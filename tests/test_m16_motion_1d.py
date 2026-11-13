"""M16a: 1D dynamics, independently checked under constant force and image forces."""
from math import isclose, pi
import json

import pytest

from conductor_sim import (ConductingPlane, ElectrostaticProblem, FixedNetCharge,
                           FixedPotential, Grounded, PointCharge, SolidSphere,
                           SphericalShell, GroundedPlaneBackend, ParallelPlateBackend,
                           ExternalSphereBackend, NestedShellBackend,
                           MultiImageShellBackend, UnsupportedProblem)
from conductor_sim.model import EPSILON_0
from conductor_sim.observables import ChargeForce, EnergyBreakdown, ForceEnergyResult
from conductor_sim.motion_1d import (Motion1DSpec, MotionIntegrationError, integrate_motion_1d)

Q = 1e-9
K = 1/(4*pi*EPSILON_0)


def single(geom, position=(0.,0.,.1), q=Q):
    return ElectrostaticProblem((PointCharge(q,position),),tuple(geom))


class ConstantForceBackend:
    """Independent exact ODE reference, not an approximation to plate induction."""
    def __init__(self, force_n=(0.,0.,1e-6), constant_j=0.):
        self.force_n=force_n
        self.constant_j=constant_j
        self.opens=0
        self.closes=0

    def supports(self, problem):return True

    def solve(self,problem):
        self.opens+=1
        parent=self
        class Solution:
            def force_energy(self):
                r=problem.charges[0].position_m
                force=parent.force_n
                energy=EnergyBreakdown(0.,parent.constant_j,-sum(force[i]*r[i] for i in range(3)),
                                       'independent constant-force ODE test double')
                return ForceEnergyResult((ChargeForce(problem.charges[0],(0.,0.,0.),(0.,0.,0.),force),),
                                         energy,'closed-form uniform-force test double')
            def close(self):parent.closes+=1
        return Solution()


def test_uniform_acceleration_with_velocity_reversal_and_exact_turning_time():
    plates=(ConductingPlane((0,0,0),(0,0,1),Grounded(),'lo'),
            ConductingPlane((0,0,.2),(0,0,-1),FixedPotential(20.),'hi'))
    p=single(plates,position=(0,0,.1))
    backend=ConstantForceBackend()
    spec=Motion1DSpec(p,0,2e-6,(0,0,2),-.05,.30,.03)
    result=integrate_motion_1d(spec,backend)
    assert result.termination=='duration'
    assert len(result.samples)==11
    assert len(result.events)==1 and result.events[0].kind=='turning-point'
    event=result.events[0]
    assert event.time_s==pytest.approx(.10,abs=2e-7)
    assert event.displacement_m==pytest.approx(-.0025,abs=1e-8)
    for sample in result.samples:
        t=sample.time_s
        assert sample.position_m[2]==pytest.approx(.1-.05*t+.25*t*t,abs=1e-11)
        assert sample.velocity_m_s==pytest.approx(-.05+.5*t,abs=1e-11)
        assert sample.acceleration_m_s2==pytest.approx(.5,rel=2e-11)
        assert sample.force_n==1e-6
        assert abs(sample.energy_drift_j)<2e-19
    assert result.max_abs_energy_drift_j<2e-19
    assert backend.opens==backend.closes and backend.opens>50
    payload=result.as_dict()
    assert payload['mode']=='elapsed-time classical dynamics'
    assert payload['termination']=='duration'
    assert payload['samples'][-1]['time_s']==pytest.approx(.3)
    json.dumps(payload,allow_nan=False)


def test_large_energy_offset_does_not_hide_mechanical_energy_drift():
    p=single((ConductingPlane(),),position=(0,0,.1))
    backend=ConstantForceBackend((0.,0.,-1e-8),constant_j=1e13)
    spec=Motion1DSpec(p,0,1e-6,(0,0,1),.02,.05,.01)
    r=integrate_motion_1d(spec,backend)
    assert r.samples[-1].interaction_energy_j==r.samples[0].interaction_energy_j
    assert abs(r.samples[-1].energy_drift_j)<1e-19
    assert r.samples[-1].kinetic_energy_j!=r.samples[0].kinetic_energy_j
    assert backend.opens==backend.closes


def test_grounded_plane_induction_force_and_energy_conservation():
    p=single((ConductingPlane(),),position=(0,0,.1))
    spec=Motion1DSpec(p,0,1e-6,(0,0,1),.1,.075,.005)
    result=integrate_motion_1d(spec,GroundedPlaneBackend())
    assert result.termination=='duration'
    assert len(result.samples)==16
    for row in result.samples:
        h=row.position_m[2]
        assert row.force_n==pytest.approx(-K*Q*Q/(4*h*h),rel=2e-12)
        assert row.interaction_energy_j==pytest.approx(-K*Q*Q/(4*h),rel=2e-12)
    assert result.max_abs_energy_drift_j<2e-17
    assert result.samples[-1].velocity_m_s<.1


def test_surface_guard_is_distinct_from_claimed_exact_impact():
    p=single((ConductingPlane(),),position=(0,0,.02))
    spec=Motion1DSpec(p,0,1e-6,(0,0,1),-.5,.15,.01,
                      surface_guard_m=1e-4,relative_tolerance=2e-6)
    result=integrate_motion_1d(spec,GroundedPlaneBackend())
    assert result.termination=='surface-guard'
    assert result.events[-1].kind=='surface-guard'
    assert result.events[-1].time_s<spec.duration_s
    assert result.samples[-1].position_m[2]>spec.surface_guard_m
    assert result.samples[-1].position_m[2]<spec.surface_guard_m*1.12
    assert 'not resolved' in result.events[-1].description or 'cutoff' in result.events[-1].description


def test_cavity_centre_stationary_and_displaced_force_points_outward():
    shell=SphericalShell(.1,.12,Grounded(),name='shell')
    p=single((shell,),position=(0,0,0))
    spec=Motion1DSpec(p,0,1e-6,(0,0,1),0.,.2,.02)
    r=integrate_motion_1d(spec,MultiImageShellBackend())
    assert all(abs(row.displacement_m)<1e-14 and abs(row.velocity_m_s)<1e-13 for row in r.samples)
    assert r.events==()
    assert r.max_abs_energy_drift_j<1e-18
    out=single((shell,),position=(0,0,.02))
    move=integrate_motion_1d(Motion1DSpec(out,0,1e-6,(0,0,1),0.,.15,.015),MultiImageShellBackend())
    assert move.samples[-1].displacement_m>0
    assert move.samples[-1].velocity_m_s>0
    for row in move.samples:
        rr=row.position_m[2]
        exact=K*Q*Q*.1*rr/(.1**2-rr*rr)**2
        assert row.force_n==pytest.approx(exact,rel=2e-12)
    assert move.max_abs_energy_drift_j<2e-16


@pytest.mark.parametrize('backend,conductors,pos,direction',[
    (ExternalSphereBackend(),(SolidSphere(.1,Grounded(),name='solid'),),(.2,0,0),(1,0,0)),
    (ExternalSphereBackend(),(SolidSphere(.1,FixedNetCharge(-.5e-9),name='solid'),),(0,0,-.2),(0,0,1)),
    (NestedShellBackend(),(SphericalShell(.1,.12,FixedNetCharge(.7e-9),name='inner'),SphericalShell(.2,.23,Grounded(),name='outer')), (0,0,.04),(0,0,1)),
    (ParallelPlateBackend(image_order=28),(ConductingPlane(name='lower'),ConductingPlane((0,0,.2),(0,0,-1),FixedPotential(20.),'upper')), (0,0,.1),(0,0,1)),
])
def test_composes_across_supported_analytical_field_families(backend,conductors,pos,direction):
    p=single(conductors,position=pos)
    result=integrate_motion_1d(Motion1DSpec(p,0,1e-6,direction,0.,.06,.012),backend)
    assert result.termination=='duration'
    assert len(result.samples)==6
    assert result.max_abs_energy_drift_j<5e-14
    assert all(abs(row.energy_drift_j)<5e-14 for row in result.samples)


def test_reverse_axis_is_just_a_coordinate_change_not_a_new_force():
    p=single((ConductingPlane(),),position=(0,0,.1))
    a=integrate_motion_1d(Motion1DSpec(p,0,1e-6,(0,0,1),-.03,.10,.01),GroundedPlaneBackend())
    b=integrate_motion_1d(Motion1DSpec(p,0,1e-6,(0,0,-1),+.03,.10,.01),GroundedPlaneBackend())
    for x,y in zip(a.samples,b.samples):
        assert x.position_m==pytest.approx(y.position_m,abs=1e-10)
        assert x.velocity_m_s==pytest.approx(-y.velocity_m_s,abs=1e-8)
        assert x.force_n==pytest.approx(-y.force_n,abs=1e-13)


@pytest.mark.parametrize('kwargs,error',[
    ({'mass_kg':0.},ValueError),
    ({'mass_kg':-1.},ValueError),
    ({'mass_kg':float('nan')},ValueError),
    ({'axis':(0,0,0)},ValueError),
    ({'axis':(1,0,0)},UnsupportedProblem),
    ({'initial_velocity_m_s':float('inf')},ValueError),
    ({'initial_velocity_m_s':.15*299792458.},UnsupportedProblem),
    ({'duration_s':-1.},ValueError),
    ({'relative_tolerance':0.3},ValueError),
    ({'surface_guard_m':.2},ValueError),
    ({'max_internal_steps':2},ValueError),
    ({'sample_interval_s':1.},ValueError),
])
def test_strict_spec_refusals(kwargs,error):
    p=single((ConductingPlane(),),position=(0,0,.1))
    args=dict(problem=p,charge_index=0,mass_kg=1e-6,axis=(0,0,1),
              initial_velocity_m_s=.1,duration_s=.1,sample_interval_s=.01)
    args.update(kwargs)
    with pytest.raises(error): Motion1DSpec(**args)


def test_one_dimensional_symmetry_rejects_off_axis_and_more_sources():
    c=SolidSphere(.1,Grounded())
    with pytest.raises(UnsupportedProblem,match='radial'):
        Motion1DSpec(single((c,),position=(0.04,.02,.2)),0,1e-6,(0,0,1),0,.1,.01)
    shell=SphericalShell(.1,.12,Grounded())
    with pytest.raises(UnsupportedProblem,match='line through'):
        Motion1DSpec(single((shell,),position=(.03,0,.04)),0,1e-6,(0,0,1),0,.1,.01)
    with pytest.raises(UnsupportedProblem,match='single moving'):
        Motion1DSpec(ElectrostaticProblem((PointCharge(Q,(0,0,.1)),PointCharge(-Q,(.03,0,.1))),
                                           (ConductingPlane(),)),0,1e-6,(0,0,1),0,.1,.01)
    with pytest.raises(UnsupportedProblem):
        Motion1DSpec(single((ConductingPlane(name='first'),ConductingPlane((0,0,.2),(0,0,1),Grounded(),name='second')),
                            position=(0,0,.1)),0,1e-6,(0,0,1),0,.1,.01)


def test_backend_without_force_provider_refused_not_silently_analytical():
    class NoForce:
        def supports(self,p):return True
        def solve(self,p):return object()
    p=single((ConductingPlane(),),position=(0,0,.1))
    with pytest.raises(UnsupportedProblem,match='force/energy'):
        integrate_motion_1d(Motion1DSpec(p,0,1e-6,(0,0,1),0.,.1,.01),NoForce())


def test_transverse_force_provider_refused_not_projected_onto_motion_axis():
    p=single((ConductingPlane(),),position=(0,0,.1))
    with pytest.raises(UnsupportedProblem,match='transverse'):
        integrate_motion_1d(Motion1DSpec(p,0,1e-6,(0,0,1),0.,.1,.01),
                            ConstantForceBackend((1e-6,0,1e-6)))


def test_real_biased_plate_electron_transit_resolves_nanosecond_times():
    # JEE-scale independent check: qE/m acceleration in an almost uniform
    # capacitor field. The real plate backend still includes its images.
    e=1.602176634e-19
    mass=9.1093837139e-31
    gap=.2
    voltage=20.
    p=single((ConductingPlane(name='lower'),
              ConductingPlane((0,0,gap),(0,0,-1),FixedPotential(voltage),'upper')),
             position=(0,0,.09),q=-e)
    time=5e-8
    a=e*voltage/(gap*mass)
    spec=Motion1DSpec(p,0,mass,(0,0,1),0.,time,5e-9,relative_tolerance=2e-8)
    result=integrate_motion_1d(spec,ParallelPlateBackend(image_order=24))
    assert result.termination=='duration'
    assert len(result.samples)==11
    assert result.samples[-1].position_m[2]==pytest.approx(.09+.5*a*time*time,rel=5e-8)
    assert result.samples[-1].velocity_m_s==pytest.approx(a*time,rel=5e-8)
    assert result.max_abs_energy_drift_j<2e-27


def test_symmetry_guard_rejects_tiny_physically_transverse_forces():
    p=single((ConductingPlane(),),position=(0,0,.1))
    with pytest.raises(UnsupportedProblem,match='transverse'):
        integrate_motion_1d(Motion1DSpec(p,0,1e-6,(0,0,1),0.,.01,.005),
                            ConstantForceBackend((8e-20,0,8e-20)))


def test_evolving_speed_outside_newtonian_regime_is_not_reported_as_valid():
    p=single((ConductingPlane(),),position=(0,0,.1))
    spec=Motion1DSpec(p,0,1e-20,(0,0,1),0.,.01,.001)
    with pytest.raises(UnsupportedProblem,match='nonrelativistic'):
        integrate_motion_1d(spec,ConstantForceBackend((0,0,1e-6)))


def test_independent_kelvin_cavity_energy_to_speed_conversion():
    a=.1
    q=Q
    r0=.02
    m=1e-6
    shell=SphericalShell(a,.12,Grounded())
    p=single((shell,),position=(0,0,r0))
    result=integrate_motion_1d(Motion1DSpec(p,0,m,(0,0,1),0.,.10,.01),
                               MultiImageShellBackend())
    last=result.samples[-1]
    r=last.position_m[2]
    initial_u=-K*q*q*a/(2*(a*a-r0*r0))
    current_u=-K*q*q*a/(2*(a*a-r*r))
    assert last.interaction_energy_j==pytest.approx(current_u,rel=1e-12)
    assert .5*m*last.velocity_m_s**2==pytest.approx(initial_u-current_u,rel=4e-8)


def test_translation_and_physical_normal_invariance():
    # An arbitrary plane anchor along its infinite surface has no impact.
    first=ConductingPlane((0,0,0),(0,0,1),Grounded(),'earth')
    shifted=ConductingPlane((1e6,-7e5,0),(0,0,1),Grounded(),'earth')
    result=[]
    for plane in (first,shifted):
        p=single((plane,),position=(.04,.02,.10))
        result.append(integrate_motion_1d(
            Motion1DSpec(p,0,1e-6,(0,0,1),0.,.10,.01),GroundedPlaneBackend()))
    for a,b in zip(result[0].samples,result[1].samples):
        assert a.position_m==pytest.approx(b.position_m,rel=1e-10,abs=1e-10)
        assert a.velocity_m_s==pytest.approx(b.velocity_m_s,rel=1e-10,abs=1e-10)


def test_real_spherical_surface_guard_with_no_crossing():
    from conductor_sim import ExternalSphereBackend
    specs=(
        (SolidSphere(.1,Grounded(),name='metal'),(.0,.0,.14),-.35, ExternalSphereBackend()),
        (SphericalShell(.1,.12,Grounded(),name='metal'),(0.,0.,.06),.25,MultiImageShellBackend()),
    )
    for metal,pos,velocity,backend in specs:
        p=single((metal,),position=pos)
        result=integrate_motion_1d(Motion1DSpec(p,0,1e-6,(0,0,1),velocity,.25,.01,
                                                surface_guard_m=2e-4),backend)
        assert result.termination=='surface-guard'
        assert len(result.samples)>=2
        assert result.events[-1].kind=='surface-guard'
        boundary=metal.radius_m if isinstance(metal,SolidSphere) else metal.inner_radius_m
        assert all((abs(row.position_m[2]) > boundary if isinstance(metal,SolidSphere)
                    else abs(row.position_m[2]) < boundary) for row in result.samples)
        assert result.samples[-1].time_s<.25
