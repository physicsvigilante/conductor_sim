"""M17a: independent vector-motion checks without browser-side physics."""
from dataclasses import replace
import json
from math import hypot
import pytest
from conductor_sim import (
    ConductingPlane, ElectrostaticProblem, ExternalSphereBackend, FixedPotential,
    Grounded, GroundedPlaneBackend, Motion2DSpec, MultiImageShellBackend,
    ParallelPlateBackend, PointCharge, SolidSphere, SphericalShell,
    integrate_motion_2d,
)
from conductor_sim.observables import ChargeForce, EnergyBreakdown, ForceEnergyResult
from conductor_sim.solver import UnsupportedProblem


def norm(v):
    return hypot(*v)


def plane_spec(**overrides):
    problem = ElectrostaticProblem((PointCharge(1e-9, (0., 0., .1)),), (ConductingPlane(),))
    options = dict(problem=problem, charge_index=0, mass_kg=1e-6,
                   initial_velocity_m_s=(.2, 0., -.03),
                   duration_s=.08, sample_interval_s=.01)
    return Motion2DSpec(**(options | overrides))


def check_result(r):
    data = json.loads(json.dumps(r.as_dict(), allow_nan=False))
    assert data['schema'] == 'conductor_sim.motion_2d.v1'
    assert 'frames' not in data and 'position_sweep' not in data
    assert len(r.samples) >= 2
    assert all(a.time_s < b.time_s for a,b in zip(r.samples, r.samples[1:]))
    for s in r.samples:
        assert s.kinetic_energy_j == pytest.approx(.5*r.spec.mass_kg*norm(s.velocity_m_s)**2)
        p = tuple(s.position_m[i]-r.plane_origin_m[i] for i in range(3))
        assert abs(sum(p[i]*r.plane_normal[i] for i in range(3))) < 1e-10
        assert abs(sum(s.velocity_m_s[i]*r.plane_normal[i] for i in range(3))) < 1e-10


def test_angled_plane_motion_preserves_tangential_speed_and_energy():
    r = integrate_motion_2d(plane_spec(), GroundedPlaneBackend())
    assert r.termination == 'duration'
    assert len(r.samples) == 9
    assert r.samples[-1].position_m[0] == pytest.approx(.016, rel=1e-8)
    assert r.samples[-1].velocity_m_s[0] == pytest.approx(.2, abs=1e-11)
    assert r.samples[-1].position_m[2] < .1 - .03*.08
    assert r.max_abs_energy_drift_j < 1e-12
    check_result(r)


def test_biased_plates_independent_uniform_field_limit():
    lower = ConductingPlane(name='lower')
    upper = ConductingPlane((0,0,.2),(0,0,-1),FixedPotential(20.),'upper')
    charge,mass = 1e-15,1e-12
    problem = ElectrostaticProblem((PointCharge(charge,(0,0,.1)),),(lower,upper))
    spec = Motion2DSpec(problem,0,mass,(.04,.03,.02),.2,.01)
    result = integrate_motion_2d(spec, ParallelPlateBackend(image_order=48))
    s = result.samples[-1]
    expected_z = .1+.02*.2 + .5*(charge*(-100.)/mass)*(.2**2)
    assert s.position_m[2] == pytest.approx(expected_z, abs=3e-5)
    assert s.position_m[0] == pytest.approx(.008,abs=1e-10)
    assert s.position_m[1] == pytest.approx(.006,abs=1e-10)
    assert result.max_abs_energy_drift_j < 1e-15
    check_result(result)


def test_sphere_curved_motion_preserves_angular_momentum_and_energy():
    problem = ElectrostaticProblem(
        (PointCharge(1e-9,(.18,0.,0.)),),(SolidSphere(.07,Grounded()),))
    spec = Motion2DSpec(problem,0,1e-6,(.015,.13,.07),.04,.005)
    r = integrate_motion_2d(spec,ExternalSphereBackend())
    assert r.termination == 'duration'
    s = r.samples[-1]
    def cross(a,b):
        return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
    assert norm(cross(s.position_m,s.velocity_m_s)) == pytest.approx(
        norm(cross((.18,0.,0.),spec.initial_velocity_m_s)), rel=1e-5)
    assert s.position_m[1] > .003
    assert r.max_abs_energy_drift_j < 1e-11
    check_result(r)


def test_offcenter_cavity_and_stationary_exact_center():
    problem = ElectrostaticProblem((PointCharge(1e-9,(.03,0.,.01)),),
                                  (SphericalShell(.1,.12,Grounded()),))
    spec = Motion2DSpec(problem,0,1e-6,(0.,.09,.03),.04,.005)
    result = integrate_motion_2d(spec, MultiImageShellBackend())
    assert result.termination == 'duration'
    assert result.samples[-1].position_m[1] > .002
    assert result.max_abs_energy_drift_j < 1e-11
    check_result(result)
    centered = replace(spec, problem=replace(problem,charges=(PointCharge(1e-9,(0.,0.,0.)),)),
                       initial_velocity_m_s=(0.,0.,0.))
    still = integrate_motion_2d(centered, MultiImageShellBackend())
    assert all(norm(row.velocity_m_s)<1e-15 and norm(row.force_n)<1e-15 for row in still.samples)


def test_guard_stops_before_metal_without_claiming_impact():
    spec = plane_spec(
        problem=ElectrostaticProblem((PointCharge(1e-9,(0.,0.,.02)),),
                                     (ConductingPlane(),)),
        initial_velocity_m_s=(.02,0.,-.5),
        duration_s=.15, surface_guard_m=1e-4)
    r = integrate_motion_2d(spec,GroundedPlaneBackend())
    assert r.termination == 'surface-guard'
    assert r.events[-1].kind == 'surface-guard'
    assert r.events[-1].time_s < spec.duration_s
    assert r.samples[-1].time_s == pytest.approx(r.events[-1].time_s)
    assert all(row.position_m[2] > spec.surface_guard_m for row in r.samples)
    assert 'not an exact collision' in r.events[-1].description


def test_rejects_bad_geometry_speed_and_charge_inventory():
    spec=plane_spec()
    with pytest.raises(UnsupportedProblem,match='one nonzero'):
        replace(spec,problem=replace(spec.problem,charges=(
            spec.problem.charges[0], PointCharge(1e-9,(.1,0,.1)))))
    with pytest.raises(UnsupportedProblem,match='vacuum side'):
        replace(spec,problem=replace(spec.problem,charges=(PointCharge(1e-9,(0,0,-.001)),)))
    upper=ConductingPlane((0,0,.2),(0,1,0),Grounded(),'oblique')
    with pytest.raises(UnsupportedProblem,match='parallel'):
        replace(spec,problem=replace(spec.problem,conductors=(*spec.problem.conductors,upper)))
    with pytest.raises(UnsupportedProblem,match='0.1c'):
        replace(spec,initial_velocity_m_s=(3e7,0,0))
    with pytest.raises(UnsupportedProblem,match='plane'):
        replace(spec,problem=replace(spec.problem,conductors=(
            spec.problem.conductors[0],SolidSphere(.02,Grounded()))))
    with pytest.raises(ValueError,match='guard'):
        replace(spec,surface_guard_m=.2)
    with pytest.raises(ValueError,match='interval'):
        replace(spec,sample_interval_s=.001,duration_s=3.)


def test_no_force_fallback_closes_owned_solution():
    closed=[]
    class NoForce:
        def close(self):closed.append(True)
    class Numerical:
        def supports(self, problem):return True
        def solve(self, problem):return NoForce()
    with pytest.raises(UnsupportedProblem,match='does not provide'):
        integrate_motion_2d(plane_spec(),Numerical())
    assert closed == [True]


def test_refuses_transverse_force_instead_of_projecting():
    closed=[]
    class Fake:
        def __init__(self,p):self.p=p
        def close(self):closed.append(True)
        def force_energy(self):
            return ForceEnergyResult(
                (ChargeForce(self.p.charges[0],(0.,1.,0.),(0.,0.,0.)),),
                EnergyBreakdown(0.,0.,0.,'test double'), 'unphysical transverse test')
    class Backend:
        def supports(self,p):return True
        def solve(self,p):return Fake(p)
    with pytest.raises(UnsupportedProblem,match='transverse force'):
        integrate_motion_2d(plane_spec(),Backend())
    assert closed == [True]


def test_exact_vector_constant_acceleration_and_energy():
    closed=[]
    class Fake:
        def __init__(self,p):self.source=p.charges[0]
        def close(self):closed.append(True)
        def force_energy(self):
            f=(0.,0.,-2e-7)
            return ForceEnergyResult(
                (ChargeForce(self.source,(0.,0.,0.),(0.,0.,0.),f),),
                EnergyBreakdown(0.,0.,-f[2]*self.source.position_m[2],'test'),
                'exact constant force')
    class Backend:
        def supports(self,p):return True
        def solve(self,p):return Fake(p)
    r=integrate_motion_2d(plane_spec(),Backend())
    t=.08
    assert r.samples[-1].position_m[0] == pytest.approx(.2*t,abs=1e-12)
    assert r.samples[-1].position_m[2] == pytest.approx(.1-.03*t-.5*.2*t*t,abs=1e-12)
    assert r.samples[-1].velocity_m_s[2] == pytest.approx(-.03-.2*t,abs=1e-12)
    assert r.max_abs_energy_drift_j < 1e-17
    assert len(closed)>len(r.samples)
    check_result(r)


def test_author_model_unchanged_and_saved_schema_is_not_browser_payload():
    spec=plane_spec()
    initial=spec.problem
    data=integrate_motion_2d(spec,GroundedPlaneBackend()).as_dict()
    assert data['mode']=='elapsed-time planar classical vector dynamics'
    assert 'motion_1d' not in data and 'frames' not in data
    assert data['samples'][0]['position_m']==pytest.approx((0,0,.1))
    assert spec.problem==initial
