"""M18a: stationary point charges are held, not secretly integrated or omitted."""
from __future__ import annotations

from dataclasses import replace
import json
from math import fsum, hypot, isclose, pi

import pytest

from conductor_sim import (
    ConductingPlane, ElectrostaticProblem, ExternalSphereBackend, FixedNetCharge,
    FixedPotential, Grounded, GroundedPlaneBackend, Motion2DSpec,
    MultiImageShellBackend, NestedShellBackend, ParallelPlateBackend,
    PointCharge, SolidSphere, SphericalShell, UnsupportedProblem,
    integrate_motion_2d, sample_motion_2d,
)
from conductor_sim.model import EPSILON_0
from conductor_sim.motion_viewer_2d import motion_slice_2d
from conductor_sim.observables import ChargeForce, EnergyBreakdown, ForceEnergyResult
from conductor_sim._motion_validation import validate_planar_motion


def norm(v):
    return hypot(*v)


def sub(a,b):
    return tuple(x-y for x,y in zip(a,b))


def spec_for(charges=None, *, moving_index=0, velocity=(.12,0.,-.01), **kwargs):
    if charges is None:
        charges=(PointCharge(1e-9,(0.,0.,.10)),PointCharge(-.5e-9,(.05,0.,.12)))
    problem=ElectrostaticProblem(tuple(charges),(ConductingPlane(),))
    return Motion2DSpec(problem,moving_index,1e-6,velocity,.03,.005,**kwargs)


def _independent_plane_force(moving, stationary):
    """Independent mirror Coulomb expression; no production force provider."""
    k=1/(4*pi*EPSILON_0)
    r=moving.position_m; q=moving.charge_c
    out=[0.,0.,-k*q*q/(4*r[2]*r[2])]
    for static in stationary:
        f=static.charge_c
        mirrored=(static.position_m[0], static.position_m[1],-static.position_m[2])
        for s,weight in ((static.position_m,1.), (mirrored,-1.)):
            delta=sub(r,s)
            d=norm(delta)
            for j in range(3):
                out[j]+=k*q*f*weight*delta[j]/d**3
    return tuple(out)


def test_index_one_identifies_moving_source_independently_of_grouped_order():
    charges=(PointCharge(-.5e-9,(.05,0.,.12)),
             PointCharge(1e-9,(0.,0.,.10)),
             PointCharge(.2e-9,(-.06,0.,.12)))
    spec=spec_for(charges,moving_index=1)
    initial=spec.problem
    changed=spec.problem_at((.007,0.,.099))
    assert changed.charges[0]==charges[0] and changed.charges[2]==charges[2]
    assert changed.charges[1].position_m==(.007,0.,.099)
    assert spec.problem is initial and spec.problem.charges==charges
    backend=GroundedPlaneBackend()
    r=integrate_motion_2d(spec,backend)
    assert r.termination=='duration' and len(r.samples)==7
    assert r.samples[0].force_n==pytest.approx(_independent_plane_force(charges[1],(charges[0],charges[2])),rel=1e-12)
    assert all(abs(row.position_m[1])<1e-12 for row in r.samples)
    assert r.max_abs_energy_drift_j < 1e-13
    data=r.as_dict()
    assert data['schema']=='conductor_sim.motion_2d.v1'
    assert data['moving_charge_index']==1
    assert [item['charge_index'] for item in data['stationary_charges']]==[0,2]
    assert data['stationary_charges'][0]['charge_c']==charges[0].charge_c
    assert data['stationary_charges'][1]['position_m']==charges[2].position_m
    assert all('velocity' not in item for item in data['stationary_charges'])
    json.dumps(data,allow_nan=False)
    validate_planar_motion(json.loads(json.dumps(data)))


def test_reordering_all_author_sources_does_not_change_motion():
    first=spec_for()
    second=spec_for(tuple(reversed(first.problem.charges)),moving_index=1)
    ra=integrate_motion_2d(first,GroundedPlaneBackend())
    rb=integrate_motion_2d(second,GroundedPlaneBackend())
    assert [row.position_m for row in ra.samples]==pytest.approx([row.position_m for row in rb.samples],abs=1e-10)
    assert [row.velocity_m_s for row in ra.samples]==pytest.approx([row.velocity_m_s for row in rb.samples],abs=1e-9)


def test_independent_force_potential_gradient_and_energy_drift():
    spec=spec_for()
    backend=GroundedPlaneBackend()
    r=integrate_motion_2d(spec,backend)
    moving=spec.selected_charge
    assert r.samples[0].force_n==pytest.approx(_independent_plane_force(moving,spec.stationary_charges),rel=1e-12)
    # Mechanical U, including the held source's constant self/image energy,
    # gives -dU/dx for the selected moving coordinate.
    x=moving.position_m
    eps=1e-5
    energy=[]
    for sign in (-1,1):
        p=spec.problem_at((x[0]+sign*eps,x[1],x[2]))
        sol=backend.solve(p)
        energy.append(sol.interaction_energy_j)
    derivative=(energy[1]-energy[0])/(2*eps)
    assert -derivative==pytest.approx(r.samples[0].force_n[0],rel=1e-7)
    assert r.max_abs_energy_drift_j<5e-13


def test_stationary_sites_choose_meridian_for_normal_launch():
    s=spec_for((PointCharge(1e-9,(0.,0.,.1)),PointCharge(.4e-9,(0.,.05,.12))),
               velocity=(0.,0.,.02))
    r=integrate_motion_2d(s,GroundedPlaneBackend())
    assert abs(r.plane_normal[0])==pytest.approx(1.,abs=1e-12)
    assert r.samples[-1].position_m[1]<0.  # like charges repel along -y
    assert all(abs(row.position_m[0])<1e-12 for row in r.samples)


def test_exterior_sphere_and_radial_launch_with_off_axis_fixed_charge():
    p=ElectrostaticProblem((PointCharge(1e-9,(.18,0.,0.)),
        PointCharge(-.4e-9,(.23,.06,0.))),(SolidSphere(.07,Grounded()),))
    spec=Motion2DSpec(p,0,1e-6,(.01,0.,0.),.04,.005)
    r=integrate_motion_2d(spec,ExternalSphereBackend())
    assert r.termination=='duration'
    assert abs(r.plane_normal[2])==pytest.approx(1.,abs=1e-12)
    assert r.samples[-1].position_m[1]>0
    assert r.samples[0].force_n[1]>0
    assert r.max_abs_energy_drift_j<5e-11


def test_isolated_sphere_fixed_charge_is_composed_not_ignored():
    p=ElectrostaticProblem((PointCharge(-1e-9,(.17,0.,0.)),
         PointCharge(1e-9,(.24,.05,0.))),
         (SolidSphere(.06,FixedNetCharge(2e-9)),))
    spec=Motion2DSpec(p,0,1e-6,(.02,.04,0.),.03,.005)
    r=integrate_motion_2d(spec,ExternalSphereBackend())
    assert r.termination=='duration'
    assert r.samples[0].force_n!=pytest.approx((0.,0.,0.),abs=1e-15)
    assert r.max_abs_energy_drift_j<2e-10
    assert r.as_dict()['stationary_charges'][0]['charge_index']==1


def test_cavity_and_nested_cavity_with_fixed_charge():
    charges=(PointCharge(1e-9,(.025,0.,0.)), PointCharge(-.5e-9,(-.03,.02,0.)))
    for conductors,backend in [
        ((SphericalShell(.1,.12,Grounded()),),MultiImageShellBackend()),
        ((SphericalShell(.1,.12,FixedNetCharge(2e-10),name='inner'),
          SphericalShell(.2,.22,Grounded(),name='outer')),NestedShellBackend()),
    ]:
        spec=Motion2DSpec(ElectrostaticProblem(charges,conductors),0,1e-6,(.02,.03,0.),.03,.005)
        r=integrate_motion_2d(spec,backend)
        assert r.termination=='duration'
        assert len(r.samples)==7
        assert r.samples[-1].position_m[1]>.0008
        assert r.max_abs_energy_drift_j<1e-11


def test_biased_plates_preserve_external_voltage_work_in_mechanical_energy():
    lower=ConductingPlane(name='lower')
    upper=ConductingPlane((0.,0.,.2),(0.,0.,-1.),FixedPotential(20.),'upper')
    charges=(PointCharge(1e-10,(0.,0.,.1)),PointCharge(1e-10,(.04,0.,.12)))
    p=ElectrostaticProblem(charges,(lower,upper))
    spec=Motion2DSpec(p,0,1e-6,(.02,0.,.01),.03,.005)
    backend=ParallelPlateBackend(image_order=16)
    r=integrate_motion_2d(spec,backend)
    assert r.termination=='duration' and r.max_abs_energy_drift_j<1e-12
    # Real force differs when the fixed source is omitted.
    lone=replace(spec,problem=replace(p,charges=(charges[0],)))
    b=integrate_motion_2d(lone,backend)
    assert abs(r.samples[0].force_n[0]-b.samples[0].force_n[0])>1e-10
    assert all(row.velocity_m_s[1]==0 for row in r.samples)


def test_charge_approach_stops_before_point_not_at_a_fictitious_collision():
    p=ElectrostaticProblem((PointCharge(-1e-10,(0.,0.,.1)),
        PointCharge(1e-10,(.03,0.,.1))),(ConductingPlane(),))
    spec=Motion2DSpec(p,0,1e-6,(.1,0.,0.),.5,.005,charge_guard_m=.005)
    r=integrate_motion_2d(spec,GroundedPlaneBackend())
    assert r.termination=='charge-guard' and len(r.events)==1
    assert r.events[0].kind=='charge-guard'
    assert r.events[0].time_s<spec.duration_s
    assert r.events[0].time_s==pytest.approx(r.samples[-1].time_s)
    assert r.events[0].position_m==r.samples[-1].position_m
    assert 'not an exact collision' in r.events[0].description
    assert min(norm(sub(row.position_m,p.charges[1].position_m)) for row in r.samples)>.005
    validate_planar_motion(json.loads(json.dumps(r.as_dict())))


def test_rejects_nonplanar_stationary_inventory_and_duplicate_sources():
    s=spec_for()
    with pytest.raises(UnsupportedProblem,match='share the physically invariant plane'):
        replace(s,problem=replace(s.problem,charges=(*s.problem.charges,PointCharge(1e-9,(.02,.03,.12)))))
    with pytest.raises(UnsupportedProblem,match='distinct sites'):
        replace(s,problem=replace(s.problem,charges=(s.problem.charges[0],s.problem.charges[0])))
    with pytest.raises(UnsupportedProblem,match='nonzero'):
        replace(s,problem=replace(s.problem,charges=(*s.problem.charges,PointCharge(0.,(.03,0,.14)))))
    with pytest.raises(UnsupportedProblem,match='31 fixed'):
        replace(s,problem=replace(s.problem,charges=tuple(PointCharge(1e-9,(i*.001,0,.1)) for i in range(33))))
    with pytest.raises(UnsupportedProblem,match='valid vacuum'):
        replace(s,problem=replace(s.problem,charges=(s.problem.charges[0],PointCharge(.5e-9,(.05,0,-.1)))))
    for bad in (0.,-1.,float('nan'),True):
        with pytest.raises(ValueError,match='charge_guard_m'):
            replace(s,charge_guard_m=bad)
    with pytest.raises(ValueError,match='stationary point-charge guard'):
        replace(s,charge_guard_m=.1)
    with pytest.raises(ValueError,match='requires at least one'):
        replace(s, problem=replace(s.problem,charges=(s.problem.charges[0],)),charge_guard_m=1e-3)


def test_nonplanar_exterior_and_nested_cavity_are_refused():
    p=ElectrostaticProblem((PointCharge(1e-9,(.18,0.,0.)),PointCharge(.5e-9,(.23,.02,.03))),
                           (SolidSphere(.07,Grounded()),))
    with pytest.raises(UnsupportedProblem,match='share the physically invariant plane'):
        Motion2DSpec(p,0,1e-6,(.02,.02,0.),.03,.005)
    p=ElectrostaticProblem((PointCharge(1e-9,(.03,0.,0.)),PointCharge(.5e-9,(-.03,.03,.03))),
                           (SphericalShell(.1,.12,Grounded()),))
    with pytest.raises(UnsupportedProblem,match='share the physically invariant plane'):
        Motion2DSpec(p,0,1e-6,(0.,.03,0.),.03,.005)


def test_no_femm_force_substitution_and_owned_solution_cleanup():
    closed=[]
    class NoForce:
        def close(self):closed.append('closed')
    class Numerical:
        def supports(self,p):return True
        def solve(self,p):return NoForce()
    with pytest.raises(UnsupportedProblem,match='does not provide'):
        integrate_motion_2d(spec_for(),Numerical())
    assert closed==['closed']


def test_fake_provider_reordered_sites_cannot_change_selected_force():
    closed=[]
    class Fake:
        def __init__(self,p):self.p=p
        def close(self):closed.append(True)
        def force_energy(self):
            moving=self.p.charges[1]
            fixed=self.p.charges[0]
            f=(1e-7,0.,0.)
            return ForceEnergyResult((ChargeForce(fixed,(0.,0.,0.),(0.,0.,0.)),
                ChargeForce(moving,(0.,0.,0.),(0.,0.,0.),f)),
                EnergyBreakdown(0.,0.,-f[0]*moving.position_m[0], 'held-source potential'),
                'test provider with sorted (not authored) sites')
    class Backend:
        def supports(self,p):return True
        def solve(self,p):return Fake(p)
    fixed=PointCharge(5e-10,(-.03,0,.12));moving=PointCharge(-1e-9,(0.,0.,.1))
    spec=spec_for((fixed,moving),moving_index=1,velocity=(0.,0.,0.))
    r=integrate_motion_2d(spec,Backend())
    assert r.samples[-1].position_m[0]==pytest.approx(.5*(1e-7/1e-6)*.03**2,abs=1e-12)
    assert r.samples[-1].force_n==(1e-7,0.,0.)
    assert r.max_abs_energy_drift_j<1e-16
    assert len(closed)>len(r.samples)


def test_reject_mismatched_backend_inventory_and_close_solution():
    closed=[]
    class Wrong:
        def __init__(self,p):self.p=p
        def close(self):closed.append(True)
        def force_energy(self):
            source=self.p.charges[0]
            return ForceEnergyResult((ChargeForce(source,(0.,0.,0.),(0.,0.,0.)),),
                                     EnergyBreakdown(0.,0.,0.,'test'), 'missing fixed site')
    class Backend:
        def supports(self,p):return True
        def solve(self,p):return Wrong(p)
    with pytest.raises(UnsupportedProblem,match='identity'):
        integrate_motion_2d(spec_for(),Backend())
    assert closed==[True]


def test_M18b_viewer_is_explicitly_not_implemented_in_M18a():
    s=spec_for()
    class MustNotSolve:
        def supports(self,p):raise AssertionError('viewer touched backend')
        def solve(self,p):raise AssertionError('viewer touched backend')
    with pytest.raises(UnsupportedProblem,match='M18b viewer'):
        sample_motion_2d(s,MustNotSolve())
    r=integrate_motion_2d(s,GroundedPlaneBackend())
    with pytest.raises(UnsupportedProblem,match='M18b viewer'):
        motion_slice_2d(s,r)


def test_one_source_M17a_payload_backward_compatible():
    s=spec_for(charges=(PointCharge(1e-9,(0.,0.,.1)),))
    r=integrate_motion_2d(s,GroundedPlaneBackend())
    result=r.as_dict()
    assert result['schema']=='conductor_sim.motion_2d.v1'
    assert 'stationary_charges' not in result and 'moving_charge_index' not in result
    assert 'charge_guard_m' not in result
    assert r.termination=='duration'
    validate_planar_motion(json.loads(json.dumps(result)))


def _independent_sphere_force(moving, static, sphere):
    """Exact external Kelvin images plus prescribed central monopole."""
    k=1/(4*pi*EPSILON_0)
    a=sphere.radius_m;center=sphere.center_m
    image=[]
    for charge in (moving,*static):
        radius=sub(charge.position_m,center)
        d=norm(radius)
        q_image=-charge.charge_c*a/d
        pos=tuple(center[i]+a*a*radius[i]/d**2 for i in range(3))
        image.append((q_image,pos))
    if isinstance(sphere.constraint,FixedNetCharge):
        center_q=sphere.constraint.charge_c-fsum(q for q,_ in image)
        image.append((center_q,center))
    # Direct fixed sources and *all* induced Kelvin images affect the
    # moving charge; the moving real source never acts on itself.
    F=[0.,0.,0.]
    for q,pos in [*((q.charge_c,q.position_m) for q in static),*image]:
        delta=sub(moving.position_m,pos)
        d=norm(delta)
        for i in range(3):F[i]+=k*moving.charge_c*q*delta[i]/d**3
    return tuple(F)


@pytest.mark.parametrize('constraint',[Grounded(),FixedNetCharge(1.6e-9)])
def test_independent_kelvin_fixed_source_force_and_reaction(constraint):
    sphere=SolidSphere(.07,constraint)
    moving=PointCharge(-1e-9,(.18,0.,0.))
    fixed=(PointCharge(1e-9,(.23,.05,0.)),)
    p=ElectrostaticProblem((moving,*fixed),(sphere,))
    r=integrate_motion_2d(Motion2DSpec(p,0,1e-6,(.02,.04,0.),.03,.005),
                          ExternalSphereBackend())
    expected=_independent_sphere_force(moving,fixed,sphere)
    assert r.samples[0].force_n==pytest.approx(expected,rel=1e-11)
    assert r.max_abs_energy_drift_j<1e-10


def test_centered_cavity_with_held_source_selects_physical_meridian():
    p=ElectrostaticProblem((PointCharge(1e-9,(0.,0.,0.)),
                            PointCharge(.5e-9,(.04,.02,0.))),
                            (SphericalShell(.1,.12,Grounded()),))
    spec=Motion2DSpec(p,0,1e-6,(0.,0.,0.),.03,.005)
    r=integrate_motion_2d(spec,MultiImageShellBackend())
    assert r.termination=='duration'
    assert r.samples[-1].position_m[0]<0  # like charges repel
    assert r.samples[-1].position_m[1]<0
    assert abs(r.plane_normal[2])==pytest.approx(1.,abs=1e-12)


def test_multi_source_zero_velocity_and_no_force_at_center_is_not_assumed():
    p=ElectrostaticProblem((PointCharge(1e-9,(0.,0.,0.)),
                            PointCharge(1e-9,(.05,0.,0.))),
                            (SphericalShell(.1,.12,Grounded()),))
    spec=Motion2DSpec(p,0,1e-6,(0.,0.,0.),.03,.005)
    r=integrate_motion_2d(spec,MultiImageShellBackend())
    assert r.samples[0].force_n[0]<0
    assert r.samples[-1].velocity_m_s[0]<0
    assert r.max_abs_energy_drift_j<1e-11


def test_standalone_M18a_cli_cases_retain_stationary_inventory(tmp_path):
    from pathlib import Path
    import subprocess, sys
    root=Path(__file__).resolve().parents[1]
    for case in ('plane','plates','sphere','cavity','nested','guard'):
        path=tmp_path/(case+'.json')
        run=subprocess.run([sys.executable,'examples/particle_motion_fixed_charges.py',
            '--case',case,'--output',str(path)],cwd=root,text=True,capture_output=True,timeout=30)
        assert run.returncode==0,run.stderr+run.stdout
        data=json.loads(path.read_text())
        assert data['example']==case
        assert len(data['stationary_charges'])==1
        assert data['moving_charge_index'] in (0,1)
        assert data['stationary_charges'][0]['charge_index'] != data['moving_charge_index']
        assert len(data['samples'])>=2
        validate_planar_motion(data)
        assert data['termination']==('charge-guard' if case=='guard' else 'duration')
        assert not path.with_suffix('.html').exists()  # M18b visualization still unimplemented
