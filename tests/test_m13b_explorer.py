"""M13b path authoring, identity, energy consistency and viewer contracts."""
from __future__ import annotations

import json
from math import hypot, pi

import pytest

from conductor_sim import (
    ConductingPlane, ElectrostaticProblem, FixedNetCharge, FixedPotential,
    Grounded, GroundedPlaneBackend, ExternalSphereBackend, ParallelPlateBackend,
    PointCharge, SolidSphere, force_energy,
)
from conductor_sim.position_sweep import (PositionSweepSpec, sample_position_sweep,
                                          sweep_slice)
from conductor_sim.solver import UnsupportedProblem
from conductor_sim.visualization import SliceSpec
from conductor_sim.viewer import render_html

Q=1e-9


def plates(charges, *, upper=Grounded()):
    return ElectrostaticProblem(tuple(charges),(
        ConductingPlane(name='lower'),
        ConductingPlane((0.,0.,.2),(0.,0.,-1.),upper,'upper')))


def quick(spec, backend, *, plane=None):
    return sample_position_sweep(spec,backend,plane=plane,
                                 resolution=9,line_steps=15,seeds=[],profile_samples=9)


def test_grounded_plane_exact_force_work_and_legacy_viewer_fields():
    start=.04;end=.13
    p=ElectrostaticProblem((PointCharge(Q,(0,0,start)),),(ConductingPlane(name='lower'),))
    original=p
    spec=PositionSweepSpec(p,0,(0.,0.,end),steps=5)
    output=quick(spec,GroundedPlaneBackend())
    assert p is original and p.charges[0].position_m==(0.,0.,start)
    series=output['position_sweep']['samples']
    k=1/(4*pi*p.permittivity_f_per_m)
    assert len(series)==5 and len(output['frames'])==5
    for r in series:
        z=r['position_m'][2]
        assert r['force_parallel_n']==pytest.approx(-k*Q*Q/(4*z*z))
        assert r['energy_j']==pytest.approx(-k*Q*Q/(4*z))
        assert r['direct_force_n']==[0.,0.,0.]
        assert r['background_force_n']==[0.,0.,0.]
        assert r['force_parallel_n']==pytest.approx(r['force_n'][2])
        assert r['charge_c']==Q and r['grouped_index']==0
        assert r['energy_reference']
    assert series[0]['external_quasistatic_work_j']==0.
    assert series[-1]['external_quasistatic_work_j']==pytest.approx(
        series[-1]['energy_j']-series[0]['energy_j'])
    assert output['playback'].lower().find('not elapsed time')>=0
    assert output['schema']=='conductor_sim.visualization.v1'
    assert output['units']['force']=='N'
    assert render_html(output).count('force-panel')>=1
    json.dumps(output,allow_nan=False)


@pytest.mark.parametrize('backend,charges,end',[
    (ParallelPlateBackend(image_order=24),(PointCharge(Q,(0.,0.,.04)),),(0.,0.,.16)),
    (ParallelPlateBackend(image_order=24),(PointCharge(-Q,(0.,0.,.04)),),(0.,0.,.16)),
])
def test_parallel_plate_grounded_midpoint_and_force_energy_derivative(backend,charges,end):
    spec=PositionSweepSpec(plates(charges),0,end,steps=7)
    bundle=quick(spec,backend)
    samples=bundle['position_sweep']['samples']
    assert samples[3]['force_parallel_n']==pytest.approx(0.,abs=1e-18)
    assert samples[0]['force_parallel_n']<0 and samples[-1]['force_parallel_n']>0
    assert samples[3]['energy_j']>samples[0]['energy_j']
    assert samples[3]['energy_j']==pytest.approx(samples[0]['energy_j']-samples[0]['external_quasistatic_work_j']+samples[3]['external_quasistatic_work_j'])
    for i in range(1,len(samples)-1):
        left,right=samples[i-1],samples[i+1]
        gradient=-(right['energy_j']-left['energy_j'])/(right['coordinate_m']-left['coordinate_m'])
        # Coarse finite differences near plates only approximate the slope.
        assert samples[i]['force_parallel_n']==pytest.approx(gradient,rel=.30,abs=2e-8)
    assert bundle['position_sweep']['conditions'].startswith('One charge moved')


def test_voltage_bias_energy_work_uses_full_background_term():
    backend=ParallelPlateBackend(image_order=48)
    spec=PositionSweepSpec(plates((PointCharge(Q,(0,0,.06)),),upper=FixedPotential(20)),0,
                           (0,0,.08),steps=5)
    output=quick(spec,backend)
    rows=output['position_sweep']['samples']
    assert rows[0]['energy_background_j']==pytest.approx(Q*20*.06/.2)
    assert rows[-1]['energy_background_j']==pytest.approx(Q*20*.08/.2)
    assert rows[0]['background_force_n'][2]==pytest.approx(-Q*100)
    assert rows[-1]['external_quasistatic_work_j']==pytest.approx(
        force_energy(backend.solve(plates((PointCharge(Q,(0,0,.08)),),upper=FixedPotential(20)))).energy.total_j -
        force_energy(backend.solve(plates((PointCharge(Q,(0,0,.06)),),upper=FixedPotential(20)))).energy.total_j)


@pytest.mark.parametrize('fixed', [False,True])
def test_external_sphere_two_constraint_cases_reuse_same_api(fixed):
    p=ElectrostaticProblem((PointCharge(Q,(0,0,.17)),),(
        SolidSphere(.1,FixedNetCharge(2e-9) if fixed else Grounded()),))
    bundle=quick(PositionSweepSpec(p,0,(0,0,.29),steps=3),ExternalSphereBackend())
    for row in bundle['position_sweep']['samples']:
        z=row['position_m'][2]
        direct=force_energy(ExternalSphereBackend().solve(ElectrostaticProblem(
            (PointCharge(Q,(0,0,z)),),p.conductors)))
        assert row['energy_j']==pytest.approx(direct.energy.total_j)
        assert row['force_n']==pytest.approx(direct.charges[0].total_force_n)
    assert bundle['position_sweep']['authored_charge_index']==0


def test_authored_charge_tracking_survives_sorted_order_swap_without_collision():
    charges=(PointCharge(Q,(-.03,0,.05)), PointCharge(-.3*Q,(0,0,.09)))
    spec=PositionSweepSpec(plates(charges),0,(.03,0,.15),steps=5)
    bundle=quick(spec,ParallelPlateBackend(image_order=18))
    rows=bundle['position_sweep']['samples']
    assert [r['grouped_index'] for r in rows][0] != [r['grouped_index'] for r in rows][-1]
    assert all(r['charge_c']==Q for r in rows)
    assert all(r['position_m']==list(spec.problem.charges[0].position_m) or
               r['position_m']!=list(spec.problem.charges[1].position_m) for r in rows)
    assert all(row['force_parallel_n']==pytest.approx(sum(row['force_n'][i]*spec.direction[i]
                                                          for i in range(3))) for row in rows)
    assert abs(sweep_slice(spec).project(spec.end_m)[2])<1e-12


def test_rotated_grounded_plane_path_and_projection():
    p=ElectrostaticProblem((PointCharge(Q,(.08,.02,0.)),),(
        ConductingPlane((0,0,0), (1,0,0),Grounded(),'side'),))
    spec=PositionSweepSpec(p,0,(.15,.045,.02),steps=3)
    plane=sweep_slice(spec,resolution=11)
    assert plane.contains(plane.project(spec.end_m)[:2])
    assert abs(plane.project(spec.end_m)[2]) < 1e-12
    out=quick(spec,GroundedPlaneBackend(),plane=plane)
    assert len(out['position_sweep']['samples'])==3
    assert all(row['charge_c']==Q for row in out['position_sweep']['samples'])


def test_sweep_rejects_invalid_authored_identity_and_physical_collisions():
    p=plates((PointCharge(Q,(0,0,.04)),PointCharge(-Q,(0,0,.1))))
    with pytest.raises(UnsupportedProblem,match='intersects another'):
        PositionSweepSpec(p,0,(0,0,.17))
    with pytest.raises(TypeError):
        PositionSweepSpec(p,True,(0,0,.06))
    with pytest.raises(IndexError):
        PositionSweepSpec(p,5,(0,0,.06))
    with pytest.raises(ValueError):
        PositionSweepSpec(p,0,(0,0,.04))
    with pytest.raises(ValueError):
        PositionSweepSpec(p,0,(0,0,.07),steps=25)
    with pytest.raises(UnsupportedProblem,match='vacuum'):
        PositionSweepSpec(plates((PointCharge(Q,(0,0,.05)),)),0,(0,0,.25))
    with pytest.raises(UnsupportedProblem,match='solid conductor'):
        PositionSweepSpec(ElectrostaticProblem((PointCharge(Q,(0,0,.2)),),(
            SolidSphere(.1,Grounded()),)),0,(0,0,-.2))
    with pytest.raises(UnsupportedProblem,match='zero-charge'):
        PositionSweepSpec(plates((PointCharge(0,(0,0,.04)),)),0,(0,0,.07))
    with pytest.raises(ValueError):
        PositionSweepSpec(plates((PointCharge(Q,(0,0,.04)),)),0,(float('inf'),0,.05))
    with pytest.raises(UnsupportedProblem,match='intersects another'):
        PositionSweepSpec(plates((PointCharge(Q,(0,0,.04)), PointCharge(-Q,(0,0,.04)))),0,(0,0,.06))


def test_refuse_numerical_backend_without_silent_analytic_substitution_and_close():
    spec=PositionSweepSpec(plates((PointCharge(Q,(0,0,.05)),)),0,(0,0,.08),steps=3)
    counts={'solves':0,'closes':0}
    class Unobserved:
        conductor_results=()
        def close(self):counts['closes']+=1
    class Backend:
        def supports(self,p):return True
        def solve(self,p):
            counts['solves']+=1
            return Unobserved()
    with pytest.raises(UnsupportedProblem,match='force/energy'):
        quick(spec,Backend())
    assert counts=={'solves':1,'closes':1}
    class UnsupportedBackend:
        def supports(self,p):return False
        def solve(self,p):raise AssertionError('should reject before solve')
    with pytest.raises(UnsupportedProblem,match='does not support'):
        quick(spec,UnsupportedBackend())
    assert counts=={'solves':1,'closes':1}


def test_legacy_viewer_is_unchanged_and_corrupt_sweep_refused():
    from conductor_sim.visualization import sample_sweep
    plain=sample_sweep([('legacy',plates((PointCharge(Q,(0,0,.05)),)))],
                       ParallelPlateBackend(image_order=8),resolution=9,seeds=[],line_steps=5)
    html=render_html(plain)
    assert 'id="force-panel" hidden' in html
    assert '"position_sweep"' not in html
    spec=PositionSweepSpec(plates((PointCharge(Q,(0,0,.05)),)),0,(0,0,.09),steps=3)
    data=quick(spec,ParallelPlateBackend(image_order=8))
    data['position_sweep']['samples'].pop()
    with pytest.raises(ValueError,match='samples must match'):
        render_html(data)


def test_reject_out_of_slice_end_before_solving():
    spec=PositionSweepSpec(plates((PointCharge(Q,(0,0,.05)),)),0,(0,0,.15),steps=3)
    fixed=SliceSpec((0,0,0),(1,0,0),(0,1,0),(-.3,.3),(-.3,.3),9)
    class GuardBackend:
        def solve(self,p):raise AssertionError('must not solve wrong slice')
    with pytest.raises(UnsupportedProblem,match='fixed slice'):
        quick(spec,GuardBackend(),plane=fixed)
