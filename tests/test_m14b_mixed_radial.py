"""M14b exact mixed conducting sphere/shell physics and connection regressions."""
from dataclasses import replace
from math import pi
import random
import json
import pytest

from conductor_sim import (ConnectionPlan, ConnectedShellBackend, ElectricalLink,
    ElectrostaticProblem, FixedNetCharge, Grounded, MixedRadialBackend,
    NestedShellBackend, PointCharge, SolidSphere, SphericalShell)
from conductor_sim.model import EPSILON_0
from conductor_sim.solver import SingularField, UnsupportedProblem
from conductor_sim.visual_adapters import context_for_solution
from conductor_sim.visualization import write_visualization
from examples.solid_sphere_connected import scenario

K=1/(4*pi*EPSILON_0)
Q=1e-9


def model(q0=Q,q1=0.,q2=None,center=(0.,0.,0.)):
    shapes=[SolidSphere(.1,FixedNetCharge(q0),center,'solid'),
            SphericalShell(.19,.2,FixedNetCharge(q1),center,'middle')]
    if q2 is not None:
        shapes.append(SphericalShell(.3,.32,FixedNetCharge(q2),center,'outer'))
    return ElectrostaticProblem(conductors=shapes)


def wire(p,names=('solid','middle'),mode='floating'):
    return ConnectedShellBackend().solve(ConnectionPlan(p,(ElectricalLink(names,mode),)))


def test_no_phantom_cavity_and_gauss_surface_charge_and_field_gradients():
    p=model(1.5*Q,-.25*Q)
    sol=MixedRadialBackend().solve(p)
    solid,shell=sol.conductor_results
    assert solid.inner_surface_charge_c==0.
    assert solid.outer_surface_charge_c==pytest.approx(1.5*Q)
    assert shell.inner_surface_charge_c==pytest.approx(-1.5*Q)
    assert shell.outer_surface_charge_c==pytest.approx(1.25*Q)
    assert solid.potential_v==pytest.approx(K*(1.5*Q*(1/.1-1/.19)+1.25*Q/.2))
    assert shell.potential_v==pytest.approx(K*1.25*Q/.2)
    assert sol.electric_field_at((0,0,0))==(0.,0.,0.)
    assert sol.electric_field_at((0,0,.195))==(0.,0.,0.)
    assert sol.electric_field_at((0,0,.15))==pytest.approx((0.,0.,K*1.5*Q/.15**2))
    assert sol.electric_field_at((0,0,.3))==pytest.approx((0.,0.,K*1.25*Q/.3**2))
    assert sol.surface_charge_density_at('solid','outer',(.1,0,0))==pytest.approx(1.5*Q/(4*pi*.1**2))
    assert sol.surface_charge_density_at('middle','inner',(.19,0,0))==pytest.approx(-1.5*Q/(4*pi*.19**2))
    for r in (.1,.19,.2):
        with pytest.raises(SingularField):
            sol.electric_field_at((r,0,0))
    with pytest.raises(ValueError,match='no inner surface'):
        sol.surface_charge_density_at('solid','inner',(.1,0,0))
    for r in (.14,.25):
        h=1e-5
        grad=-(sol.potential_at((r+h,0,0))-sol.potential_at((r-h,0,0)))/(2*h)
        assert sol.electric_field_at((r,0,0))[0]==pytest.approx(grad,rel=2e-8)


def test_pdf_page3_solid_sphere_and_shell_release_nine_joules():
    plan=scenario('pdf-p03-q27')
    result=ConnectedShellBackend().solve(plan)
    exact=K*(20e-6)**2/2*(1/.1-1/.2)
    assert exact==pytest.approx(8.987551792,rel=2e-10)
    assert result.released_field_energy_j==pytest.approx(exact,rel=1e-5)
    assert result.released_field_energy_j==pytest.approx(9.,abs=.02)
    assert result.after.conductor_by_name('solid').net_charge_c==pytest.approx(0,abs=1e-18)
    assert result.after.conductor_by_name('shell').net_charge_c==pytest.approx(20e-6)
    for r in (.12,.15,.19):
        assert result.after.electric_field_at((r,0,0))==(0.,0.,0.)
    for r in (.25,.3):
        assert result.after.electric_field_at((r,0,0))==pytest.approx(result.before.electric_field_at((r,0,0)))


def test_pdf_page10_shell_only_reference_reuses_m14a_without_fake_solid():
    plan=scenario('pdf-p10-q13-q14')
    assert NestedShellBackend().supports(plan.initial)
    result=ConnectedShellBackend().solve(plan)
    q=plan.initial.conductors[0].constraint.charge_c
    assert K*q/.1**2==pytest.approx(1.8e6)
    assert result.released_field_energy_j==pytest.approx(.09,rel=.003)
    assert result.radial_field_energy_after_j==pytest.approx(0,abs=1e-20)
    assert all(abs(r.net_charge_c)<1e-17 for r in result.after.conductor_results)


def test_grounded_shell_reservoir_and_grounded_solid():
    p=ElectrostaticProblem(conductors=(SolidSphere(.1,FixedNetCharge(Q),name='solid'),
            SphericalShell(.19,.2,Grounded(),name='middle')))
    initial=MixedRadialBackend().solve(p)
    assert initial.conductor_by_name('middle').net_charge_c==pytest.approx(-Q)
    result=wire(p,mode='grounded')
    assert all(abs(r.potential_v)<1e-10 for r in result.after.conductor_results)
    assert all(abs(r.net_charge_c)<1e-18 for r in result.after.conductor_results)
    assert result.released_field_energy_j==pytest.approx(K*Q**2/2*(1/.1-1/.19))
    p2=ElectrostaticProblem(conductors=(SolidSphere(.1,Grounded(),name='solid'),
            SphericalShell(.19,.2,FixedNetCharge(2*Q),name='middle')))
    sol=MixedRadialBackend().solve(p2)
    qs=-2*Q*(1/.2)/(1/.1-1/.19+1/.2)
    assert sol.conductor_by_name('solid').net_charge_c==pytest.approx(qs)
    assert sol.conductor_by_name('solid').potential_v==pytest.approx(0,abs=1e-12)
    assert not ConnectedShellBackend().supports(ConnectionPlan(p2,(ElectricalLink(('solid','middle')),)))


def test_nonadjacent_link_reuses_general_capacitance_matrix():
    result=wire(model(1.2*Q,.4*Q,-.3*Q),('solid','outer'))
    assert result.after.conductor_by_name('middle').net_charge_c==pytest.approx(.4*Q)
    assert result.after.conductor_by_name('solid').potential_v==pytest.approx(
        result.after.conductor_by_name('outer').potential_v)
    assert (result.after.conductor_by_name('solid').net_charge_c+
            result.after.conductor_by_name('outer').net_charge_c)==pytest.approx(.9*Q)
    assert result.released_field_energy_j>=0


def test_translation_scaling_and_authored_conductor_order():
    p=model(2*Q,-.3*Q,.1*Q)
    base=MixedRadialBackend().solve(p)
    reverse=MixedRadialBackend().solve(ElectrostaticProblem(conductors=tuple(reversed(p.conductors))))
    assert [r.name for r in reverse.conductor_results]==['outer','middle','solid']
    for c in p.conductors:
        assert base.conductor_by_name(c.name)==reverse.conductor_by_name(c.name)
    shift=(1.,-.4,.6)
    moved=MixedRadialBackend().solve(ElectrostaticProblem(
        conductors=tuple(replace(c,center_m=shift) for c in p.conductors)))
    assert moved.potential_at((1.,-.4,.85))==pytest.approx(base.potential_at((0.,0.,.25)))
    doubled=wire(model(2*Q,-.6*Q))
    nominal=wire(model(Q,-.3*Q))
    assert doubled.released_field_energy_j==pytest.approx(4*nominal.released_field_energy_j)
    perm=ElectrostaticProblem(nominal.plan.initial.charges,nominal.plan.initial.conductors,2*EPSILON_0)
    assert wire(perm).released_field_energy_j==pytest.approx(.5*nominal.released_field_energy_j)


def test_reject_invalid_shapes_and_point_sources_before_numerical_solver():
    valid=model()
    bad=[
        ElectrostaticProblem(conductors=(valid.conductors[0],)),
        ElectrostaticProblem(charges=(PointCharge(Q,(0,0,.16)),),conductors=valid.conductors),
        ElectrostaticProblem(charges=(PointCharge(Q,(0,0,0)),),conductors=valid.conductors),
        ElectrostaticProblem(conductors=(valid.conductors[0],
           replace(valid.conductors[1],inner_radius_m=.09))),
        ElectrostaticProblem(conductors=(valid.conductors[0],
           replace(valid.conductors[1],center_m=(.01,0,0)))),
        ElectrostaticProblem(conductors=(valid.conductors[0],
           SolidSphere(.25,FixedNetCharge(0),name='another'))),
    ]
    for p in bad:
        assert not MixedRadialBackend().supports(p)
        with pytest.raises(UnsupportedProblem):
            MixedRadialBackend().solve(p)
        if len(p.conductors)>=2:
            assert not ConnectedShellBackend().supports(ConnectionPlan(p,(
                ElectricalLink((p.conductors[0].name,p.conductors[1].name)),)))


def test_viewer_reuses_solid_faces_and_existing_static_state_schema(tmp_path):
    p=model()
    result=wire(p)
    ctx=context_for_solution(p,MixedRadialBackend().solve(p))
    assert ctx.density_at is not None
    assert ctx.region((0,0,.05))=='metal' and ctx.region((0,0,.15))=='vacuum'
    bundle=result.visualization(resolution=13,line_steps=22,seeds=[],profile_samples=13)
    assert len(bundle['frames'])==2
    for f in bundle['frames']:
        assert [g['inner_radius_m'] for g in f['geometry']]==pytest.approx([0.,.19])
        assert [(r['conductor'],r['surface']) for r in f['profiles']]==[
            ('solid','outer'),('middle','inner'),('middle','outer')]
        assert f['backend']=='MixedRadialSolution'
    assert bundle['connection']['cavity_field_energy_omitted'] is False
    file=tmp_path/'connected.html'
    write_visualization(bundle,file)
    assert 'id="scene"' in file.read_text(encoding='utf-8')
    json.dumps(bundle,allow_nan=False)


def test_energy_is_minimum_at_floating_equilibrium():
    from conductor_sim.connections import _field_energy
    result=wire(model(1.2*Q,.5*Q))
    qi=result.after.conductor_by_name('solid').net_charge_c
    qo=result.after.conductor_by_name('middle').net_charge_c
    for delta in (-.13*Q,-.07*Q,.07*Q,.13*Q):
        alternative=MixedRadialBackend().solve(model(qi+delta,qo-delta))
        assert _field_energy(alternative)>result.radial_field_energy_after_j


@pytest.mark.parametrize('seed',range(5))
def test_random_signed_charge_groups_obey_equal_potential_and_charge_conservation(seed):
    rng=random.Random(seed)
    for _ in range(12):
        x,y,z=[rng.uniform(-3,3)*Q for i in range(3)]
        p=model(x,y,z)
        names=rng.choice([('solid','middle'),('solid','outer'),('middle','outer')])
        result=wire(p,names)
        a,b=[result.after.conductor_by_name(name) for name in names]
        before=sum(result.before.conductor_by_name(n).net_charge_c for n in names)
        assert a.potential_v==pytest.approx(b.potential_v,rel=1e-9,abs=1e-8)
        assert a.net_charge_c+b.net_charge_c==pytest.approx(before,abs=1e-20)
        assert result.released_field_energy_j>=0
