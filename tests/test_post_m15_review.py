"""Focused reproductions from the post-M15 correctness review."""
from dataclasses import replace
from math import fsum, pi
import pytest
from conductor_sim import (
    ConductingPlane, ElectrostaticProblem, FixedNetCharge, Grounded,
    MultiImageShellBackend, NestedShellBackend, PointCharge, PositionSweepSpec,
    SolidSphere, SphericalShell, force_energy, sample_position_sweep,
)
from conductor_sim.position_sweep import sweep_slice

K=1/(4*pi*8.8541878128e-12)


@pytest.mark.parametrize('conductors',[
    (ConductingPlane(),),
    (SphericalShell(.1,.12,Grounded(),name='cavity'),ConductingPlane(name='plane')),
    (SolidSphere(.1,Grounded()),),
])
def test_nested_capability_query_refuses_other_shapes_instead_of_crashing(conductors):
    from conductor_sim.solver import UnsupportedProblem
    problem=ElectrostaticProblem(conductors=conductors)
    assert NestedShellBackend().supports(problem) is False
    with pytest.raises(UnsupportedProblem):
        NestedShellBackend().solve(problem)


def test_default_cavity_slice_includes_all_shells_independent_of_author_order():
    shells=(SphericalShell(.1,.12,FixedNetCharge(0),name='inner'),
            SphericalShell(.5,.55,Grounded(),name='outer'))
    p=ElectrostaticProblem((PointCharge(1e-9,(0,0,.02)),),shells)
    spec=PositionSweepSpec(p,0,(0,0,.06),steps=3)
    first=sweep_slice(spec)
    second=sweep_slice(replace(spec,problem=replace(p,conductors=shells[::-1])))
    assert first==second
    for c in shells:
        x,y,_=first.project(c.center_m)
        for xy in ((x+c.outer_radius_m,y),(x-c.outer_radius_m,y),
                   (x,y+c.outer_radius_m),(x,y-c.outer_radius_m)):
            assert first.contains(xy)


def test_default_plate_sweep_frame_does_not_depend_on_arbitrary_tangential_anchor():
    p=ElectrostaticProblem((PointCharge(1e-9,(.03,0,.06)),),
                          (ConductingPlane(),))
    spec=PositionSweepSpec(p,0,(.03,0,.12),steps=3)
    other=replace(spec,problem=replace(p,conductors=(ConductingPlane((1e6,0,0)),)))
    a,b=sweep_slice(spec),sweep_slice(other)
    assert b.x_range_m==pytest.approx(a.x_range_m)
    assert b.y_range_m==pytest.approx(a.y_range_m)
    assert b.origin_m==pytest.approx(a.origin_m)


@pytest.mark.parametrize('nested', [False,True])
@pytest.mark.parametrize('shell_charge',[0.,1e9,-1e9])
def test_cavity_induced_energy_is_not_lost_when_subtracting_large_background(nested,shell_charge):
    # Deliberate dynamic-range stress, not a physical field-strength claim.
    q,a,r=1e-9,.1,.03
    shells=[SphericalShell(a,.2,FixedNetCharge(shell_charge),name='inner')]
    if nested:
        shells.append(SphericalShell(.3,.4,Grounded(),name='outer'))
    p=ElectrostaticProblem((PointCharge(q,(0,0,r)),),shells)
    backend=NestedShellBackend() if nested else MultiImageShellBackend()
    obs=force_energy(backend.solve(p))
    response=(1/.2-1/.3) if nested else 1/.2
    expected=-K*q*q*a/(2*(a*a-r*r))+.5*K*q*q*response
    assert obs.energy.induced_j==pytest.approx(expected,rel=2e-13,abs=1e-22)
    assert obs.energy.background_j==pytest.approx(q*K*shell_charge*response)
    assert obs.charges[0].total_force_n[2]==pytest.approx(K*q*q*a*r/(a*a-r*r)**2)


def test_displacement_work_does_not_disappear_beneath_a_large_constant_energy():
    q,a=1e-9,.1
    p=ElectrostaticProblem((PointCharge(q,(0,0,.02)),),
          (SphericalShell(a,.12,FixedNetCharge(1e9)),))
    spec=PositionSweepSpec(p,0,(0,0,.06),steps=3)
    data=sample_position_sweep(spec,MultiImageShellBackend(),resolution=9,
                              line_steps=5,seeds=[],profile_samples=9)
    first,last=data['position_sweep']['samples'][::2]
    expected=-K*q*q*a/2*(1/(a*a-.06**2)-1/(a*a-.02**2))
    assert last['external_quasistatic_work_j']==pytest.approx(expected,rel=2e-12,abs=1e-22)
    assert first['external_quasistatic_work_j']==0.


@pytest.mark.parametrize('constraints',[
    (FixedNetCharge(2e-9),FixedNetCharge(-3e-9),Grounded()),
    (FixedNetCharge(2e-9),Grounded(),FixedNetCharge(4e-9)),
    (Grounded(),FixedNetCharge(-3e-9),FixedNetCharge(4e-9)),
    (FixedNetCharge(2e-9),FixedNetCharge(-3e-9),FixedNetCharge(4e-9)),
])
def test_uniform_voltage_decomposition_matches_solved_response_at_ordinary_scales(constraints):
    radii=((.1,.12),(.2,.22),(.3,.32))
    shapes=tuple(SphericalShell(a,b,c,name=f's{i}')
                 for i,((a,b),c) in enumerate(zip(radii,constraints)))
    charges=(PointCharge(1e-9,(0,0,.02)),PointCharge(-.2e-9,(.03,0,-.02)))
    p=ElectrostaticProblem(charges,shapes)
    solver=NestedShellBackend()
    sol=solver.solve(p)
    empty=solver.solve(replace(p,charges=()))
    observed=force_energy(sol)
    q=fsum(c.charge_c for c in charges)
    assert observed.energy.background_j==pytest.approx(q*empty.conductor_by_name('s0').potential_v,abs=1e-21)
    grounded=force_energy(MultiImageShellBackend().solve(
        replace(p,conductors=(replace(shapes[0],constraint=Grounded()),))))
    expected=grounded.energy.induced_j+.5*q*(sol.conductor_by_name('s0').potential_v-empty.conductor_by_name('s0').potential_v)
    assert observed.energy.induced_j==pytest.approx(expected,rel=2e-12,abs=1e-21)
