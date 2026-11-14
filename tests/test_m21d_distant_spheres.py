"""Independent JEE relations, reusable grouping and strict approximation boundaries."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from math import fsum, isclose, pi
import json
import pytest

from conductor_sim import (EPSILON_0, ConnectionPlan, ElectricalLink, ElectrostaticProblem,
                           FixedNetCharge, Grounded, SolidSphere, SphericalShell, PointCharge, ConductingPlane)
from conductor_sim.distant_spheres import APPROXIMATION, DistantSphereConnectionBackend
from conductor_sim.distant_sphere_viewer import render_distant_spheres, write_distant_spheres
from conductor_sim.solver import UnsupportedProblem
from examples.distant_connected_spheres import make_case, CASES

K=1/(4*pi*EPSILON_0)
BACKEND=DistantSphereConnectionBackend()


def plan(q=(1e-9,2e-9),r=(.1,.2),d=20.,eps=EPSILON_0):
    return ConnectionPlan(ElectrostaticProblem(conductors=tuple(
        SolidSphere(rad,FixedNetCharge(charge),(i*d,0.,0.),f's{i}')
        for i,(rad,charge) in enumerate(zip(r,q))),permittivity_f_per_m=eps),
        (ElectricalLink(('s0','s1')),))


def solve(p):
    return BACKEND.solve(p,approximation=APPROXIMATION)


@pytest.mark.parametrize('case',CASES)
def test_source_cases_conserve_groups_and_positive_release(case):
    result,_=make_case(case)
    assert result.before.total_charge_c==pytest.approx(result.after.total_charge_c,abs=1e-24)
    assert result.released_field_energy_j>=0
    for link in result.plan.links:
        before=[result.before.by_name(n) for n in link.members]
        after=[result.after.by_name(n) for n in link.members]
        assert fsum(x.charge_c for x in after)==pytest.approx(fsum(x.charge_c for x in before),abs=1e-24)
        assert all(isclose(x.potential_v,after[0].potential_v,rel_tol=1e-13,abs_tol=1e-13) for x in after)
    assert result.as_dict()['qualification']['is_relative_error_bound'] is False


def test_2011_official_all_four_relations_and_shapes():
    r,src=make_case('jee2011');a,b=r.after.conductors
    assert a.kind=='SphericalShell' and b.kind=='SolidSphere'
    assert a.interior_field_v_m==b.interior_field_v_m==0
    assert a.inner_surface_charge_c==0
    assert a.charge_c>b.charge_c
    assert a.mean_outer_sigma_c_m2/b.mean_outer_sigma_c_m2==pytest.approx(b.outer_radius_m/a.outer_radius_m)
    assert a.outer_field_normal_v_m<b.outer_field_normal_v_m
    assert a.charge_c==pytest.approx(4e-9/3,rel=1e-14)
    assert b.charge_c==pytest.approx(2e-9/3,rel=1e-14)
    assert 'Q31' in src['official_reference']


def test_notes_literal_millicoulombs_and_release():
    r,source=make_case('notes-sharing')
    assert [x.charge_c for x in r.before.conductors]==[.5e-3,.7e-3]
    assert [x.charge_c for x in r.after.conductors]==pytest.approx([.4e-3,.8e-3],rel=1e-14)
    # Independent two-capacitor identity: 1/2 C1 C2/(C1+C2) (V1-V2)^2.
    c1,c2=4*pi*EPSILON_0*.1,4*pi*EPSILON_0*.2
    expected=.5*c1*c2/(c1+c2)*(K*(.5e-3/.1-.7e-3/.2))**2
    assert r.released_field_energy_j==pytest.approx(expected,rel=2e-14)
    assert r.released_field_energy_j==pytest.approx(674.0663844196,rel=1e-12)
    assert 'breakdown' in source['additional_choices']


def test_equal_density_and_surface_ratio():
    r,_=make_case('equal-density')
    assert r.after.by_name('large').mean_outer_sigma_c_m2/1e-8==pytest.approx(5/6)
    assert r.after.by_name('small').mean_outer_sigma_c_m2/1e-8==pytest.approx(5/3)
    r,_=make_case('surface-ratio');a,b=r.after.conductors
    assert a.outer_field_normal_v_m/b.outer_field_normal_v_m==pytest.approx(2.)
    assert r.released_field_energy_j==pytest.approx(0.,abs=1e-30)


@pytest.mark.parametrize('q',[(0.,0.),(1e-9,-1e-9),(-1e-9,-2e-9),(0.,3e-9),(9e-9,-3e-9),(-1e-9,2e-9)])
@pytest.mark.parametrize('radii',[(.1,.1),(.1,.2),(.07,.21)])
def test_signed_and_zero_charge_analytic_reference(q,radii):
    result=solve(plan(q=q,r=radii))
    qf=[sum(q)*x/sum(radii) for x in radii]
    assert [x.charge_c for x in result.after.conductors]==pytest.approx(qf,rel=1e-14,abs=1e-25)
    initial=.5*K*sum(x*x/r for x,r in zip(q,radii))
    final=.5*K*sum(q)**2/sum(radii)
    assert result.before.self_field_energy_j==pytest.approx(initial,rel=2e-14)
    assert result.after.self_field_energy_j==pytest.approx(final,rel=2e-14,abs=1e-30)
    release=.5*K*radii[0]*radii[1]/sum(radii)*(q[0]/radii[0]-q[1]/radii[1])**2
    assert result.released_field_energy_j==pytest.approx(release,rel=2e-13,abs=1e-30)


def test_groups_unlinked_and_disconnected_continuation():
    p=plan(q=(1e-9,2e-9,-3e-9,4e-9,8e-9),r=(.1,.2,.15,.1,.2))
    p=replace(p,links=(ElectricalLink(('s0','s2')),ElectricalLink(('s1','s3'))))
    before=deepcopy(p);r=solve(p)
    assert p==before
    assert r.after.by_name('s4').charge_c==8e-9
    assert all(a.name==b.name and a.center_m==b.center_m for a,b in zip(p.initial.conductors,r.resolved_problem.conductors))
    # Existing named parts can be disconnected and reconnected, not mutated.
    second=solve(ConnectionPlan(r.resolved_problem,(ElectricalLink(('s0','s4')),)))
    assert second.before.by_name('s0').charge_c==r.after.by_name('s0').charge_c
    assert second.after.by_name('s1').charge_c==r.after.by_name('s1').charge_c
    assert p==before
    with pytest.raises(FrozenInstanceError):
        r.released_field_energy_j=0.
    audit=r.as_dict();audit['after']['conductors'][0]['charge_c']=42
    assert r.after.conductors[0].charge_c!=42


def test_ordering_rotation_translation_and_permittivity():
    p=plan(q=(1e-9,3e-9));r=solve(p)
    reordered=solve(replace(p,initial=replace(p.initial,conductors=p.initial.conductors[::-1])))
    for name in ('s0','s1'):
        assert r.after.by_name(name)==reordered.after.by_name(name)
    transformed=tuple(replace(c,center_m=(3.,4.+c.center_m[0],-2.)) for c in p.initial.conductors)
    rt=solve(replace(p,initial=replace(p.initial,conductors=transformed)))
    assert rt.released_field_energy_j==r.released_field_energy_j
    eps=solve(replace(p,initial=replace(p.initial,permittivity_f_per_m=2*EPSILON_0)))
    assert [c.charge_c for c in eps.after.conductors]==[c.charge_c for c in r.after.conductors]
    assert eps.released_field_energy_j==pytest.approx(r.released_field_energy_j/2)
    assert eps.after.conductors[0].potential_v==pytest.approx(r.after.conductors[0].potential_v/2)


def test_omitted_effects_exposed_not_added_to_energy():
    r=solve(plan(q=(0.,1e-9)))
    assert r.before.conductors[0].potential_v==0
    assert r.before.omitted_monopole_potential_estimate_v[0]==pytest.approx(K*1e-9/20)
    assert r.after.omitted_pair_energy_abs_scale_j>0
    assert not hasattr(r.after,'potential_at') and not hasattr(r.after,'electric_field_at')
    r2=solve(plan(q=(0.,1e-9),d=40))
    assert r2.released_field_energy_j==r.released_field_energy_j
    assert r2.after.omitted_pair_energy_abs_scale_j==pytest.approx(r.after.omitted_pair_energy_abs_scale_j/2)


@pytest.mark.parametrize('distance',[0.,.1,.3,1.,3.9])
def test_close_nested_touching_refused(distance):
    p=plan(d=distance)
    assert BACKEND.supports(p,approximation=APPROXIMATION) is False
    with pytest.raises(UnsupportedProblem): solve(p)


def test_geometry_collective_row_sum_not_only_pairwise():
    # Individually R/d < .05, but the sum from multiple bodies exceeds policy.
    p=plan(q=(1e-9,)*8,r=(.2,)*8,d=5.)
    with pytest.raises(UnsupportedProblem,match='far-separation'): solve(p)


@pytest.mark.parametrize('change',[
    lambda p:replace(p,links=(ElectricalLink(('s0','unknown')),)),
    lambda p:replace(p,links=(ElectricalLink(('s0','s1'),'grounded'),)),
    lambda p:replace(p,links=p.links+p.links),
    lambda p:replace(p,initial=replace(p.initial,charges=(PointCharge(1e-9,(0,0,5)),))),
    lambda p:replace(p,initial=replace(p.initial,conductors=(replace(p.initial.conductors[0],constraint=Grounded()),p.initial.conductors[1]))),
    lambda p:replace(p,initial=replace(p.initial,conductors=(ConductingPlane(),p.initial.conductors[1]))),
])
def test_unqualified_relationships_rejected(change):
    p=change(plan())
    with pytest.raises(UnsupportedProblem): solve(p)
    assert not BACKEND.supports(p,approximation=APPROXIMATION)


@pytest.mark.parametrize('approximation',[None,'exact','monopole-with-coupling','automatic'])
def test_explicit_approximation_required(approximation):
    with pytest.raises(UnsupportedProblem): BACKEND.solve(plan(),approximation=approximation)
    with pytest.raises(TypeError): BACKEND.solve(plan())


def test_finite_and_resolved_numerical_domain():
    p=plan(q=(1e308,1e308))
    with pytest.raises(ValueError): solve(p)
    p=plan(q=(True,1e-9))
    with pytest.raises(ValueError): solve(p)
    p=plan();p=replace(p,initial=replace(p.initial,conductors=tuple(replace(c,center_m=(1e20+i*1e6,0,0)) for i,c in enumerate(p.initial.conductors))))
    with pytest.raises(UnsupportedProblem,match='unresolved'): solve(p)


def test_small_energy_release_survives_large_absolute_energy():
    p=plan(q=(1.,2.+1e-8));r=solve(p)
    assert r.released_field_energy_j>1e-8
    c1,c2=[c.capacitance_f for c in r.before.conductors]
    expect=.5*c1*c2/(c1+c2)*(r.before.conductors[0].potential_v-r.before.conductors[1].potential_v)**2
    assert r.released_field_energy_j==pytest.approx(expect,rel=1e-6)
    assert r.before.self_field_energy_j > 1e10


@pytest.mark.parametrize('case',CASES)
def test_offline_viewer_and_verified_data(tmp_path,case):
    r,source=make_case(case)
    path=write_distant_spheres(r,tmp_path/(case+'.html'),title='M21d '+case,provenance=source)
    text=path.read_text();record=json.loads(path.with_suffix('.json').read_text())
    assert 'Leading self-capacitance approximation' in text
    assert 'not a discharge animation' in text
    assert 'hasTimeEvolution:false' in text
    assert '<script src=' not in text and 'fetch(' not in text
    assert record['before']==r.as_dict()['before']
    bad=replace(r,released_field_energy_j=r.released_field_energy_j+1.)
    with pytest.raises(ValueError,match='recomputed'):
        render_distant_spheres(bad,title='invalid',provenance={})


def test_safe_authored_text():
    r,_=make_case('jee2011')
    html=render_distant_spheres(r,title='<script>alert(1)</script>',provenance={'note':'</script><script>alert(2)</script>'})
    assert '<script>alert(1)' not in html and '<script>alert(2)' not in html
    assert '\\u003c/script\\u003e' in html


def test_eight_conductors_many_groups_and_distant_requalification():
    p=plan(q=tuple(i*1e-9 for i in range(8)),r=(.1,)*8,d=100.)
    p=replace(p,links=tuple(ElectricalLink((f's{i}',f's{i+1}')) for i in range(0,8,2)))
    result=solve(p)
    assert len(result.after.conductors)==8 and len(result.plan.links)==4
    html=render_distant_spheres(result,title='Eight conductors',provenance={})
    assert 'Math.max(700,220*n)' in html
    assert all(name in html for name in result.as_dict()['conductor_ids'])
    changed=replace(p,initial=replace(p.initial,conductors=tuple(
        replace(c,center_m=(i*.5,0,0)) for i,c in enumerate(p.initial.conductors))))
    with pytest.raises(UnsupportedProblem): solve(changed)
    assert p.initial.conductors[1].center_m==(100.,0.,0.)
