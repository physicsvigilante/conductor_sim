"""M15 independent Kelvin cavity force/energy and reusable sweep contracts.

All reference formulas here are constructed from explicit image positions and
charges, NOT the engine's dimensionless image_term/force_energy helpers.
"""
from __future__ import annotations

from dataclasses import replace
import json
from math import hypot, pi

import pytest

from conductor_sim import (
    EPSILON_0, ElectrostaticProblem, FixedNetCharge, Grounded,
    MultiImageShellBackend, NestedShellBackend, PointCharge, SphericalShell,
    PositionSweepSpec, sample_position_sweep, sweep_slice, force_energy,
    UnsupportedProblem, SingularField,
)
from conductor_sim.visualization import SliceSpec
from conductor_sim.visual_adapters import context_for_solution
from conductor_sim.observables import _cavity_observables

K = 1/(4*pi*EPSILON_0)
Q = 1e-9
A = .1


def one(charges, constraint=Grounded(), *, a=A, b=.12, center=(0.,0.,0.)):
    return ElectrostaticProblem(tuple(charges), (
        SphericalShell(a,b,constraint,center,name='inner'),))


def nested(charges, inner=FixedNetCharge(.4*Q), outer=Grounded(),
           *, center=(0.,0.,0.)):
    return ElectrostaticProblem(tuple(charges), (
        SphericalShell(.1,.12,inner,center,'inner'),
        SphericalShell(.2,.22,outer,center,'outer'),))


def _delta(a,b):
    return tuple(a[i]-b[i] for i in range(3))


def _dot(a,b):
    return sum(a[i]*b[i] for i in range(3))


def _image_reference(point, source, a=A, center=(0.,0.,0.)):
    """Textbook image q'/q=-a/d at y'=c+(a/d)^2(y-c)."""
    y=_delta(source,center)
    d=hypot(*y)
    if d == 0:
        return -1/a, (0.,0.,0.)
    source_image=tuple(center[i]+a*a/(d*d)*y[i] for i in range(3))
    factor=-a/d
    difference=_delta(point,source_image)
    sep=hypot(*difference)
    return factor/sep, tuple(factor*x/sep**3 for x in difference)


def independent(sites, a=A, center=(0.,0.,0.), *, background_v=0., uniform_source_v=0.):
    """Return F vectors, separated energy contributions, no engine solver calls."""
    f=[[0.,0.,0.] for s in sites]
    pair=0.
    for i,p in enumerate(sites):
        for j in range(i+1,len(sites)):
            other=sites[j]
            difference=_delta(p.position_m,other.position_m)
            d=hypot(*difference)
            term=K*p.charge_c*other.charge_c/d
            pair+=term
            for c in range(3):
                value=term*difference[c]/d**2
                f[i][c]+=value
                f[j][c]-=value
    image_energy=0.
    for i,p in enumerate(sites):
        for j,other in enumerate(sites):
            value,e=_image_reference(p.position_m,other.position_m,a,center)
            image_energy+=.5*K*p.charge_c*other.charge_c*value
            for c in range(3):
                f[i][c]+=K*p.charge_c*other.charge_c*e[c]
    qt=sum(p.charge_c for p in sites)
    return f, (pair,image_energy+.5*qt*uniform_source_v,qt*background_v)


def test_single_grounded_offcenter_force_energy_matches_closed_form_and_direction():
    for q in (Q,-Q):
        for d in (.0,.01,.035,.065,.092):
            p=one((PointCharge(q,(0.,0.,d)),))
            backend=MultiImageShellBackend()
            s=backend.solve(p)
            output=force_energy(s)
            assert len(output.charges)==1
            c=output.charges[0]
            assert c.direct_force_n==(0.,0.,0.)
            assert c.background_force_n==(0.,0.,0.)
            assert c.induced_force_n==pytest.approx((0.,0.,K*q*q*A*d/(A*A-d*d)**2),abs=2e-13)
            assert output.energy.induced_j==pytest.approx(-K*q*q*A/(2*(A*A-d*d)))
            assert output.energy.background_j==0.
            assert output.energy.direct_j==0.
            assert output.image_order is None
            if d==0:
                assert c.total_force_n==(0.,0.,0.)
                assert output.energy.total_j==pytest.approx(-K*q*q/(2*A))
            with pytest.raises(SingularField):s.electric_field_at((0.,0.,d))


def test_offcentre_three_dimensional_kelvin_image_is_independent_of_engine_helper():
    c=(.05,-.03,.02)
    p=one((PointCharge(Q,(.073,-.038,.041)),),center=c)
    observed=force_energy(MultiImageShellBackend().solve(p))
    site=p.charges[0]
    v,e=_image_reference(site.position_m,site.position_m,center=c)
    assert observed.charges[0].total_force_n==pytest.approx(tuple(K*Q*Q*x for x in e),rel=2e-13,abs=1e-12)
    assert observed.energy.total_j==pytest.approx(.5*K*Q*Q*v,rel=2e-13)
    # Force must align with its displacement from the cavity center.
    d=_delta(site.position_m,c)
    f=observed.charges[0].total_force_n
    assert all(f[i]*d[j]==pytest.approx(f[j]*d[i],rel=1e-12,abs=1e-16)
               for i in range(3) for j in range(3))


@pytest.mark.parametrize('constraint,expected_background',[
    (Grounded(),0.),
    (FixedNetCharge(2*Q),K*2*Q/.12),
    (FixedNetCharge(-.6*Q),-K*.6*Q/.12),
])
def test_shell_net_charge_is_spatially_uniform_background(constraint,expected_background):
    for z in (.02,.065):
        p=one((PointCharge(Q,(0,0,z)),),constraint)
        o=force_energy(MultiImageShellBackend().solve(p))
        f=o.charges[0]
        assert f.induced_force_n[2]==pytest.approx(K*Q**2*A*z/(A*A-z*z)**2)
        assert f.background_force_n==(0.,0.,0.)
        # At fixed-Q the inner conductor's *change* in voltage is kq/b.
        expected_img=-K*Q**2*A/(2*(A*A-z*z))
        uniform=.5*K*Q*Q/.12 if isinstance(constraint,FixedNetCharge) else 0.
        assert o.energy.induced_j==pytest.approx(expected_img+uniform)
        assert o.energy.background_j==pytest.approx(Q*expected_background)
        assert 'mechanical' in o.energy.reference


def test_multi_source_3d_signed_direct_and_induced_forces_match_image_construction():
    sources=(PointCharge(1.2*Q,(.025,.007,.015)),
             PointCharge(-.65*Q,(-.038,.021,.029)),
             PointCharge(.4*Q,(.01,-.024,-.032)))
    p=one(sources,FixedNetCharge(1.5*Q))
    sol=MultiImageShellBackend().solve(p)
    o=force_energy(sol)
    forces,(pair,images,background)=independent(sol.charges,background_v=K*1.5*Q/.12,
                                                uniform_source_v=K*sum(c.charge_c for c in sources)/.12)
    assert len(o.charges)==3
    for i,c in enumerate(o.charges):
        assert c.total_force_n==pytest.approx(forces[i],rel=1e-12,abs=1e-11)
    assert o.energy.direct_j==pytest.approx(pair,rel=1e-12)
    assert o.energy.induced_j==pytest.approx(images,rel=1e-12)
    assert o.energy.background_j==pytest.approx(background,rel=1e-12)
    assert o.energy.total_j==pytest.approx(sum((pair,images,background)),rel=1e-12)


def test_all_three_energy_components_obey_negative_position_gradient():
    sources=(PointCharge(Q,(.018,.007,.025)),PointCharge(-.6*Q,(-.024,.014,-.035)))
    p=one(sources,FixedNetCharge(1.2*Q))
    b=MultiImageShellBackend()
    o=force_energy(b.solve(p))
    for idx,c in enumerate(o.charges):
        for axis in range(3):
            step=1e-6
            energy=[]
            for t in (-1.,1.):
                q=replace(c.source,position_m=tuple(c.source.position_m[i]+(t*step if i==axis else 0.)
                                                     for i in range(3)))
                sites=list(o.charges[j].source for j in range(len(o.charges)))
                sites[idx]=q
                energy.append(force_energy(b.solve(replace(p,charges=tuple(sites)))).energy)
            for name,force in [('direct_j',c.direct_force_n),('induced_j',c.induced_force_n),
                               ('background_j',c.background_force_n)]:
                gradient=-(getattr(energy[1],name)-getattr(energy[0],name))/(2*step)
                assert force[axis]==pytest.approx(gradient,rel=5e-7,abs=3e-9)
            total_grad=-(energy[1].total_j-energy[0].total_j)/(2*step)
            assert c.total_force_n[axis]==pytest.approx(total_grad,rel=5e-7,abs=3e-9)


def test_nested_shells_change_uniform_energy_not_cavity_induced_force():
    sources=(PointCharge(1.2*Q,(0,.014,.041)),PointCharge(-.5*Q,(.027,0,-.033)))
    shell=one(sources,FixedNetCharge(.4*Q))
    multi=nested(sources,FixedNetCharge(.4*Q),Grounded())
    c=force_energy(MultiImageShellBackend().solve(shell))
    n=force_energy(NestedShellBackend().solve(multi))
    assert len(c.charges)==len(n.charges)==2
    for lhs,rhs in zip(c.charges,n.charges):
        assert lhs.total_force_n==pytest.approx(rhs.total_force_n,rel=1e-13,abs=1e-10)
        assert lhs.induced_force_n==pytest.approx(rhs.induced_force_n,rel=1e-13,abs=1e-10)
    # Source-free shells can have a different uniform potential; it is not a
    # spurious force on a charge somewhere in the cavity.
    assert c.energy.background_j!=pytest.approx(n.energy.background_j,abs=1e-10)
    assert n.energy.reference.startswith('Spherical cavity:')


@pytest.mark.parametrize('inner,outer',[
    (Grounded(),Grounded()),
    (FixedNetCharge(.7*Q),FixedNetCharge(-.3*Q)),
    (Grounded(),FixedNetCharge(-.4*Q)),
    (FixedNetCharge(.5*Q),Grounded()),
])
def test_nested_shell_coupled_uniform_potential_is_independent_of_source_displacement(inner,outer):
    b=NestedShellBackend()
    reports=[]
    for z in (.015,.045,.07):
        p=nested((PointCharge(Q,(0.,0.,z)),),inner,outer)
        s=b.solve(p)
        o=force_energy(s)
        reports.append(o)
        assert o.charges[0].total_force_n[2]==pytest.approx(K*Q**2*A*z/(A*A-z*z)**2)
        assert o.energy.direct_j==0.
    assert reports[0].energy.background_j==pytest.approx(reports[-1].energy.background_j,abs=1e-15)
    for a,b in zip(reports,reports[1:]):
        assert a.energy.background_j==pytest.approx(b.energy.background_j,abs=1e-15)
        # The source-induced constant is also position-independent, so the
        # induced energy difference equals the image-only energy difference.
        expected=-K*Q*Q*A/2*(1/(A*A-.045**2)-1/(A*A-.015**2)) if a==reports[0] else -K*Q*Q*A/2*(1/(A*A-.07**2)-1/(A*A-.045**2))
        assert b.energy.induced_j-a.energy.induced_j==pytest.approx(expected,rel=1e-12,abs=1e-15)


def test_centered_source_and_exactly_canceled_source_do_not_create_singular_forces():
    p=one((PointCharge(Q,(0.,0.,0.)),PointCharge(Q,(.02,0,0)),
           PointCharge(-Q,(.02,0,0))),Grounded())
    s=MultiImageShellBackend().solve(p)
    o=force_energy(s)
    assert len(s.charges)==len(o.charges)==1
    assert o.charges[0].source.position_m==(0.,0.,0.)
    assert o.charges[0].total_force_n==(0.,0.,0.)
    assert o.energy.total_j==pytest.approx(-K*Q**2/(2*A))
    empty=force_energy(MultiImageShellBackend().solve(one((),Grounded())))
    assert empty.charges==() and empty.energy.total_j==0.


def test_nested_source_order_and_world_rotation_do_not_change_observations():
    center=(.8,-.5,.3)
    original=nested((PointCharge(Q,(.027,.01,.018)),PointCharge(-.35*Q,(-.018,.017,.043))),
                    FixedNetCharge(.5*Q),Grounded())
    def trans(p):return (center[0]+p[2],center[1]+p[1],center[2]-p[0])
    transformed=replace(original,
        charges=tuple(PointCharge(c.charge_c,trans(c.position_m)) for c in reversed(original.charges)),
        conductors=tuple(replace(c,center_m=center) for c in reversed(original.conductors)))
    s0=force_energy(NestedShellBackend().solve(original))
    s1=force_energy(NestedShellBackend().solve(transformed))
    assert s1.energy.total_j==pytest.approx(s0.energy.total_j,rel=1e-12,abs=1e-14)
    assert s1.energy.induced_j==pytest.approx(s0.energy.induced_j,rel=1e-12,abs=1e-14)
    fs0={c.source.charge_c:c.total_force_n for c in s0.charges}
    fs1={c.source.charge_c:c.total_force_n for c in s1.charges}
    for q,f in fs0.items():
        assert fs1[q]==pytest.approx((f[2],f[1],-f[0]),rel=1e-12,abs=1e-11)


def test_position_sweep_works_from_exact_cavity_center_and_preserves_original_model():
    p=one((PointCharge(Q,(0.,0.,0.)),),FixedNetCharge(.6*Q))
    spec=PositionSweepSpec(p,0,(0.,0.,.07),steps=5)
    plane=sweep_slice(spec,resolution=9)
    assert all(abs(plane.project(point)[2])<1e-13 for point in (spec.start_m,spec.end_m))
    data=sample_position_sweep(spec,MultiImageShellBackend(),plane=plane,
                               line_steps=12,seeds=[],profile_samples=9)
    samples=data['position_sweep']['samples']
    assert len(samples)==5
    assert samples[0]['force_n']==[0.,0.,0.]
    assert samples[0]['external_quasistatic_work_j']==0.
    assert samples[-1]['energy_j']<samples[0]['energy_j']
    assert samples[-1]['external_quasistatic_work_j']==pytest.approx(
        samples[-1]['energy_j']-samples[0]['energy_j'])
    assert p.charges[0].position_m==(0.,0.,0.)
    assert 'not elapsed time' in data['playback']
    assert all(f['force_energy'] for f in data['frames'])
    json.dumps(data,allow_nan=False)


def test_multi_source_nested_cavity_sweep_reuses_one_backend_and_viewer():
    p=nested((PointCharge(Q,(0.,0.,-.045)),PointCharge(-.3*Q,(.038,.012,.018))),
             Grounded(),FixedNetCharge(.9*Q))
    spec=PositionSweepSpec(p,0,(0.,0.,.073),steps=5)
    data=sample_position_sweep(spec,NestedShellBackend(),resolution=9,
                               line_steps=12,seeds=[],profile_samples=9)
    rows=data['position_sweep']['samples']
    assert len(rows)==5
    assert all(r['charge_c']==Q for r in rows)
    assert len({r['position_m'][2] for r in rows})==5
    assert len({r['energy_j'] for r in rows})>2
    assert all(len(f['profiles'])==4 for f in data['frames'])
    for i in range(len(rows)):
        assert rows[i]['force_parallel_n']==pytest.approx(rows[i]['force_n'][2])


def test_cavity_path_crossing_metal_or_other_source_is_refused_before_any_solve():
    p=one((PointCharge(Q,(0,0,.015)),),Grounded())
    for end in ((0,0,A),(.0,0,.12),(0,0,-.12)):
        with pytest.raises(UnsupportedProblem,match='within the spherical cavity'):
            PositionSweepSpec(p,0,end,steps=5)
    p2=one((PointCharge(Q,(0,0,.015)),PointCharge(-.5*Q,(0,0,.045))),Grounded())
    with pytest.raises(UnsupportedProblem,match='intersects another'):
        PositionSweepSpec(p2,0,(0,0,.075),steps=5)
    # Other electrostatic domains have not accidentally become cavity force
    # providers; in particular an all-metal source-free sphere has no source.
    with pytest.raises(UnsupportedProblem):
        PositionSweepSpec(one((PointCharge(0,(0.,0.,0.)),)),0,(0,0,.02))


def test_nested_sweep_solution_still_refuses_phantom_numerical_provider():
    p=nested((PointCharge(Q,(0,0,.015)),),Grounded(),Grounded())
    spec=PositionSweepSpec(p,0,(0,0,.05),steps=3)
    calls={'solve':0,'close':0}
    class Solution:
        conductor_results=()
        def close(self):calls['close']+=1
    class Backend:
        def supports(self,problem):return True
        def solve(self,problem):
            calls['solve']+=1
            return Solution()
    with pytest.raises(UnsupportedProblem,match='does not provide'):
        sample_position_sweep(spec,Backend(),resolution=9,seeds=[],line_steps=12,profile_samples=9)
    assert calls=={'solve':1,'close':1}


def test_legacy_centered_and_one_source_cavity_backends_share_observables():
    from conductor_sim import ImageChargeShellBackend, ConcentricShellBackend
    for constraint in (Grounded(),FixedNetCharge(1.6*Q)):
        for z in (0.,.04):
            p=one((PointCharge(Q,(0.,0.,z)),),constraint)
            reference=force_energy(MultiImageShellBackend().solve(p))
            choices=[ImageChargeShellBackend()]
            if z==0.:
                choices.append(ConcentricShellBackend())
            for backend in choices:
                old=force_energy(backend.solve(p))
                assert old.charges[0].total_force_n==pytest.approx(reference.charges[0].total_force_n)
                assert old.energy.total_j==pytest.approx(reference.energy.total_j)
                assert old.energy.induced_j==pytest.approx(reference.energy.induced_j)
                assert old.energy.background_j==pytest.approx(reference.energy.background_j)


def test_legacy_centered_canceling_sources_produce_zero_force_sources():
    from conductor_sim import ConcentricShellBackend
    p=one((PointCharge(Q,(0.,0.,0.)),PointCharge(-Q,(0.,0.,0.))),FixedNetCharge(Q))
    old=force_energy(ConcentricShellBackend().solve(p))
    assert old.charges==() and old.energy.total_j==0.


@pytest.mark.parametrize('name',(
    'cavity-grounded','cavity-isolated','cavity-nested','cavity-multi','cavity-centered'))
def test_all_five_cavity_examples_reuse_position_sweep_contract(name):
    from examples.force_energy_explorer import scenario
    spec,backend=scenario(name,48)
    assert backend.supports(spec.problem)
    assert spec.start_m != spec.end_m
    assert all(isinstance(c,SphericalShell) for c in spec.problem.conductors)
    entries=list(spec.states())
    assert len(entries)==spec.steps
    assert all(backend.supports(p) for _,p in entries)
    assert all(all(q.position_m==spec.problem.charges[i].position_m
                   for i,q in enumerate(p.charges) if i!=spec.charge_index)
               for _,p in entries)
    first=force_energy(backend.solve(entries[0][1]))
    last=force_energy(backend.solve(entries[-1][1]))
    assert len(first.charges)>=1 and len(last.charges)>=1
    if name=='cavity-centered':
        assert first.charges[0].total_force_n==(0.,0.,0.)
        assert last.charges[0].total_force_n!=(0.,0.,0.)
    if name=='cavity-nested':
        assert len(spec.problem.conductors)==2


def test_legacy_single_offcenter_exact_image_backend_can_run_reusable_sweep():
    from conductor_sim import ImageChargeShellBackend, ConcentricShellBackend
    p=one((PointCharge(Q,(0.,0.,.025)),),Grounded())
    spec=PositionSweepSpec(p,0,(0.,0.,.063),steps=3)
    result=sample_position_sweep(spec,ImageChargeShellBackend(),resolution=9,
                                 line_steps=12,seeds=[],profile_samples=9)
    assert len(result['frames'])==3
    assert result['position_sweep']['samples'][-1]['force_parallel_n']>0
    # The historically centered-only backend must still refuse an offcenter
    # sweep, rather than silently pretending to solve offcenter image physics.
    with pytest.raises(UnsupportedProblem,match='does not support'):
        sample_position_sweep(spec,ConcentricShellBackend(),resolution=9,
                              line_steps=12,seeds=[],profile_samples=9)
