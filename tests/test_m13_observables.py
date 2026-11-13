"""M13a independent identities, energy gradients, capability and legacy checks."""
from dataclasses import asdict, replace
import json
from math import fsum, hypot, log, pi

import pytest

from conductor_sim import (
    EPSILON_0, ChargeForce, EnergyBreakdown, ForceEnergyResult, force_energy,
    ConductingPlane, ElectrostaticProblem, FixedNetCharge, FixedPotential, Grounded,
    GroundedPlaneBackend, ExternalSphereBackend, ParallelPlateBackend,
    NestedShellBackend, PointCharge, SolidSphere, SphericalShell,
    Femm5PlaneBackend, Femm5ParallelPlateBackend, UnsupportedProblem, SingularField,
)
from conductor_sim.observables import _direct
from test_femm5_adapter import FakeClient, FakeSession

K = 1/(4*pi*EPSILON_0)
Q = 1e-9
D = .2


def slab(charges=(), v0=0., v1=0., d=D):
    return ElectrostaticProblem(tuple(charges), (
        ConductingPlane(constraint=FixedPotential(v0), name='lower'),
        ConductingPlane((0.,0.,d),(0.,0.,-1.),FixedPotential(v1),'upper')))


def point(h, q=Q):
    return PointCharge(q, (0.,0.,h))


def plane_problem(charges):
    return ElectrostaticProblem(tuple(charges), (ConductingPlane(),))


def sphere_problem(charges, constraint=Grounded()):
    return ElectrostaticProblem(tuple(charges), (SolidSphere(.1,constraint),))


def assert_gradient(problem, backend, *, step=1e-6, rel=3e-7):
    observations = force_energy(backend.solve(problem))
    sites = [c.source for c in observations.charges]
    for i, observation in enumerate(observations.charges):
        for axis in range(3):
            def displaced_energy(delta):
                charges = list(sites)
                q = charges[i]
                position = tuple(q.position_m[j]+(delta if axis==j else 0.) for j in range(3))
                charges[i] = PointCharge(q.charge_c,position)
                changed = replace(problem, charges=tuple(charges))
                return force_energy(backend.solve(changed)).energy.total_j
            fd = -(displaced_energy(step)-displaced_energy(-step))/(2*step)
            assert observation.total_force_n[axis] == pytest.approx(fd, rel=rel, abs=8e-15)
    return observations


@pytest.mark.parametrize('q',[Q,-Q,2*Q])
def test_grounded_plane_half_image_energy_and_self_force(q):
    h = .073
    s = GroundedPlaneBackend().solve(plane_problem((point(h,q),)))
    out = force_energy(s)
    f = out.charges[0]
    assert f.source == point(h,q)
    assert f.total_force_n == pytest.approx((0.,0.,-K*q*q/(4*h*h)),rel=3e-14,abs=1e-25)
    assert f.induced_force_n == f.conductor_force_n == s.induced_force_on_charge_n
    assert out.energy.induced_j == pytest.approx(-K*q*q/(4*h),rel=3e-14)
    assert out.energy.total_j == pytest.approx(s.interaction_energy_j,rel=3e-14)
    assert out.energy.direct_j == out.energy.background_j == 0.
    assert f.direct_force_n == f.background_force_n == (0.,0.,0.)
    # Observing force never evaluates the full singular field at the source.
    with pytest.raises(SingularField): s.electric_field_at(f.source.position_m)


@pytest.mark.parametrize('fixed_q',[None,0.,1.3e-9,-.8e-9])
@pytest.mark.parametrize('multiple',[False,True])
def test_sphere_legacy_reuse_and_clear_polarization_background_split(fixed_q,multiple):
    constraint = Grounded() if fixed_q is None else FixedNetCharge(fixed_q)
    sites = (PointCharge(Q,(.03,.02,.23)),)
    if multiple:
        sites += (PointCharge(-.4*Q,(-.19,.04,-.08)),)
    s = ExternalSphereBackend().solve(sphere_problem(sites,constraint))
    out = force_energy(s)
    assert out.energy.total_j == pytest.approx(s.interaction_energy_j,rel=3e-14,abs=1e-24)
    for i,f in enumerate(out.charges):
        legacy = s.force_on_charge_n(i) if multiple else s.force_on_charge_n
        assert f.total_force_n == pytest.approx(legacy,rel=3e-14,abs=1e-23)
        d = hypot(*f.source.position_m)
        bg = tuple(K*f.source.charge_c*(fixed_q or 0.)*x/d**3 for x in f.source.position_m)
        assert f.background_force_n == pytest.approx(bg,rel=3e-14,abs=1e-24)
    expected = sum(K*(fixed_q or 0.)*q.charge_c/hypot(*q.position_m) for q in sites)
    assert out.energy.background_j == pytest.approx(expected,rel=3e-14,abs=1e-24)
    assert_gradient(sphere_problem(sites,constraint),ExternalSphereBackend())


def test_fixed_Q_changes_background_not_polarization():
    sites = (point(.22),point(-.31,-.6*Q))
    neutral = force_energy(ExternalSphereBackend().solve(sphere_problem(sites,FixedNetCharge(0.))))
    charged = force_energy(ExternalSphereBackend().solve(sphere_problem(sites,FixedNetCharge(1.5*Q))))
    assert charged.energy.induced_j == neutral.energy.induced_j
    for a,b in zip(charged.charges,neutral.charges):
        assert a.induced_force_n == b.induced_force_n
        assert a.direct_force_n == b.direct_force_n


@pytest.mark.parametrize('q',[Q,-Q])
@pytest.mark.parametrize('order',[1,8,48,128])
def test_midpoint_force_is_symmetric_at_every_order_without_zero_override(q,order):
    backend = ParallelPlateBackend(image_order=order)
    s = backend.solve(slab((point(D/2,q),)))
    out = force_energy(s)
    assert out.image_order == order
    assert out.charges[0].induced_force_n == pytest.approx((0.,0.,0.),abs=3e-21)
    # Small displacements show an UNSTABLE equilibrium, not a forced zero band.
    fminus=force_energy(backend.solve(slab((point(.099,q),)))).charges[0].total_force_n[2]
    fplus=force_energy(backend.solve(slab((point(.101,q),)))).charges[0].total_force_n[2]
    assert fminus < 0 < fplus
    assert fminus == pytest.approx(-fplus,rel=3e-12,abs=1e-22)


def test_midpoint_energy_converges_to_independent_log2_identity():
    exact = -K*Q*Q*log(2)/D
    errors=[]
    for n in (8,16,32,64,128):
        out=force_energy(ParallelPlateBackend(image_order=n).solve(slab((point(D/2),))))
        errors.append(abs(out.energy.induced_j-exact))
    assert all(b < .3*a for a,b in zip(errors,errors[1:]))
    assert errors[-1] < abs(exact)*6e-6


@pytest.mark.parametrize('fraction',[.17,.325,.69,.87])
def test_single_slab_force_agrees_with_independent_one_dimensional_series(fraction):
    # Pair the image forces by their distance to opposite plates, not with the
    # implementation's potential kernel; an independent convergent series.
    ref = -K*Q*Q/(4*D*D)*fsum(1/(m+fraction)**2-1/(m+1-fraction)**2 for m in range(30000))
    p = slab((point(fraction*D),))
    coarse = force_energy(ParallelPlateBackend(image_order=24).solve(p)).charges[0].induced_force_n[2]
    fine = force_energy(ParallelPlateBackend(image_order=256).solve(p)).charges[0].induced_force_n[2]
    assert abs(fine-ref) < abs(coarse-ref)/10
    assert fine == pytest.approx(ref,rel=2e-6,abs=1e-17)


@pytest.mark.parametrize('upper',[False,True])
def test_near_plate_force_reduces_to_single_plane_limit(upper):
    h=D/1000
    z=D-h if upper else h
    out=force_energy(ParallelPlateBackend(image_order=128).solve(slab((point(z),))))
    expected=K*Q*Q/(4*h*h)*(1 if upper else -1)
    assert out.charges[0].total_force_n[2] == pytest.approx(expected,rel=2e-8)


@pytest.mark.parametrize('order',[1,12,48])
@pytest.mark.parametrize('volts',[(0.,0.),(-7.,19.),(8.,8.)])
def test_slab_force_energy_gradient_for_arbitrary_3d_signed_sources(order,volts):
    charges=(PointCharge(Q,(.017,-.009,.045)),PointCharge(-.7*Q,(-.053,.023,.142)),
             PointCharge(.35*Q,(.084,-.019,.091)))
    p=slab(charges,*volts)
    out=assert_gradient(p,ParallelPlateBackend(image_order=order))
    assert tuple(fsum(c.direct_force_n[i] for c in out.charges) for i in range(3)) == pytest.approx((0,0,0),abs=1e-22)
    assert out.energy.total_j == pytest.approx(fsum((out.energy.direct_j,out.energy.induced_j,out.energy.background_j)),rel=1e-14)


def test_bias_has_full_qV_and_qE_not_half():
    sites=(point(.063,Q),point(.141,-.6*Q))
    a=force_energy(ParallelPlateBackend().solve(slab(sites)))
    b=force_energy(ParallelPlateBackend().solve(slab(sites,-12.,28.)))
    assert b.energy.induced_j == a.energy.induced_j
    expected=sum(q.charge_c*(-12.+40*q.position_m[2]/D) for q in sites)
    assert b.energy.background_j == pytest.approx(expected,rel=2e-14)
    for f in b.charges:
        assert f.background_force_n == pytest.approx((0,0,-200*f.source.charge_c),abs=1e-24)


def test_common_voltage_shift_changes_energy_gauge_but_not_force_or_work():
    b=ParallelPlateBackend(image_order=24)
    initial=(point(.06),point(.14,-.4*Q))
    final=(point(.075),point(.14,-.4*Q))
    observations = [force_energy(b.solve(slab(sites,lo,hi)))
                    for sites in (initial,final) for lo,hi in ((-3.,17.),(10.,30.))]
    a,ag,z,zg=observations
    assert ag.energy.total_j-a.energy.total_j == pytest.approx(13*.6*Q,rel=1e-13)
    assert zg.energy.total_j-ag.energy.total_j == pytest.approx(z.energy.total_j-a.energy.total_j,rel=1e-13)
    for f,g in zip(a.charges,ag.charges):
        assert f.total_force_n == g.total_force_n


def test_reciprocity_and_plate_order_invariance_for_fields_and_observations():
    sites=(PointCharge(Q,(.02,.01,.06)),PointCharge(-.5*Q,(-.03,.04,.14)))
    p=slab(sites,-5.,8.)
    b=ParallelPlateBackend(image_order=3)
    a=b.solve(p);z=b.solve(replace(p,conductors=tuple(reversed(p.conductors))))
    for pt in ((.08,.015,.11),(-.08,.01,.08)):
        assert a.potential_at(pt)==pytest.approx(z.potential_at(pt),rel=1e-13,abs=1e-11)
        assert a.electric_field_at(pt)==pytest.approx(z.electric_field_at(pt),rel=1e-13,abs=1e-10)
    oa,oz=force_energy(a),force_energy(z)
    assert oa.energy.total_j==pytest.approx(oz.energy.total_j,rel=1e-13)
    for f,g in zip(oa.charges,oz.charges):
        assert f.total_force_n==pytest.approx(g.total_force_n,rel=1e-13,abs=1e-22)
    x,y=sites
    ha=a._source_potential(x.position_m,y,exclude_direct=True)/y.charge_c
    hb=a._source_potential(y.position_m,x,exclude_direct=True)/x.charge_c
    assert ha==pytest.approx(hb,rel=3e-14)


def test_world_rotation_and_translation_transform_forces_not_energy():
    p=slab((PointCharge(Q,(.02,.01,.06)),PointCharge(-.4*Q,(-.04,.03,.145))),3.,21.)
    def vector(v): return (v[2],v[0],v[1])
    def tr(v): return tuple(a+b for a,b in zip(vector(v),(2.,-1.,.5)))
    new=replace(p,charges=tuple(PointCharge(q.charge_c,tr(q.position_m)) for q in p.charges),
                conductors=tuple(replace(c,point_m=tr(c.point_m),normal=vector(c.normal)) for c in p.conductors))
    b=ParallelPlateBackend(image_order=48)
    a,z=force_energy(b.solve(p)),force_energy(b.solve(new))
    assert a.energy.total_j==pytest.approx(z.energy.total_j,rel=2e-13)
    # Sorting changes under rotation; identify sites by their transformed coordinates.
    for f in a.charges:
        g=next(c for c in z.charges if c.source.position_m==tr(f.source.position_m))
        for attr in ('total_force_n','induced_force_n','direct_force_n','background_force_n'):
            assert vector(getattr(f,attr))==pytest.approx(getattr(g,attr),rel=2e-12,abs=1e-20)


@pytest.mark.parametrize('volts',[(0.,0.),(0.,20.)])
def test_empty_or_exactly_canceled_slab_has_zero_mechanical_interaction(volts):
    b=ParallelPlateBackend()
    for sites in ((),(point(.06),point(.06,-Q)),(point(.06,0.),)):
        s=b.solve(slab(sites,*volts));out=force_energy(s)
        assert out.charges==()
        assert out.energy.total_j==0.
        if volts[0]!=volts[1]:
            assert s.conductor_results[0].net_charge_c is None
    out=force_energy(b.solve(slab((point(.06),point(.14,-Q)),*volts)))
    assert len(out.charges)==2 and out.energy.total_j!=0.


def test_grouped_sources_and_component_permutations_preserve_observations():
    base=(point(.06,.6*Q),point(.06,.4*Q),point(.14,-Q),point(.09,Q),point(.09,-Q))
    b=ParallelPlateBackend(image_order=16)
    out=force_energy(b.solve(slab(base)))
    assert len(out.charges)==2
    assert out==force_energy(b.solve(slab(tuple(reversed(base)))))
    assert out.charges[0].source.charge_c==pytest.approx(Q,rel=1e-14)
    assert_gradient(slab(tuple(c.source for c in out.charges)),b)


def test_charge_and_permittivity_scaling():
    b=ParallelPlateBackend(image_order=16)
    p=slab((point(.06),point(.14,-.4*Q)))
    a=force_energy(b.solve(p))
    z=force_energy(b.solve(replace(p,charges=tuple(replace(q,charge_c=3*q.charge_c) for q in p.charges))))
    e=force_energy(b.solve(replace(p,permittivity_f_per_m=2*EPSILON_0)))
    assert z.energy.total_j==pytest.approx(9*a.energy.total_j,rel=2e-14)
    assert e.energy.total_j==pytest.approx(.5*a.energy.total_j,rel=2e-14)
    for f,g,h in zip(a.charges,z.charges,e.charges):
        assert g.total_force_n==pytest.approx(tuple(9*x for x in f.total_force_n),rel=3e-14,abs=1e-23)
        assert h.total_force_n==pytest.approx(tuple(.5*x for x in f.total_force_n),rel=3e-14,abs=1e-23)


def test_quasistatic_work_equals_negative_force_integral():
    b=ParallelPlateBackend(image_order=32)
    lo,hi=.055,.085
    def result(z):return force_energy(b.solve(slab((point(z),),-3.,17.)))
    energy_work=result(hi).energy.total_j-result(lo).energy.total_j
    n=120;dz=(hi-lo)/n
    values=[result(lo+i*dz).charges[0].total_force_n[2] for i in range(n+1)]
    integral=dz/3*(values[0]+values[-1]+4*fsum(values[1:-1:2])+2*fsum(values[2:-1:2]))
    assert energy_work==pytest.approx(-integral,rel=1e-8,abs=1e-18)


@pytest.mark.parametrize('bad',[True,False,1.5,'0',None])
def test_new_force_index_rejects_nonintegers(bad):
    s=ParallelPlateBackend().solve(slab((point(.06),)))
    with pytest.raises(TypeError): force_energy(s).force_on_charge_n(bad)
    with pytest.raises(TypeError): s.force_on_charge_n(bad)


@pytest.mark.parametrize('bad',[-1,1,100])
def test_new_force_index_rejects_out_of_range(bad):
    s=ParallelPlateBackend().solve(slab((point(.06),)))
    with pytest.raises(IndexError): force_energy(s).force_on_charge_n(bad)
    with pytest.raises(IndexError): s.force_on_charge_n(bad)


def test_slab_convenience_api_and_json_finite_results():
    s=ParallelPlateBackend().solve(slab((point(.06),point(.14,-.4*Q)),0.,20.))
    out=force_energy(s)
    assert s.interaction_energy_j==out.energy.total_j
    for i,f in enumerate(out.charges):
        assert s.force_on_charge_n(i)==f.total_force_n
        assert s.force_on_charge_n(i,include_direct=False)==f.conductor_force_n
    json.dumps(asdict(out),allow_nan=False)
    with pytest.raises(ValueError): EnergyBreakdown(float('inf'),0.,0.,'invalid')
    with pytest.raises(ValueError): ChargeForce(point(.06),(float('nan'),0,0),(0,0,0))
    with pytest.raises(ValueError): _direct((point(.06),point(.06)),K)


def test_optional_provider_supports_custom_solutions_and_propagates_failures():
    expected=ForceEnergyResult((),EnergyBreakdown(0.,0.,0.,'custom'),'test provider')
    class Custom:
        def force_energy(self):return expected
    class Bad:
        def force_energy(self):raise RuntimeError('provider failed')
    class Wrong:
        def force_energy(self):return {'fake':True}
    assert force_energy(Custom()) is expected
    with pytest.raises(RuntimeError,match='provider failed'):force_energy(Bad())
    with pytest.raises(TypeError,match='ForceEnergyResult'):force_energy(Wrong())
    with pytest.raises(UnsupportedProblem):force_energy(object())


@pytest.mark.parametrize('pair',[False,True])
def test_numerical_observation_refusal_does_not_sample_or_close_session(pair):
    p=slab((point(.06),)) if pair else plane_problem((point(.06),))
    client=FakeClient();session=FakeSession(client)
    backend=Femm5ParallelPlateBackend if pair else Femm5PlaneBackend
    with backend(session_factory=lambda:session).solve(p) as s:
        calls=list(client.calls)
        with pytest.raises(UnsupportedProblem,match='does not provide'):force_energy(s)
        assert client.calls==calls
        assert session.closes==0
    assert session.closes==1


def test_cavity_force_not_fabricated_from_singular_total_field():
    p=ElectrostaticProblem((point(.03),),(SphericalShell(.1,.12,Grounded()),))
    s=NestedShellBackend().solve(p)
    with pytest.raises(UnsupportedProblem,match='does not provide'):force_energy(s)


@pytest.mark.parametrize('family',['plane','sphere','slab'])
def test_each_energy_component_generates_its_corresponding_force(family):
    charges=(PointCharge(Q,(.018,.01,.045)),PointCharge(-.5*Q,(-.049,.03,.143)))
    if family=='plane':
        p=plane_problem(charges);b=GroundedPlaneBackend()
    elif family=='sphere':
        charges=tuple(replace(q,position_m=tuple(3*x for x in q.position_m)) for q in charges)
        p=sphere_problem(charges,FixedNetCharge(1.3*Q));b=ExternalSphereBackend()
    else:
        p=slab(charges,-4.,17.);b=ParallelPlateBackend(image_order=8)
    out=force_energy(b.solve(p));sites=tuple(c.source for c in out.charges)
    for i,c in enumerate(out.charges):
        for axis in range(3):
            energies=[]
            for delta in (-1e-6,1e-6):
                moved=list(sites);q=moved[i]
                moved[i]=replace(q,position_m=tuple(x+(delta if j==axis else 0) for j,x in enumerate(q.position_m)))
                energies.append(force_energy(b.solve(replace(p,charges=tuple(moved)))).energy)
            for field,component in [('direct_force_n','direct_j'),('induced_force_n','induced_j'),('background_force_n','background_j')]:
                fd=-(getattr(energies[1],component)-getattr(energies[0],component))/2e-6
                assert getattr(c,field)[axis]==pytest.approx(fd,rel=5e-7,abs=7e-15)


def test_regular_slab_force_matches_limit_of_existing_field_samples():
    s=ParallelPlateBackend(image_order=12).solve(slab(
        (PointCharge(Q,(.017,.01,.045)),PointCharge(-.4*Q,(-.049,.03,.143))),-4.,17.))
    for observation in force_energy(s).charges:
        q=observation.source
        residual=[]
        for delta in (-2e-5,2e-5):
            p=(q.position_m[0]+delta,q.position_m[1],q.position_m[2])
            e=s.electric_field_at(p)
            self_x=K*q.charge_c*delta/abs(delta)**3
            residual.append((e[0]-self_x,e[1],e[2]))
        predicted=tuple(q.charge_c*(residual[0][j]+residual[1][j])/2 for j in range(3))
        assert observation.total_force_n==pytest.approx(predicted,rel=8e-6,abs=1e-14)


def test_force_energy_example_uses_common_api_and_writes_finite_report(tmp_path):
    from pathlib import Path
    import subprocess
    import sys
    script=Path(__file__).resolve().parents[1]/'examples'/'force_energy.py'
    output=tmp_path/'report.json'
    subprocess.run([sys.executable,str(script),'--image-order','16','--output',str(output)],
                   check=True,capture_output=True,text=True,timeout=30)
    report=json.loads(output.read_text(encoding='utf-8'))
    assert len(report['cases'])==7
    assert all(c['charges'] and 'reference' in c['energy'] for c in report['cases'])
    work=report['work_example']
    assert work['external_quasistatic_work_j']==-work['work_by_electric_force_j']
    assert work['external_quasistatic_work_j']>0
