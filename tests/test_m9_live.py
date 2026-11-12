"""M9 actual FEMM 5 integration. Never pass by substituting exact images."""
from math import pi, sin
from os import environ

import pytest

from conductor_sim import (ElectrostaticProblem, ExternalSphereBackend,
                           Femm5ExternalSphereBackend, Femm5Settings,
                           FixedNetCharge, Grounded, PointCharge, SolidSphere)
from conductor_sim.solver import UnsupportedProblem
from test_m9_external import problem, A

pytestmark=[pytest.mark.femm5,
            pytest.mark.skipif(not environ.get('FEMM5_BINDING'),
                               reason='real FEMM 5 binding not configured')]

@pytest.mark.parametrize('constraint',[
    Grounded(),FixedNetCharge(0.),FixedNetCharge(1.1e-9),FixedNetCharge(-.75e-9)])
def test_real_femm_multi_external_charge_potential_field_net_charge(constraint):
    p=problem(constraint)
    exact=ExternalSphereBackend().solve(p)
    with Femm5ExternalSphereBackend().solve(p) as sol:
        got=sol.conductor_results[0]
        exp=exact.conductor_results[0]
        assert got.net_charge_c==pytest.approx(exp.net_charge_c,rel=.14,abs=2e-11)
        assert got.potential_v==pytest.approx(exp.potential_v,rel=.13,abs=1.0)
        assert len(sol.numerical_sources)==2
        assert sol.numerical_source_charge_c==pytest.approx(.5e-9,rel=.02)
        for num in sol.numerical_sources:
            q=next(p.charge_c for p in p.charges if p.position_m==num.position_m)
            assert num.measured_charge_c==pytest.approx(q,rel=.03,abs=1e-12)
        for sample in ((0,0,.17),(.2,0,-.12),(.23,0,.08),(.19,.10,0),(.4,0,.10)):
            assert sol.potential_at(sample)==pytest.approx(exact.potential_at(sample),rel=.20,abs=3)
            assert sol.electric_field_at(sample)==pytest.approx(exact.electric_field_at(sample),rel=.43,abs=170)
        assert sol.potential_at((0,0,0))==pytest.approx(exp.potential_v,rel=.13,abs=1)
        assert max(abs(v) for v in sol.electric_field_at((0,0,.03)))<7
        assert sol.source_radius_m is None


def test_real_femm_two_opposite_source_density_and_flux():
    p=problem(Grounded())
    ref=ExternalSphereBackend().solve(p)
    with Femm5ExternalSphereBackend(settings=Femm5Settings(mesh_scale=.7,arc_degrees=3)).solve(p) as s:
        sigma=[s.surface_charge_density('outer',angle) for angle in (.15,.6,1.5,2.55,pi-.15)]
        reference=[ref.surface_charge_density('outer',angle) for angle in (.15,.6,1.5,2.55,pi-.15)]
        assert sigma[0]>0 and sigma[-1]<0
        for x,y in zip(sigma,reference):
            assert x==pytest.approx(y,rel=.7,abs=1.5e-9)
        n=60;step=pi/n
        charge=2*pi*A*A*step*sum(s.surface_charge_density('outer',(i+.5)*step)*
                                  sin((i+.5)*step) for i in range(n))
        assert charge==pytest.approx(ref.conductor_results[0].net_charge_c,rel=.42,abs=6e-11)


def test_real_femm_zero_total_external_charge_does_not_remove_independent_sources():
    p=problem(FixedNetCharge(0.),sites=((1e-9,(0.,0.,.27)),(-1e-9,(0.,0.,-.27))))
    exact=ExternalSphereBackend().solve(p)
    assert exact.conductor_results[0].potential_v==pytest.approx(0,abs=1e-8)
    with Femm5ExternalSphereBackend().solve(p) as s:
        assert len(s.numerical_sources)==2
        assert s.numerical_source_charge_c==pytest.approx(0,abs=2e-20)
        assert sorted(c.measured_charge_c for c in s.numerical_sources)==pytest.approx([-1e-9,1e-9],rel=.02)
        assert s.conductor_results[0].net_charge_c==pytest.approx(0,abs=2e-20)
        assert s.conductor_results[0].potential_v==pytest.approx(0,abs=1.0)
        for point in ((0,0,.18),(0,0,-.18),(.24,.04,.1)):
            assert s.potential_at(point)==pytest.approx(exact.potential_at(point),rel=.20,abs=2.0)


def test_real_femm_axis_rotation_and_translated_sphere():
    pz=problem(Grounded())
    px=problem(Grounded(),sites=((1e-9,(.27,0,0)),(-.5e-9,(-.32,0,0))),center=(1.,-.7,.2))
    with Femm5ExternalSphereBackend().solve(pz) as z, Femm5ExternalSphereBackend().solve(px) as x:
        assert x.conductor_results[0].net_charge_c==pytest.approx(z.conductor_results[0].net_charge_c,rel=.03)
        pt=(.19,.07,.13)
        mapped=(1.+pt[2],-.7+pt[1],.2+pt[0])
        assert x.potential_at(mapped)==pytest.approx(z.potential_at(pt),rel=.1,abs=1)
        old=z.electric_field_at(pt)
        assert x.electric_field_at(mapped)==pytest.approx((old[2],old[1],old[0]),rel=.18,abs=45)


def test_real_femm_multi_source_mesh_domain_and_core_sensitivity():
    p=problem(Grounded())
    exact=ExternalSphereBackend().solve(p)
    point=(.23,.09,.12)
    expected=exact.potential_at(point)
    values=[]
    for settings in (Femm5Settings(source_radius_fraction=.04,outer_radius_factor=6),
                     Femm5Settings(source_radius_fraction=.08,outer_radius_factor=8),
                     Femm5Settings(source_radius_fraction=.06,outer_radius_factor=10,mesh_scale=.8)):
        with Femm5ExternalSphereBackend(settings=settings).solve(p) as s:
            values.append(s.potential_at(point))
            assert s.conductor_results[0].net_charge_c==pytest.approx(exact.conductor_results[0].net_charge_c,rel=.20)
    assert all(v==pytest.approx(expected,rel=.23,abs=2) for v in values)
    assert max(values)-min(values) < .20*max(abs(expected),2)


def test_real_femm_unsupported_noncollinear_requires_3d_solver():
    p=problem(FixedNetCharge(0.),sites=((1e-9,(0,0,.27)),(-.5e-9,(.3,0,0))))
    assert ExternalSphereBackend().supports(p)
    b=Femm5ExternalSphereBackend()
    assert not b.supports(p)
    with pytest.raises(UnsupportedProblem,match='non-collinear'):
        b.solve(p)
