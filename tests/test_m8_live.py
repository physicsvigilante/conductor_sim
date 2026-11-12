"""Real FEMM 5 M8 isolated-sphere integration; mock clients NEVER satisfy this gate."""
from math import pi, sin
from os import environ

import pytest

from conductor_sim import (
    ElectrostaticProblem, ExternalIsolatedSphereBackend,
    ExternalGroundedSphereBackend, Femm5ExternalSphereBackend,
    Femm5Settings, FixedNetCharge, Grounded, PointCharge, SolidSphere,
)

pytestmark = [pytest.mark.femm5, pytest.mark.skipif(
    not environ.get('FEMM5_BINDING'), reason='real FEMM 5 installation is required'
)]

A = .12
D = .27
Q = 1e-9


def problem(net=0., q=Q, center=(0.,0.,0.), direction=(0.,0.,1.), grounded=False):
    sphere = SolidSphere(A, Grounded() if grounded else FixedNetCharge(net),
                         center_m=center, name='physical-sphere')
    source = PointCharge(q, tuple(center[i]+D*direction[i] for i in range(3)))
    return ElectrostaticProblem(charges=(source,), conductors=(sphere,))


@pytest.mark.parametrize('net', (0.,1e-9,-.75e-9))
def test_real_femm_isolated_sphere_net_charge_potential_and_exterior_field(net):
    p=problem(net)
    reference=ExternalIsolatedSphereBackend().solve(p)
    with Femm5ExternalSphereBackend().solve(p) as numeric:
        exact=reference.conductor_results[0]
        got=numeric.conductor_results[0]
        assert got.net_charge_c == pytest.approx(net,rel=.08,abs=2e-11)
        assert got.potential_v == pytest.approx(exact.potential_v,rel=.16,abs=2)
        assert numeric.numerical_source_charge_c == pytest.approx(Q,rel=.05)
        # All expected physical conductor properties must be measured, not
        # imposed as an analytical reference substitute.
        for point in ((0,0,.17),(.08,0,.22),(-.19,.07,.08),(.32,0,-.14),(.22,0,.15)):
            assert numeric.potential_at(point) == pytest.approx(
                reference.potential_at(point),rel=.25,abs=3.5)
            assert numeric.electric_field_at(point) == pytest.approx(
                reference.electric_field_at(point),rel=.40,abs=130)
        assert numeric.potential_at((0,0,0)) == pytest.approx(exact.potential_v,rel=.2,abs=2)
        assert numeric.electric_field_at((0,0,.05)) == pytest.approx((0,0,0),abs=6)
        for theta in (.2,1.,2.8):
            assert numeric.surface_charge_density('outer',theta) == pytest.approx(
                reference.surface_charge_density('outer',theta),rel=.70,abs=2.5e-9)


def test_real_femm_grounded_vs_isolated_constraint_is_not_silently_confused():
    grounded_p=problem(grounded=True)
    neutral_p=problem(net=0.)
    exact_g=ExternalGroundedSphereBackend().solve(grounded_p)
    exact_i=ExternalIsolatedSphereBackend().solve(neutral_p)
    with Femm5ExternalSphereBackend().solve(grounded_p) as g, \
         Femm5ExternalSphereBackend().solve(neutral_p) as i:
        assert g.conductor_results[0].net_charge_c == pytest.approx(
            exact_g.conductor_results[0].net_charge_c,rel=.13,abs=1e-11)
        assert abs(g.conductor_results[0].potential_v) < .1
        assert abs(i.conductor_results[0].net_charge_c) < 2e-11
        assert i.conductor_results[0].potential_v == pytest.approx(
            exact_i.conductor_results[0].potential_v,rel=.15)
        assert i.conductor_results[0].potential_v-g.conductor_results[0].potential_v > 15
        # Source data comes from two independent FEMM solves, never the image backend.
        assert i.numerical_source_charge_c == pytest.approx(Q,rel=.04)
        assert g.numerical_source_charge_c == pytest.approx(Q,rel=.04)


def test_real_femm_neutral_induced_surface_density_changes_sign_and_integrates_to_zero():
    p=problem(0.)
    with Femm5ExternalSphereBackend(settings=Femm5Settings(arc_degrees=3,mesh_scale=.65)).solve(p) as n:
        near=n.surface_charge_density('outer',.12)
        far=n.surface_charge_density('outer',pi-.12)
        assert near < 0 < far
        h=pi/72
        observed = 2*pi*A*A*h*sum(n.surface_charge_density('outer',(j+.5)*h)*
                                   sin((j+.5)*h) for j in range(72))
        # One-sided numerical sigma near a discretized surface is not a
        # perfect boundary integral; zero-net Q is the independent target.
        assert abs(observed) < 3.5e-10


def test_real_femm_isolated_mesh_outer_domain_and_core_size_sensitivity():
    p=problem(net=1e-9)
    exact=ExternalIsolatedSphereBackend().solve(p)
    sample=(0,0,.18)
    target=exact.potential_at(sample)
    results=[]
    for settings in (
        Femm5Settings(source_radius_fraction=.04,outer_radius_factor=6),
        Femm5Settings(source_radius_fraction=.08,outer_radius_factor=8),
        Femm5Settings(source_radius_fraction=.08,outer_radius_factor=10,mesh_scale=.65),
    ):
        with Femm5ExternalSphereBackend(settings=settings).solve(p) as n:
            results.append(n.potential_at(sample))
            assert n.conductor_results[0].net_charge_c == pytest.approx(1e-9,rel=.07)
    assert all(v==pytest.approx(target,rel=.26,abs=2.5) for v in results)
    assert max(results)-min(results) < .22*max(abs(target),2)


def test_real_femm_isolated_world_rotation_translation_covariance():
    pz=problem(net=-.6e-9)
    px=problem(net=-.6e-9,center=(1.,-2.,.5),direction=(1.,0.,0.))
    with Femm5ExternalSphereBackend().solve(pz) as z, Femm5ExternalSphereBackend().solve(px) as x:
        point=(.16,.07,.19)
        rotated=(1.+point[2],-2.+point[1],.5+point[0])
        assert x.conductor_results[0].potential_v == pytest.approx(
            z.conductor_results[0].potential_v,rel=.07,abs=1)
        assert x.potential_at(rotated) == pytest.approx(z.potential_at(point),rel=.07,abs=2)
        ez=z.electric_field_at(point)
        ex=x.electric_field_at(rotated)
        assert ex == pytest.approx((ez[2],ez[1],ez[0]),rel=.15,abs=30)
