"""REAL FEMM 5 integration checks. Never substitute mocked solver data.

These are skipped unless FEMM5_BINDING is set to femm-qt/pyfemm/femm.py.
Do not interpret normal CI passing with a skip as FEMM numerical validation.
"""

from math import isclose
from os import environ

import pytest

from conductor_sim import (
    ConcentricShellBackend, ElectrostaticProblem, FixedNetCharge, Grounded, PointCharge, SphericalShell,
)
from conductor_sim.femm5 import Femm5Backend, Femm5Settings

pytestmark = [
    pytest.mark.femm5,
    pytest.mark.skipif(not environ.get("FEMM5_BINDING"), reason="real FEMM 5 binding not configured"),
]


def shell_problem():
    return ElectrostaticProblem(conductors=(
        SphericalShell(inner_radius_m=.1, outer_radius_m=.12, constraint=FixedNetCharge(1e-9)),
    ))


def test_real_femm5_charged_empty_shell():
    """Compare independently computed V, E, and total Q with Gauss-law reference."""
    p = shell_problem()
    reference = ConcentricShellBackend().solve(p)
    with Femm5Backend().solve(p) as numerical:
        numeric_conductor = numerical.conductor_results[0]
        analytic_conductor = reference.conductor_results[0]
        assert numeric_conductor.net_charge_c == pytest.approx(1e-9, rel=.04)
        assert numeric_conductor.potential_v == pytest.approx(analytic_conductor.potential_v, rel=.08)
        for r in (.04, .11, .18, .36):
            got = numerical.potential_at((r, 0., 0.))
            expected = reference.potential_at((r, 0., 0.))
            assert got == pytest.approx(expected, rel=.08, abs=.03)
        # Field in cavity and metal is ~zero; outside follows Coulomb's law.
        for r in (.04, .11):
            assert abs(numerical.electric_field_at((r, 0., 0.))[0]) < 1.0
        for r in (.18, .36):
            assert numerical.electric_field_at((r, 0., 0.))[0] == pytest.approx(
                reference.electric_field_at((r, 0., 0.))[0], rel=.15
            )


def test_real_femm5_mesh_and_outer_domain_stability():
    """Do not claim convergence until these real FEMM runs succeed."""
    p = shell_problem()
    reference = ConcentricShellBackend().solve(p)
    target = reference.potential_at((.24, 0, 0))
    cases = [
        Femm5Settings(outer_radius_factor=5, mesh_scale=1.0),
        Femm5Settings(outer_radius_factor=8, mesh_scale=1.0),
        Femm5Settings(outer_radius_factor=8, mesh_scale=.5),
    ]
    values = []
    for cfg in cases:
        with Femm5Backend(settings=cfg).solve(p) as sol:
            values.append(sol.potential_at((.24, 0, 0)))
    assert all(v == pytest.approx(target, rel=.08) for v in values)
    # Bounds on sensitivity, not an unjustified assertion of monotonic convergence.
    assert abs(values[0] - values[1]) / abs(target) < .08
    assert abs(values[1] - values[2]) / abs(target) < .08


def centered_charge_problem(constraint, q=1e-9):
    return ElectrostaticProblem(
        charges=(PointCharge(q),),
        conductors=(SphericalShell(.1, .12, constraint),),
    )


@pytest.mark.parametrize("constraint", [FixedNetCharge(0), Grounded(), FixedNetCharge(2e-9)])
def test_real_femm5_centered_charge_conductor_physics(constraint):
    """Real numerical induction and source Q, never copied from the reference."""
    p = centered_charge_problem(constraint)
    exact = ConcentricShellBackend().solve(p)
    with Femm5Backend().solve(p) as numeric:
        assert numeric.source_radius_m == pytest.approx(.01)
        assert numeric.numerical_source_charge_c == pytest.approx(1e-9, rel=.04)
        expected = exact.conductor_results[0]
        got = numeric.conductor_results[0]
        assert got.potential_v == pytest.approx(expected.potential_v, rel=.08, abs=.5)
        assert got.net_charge_c == pytest.approx(expected.net_charge_c, rel=.05, abs=2e-11)
        assert got.inner_surface_charge_c == pytest.approx(-1e-9, rel=.04)
        assert got.outer_surface_charge_c == pytest.approx(expected.outer_surface_charge_c, rel=.05, abs=2e-11)
        for r in (.03, .05, .075, .11, .18, .3):
            point = (r, 0, 0)
            assert numeric.potential_at(point) == pytest.approx(
                exact.potential_at(point), rel=.10, abs=.65
            )
        for r in (.03, .05, .075, .18, .3):
            point = (r, 0, 0)
            expected_field = exact.electric_field_at(point)[0]
            got_field = numeric.electric_field_at(point)[0]
            assert got_field == pytest.approx(expected_field, rel=.20, abs=2.)
        assert abs(numeric.electric_field_at((.11,0,0))[0]) < 2


@pytest.mark.parametrize("charge", [1e-9, -1e-9])
def test_real_femm5_signed_centered_charge_grounded(charge):
    """Both source signs should induce opposite inner surface signs."""
    p = centered_charge_problem(Grounded(), charge)
    exact = ConcentricShellBackend().solve(p)
    with Femm5Backend().solve(p) as numeric:
        assert numeric.numerical_source_charge_c == pytest.approx(charge, rel=.04)
        assert numeric.conductor_results[0].net_charge_c == pytest.approx(-charge, rel=.05)
        assert numeric.electric_field_at((.05,0,0))[0] == pytest.approx(
            exact.electric_field_at((.05,0,0))[0], rel=.20
        )
        assert abs(numeric.potential_at((.3, 0, 0))) < .75


def test_real_femm5_source_core_size_and_mesh_sensitivity():
    """Error at a safe r > numerical core stays bounded under radius/mesh changes."""
    p = centered_charge_problem(FixedNetCharge(0))
    exact = ConcentricShellBackend().solve(p)
    point = (.05, 0., 0.)
    reference = exact.potential_at(point)
    results = []
    for cfg in (
        Femm5Settings(source_radius_fraction=.05, mesh_scale=1.0),
        Femm5Settings(source_radius_fraction=.10, mesh_scale=1.0),
        Femm5Settings(source_radius_fraction=.10, mesh_scale=.65),
    ):
        with Femm5Backend(settings=cfg).solve(p) as numeric:
            results.append(numeric.potential_at(point))
            assert numeric.numerical_source_charge_c == pytest.approx(1e-9, rel=.04)
    assert all(x == pytest.approx(reference, rel=.10) for x in results)
    assert max(results) - min(results) < .10 * abs(reference)


def test_real_femm5_grounded_empty_shell():
    p = ElectrostaticProblem(conductors=(SphericalShell(.1, .12, Grounded()),))
    with Femm5Backend().solve(p) as numeric:
        assert numeric.source_radius_m is None
        assert abs(numeric.conductor_results[0].potential_v) < .01
        assert abs(numeric.conductor_results[0].net_charge_c) < 1e-11
        for r in (.05,.11,.18):
            assert abs(numeric.potential_at((r,0,0))) < .01
            assert abs(numeric.electric_field_at((r,0,0))[0]) < .02


def offcenter_problem(constraint, q=1e-9, position=(0, 0, .04)):
    return ElectrostaticProblem(
        charges=(PointCharge(q, position),),
        conductors=(SphericalShell(.1, .12, constraint),),
    )


@pytest.mark.parametrize("constraint", [Grounded(), FixedNetCharge(0), FixedNetCharge(2e-9)])
def test_real_femm5_offcenter_potential_field_and_charge(constraint):
    """Real off-center FEMM fields compared to Kelvin image-charge reference."""
    from conductor_sim.image_charge import ImageChargeShellBackend
    p = offcenter_problem(constraint)
    exact = ImageChargeShellBackend().solve(p)
    with Femm5Backend().solve(p) as numeric:
        assert numeric.source_displacement_m == pytest.approx(.04)
        assert numeric.source_radius_m == pytest.approx(.01)
        assert numeric.numerical_source_charge_c == pytest.approx(1e-9, rel=.04)
        e = exact.conductor_results[0]
        g = numeric.conductor_results[0]
        assert g.potential_v == pytest.approx(e.potential_v, rel=.09, abs=.75)
        assert g.net_charge_c == pytest.approx(e.net_charge_c, rel=.05, abs=2e-11)
        assert g.inner_surface_charge_c == pytest.approx(-1e-9, rel=.04)
        assert g.outer_surface_charge_c == pytest.approx(e.outer_surface_charge_c, rel=.06, abs=3e-11)
        for pt in [(.05,0,0), (0,0,-.06), (0,0,.075), (.075,0,.015),
                   (.11,0,0), (.2,0,.05)]:
            assert numeric.potential_at(pt) == pytest.approx(
                exact.potential_at(pt), rel=.16, abs=1.5
            )
        for pt in [(.05,0,0), (0,0,-.06), (0,0,.075), (.075,0,.015),
                   (.11,0,0), (.2,0,.05)]:
            assert numeric.electric_field_at(pt) == pytest.approx(
                exact.electric_field_at(pt), rel=.32, abs=130
            )


def test_real_femm5_nonuniform_induced_surface_charge_density():
    """Local sigma is estimated from the *numerical field*, not analytical inference."""
    from math import pi, sin
    from conductor_sim.image_charge import ImageChargeShellBackend
    p = offcenter_problem(FixedNetCharge(0))
    exact = ImageChargeShellBackend().solve(p)
    with Femm5Backend(settings=Femm5Settings(mesh_scale=.7, arc_degrees=3.)).solve(p) as numeric:
        thetas = (.14, .4, .85, 1.4, 2., 2.65, 3.0)
        observed = [numeric.surface_charge_density("inner", t) for t in thetas]
        reference = [exact.surface_charge_density("inner", t) for t in thetas]
        assert all(value < 0 for value in observed)
        assert abs(observed[0]) > 3*abs(observed[-1])
        for got, expected in zip(observed, reference):
            assert got == pytest.approx(expected, rel=.55)
        # Numerically integrate FEMM's near-surface E over the inner sphere,
        # without using FEMM's conductor total charge or Gauss partition result.
        n = 60
        integral = sum(numeric.surface_charge_density("inner", (k+.5)*pi/n)*
                       sin((k+.5)*pi/n) for k in range(n))
        integral *= 2*pi*.1**2*pi/n
        assert integral == pytest.approx(-1e-9, rel=.30)
        # An isolated spherical outer conductor has uniform positive sigma.
        outer_ref = exact.surface_charge_density("outer", .3)
        for t in (.3, .95, 1.6, 2.5):
            assert numeric.surface_charge_density("outer", t) == pytest.approx(outer_ref, rel=.40)


@pytest.mark.parametrize("q", [1e-9, -1e-9])
def test_real_femm5_grounded_offcenter_and_charge_sign(q):
    from conductor_sim.image_charge import ImageChargeShellBackend
    p = offcenter_problem(Grounded(), q)
    exact = ImageChargeShellBackend().solve(p)
    with Femm5Backend().solve(p) as numerical:
        assert numerical.conductor_results[0].net_charge_c == pytest.approx(-q, rel=.05)
        assert numerical.surface_charge_density("inner", .3) * q < 0
        for pt in [(.04,0,.20), (0,0,.30), (.3,0,0)]:
            assert numerical.potential_at(pt) == pytest.approx(0, abs=1.0)
            assert numerical.electric_field_at(pt) == pytest.approx((0,0,0), abs=5.)
        assert numerical.potential_at((0,0,-.04)) == pytest.approx(
            exact.potential_at((0,0,-.04)), rel=.15, abs=1
        )


def test_real_femm5_offcenter_world_axis_covariance():
    """World orientations use the same axisymmetric numerical geometry."""
    px = offcenter_problem(FixedNetCharge(0), position=(.04,0,0))
    pz = offcenter_problem(FixedNetCharge(0), position=(0,0,.04))
    with Femm5Backend().solve(px) as a, Femm5Backend().solve(pz) as b:
        pa=(.065,.025,0); pb=(0,.025,.065)
        assert a.potential_at(pa)==pytest.approx(b.potential_at(pb), rel=.03)
        fa=a.electric_field_at(pa); fb=b.electric_field_at(pb)
        assert fa == pytest.approx((fb[2],fb[1],fb[0]),rel=.08,abs=10.)


def test_real_femm5_offcenter_source_size_stability():
    """Finite conductor core should approach point-source field away from core."""
    from conductor_sim.image_charge import ImageChargeShellBackend
    p = offcenter_problem(FixedNetCharge(0))
    exact = ImageChargeShellBackend().solve(p)
    pt=(.05,0,0)
    expected=exact.potential_at(pt)
    potentials=[]
    for fraction in (.04,.07,.10):
        with Femm5Backend(settings=Femm5Settings(source_radius_fraction=fraction,
                                                 mesh_scale=.8)).solve(p) as result:
            potentials.append(result.potential_at(pt))
            assert result.numerical_source_charge_c == pytest.approx(1e-9,rel=.04)
    assert all(x == pytest.approx(expected,rel=.15) for x in potentials)
    assert max(potentials)-min(potentials) < .14*abs(expected)
