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
