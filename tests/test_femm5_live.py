"""REAL FEMM 5 integration checks. Never substitute mocked solver data.

These are skipped unless FEMM5_BINDING is set to femm-qt/pyfemm/femm.py.
Do not interpret normal CI passing with a skip as FEMM numerical validation.
"""

from math import isclose
from os import environ

import pytest

from conductor_sim import (
    ConcentricShellBackend, ElectrostaticProblem, FixedNetCharge, SphericalShell,
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
