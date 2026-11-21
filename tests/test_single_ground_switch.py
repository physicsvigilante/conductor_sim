"""Workbook p14 numerical Q2 and composable ground-reservoir transfers."""
from math import isclose

import pytest

from conductor_sim import (
    ConnectionPlan, ConnectedShellBackend, ElectricalLink, ElectrostaticProblem,
    FixedNetCharge, Grounded, SphericalShell,
)


def shell(name, inner, outer, charge):
    return SphericalShell(inner, outer, FixedNetCharge(charge), name=name)


def ground_inner(q, width):
    # Teacher radius and shell thickness; the original gives thin radii r and 3r.
    inner = shell("inner", 0.1 - width, 0.1, 0.0)
    outer = shell("outer", 0.3 - width, 0.3, q)
    link = ElectricalLink(("inner",), mode="grounded")
    plan = ConnectionPlan(ElectrostaticProblem(conductors=(outer, inner)), (link,))
    return ConnectedShellBackend().solve(plan), link


@pytest.mark.parametrize("q", [3e-9, -3e-9])
def test_original_direction_and_thin_shell_limit(q):
    previous_error = float("inf")
    for width in (1e-3, 1e-5, 1e-7):
        run, link = ground_inner(q, width)
        to_earth = run.charge_to_ground_c(link)
        # Zero final potential = k*(Q_inner/r + Q_outer_faces' contributions).
        n = 1 + 0.3 * (1/0.1 - 1/(0.3-width))
        assert isclose(to_earth, q/n, rel_tol=1e-13)
        assert isclose(run.after.conductor_by_name("inner").net_charge_c, -to_earth, rel_tol=2e-15)
        assert run.after.conductor_by_name("outer").net_charge_c == q
        error = abs(q/to_earth - 3)
        assert error < previous_error
        previous_error = error
    assert previous_error < 1e-6


def test_independent_ground_and_floating_connection():
    ground = ElectricalLink(("inner",), mode="grounded")
    floating = ElectricalLink(("middle", "outer"))
    shells = (shell("inner", .02, .03, 1e-9),
              shell("middle", .05, .06, 2e-9),
              shell("outer", .09, .10, 4e-9))
    run = ConnectedShellBackend().solve(ConnectionPlan(
        ElectrostaticProblem(conductors=shells), (ground, floating)))
    assert isclose(sum(run.after.conductor_by_name(k).net_charge_c
                       for k in floating.members), 6e-9, rel_tol=1e-13)
    assert isclose(run.charge_to_ground_c(ground),
                   7e-9 - sum(r.net_charge_c for r in run.after.conductor_results),
                   rel_tol=1e-13)
    with pytest.raises(ValueError):
        run.charge_to_ground_c(floating)
    with pytest.raises(ValueError):
        run.charge_to_ground_c(ElectricalLink(("outer",), mode="grounded"))


def test_one_member_floating_wire_still_refused():
    with pytest.raises(ValueError):
        ElectricalLink(("inner",))


def test_existing_earth_connection_makes_new_wire_transfer_ambiguous():
    old = SphericalShell(.02, .03, Grounded(), name="old")
    new = shell("new", .04, .05, 1e-9)
    link = ElectricalLink(("old", "new"), mode="grounded")
    run = ConnectedShellBackend().solve(ConnectionPlan(
        ElectrostaticProblem(conductors=(old, new)), (link,)))
    with pytest.raises(ValueError, match="existing earth path"):
        run.charge_to_ground_c(link)


@pytest.mark.parametrize("a,b,c,q0,q", [
    (.09429007747323771, .6870894465573995, .766280973830598, -.12481917031316854, 1.0),
    (.1, .299, .3, -1.0011160714285713e-9, 3e-9),
    (.125, .25, .5, -.125, .375),
    (.125, .25, .5, 0.0, 0.0),
])
def test_nearly_balanced_switch_preserves_sign(a, b, c, q0, q):
    from fractions import Fraction

    initial = (shell("inner", a/2, a, q0), shell("outer", b, c, q))
    link = ElectricalLink(("inner",), mode="grounded")
    result = ConnectedShellBackend().solve(ConnectionPlan(ElectrostaticProblem(conductors=initial), (link,)))
    a, b, c = map(Fraction, (a, b, c))
    expected = Fraction(q0) + Fraction(q) / (1 + c*(1/a-1/b))
    assert result.charge_to_ground_c(link) == float(expected)
