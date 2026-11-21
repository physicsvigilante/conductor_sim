from fractions import Fraction

import pytest

from conductor_sim import (ConnectedShellBackend, ConnectionPlan, ElectricalLink, ElectrostaticProblem,
                           FixedNetCharge, Grounded, SphericalShell)
from conductor_sim.solver import UnsupportedProblem
from conductor_sim.thin_shells import ConcentricThinShell, solve_thin_shell_charges


def shell(name, r, q):
    return ConcentricThinShell(name, r, FixedNetCharge(q))


@pytest.mark.parametrize("r", [1, 1e-90, 2**100])
def test_q36_independent_ground_leads(r):
    # Integer multiples preserve the printed ratio exactly for these binary inputs.
    r = Fraction(r)
    shells = (shell("outer", 3*r, 0), shell("inner", r, 0), shell("middle", 2*r, 4))
    links = (ElectricalLink(("inner",), "grounded"), ElectricalLink(("outer",), "grounded"))
    assert solve_thin_shell_charges(shells, links) == {"inner": -1, "middle": 4, "outer": -3}


def test_floating_pair_and_preexisting_ground_requalify_together():
    shells = (shell("A", 1, -3), ConcentricThinShell("B", 2, Grounded()), shell("C", 5, 2))
    q = solve_thin_shell_charges(shells, (ElectricalLink(("A", "C")),))
    assert q["A"] + q["C"] == -1
    voltage = {s.name: sum(q[t.name]/max(s.radius_m,t.radius_m) for t in shells) for s in shells}
    assert voltage["A"] == voltage["C"]
    assert voltage["B"] == 0
    assert all(type(v) is Fraction for v in q.values())


def test_resolved_finite_shells_converge_to_the_ideal_surface_charge_split():
    shells = (shell("inner", 1, 0), shell("middle", 2, 4e-9), shell("outer", 3, 0))
    links = (ElectricalLink(("inner", "outer"), "grounded"),)
    ideal = solve_thin_shell_charges(shells, links)
    errors = []
    for width in (0.1, 0.01, 0.001):
        geometry = ElectrostaticProblem(conductors=tuple(
            SphericalShell(s.radius_m-width, s.radius_m, s.constraint, name=s.name) for s in shells))
        result = ConnectedShellBackend().solve(ConnectionPlan(geometry, links))
        errors.append(abs(result.after.conductor_by_name("inner").net_charge_c - float(ideal["inner"])))
    assert 0 < errors[2] < 0.11*errors[1] < 0.11**2*errors[0]


def test_rejects_coincident_shells_and_overlapping_switches():
    with pytest.raises(ValueError, match="distinct"):
        solve_thin_shell_charges((shell("A", 1, 1), shell("B", 1, -1)))
    with pytest.raises(UnsupportedProblem, match="overlapping"):
        solve_thin_shell_charges((shell("A", 1, 1), shell("B", 2, -1)),
                                (ElectricalLink(("A", "B")), ElectricalLink(("B",), "grounded")))
