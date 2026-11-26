"""Phase 0d: FEMM capability query (M01 C7), force-index policy (C8), single-shell grounding (C11).

The single-shell oracle is exact rational Gauss/Dirichlet bookkeeping written
from the radial closed form, not from the code under test.
"""
from fractions import Fraction as F
from math import isclose, pi

import pytest

import conductor_sim as cs
from conductor_sim import (
    EPSILON_0, ConductingPlane, ConnectedShellBackend, ConnectionPlan, ElectricalLink,
    ElectrostaticProblem, ExternalMultiSphereBackend, FixedNetCharge, FixedPotential, Grounded,
    GroundedPlaneBackend, NestedShellBackend, PointCharge, SolidSphere, SphericalShell,
)


def _never():
    raise AssertionError("a capability query must not open a FEMM session")


FAMILIES = {
    "plane": ElectrostaticProblem((PointCharge(1e-9, (0., 0., 1.)),), (ConductingPlane(),)),
    "plates": ElectrostaticProblem((), (ConductingPlane(name="lo"),
                                        ConductingPlane((0., 0., 1.), (0., 0., -1.), FixedPotential(10.), "up"))),
    "shell": ElectrostaticProblem((PointCharge(1e-9, (.1, 0., 0.)),), (SphericalShell(1., 2., Grounded()),)),
    "solid-exterior": ElectrostaticProblem((PointCharge(1e-9, (3., 0., 0.)),), (SolidSphere(1., Grounded()),)),
    "mixed": ElectrostaticProblem((), (SolidSphere(1., FixedNetCharge(1e-9), name="b"),
                                       SphericalShell(2., 3., Grounded(), name="s"))),
    "plane-and-shell": ElectrostaticProblem((), (SphericalShell(1., 2., Grounded(), name="s"),
                                                 ConductingPlane((0., 0., 5.), (0., 0., 1.), name="p"))),
    "empty": ElectrostaticProblem((), ()),
}
FEMM_BACKENDS = sorted(name for name, obj in vars(cs).items()
                       if isinstance(obj, type) and name.startswith("Femm") and name.endswith("Backend"))


@pytest.mark.parametrize("backend_name", FEMM_BACKENDS)
@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_femm_capability_queries_return_bool(backend_name, family):
    # W-d1 (D4 = F2): supports() answers for every authored family.
    backend = getattr(cs, backend_name)(session_factory=_never)
    assert isinstance(backend.supports(FAMILIES[family]), bool)


def test_femm_shell_backend_admission_otherwise_unchanged():
    backend = cs.Femm5Backend(session_factory=_never)
    assert backend.supports(FAMILIES["shell"]) is True
    assert backend.supports(FAMILIES["plane"]) is False
    assert backend.supports(FAMILIES["plane-and-shell"]) is False


def _plane_solution():
    charges = (PointCharge(3e-9, (0., 0., 2.)), PointCharge(-2e-9, (1., 0., 1.)))
    return GroundedPlaneBackend().solve(ElectrostaticProblem(charges, (ConductingPlane(),)))


def _multi_solution():
    charges = (PointCharge(-2e-9, (3., 0., 0.)), PointCharge(3e-9, (2., 0., 0.)))
    return ExternalMultiSphereBackend().solve(ElectrostaticProblem(charges, (SolidSphere(1., Grounded()),)))


METHODS = (
    lambda: (_plane_solution(), "force_on_charge_n"),
    lambda: (_multi_solution(), "force_on_charge_n"),
    lambda: (_multi_solution(), "conductor_force_on_charge_n"),
)


@pytest.mark.parametrize("make", METHODS)
def test_force_index_policy_matches_shared_result(make):
    # W-d2 (D5): -1 no longer selects a site and then divides by its own distance.
    solution, name = make()
    method = getattr(solution, name)
    for bad, error in ((-1, IndexError), (2, IndexError), (True, TypeError), (1.0, TypeError)):
        with pytest.raises(error):
            method(bad)
    shared = cs.force_energy(solution)
    with pytest.raises(IndexError, match="outside the grouped active sites"):
        shared.force_on_charge_n(-1)
    with pytest.raises(IndexError, match="outside the grouped active sites"):
        method(-1)
    with pytest.raises(TypeError, match="source index must be an integer"):
        method(False)


def test_valid_force_indices_unchanged():
    # Valid-index results against an independent Kelvin closed form (x axis).
    solution = _multi_solution()
    k = 1/(4*pi*EPSILON_0)
    sites = ((F(2), F(3, 10**9)), (F(3), F(-2, 10**9)))
    for index, (x, q) in enumerate(sites):
        images = [(F(1)/p, -c/p) for p, c in sites]
        metal = sum(c*(x-p)/abs(x-p)**3 for p, c in images)
        direct = sum(c*(x-p)/abs(x-p)**3 for p, c in sites if p != x)
        assert isclose(solution.conductor_force_on_charge_n(index)[0], k*float(q*metal), rel_tol=1e-14)
        assert isclose(solution.force_on_charge_n(index)[0], k*float(q*(metal+direct)), rel_tol=1e-14)


def _single_shell_plan(link_members=("s",)):
    shell = SphericalShell(.15, .30, FixedNetCharge(-1e-9), name="s")
    problem = ElectrostaticProblem((PointCharge(-2e-9, (.05, 0., 0.)),), (shell,))
    return ConnectionPlan(problem, (ElectricalLink(link_members, "grounded"),))


def test_single_shell_can_be_grounded_through_a_connection_plan():
    # W-d3 (F5). Exact oracle: Q_cavity=-2 nC, Q_metal=-1 nC, b=0.30 m.
    plan = _single_shell_plan()
    backend = ConnectedShellBackend()
    assert backend.supports(plan)
    result = backend.solve(plan)
    k = 1/(4*pi*EPSILON_0)
    q_cav, q_metal, b = F(-2, 10**9), F(-1, 10**9), F(3, 10)
    before = result.before.conductor_by_name("s")
    after = result.after.conductor_by_name("s")
    assert isclose(before.potential_v, k*float((q_cav+q_metal)/b), rel_tol=1e-14)
    assert isclose(before.potential_v, -89.8755179226, rel_tol=1e-10)
    assert after.potential_v == 0.0
    assert after.outer_surface_charge_c == 0.0
    assert isclose(after.net_charge_c, float(-q_cav), rel_tol=1e-15)
    link = plan.links[0]
    assert isclose(result.charge_to_ground_c(link), float(q_metal-(-q_cav)), rel_tol=1e-15)
    released = (q_cav+q_metal)**2/(2*b)  # exterior field energy before; zero after
    assert isclose(result.released_field_energy_j, k*float(released), rel_tol=1e-12)
    assert isclose(result.released_field_energy_j, 1.34813276884e-7, rel_tol=1e-10)


def test_single_shell_admission_keeps_other_refusals():
    backend = ConnectedShellBackend()
    with pytest.raises(Exception):
        backend.solve(_single_shell_plan(("s", "ghost")))
    assert not backend.supports(_single_shell_plan(("s", "ghost")))


def test_two_shell_connection_unchanged():
    shells = (SphericalShell(.1, .12, FixedNetCharge(2e-9), name="a"),
              SphericalShell(.2, .22, FixedNetCharge(-1e-9), name="b"))
    plan = ConnectionPlan(ElectrostaticProblem((), shells), (ElectricalLink(("a", "b"), "floating"),))
    result = ConnectedShellBackend().solve(plan)
    # Inner shell's charge moves to the outer shell (Faraday cage), total kept.
    assert isclose(result.after.conductor_by_name("a").net_charge_c, 0.0, abs_tol=1e-24)
    assert isclose(result.after.conductor_by_name("b").net_charge_c, 1e-9, rel_tol=1e-14)
