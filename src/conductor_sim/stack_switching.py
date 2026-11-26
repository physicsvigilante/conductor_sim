"""Charge transfers and energy ledgers for fixed-geometry switching and quasi-static transitions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from fractions import Fraction
from types import MappingProxyType

from .model import FixedNetCharge, FixedPotential, Grounded
from .planar_stack import PlanarStackProblem, PlanarStackSolution, _solve_planar_stack
from .solver import UnsupportedProblem
from .spherical_conductor import _representable
from .spherical_stack import SphericalStackProblem, SphericalStackSolution, _solve_spherical_stack
from .stack_coupling import ConductorGroup, Floating, PlanarVoltageReference

__all__ = [
    "TerminalTransfer",
    "StackSwitchResult",
    "switch_conductor_stack",
    "QuasistaticTransitionResult",
    "quasistatic_stack_transition",
]

_Solution = SphericalStackSolution | PlanarStackSolution
_Problem = SphericalStackProblem | PlanarStackProblem
_ZERO = Fraction()


@dataclass(frozen=True, slots=True)
class TerminalTransfer:
    """One final terminal's signed conventional charge received from the stack."""

    members: tuple[str, ...]
    constraint: Grounded | FixedPotential
    charge_delivered_c: float


@dataclass(frozen=True, slots=True)
class StackSwitchResult:
    """Static endpoints and charge and energy accounting for one ideal switch."""

    before: _Solution
    after: _Solution
    delta_q_c: Mapping[str, float]
    terminal_transfers: tuple[TerminalTransfer, ...]
    charge_delivered_to_ground_c: float
    charge_delivered_to_fixed_potential_supplies_c: float
    initial_field_energy_j: float
    final_field_energy_j: float
    delta_field_energy_j: float
    supply_work_j: float
    external_work_j: float
    energy_residual_j: float


@dataclass(frozen=True, slots=True)
class QuasistaticTransitionResult:
    """Reversible endpoint work and energy ledger for a quasi-static geometry/dielectric change.

    Models an ideal reversible transition between two static endpoints with unchanged
    electrical connections and constant terminal voltages (or conserved floating charges).
    Does not claim 1D field validity for partial lateral dielectric insertion.
    """

    before: _Solution
    after: _Solution
    delta_q_c: Mapping[str, float]
    terminal_transfers: tuple[TerminalTransfer, ...]
    charge_delivered_to_ground_c: float
    charge_delivered_to_fixed_potential_supplies_c: float
    initial_field_energy_j: float
    final_field_energy_j: float
    delta_field_energy_j: float
    supply_work_j: float
    external_work_j: float
    electrostatic_work_on_agent_j: float
    energy_residual_j: float
    idealization: str = "endpoint-capacitance-work (fringe-free; no partial lateral insertion 1D field claims)"


def _terminal_accounting(
    ids: tuple[str, ...],
    groups: tuple[ConductorGroup, ...],
    q_before: tuple[Fraction, ...],
    q_after: tuple[Fraction, ...],
):
    delta = {name: qa - qb for name, qb, qa in zip(ids, q_before, q_after)}
    terminals = []
    ground = fixed = work = _ZERO
    for group in groups:
        constraint = group.constraint
        if type(constraint) not in (Grounded, FixedPotential):
            continue
        delivered = -sum((delta[name] for name in group.members), _ZERO)
        terminals.append(TerminalTransfer(group.members, constraint, _representable(delivered, "terminal charge")))
        if type(constraint) is Grounded:
            ground += delivered
        else:
            fixed += delivered
            work -= Fraction(constraint.potential_v) * delivered
    return delta, tuple(terminals), ground, fixed, work


def switch_conductor_stack(
    before: _Solution, groups: tuple[ConductorGroup, ...], *,
    voltage_reference: PlanarVoltageReference | None = None,
) -> StackSwitchResult:
    """Reconnect a solved stack to ideal wires and constant-potential terminals.

    Floating groups conserve the sum of their members' initial charges.
    An explicit FixedNetCharge in a final group must match that conserved
    charge. Use a planar voltage reference when all final groups float.
    A Floating constraint in the returned problem derives its charge from this
    switch's initial state. Use the returned solution as the next initial state.

    Positive supply work adds energy to the field system. Charge delivered to
    the terminals has the opposite sign to the stack's charge gain. The
    residual is the energy balance for the switch, whose dynamics are unspecified.
    """
    groups = tuple(groups)
    if type(before) is SphericalStackSolution:
        if voltage_reference is not None:
            raise UnsupportedProblem("spherical voltage is referenced to infinity")
        if before.has_active_sources:
            raise UnsupportedProblem("total field energy is undefined with active point charges")
        ids = before.problem.conductor_ids
        after = _solve_spherical_stack(replace(before.problem, groups=groups), before._state.charges)
    elif type(before) is PlanarStackSolution:
        ids = before.problem.plate_ids
        problem = replace(before.problem, groups=groups, voltage_reference=voltage_reference)
        after = _solve_planar_stack(problem, before._state.charges)
    else:
        raise UnsupportedProblem("switching requires a solved spherical or planar stack")

    delta, terminals, ground, fixed, work = _terminal_accounting(
        ids, groups, before._state.charges, after._state.charges
    )
    initial_energy, final_energy = before._state.energy, after._state.energy
    delta_energy = final_energy - initial_energy
    return StackSwitchResult(
        before, after, MappingProxyType({name: _representable(q, "charge transfer") for name, q in delta.items()}),
        terminals, _representable(ground, "ground charge transfer"),
        _representable(fixed, "supply charge transfer"),
        _representable(initial_energy, "initial field energy"), _representable(final_energy, "final field energy"),
        _representable(delta_energy, "field energy change"), _representable(work, "supply work"),
        0.0, _representable(work - delta_energy, "energy residual"),
    )


def _same_reversible_topology(
    before_groups: tuple[ConductorGroup, ...],
    after_groups: tuple[ConductorGroup, ...],
) -> bool:
    """Verify identical wire partitions and constant terminal voltages across a quasi-static transition."""
    before_map = {frozenset(g.members): g.constraint for g in before_groups}
    after_map = {frozenset(g.members): g.constraint for g in after_groups}
    if set(before_map) != set(after_map):
        return False
    for members, c_before in before_map.items():
        c_after = after_map[members]
        tb, ta = type(c_before), type(c_after)
        if tb in (Grounded, FixedPotential):
            if ta not in (Grounded, FixedPotential):
                return False
            vb = _ZERO if tb is Grounded else Fraction(c_before.potential_v)
            va = _ZERO if ta is Grounded else Fraction(c_after.potential_v)
            if vb != va:
                return False
        else:
            if ta not in (FixedNetCharge, Floating):
                return False
    return True


def quasistatic_stack_transition(
    before: _Solution, after_problem: _Problem,
) -> QuasistaticTransitionResult:
    """Compute reversible endpoint work for a quasi-static geometry or dielectric change.

    Requires unchanged conductor IDs, unchanged electrical wire groups, constant
    prescribed terminal voltages, and conserved floating-group charges (source-free).
    Energy balance: W_supply + W_external = Delta_U_field, with energy_residual_j == 0.0.
    """
    if type(before) is SphericalStackSolution:
        if type(after_problem) is not SphericalStackProblem:
            raise UnsupportedProblem("spherical transition requires a SphericalStackProblem endpoint")
        if before.has_active_sources:
            raise UnsupportedProblem("quasi-static field energy ledger is undefined with active point charges")
        if before.problem.conductor_ids != after_problem.conductor_ids:
            raise UnsupportedProblem("quasi-static transition requires matching conductor IDs")
        if not _same_reversible_topology(before.problem.groups, after_problem.groups):
            raise UnsupportedProblem(
                "quasistatic_stack_transition requires unchanged electrical connections and constant "
                "terminal voltages; use switch_conductor_stack for switching"
            )
        ids = before.problem.conductor_ids
        after = _solve_spherical_stack(after_problem, before._state.charges)
        if after.has_active_sources:
            raise UnsupportedProblem("quasi-static field energy ledger is undefined with active point charges")
    elif type(before) is PlanarStackSolution:
        if type(after_problem) is not PlanarStackProblem:
            raise UnsupportedProblem("planar transition requires a PlanarStackProblem endpoint")
        if before.problem.plate_ids != after_problem.plate_ids:
            raise UnsupportedProblem("quasi-static transition requires matching plate IDs")
        if not _same_reversible_topology(before.problem.groups, after_problem.groups):
            raise UnsupportedProblem(
                "quasistatic_stack_transition requires unchanged electrical connections and constant "
                "terminal voltages; use switch_conductor_stack for switching"
            )
        ids = before.problem.plate_ids
        after = _solve_planar_stack(after_problem, before._state.charges)
    else:
        raise UnsupportedProblem("quasi-static transition requires a solved spherical or planar stack")

    delta, terminals, ground, fixed, work = _terminal_accounting(
        ids, after_problem.groups, before._state.charges, after._state.charges
    )
    initial_energy, final_energy = before._state.energy, after._state.energy
    delta_energy = final_energy - initial_energy
    external_work = delta_energy - work
    return QuasistaticTransitionResult(
        before, after, MappingProxyType({name: _representable(q, "charge transfer") for name, q in delta.items()}),
        terminals, _representable(ground, "ground charge transfer"),
        _representable(fixed, "supply charge transfer"),
        _representable(initial_energy, "initial field energy"), _representable(final_energy, "final field energy"),
        _representable(delta_energy, "field energy change"), _representable(work, "supply work"),
        _representable(external_work, "external work"),
        _representable(-external_work, "electrostatic work on agent"),
        0.0,
    )
