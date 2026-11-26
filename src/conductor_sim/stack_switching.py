"""Charge transfers and energy ledger for fixed-geometry conductor switching."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from fractions import Fraction
from types import MappingProxyType

from .model import FixedPotential, Grounded
from .planar_stack import PlanarStackSolution, _solve_planar_stack
from .solver import UnsupportedProblem
from .spherical_conductor import _representable
from .spherical_stack import SphericalStackSolution, _solve_spherical_stack
from .stack_coupling import ConductorGroup, PlanarVoltageReference

__all__ = ["TerminalTransfer", "StackSwitchResult", "switch_conductor_stack"]

_Solution = SphericalStackSolution | PlanarStackSolution


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

    delta = {name: q_after - q_before
             for name, q_before, q_after in zip(ids, before._state.charges, after._state.charges)}
    terminals = []
    ground = fixed = work = Fraction()
    for group in groups:
        constraint = group.constraint
        if type(constraint) not in (Grounded, FixedPotential):
            continue
        delivered = -sum(delta[name] for name in group.members)
        terminals.append(TerminalTransfer(group.members, constraint, _representable(delivered, "terminal charge")))
        if type(constraint) is Grounded:
            ground += delivered
        else:
            fixed += delivered
            work -= Fraction(constraint.potential_v) * delivered

    initial_energy, final_energy = before._state.energy, after._state.energy
    delta_energy = final_energy - initial_energy
    return StackSwitchResult(
        before, after, MappingProxyType({name: _representable(q, "charge transfer") for name, q in delta.items()}),
        tuple(terminals), _representable(ground, "ground charge transfer"),
        _representable(fixed, "supply charge transfer"),
        _representable(initial_energy, "initial field energy"), _representable(final_energy, "final field energy"),
        _representable(delta_energy, "field energy change"), _representable(work, "supply work"),
        0.0, _representable(work - delta_energy, "energy residual"),
    )
