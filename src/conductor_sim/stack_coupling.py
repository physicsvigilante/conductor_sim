"""Electrostatic coupling for a chain of conductors and an optional exterior."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .model import FixedNetCharge, FixedPotential, Grounded
from .nested_shells import _solve_linear
from .solver import UnsupportedProblem
from .spherical_conductor import _finite_si, _identity, _representable

__all__ = ["ConductorGroup", "Floating", "PlanarVoltageReference"]

_ZERO = Fraction()


@dataclass(frozen=True, slots=True)
class Floating:
    """In a switch, conserve the total initial charge of this group's members."""


@dataclass(frozen=True, slots=True)
class ConductorGroup:
    """An equipotential singleton or wire group with one electrical constraint.

    FixedNetCharge sets the whole group's charge in a static solve. Use
    Floating in a switch to obtain that total from the initial equilibrium.
    Static charge targets use the supplied number's decimal spelling.
    """

    members: tuple[str, ...]
    constraint: Grounded | FixedPotential | FixedNetCharge | Floating

    def __post_init__(self) -> None:
        if not isinstance(self.members, (tuple, list)) or not self.members:
            raise ValueError("a conductor group needs a sequence of member IDs")
        members = tuple(self.members)
        for member in members:
            _identity(member, "conductor ID")
        if len(set(members)) != len(members):
            raise ValueError("conductor group member IDs must be distinct")
        object.__setattr__(self, "members", members)
        constraint = self.constraint
        if type(constraint) is FixedNetCharge:
            _finite_si(constraint.charge_c, "group net charge")
        elif type(constraint) is FixedPotential:
            _finite_si(constraint.potential_v, "group potential")
        elif type(constraint) not in (Grounded, Floating):
            raise UnsupportedProblem("unsupported conductor group constraint")


@dataclass(frozen=True, slots=True)
class PlanarVoltageReference:
    """Gauge for a charge-constrained neutral planar stack, without a supply."""

    plate_id: str
    potential_v: float = 0.0

    def __post_init__(self) -> None:
        _identity(self.plate_id, "reference plate ID")
        _finite_si(self.potential_v, "reference potential")


@dataclass(frozen=True, slots=True)
class _Gap:
    """Source-free elastance and the two zero-potential face-charge offsets."""

    elastance: Fraction
    left_induced_c: Fraction = _ZERO
    right_induced_c: Fraction = _ZERO


@dataclass(frozen=True, slots=True)
class _Coupling:
    gaps: tuple[_Gap, ...]
    exterior: _Gap | None = None
    first_left_induced_c: Fraction = _ZERO

    def edges(self):
        """Yield (left, right, capacitance), using None for infinity."""
        for i, gap in enumerate(self.gaps):
            yield i, i + 1, 1 / gap.elastance
        if self.exterior is not None:
            yield len(self.gaps), None, 1 / self.exterior.elastance

    def face_charges(self, voltages: tuple[Fraction, ...]):
        left = [self.first_left_induced_c]
        right = []
        for i, gap in enumerate(self.gaps):
            charge = (voltages[i] - voltages[i + 1]) / gap.elastance
            right.append(charge + gap.left_induced_c)
            left.append(-charge + gap.right_induced_c)
        exterior = self.exterior
        last = (_ZERO if exterior is None else
                voltages[-1] / exterior.elastance + exterior.left_induced_c)
        return tuple(left), (*right, last)

    @property
    def capacitance_matrix_f(self) -> tuple[tuple[float, ...], ...]:
        n = len(self.gaps) + 1
        matrix = [[_ZERO] * n for _ in range(n)]
        for i, j, capacitance in self.edges():
            _add_capacitance(matrix, i, j, capacitance)
        return tuple(tuple(_representable(c, "capacitance") for c in row) for row in matrix)


def _add_capacitance(matrix, i, j, capacitance):
    matrix[i][i] += capacitance
    if j is not None:
        matrix[j][j] += capacitance
        matrix[i][j] -= capacitance
        matrix[j][i] -= capacitance


@dataclass(frozen=True, slots=True)
class _State:
    coupling: _Coupling
    voltages: tuple[Fraction, ...]
    left_faces: tuple[Fraction, ...]
    right_faces: tuple[Fraction, ...]
    charges: tuple[Fraction, ...]

    @property
    def energy(self) -> Fraction:
        """Source-free field energy from the gap and exterior voltage drops."""
        return sum(((self.voltages[i] - (self.voltages[j] if j is not None else 0)) ** 2 * c
                    for i, j, c in self.coupling.edges()), _ZERO) / 2


def _group_layout(ids, groups):
    indices = {name: i for i, name in enumerate(ids)}
    assigned = {}
    members = []
    for group_index, group in enumerate(groups):
        if type(group) is not ConductorGroup:
            raise UnsupportedProblem("expected ConductorGroup entries for electrical constraints")
        for name in group.members:
            if name not in indices:
                raise UnsupportedProblem(f"unknown conductor ID {name!r} in group")
            if name in assigned:
                raise UnsupportedProblem(f"overlapping groups at conductor {name!r}")
            assigned[name] = group_index
        members.append(tuple(indices[name] for name in group.members))
    if len(assigned) != len(ids):
        raise UnsupportedProblem("every conductor must belong to one electrical group")
    return tuple(assigned[name] for name in ids), tuple(members)


def _solve_coupling(
    coupling: _Coupling, ids: tuple[str, ...], groups: tuple[ConductorGroup, ...], *,
    reference: PlanarVoltageReference | None = None,
    initial_charges: tuple[Fraction, ...] | None = None,
) -> _State:
    """Collapse wired vertices and solve the charge equations on free groups."""
    group_at, members = _group_layout(ids, groups)
    prescribed = {}
    targets = {}
    for i, (group, indices) in enumerate(zip(groups, members)):
        constraint = group.constraint
        if type(constraint) is Grounded:
            prescribed[i] = _ZERO
        elif type(constraint) is FixedPotential:
            prescribed[i] = Fraction(constraint.potential_v)
        elif initial_charges is None:
            if type(constraint) is Floating:
                raise UnsupportedProblem("static floating groups require FixedNetCharge")
            # Charge targets use the authored decimal spelling, so e.g.
            # (1e-9, 2e-9, -3e-9) is exactly neutral.
            targets[i] = Fraction(str(constraint.charge_c))
        else:
            targets[i] = sum((initial_charges[j] for j in indices), _ZERO)
            if (type(constraint) is FixedNetCharge
                    and constraint.charge_c != float(targets[i])):
                raise UnsupportedProblem("a switch must conserve each final floating group's charge")

    if coupling.exterior is None:
        if prescribed:
            if reference is not None:
                raise UnsupportedProblem("a prescribed terminal already sets the planar voltage reference")
        else:
            if sum(targets.values()) != 0:
                raise UnsupportedProblem("zero planar exterior displacement requires neutral total charge")
            if type(reference) is not PlanarVoltageReference or reference.plate_id not in ids:
                raise UnsupportedProblem("a charge-constrained planar stack requires a voltage reference on a plate")
            gauge = group_at[ids.index(reference.plate_id)]
            prescribed[gauge] = Fraction(reference.potential_v)
            del targets[gauge]  # Its charge equation follows from neutrality.

    induced_left, induced_right = coupling.face_charges((_ZERO,) * len(ids))
    free = {group: i for i, group in enumerate(targets)}
    matrix = [[_ZERO] * len(free) for _ in free]
    rhs = [targets[group] - sum(induced_left[i] + induced_right[i] for i in members[group])
           for group in free]
    for i, j, capacitance in coupling.edges():
        left = group_at[i]
        right = group_at[j] if j is not None else None
        if left == right:
            # A short removes the voltage drop, while its induced offsets
            # still contribute to the group's free charge equation.
            continue
        if left in free and right in free:
            _add_capacitance(matrix, free[left], free[right], capacitance)
        else:
            for a, b in ((left, right), (right, left)):
                if a in free:
                    row = free[a]
                    matrix[row][row] += capacitance
                    rhs[row] += capacitance * prescribed.get(b, _ZERO)

    solved = dict(zip(free, _solve_linear(matrix, rhs, exact=True)))
    solved.update(prescribed)
    voltages = tuple(solved[group] for group in group_at)
    left, right = coupling.face_charges(voltages)
    return _State(coupling, voltages, left, right, tuple(a + b for a, b in zip(left, right)))
