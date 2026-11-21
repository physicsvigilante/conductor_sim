"""Named ideal concentric conducting surfaces in a homogeneous dielectric."""
from dataclasses import dataclass
from fractions import Fraction

from .connections import ElectricalLink, _coefficients, _connection_charges, _wire_groups
from .model import FixedNetCharge, Grounded
from .solver import UnsupportedProblem


def _rational(value, label):
    if type(value) not in (int, float, Fraction):
        raise ValueError(f"{label} requires a finite real number")
    try:
        return Fraction(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{label} requires a finite real number") from exc


@dataclass(frozen=True, slots=True)
class ConcentricThinShell:
    name: str
    radius_m: float
    constraint: FixedNetCharge | Grounded

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("shell name must be nonempty")
        if _rational(self.radius_m, "radius") <= 0:
            raise ValueError("shell radius must be positive")
        if type(self.constraint) is FixedNetCharge:
            _rational(self.constraint.charge_c, "net charge")
        elif type(self.constraint) is not Grounded:
            raise UnsupportedProblem("shell must be isolated or grounded")

    @property
    def inner_radius_m(self):
        return self.radius_m

    @property
    def outer_radius_m(self):
        return self.radius_m


def solve_thin_shell_charges(shells, links=()):
    """Return exact named net charges (C) after all named switches close together.

    Radii share one centre. The cavity is empty, with no external sources.
    Presenting a body with two distinct face radii requires a finite-shell model.
    """
    shells = tuple(shells)
    if not 2 <= len(shells) <= 8 or any(type(shell) is not ConcentricThinShell for shell in shells):
        raise UnsupportedProblem("require two to eight concentric ideal thin shells")
    if len({s.radius_m for s in shells}) != len(shells) or len({s.name for s in shells}) != len(shells):
        raise ValueError("shell radii and names must each be distinct")
    links = tuple(links)
    if any(type(link) is not ElectricalLink for link in links):
        raise TypeError("expected named ElectricalLink objects")
    ordered, groups, grouped = _wire_groups(tuple(sorted(shells, key=lambda s: s.radius_m)), links)
    matrix, offsets = _coefficients(ordered, Fraction(0), exact=True)
    charges = _connection_charges(ordered, groups, grouped, matrix, offsets, exact=True)
    return {shape.name: q for shape, q in zip(ordered, charges)}
