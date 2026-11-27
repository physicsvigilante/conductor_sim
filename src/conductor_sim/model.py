"""Immutable, solver-independent electrostatic problem definitions (SI units)."""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from math import fsum, hypot, isfinite
from numbers import Integral
from sys import float_info
from typing import TypeAlias

from ._directions import unit_direction

EPSILON_0 = 8.8541878128e-12
_MIN_NORMAL = float_info.min
Vector3: TypeAlias = tuple[float, float, float]
ORIGIN: Vector3 = (0.0, 0.0, 0.0)

def _finite(value: float, name: str) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")

def _position(value: Vector3, name: str) -> Vector3:
    try:
        valid = len(value) == 3 and all(isfinite(v) for v in value)
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f"{name} must contain exactly three finite coordinates")
    normalized = tuple(float(v) for v in value)
    if any(isinstance(v, Integral) and int(v) != stored for v, stored in zip(value, normalized)):
        raise ValueError(f"{name} contains an integer coordinate that loses precision in binary64")
    return normalized

@dataclass(frozen=True, slots=True)
class Grounded:
    """Zero-potential constraint on an entire connected conductor."""

@dataclass(frozen=True, slots=True)
class FixedNetCharge:
    """Isolated conductor with prescribed total charge."""
    charge_c: float
    def __post_init__(self) -> None:
        _finite(self.charge_c, "net charge")

@dataclass(frozen=True, slots=True)
class FixedPotential:
    """Prescribed potential of a connected conductor in volts."""
    potential_v: float

    def __post_init__(self) -> None:
        _finite(self.potential_v, "fixed potential")

ElectricalConstraint: TypeAlias = Grounded | FixedNetCharge | FixedPotential

@dataclass(frozen=True, slots=True)
class PointCharge:
    charge_c: float
    position_m: Vector3 = ORIGIN
    def __post_init__(self) -> None:
        _finite(self.charge_c, "charge")
        object.__setattr__(self, "position_m", _position(self.position_m, "charge position"))

@dataclass(frozen=True, slots=True)
class SphericalShell:
    inner_radius_m: float
    outer_radius_m: float
    constraint: ElectricalConstraint
    center_m: Vector3 = ORIGIN
    name: str = "shell"
    def __post_init__(self) -> None:
        _finite(self.inner_radius_m, "inner radius")
        _finite(self.outer_radius_m, "outer radius")
        if not 0 < self.inner_radius_m < self.outer_radius_m:
            raise ValueError("shell requires 0 < inner radius < outer radius")
        if not isinstance(self.constraint, (Grounded, FixedNetCharge)):
            raise TypeError("unsupported electrical constraint")
        if (not isinstance(self.name, str) or not self.name.strip()
                or any(ord(c) < 32 or c in '\"\\' for c in self.name)):
            raise ValueError("conductor name must be nonempty text without quotes, backslashes or controls")
        object.__setattr__(self, "center_m", _position(self.center_m, "shell center"))

@dataclass(frozen=True, slots=True)
class SolidSphere:
    """Connected solid spherical conductor; no inner cavity or inner face.

    Supports the same electrical-constraint vocabulary as shells. Numerical
    backend support is separately restricted by geometry and source placement.
    """
    radius_m: float
    constraint: ElectricalConstraint
    center_m: Vector3 = ORIGIN
    name: str = "sphere"

    def __post_init__(self) -> None:
        _finite(self.radius_m, "solid sphere radius")
        if self.radius_m <= 0:
            raise ValueError("solid sphere radius must be positive")
        if not isinstance(self.constraint, (Grounded, FixedNetCharge)):
            raise TypeError("unsupported electrical constraint")
        if (not isinstance(self.name, str) or not self.name.strip()
                or any(ord(c) < 32 or c in '"\\' for c in self.name)):
            raise ValueError("conductor name must be nonempty text without quotes, backslashes or controls")
        object.__setattr__(self, "center_m", _position(self.center_m, "sphere center"))

    @property
    def inner_radius_m(self) -> float:
        """For shape-aware visualization only; zero means no cavity/inner face."""
        return 0.0

    @property
    def outer_radius_m(self) -> float:
        return self.radius_m



class _PlaneNormal(tuple):
    """Unit-length float 3-tuple preserving the exact authored plane direction."""

    def __new__(cls, unit: Vector3, raw: Vector3 | None = None):
        unit_t = tuple(float(x) for x in unit)
        obj = super().__new__(cls, unit_t)
        obj._raw = unit_t if raw is None else tuple(float(x) for x in raw)
        return obj

    def __reduce__(self):
        return (_PlaneNormal, (tuple(self), self._raw))


def _canonical_exact_direction(raw: Vector3) -> tuple[Fraction, Fraction, Fraction]:
    fr = tuple(Fraction(x) for x in raw)
    m = max(abs(x) for x in fr)
    return tuple(x / m for x in fr)


def _plane_raw_normal(normal: Vector3, raw_normal: Vector3 | None = None) -> Vector3:
    if raw_normal is not None:
        return raw_normal
    if isinstance(normal, _PlaneNormal):
        return normal._raw
    return normal


def _plane_exact_numerator(
    point: Vector3,
    point_m: Vector3,
    normal: Vector3,
    raw_normal: Vector3 | None = None,
) -> Fraction:
    rn = _plane_raw_normal(normal, raw_normal)
    return sum(
        (Fraction(point[i]) - Fraction(point_m[i])) * Fraction(rn[i])
        for i in range(3)
    )


def _plane_signed_distance(
    point: Vector3,
    point_m: Vector3,
    normal: Vector3,
    raw_normal: Vector3 | None = None,
) -> float:
    """Cancellation-, overflow-, and subnormal-safe signed distance to a plane."""
    rn = _plane_raw_normal(normal, raw_normal)
    max_coord = max(
        (max(abs(point[i]), abs(point_m[i])) for i in range(3) if rn[i] != 0.0),
        default=0.0,
    )
    if max_coord == 0.0:
        return 0.0
    if (
        _MIN_NORMAL * 1e4 <= max_coord <= 1e150
        and all((rn[i] == 0.0) or (abs(normal[i]) >= _MIN_NORMAL * 1e4) for i in range(3))
    ):
        terms = tuple((point[i] - point_m[i]) * normal[i] for i in range(3))
        if all(isfinite(t) for t in terms):
            raw = fsum(terms)
            bound = max(
                (max(abs(point[i]), abs(point_m[i])) * abs(normal[i]) for i in range(3)),
                default=0.0,
            )
            if isfinite(raw) and abs(raw) >= 0.01 * bound and abs(raw) >= _MIN_NORMAL * 1e4:
                return raw
    exact_num = sum(
        (Fraction(point[i]) - Fraction(point_m[i])) * Fraction(rn[i])
        for i in range(3)
    )
    if exact_num == 0:
        return 0.0
    r_max = max(abs(Fraction(x)) for x in rn)
    r_norm_scaled = hypot(*(float(Fraction(x) / r_max) for x in rn))
    exact = exact_num / (r_max * Fraction(r_norm_scaled))
    try:
        return float(exact)
    except OverflowError:
        return float('inf') if exact > 0 else float('-inf')


@dataclass(frozen=True, slots=True, eq=False)
class ConductingPlane:
    """Ideal infinite conductor occupying the negative-normal half-space.

    ``point_m`` lies on the boundary; ``normal`` points into the physical
    vacuum half-space (where charges may be placed). The plane has no finite
    area and cannot have a prescribed *total* charge. Supports Grounded() or
    FixedPotential(volts); a numerical backend may support fewer constraints.
    Solvers remain free to
    reject more complicated configurations involving this component.
    """
    point_m: Vector3 = ORIGIN
    normal: Vector3 = (0.0, 0.0, 1.0)
    constraint: ElectricalConstraint = Grounded()
    name: str = "plane"

    def __post_init__(self) -> None:
        object.__setattr__(self, "point_m", _position(self.point_m, "plane point"))
        raw_input = self.normal
        if isinstance(raw_input, _PlaneNormal):
            vector = _position(raw_input._raw, "plane normal")
        else:
            vector = _position(raw_input, "plane normal")
        unit = unit_direction(vector, "plane normal must be finite and nonzero")
        object.__setattr__(self, "normal", _PlaneNormal(unit, vector))
        if not isinstance(self.constraint, (Grounded, FixedPotential)):
            raise TypeError("an infinite conducting plane requires Grounded() or FixedPotential(V)")
        if (not isinstance(self.name, str) or not self.name.strip()
                or self.name == "<None>"
                or any(ord(c) < 32 or c in '"\\' for c in self.name)):
            raise ValueError("invalid conducting-plane name")

    @property
    def _raw_normal(self) -> Vector3:
        return _plane_raw_normal(self.normal)

    def _identity_key(self):
        return (
            self.point_m,
            tuple(self.normal),
            _canonical_exact_direction(self._raw_normal),
            self.constraint,
            self.name,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ConductingPlane):
            return NotImplemented
        return self._identity_key() == other._identity_key()

    def __hash__(self) -> int:
        return hash(self._identity_key())

    def signed_distance_m(self, point_m: Vector3) -> float:
        """Positive in vacuum; zero at metal surface; negative inside metal."""
        return _plane_signed_distance(
            _position(point_m, "point"),
            self.point_m,
            self.normal,
            self._raw_normal,
        )

Conductor: TypeAlias = SphericalShell | SolidSphere | ConductingPlane

@dataclass(frozen=True, slots=True)
class ElectrostaticProblem:
    charges: tuple[PointCharge, ...] = ()
    conductors: tuple[Conductor, ...] = ()
    permittivity_f_per_m: float = EPSILON_0
    def __post_init__(self) -> None:
        object.__setattr__(self, "charges", tuple(self.charges))
        object.__setattr__(self, "conductors", tuple(self.conductors))
        _finite(self.permittivity_f_per_m, "permittivity")
        if self.permittivity_f_per_m <= 0:
            raise ValueError("permittivity must be positive")
        if any(not isinstance(c, PointCharge) for c in self.charges):
            raise TypeError("all charges must be PointCharge objects")
        if any(not isinstance(c, (SphericalShell, SolidSphere, ConductingPlane)) for c in self.conductors):
            raise TypeError("all conductors must be supported objects")
        names = [c.name for c in self.conductors]
        if len(names) != len(set(names)):
            raise ValueError("conductor names must be unique")
