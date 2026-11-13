"""Immutable, solver-independent electrostatic problem definitions (SI units)."""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from typing import TypeAlias

EPSILON_0 = 8.8541878128e-12
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
    return tuple(float(v) for v in value)

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



@dataclass(frozen=True, slots=True)
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
        from math import hypot
        object.__setattr__(self, "point_m", _position(self.point_m, "plane point"))
        vector = _position(self.normal, "plane normal")
        magnitude = hypot(*vector)
        if not isfinite(magnitude) or magnitude == 0:
            raise ValueError("plane normal must be finite and nonzero")
        object.__setattr__(self, "normal", tuple(v / magnitude for v in vector))
        if not isinstance(self.constraint, (Grounded, FixedPotential)):
            raise TypeError("an infinite conducting plane requires Grounded() or FixedPotential(V)")
        if (not isinstance(self.name, str) or not self.name.strip()
                or self.name == "<None>"
                or any(ord(c) < 32 or c in '"\\' for c in self.name)):
            raise ValueError("invalid conducting-plane name")

    def signed_distance_m(self, point_m: Vector3) -> float:
        """Positive in vacuum; zero at metal surface; negative inside metal."""
        point = _position(point_m, "point")
        return sum((point[i] - self.point_m[i]) * self.normal[i] for i in range(3))

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
