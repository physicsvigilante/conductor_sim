"""Static SI solution for one connected spherical conductor, with V(infinity) = 0."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass, field
from fractions import Fraction
from itertools import combinations
from math import copysign, fsum, hypot, isfinite, pi, ulp
from types import MappingProxyType
from typing import Literal

from .model import (
    EPSILON_0, ORIGIN, ElectricalConstraint, FixedNetCharge, FixedPotential,
    Grounded, Vector3, _position,
)
from .solver import ConductorResult, SingularField, UnsupportedProblem

__all__ = [
    "SphericalCavity", "SphericalPointSource", "SphericalConductorProblem",
    "SphericalRegion", "SphericalConductorSolution", "solve_spherical_conductor",
]


def _finite_si(value: float, label: str, *, positive: bool = False) -> None:
    try:
        finite = type(value) in (int, float) and isfinite(value)
    except OverflowError:
        finite = False
    if not finite or (positive and value <= 0):
        requirement = "finite and positive" if positive else "finite"
        raise ValueError(f"{label} must be {requirement} in SI")


def _identity(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")


def _difference(a: Vector3, b: Vector3) -> Vector3:
    return tuple(x - y for x, y in zip(a, b))


def _squared_distance(a: Vector3, b: Vector3) -> Fraction:
    return sum(((Fraction(x) - Fraction(y)) ** 2 for x, y in zip(a, b)), Fraction())


def _distance(vector: Vector3) -> float:
    length = hypot(*vector)
    if not isfinite(length):
        raise UnsupportedProblem("distance outside the finite binary64 range")
    return length


def _radial_gap(point: Vector3, center: Vector3, distance: float, radius: float) -> float:
    gap = distance - radius
    if abs(gap) > radius * 1e-4:
        return gap
    # Recover the signed gap without subtracting two nearly equal rounded
    # radii. This also distinguishes an exact wall from rounded hypot == radius.
    squared_gap = _squared_distance(point, center) - Fraction(radius) ** 2
    gap = float(squared_gap / (Fraction(distance) + Fraction(radius)))
    if gap == 0 and squared_gap:
        return copysign(ulp(0.), -1. if squared_gap < 0 else 1.)
    return gap


def _representable(value: Fraction, label: str) -> float:
    try:
        result = float(value)
    except OverflowError as exc:
        raise UnsupportedProblem(f"{label} outside the binary64 range") from exc
    if result == 0 and value:
        raise UnsupportedProblem(f"{label} outside the binary64 range")
    return result


def _sum_finite(terms) -> float:
    try:
        total = fsum(terms)
    except (OverflowError, ValueError) as exc:
        raise UnsupportedProblem("local observable outside the binary64 range") from exc
    if not isfinite(total):
        raise UnsupportedProblem("local observable outside the binary64 range")
    return total


@dataclass(frozen=True, slots=True)
class SphericalCavity:
    """One sealed void inside the connected metal."""

    cavity_id: str
    center_m: Vector3
    radius_m: float

    def __post_init__(self) -> None:
        _identity(self.cavity_id, "cavity ID")
        _finite_si(self.radius_m, "cavity radius", positive=True)
        object.__setattr__(self, "radius_m", float(self.radius_m))
        object.__setattr__(self, "center_m", _position(self.center_m, "cavity center"))


@dataclass(frozen=True, slots=True)
class SphericalPointSource:
    """A held point charge with a stable ID and a world-frame position."""

    source_id: str
    charge_c: float
    position_m: Vector3

    def __post_init__(self) -> None:
        _identity(self.source_id, "source ID")
        _finite_si(self.charge_c, "source charge")
        object.__setattr__(self, "position_m", _position(self.position_m, "source position"))


@dataclass(frozen=True, slots=True)
class SphericalConductorProblem:
    """One outer sphere and any number of named, disjoint spherical cavities.

    A single homogeneous permittivity applies in the exterior and all cavities.
    Supply held sources in world coordinates; geometry determines their regions.
    Use Grounded() or FixedNetCharge(coulombs) for the connected metal.
    """

    outer_radius_m: float
    constraint: ElectricalConstraint
    _: KW_ONLY
    center_m: Vector3 = ORIGIN
    cavities: tuple[SphericalCavity, ...] = ()
    sources: tuple[SphericalPointSource, ...] = ()
    permittivity_f_per_m: float = EPSILON_0
    conductor_id: str = "sphere"

    def __post_init__(self) -> None:
        _identity(self.conductor_id, "conductor ID")
        _finite_si(self.outer_radius_m, "outer radius", positive=True)
        _finite_si(self.permittivity_f_per_m, "permittivity", positive=True)
        object.__setattr__(self, "outer_radius_m", float(self.outer_radius_m))
        object.__setattr__(self, "center_m", _position(self.center_m, "outer sphere center"))
        object.__setattr__(self, "cavities", tuple(self.cavities))
        object.__setattr__(self, "sources", tuple(self.sources))
        if any(type(cavity) is not SphericalCavity for cavity in self.cavities):
            raise UnsupportedProblem("the single conductor requires sealed spherical cavities")
        if any(type(source) is not SphericalPointSource for source in self.sources):
            raise UnsupportedProblem("sources must be held spherical-conductor point sources")
        if len({c.cavity_id for c in self.cavities}) != len(self.cavities):
            raise ValueError("cavity IDs must be unique")
        if len({s.source_id for s in self.sources}) != len(self.sources):
            raise ValueError("source IDs must be unique")


@dataclass(frozen=True, slots=True)
class SphericalRegion:
    """Location relative to the metal. Cavity locations carry their cavity ID."""

    kind: Literal["exterior", "metal", "cavity", "outer_wall", "cavity_wall"]
    cavity_id: str | None = None


_EXTERIOR = SphericalRegion("exterior")


def _validate_geometry(problem: SphericalConductorProblem) -> None:
    outer_radius = Fraction(problem.outer_radius_m)
    for cavity in problem.cavities:
        clearance = outer_radius - Fraction(cavity.radius_m)
        if clearance <= 0 or _squared_distance(cavity.center_m, problem.center_m) >= clearance**2:
            raise UnsupportedProblem(f"cavity {cavity.cavity_id!r} must be strictly inside the outer sphere")
    for left, right in combinations(problem.cavities, 2):
        separation = Fraction(left.radius_m) + Fraction(right.radius_m)
        if _squared_distance(left.center_m, right.center_m) <= separation**2:
            raise UnsupportedProblem(f"cavities {left.cavity_id!r} and {right.cavity_id!r} touch or overlap")


def _region_at(problem: SphericalConductorProblem, point: Vector3) -> SphericalRegion:
    r = _distance(_difference(point, problem.center_m))
    gap = _radial_gap(point, problem.center_m, r, problem.outer_radius_m)
    if gap == 0:
        return SphericalRegion("outer_wall")
    unresolved = abs(gap) <= 8 * ulp(problem.outer_radius_m)
    if gap > 0:
        if unresolved:
            raise UnsupportedProblem("unresolved near-wall position")
        return _EXTERIOR
    region = SphericalRegion("metal")
    for cavity in problem.cavities:
        r = _distance(_difference(point, cavity.center_m))
        gap = _radial_gap(point, cavity.center_m, r, cavity.radius_m)
        if gap == 0:
            return SphericalRegion("cavity_wall", cavity.cavity_id)
        unresolved |= abs(gap) <= 8 * ulp(cavity.radius_m)
        if gap < 0:
            region = SphericalRegion("cavity", cavity.cavity_id)
            break
    if unresolved:
        raise UnsupportedProblem("unresolved near-wall position")
    return region


@dataclass(frozen=True, slots=True)
class _KelvinSite:
    """A real source and its grounded-sphere image, in a local sphere frame."""

    position_m: Vector3
    charge_c: float
    distance_m: float
    gap_m: float
    direction: Vector3

    def sample(self, point: Vector3, radial: Vector3, radius: float) -> _KelvinPair:
        direct = _difference(point, self.position_m)
        distance = _distance(direct)
        if distance == 0:
            raise SingularField("potential and electric field are singular at an active point source")
        if distance <= 8 * ulp(max(radius, self.distance_m)):
            raise UnsupportedProblem("unresolved near-source position")
        t = self.distance_m / radius
        h = self.gap_m * (1 + t)
        # z is the image separation scaled by d/a. When the probe is close
        # to the source, form z from their directly subtracted coordinates.
        if distance < radius / 2 and 0.5 < t < 2:
            z = tuple(t * v + h * u for v, u in zip(direct, self.direction))
        else:
            z = tuple(t * x - radius * u for x, u in zip(radial, self.direction))
        image_distance = _distance(z)
        if image_distance == 0:
            raise UnsupportedProblem("unresolved Kelvin image separation")
        return _KelvinPair(self, direct, distance, z, image_distance, t, h)


@dataclass(frozen=True, slots=True)
class _KelvinPair:
    site: _KelvinSite
    direct: Vector3
    distance: float
    scaled_image: Vector3
    image_distance: float
    t: float
    h: float

    def reciprocal_difference(self, r: float, gap: float, radius: float) -> float:
        """The factor 1 - A/B, with A real distance and B scaled image distance."""
        a, b = self.distance, self.image_distance
        if a < b / 2:
            return 1 - a / b
        # B² - A² = (r² - radius²)(d² - radius²) / radius².
        return ((gap / b) * (self.site.gap_m / b) * (1 + r / radius)
                * (1 + self.t) / (1 + a / b))

    def field(self, k: float, r: float, gap: float, radius: float) -> Vector3:
        a, b, q = self.distance, self.image_distance, self.site.charge_c
        real_scale = (k * (q / a)) / a
        beta = a / b
        if beta >= 0.5 and 0.5 < self.t < 2:
            # z = t*direct + h*u. Factor 1 - t²*(A/B)³ to preserve the
            # field of a near-wall source/image pair away from the source.
            direct_factor = fsum((
                self.reciprocal_difference(r, gap, radius) * (1 + beta + beta**2),
                -(self.h / radius) * beta**3,
            ))
            image_factor = self.t * (self.h / b) * beta**2
            return tuple(real_scale * fsum((direct_factor * (v / a), -image_factor * u))
                         for v, u in zip(self.direct, self.site.direction))
        image_scale = self.t * ((k * (q / b)) / b)
        return tuple(_sum_finite((real_scale * (v / a), -image_scale * (z / b)))
                     for v, z in zip(self.direct, self.scaled_image))


@dataclass(frozen=True, slots=True)
class _RegionKernel:
    center_m: Vector3
    radius_m: float
    sites: tuple[_KelvinSite, ...]

    def sample(self, point: Vector3) -> tuple[Vector3, float, float, tuple[_KelvinPair, ...]]:
        radial = _difference(point, self.center_m)
        r = _distance(radial)
        gap = _radial_gap(point, self.center_m, r, self.radius_m)
        pairs = tuple(site.sample(point, radial, self.radius_m) for site in self.sites)
        return radial, r, gap, pairs


@dataclass(frozen=True, slots=True)
class SphericalConductorSolution:
    """Metal state, free charge by face, source provenance, and spatial observables."""

    problem: SphericalConductorProblem
    potential_v: float
    net_charge_c: float
    outer_surface_charge_c: float
    inner_surface_charge_c: Mapping[str, float]
    source_regions: Mapping[str, SphericalRegion]
    _inner_total_c: float = field(repr=False)
    _k: float = field(repr=False)
    _kernels: Mapping[SphericalRegion, _RegionKernel] = field(repr=False)

    @property
    def conductor_results(self) -> tuple[ConductorResult, ...]:
        """FieldSolution summary, with the aggregate free charge on inner faces."""
        return (ConductorResult(self.problem.conductor_id, self.potential_v, self.net_charge_c,
                                self._inner_total_c, self.outer_surface_charge_c),)

    def region_at(self, point_m: Vector3) -> SphericalRegion:
        """Classify a world point. Exact sphere membership identifies a wall."""
        return _region_at(self.problem, _position(point_m, "sample point"))

    def potential_at(self, point_m: Vector3) -> float:
        point = _position(point_m, "sample point")
        region = _region_at(self.problem, point)
        kernel = self._kernels.get(region)
        if kernel is None:
            return self.potential_v
        _, r, gap, pairs = kernel.sample(point)
        radius = kernel.radius_m
        background = self.potential_v * (radius / r) if region == _EXTERIOR else self.potential_v
        return _sum_finite((
            background,
            *((self._k * (pair.site.charge_c / pair.distance))
              * pair.reciprocal_difference(r, gap, radius) for pair in pairs),
        ))

    def electric_field_at(self, point_m: Vector3) -> Vector3:
        point = _position(point_m, "sample point")
        region = _region_at(self.problem, point)
        if region.kind in ("outer_wall", "cavity_wall"):
            raise SingularField("electric field has distinct one-sided limits at a conductor wall")
        kernel = self._kernels.get(region)
        if kernel is None:
            return ORIGIN
        radial, r, gap, pairs = kernel.sample(point)
        radius = kernel.radius_m
        background = ORIGIN
        if region == _EXTERIOR:
            factor = (self.potential_v * (radius / r)) / r
            background = tuple(factor * (x / r) for x in radial)
        return tuple(_sum_finite(axis) for axis in zip(
            background, *(pair.field(self._k, r, gap, radius) for pair in pairs),
        ))


def _solve_state(
    constraint: ElectricalConstraint, radius: Fraction, inverse_k: Fraction,
    cavity_charge: Fraction, exterior_sum: Fraction,
) -> tuple[Fraction, Fraction, Fraction]:
    """Return (metal potential, net metal charge, outer-face charge)."""
    if type(constraint) is FixedNetCharge:
        metal_charge = Fraction(constraint.charge_c)
        outer_charge = metal_charge + cavity_charge
        voltage = (outer_charge / radius + exterior_sum) / inverse_k
        return voltage, metal_charge, outer_charge

    # The public constraint check currently admits only zero Dirichlet data.
    voltage = Fraction(constraint.potential_v) if type(constraint) is FixedPotential else Fraction()
    outer_charge = radius * (voltage * inverse_k - exterior_sum)
    return voltage, outer_charge - cavity_charge, outer_charge


def solve_spherical_conductor(problem: SphericalConductorProblem) -> SphericalConductorSolution:
    """Solve one connected spherical conductor with grounded or fixed-charge metal."""
    if type(problem) is not SphericalConductorProblem:
        raise UnsupportedProblem("expected one SphericalConductorProblem")
    constraint = problem.constraint
    if type(constraint) is FixedNetCharge:
        _finite_si(constraint.charge_c, "net metal charge")
    elif not (type(constraint) is Grounded
              or (type(constraint) is FixedPotential and constraint.potential_v == 0)):
        raise UnsupportedProblem("requires Grounded() or FixedNetCharge(); nonzero FixedPotential is unsupported")
    _validate_geometry(problem)

    regions = {}
    geometry = {_EXTERIOR: (problem.center_m, problem.outer_radius_m)}
    geometry.update({
        SphericalRegion("cavity", c.cavity_id): (c.center_m, c.radius_m)
        for c in problem.cavities
    })
    # First qualify each authored component, including any zero or canceled charge.
    for source in problem.sources:
        region = _region_at(problem, source.position_m)
        if region not in geometry:
            raise UnsupportedProblem(f"source {source.source_id!r} is on a wall or inside metal")
        regions[source.source_id] = region

    grouped: dict[SphericalRegion, dict[Vector3, Fraction]] = {region: {} for region in geometry}
    for source in problem.sources:
        sites = grouped[regions[source.source_id]]
        position = source.position_m
        sites[position] = sites.get(position, Fraction()) + Fraction(source.charge_c)

    kernels = {}
    exterior_sum = Fraction()
    inner = {}
    for region, (center, radius) in geometry.items():
        sites = grouped[region]
        if region.cavity_id is not None:
            inner[region.cavity_id] = -sum(sites.values(), Fraction())
        active = []
        for position, charge in sites.items():
            if not charge:
                continue
            radial = _difference(position, center)
            d = _distance(radial)
            if region == _EXTERIOR:
                exterior_sum += charge / Fraction(d)
            gap = _radial_gap(position, center, d, radius)
            direction = tuple(x / d for x in radial) if d else (1.0, 0.0, 0.0)
            active.append(_KelvinSite(position, _representable(charge, "site charge"), d, gap, direction))
        kernels[region] = _RegionKernel(center, radius, tuple(active))

    inner_total = sum(inner.values(), Fraction())
    inverse_k = Fraction(4 * pi) * Fraction(problem.permittivity_f_per_m)
    voltage, net_charge, outer_charge = _solve_state(
        constraint, Fraction(problem.outer_radius_m), inverse_k, -inner_total, exterior_sum,
    )
    return SphericalConductorSolution(
        problem,
        _representable(voltage, "metal potential"),
        _representable(net_charge, "net metal charge"),
        _representable(outer_charge, "outer-face charge"),
        MappingProxyType({name: _representable(q, "inner-face charge") for name, q in inner.items()}),
        MappingProxyType(regions),
        _representable(inner_total, "aggregate inner-face charge"),
        _representable(1 / inverse_k, "Coulomb constant"),
        MappingProxyType(kernels),
    )
