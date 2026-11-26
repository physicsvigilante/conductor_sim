"""Static SI solution for one connected spherical conductor and its sealed cavities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass, field
from fractions import Fraction
from itertools import combinations
from math import copysign, fsum, hypot, isfinite, pi, ulp
from types import MappingProxyType
from typing import Literal

from ._spherical_precision import _RootSum, _round_vector
from .model import (
    EPSILON_0, ORIGIN, ElectricalConstraint, FixedNetCharge, FixedPotential,
    Grounded, Vector3, _position,
)
from .solver import ConductorResult, SingularField, UnsupportedProblem

__all__ = [
    "SphericalCavity", "SphericalPointSource", "SphericalConductorProblem",
    "SphericalRegion", "SphericalConductorSolution", "solve_spherical_conductor",
    "UniformBackgroundField",
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


_ExactVector = tuple[Fraction, Fraction, Fraction]


def _exact_difference(a: Vector3, b: Vector3) -> _ExactVector:
    return tuple(Fraction(x) - Fraction(y) for x, y in zip(a, b))


def _norm_sq(vector: _ExactVector) -> Fraction:
    return sum((x * x for x in vector), Fraction())


def _squared_distance(a: Vector3, b: Vector3) -> Fraction:
    return _norm_sq(_exact_difference(a, b))


def _checked_position(value: Vector3, label: str) -> Vector3:
    try:
        point = _position(value, label)
        for coord in value:
            _finite_si(coord, label)
        return point
    except (ValueError, TypeError, OverflowError) as exc:
        raise UnsupportedProblem(str(exc)) from exc


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
class UniformBackgroundField:
    """V_bg(x) = -E0 dot (x - reference_point_m); None uses the sphere center."""

    electric_field_v_per_m: Vector3
    reference_point_m: Vector3 | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "electric_field_v_per_m", _checked_position(
            self.electric_field_v_per_m, "background electric field"))
        if self.reference_point_m is not None:
            object.__setattr__(self, "reference_point_m", _checked_position(
                self.reference_point_m, "background reference point"))


@dataclass(frozen=True, slots=True)
class SphericalConductorProblem:
    """One outer sphere and any number of named, disjoint spherical cavities.

    A single homogeneous permittivity applies in the exterior and all cavities.
    Supply held sources in world coordinates; geometry determines their regions.
    With a UniformBackgroundField, use Grounded(), FixedNetCharge(coulombs),
    or FixedPotential(volts). Without one, the potential decays to zero at
    infinity and the legacy constraint rules apply.
    """

    outer_radius_m: float
    constraint: ElectricalConstraint
    _: KW_ONLY
    center_m: Vector3 = ORIGIN
    cavities: tuple[SphericalCavity, ...] = ()
    sources: tuple[SphericalPointSource, ...] = ()
    permittivity_f_per_m: float = EPSILON_0
    conductor_id: str = "sphere"
    background_field: UniformBackgroundField | None = None

    def __post_init__(self) -> None:
        if self.background_field is not None and type(self.background_field) is not UniformBackgroundField:
            raise UnsupportedProblem("background_field must be a UniformBackgroundField")
        try:
            self._validate_inputs()
        except (ValueError, TypeError, OverflowError) as exc:
            if self.background_field is None or isinstance(exc, UnsupportedProblem):
                raise
            raise UnsupportedProblem(str(exc)) from exc

    def _validate_inputs(self) -> None:
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
    if problem.background_field is not None:
        outer_gap = _squared_distance(point, problem.center_m) - Fraction(problem.outer_radius_m)**2
        if outer_gap > 0:
            return _EXTERIOR
        if outer_gap == 0:
            return SphericalRegion("outer_wall")
        for cavity in problem.cavities:
            gap = _squared_distance(point, cavity.center_m) - Fraction(cavity.radius_m)**2
            if gap <= 0:
                return SphericalRegion("cavity" if gap < 0 else "cavity_wall", cavity.cavity_id)
        return SphericalRegion("metal")

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
class _ExactKelvinSite:
    position_m: Vector3
    radial_m: _ExactVector
    distance_sq: Fraction
    charge_c: Fraction

    def pair(
        self, point: Vector3, radial: _ExactVector, radius: Fraction, *, legacy: bool,
    ) -> tuple[tuple[_ExactVector, Fraction, Fraction, Fraction], ...]:
        """Return (separation, separation squared, charge numerator, charge root) terms."""
        direct = _exact_difference(point, self.position_m)
        direct_sq = _norm_sq(direct)
        if not direct_sq:
            raise SingularField("potential and electric field are singular at an active point source")
        if legacy:
            scale = max(float(radius), _distance(tuple(float(x) for x in self.radial_m)))
            if direct_sq <= Fraction(8 * ulp(scale))**2:
                raise UnsupportedProblem("unresolved near-source position")
        real = (direct, direct_sq, self.charge_c, Fraction(1))
        if not self.distance_sq:
            # At the cavity center the image contributes a constant -k*q/R.
            return real, ((Fraction(),) * 3, radius**2, -self.charge_c, Fraction(1))
        image_scale = radius**2 / self.distance_sq
        image = tuple(x - image_scale * s for x, s in zip(radial, self.radial_m))
        return real, (image, _norm_sq(image), -self.charge_c * radius, self.distance_sq)


@dataclass(frozen=True, slots=True)
class _ExactRegionKernel:
    center_m: Vector3
    radius_m: Fraction
    sites: tuple[_ExactKelvinSite, ...]

    def pairs(self, point: Vector3, radial: _ExactVector, *, legacy: bool):
        for site in self.sites:
            yield from site.pair(point, radial, self.radius_m, legacy=legacy)


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
    _inverse_k: Fraction = field(repr=False)
    _kernels: Mapping[SphericalRegion, _ExactRegionKernel] = field(repr=False)
    _field: _ExactVector = field(repr=False)
    _background_center_v: Fraction = field(repr=False)
    _voltage: _RootSum = field(repr=False, compare=False)
    _relative_voltage: _RootSum = field(repr=False, compare=False)
    _outer_charge: _RootSum = field(repr=False, compare=False)

    @property
    def conductor_results(self) -> tuple[ConductorResult, ...]:
        """FieldSolution summary, with the aggregate free charge on inner faces."""
        return (ConductorResult(self.problem.conductor_id, self.potential_v, self.net_charge_c,
                                self._inner_total_c, self.outer_surface_charge_c),)

    def _sample_position(self, point_m: Vector3) -> Vector3:
        if self.problem.background_field is None:
            return _position(point_m, "sample point")
        return _checked_position(point_m, "sample point")

    def region_at(self, point_m: Vector3) -> SphericalRegion:
        """Classify a world point. Exact sphere membership identifies a wall."""
        return _region_at(self.problem, self._sample_position(point_m))

    def potential_at(self, point_m: Vector3) -> float:
        point = self._sample_position(point_m)
        region = _region_at(self.problem, point)
        kernel = self._kernels.get(region)
        if kernel is None:
            return self.potential_v
        radial = _exact_difference(point, kernel.center_m)
        total = _RootSum()
        if region == _EXTERIOR:
            r_sq = _norm_sq(radial)
            radius = kernel.radius_m
            projection = _dot(self._field, radial)
            total.add(self._background_center_v - projection)
            total.add(projection * radius**3 / r_sq, 1 / r_sq)
            total.add_scaled(self._relative_voltage, radius, 1 / r_sq)
        else:
            total.add_scaled(self._voltage, Fraction(1))
        for _, separation_sq, q, charge_root in kernel.pairs(
            point, radial, legacy=self.problem.background_field is None,
        ):
            total.add(q / self._inverse_k, 1 / (separation_sq * charge_root))
        return total.rounded("local potential")

    def electric_field_at(self, point_m: Vector3) -> Vector3:
        point = self._sample_position(point_m)
        region = _region_at(self.problem, point)
        if region.kind in ("outer_wall", "cavity_wall"):
            raise SingularField("electric field has distinct one-sided limits at a conductor wall")
        kernel = self._kernels.get(region)
        if kernel is None:
            return ORIGIN
        radial = _exact_difference(point, kernel.center_m)
        components = tuple(_RootSum() for _ in radial)
        if region == _EXTERIOR:
            r_sq = _norm_sq(radial)
            radius = kernel.radius_m
            projection = _dot(self._field, radial)
            for total, e, x in zip(components, self._field, radial):
                total.add(e)
                total.add(radius**3 * (3 * projection * x / r_sq - e) / r_sq, 1 / r_sq)
                total.add_scaled(self._relative_voltage, radius * x / r_sq, 1 / r_sq)
        for separation, separation_sq, q, charge_root in kernel.pairs(
            point, radial, legacy=self.problem.background_field is None,
        ):
            for total, delta in zip(components, separation):
                total.add(q * delta / (self._inverse_k * separation_sq),
                          1 / (separation_sq * charge_root))
        return _round_vector(components, "electric field")

    @property
    def background_induced_dipole_c_m(self) -> Vector3:
        """Dipole moment about the sphere center induced by the uniform field alone."""
        radius = self._kernels[_EXTERIOR].radius_m
        components = tuple(_RootSum(self._inverse_k * radius**3 * e) for e in self._field)
        return _round_vector(components, "induced dipole")

    def _outer_wall_density(self, point_m: Vector3) -> tuple[_RootSum, _ExactVector]:
        point = _checked_position(point_m, "outer wall point")
        kernel = self._kernels[_EXTERIOR]
        radius = kernel.radius_m
        radial = _exact_difference(point, kernel.center_m)
        if _norm_sq(radial) != radius**2:
            raise UnsupportedProblem("outer wall observables require an exact outer wall point")
        normal = tuple(x / radius for x in radial)
        epsilon = Fraction(self.problem.permittivity_f_per_m)
        total = _RootSum(3 * epsilon * _dot(self._field, normal))
        total.add_scaled(self._relative_voltage, epsilon / radius)
        for site in kernel.sites:
            # The grounded Kelvin density integrates to -q*R/d.
            separation_sq = _squared_distance(point, site.position_m)
            total.add(-site.charge_c * (site.distance_sq - radius**2)
                      / (Fraction(4 * pi) * radius * separation_sq), 1 / separation_sq)
        return total, normal

    def outer_surface_charge_density_at(self, point_m: Vector3) -> float:
        """Free charge density in C/m² at an exact outer wall point."""
        density, _ = self._outer_wall_density(point_m)
        return density.rounded("outer surface charge density")

    def outer_wall_electric_field_at(self, point_m: Vector3) -> Vector3:
        """One-sided electric field on the dielectric side of the outer wall."""
        density, normal = self._outer_wall_density(point_m)
        epsilon = Fraction(self.problem.permittivity_f_per_m)
        components = []
        for n in normal:
            component = _RootSum()
            component.add_scaled(density, n / epsilon)
            components.append(component)
        return _round_vector(tuple(components), "outer wall electric field")

    def outer_wall_pressure_at(self, point_m: Vector3) -> float:
        """Outward electrostatic traction magnitude sigma²/(2*eps), in pascals."""
        density, _ = self._outer_wall_density(point_m)
        epsilon = Fraction(self.problem.permittivity_f_per_m)
        return density.rounded("outer wall pressure", square_over=2 * epsilon)

    @property
    def outer_wall_force_n(self) -> Vector3:
        """Integrated stress on the outer face, with no active exterior point charges."""
        if self._kernels[_EXTERIOR].sites:
            raise UnsupportedProblem("outer wall force with active exterior point charges is unsupported")
        components = []
        for e in self._field:
            component = _RootSum()
            component.add_scaled(self._outer_charge, e)
            components.append(component)
        return _round_vector(tuple(components), "outer wall force")


def _dot(a: _ExactVector, b: _ExactVector) -> Fraction:
    return sum((x * y for x, y in zip(a, b)), Fraction())


def solve_spherical_conductor(problem: SphericalConductorProblem) -> SphericalConductorSolution:
    """Solve one connected spherical conductor with optional uniform background data."""
    if type(problem) is not SphericalConductorProblem:
        raise UnsupportedProblem("expected one SphericalConductorProblem")
    constraint = problem.constraint
    background = problem.background_field
    if type(constraint) is FixedNetCharge:
        try:
            _finite_si(constraint.charge_c, "net metal charge")
        except ValueError as exc:
            if background is None:
                raise
            raise UnsupportedProblem(str(exc)) from exc
    elif background is not None and type(constraint) is FixedPotential:
        try:
            _finite_si(constraint.potential_v, "fixed potential")
        except ValueError as exc:
            raise UnsupportedProblem(str(exc)) from exc
    elif not (type(constraint) is Grounded
              or (type(constraint) is FixedPotential and constraint.potential_v == 0)):
        if background is not None:
            raise UnsupportedProblem("requires Grounded(), FixedNetCharge(), or FixedPotential()")
        raise UnsupportedProblem("requires Grounded() or FixedNetCharge(); nonzero FixedPotential is unsupported")
    _validate_geometry(problem)

    radius = Fraction(problem.outer_radius_m)
    inverse_k = Fraction(4 * pi) * Fraction(problem.permittivity_f_per_m)
    if background is None:
        _representable(1 / inverse_k, "Coulomb constant")
    field_vector = tuple(Fraction(e) for e in (background.electric_field_v_per_m if background else ORIGIN))
    reference = background.reference_point_m if background else None
    background_center = (Fraction() if reference is None else
                         -_dot(field_vector, _exact_difference(problem.center_m, reference)))

    geometry = {_EXTERIOR: (problem.center_m, radius)}
    geometry.update({
        SphericalRegion("cavity", c.cavity_id): (c.center_m, Fraction(c.radius_m))
        for c in problem.cavities
    })
    # Qualify each authored source before coalescing coincident sites.
    regions = {}
    grouped: dict[SphericalRegion, dict[Vector3, Fraction]] = {region: {} for region in geometry}
    for source in problem.sources:
        region = _region_at(problem, source.position_m)
        if region not in geometry:
            raise UnsupportedProblem(f"source {source.source_id!r} is on a wall or inside metal")
        regions[source.source_id] = region
        sites = grouped[region]
        position = source.position_m
        sites[position] = sites.get(position, Fraction()) + Fraction(source.charge_c)

    kernels = {}
    exterior_sum = _RootSum()
    inner = {}
    for region, (center, region_radius) in geometry.items():
        sites = grouped[region]
        if region.cavity_id is not None:
            inner[region.cavity_id] = -sum(sites.values(), Fraction())
        active = []
        for position, charge in sites.items():
            if not charge:
                continue
            if background is None:
                _representable(charge, "site charge")
            radial = _exact_difference(position, center)
            distance_sq = _norm_sq(radial)
            if region == _EXTERIOR:
                exterior_sum.add(charge, 1 / distance_sq)
            active.append(_ExactKelvinSite(position, radial, distance_sq, charge))
        kernels[region] = _ExactRegionKernel(center, region_radius, tuple(active))

    inner_total = sum(inner.values(), Fraction())
    if type(constraint) is FixedNetCharge:
        metal_charge = Fraction(constraint.charge_c)
        outer = metal_charge - inner_total
        outer_charge = _RootSum(outer)
        net_charge = _RootSum(metal_charge)
        relative_voltage = _RootSum(outer / (radius * inverse_k))
        relative_voltage.add_scaled(exterior_sum, 1 / inverse_k)
        voltage = _RootSum(background_center)
        voltage.add_scaled(relative_voltage, Fraction(1))
    else:
        prescribed_v = Fraction(constraint.potential_v) if type(constraint) is FixedPotential else Fraction()
        voltage = _RootSum(prescribed_v)
        relative_v = prescribed_v - background_center
        relative_voltage = _RootSum(relative_v)
        outer_charge = _RootSum(radius * inverse_k * relative_v)
        outer_charge.add_scaled(exterior_sum, -radius)
        net_charge = _RootSum(inner_total)
        net_charge.add_scaled(outer_charge, Fraction(1))

    return SphericalConductorSolution(
        problem,
        voltage.rounded("metal potential"),
        net_charge.rounded("net metal charge"),
        outer_charge.rounded("outer-face charge"),
        MappingProxyType({name: _representable(q, "inner-face charge") for name, q in inner.items()}),
        MappingProxyType(regions),
        _representable(inner_total, "aggregate inner-face charge"),
        inverse_k, MappingProxyType(kernels), field_vector, background_center,
        voltage, relative_voltage, outer_charge,
    )
