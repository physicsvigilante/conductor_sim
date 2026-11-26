"""Concentric spherical conductor stacks with homogeneous or layered linear dielectrics, SI."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass, field
from fractions import Fraction
from math import pi
from types import MappingProxyType
from typing import Literal

from .dielectric_layers import (
    DielectricInterfaceCharge,
    DielectricLayer,
    _validated_exterior_layers,
    _validated_gap_layers,
)
from .model import EPSILON_0, ORIGIN, Vector3, _position
from .solver import ConductorResult, SingularField, UnsupportedProblem
from .spherical_conductor import (
    _KelvinSite, _RegionKernel, _difference, _distance, _finite_si, _identity,
    _radial_gap, _representable, _sum_finite,
)
from .stack_coupling import ConductorGroup, _Coupling, _Gap, _State, _solve_coupling

__all__ = [
    "ConcentricConductor", "StackPointCharge", "SphericalStackProblem",
    "SphericalStackRegion", "SphericalStackSolution", "solve_spherical_stack",
]

_ZERO = Fraction()


def _world_point(value: Vector3, label: str) -> Vector3:
    point = _position(value, label)
    for coordinate in value:
        _finite_si(coordinate, label)
    return point


@dataclass(frozen=True, slots=True)
class ConcentricConductor:
    """A shell, including coincident faces for a thin shell or radius 0 for a solid core."""

    conductor_id: str
    inner_radius_m: float
    outer_radius_m: float

    def __post_init__(self) -> None:
        _identity(self.conductor_id, "conductor ID")
        _finite_si(self.inner_radius_m, "inner radius")
        _finite_si(self.outer_radius_m, "outer radius", positive=True)
        if not 0 <= self.inner_radius_m <= self.outer_radius_m:
            raise ValueError("expected 0 <= inner radius <= outer radius")


@dataclass(frozen=True, slots=True)
class StackPointCharge:
    """One held point charge with a stable ID and world-frame position."""

    source_id: str
    charge_c: float
    position_m: Vector3

    def __post_init__(self) -> None:
        _identity(self.source_id, "source ID")
        _finite_si(self.charge_c, "source charge")
        object.__setattr__(self, "position_m", _world_point(self.position_m, "source position"))


@dataclass(frozen=True, slots=True)
class SphericalStackProblem:
    """Conductors in increasing radial order, with voltage referenced to infinity.

    Each conductor belongs to exactly one group. The group can be a singleton
    or an ideal wire joining any subset of conductors. Held charges are allowed
    in the cavity, homogeneous gaps, and an uncoated exterior.
    Optionally supply gap_layers (one ordered tuple of DielectricLayer per gap,
    relative to vacuum EPSILON_0) and/or exterior_layers (finite concentric
    dielectric coatings outside the outermost conductor, surrounded by
    permittivity_f_per_m to infinity).
    """

    conductors: tuple[ConcentricConductor, ...]
    groups: tuple[ConductorGroup, ...]
    _: KW_ONLY
    center_m: Vector3 = ORIGIN
    sources: tuple[StackPointCharge, ...] = ()
    permittivity_f_per_m: float = EPSILON_0
    gap_layers: tuple[tuple[DielectricLayer, ...], ...] | None = None
    exterior_layers: tuple[DielectricLayer, ...] = ()

    def __post_init__(self) -> None:
        _finite_si(self.permittivity_f_per_m, "permittivity", positive=True)
        object.__setattr__(self, "center_m", _world_point(self.center_m, "stack center"))
        object.__setattr__(self, "conductors", tuple(self.conductors))
        object.__setattr__(self, "groups", tuple(self.groups))
        object.__setattr__(self, "sources", tuple(self.sources))
        if not self.conductors or any(type(c) is not ConcentricConductor for c in self.conductors):
            raise UnsupportedProblem("a spherical stack requires one or more concentric conductors")
        if len(set(self.conductor_ids)) != len(self.conductors):
            raise ValueError("conductor IDs must be unique")
        for left, right in zip(self.conductors, self.conductors[1:]):
            if left.outer_radius_m >= right.inner_radius_m:
                raise UnsupportedProblem("concentric conductors must be ordered and strictly separated")
        if any(type(s) is not StackPointCharge for s in self.sources):
            raise UnsupportedProblem("spherical stack sources must be held StackPointCharge objects")
        if len({s.source_id for s in self.sources}) != len(self.sources):
            raise ValueError("source IDs must be unique")
        if (self.gap_layers is not None or self.exterior_layers) and self.permittivity_f_per_m < EPSILON_0:
            raise UnsupportedProblem("background permittivity must be >= EPSILON_0 when dielectric layers are specified")
        if self.gap_layers is not None:
            if not isinstance(self.gap_layers, (tuple, list)):
                raise UnsupportedProblem("gap_layers must be a sequence of per-gap DielectricLayer tuples")
            normalized = tuple(
                tuple(row) if isinstance(row, (tuple, list)) else row for row in self.gap_layers
            )
            spans = tuple(
                (Fraction(left.outer_radius_m), Fraction(right.inner_radius_m))
                for left, right in zip(self.conductors, self.conductors[1:])
            )
            _validated_gap_layers(normalized, spans, "spherical stack")
            object.__setattr__(self, "gap_layers", normalized)
        if not isinstance(self.exterior_layers, (tuple, list)):
            raise UnsupportedProblem("exterior_layers must be a sequence of DielectricLayer entries")
        ext_layers = tuple(self.exterior_layers)
        if ext_layers:
            _validated_exterior_layers(ext_layers, Fraction(self.conductors[-1].outer_radius_m))
        object.__setattr__(self, "exterior_layers", ext_layers)

    @property
    def conductor_ids(self) -> tuple[str, ...]:
        return tuple(c.conductor_id for c in self.conductors)


@dataclass(frozen=True, slots=True)
class SphericalStackRegion:
    """Point classification; a gap index counts outward from zero."""

    kind: Literal["cavity", "annulus", "exterior", "metal", "inner_wall", "outer_wall", "thin_wall"]
    conductor_id: str | None = None
    gap_index: int | None = None


def _sample_region(problem: SphericalStackProblem, point: Vector3) -> tuple[SphericalStackRegion, int, Vector3, float]:
    radial = _difference(point, problem.center_m)
    radius = _distance(radial)
    for i, conductor in enumerate(problem.conductors):
        inner, outer = conductor.inner_radius_m, conductor.outer_radius_m
        name = conductor.conductor_id
        if inner:
            gap = _radial_gap(point, problem.center_m, radius, inner)
            if gap == 0:
                kind = "thin_wall" if inner == outer else "inner_wall"
                return SphericalStackRegion(kind, name), i, radial, radius
            if gap < 0:
                if i == 0:
                    return SphericalStackRegion("cavity", name), -1, radial, radius
                return SphericalStackRegion("annulus", gap_index=i - 1), i - 1, radial, radius
        gap = _radial_gap(point, problem.center_m, radius, outer)
        if gap == 0:
            return SphericalStackRegion("outer_wall", name), i, radial, radius
        if gap < 0:
            return SphericalStackRegion("metal", name), i, radial, radius
    return SphericalStackRegion("exterior"), len(problem.conductors) - 1, radial, radius


def _inner_weight(point: Vector3, center: Vector3, r: float, inner: float, outer: float) -> Fraction:
    # a(b-r) / [a(b-r) + b(r-a)] equals the reciprocal-radius weight.
    # Use signed geometric gaps so a rounded r can still resolve either wall.
    left = -Fraction(_radial_gap(point, center, r, outer)) * Fraction(inner)
    right = Fraction(_radial_gap(point, center, r, inner)) * Fraction(outer)
    return left / (left + right)


def _kelvin_kernel(center: Vector3, radius: float, sources: Mapping[Vector3, Fraction]) -> _RegionKernel:
    sites = []
    for position, charge in sources.items():
        radial = _difference(position, center)
        distance = _distance(radial)
        gap = _radial_gap(position, center, distance, radius)
        direction = tuple(x / distance for x in radial) if distance else (1., 0., 0.)
        sites.append(_KelvinSite(position, _representable(charge, "site charge"), distance, gap, direction))
    return _RegionKernel(center, radius, tuple(sites))


@dataclass(frozen=True, slots=True)
class SphericalStackSolution:
    """Conductor state, complete face totals, dielectric interfaces, and supported local observables."""

    problem: SphericalStackProblem
    conductor_results: tuple[ConductorResult, ...]
    source_regions: Mapping[str, SphericalStackRegion]
    dielectric_interfaces: tuple[DielectricInterfaceCharge, ...]
    _state: _State = field(repr=False)
    _inverse_k: Fraction = field(repr=False)
    _inverse_k0: Fraction = field(repr=False)
    _annulus_sites: tuple[frozenset[Vector3], ...] = field(repr=False)
    _gap_sublayers: tuple[tuple[tuple[Fraction, Fraction, Fraction], ...], ...] | None = field(repr=False)
    _exterior_sublayers: tuple[tuple[Fraction, Fraction, Fraction], ...] = field(repr=False)
    _cavity: _RegionKernel | None = field(repr=False)
    _exterior: _RegionKernel = field(repr=False)

    @property
    def capacitance_matrix_f(self) -> tuple[tuple[float, ...], ...]:
        """Maxwell matrix in problem order, before applying wire or supply constraints."""
        return self._state.coupling.capacitance_matrix_f

    @property
    def has_active_sources(self) -> bool:
        """Whether any held charge remains after coalescing coincident sources."""
        return bool(any(self._annulus_sites) or self._exterior.sites
                    or (self._cavity is not None and self._cavity.sites))

    @property
    def field_energy_j(self) -> float | None:
        """Total field energy for source-free states; None for active point charges."""
        if self.has_active_sources:
            return None
        return _representable(self._state.energy, "field energy")

    def region_at(self, point_m: Vector3) -> SphericalStackRegion:
        return _sample_region(self.problem, _world_point(point_m, "sample point"))[0]

    def _require_source_free_annulus(self, i: int, point: Vector3) -> None:
        sites = self._annulus_sites[i]
        if point in sites:
            raise SingularField("potential and electric field are singular at an active point source")
        if sites:
            raise UnsupportedProblem("local observations in a charged annulus require its Green function")

    def _layered_gap_potential(self, i: int, point: Vector3, r: float) -> float:
        assert self._gap_sublayers is not None
        layers = self._gap_sublayers[i]
        q_enc = self._state.right_faces[i]
        v_outer = self._state.voltages[i + 1]
        r_frac = Fraction(r)
        # Find layer containing r using exact signed gap at each interface.
        for idx, (lo, hi, kappa) in enumerate(layers):
            g_hi = _radial_gap(point, self.problem.center_m, r, float(hi))
            if g_hi <= 0:
                if g_hi == 0:
                    elastance_out = sum(
                        ((1 / l_lo - 1 / l_hi) / (self._inverse_k0 * l_k) for l_lo, l_hi, l_k in layers[idx + 1:]),
                        _ZERO,
                    )
                else:
                    elastance_out = (1 / r_frac - 1 / hi) / (self._inverse_k0 * kappa) + sum(
                        ((1 / l_lo - 1 / l_hi) / (self._inverse_k0 * l_k) for l_lo, l_hi, l_k in layers[idx + 1:]),
                        _ZERO,
                    )
                return _representable(v_outer + q_enc * elastance_out, "layered gap potential")
        return _representable(v_outer, "layered gap potential")

    def _layered_gap_field(self, i: int, point: Vector3, radial: Vector3, r: float) -> Vector3:
        assert self._gap_sublayers is not None
        layers = self._gap_sublayers[i]
        q_enc = self._state.right_faces[i]
        r_frac = Fraction(r)
        for idx, (lo, hi, kappa) in enumerate(layers):
            g_hi = _radial_gap(point, self.problem.center_m, r, float(hi))
            if g_hi < 0:
                factor = _representable(q_enc / (self._inverse_k0 * kappa * r_frac**2), "layered radial field")
                return tuple(factor * (x / r) for x in radial)
            if g_hi == 0 and idx + 1 < len(layers):
                next_kappa = layers[idx + 1][2]
                if kappa != next_kappa:
                    raise SingularField("electric field has a jump discontinuity at a dielectric interface")
                factor = _representable(q_enc / (self._inverse_k0 * kappa * r_frac**2), "layered radial field")
                return tuple(factor * (x / r) for x in radial)
        kappa = layers[-1][2]
        factor = _representable(q_enc / (self._inverse_k0 * kappa * r_frac**2), "layered radial field")
        return tuple(factor * (x / r) for x in radial)

    def _coated_exterior_potential(self, point: Vector3, r: float) -> float:
        layers = self._exterior_sublayers
        q_enc = self._state.right_faces[-1]
        r_frac = Fraction(r)
        r_coat_out = layers[-1][1]
        tail_elastance = 1 / (self._inverse_k * r_coat_out)
        for idx, (lo, hi, kappa) in enumerate(layers):
            g_hi = _radial_gap(point, self.problem.center_m, r, float(hi))
            if g_hi <= 0:
                if g_hi == 0:
                    elastance = tail_elastance + sum(
                        ((1 / l_lo - 1 / l_hi) / (self._inverse_k0 * l_k) for l_lo, l_hi, l_k in layers[idx + 1:]),
                        _ZERO,
                    )
                else:
                    elastance = (1 / r_frac - 1 / hi) / (self._inverse_k0 * kappa) + tail_elastance + sum(
                        ((1 / l_lo - 1 / l_hi) / (self._inverse_k0 * l_k) for l_lo, l_hi, l_k in layers[idx + 1:]),
                        _ZERO,
                    )
                return _representable(q_enc * elastance, "coated exterior potential")
        return _representable(q_enc / (self._inverse_k * r_frac), "coated exterior potential")

    def _coated_exterior_field(self, point: Vector3, radial: Vector3, r: float) -> Vector3:
        layers = self._exterior_sublayers
        q_enc = self._state.right_faces[-1]
        r_frac = Fraction(r)
        kappa_ext = self._inverse_k / self._inverse_k0
        for idx, (lo, hi, kappa) in enumerate(layers):
            g_hi = _radial_gap(point, self.problem.center_m, r, float(hi))
            if g_hi < 0:
                factor = _representable(q_enc / (self._inverse_k0 * kappa * r_frac**2), "coated exterior field")
                return tuple(factor * (x / r) for x in radial)
            if g_hi == 0:
                next_kappa = layers[idx + 1][2] if idx + 1 < len(layers) else kappa_ext
                if kappa != next_kappa:
                    raise SingularField("electric field has a jump discontinuity at a dielectric interface")
                factor = _representable(q_enc / (self._inverse_k0 * kappa * r_frac**2), "coated exterior field")
                return tuple(factor * (x / r) for x in radial)
        factor = _representable(q_enc / (self._inverse_k * r_frac**2), "coated exterior field")
        return tuple(factor * (x / r) for x in radial)

    def potential_at(self, point_m: Vector3) -> float:
        point = _world_point(point_m, "sample point")
        region, i, _, r = _sample_region(self.problem, point)
        voltages = self._state.voltages
        if region.kind == "annulus":
            self._require_source_free_annulus(i, point)
            if self._gap_sublayers is not None:
                return self._layered_gap_potential(i, point, r)
            inner = self.problem.conductors[i].outer_radius_m
            outer = self.problem.conductors[i + 1].inner_radius_m
            weight = _inner_weight(point, self.problem.center_m, r, inner, outer)
            drop = (voltages[i] - voltages[i + 1]) * weight
            return _representable(voltages[i + 1] + drop, "gap potential")
        if region.kind not in ("cavity", "exterior"):
            return self.conductor_results[i].potential_v

        if region.kind == "exterior" and self._exterior_sublayers:
            return self._coated_exterior_potential(point, r)

        cavity = region.kind == "cavity"
        kernel = self._cavity if cavity else self._exterior
        assert kernel is not None
        _, r, gap, pairs = kernel.sample(point)
        radius = kernel.radius_m
        voltage = self.conductor_results[0 if cavity else -1].potential_v
        background = voltage if cavity else voltage * (radius / r)
        k = _representable(1 / self._inverse_k, "Coulomb constant")
        return _sum_finite((background, *((k * (pair.site.charge_c / pair.distance))
                                         * pair.reciprocal_difference(r, gap, radius) for pair in pairs)))

    def electric_field_at(self, point_m: Vector3) -> Vector3:
        point = _world_point(point_m, "sample point")
        region, i, radial, r = _sample_region(self.problem, point)
        if region.kind.endswith("wall"):
            raise SingularField("electric field requires a side at a conductor wall")
        if region.kind == "metal":
            return ORIGIN
        if region.kind == "annulus":
            self._require_source_free_annulus(i, point)
            if self._gap_sublayers is not None:
                return self._layered_gap_field(i, point, radial, r)
            factor = _representable(self._state.right_faces[i] / (self._inverse_k * Fraction(r)**2),
                                    "radial gap field")
            return tuple(factor * (x / r) for x in radial)

        if region.kind == "exterior" and self._exterior_sublayers:
            return self._coated_exterior_field(point, radial, r)

        cavity = region.kind == "cavity"
        kernel = self._cavity if cavity else self._exterior
        assert kernel is not None
        radial, r, gap, pairs = kernel.sample(point)
        radius = kernel.radius_m
        background = ORIGIN
        if not cavity:
            factor = (self.conductor_results[-1].potential_v * (radius / r)) / r
            background = tuple(factor * (x / r) for x in radial)
        k = _representable(1 / self._inverse_k, "Coulomb constant")
        return tuple(_sum_finite(axis) for axis in zip(
            background, *(pair.field(k, r, gap, radius) for pair in pairs),
        ))


def _solve_spherical_stack(
    problem: SphericalStackProblem, initial_charges: tuple[Fraction, ...] | None = None,
) -> SphericalStackSolution:
    conductors = problem.conductors
    center = problem.center_m
    four_pi = Fraction(4 * pi)
    inverse_k = four_pi * Fraction(problem.permittivity_f_per_m)
    inverse_k0 = four_pi * Fraction(EPSILON_0)

    buckets: list[dict[Vector3, Fraction]] = [{} for _ in range(len(conductors) + 1)]
    regions = {}
    for source in problem.sources:
        region, i, _, _ = _sample_region(problem, source.position_m)
        if region.kind not in ("cavity", "annulus", "exterior"):
            raise UnsupportedProblem(f"source {source.source_id!r} is on a wall or inside metal")
        regions[source.source_id] = region
        bucket = buckets[i + 1]
        point = source.position_m
        bucket[point] = bucket.get(point, _ZERO) + Fraction(source.charge_c)
    cavity_sites, *annulus_sites, exterior_sites = (
        {point: q for point, q in bucket.items() if q} for bucket in buckets
    )

    spans = tuple(
        (Fraction(left.outer_radius_m), Fraction(right.inner_radius_m))
        for left, right in zip(conductors, conductors[1:])
    )
    validated_gaps = (
        _validated_gap_layers(problem.gap_layers, spans, "spherical stack")
        if problem.gap_layers is not None else None
    )
    outer = Fraction(conductors[-1].outer_radius_m)
    validated_ext = _validated_exterior_layers(problem.exterior_layers, outer)

    if validated_ext and exterior_sites:
        raise UnsupportedProblem("active point charges outside a dielectric-coated spherical stack are unsupported")

    gaps = []
    for g, (left, right, sites) in enumerate(zip(conductors, conductors[1:], annulus_sites)):
        a, b = Fraction(left.outer_radius_m), Fraction(right.inner_radius_m)
        if validated_gaps is None:
            elastance = (1 / a - 1 / b) / inverse_k
        else:
            layers = validated_gaps[g]
            if sites and len({kappa for _, _, kappa in layers}) > 1:
                raise UnsupportedProblem("active point charges inside a multi-layer dielectric gap are unsupported")
            elastance = sum(((1 / lo - 1 / hi) / (inverse_k0 * kappa) for lo, hi, kappa in layers), _ZERO)
        induced = _ZERO
        for point, charge in sites.items():
            r = _distance(_difference(point, center))
            weight = _inner_weight(point, center, r, left.outer_radius_m, right.inner_radius_m)
            induced -= charge * weight
        gaps.append(_Gap(elastance, induced, -sum(sites.values(), _ZERO) - induced))

    if validated_ext:
        coat_outer = validated_ext[-1][1]
        ext_elastance = sum(
            ((1 / lo - 1 / hi) / (inverse_k0 * kappa) for lo, hi, kappa in validated_ext),
            _ZERO,
        ) + 1 / (coat_outer * inverse_k)
        exterior_offset = _ZERO
    else:
        ext_elastance = 1 / (outer * inverse_k)
        exterior_offset = -sum(
            (q * outer / Fraction(_distance(_difference(p, center))) for p, q in exterior_sites.items()),
            _ZERO,
        )

    coupling = _Coupling(tuple(gaps), _Gap(ext_elastance, exterior_offset),
                         -sum(cavity_sites.values(), _ZERO))
    state = _solve_coupling(coupling, problem.conductor_ids, problem.groups, initial_charges=initial_charges)
    results = tuple(ConductorResult(name, _representable(v, "conductor potential"),
                                    _representable(q, "conductor charge"),
                                    _representable(a, "inner-face charge"),
                                    _representable(b, "outer-face charge"))
                    for name, v, q, a, b in zip(problem.conductor_ids, state.voltages, state.charges,
                                                state.left_faces, state.right_faces))

    # Bound surface charge ledger on all source-free layered dielectric interfaces.
    interfaces: list[DielectricInterfaceCharge] = []
    if validated_gaps is not None:
        for g, (layers, sites) in enumerate(zip(validated_gaps, annulus_sites)):
            if sites:
                continue  # Off-center point charge makes angular bound density non-uniform.
            q_enc = state.right_faces[g]
            r0, _, k0 = layers[0]
            q_b0 = -(1 - 1 / k0) * q_enc
            interfaces.append(DielectricInterfaceCharge(
                "gap", g, float(r0), "conductor_outer", None, float(k0),
                _representable(q_b0 / (four_pi * r0**2), "bound surface density"),
                _representable(q_b0, "bound face charge"),
            ))
            for idx in range(len(layers) - 1):
                _, r_if, k1 = layers[idx]
                k2 = layers[idx + 1][2]
                q_bif = (1 / k2 - 1 / k1) * q_enc
                interfaces.append(DielectricInterfaceCharge(
                    "gap", g, float(r_if), "dielectric_interface", float(k1), float(k2),
                    _representable(q_bif / (four_pi * r_if**2), "bound surface density"),
                    _representable(q_bif, "bound interface charge"),
                ))
            _, r_end, kl = layers[-1]
            q_bl = (1 - 1 / kl) * q_enc
            interfaces.append(DielectricInterfaceCharge(
                "gap", g, float(r_end), "conductor_inner", float(kl), None,
                _representable(q_bl / (four_pi * r_end**2), "bound surface density"),
                _representable(q_bl, "bound face charge"),
            ))
    if validated_ext:
        q_enc = state.right_faces[-1]
        kappa_ext = inverse_k / inverse_k0
        r0, _, k0 = validated_ext[0]
        q_b0 = -(1 - 1 / k0) * q_enc
        interfaces.append(DielectricInterfaceCharge(
            "exterior", None, float(r0), "conductor_outer", None, float(k0),
            _representable(q_b0 / (four_pi * r0**2), "bound surface density"),
            _representable(q_b0, "bound face charge"),
        ))
        for idx in range(len(validated_ext) - 1):
            _, r_if, k1 = validated_ext[idx]
            k2 = validated_ext[idx + 1][2]
            q_bif = (1 / k2 - 1 / k1) * q_enc
            interfaces.append(DielectricInterfaceCharge(
                "exterior", None, float(r_if), "dielectric_interface", float(k1), float(k2),
                _representable(q_bif / (four_pi * r_if**2), "bound surface density"),
                _representable(q_bif, "bound interface charge"),
            ))
        _, r_end, kl = validated_ext[-1]
        q_bend = (1 / kappa_ext - 1 / kl) * q_enc
        interfaces.append(DielectricInterfaceCharge(
            "exterior", None, float(r_end), "dielectric_interface", float(kl), float(kappa_ext),
            _representable(q_bend / (four_pi * r_end**2), "bound surface density"),
            _representable(q_bend, "bound interface charge"),
        ))

    inner = conductors[0].inner_radius_m
    return SphericalStackSolution(
        problem, results, MappingProxyType(regions), tuple(interfaces), state, inverse_k, inverse_k0,
        tuple(frozenset(sites) for sites in annulus_sites),
        validated_gaps, validated_ext,
        _kelvin_kernel(center, float(inner), cavity_sites) if inner else None,
        _kelvin_kernel(center, float(outer), exterior_sites),
    )


def solve_spherical_stack(problem: SphericalStackProblem) -> SphericalStackSolution:
    """Solve concentric spherical geometry with homogeneous or layered linear dielectrics."""
    if type(problem) is not SphericalStackProblem:
        raise UnsupportedProblem("expected a SphericalStackProblem")
    return _solve_spherical_stack(problem)
