"""M40a: local surface sigma on genuinely spherical, disjoint sealed cavities.

This is NOT the geometry-free M39a ledger acquiring a fictional field map.
Physical charges/identities/grounding remain authoritative in M39a; the
new, separately qualified geometry supplies only bounded local surface
density probes. No time, force, FEMM or arbitrary irregular 3D solver.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from math import fsum, hypot, isfinite, pi
import re

from .cavity_gauss import GaussCavityAssembly, GaussCavityLedger, solve_cavity_gauss
from .solver import UnsupportedProblem

SCHEMA = "conductor_sim.m40a_spherical_multicavity_sigma.v1"
ROLE = "teacher_hypothetical_spherical_multicavity_only"
ASSUMPTIONS = (
    "one connected ideal metal with spherical outer surface and disjoint sealed spherical voids",
    "all point charges have verified positions strictly within their assigned spherical voids",
    "homogeneous positive permittivity, no exterior sources/fields, no dielectric interfaces",
    "only isolated fixed-metal-charge or fully grounded static equilibria",
    "spherical inner/outer surface sigma samples ONLY, not a 3D field or force solver",
)
UNSUPPORTED = (
    "irregular/non-spherical cavities, local 3D field/potential maps",
    "source forces, motion, ground-wire currents or transient grounding",
    "source-faithful solutions of workbook p11 or notes p70 irregular 3D local-density items",
)
_ID = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}\Z")
_NORMALS = ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _id(v, label):
    if type(v) is not str or _ID.fullmatch(v) is None:
        raise ValueError(label + " needs a stable physical ID")
    return v


def _real(v, label, *, positive=False):
    if type(v) not in (int, float):
        raise ValueError(label + " must be non-Boolean finite SI")
    try:
        n = float(v)
    except (OverflowError, ValueError) as exc:
        raise ValueError(label + " outside finite binary64 SI") from exc
    if not isfinite(n) or (positive and n <= 0):
        raise ValueError(label + " must be finite" + (" and positive" if positive else ""))
    return n


def _point(value, label):
    if type(value) not in (tuple, list) or len(value) != 3:
        raise ValueError(label + " must have exactly three physical coordinates")
    return tuple(_real(x, label + " component") for x in value)


def _difference(p, q):
    return tuple(p[i] - q[i] for i in range(3))


def _distance(p, q):
    return hypot(*_difference(p, q))


def _normal(value):
    n = _point(value, "unit surface normal")
    length = hypot(*n)
    if not isfinite(length) or abs(length - 1.0) > 1e-12:
        raise ValueError("surface query requires a genuine unit-normal vector")
    return n


@dataclass(frozen=True, slots=True)
class NamedSourcePosition:
    """Actual 3D source placement linked by its ORIGINAL M39a source identity."""
    source_id: str
    position_m: tuple[float, float, float]

    def __post_init__(self):
        _id(self.source_id, "named point source")
        object.__setattr__(self, "position_m", _point(self.position_m, "source position"))


@dataclass(frozen=True, slots=True)
class SphericalVoid:
    """One actual sealed spherical hole cut from the SAME connected metal."""
    cavity_id: str
    center_m: tuple[float, float, float]
    radius_m: float
    positions: tuple[NamedSourcePosition, ...] = ()

    def __post_init__(self):
        _id(self.cavity_id, "spherical cavity")
        object.__setattr__(self, "center_m", _point(self.center_m, "cavity center"))
        radius = _real(self.radius_m, "spherical cavity radius", positive=True)
        if not 1e-6 <= radius <= 1e6:
            raise UnsupportedProblem("spherical cavity radius outside qualified SI scale 1e-6..1e6 m")
        object.__setattr__(self, "radius_m", radius)
        if type(self.positions) not in (tuple, list) or any(
            type(x) is not NamedSourcePosition for x in self.positions
        ):
            raise ValueError("spherical cavity needs typed source positions")
        if len(self.positions) > 16:
            raise UnsupportedProblem("one cavity permits no more than 16 source positions")
        object.__setattr__(self, "positions", tuple(self.positions))
        ids = [p.source_id for p in self.positions]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate named physical source placement")


@dataclass(frozen=True, slots=True)
class SphericalMetalGeometry:
    """Spherical exterior and multiple separate actual spherical inner faces."""
    outer_center_m: tuple[float, float, float]
    outer_radius_m: float
    voids: tuple[SphericalVoid, ...]

    def __post_init__(self):
        object.__setattr__(self, "outer_center_m", _point(self.outer_center_m, "outer metal center"))
        radius = _real(self.outer_radius_m, "outer metal radius", positive=True)
        if not 1e-6 <= radius <= 1e6:
            raise UnsupportedProblem("outer spherical metal radius outside qualified SI scale 1e-6..1e6 m")
        object.__setattr__(self, "outer_radius_m", radius)
        if type(self.voids) not in (tuple, list) or any(type(v) is not SphericalVoid for v in self.voids):
            raise ValueError("spherical metal needs typed sealed spherical voids")
        if not 1 <= len(self.voids) <= 8:
            raise UnsupportedProblem("spherical metal admits 1..8 genuinely disjoint sealed voids")
        object.__setattr__(self, "voids", tuple(self.voids))


def _validated(gauss, geometry):
    """Defensively rebuild every value and prove physical containment."""
    if type(gauss) is not GaussCavityAssembly or type(geometry) is not SphericalMetalGeometry:
        raise TypeError("M40a requires actual typed M39a charges and spherical geometry")
    ledger = solve_cavity_gauss(gauss)
    a = ledger.assembly
    if type(geometry.voids) is not tuple:
        raise ValueError("outer geometry inventory must be frozen")
    voids = []
    for v in geometry.voids:
        if type(v) is not SphericalVoid or type(v.positions) is not tuple:
            raise ValueError("nested physical cavity geometry has been tampered")
        placements = tuple(NamedSourcePosition(p.source_id, p.position_m)
                           if type(p) is NamedSourcePosition else
                           (_ for _ in ()).throw(ValueError("invalid nested source placement"))
                           for p in v.positions)
        voids.append(SphericalVoid(v.cavity_id, v.center_m, v.radius_m, placements))
    g = SphericalMetalGeometry(geometry.outer_center_m, geometry.outer_radius_m, tuple(voids))
    if len(g.voids) != len(a.cavities):
        raise ValueError("spherical geometry has missing or extra M39a cavities")
    if any(v.cavity_id != c.cavity_id for v, c in zip(g.voids, a.cavities)):
        raise ValueError("spherical cavity identities/order contradict authoritative Gauss topology")
    outer_c, outer_r = g.outer_center_m, g.outer_radius_m
    if any(abs(x) > 1e6 * outer_r for x in outer_c):
        raise UnsupportedProblem("large absolute world-coordinate offsets are numerically unqualified")
    nonzero = []
    for void, cav in zip(g.voids, a.cavities):
        if any(abs(x) > 1e6 * outer_r for x in void.center_m):
            raise UnsupportedProblem("cavity coordinates exceed qualified precision domain")
        expected = {s.source_id for s in cav.sources}
        actual = {p.source_id for p in void.positions}
        if expected != actual or len(void.positions) != len(cav.sources):
            raise ValueError("spherical source placements do not match original M39a source IDs")
        if _distance(void.center_m, outer_c) + void.radius_m >= outer_r * (1.0 - 1e-8):
            raise UnsupportedProblem("a spherical void touches or escapes the outer metal surface")
        for p in void.positions:
            if any(abs(x) > 1e6 * outer_r for x in p.position_m):
                raise UnsupportedProblem("source coordinates exceed qualified precision domain")
            if _distance(p.position_m, void.center_m) >= void.radius_m * (1.0 - 1e-8):
                raise UnsupportedProblem("source is outside/on/too close to its assigned spherical wall")
        nonzero.extend(abs(s.charge_c) for s in cav.sources if s.charge_c != 0.0)
    for i, left in enumerate(g.voids):
        for right in g.voids[i+1:]:
            if _distance(left.center_m, right.center_m) <= (
                left.radius_m + right.radius_m + outer_r * 1e-8
            ):
                raise UnsupportedProblem("spherical cavity voids overlap/touch or leave no qualified metal gap")
    if nonzero:
        if min(nonzero) < 1e-240 or max(nonzero) > 1e180 or max(nonzero)/min(nonzero) > 1e20:
            raise UnsupportedProblem("local sigma source magnitudes/dynamic range are not numerically qualified")
    return a, g, ledger


@dataclass(frozen=True, slots=True)
class SphericalSigmaModel:
    gauss_assembly: GaussCavityAssembly
    geometry: SphericalMetalGeometry

    def __post_init__(self):
        _validated(self.gauss_assembly, self.geometry)


@dataclass(frozen=True, slots=True)
class SphericalSigmaSolution:
    """Actual geometry-qualified local surface density, NOT general E/V solver."""
    model: SphericalSigmaModel
    gauss_ledger: GaussCavityLedger

    def _fresh(self):
        if type(self.model) is not SphericalSigmaModel or type(self.gauss_ledger) is not GaussCavityLedger:
            raise ValueError("invalid typed spherical sigma result")
        a, g, fresh = _validated(self.model.gauss_assembly, self.model.geometry)
        def canonical(v):
            return json.dumps(v, sort_keys=True, allow_nan=False, separators=(",", ":"))
        if canonical(self.gauss_ledger.as_dict()) != canonical(fresh.as_dict()):
            raise ValueError("saved Gauss source IDs/charges/faces differ from physical re-execution")
        return a, g, fresh

    def inner_surface_charge_density(self, cavity_id: str, normal):
        n = _normal(normal)
        a, g, _ = self._fresh()
        _id(cavity_id, "requested cavity")
        for cavity, void in zip(a.cavities, g.voids):
            if cavity.cavity_id != cavity_id:
                continue
            R = void.radius_m
            lookup = {p.source_id: p.position_m for p in void.positions}
            terms = []
            for source in cavity.sources:
                if source.charge_c == 0.0:
                    continue
                dvec = _difference(lookup[source.source_id], void.center_m)
                d = hypot(*dvec)
                gap = R-d
                sep = hypot(*(R*n[k]-dvec[k] for k in range(3)))
                if sep <= 0 or not isfinite(sep) or gap <= 0:
                    raise UnsupportedProblem("source-to-wall singular or unqualified density query")
                term = -source.charge_c * ((gap * (R+d))/(4*pi*R*sep**3))
                if not isfinite(term) or term == 0.0:
                    raise UnsupportedProblem("nonzero spherical sigma contribution overflowed/underflowed binary64")
                terms.append(term)
            try:
                total = fsum(terms)
                magnitude = fsum(abs(v) for v in terms)
            except (OverflowError, ValueError) as exc:
                raise UnsupportedProblem("surface density not representable in qualified binary64") from exc
            if not isfinite(total) or not isfinite(magnitude):
                raise UnsupportedProblem("surface density overflowed")
            if magnitude and abs(total) <= magnitude * 1e-12:
                # A zero caused by near-cancellation of signed approximate
                # contributions is NOT a certified physical theorem zero.
                # Exact same-site cancellations are a separate safe case.
                by_position = {}
                for s in cavity.sources:
                    by_position.setdefault(lookup[s.source_id], []).append(s.charge_c)
                if any(fsum(group) != 0.0 for group in by_position.values()):
                    raise UnsupportedProblem("ill-conditioned signed local-sigma cancellation cannot certify zero")
                return 0.0
            return 0.0 if total == 0 else total
        raise ValueError("unknown cavity identity for surface-density query")

    def outer_surface_charge_density(self, normal):
        _normal(normal)
        _, g, ledger = self._fresh()
        denom = 4*pi*g.outer_radius_m*g.outer_radius_m
        sigma = ledger.outer_face_charge_c / denom
        if not isfinite(sigma) or (ledger.outer_face_charge_c != 0 and sigma == 0.0):
            raise UnsupportedProblem("nonzero outer surface density overflows/underflows qualified binary64")
        return 0.0 if sigma == 0.0 else sigma

    def as_dict(self, probes=()):
        """Export geometry and *explicitly requested* samples, not fictional maps."""
        a, g, ledger = self._fresh()
        if type(probes) not in (tuple, list) or len(probes) > 64:
            raise ValueError("sigma record needs up to 64 explicit signed surface probes")
        samples = []
        for query in probes:
            if type(query) not in (tuple, list) or len(query) != 2:
                raise ValueError("probe needs (cavity_id or outer, unit normal)")
            surface, raw_normal = query
            n = _normal(raw_normal)
            if surface == "outer" and type(surface) is str:
                density = self.outer_surface_charge_density(n)
            else:
                density = self.inner_surface_charge_density(surface, n)
            samples.append({"surface_id":surface, "unit_normal":list(n),
                            "sigma_c_per_m2":density})
        return {
            "schema": SCHEMA, "source_role": ROLE,
            "original_workbook_items_newly_solved":0,
            "assumptions": list(ASSUMPTIONS), "unsupported": list(UNSUPPORTED),
            "geometry_qualified_spherical_only": True,
            "irregular_3d_fields_qualified": False, "time_evolution_defined": False,
            "units":{"charge":"C","position":"m","radius":"m","surface_density":"C/m^2"},
            "outer_sphere":{"center_m":list(g.outer_center_m),"radius_m":g.outer_radius_m},
            "voids":[{"cavity_id":v.cavity_id,"center_m":list(v.center_m),
                      "radius_m":v.radius_m,
                      "source_positions":[{"source_id":s.source_id,"position_m":list(s.position_m)}
                                          for s in v.positions]}
                     for v in g.voids],
            "gauss":ledger.as_dict(), "samples":samples,
        }

    def electric_field_at(self, *args, **kwargs):
        raise UnsupportedProblem("M40a surface density samples do NOT qualify local E(x) or 3D maps")

    def potential_at(self, *args, **kwargs):
        raise UnsupportedProblem("M40a surface density samples do NOT qualify local V(x)")

    def force_on_source(self, *args, **kwargs):
        raise UnsupportedProblem("M40a source-force dynamics are not qualified")

    def ground_wire_current(self, *args, **kwargs):
        raise UnsupportedProblem("separate static M40a states have no wire-current chronology")


def solve_spherical_sigma(model: SphericalSigmaModel) -> SphericalSigmaSolution:
    if type(model) is not SphericalSigmaModel:
        raise TypeError("requires actual SphericalSigmaModel")
    a, g, ledger = _validated(model.gauss_assembly, model.geometry)
    fresh = SphericalSigmaModel(a, g)
    return SphericalSigmaSolution(fresh, ledger)
