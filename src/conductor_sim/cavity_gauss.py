"""M39a: exact Gauss-law *integrals* for explicitly authored sealed-cavity topology.

This is a composable topology-only theorem ledger, NOT a geometry, mesh,
surface-density, electric-field, force, or time-domain solver. Membership
of charges in disjoint sealed cavities is a physical author declaration,
NOT an independently verified spatial containment assertion.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from math import fsum, isfinite
import re

from .model import EPSILON_0, FixedNetCharge, Grounded
from .solver import UnsupportedProblem


SCHEMA = "conductor_sim.gauss_cavity_network.v1"
ROLE = "teacher_hypothetical_topology_gauss_only"
ANALYSIS = "conditional exact Gauss-law conductor-face charge TOTALS, never local geometry"
MEMBERSHIP = (
    "TEACHER-DECLARED membership in mutually sealed noncontacting cavities; "
    "no geometry or containment was verified"
)
ASSUMPTIONS = (
    "one connected ideal electrostatic metal with 1..8 disjoint fully sealed voids",
    "all charges are fixed inside their declared void and do not contact metal",
    "no exterior sources, applied exterior field, other metal, internal electrode or aperture",
    "homogeneous positive permittivity; vacuum/infinity reference V=0 for grounding",
    "no time evolution; each revised assembly is a separate static hypothetical",
)
UNSUPPORTED = (
    "pointwise electric field E(x), including within individual cavities",
    "pointwise potential V(x), except the grounded metal boundary theorem",
    "surface charge density sigma(x) or arbitrary-shaped induced charge map",
    "geometric cavity containment, radius, area, mesh and visualization",
    "source force, energy, wire current, discharge time, heat or radiation",
)
UNITS = {
    "charge": "C",
    "permittivity": "F/m",
    "exterior_electric_flux": "V*m",
}
_ID = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}\Z")


def _identifier(value, what):
    if type(value) is not str or _ID.fullmatch(value) is None:
        raise ValueError(f"{what} must be a stable 1..64-character ASCII physical ID")
    return value


def _real(value, what, *, positive=False):
    if type(value) not in (int, float):
        raise ValueError(f"{what} must be a finite non-Boolean real SI value")
    try:
        number = float(value)
    except (OverflowError, ValueError) as ex:
        raise ValueError(f"{what} cannot be represented as finite SI") from ex
    if not isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{what} must be finite" + (" and positive" if positive else ""))
    return number


def _sum(values, what):
    try:
        number = fsum(values)
    except (ValueError, OverflowError) as ex:
        raise UnsupportedProblem(f"{what} is not representable in the supported SI float domain") from ex
    if not isfinite(number):
        raise UnsupportedProblem(f"{what} overflowed")
    return 0.0 if number == 0 else number


def _zero(value):
    return 0.0 if value == 0 else value


@dataclass(frozen=True, slots=True)
class CavityCharge:
    """Teacher-declared fixed charge ID and signed Coulombs; NO coordinates."""

    source_id: str
    charge_c: float

    def __post_init__(self):
        _identifier(self.source_id, "source identity")
        object.__setattr__(self, "charge_c", _real(self.charge_c, "source charge"))


@dataclass(frozen=True, slots=True)
class SealedCavity:
    """One hole within the SAME connected conductor, not another metal."""

    cavity_id: str
    sources: tuple[CavityCharge, ...] = ()

    def __post_init__(self):
        _identifier(self.cavity_id, "cavity identity")
        if type(self.sources) not in (tuple, list) or any(
            type(x) is not CavityCharge for x in self.sources
        ):
            raise TypeError("sealed cavity needs an ordered sequence of typed CavityCharge sources")
        if len(self.sources) > 16:
            raise UnsupportedProblem("a cavity may hold no more than 16 declared sources")
        object.__setattr__(self, "sources", tuple(self.sources))


@dataclass(frozen=True, slots=True)
class GaussCavityAssembly:
    """Topology declaration, NOT an ElectrostaticProblem or spatial solver."""

    conductor_id: str
    cavities: tuple[SealedCavity, ...]
    constraint: FixedNetCharge | Grounded
    permittivity_f_per_m: float = EPSILON_0

    def __post_init__(self):
        _identifier(self.conductor_id, "metal identity")
        if type(self.cavities) not in (tuple, list) or any(
            type(c) is not SealedCavity for c in self.cavities
        ):
            raise TypeError("cavity assembly needs a sequence of typed sealed cavities")
        if not 1 <= len(self.cavities) <= 8:
            raise UnsupportedProblem("one connected conductor needs 1..8 sealed cavities")
        if type(self.constraint) not in (FixedNetCharge, Grounded):
            raise UnsupportedProblem("only fixed-net-charge isolated or truly grounded metal")
        if type(self.constraint) is FixedNetCharge:
            _real(self.constraint.charge_c, "isolated metal charge")
        object.__setattr__(
            self, "permittivity_f_per_m",
            _real(self.permittivity_f_per_m, "homogeneous permittivity", positive=True),
        )
        object.__setattr__(self, "cavities", tuple(self.cavities))
        seen = {self.conductor_id}
        count = 0
        for cavity in self.cavities:
            _identifier(cavity.cavity_id, "cavity identity")
            if cavity.cavity_id in seen:
                raise ValueError("duplicate physical conductor/cavity/source identity")
            seen.add(cavity.cavity_id)
            if type(cavity.sources) is not tuple:
                raise TypeError("nested cavity source inventory is not frozen")
            for source in cavity.sources:
                if type(source) is not CavityCharge:
                    raise TypeError("source inventory contains a non-CavityCharge")
                _identifier(source.source_id, "point-source identity")
                _real(source.charge_c, "point source charge")
                if source.source_id in seen:
                    raise ValueError("duplicate or multiply assigned physical charge source identity")
                seen.add(source.source_id)
                count += 1
        if count > 16:
            raise UnsupportedProblem("one conductor permits at most 16 authored source charges")


def _validated_assembly(model):
    """Rebuild every nested value; frozen Python dataclasses can be tampered."""
    if type(model) is not GaussCavityAssembly:
        raise TypeError("Gauss ledger requires a typed GaussCavityAssembly")
    if type(model.cavities) is not tuple:
        raise TypeError("authored sealed-cavity inventory is not immutable")
    cavities = []
    for cavity in model.cavities:
        if type(cavity) is not SealedCavity or type(cavity.sources) is not tuple:
            raise ValueError("authored nested cavity identity/inventory is not valid")
        sources = []
        for source in cavity.sources:
            if type(source) is not CavityCharge:
                raise ValueError("authored source is not a typed charge")
            sources.append(CavityCharge(source.source_id, source.charge_c))
        cavities.append(SealedCavity(cavity.cavity_id, tuple(sources)))
    if type(model.constraint) is Grounded:
        constraint = Grounded()
    elif type(model.constraint) is FixedNetCharge:
        constraint = FixedNetCharge(
            _real(model.constraint.charge_c, "isolated metal authored net charge")
        )
    else:
        raise UnsupportedProblem("unsupported conductor electrical constraint")
    return GaussCavityAssembly(
        model.conductor_id, tuple(cavities), constraint, model.permittivity_f_per_m
    )


@dataclass(frozen=True, slots=True)
class CavityFaceTotal:
    cavity_id: str
    enclosed_source_charge_c: float
    induced_inner_face_charge_c: float


@dataclass(frozen=True, slots=True)
class GaussCavityLedger:
    """Verified exact-integral summary; no geometry-dependent observables."""

    assembly: GaussCavityAssembly
    faces: tuple[CavityFaceTotal, ...]
    total_source_charge_c: float
    net_metal_charge_c: float
    outer_face_charge_c: float
    exterior_electric_flux_v_m: float
    exterior_field_zero_theorem: bool

    def _record(self):
        a = self.assembly
        isolated = type(a.constraint) is FixedNetCharge
        return {
            "schema": SCHEMA,
            "analysis": ANALYSIS,
            "source_role": ROLE,
            "original_workbook_items_newly_solved": 0,
            "membership_provenance": MEMBERSHIP,
            "assumptions": list(ASSUMPTIONS),
            "unsupported": list(UNSUPPORTED),
            "units": dict(UNITS),
            "conductor": {
                "conductor_id": a.conductor_id,
                "constraint": "isolated_fixed_net_charge" if isolated else "grounded_zero_potential",
                "authored_isolated_metal_charge_c": a.constraint.charge_c if isolated else None,
                "equilibrium_net_metal_charge_c": self.net_metal_charge_c,
            },
            "permittivity_f_per_m": a.permittivity_f_per_m,
            "cavities": [
                {
                    "cavity_id": cav.cavity_id,
                    "sealed": True,
                    "authored_sources": [
                        {"source_id": s.source_id, "charge_c": s.charge_c}
                        for s in cav.sources
                    ],
                    "source_charge_sum_c": face.enclosed_source_charge_c,
                    "induced_inner_face_total_charge_c": face.induced_inner_face_charge_c,
                } for cav, face in zip(a.cavities, self.faces)
            ],
            "total_interior_source_charge_c": self.total_source_charge_c,
            "outer_face_total_charge_c": self.outer_face_charge_c,
            "exterior_electric_flux_v_m": self.exterior_electric_flux_v_m,
            "exterior_electric_field_zero_theorem": self.exterior_field_zero_theorem,
            "geometry_verified": False,
            "source_position_defined": False,
            "time_evolution_defined": False,
        }

    def as_dict(self):
        validate_cavity_gauss_ledger(self)
        return json.loads(json.dumps(self._record(), allow_nan=False))

    def electric_field_at(self, *args, **kwargs):
        raise UnsupportedProblem("topology-only Gauss totals do NOT supply pointwise E(x)")

    def potential_at(self, *args, **kwargs):
        raise UnsupportedProblem("topology-only Gauss totals do NOT supply pointwise V(x)")

    def surface_charge_density_at(self, *args, **kwargs):
        raise UnsupportedProblem("topology-only Gauss totals do NOT supply local sigma(x)")

    def force_on_source(self, *args, **kwargs):
        raise UnsupportedProblem("topology-only Gauss totals do NOT supply source force")

    def ground_wire_current(self, *args, **kwargs):
        raise UnsupportedProblem("separate static Gauss ledgers do NOT define wire current")


def solve_cavity_gauss(model: GaussCavityAssembly) -> GaussCavityLedger:
    """Gauss integrals only, conditional on declared closed-cavity topology."""
    a = _validated_assembly(model)
    faces = []
    for cavity in a.cavities:
        amount = _sum((s.charge_c for s in cavity.sources), "cavity source sum")
        faces.append(CavityFaceTotal(cavity.cavity_id, amount, _zero(-amount)))
    total = _sum((f.enclosed_source_charge_c for f in faces), "all enclosed source charge")
    grounded = type(a.constraint) is Grounded
    metal = _zero(-total) if grounded else _real(a.constraint.charge_c, "isolated metal charge")
    outer = 0.0 if grounded else _sum((metal, total), "outer metal face")
    try:
        flux = outer / a.permittivity_f_per_m
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        raise UnsupportedProblem("exterior electric flux is not representable") from exc
    if not isfinite(flux):
        raise UnsupportedProblem("exterior electric flux overflows qualified SI range")
    return GaussCavityLedger(
        a, tuple(faces), total, metal, outer, _zero(flux), grounded
    )


def validate_cavity_gauss_ledger(value: GaussCavityLedger) -> None:
    """Refuse forged native saved data, including Boolean zero time-style attacks.

    This is an in-process consistency check, NOT a cryptographic certificate
    of origin for arbitrary self-consistent JSON or a geometry proof.
    """
    if type(value) is not GaussCavityLedger:
        raise TypeError("requires native GaussCavityLedger")
    if type(value.faces) is not tuple or any(
        type(x) is not CavityFaceTotal for x in value.faces
    ):
        raise ValueError("native cavity-face ledger structure is forged")
    if (len(value.faces) != len(value.assembly.cavities)
            or any(type(face.cavity_id) is not str
                   or face.cavity_id != cavity.cavity_id
                   for cavity,face in zip(value.assembly.cavities,value.faces))):
        raise ValueError("native cavity face identities/order contradict authored physical topology")
    genuine = solve_cavity_gauss(value.assembly)
    try:
        def stable(raw):
            return json.dumps(raw, sort_keys=True, allow_nan=False, separators=(",", ":"))
        if stable(genuine._record()) != stable(value._record()):
            raise ValueError("Gauss ledger totals, source IDs or SI record differ from native re-execution")
    except (TypeError, OverflowError, AttributeError) as exc:
        raise ValueError("forged native Gauss evidence is malformed") from exc
