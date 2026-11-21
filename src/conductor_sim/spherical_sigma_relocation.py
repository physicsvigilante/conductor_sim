"""M40c: one immutable static named-source relocation, with genuine M40a sigma re-solves.

Two separately qualified electrostatic equilibria, NOT motion, wire current, force,
work, or a claim that irregular conductor cavities have been solved.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import isclose, isfinite, ulp

from .spherical_multicavity import (
    NamedSourcePosition, SphericalSigmaModel, _near_wall_sigma_precision,
    SphericalMetalGeometry, SphericalVoid, solve_spherical_sigma,
)
from .solver import UnsupportedProblem

SCHEMA = "conductor_sim.m40c_static_spherical_relocation.v1"
ROLE = "teacher_hypothetical_static_spherical_source_relocation_only"
NO_CHRONOLOGY = (
    "Before and after are separately qualified STATIC hypothetical equilibria; "
    "they do not define a physical motion path, elapsed time, grounding current, "
    "force, work or energy transfer."
)
DEFAULT_RAYS = ((1., 0., 0.), (-1., 0., 0.), (0., 1., 0.),
                (0., -1., 0.), (0., 0., 1.), (0., 0., -1.))


@dataclass(frozen=True, slots=True)
class StaticSourceRelocation:
    baseline: SphericalSigmaModel
    cavity_id: str
    source_id: str
    new_position_m: tuple[float, float, float]

    def __post_init__(self):
        # Keep the requalified value snapshot, including the electrical constraint.
        object.__setattr__(self, "baseline", solve_spherical_sigma(self.baseline).model)
        if type(self.cavity_id) is not str or not self.cavity_id:
            raise ValueError("relocation requires an original cavity ID")
        # Reuse the scientific source identity/point validation, including Boolean refusal.
        p = NamedSourcePosition(self.source_id, self.new_position_m)
        object.__setattr__(self, "new_position_m", p.position_m)


def relocate_spherical_source(model: SphericalSigmaModel,
                              change: StaticSourceRelocation) -> SphericalSigmaModel:
    """Revalidate baseline, reposition exactly one owned source, requalify the result."""
    if type(model) is not SphericalSigmaModel or type(change) is not StaticSourceRelocation:
        raise TypeError("requires real M40a spherical model and M40c relocation")
    fresh = solve_spherical_sigma(model).model
    if fresh != change.baseline:
        raise ValueError("instruction belongs to a different baseline")
    change = replace(change)
    cavities = []
    count = 0
    for void in fresh.geometry.voids:
        placements = []
        for p in void.positions:
            if p.source_id == change.source_id:
                if void.cavity_id != change.cavity_id:
                    raise ValueError("original source ID belongs to a different cavity")
                if p.position_m == change.new_position_m:
                    raise ValueError("source relocation must change the actual position")
                placements.append(NamedSourcePosition(p.source_id, change.new_position_m))
                count += 1
            else:
                placements.append(p)
        cavities.append(SphericalVoid(void.cavity_id, void.center_m, void.radius_m,
                                     tuple(placements)))
    if count != 1:
        raise ValueError("relocation must identify exactly one existing source")
    g = fresh.geometry
    next_geometry = SphericalMetalGeometry(g.outer_center_m, g.outer_radius_m,
                                           tuple(cavities))
    # Uses unchanged M40a admission: geometry gaps, SI, stable IDs and source ownership.
    return SphericalSigmaModel(fresh.gauss_assembly, next_geometry)


def _rays(value):
    if value is None:
        value = DEFAULT_RAYS
    if type(value) not in (tuple, list) or not 1 <= len(value) <= 24:
        raise ValueError("relocation comparison permits 1..24 original raw surface rays")
    out = []
    for ray in value:
        if type(ray) not in (tuple, list) or len(ray) != 3:
            raise ValueError("each raw spherical ray requires 3 coordinates")
        if any(type(v) not in (float, int) for v in ray):
            raise ValueError("raw ray components must be non-Boolean finite SI")
        out.append(tuple(float(v) for v in ray))
    return tuple(out)


def _delta(before: float, after: float, *, moved_terms):
    if moved_terms is None:
        if before != after:
            raise ValueError("untouched cavity falsely changes local sigma")
        return "unchanged_exact", 0.0
    difference = after - before
    if not isfinite(difference):
        raise UnsupportedProblem("static local-sigma difference overflowed")
    # Resolution is a property of the numeric comparison, not a theorem zero.
    # Stationary sources cancel analytically. Check the observed total against
    # the relocated source alone, evaluated using the precise spherical kernel.
    old_term, new_term = moved_terms
    scale = max(abs(before), abs(after), abs(old_term), abs(new_term))
    source_delta = new_term - old_term
    if (abs(source_delta) <= 32 * ulp(scale)
            or not isclose(difference, source_delta, rel_tol=2e-10, abs_tol=0.0)):
        return "unresolved_numerical_difference", None
    return "resolved_difference", difference


@dataclass(frozen=True, slots=True)
class StaticSigmaComparison:
    baseline: SphericalSigmaModel
    revised: SphericalSigmaModel
    instruction: StaticSourceRelocation
    rays: tuple[tuple[float, float, float], ...]

    def as_dict(self) -> dict:
        """Re-execute both native states and reject re-used/tampered revised models."""
        if type(self.baseline) is not SphericalSigmaModel or type(self.revised) is not SphericalSigmaModel:
            raise TypeError("comparison requires genuine qualified static spherical models")
        if type(self.instruction) is not StaticSourceRelocation:
            raise TypeError("comparison requires real typed relocation")
        # Constructors can be mutated via Python internals, so rebuild the authority.
        instruction = replace(self.instruction)
        expected = relocate_spherical_source(self.baseline, instruction)
        fresh_revision = solve_spherical_sigma(self.revised).model
        # Compare original typed immutable geometry including exact source position inventory.
        if expected != fresh_revision:
            raise ValueError("comparison's revised model is not the requested static relocation")
        before = solve_spherical_sigma(self.baseline)
        after = solve_spherical_sigma(self.revised)
        directions = _rays(self.rays)
        source_before = before.as_dict()
        source_after = after.as_dict()
        if source_before["gauss"] != source_after["gauss"]:
            raise ValueError("one static position revision must preserve all Gauss totals")
        target = next(v for v in source_before["voids"] if v["cavity_id"] == instruction.cavity_id)
        moved = next(s for s in target["source_positions"] if s["source_id"] == instruction.source_id)
        cavity = next(c for c in self.baseline.gauss_assembly.cavities if c.cavity_id == instruction.cavity_id)
        charge = next(s.charge_c for s in cavity.sources if s.source_id == instruction.source_id)
        records = {"baseline": [], "revised": []}
        rows = []
        for void in source_before["voids"]:
            cavity_id = void["cavity_id"]
            probes = tuple((cavity_id, ray) for ray in directions)
            left, right = before.as_dict(probes), after.as_dict(probes)
            records["baseline"].append(left)
            records["revised"].append(right)
            samples = []
            for a, b in zip(left["samples"], right["samples"]):
                if a["input_direction"] != b["input_direction"]:
                    raise ValueError("sample rays are not identical between static scenes")
                terms = None
                if cavity_id == instruction.cavity_id:
                    terms = tuple(_near_wall_sigma_precision(
                        charge, void["radius_m"], position, void["center_m"], a["input_direction"]
                    ) for position in (moved["position_m"], instruction.new_position_m))
                flag, value = _delta(a["sigma_c_per_m2"], b["sigma_c_per_m2"], moved_terms=terms)
                samples.append({
                    "input_direction": a["input_direction"],
                    "unit_normal_display_only": a["unit_normal"],
                    "sigma_baseline_c_per_m2": a["sigma_c_per_m2"],
                    "sigma_revised_c_per_m2": b["sigma_c_per_m2"],
                    "delta_status": flag, "delta_c_per_m2": value,
                })
            face = next(c for c in source_before["gauss"]["cavities"]
                        if c["cavity_id"] == cavity_id)
            rows.append({"cavity_id": cavity_id, "radius_m": void["radius_m"],
                         "center_m": void["center_m"],
                         "whole_face_induced_charge_c": face["induced_inner_face_total_charge_c"],
                         "samples": samples})
        outer_ray = (("outer", (1.0, 0.0, 0.0)),)
        out_a, out_b = before.as_dict(outer_ray), after.as_dict(outer_ray)
        records["baseline"].append(out_a)
        records["revised"].append(out_b)
        if out_a["samples"] != out_b["samples"]:
            raise ValueError("outer uniform sigma cannot change with source-only relocation")
        return {"schema": SCHEMA, "source_role": ROLE,
                "original_workbook_items_newly_solved": 0,
                "time_evolution_defined": False, "no_chronology": NO_CHRONOLOGY,
                "geometry_qualified_spherical_only": True,
                "local_3d_fields_or_forces_qualified": False,
                "units": {"position": "m", "charge": "C", "surface_density": "C/m^2"},
                "relocation": {"source_id": instruction.source_id,
                               "cavity_id": instruction.cavity_id,
                               "original_position_m": moved["position_m"],
                               "revised_position_m": list(instruction.new_position_m)},
                "outer_sphere": source_before["outer_sphere"],
                "baseline_voids": source_before["voids"],
                "revised_voids": source_after["voids"],
                "gauss": source_before["gauss"], "cavities": rows,
                "outer_uniform_sigma_c_per_m2": out_a["samples"][0]["sigma_c_per_m2"],
                "m40a_v2_witness_records": records}


def compare_static_spherical_relocation(model: SphericalSigmaModel,
                                        change: StaticSourceRelocation,
                                        *, rays=None) -> StaticSigmaComparison:
    """Return a re-checkable comparison of two separately solved static hypotheses."""
    original = solve_spherical_sigma(model).model
    new = relocate_spherical_source(original, change)
    result = StaticSigmaComparison(original, new, change, _rays(rays))
    result.as_dict()  # Fail closed at creation as well as export.
    return result
