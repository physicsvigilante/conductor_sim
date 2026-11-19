"""Independent stdlib M40b verifier: v2 real+Kelvin fields and exact viewer evidence.

Imports no conductor_sim native physics. Self-consistent external JSON is NOT
cryptographic proof of historical teacher authorship without a trusted anchor.
"""
from __future__ import annotations

import argparse
import json
from math import atan2, degrees, hypot
from pathlib import Path

from tools.m40a_spherical_reference import check_record

SCHEMA = "conductor_sim.m40b_spherical_surface_atlas.v1"
ROLE = "teacher_hypothetical_spherical_multicavity_only"
WARNING = (
    "ACTUAL 3D SPHERES SHOWN AS XY ORTHOGRAPHIC PROJECTIONS; z IS LISTED "
    "SEPARATELY. XY OVERLAP DOES NOT PROVE 3D CONTACT OR INTERSECTION."
)
LIMITS = (
    "Signed tiles are distinct surface POINT SAMPLES, not continuous "
    "fields, area averages, integrated fluxes, or time animation. "
    "No irregular cavities, source forces, E(x), V(x), ground-wire "
    "currents, newly solved source JEE problems, or 3D field maps."
)


def canonical(v):
    return json.dumps(v, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":"))


def equal(actual, expected, label):
    if canonical(actual) != canonical(expected):
        raise ValueError(label + " differs from independently validated physical witness")


def require(flag, label):
    if not flag:
        raise ValueError(label)


def check_manifest(doc):
    if type(doc) is not dict:
        raise ValueError("viewer record must be a mapping")
    flags = {
        "schema": SCHEMA, "source_role": ROLE,
        "original_workbook_items_newly_solved": 0,
        "geometry_qualified_spherical_only": True,
        "visual_is_xy_orthographic_projection": True,
        "projection_is_2d_field_or_cross_section": False,
        "sample_tiles_are_point_observations_only": True,
        "time_evolution_defined": False,
        "irregular_3d_fields_qualified": False,
        "warning": WARNING, "limitations": LIMITS,
        "units": {"coordinate": "m", "radius": "m", "charge": "C",
                  "surface_density": "C/m^2"},
    }
    for k, v in flags.items():
        require(k in doc, "missing fixed M40b metadata " + k)
        equal(doc[k], v, k)
    require(set(doc) == set(flags) | {
        "outer_sphere", "cavities", "outer", "native_m40a_v2_witness_records",
    }, "unexpected/missing M40b manifest keys")
    witness = doc["native_m40a_v2_witness_records"]
    if type(witness) is not dict or set(witness) != {"isolated", "grounded"}:
        raise ValueError("real isolated/grounded v2 native source evidence required")
    a, b = witness["isolated"], witness["grounded"]
    if type(a) is not list or type(b) is not list or not 2 <= len(a) <= 9 or len(a) != len(b):
        raise ValueError("expected 1..8 cavities plus one outer witness per state")
    if type(doc["cavities"]) is not list or len(doc["cavities"]) != len(a) - 1:
        raise ValueError("actual cavity count differs from native source evidence")
    for r in a + b:
        check_record(r)  # Independent original real+Kelvin image normal electric field.
    first, other = a[0], b[0]
    # Every face witness in one state must be from THAT exact physical state;
    # reject an internally consistent grounded record swapped into isolated.
    for label, records in (("isolated", a), ("grounded", b)):
        baseline = records[0]
        for record in records[1:]:
            equal(record["gauss"], baseline["gauss"],
                  label + " all-face source and constraint identity")
            equal(record["voids"], baseline["voids"],
                  label + " all-face 3D cavity geometry")
            equal(record["outer_sphere"], baseline["outer_sphere"],
                  label + " all-face outer geometry")
    equal(doc["outer_sphere"], first["outer_sphere"], "outer 3D geometry")
    require(first["gauss"]["conductor"]["constraint"] == "isolated_fixed_net_charge",
            "first state must be genuinely isolated")
    require(other["gauss"]["conductor"]["constraint"] == "grounded_zero_potential",
            "second state must be genuinely grounded")
    equal(first["voids"], other["voids"], "paired 3D spherical voids")
    equal(first["gauss"]["cavities"], other["gauss"]["cavities"],
          "paired Gauss source inventory and inner totals")
    equal(first["outer_sphere"], other["outer_sphere"], "paired outer geometry")
    equal(first["gauss"]["conductor"]["conductor_id"],
          other["gauss"]["conductor"]["conductor_id"], "same physical metal identity")
    equal(first["gauss"]["permittivity_f_per_m"],
          other["gauss"]["permittivity_f_per_m"], "same permittivity")
    count = 0
    source_count = 0
    for i, row in enumerate(doc["cavities"]):
        if type(row) is not dict:
            raise ValueError("cavity display row not a mapping")
        left, right = a[i], b[i]
        for record in (left,right):
            equal(record["voids"],first["voids"],"all cavity record geometry")
            equal(record["outer_sphere"],first["outer_sphere"],"all outer record geometry")
            equal(record["gauss"]["cavities"],first["gauss"]["cavities"],
                  "all cavity record source inventory")
        void = first["voids"][i]
        charge = first["gauss"]["cavities"][i]
        require(void["cavity_id"] == charge["cavity_id"],
                "source and geometry cavity identities differ")
        positions = {p["source_id"]:p["position_m"] for p in void["source_positions"]}
        samples = []
        require(1 <= len(left["samples"]) <= 48 and len(left["samples"]) == len(right["samples"]),
                "each cavity atlas needs 1..48 source-pinned samples")
        for x, y in zip(left["samples"],right["samples"]):
            require(x["surface_id"] == void["cavity_id"] and y["surface_id"] == void["cavity_id"],
                    "sample surface source identity mismatch")
            equal(x["input_direction"], y["input_direction"], "same physical raw ray")
            equal(x["unit_normal"], y["unit_normal"], "same display-only normal")
            equal(x["sigma_c_per_m2"], y["sigma_c_per_m2"],
                  "inner local sigma unchanged between grounded/isolated")
            raw = x["input_direction"]
            samples.append({
                "input_direction": raw, "unit_normal": x["unit_normal"],
                "theta_deg_display": degrees(atan2(hypot(raw[0], raw[1]), raw[2])),
                "phi_deg_display": degrees(atan2(raw[1], raw[0])) % 360.,
                "isolated_sigma_c_per_m2": x["sigma_c_per_m2"],
                "grounded_sigma_c_per_m2": y["sigma_c_per_m2"],
            })
        sources = [
            {"source_id": x["source_id"], "charge_c": x["charge_c"],
             "position_m": positions[x["source_id"]]}
            for x in charge["authored_sources"]
        ]
        source_count += len(sources)
        equal(row, {
            "cavity_id": void["cavity_id"], "center_m": void["center_m"],
            "radius_m": void["radius_m"], "sources": sources,
            "source_total_charge_c": charge["source_charge_sum_c"],
            "whole_inner_face_charge_c": charge["induced_inner_face_total_charge_c"],
            "samples": samples,
        }, "visible cavity numbers, geometry, signed sigma and source provenance")
        count += len(samples)
    for x,y in zip(a[-1:],b[-1:]):
        require(len(x["samples"]) == len(y["samples"]) == 1,
                "exactly one outer uniform witness ray")
        require(x["samples"][0]["surface_id"] == y["samples"][0]["surface_id"] == "outer",
                "outer witness must be a spherical outer probe")
        equal(x["input_direction"] if "input_direction" in x else x["samples"][0]["input_direction"],
              y["samples"][0]["input_direction"], "outer physical ray")
    for record in (a[-1],b[-1]):
        equal(record["voids"],first["voids"],"outer record physical geometry")
        equal(record["gauss"]["cavities"],first["gauss"]["cavities"],
              "outer record physical source identity")
    outer = {
        "isolated_whole_face_charge_c": a[-1]["gauss"]["outer_face_total_charge_c"],
        "grounded_whole_face_charge_c": b[-1]["gauss"]["outer_face_total_charge_c"],
        "isolated_uniform_sigma_c_per_m2": a[-1]["samples"][0]["sigma_c_per_m2"],
        "grounded_uniform_sigma_c_per_m2": b[-1]["samples"][0]["sigma_c_per_m2"],
    }
    equal(doc["outer"], outer, "uniform spherical outer Gauss density and whole-face charge")
    require(outer["grounded_uniform_sigma_c_per_m2"] == 0.0,
            "grounded outer sphere must have zero sigma")
    return {"schema":"conductor_sim.m40b_independent_viewer_witness.v1",
            "cavities_checked":len(doc["cavities"]),
            "sources_checked":source_count,
            "signed_inner_samples_verified_by_images":count * 2,
            "outer_samples_verified_by_images":2,
            "new_source_jee_items_solved":0,
            "external_teacher_history_cryptographically_authenticated":False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = check_manifest(json.loads(args.input.read_text(encoding="utf-8")))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(result,sort_keys=True))


if __name__ == "__main__":
    main()
