"""Standalone M39b static topology envelope and paired M39a SI evidence verifier.

Pure stdlib, imports only separately qualified standalone M39a theorem checker.
NEVER imports conductor_sim or a generated HTML/viewer module. This validates
self-consistent author-declared evidence, not externally authenticated history.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.m39a_gauss_reference import check_gauss_record

SCHEMA = "conductor_sim.m39b_topology_view.v1"
TEACHER = "teacher_hypothetical_topology_gauss_only"
WARNING = "TOPOLOGY ONLY — NOT TO SCALE — CARD ORDER IS NOT PHYSICAL POSITION"
PROVENANCE = (
    "Teacher-authored source membership. No physical shape, coordinates or "
    "geometric charge containment was verified."
)
REFUSAL = (
    "Whole-face Gauss-law totals ONLY. No local E(x), V(x), sigma(x), "
    "forces, ground-wire current or real 3D field map."
)


def _required(doc, keys, what):
    if type(doc) is not dict or set(doc) != set(keys):
        raise ValueError(what + " has extra/missing evidence fields")


def _canonical(doc):
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False,
                      ensure_ascii=False)


def check_topology_view_manifest(doc):
    _required(doc, (
        "schema", "source_role", "original_workbook_items_newly_solved",
        "warning", "membership_provenance", "unsupported", "geometry_verified",
        "diagram_is_spatial_geometry", "time_evolution_defined", "metal_id",
        "permittivity_f_per_m", "cavities", "states",
    ), "teacher topology-view manifest")
    if (doc["schema"] != SCHEMA or doc["source_role"] != TEACHER
            or type(doc["original_workbook_items_newly_solved"]) is not int
            or doc["original_workbook_items_newly_solved"] != 0):
        raise ValueError("topology-view source role or original-JEE count falsified")
    if (doc["warning"] != WARNING
            or doc["membership_provenance"] != PROVENANCE
            or doc["unsupported"] != REFUSAL):
        raise ValueError("topology warning, qualification or local-field refusals falsified")
    for key in ("geometry_verified", "diagram_is_spatial_geometry", "time_evolution_defined"):
        if doc[key] is not False:
            raise ValueError("topology view falsely claims verified geometry or dynamics")
    _required(doc["states"], ("isolated", "grounded"), "two distinct static scenarios")
    a = doc["states"]["isolated"]
    b = doc["states"]["grounded"]
    isolated_witness = check_gauss_record(a)
    grounded_witness = check_gauss_record(b)
    if a["conductor"]["constraint"] != "isolated_fixed_net_charge":
        raise ValueError("first teacher scenario is not authentically isolated")
    if b["conductor"]["constraint"] != "grounded_zero_potential":
        raise ValueError("second teacher scenario is not authentically grounded")
    if (type(doc["metal_id"]) is not str
            or a["conductor"]["conductor_id"] != doc["metal_id"]
            or b["conductor"]["conductor_id"] != doc["metal_id"]):
        raise ValueError("physical metal source identity mismatch in paired teacher view")
    if (type(doc["permittivity_f_per_m"]) not in (int, float)
            or _canonical(a["permittivity_f_per_m"]) != _canonical(doc["permittivity_f_per_m"])
            or _canonical(b["permittivity_f_per_m"]) != _canonical(doc["permittivity_f_per_m"])):
        raise ValueError("paired teacher scenarios have mismatched permittivity")
    def inventory(s):
        return [(row["cavity_id"], row["sealed"], row["authored_sources"])
                for row in s["cavities"]]
    if _canonical(inventory(a)) != _canonical(inventory(b)):
        raise ValueError("paired teacher scenarios have different cavity/source inventories")
    if any(_canonical({k: x[k] for k in ("source_charge_sum_c",
                                       "induced_inner_face_total_charge_c")})
           != _canonical({k: y[k] for k in ("source_charge_sum_c",
                                         "induced_inner_face_total_charge_c")})
           for x, y in zip(a["cavities"], b["cavities"])):
        raise ValueError("paired induced-face totals differ for identical teacher source inventory")
    expected = [{
        "cavity_id": row["cavity_id"],
        "authored_sources": row["authored_sources"],
        "source_charge_sum_c": row["source_charge_sum_c"],
        "induced_inner_face_total_charge_c": row["induced_inner_face_total_charge_c"],
    } for row in a["cavities"]]
    if _canonical(doc["cavities"]) != _canonical(expected):
        raise ValueError("diagram source identities, signs or complete-face values are forged")
    return {
        "schema": "conductor_sim.m39b_independent_topology_witness.v1",
        "source_role": TEACHER,
        "original_workbook_items_newly_solved": 0,
        "cavities_checked": len(expected),
        "source_identities_checked": isolated_witness["point_source_identities_checked"],
        "isolated_metal_constraint": isolated_witness["metal_constraint"],
        "grounded_metal_constraint": grounded_witness["metal_constraint"],
        "local_e_v_sigma_qualified": False,
        "physical_positions_certified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    doc = json.loads(args.input.read_text(encoding="utf-8"))
    result = check_topology_view_manifest(doc)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
