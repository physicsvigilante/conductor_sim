"""Independent stdlib-only physics/provenance checker for two JEE walkthroughs.

No conductor_sim or developer native-runner imports. Check detached recorded SI
against separately derived source equations; do not authenticate arbitrary HTML.
"""
from __future__ import annotations

import argparse
import json
import math
from hashlib import sha256
from pathlib import Path

SCHEMA = "conductor_sim.jee12_original_walkthrough.v1"
PDF = "02_conductors_half_size.pdf"
PDF_SHA = "9dff3c02a9f1848d2c3d99cb1a6e7a44dc7f5a51f7a3a2b8cc92d10ce1fbca5d"
ELECTRON_C = 1.602176634e-19
EPSILON0 = 8.8541878128e-12
K = 1 / (4 * math.pi * EPSILON0)


def demand(value: bool, label: str):
    if not value:
        raise AssertionError("independent walkthrough verification: " + label)


def number(value, label: str, expected: float, *, rtol: float = 3e-11, atol: float = 1e-16):
    demand(type(value) in (int, float) and math.isfinite(value), label + " numeric SI")
    demand(math.isclose(value, expected, rel_tol=rtol, abs_tol=atol),
           label + " independent SI mismatch " + repr((value, expected)))


def exact(value, expected, label: str):
    demand(type(value) is type(expected) and value == expected, label + " source/provenance mismatch")


def source(record: dict, *, case: str, page: int, anchor: str):
    demand(type(record) is dict and record.get("schema") == SCHEMA, "versioned schema")
    meta = record["source"]
    exact(meta, {
        "pdf": PDF, "sha256": PDF_SHA, "pdf_page_index": page,
        "anchor": anchor, "case_id": case,
        "provenance": "original printed workbook exercise; not automatically official JEE exam",
    }, case + " immutable source metadata")


def verify_work(r: dict):
    source(r, case="p01-single-08", page=1, anchor="Single Correct Q8")
    exact(r["source_given"], {
        "uniform_field_x_v_m": 10.0, "electron_charge_c": -ELECTRON_C,
        "ab_m": 5.0, "transverse_bc_m": 3.0,
    }, "work source givens")
    exact(r["derived"], {
        "horizontal_displacement_m": 4.0, "start_m": [0.0, 0.0, 0.0],
        "end_m": [4.0, 3.0, 0.0],
    }, "work derived physical endpoints")
    exact(r["original_asked"], "work done by us on an electron from A to B",
          "original work question meaning")
    exact(r["mode"], "static endpoint work; no particle trajectory or elapsed time",
          "work static qualification")
    exact(r["printed_answer"], {"option": 2, "external_work_eV": 40.0},
          "original option/answer")
    expected_field = -ELECTRON_C * 10.0 * 4.0
    n = r["native"]
    demand(set(n) == {"start_potential_v", "end_potential_v", "field_work_j",
                       "external_work_j"}, "work native ledger completeness")
    number(n["start_potential_v"], "original V(A)", 0.0)
    number(n["end_potential_v"], "original V(B)", -40.0)
    number(n["field_work_j"], "original field work", expected_field, atol=1e-30)
    number(n["external_work_j"], "original work by us", -expected_field, atol=1e-30)
    t = r["teacher_hypothetical"]
    exact(t["role"], "TEACHER HYPOTHETICAL, not printed givens",
          "original/teacher work role")
    number(t["changed_field_x_v_m"], "teacher E", 20.0)
    number(t["native_field_work_j"], "native teacher field work",
           -ELECTRON_C * 20.0 * 4.0, atol=1e-30)
    number(t["native_external_work_j"], "native teacher external work",
           ELECTRON_C * 20.0 * 4.0, atol=1e-30)


def verify_energy(r: dict):
    source(r, case="p03-single-27", page=3, anchor="Single Correct Q27")
    exact(r["mode"], "two static equilibria; not current, heat-rate or time evolution",
          "connected-sphere static scope")
    exact(r["original_asked"],
          "heat generated on joining the solid sphere and thin shell, inferred from loss of field energy",
          "energy question meaning")
    exact(r["source_given"], {
        "solid_radius_m": 0.1, "shell_radius_m": 0.2,
        "initial_solid_charge_c": 20e-6, "initial_shell_net_charge_c": 0.0,
        "wire_connection": "floating charge-conserving",
    }, "original solid and thin shell source facts")
    qualified = r["teacher_model_qualification"]
    demand(qualified["actual_inner_body"] == "SOLID conducting sphere, no interior cavity",
           "not a hollow-sphere proxy")
    exact(qualified["line_width_on_svg_is_not_physical_thickness"], True,
          "disclose exaggerated thin-shell stroke")
    number(qualified["finite_shell_inner_radius_m"], "finite shell inner", .199999)
    number(qualified["finite_shell_outer_radius_m"], "finite shell outer", .2)
    number(qualified["metal_thickness_m"], "finite metal thickness", 1e-6, atol=1e-16)
    rsolid = 0.1
    inner, outer, q = .199999, .2, 20e-6
    # Entire gap+outside electrostatic field, no current or heat-flow time.
    gap = 0.5 * K * q*q * (1 / rsolid - 1 / inner)
    outside = 0.5 * K * q*q / outer
    ideal = 0.5 * K * q*q * (1 / rsolid - 1 / outer)
    n = r["native"]
    demand(set(n) == {"before", "after", "energy_before_j", "energy_after_j",
                       "energy_released_j"}, "complete native static energy ledger")
    for label, actual, expected in (
        ("before energy", n["energy_before_j"], gap + outside),
        ("after energy", n["energy_after_j"], outside),
        ("released energy", n["energy_released_j"], gap),
    ):
        number(actual, label, expected, atol=1e-11)
    vb = K*q/outer
    vs = vb + K*q*(1/rsolid - 1/inner)
    for name, state, correct in (
        ("before", n["before"], {
            "solid_net_charge_c": q, "solid_outer_face_c": q,
            "shell_net_charge_c": 0.0, "shell_inner_face_c": -q,
            "shell_outer_face_c": q, "solid_potential_v": vs,
            "shell_potential_v": vb,
        }),
        ("after", n["after"], {
            "solid_net_charge_c": 0.0, "solid_outer_face_c": 0.0,
            "shell_net_charge_c": q, "shell_inner_face_c": 0.0,
            "shell_outer_face_c": q, "solid_potential_v": vb,
            "shell_potential_v": vb,
        }),
    ):
        demand(set(state) == set(correct), name + " real conductor/face record")
        for key, expected in correct.items():
            number(state[key], name + " " + key, expected,
                   atol=1e-10 if key.endswith("_v") else 1e-16)
    p = r["printed_answer"]
    exact(p["option"], 2, "printed energy choice")
    number(p["heat_j_rounded"], "printed heat energy J", 9.0)
    number(p["ideal_thin_limit_j"], "original ideal thin shell limit", ideal, atol=1e-11)
    t = r["teacher_hypothetical"]
    exact(t["role"], "TEACHER HYPOTHETICAL, not printed givens",
          "original/teacher energy role")
    number(t["changed_initial_solid_charge_c"], "teacher charge", 10e-6)
    number(t["native_energy_released_j"], "teacher rerun field energy",
           gap * 0.25, atol=1e-11)


def verify(directory: Path, root: Path | None = None) -> dict:
    expected = {
        "p1-external-work": ("p01-single-08", verify_work),
        "p3-solid-shell-energy": ("p03-single-27", verify_energy),
    }
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    exact(manifest, {
        "schema": "conductor_sim.jee12_walkthrough_index.v1",
        "original_complete_answers_certified_in_frozen_jee12_cohort": 0,
        "pages": [
            {"case_id":"p01-single-08", "json":"p1-external-work.json",
             "html":"p1-external-work.html"},
            {"case_id":"p03-single-27", "json":"p3-solid-shell-energy.json",
             "html":"p3-solid-shell-energy.html"},
        ],
    }, "exactly two original exercise views and honest coverage")
    if root is not None:
        demand(sha256((root / PDF).read_bytes()).hexdigest() == PDF_SHA,
               "actual frozen original raw PDF bytes")
    for name, (case, check) in expected.items():
        payload = json.loads((directory / (name + ".json")).read_text(encoding="utf-8"))
        demand(payload["source"]["case_id"] == case, name + " source identity")
        check(payload)
        html = (directory / (name + ".html")).read_text(encoding="utf-8")
        demand("<svg " in html and "<script" not in html and "http://" not in html
               and "https://" not in html, "offline SVG static source output")
        demand("TEACHER HYPOTHETICAL" in html and "Original requested answer" in html,
               "visible original-versus-teacher source labels")
        demand("not cryptographically authenticated" in html, "HTML trust boundary")
    return {"schema": "conductor_sim.jee12_walkthrough_reference.v1",
            "verified_original_source_cases": list(expected),
            "independent_si": "PASS",
            "chromium_still_a_separate_gate": True,
            "frozen_jee12_original_answer_count_changed": False}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(verify(args.input_dir, root), indent=2) + "\n")
