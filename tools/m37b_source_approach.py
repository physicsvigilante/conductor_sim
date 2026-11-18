"""M37b: source-faithful workbook p8 Q21 with teacher numerical illustrations.

The printed workbook gives symbolic positive q, R and *prescribed constant*
approach speed v, with a grounded sphere; no printed numerical q/R/v/m.
The authentic answer indices are (1),(3) (original p17 answer key).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from conductor_sim import (
    PrescribedSphereRadialSpec, sample_prescribed_sphere_radial,
    author_prescribed_grounded_sphere,
)
from conductor_sim.prescribed_wire_viewer import (
    render_prescribed_ground_wire, viewer_manifest,
)
from tools.m36a_jee30_audit import ROOT, audit_inventory, load_inventory

PDF = "02_conductors_half_size.pdf"
CASE = "p08-multiple-21"
ANCHOR = "Multiple Correct Q21"
SOURCE = {
    "role": "original_workbook_teacher_example",
    "source_file": PDF,
    "physical_pdf_page": 8,
    "printed_anchor": ANCHOR,
    "case_id": CASE,
    "source_kind": "workbook exercise, NOT an asserted official JEE exam question",
    "statement": ("Positive point charge q approaches a grounded metallic sphere of "
                  "radius R at externally maintained constant small speed v; "
                  "questions ask conventional ground-wire current direction "
                  "and its increasing magnitude."),
    "original_symbolic_givens": ["q > 0", "R > 0", "grounded metal sphere", "constant small inward v > 0"],
    "original_correct_option_indices": [1, 3],
    "numeric_attribution": ("R=0.10m, q=+1nC, d0=0.30m, speed=0.04m/s, "
                            "duration=3.0s and all additional numeric quantities "
                            "are TEACHER-CHOSEN; NONE is a printed numerical given."),
    "original_question_apparatus_count": 1,
    "teacher_hypotheticals_counted_as_original": 0,
}


def make_example():
    problem = author_prescribed_grounded_sphere(
        source_charge_c=1e-9,
        source_position_m=(0., 0., .30),
        sphere_radius_m=.10,
        sphere_name="Sphere A",
    )
    # Original workbook supplies the *type* of protocol, not these numbers.
    return PrescribedSphereRadialSpec(
        problem=problem, radial_speed_m_s=-.04,
        duration_s=3., sample_count=31, surface_guard_m=1e-4,
    )


def run_all(output_dir: Path):
    audit_inventory()  # Retain original PDF SHA/source identity gates.
    case = next((c for c in load_inventory(ROOT/"docs/m36a_jee30_coverage.json")["cases"]
                 if c["id"] == CASE), None)
    if (case is None or case["source"] != PDF or case["pdf_page"] != 8
            or case["printed_anchor"] != ANCHOR or case["candidate_route"] != "partial"):
        raise ValueError("original workbook p8 Q21 anchoring unexpectedly changed")
    spec = make_example()
    trace = sample_prescribed_sphere_radial(spec)
    if not all(row.wire_current_sphere_to_ground_a > 0 for row in trace.samples):
        raise AssertionError("positive source approaching grounded sphere must drive conventional sphere→ground current")
    if not all(b.wire_current_sphere_to_ground_a > a.wire_current_sphere_to_ground_a
               for a,b in zip(trace.samples,trace.samples[1:])):
        raise AssertionError("source constant-speed approach current must increase as distance decreases")
    manifest = viewer_manifest(trace, SOURCE)
    output_dir.mkdir(exist_ok=True,parents=True)
    (output_dir/"p8-ground-current.json").write_text(
        json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (output_dir/"p8-ground-current.html").write_text(
        render_prescribed_ground_wire(trace,source=SOURCE),encoding="utf-8")
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path("output/m37b"))
    args=parser.parse_args()
    data=run_all(args.output_dir)
    print("Original workbook apparatus:",data["source"]["original_question_apparatus_count"])
    print("Teacher numeric example at prescribed times:",len(data["trace"]["samples"]))
    print("Printed source correct options:",data["source"]["original_correct_option_indices"])
    print("Renderer:",args.output_dir/"p8-ground-current.html")
    print("Source-specific independent SI and Chromium validations are separate release gates")


if __name__=="__main__":
    main()
