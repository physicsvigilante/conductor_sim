"""Developer-only JEE-12 source-faithful p26 solid-sphere native probe.

The original notes p26 Illustration 1 gives symbolic R and Q, asking
V(R/2) and V(3R/2). Numeric choices are teacher examples.
This is NOT independent review or authentication of edited HTML.
"""
from __future__ import annotations

from hashlib import sha256
from math import isclose, pi
from pathlib import Path
import json

from conductor_sim import ElectrostaticProblem, FixedNetCharge, MixedRadialBackend, SolidSphere

PDF_SHA = "8d0d9b5cf821219058cfc4b7685a39b777e460944df44c3f9cffcb68157a60e5"
K = 1 / (4 * pi * 8.8541878128e-12)


def check(root: Path) -> dict:
    source_pdf = root / "POTENTIAL_CONDUCTORS1.pdf"
    if sha256(source_pdf.read_bytes()).hexdigest() != PDF_SHA:
        raise AssertionError("frozen source PDF byte identity mismatch")
    history = json.loads((root / "docs/m36a_jee30_coverage.json").read_text(encoding="utf-8"))
    row = next(x for x in history["cases"] if x["id"] == "p26-conductor-potential")
    if (row["pdf_page"] != 26 or row["source"] != source_pdf.name
            or row["printed_anchor"] != "Illustration 1"):
        raise AssertionError("historical source anchor mismatch")
    backend = MixedRadialBackend()
    results = []
    for R, Q, origin in (
        (.1, +1e-9, (0., 0., 0.)),
        (.12, -2e-9, (.025, -.01, .02)),
        (.04, +3e-10, (-.1, .15, -.04)),
    ):
        sphere = SolidSphere(R, FixedNetCharge(Q), center_m=origin, name="Sphere A")
        problem = ElectrostaticProblem(conductors=(sphere,))
        if not backend.supports(problem):
            raise AssertionError("true solid refused")
        solution = backend.solve(problem)
        c = sphere.center_m
        inside = solution.potential_at((c[0] + R/2, c[1], c[2]))
        outside = solution.potential_at((c[0] + 1.5*R, c[1], c[2]))
        expected_inside = K * Q / R
        expected_outside = 2 * K * Q / (3*R)
        if not isclose(inside, expected_inside, rel_tol=3e-14, abs_tol=1e-14):
            raise AssertionError("native R/2 differs from source equation")
        if not isclose(outside, expected_outside, rel_tol=3e-14, abs_tol=1e-14):
            raise AssertionError("native 3R/2 differs from source equation")
        if sphere.inner_radius_m != 0 or len(problem.conductors) != 1:
            raise AssertionError("hollow-shell proxy was used")
        results.append({
            "teacher_radius_m": R, "teacher_charge_c": Q, "teacher_center_m": list(c),
            "native_v_half_r": inside, "native_v_three_half_r": outside,
            "independently_expected_v_half_r": expected_inside,
            "independently_expected_v_three_half_r": expected_outside,
        })
    return {
        "schema": "conductor_sim.jee12_p26_developer_source_probe.v1",
        "status": "DEVELOPER PHYSICAL PRE-REVIEW, NOT INDEPENDENT PASS",
        "original_source": {
            "pdf": source_pdf.name, "sha256": PDF_SHA, "pdf_page": 26,
            "printed_anchor": "Illustration 1",
            "source_provenance": "notes illustration, not automatically official JEE",
            "original_parameters": "symbolic R and Q",
            "original_asked": ["V(R/2)", "V(3R/2)"],
            "original_answer_symbolic": ["kQ/R", "2kQ/(3R)"],
        },
        "actual_geometry": "one isolated SOLID conducting sphere; no inner conductor face",
        "teacher_numeric_cases": results,
        "independent_reference": "Gauss law and solid-metal equipotential equations",
        "original_numeric_values_claimed": False,
        "original_problem_answer_verified_by_independent_reviewer": False,
        "browser_case_newly_checked": False,
        "physical_elapsed_time": False,
    }


if __name__ == "__main__":
    print(json.dumps(check(Path(__file__).resolve().parents[1]), indent=2, allow_nan=False))
