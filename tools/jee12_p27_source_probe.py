"""Source-anchored, developer-only p27 Illustration 4 review of two thin shells.

Uses the *existing* released M36c native replay and separate reference. Checks
additional signed/translated authentic two-shell states against directly
written Gauss-law face-charge equations, without inventing original numerical
inputs or claiming a new original JEE solution.
"""
from __future__ import annotations

import argparse
import json
from hashlib import sha256
from math import isclose, pi
from pathlib import Path
from tempfile import TemporaryDirectory

from conductor_sim import (
    EPSILON_0, ElectrostaticProblem, FixedNetCharge, NestedShellBackend,
    PointCharge, SphericalShell,
)
from tools.m36c_jee_conductor_replay import _p27_original_A_B
from tools.m36c_jee_conductor_reference import original_p27_AB

SOURCE_DIGEST = "8d0d9b5cf821219058cfc4b7685a39b777e460944df44c3f9cffcb68157a60e5"
K = 1.0 / (4.0 * pi * EPSILON_0)


def _check_signed_case(q: float, q1: float, q2: float, origin: tuple[float, float, float]) -> dict:
    a, b, thickness, r, r1 = .10, .20, 1e-6, .04, .15
    ai, bi = a - thickness, b - thickness
    shells = (
        SphericalShell(ai, a, FixedNetCharge(q1), center_m=origin, name="inner"),
        SphericalShell(bi, b, FixedNetCharge(q2), center_m=origin, name="outer"),
    )
    world_a = (origin[0] + r, origin[1], origin[2])
    world_b = (origin[0] + r1, origin[1], origin[2])
    physical = ElectrostaticProblem(
        conductors=shells, charges=(PointCharge(q, origin),)
    )
    backend = NestedShellBackend()
    if not backend.supports(physical):
        raise AssertionError("valid source-concentric shells unexpectedly refused")
    result = backend.solve(physical)

    # First compute the IDEAL ZERO-THICKNESS equations from the actual figure.
    # The source specifies symbolic q,q1,q2,a,b,r,r1, not these SI values.
    printed_va = K * (q / r + q1 / a + q2 / b)
    printed_vb = K * ((q + q1) / r1 + q2 / b)

    # Independently enforce Gauss charges on all four actual finite metal faces
    # and evaluate the resulting Coulomb shell potentials in the vacuum regions.
    # A is INSIDE the first cavity. B is in the intershell vacuum gap.
    finite_va = K * (q / r - q / ai + (q + q1) / a
                     - (q + q1) / bi + (q + q1 + q2) / b)
    finite_vb = K * ((q + q1) / r1 - (q + q1) / bi
                     + (q + q1 + q2) / b)
    actual_va = result.potential_at(world_a)
    actual_vb = result.potential_at(world_b)
    for label, actual, ref in (
        ("physical finite V(A)", actual_va, finite_va),
        ("physical finite V(B)", actual_vb, finite_vb),
        ("inner face -q", result.conductor_by_name("inner").inner_surface_charge_c, -q),
        ("inner outer face q+q1", result.conductor_by_name("inner").outer_surface_charge_c, q + q1),
        ("outer inner face -(q+q1)", result.conductor_by_name("outer").inner_surface_charge_c, -(q + q1)),
        ("outer outer face q+q1+q2", result.conductor_by_name("outer").outer_surface_charge_c, q + q1 + q2),
    ):
        if not isclose(actual, ref, rel_tol=4e-13, abs_tol=2e-13):
            raise AssertionError(f"{label}: native {actual!r} vs Gauss {ref!r}")
    return dict(
        teacher_q_c=q, teacher_q1_c=q1, teacher_q2_c=q2,
        teacher_origin_m=list(origin), modelled_metal_thickness_m=thickness,
        native_finite_metal_va_v=actual_va,
        native_finite_metal_vb_v=actual_vb,
        independent_finite_metal_va_v=finite_va,
        independent_finite_metal_vb_v=finite_vb,
        original_zero_thickness_symbolic_va_instantiated_v=printed_va,
        original_zero_thickness_symbolic_vb_instantiated_v=printed_vb,
        finite_minus_ideal_va_v=finite_va - printed_va,
        finite_minus_ideal_vb_v=finite_vb - printed_vb,
    )


def check(root: Path, output_dir: Path | None = None) -> dict:
    frozen = root / "POTENTIAL_CONDUCTORS1.pdf"
    if sha256(frozen.read_bytes()).hexdigest() != SOURCE_DIGEST:
        raise AssertionError("source PDF bytes differ from audited repository original")
    historical = json.loads((root / "docs/m36a_jee30_coverage.json").read_text(encoding="utf-8"))
    case = next(x for x in historical["cases"] if x["id"] == "p27-cavity-two-shells")
    if (case["pdf_page"] != 27 or case["printed_anchor"] != "Illustration 4"
            or case["source"] != frozen.name):
        raise AssertionError("original printed source case identity changed")

    def _existing_replay(folder: Path) -> dict:
        folder.mkdir(exist_ok=True, parents=True)
        card = _p27_original_A_B(folder)
        reference = original_p27_AB(card)
        html = (folder / card["viewer_file"]).read_text(encoding="utf-8")
        for marker in (
            'id="p27-original-A-B"', "Printed ideal-thin formula comparison",
            "Actual finite-metal V(A)", "V(B)", "SOURCE ORIGINAL",
            "TEACHER EXTENSION ONLY", "1-micrometre metal thickness",
        ):
            if marker not in html:
                raise AssertionError("missing original-source vs teacher visual disclosure: " + marker)
        original_state, teacher_state = card["authored_states"]
        if original_state["charges"][0]["position_m"] != [0., 0., 0.]:
            raise AssertionError("original printed charge is not central")
        if teacher_state["charges"][0]["position_m"] != [0., 0., .02]:
            raise AssertionError("teacher extension has changed without attribution")
        if card["original_asked_observables"] != ["V(A)", "V(B)"]:
            raise AssertionError("original source answer contract changed")
        return dict(
            original_source_native_reference=reference,
            source_replay_visible_html_filename=card["viewer_file"],
            original_static_state_charge_position_m=[0., 0., 0.],
            nonoriginal_teacher_state_position_m=[0., 0., .02],
            two_original_observables_visible=True,
        )

    if output_dir is None:
        with TemporaryDirectory(prefix="jee12_p27_") as temp:
            existing = _existing_replay(Path(temp))
    else:
        existing = _existing_replay(output_dir)
    signed = [
        _check_signed_case(+1e-9, +2e-9, +3e-9, (0., 0., 0.)),
        _check_signed_case(-1.5e-9, +2.5e-9, -0.8e-9, (.1, -.14, .07)),
        _check_signed_case(+.8e-9, -2.1e-9, +.4e-9, (-.21, .03, -.08)),
    ]
    return dict(
        schema="conductor_sim.jee12_p27_developer_source_probe.v1",
        status="DEVELOPER NATIVE/REFERENCE PROBE, NOT NEW INDEPENDENT PASS OR EXTERNAL HTML AUTHENTICATION",
        source=dict(pdf=frozen.name, sha256=SOURCE_DIGEST, page=27,
                    original_anchor="Illustration 4", provenance="notes illustration, not an official exam claim",
                    actual_original_geometry="central point q, two ideal thin concentric shells q1/q2",
                    originally_asked=["V(A)", "V(B)"],
                    original_symbolic_equations=["kq/r + kq1/a + kq2/b",
                                                  "k(q+q1)/r1 + kq2/b"]),
        existing_source_backed_replay=existing,
        teacher_finite_thickness_signed_cases=signed,
        original_textbook_was_not_numeric=True,
        browser_against_this_evaluation_newly_independently_checked=False,
        original_problem_answer_independently_reviewed_for_jee12=False,
        physical_time_or_charge_motion=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Write authentic source-specific HTML here; otherwise only examine temp output")
    parser.add_argument("--output", type=Path, default=None,
                        help="Save developer JSON evidence to this path")
    args = parser.parse_args()
    data = check(Path(__file__).resolve().parents[1], args.output_dir)
    encoded = json.dumps(data, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
