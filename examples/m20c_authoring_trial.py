"""M20c: independent of Ring A/B, curated non-Ring JEE authoring trial.

Six new fixed-source systems plus six expected refusal/boundary requests.
No source in this module is a conducting object; all are held and nonresponding.
This evaluates existing authoring and comparison interfaces, not a new solver.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from math import cos, hypot, isclose, pi, sqrt
from pathlib import Path

from conductor_sim import (
    EPSILON_0, FixedPointField, PointCharge, PrescribedField,
    TestChargeMotionSpec, UniformDiskAxisField, UniformElectricField,
    UniformRodField, UniformShellField, UniformSphereField,
    prepare_scenario_comparison, write_scenario_comparison,
)
from conductor_sim.scenarios import (
    PrescribedScenario, revise_scenario, sample_scenario_viewer,
)
from conductor_sim.solver import UnsupportedProblem

PAIRED = ("U1", "V1", "R1", "P1", "S1", "D1")
NEGATIVE = ("N1", "N2", "N3", "N4", "N5", "N6")
K = 1.0 / (4.0 * pi * EPSILON_0)
VIEW = dict(visual_frames=9, profile_samples=41, resolution=11,
            extent_m=0.5, axis_orientation="vertical")

NOTES = {
    "U1": "Double the particle mass in the same uniform field.",
    "V1": "Increase the fixed volume sphere total charge by 50 percent.",
    "R1": "Increase the held rod charge from 1 to 1.7 nanocoulombs.",
    "P1": "Increase the fixed point source charge from 2 to 3 nanocoulombs.",
    "S1": "Double fixed thin-shell charge; do not add conductor induction.",
    "D1": "Increase the fixed disc radius from 0.08 to 0.11 metres.",
}


def baseline(case: str) -> PrescribedScenario:
    """Author a fresh physical system; no Ring A/B data or helper imports."""
    if case == "U1":
        sources = (UniformElectricField((0., 0., -100.)),)
        bindings = (("field:U1", "Applied uniform E"),)
        p, q, mass, v, duration, scale = (0., 0., 0.), 1e-9, 1e-6, (.2, 0., 0.), .3, .1
        problem = "Uniform E deflects a charged particle launched across field lines."
    elif case == "V1":
        sources = (UniformSphereField(1e-9, .1),)
        bindings = (("sphere:V1", "Fixed volume sphere"),)
        p, q, mass, v, duration, scale = (.02, 0., 0.), -1e-9, 1e-6, (0., 0., 0.), .4, .1
        problem = "Oppositely charged test particle oscillates inside uniform volume charge."
    elif case == "R1":
        sources = (UniformRodField(1e-9, (-.08, 0., 0.), (.08, 0., 0.)),)
        bindings = (("rod:R1", "Fixed finite rod"),)
        p, q, mass, v, duration, scale = (0., 0., .15), 1e-9, 1e-6, (.02, 0., 0.), .15, .1
        problem = "Two-dimensional flyby of a finite fixed nonconducting line charge."
    elif case == "P1":
        sources = (FixedPointField(2e-9), UniformElectricField((0., 0., -10.)))
        bindings = (("point:P1", "Fixed point source"), ("field:P1", "Applied uniform E"))
        p, q, mass, v, duration, scale = (.12, 0., .02), 1e-9, 1e-6, (0., 0., .15), .15, .1
        problem = "Two independently held sources superpose during 2D test-charge motion."
    elif case == "S1":
        sources = (UniformShellField(4e-9, .12),)
        bindings = (("shell:S1", "Fixed thin shell"),)
        p, q, mass, v, duration, scale = (.01, 0., 0.), 1e-9, 1e-6, (0., 0., .02), .2, .1
        problem = "A nonconducting uniform surface-charge shell has zero interior field."
    elif case == "D1":
        sources = (UniformDiskAxisField(2e-9, .08),)
        bindings = (("disc:D1", "Fixed charged disc"),)
        p, q, mass, v, duration, scale = (0., 0., .15), 1e-9, 1e-6, (0., 0., -.03), .15, .1
        problem = "Axial motion near a held uniformly charged thin disc; no off-axis field."
    else:
        raise ValueError("unknown held-out positive card")
    spec = TestChargeMotionSpec(
        PrescribedField(sources), PointCharge(q, p), mass, v, duration, .01,
        plane_normal=(0., 1., 0.), length_scale_m=scale, source_guard_m=1e-5,
    )
    return PrescribedScenario(
        "m20c-" + case + "-base", spec, bindings,
        "Teacher-authored M20c " + case + ": " + problem,
    )


def pair(case: str):
    parent = baseline(case)
    if case == "U1":
        patch = dict(motion_changes={"mass_kg": 2e-6})
    elif case == "V1":
        patch = dict(source_changes={"sphere:V1": {"charge_c": 1.5e-9}})
    elif case == "R1":
        patch = dict(source_changes={"rod:R1": {"charge_c": 1.7e-9}})
    elif case == "P1":
        patch = dict(source_changes={"point:P1": {"charge_c": 3e-9}})
    elif case == "S1":
        patch = dict(source_changes={"shell:S1": {"charge_c": 8e-9}})
    elif case == "D1":
        patch = dict(source_changes={"disc:D1": {"radius_m": .11}})
    else:
        raise ValueError("unknown positive card")
    derived = revise_scenario(
        parent, scenario_id="m20c-" + case + "-revised", note=NOTES[case], **patch,
    )
    return parent, derived


def bundles_and_manifest(case: str):
    a, b = pair(case)
    original_record = a.as_dict()
    left = sample_scenario_viewer(a, **VIEW)
    right = sample_scenario_viewer(b, **VIEW)
    manifest = prepare_scenario_comparison(
        left, right, baseline_id=a.scenario_id, variation_id=b.scenario_id,
        baseline_title="Baseline " + case, variation_title="Teacher revision " + case,
        baseline_provenance=a.viewer_provenance,
        variation_provenance=b.viewer_provenance,
    )
    if a.as_dict() != original_record:
        raise AssertionError("baseline model was changed by simulation")
    if left["prescribed_motion"]["termination"] != "duration" or right["prescribed_motion"]["termination"] != "duration":
        raise AssertionError("paired scenario has a censored trajectory")
    if manifest["time_interpolation"] is not False:
        raise AssertionError("comparison interpolates physical time")
    expected = json.loads(json.dumps([asdict(c) for c in b.changes], allow_nan=False))
    if manifest["revisions"] != expected:
        raise AssertionError("the independent comparison revision ledger disagrees")
    return a, b, left, right, manifest


def _check(actual, expected, *, rel=1e-7, absolute=1e-8):
    if not isclose(actual, expected, rel_tol=rel, abs_tol=absolute):
        raise AssertionError("independent reference mismatch: actual %.12g, expected %.12g" %
                             (actual, expected))
    return abs(actual - expected)


def scientific_reference(case, models, bundles):
    """External closed-form or quadrature checks, not a call to field providers."""
    a, b = models
    left, right = bundles
    errors = []
    if case == "U1":
        for model, bundle in ((a, left), (b, right)):
            q, m = model.spec.particle.charge_c, model.spec.mass_kg
            for row in bundle["prescribed_motion"]["samples"]:
                t = row["time_s"]
                errors.extend((
                    _check(row["position_m"][0], .2*t, absolute=1e-10),
                    _check(row["position_m"][2], -.5*(100.*q/m)*t*t, absolute=1e-10),
                    _check(row["velocity_m_s"][0], .2, absolute=1e-10),
                    _check(row["velocity_m_s"][2], -(100.*q/m)*t, absolute=1e-10),
                ))
        claim = "Exact constant-acceleration transverse parabola (m and m/s)"
    elif case == "V1":
        for model, bundle in ((a, left), (b, right)):
            q = abs(model.spec.particle.charge_c)
            Q = model.spec.field.sources[0].charge_c
            m = model.spec.mass_kg
            radius = model.spec.field.sources[0].radius_m
            omega = sqrt(K * q * Q / (m * radius**3))
            for row in bundle["prescribed_motion"]["samples"]:
                errors.append(_check(row["position_m"][0],
                                     .02*cos(omega*row["time_s"]), absolute=4e-7))
                if hypot(*row["position_m"]) >= radius:
                    raise AssertionError("interior sphere reference crossed its domain")
        claim = "Exact interior harmonic oscillator and certified interior positions (m)"
    elif case == "R1":
        # Independent Coulomb midpoint integral at the first particle position.
        n = 4000
        for model, bundle in ((a, left), (b, right)):
            source = model.spec.field.sources[0]
            z = model.spec.particle.position_m[2]
            L = source.end_m[0] - source.start_m[0]
            x0 = source.start_m[0]
            integral_V = 0.
            integral_Ez = 0.
            for j in range(n):
                x = x0 + L*(j+.5)/n
                r = hypot(x, z)
                integral_V += 1/r
                integral_Ez += z/r**3
            predicted_V = K*source.charge_c*integral_V/n
            predicted_Ez = K*source.charge_c*integral_Ez/n
            row = bundle["prescribed_motion"]["samples"][0]
            errors.append(_check(row["potential_v"], predicted_V, rel=4e-7, absolute=1e-5))
            errors.append(_check(row["electric_field_v_m"][2], predicted_Ez,
                                 rel=4e-7, absolute=1e-4))
        claim = "Independent 4000-panel Coulomb rod quadrature (V and V/m)"
    elif case == "P1":
        for model, bundle in ((a, left), (b, right)):
            p = model.spec.particle.position_m
            Q = model.spec.field.sources[0].charge_c
            r = hypot(*p)
            expected_E = (K*Q*p[0]/r**3, 0., K*Q*p[2]/r**3 - 10.)
            expected_V = K*Q/r + 10.*p[2]
            row = bundle["prescribed_motion"]["samples"][0]
            for actual, expected in zip(row["electric_field_v_m"], expected_E):
                errors.append(_check(actual, expected, rel=1e-11, absolute=1e-10))
            errors.append(_check(row["potential_v"], expected_V,
                                 rel=1e-11, absolute=1e-10))
        claim = "Independent Coulomb point charge plus uniform-field E and V"
    elif case == "S1":
        delta_Q = b.spec.field.sources[0].charge_c - a.spec.field.sources[0].charge_c
        expected_dU = K*a.spec.particle.charge_c*delta_Q/a.spec.field.sources[0].radius_m
        for ra, rb in zip(left["prescribed_motion"]["samples"], right["prescribed_motion"]["samples"]):
            for row in (ra, rb):
                for comp in row["electric_field_v_m"]:
                    errors.append(_check(comp, 0., absolute=1e-13))
            for item in ("position_m", "velocity_m_s"):
                for x, y in zip(ra[item], rb[item]):
                    errors.append(_check(x, y, absolute=1e-11))
            errors.append(_check(rb["potential_energy_j"]-ra["potential_energy_j"],
                                 expected_dU, rel=1e-11, absolute=1e-13))
        claim = "Zero interior field and identical motion, with analytic q delta V"
    elif case == "D1":
        for model, bundle in ((a, left), (b, right)):
            source = model.spec.field.sources[0]
            z = model.spec.particle.position_m[2]
            h = hypot(z, source.radius_m)
            expected_V = 2*K*source.charge_c/(h+z)
            expected_E = 2*K*source.charge_c/(h*(h+z))
            row = bundle["prescribed_motion"]["samples"][0]
            errors.append(_check(row["potential_v"], expected_V,
                                 rel=1e-11, absolute=1e-10))
            errors.append(_check(row["electric_field_v_m"][2], expected_E,
                                 rel=1e-11, absolute=1e-10))
            if any(s["position_m"][2] <= 0 for s in bundle["prescribed_motion"]["samples"]):
                raise AssertionError("disc trajectory crossed the ideal surface")
        claim = "Exact positive-side disc axial potential and electric field"
    else:
        raise ValueError("unknown science reference card")
    return dict(reference=claim, max_abs_reference_error=max(errors, default=0.),
                checked_scalar_observations=len(errors))


def _rejection(card):
    if card == "N1":
        parent = baseline("D1")
        return revise_scenario(parent, scenario_id="m20c-N1", note="Off-axis disc launch.",
                               motion_changes={"initial_velocity_m_s": (.01, 0., -.03)})
    if card == "N2":
        parent = baseline("S1")
        return revise_scenario(parent, scenario_id="m20c-N2", note="Shell surface start.",
                               particle_changes={"position_m": (.12, 0., 0.)})
    if card == "N3":
        parent = baseline("U1")
        return revise_scenario(parent, scenario_id="m20c-N3", note="Add an unbound source.",
                               source_changes={"unbound": {"charge_c": 1e-9}})
    if card == "N4":
        parent = baseline("V1")
        return revise_scenario(parent, scenario_id="m20c-N4", note="Impose conductor grounding.",
                               source_changes={"sphere:V1": {"constraint": "Grounded"}})
    raise ValueError("unknown expected rejection")


def _grade_refusal(card):
    expected = {
        "N1": (UnsupportedProblem, "axial"),
        "N2": (ValueError, "guard"),
        "N3": (ValueError, "unknown stable"),
        "N4": (ValueError, "unsupported property"),
    }[card]
    try:
        _rejection(card)
    except Exception as exc:
        if not isinstance(exc, expected[0]) or expected[1] not in str(exc):
            return dict(card=card, status="unexpected-exception",
                        exception=type(exc).__name__, reason=str(exc))
        return dict(card=card, status="refused",
                    exception=type(exc).__name__, reason=str(exc))
    return dict(card=card, status="unexpected-acceptance",
                reason="An out-of-scope request was incorrectly admitted")


def _grade_incomparable(card):
    if card == "N5":
        parent = baseline("U1")
        derived = revise_scenario(
            parent, scenario_id="m20c-N5", note="Change actual initial position.",
            particle_changes={"position_m": (.01, 0., 0.)},
        )
    elif card == "N6":
        spec = TestChargeMotionSpec(
            PrescribedField((FixedPointField(-1e-9),)),
            PointCharge(1e-9, (.1, 0., 0.)), 1., (-1., 0., 0.), .2, .02,
            plane_normal=(0., 1., 0.), length_scale_m=.1, source_guard_m=.01,
        )
        parent = PrescribedScenario(
            "m20c-N6-base", spec, (("point:N6", "Attracting point source"),),
            "Genuine attractive approach ends at the prescribed singularity guard.",
        )
        derived = revise_scenario(
            parent, scenario_id="m20c-N6-revised",
            note="Double the test mass while retaining the guard.",
            motion_changes={"mass_kg": 2.},
        )
    else:
        raise ValueError("unknown admitted-but-incomparable case")
    left, right = (sample_scenario_viewer(s, **VIEW) for s in (parent, derived))
    terms = [left["prescribed_motion"]["termination"],
             right["prescribed_motion"]["termination"]]
    expected_terms = (["duration", "duration"] if card == "N5"
                      else ["source-guard", "source-guard"])
    if terms != expected_terms:
        return dict(card=card, status="unexpected-termination", terminations=terms)
    try:
        prepare_scenario_comparison(
            left, right, baseline_id=parent.scenario_id,
            variation_id=derived.scenario_id,
        )
    except ValueError as exc:
        expected = ("slice",) if card == "N5" else ("time grids", "complete duration", "frame times")
        if not any(token in str(exc) for token in expected):
            return dict(card=card, status="unexpected-exception",
                        exception=type(exc).__name__, reason=str(exc))
        return dict(card=card, status="admitted-not-comparable",
                    reason=str(exc), terminations=terms,
                    physical_revision=[asdict(c) for c in derived.changes],
                    parent_unchanged=parent.revision_id == parent.as_dict()["revision_id"])
    return dict(card=card, status="unexpected-comparison",
                reason="M20a accepted physical data it should reject")


def grade_trial():
    """All predeclared cases, including expected refusals. Never drop failures."""
    results = []
    for case in PAIRED:
        try:
            a, b, left, right, manifest = bundles_and_manifest(case)
            science = scientific_reference(case, (a, b), (left, right))
            results.append(dict(card=case, status="paired",
                                baseline_revision_id=a.revision_id,
                                revision_id=b.revision_id,
                                parent_revision_id=b.parent_revision_id,
                                source_ids=manifest["source_ids"],
                                changes=manifest["revisions"],
                                physical_samples=manifest["physical_sample_count"],
                                saved_visual_frames=len(manifest["frame_times_s"]),
                                reference=science,
                                endpoint_m=[left["prescribed_motion"]["samples"][-1]["position_m"],
                                            right["prescribed_motion"]["samples"][-1]["position_m"]],
                                max_energy_drift_j=[left["prescribed_motion"]["max_abs_energy_drift_j"],
                                                    right["prescribed_motion"]["max_abs_energy_drift_j"]]))
        except Exception as exc:
            results.append(dict(card=case, status="unexpected-exception",
                                exception=type(exc).__name__, reason=str(exc)))
    for case in NEGATIVE:
        try:
            results.append(_grade_refusal(case) if case in NEGATIVE[:4]
                           else _grade_incomparable(case))
        except Exception as exc:
            results.append(dict(card=case, status="unexpected-exception",
                                exception=type(exc).__name__, reason=str(exc)))
    expected = (["paired"]*6 + ["refused"]*4 + ["admitted-not-comparable"]*2)
    actual = [r["status"] for r in results]
    return dict(schema="conductor_sim.m20c_authoring_trial.v1",
                scope="Teacher-authored fixed nonresponding sources; no conductor induction",
                design="Predeclared curated cross-source questions; not an independent blinded sample",
                charter="docs/m20c_trial_charter.md",
                expected_cards=list(PAIRED + NEGATIVE),
                case_count=len(results), case_results=results,
                status_counts={k:actual.count(k) for k in set(actual)},
                accepted=actual == expected,
                dimensions=dict(
                    interpretation="Explicit teacher hypotheticals with qualified source types",
                    physical_correctness="Independent closed forms or quadrature plus existing solver validation",
                    simulation_coverage="Six complete pairs, four expected refusals, two valid but incomparable",
                    educational_usefulness="Source identities and change ledgers survive model revisions",
                    textbook_answer_completeness="Not assessed; not textbook question replicas"))


def render_case(case, output: Path):
    a, b, left, right, manifest = bundles_and_manifest(case)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_scenario_comparison(
        left, right, output, title="M20c cross-source " + case + " teaching comparison",
        baseline_id=a.scenario_id, variation_id=b.scenario_id,
        baseline_title="Baseline " + case, variation_title="Teacher revision " + case,
        baseline_provenance=a.viewer_provenance,
        variation_provenance=b.viewer_provenance,
    )
    sibling = output.with_suffix(".json")
    sibling.write_text(json.dumps(dict(baseline=a.as_dict(), variation=b.as_dict(),
                                       manifest=manifest,
                                       independent_physics=scientific_reference(
                                           case, (a,b), (left,right))),
                                  indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output, sibling


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List 12 preregistered question cards")
    grade = commands.add_parser("grade", help="Grade all positive and negative cards")
    grade.add_argument("--output", type=Path, default=Path("output/m20c-trial.json"))
    render = commands.add_parser("render", help="Export independent M20c paired viewers")
    render.add_argument("case", choices=PAIRED)
    render.add_argument("--output", type=Path, default=Path("output/m20c-comparison.html"))
    args = parser.parse_args()
    if args.command == "list":
        for case in PAIRED + NEGATIVE:
            print(case)
    elif args.command == "grade":
        report = grade_trial()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
        print("M20c trial:", report["status_counts"], "accepted:", report["accepted"])
        print("Report:", args.output)
        if not report["accepted"]:
            raise SystemExit(1)
    else:
        html, audit = render_case(args.case, args.output)
        print("M20c independent physical histories:", html, audit)


if __name__ == "__main__":
    main()
