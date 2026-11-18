"""M36b source-backed *actual* JEE apparatus replays, no new production physics.

Run from a clean repository checkout:
    python -m tools.m36b_jee_replay --output-dir output/m36b

Seven source-anchored apparatus call existing real solver/observation APIs.
The unsupported 3D multicavity card is an explicitly NOT SIMULATED refusal.
Data is raw measured native output; independent arithmetic lives elsewhere.
"""
from __future__ import annotations

import argparse
import json
from math import pi
from pathlib import Path

from conductor_sim.dipoles import dipole_work
from conductor_sim.dipole_viewer import write_dipole_explorer
from conductor_sim.motion_events import investigate_motion
from conductor_sim.motion_event_viewer import render_event_run
from conductor_sim.scenarios import sample_scenario_viewer
from conductor_sim.viewer import render_html
from conductor_sim.distant_sphere_viewer import write_distant_spheres
from examples.distant_connected_spheres import make_case as distant_case
from examples.electric_gravity_motion import make_bundle as gravity_bundle
from examples.jee_potential_explorer import make_case as static_case
from examples.motion_event_studies import make_case as event_case
from examples.rigid_dipoles import make_model as dipole_case
from tools.m36a_jee30_audit import audit_inventory, load_inventory, ROOT

SOURCE_IDS = (
    "p02-single-19", "p03-single-30", "p05-single-47",
    "p12-linked-40", "p15-archive-02", "p02-single-15",
    "jee2021-p1-q09-q10-equipotential",
)


def _save_html(directory: Path, name: str, html: str) -> str:
    path = directory / (name + ".html")
    path.write_text(html, encoding="utf-8")
    if len(html) < 800 or "<html" not in html.lower():
        raise ValueError("rendering did not produce a complete offline HTML document: " + name)
    return str(path.name)


def _event_record(study_name: str, output_dir: Path) -> dict:
    study = event_case(study_name)
    result = investigate_motion(study)
    spec = study.scenario.spec
    html = render_event_run(result, title="JEE source replay · " + study_name)
    filename = _save_html(output_dir, study_name, html)
    return {
        "viewer_file": filename,
        "native": True,
        "kind": "actual_elapsed_time_trajectory_and_event",
        "source_notes": study.scenario.provenance,
        "author_choices": "finite start distance for sheet, d=0.01m/V=100V for plates",
        "mass_kg": spec.mass_kg,
        "particle_charge_c": spec.particle.charge_c,
        "source_field_v_m": list(spec.field.sources[0].field_v_m),
        "initial_position_m": list(spec.particle.position_m),
        "initial_velocity_m_s": list(spec.initial_velocity_m_s),
        "duration_s": spec.duration_s,
        "termination": result.trajectory.termination,
        "initial_conditions": dict(result.initial_conditions),
        "events": [
            {
                "id": e.event_id,
                "time_s": e.sample.time_s,
                "position_m": list(e.sample.position_m),
                "velocity_m_s": list(e.sample.velocity_m_s),
                "kinetic_energy_j": e.sample.kinetic_energy_j,
                "full_stop": e.full_stop_within_tolerance,
            }
            for e in result.events
        ],
        "sample_count": len(result.trajectory.samples),
    }


def run_all(output_dir: Path) -> dict:
    audit_inventory()  # Check unchanged original source PDF bytes and 30-source identities.
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = load_inventory(ROOT / "docs/m36a_jee30_coverage.json")
    by_id = {c["id"]: c for c in inventory["cases"]}
    for case_id in SOURCE_IDS:
        if by_id[case_id]["candidate_route"] != "supported_scoped":
            raise ValueError("cannot claim source replay for unqualified apparatus: " + case_id)

    data: dict[str, dict] = {}

    sheet = _event_record("sheet-stop", output_dir)
    sheet["case_id"] = "p02-single-19"
    sheet["source_given"] = {"kinetic_energy_eV": 100., "surface_sigma_c_m2": -2e-6}
    sheet["teacher_launch_distance_m"] = .001
    # One additional observation of the SAME apparatus, never a new solved-case count.
    encounter = _event_record("sheet-encounter", output_dir)
    sheet["early_surface_variant"] = {
        "teacher_launch_distance_m": .0002,
        "termination": encounter["termination"],
        "event_id": encounter["events"][0]["id"] if encounter["events"] else None,
        "event_time_s": encounter["events"][0]["time_s"] if encounter["events"] else None,
    }
    data["p02-single-19"] = sheet

    plate = _event_record("plate-transit", output_dir)
    plate["case_id"] = "p03-single-30"
    plate["source_given"] = "original d and V are symbolic, numerical values below teacher-supplied"
    plate["teacher_gap_m"] = .01
    plate["teacher_potential_difference_v"] = 100.
    data["p03-single-30"] = plate

    model, _revision = dipole_case("workbook-q47")
    angles_deg = (40, 90, 115, 180)
    changes = [
        dipole_work(model, 0., angle*pi/180.)["external_quasistatic_work_j"]
        for angle in angles_deg
    ]
    write_dipole_explorer(
        model, tuple(i*pi/36. for i in range(73)),
        output_dir / "dipole-q47.html", title="JEE source replay · dipole energy Q47",
    )
    data["p05-single-47"] = {
        "case_id": "p05-single-47",
        "native": True,
        "kind": "static_orientation_observations_not_time_evolution",
        "viewer_file": "dipole-q47.html",
        "source_given": "parallel initial dipole, four qualitative angle families; no numeric q, separation, E, mass, inertia, or time",
        "teacher_charge_c": model.dipole.charge_c,
        "teacher_separation_m": model.dipole.separation_m,
        "teacher_field_v_m": list(model.sources[0].source.field_v_m),
        "illustrative_angles_deg": list(angles_deg),
        "observed_delta_u_j": changes,
        "provenance": model.provenance,
    }

    ring_html, ring_record = static_case("ring-work")
    data["p12-linked-40"] = {
        "case_id": "p12-linked-40",
        "native": True,
        "kind": "static_endpoint_work_not_time_evolution",
        "viewer_file": _save_html(output_dir, "ring-work", ring_html),
        "source_anchor": ring_record["anchor"],
        "observed_field_work_j": ring_record["endpoint_work"]["field_work_j"],
        "observed_source_work_components": ring_record["endpoint_work"]["source_work_components"],
        "start_electric_field_v_m": ring_record["endpoint_work"]["start_electric_field_v_m"],
        "end_electric_field_v_m": ring_record["endpoint_work"]["end_electric_field_v_m"],
        "transverse_profile_max_abs_v_m": max(
            r["field_transverse_abs_v_m"] for r in ring_record["profile"]["rows"]
        ),
    }

    scenario, bundle, anchor = gravity_bundle("jee2012-proton")
    html = render_html(bundle, title="JEE source replay · 2012 proton electric and gravity")
    rows = bundle["prescribed_motion"]["samples"]
    data["p15-archive-02"] = {
        "case_id": "p15-archive-02",
        "native": True,
        "kind": "actual_elapsed_time_trajectory",
        "viewer_file": _save_html(output_dir, "proton-2012", html),
        "source_anchor": anchor,
        "source_notes": scenario.provenance,
        "source_electric_field_v_m": list(scenario.spec.field.sources[0].field_v_m),
        "source_gravity_m_s2": list(scenario.spec.gravity.acceleration_m_s2),
        "mass_kg": scenario.spec.mass_kg,
        "charge_c": scenario.spec.particle.charge_c,
        "initial_velocity_m_s": list(scenario.spec.initial_velocity_m_s),
        "termination": bundle["prescribed_motion"]["termination"],
        "saved_samples": [
            {key: sample[key] for key in (
                "time_s", "position_m", "velocity_m_s",
                "electric_force_n", "gravitational_force_n", "kinetic_energy_j",
            )} for sample in rows
        ],
    }

    distant, source = distant_case("surface-ratio")
    write_distant_spheres(
        distant, output_dir / "connected-spheres.html",
        title="JEE source replay · far-separated connected spheres", provenance=source,
    )
    data["p02-single-15"] = {
        "case_id": "p02-single-15",
        "native": True,
        "kind": "approximate_static_before_and_after_equilibria",
        "viewer_file": "connected-spheres.html",
        "source_given": source["source_given"],
        "teacher_choices": source["additional_choices"],
        "model_qualification": distant.as_dict()["qualification"],
        "before": [
            {"name": c.name, "radius_m": c.outer_radius_m, "charge_c": c.charge_c}
            for c in distant.before.conductors
        ],
        "after": [
            {"name": c.name, "radius_m": c.outer_radius_m, "charge_c": c.charge_c,
             "outer_field_v_m": c.outer_field_normal_v_m}
            for c in distant.after.conductors
        ],
        "release_energy_j": distant.released_field_energy_j,
    }

    circle_html, circle = static_case("2021-equipotential")
    frame = circle["frames"][0]
    highlights = [x for x in frame["contours"] if x.get("highlight") is True]
    if len(highlights) != 1 or not highlights[0]["segments_m"]:
        raise ValueError("actual native sampled V=0 contour unavailable")
    points = [p for segment in highlights[0]["segments_m"] for p in segment]
    stride = max(1, len(points) // 160)
    origin_x,origin_y = frame["plane"]["origin_m"][:2]
    world_points = [[origin_x+p[0], origin_y+p[1]] for p in points[::stride]]
    data["jee2021-p1-q09-q10-equipotential"] = {
        "case_id": "jee2021-p1-q09-q10-equipotential",
        "native": True,
        "kind": "sampled_static_two_charge_zero_potential_contour",
        "viewer_file": _save_html(output_dir, "jee2021-zero-potential", circle_html),
        "source_anchor": circle["jee_anchor"],
        "sources": circle["static_fixed_source_view"]["field"]["sources"],
        "actual_highlighted_level_v": highlights[0]["level_v"],
        "actual_segment_count": len(highlights[0]["segments_m"]),
        "sampled_contour_world_xy_m": world_points,
        "source_mask_count": len(frame["source_masks"]),
        "has_motion": any(k in circle for k in ("motion_1d","motion_2d","prescribed_motion")),
    }

    refusal_id = "p70-three-cavities"
    if by_id[refusal_id]["candidate_route"] != "unsupported":
        raise ValueError("cannot fabricate local-3D cavity surface density")
    refused = {
        "case_id": refusal_id,
        "native": False,
        "kind": "honest_unsupported_local_3d_field_refusal",
        "reason": "Nonconcentric three-cavity local surface density and electric field are not qualified; Gauss-law totals alone are insufficient.",
        "viewer_file": None,
    }
    if len(data) != 7 or any(not x["native"] for x in data.values()):
        raise AssertionError("seven genuine source-model observations required")
    report = {
        "schema": "conductor_sim.m36b_source_replays.v1",
        "evaluated_against": "M36a frozen 30-apparatus provisional cohort",
        "source_apparatus_count": len(data),
        "source_apparatus": data,
        "unsupported_refusal_count": 1,
        "explicit_refusal": refused,
        "scientific_verification_status": "pending independent source-free reference checker",
        "browser_verification_status": "pending actual real Chromium",
        "textbook_answer_completeness": "not_attempted",
    }
    (output_dir / "native-replays.json").write_text(
        json.dumps(report, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/m36b"))
    args = parser.parse_args()
    r = run_all(args.output_dir)
    print("M36b genuine source apparatus:", r["source_apparatus_count"])
    print("Honest local-3D refusal (not simulated):", r["unsupported_refusal_count"])
    print("REPLAY_ONLY; physical and browser reference gates still pending")


if __name__ == "__main__":
    main()
