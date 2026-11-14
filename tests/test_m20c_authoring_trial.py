"""M20c held-out-in-kind authoring trial: no Ring A/B fixtures or engine edits."""
from dataclasses import FrozenInstanceError, asdict
import json
from math import hypot

import pytest

from conductor_sim import (
    FixedPointField, UniformDiskAxisField, UniformElectricField,
    UniformRodField, UniformShellField, UniformSphereField,
)
from conductor_sim.scenarios import revise_scenario, sample_scenario_viewer
from examples.m20c_authoring_trial import (
    NEGATIVE, NOTES, PAIRED, VIEW, _grade_incomparable, _grade_refusal,
    baseline, bundles_and_manifest, grade_trial, pair, render_case,
    scientific_reference,
)


def test_charter_cases_all_accounted_for_and_no_rings():
    assert PAIRED == ("U1", "V1", "R1", "P1", "S1", "D1")
    assert NEGATIVE == ("N1", "N2", "N3", "N4", "N5", "N6")
    expected = {
        "U1": (UniformElectricField,),
        "V1": (UniformSphereField,),
        "R1": (UniformRodField,),
        "P1": (FixedPointField, UniformElectricField),
        "S1": (UniformShellField,),
        "D1": (UniformDiskAxisField,),
    }
    for case in PAIRED:
        b = baseline(case)
        assert tuple(type(s) for s in b.spec.field.sources) == expected[case]
        assert len({x for x, label in b.source_annotations}) == len(b.spec.field.sources)
        assert b.origin == "teacher-hypothetical"
        assert "Ring A" not in b.provenance and "Ring B" not in b.provenance


@pytest.mark.parametrize("case", PAIRED)
def test_six_independently_solved_nonring_models(case):
    base, revision, left, right, manifest = bundles_and_manifest(case)
    assert base.revision_id == revision.parent_revision_id
    assert base.spec != revision.spec
    assert base.source_annotations == revision.source_annotations
    assert revision.origin == "teacher-hypothetical"
    assert revision.provenance == NOTES[case]
    assert revision.as_dict()["changes"] == json.loads(json.dumps(
        [asdict(c) for c in revision.changes]))
    assert manifest["revisions"] == revision.as_dict()["changes"]
    assert len(manifest["revisions"]) == 1
    assert manifest["source_ids"] == [x for x, label in base.source_annotations]
    assert manifest["time_interpolation"] is False
    assert manifest["plane"] == left["frames"][0]["plane"] == right["frames"][0]["plane"]
    for b in (left, right):
        assert b["prescribed_motion"]["termination"] == "duration"
        assert len(b["prescribed_motion"]["samples"]) > 10
        assert all(item["electric_field_v_m"] is not None for item in
                   b["prescribed_motion"]["samples"])
        assert b["prescribed_motion"]["energy_convention"].startswith("U=q*V")
        assert b["prescribed_motion"]["profile_mode"] == (
            "axis" if case == "D1" else "elapsed-time")
    science = scientific_reference(case, (base, revision), (left, right))
    assert science["checked_scalar_observations"] > 0
    assert science["max_abs_reference_error"] >= 0


@pytest.mark.parametrize("case", PAIRED)
def test_parent_preservation_stable_identity_and_actual_fresh_spec(case):
    base = baseline(case)
    old = base.as_dict()
    original_sources = base.spec.field.sources
    _, revised = pair(case)
    assert base.as_dict() == old
    assert base.spec.field.sources == original_sources
    assert revised.spec is not base.spec
    assert revised.spec.field is not base.spec.field
    assert revised.spec.field.sources != base.spec.field.sources or (
        revised.spec.mass_kg != base.spec.mass_kg)
    with pytest.raises(FrozenInstanceError):
        revised.spec.mass_kg = 0.25
    # Detached JSON export must not permit mutation through an audit copy.
    audit = revised.as_dict()
    audit["spec"]["mass_kg"] = 999.
    assert revised.spec.mass_kg != 999.


def test_uniform_field_has_real_2d_acceleration_without_changing_initial_velocity():
    a, b, left, right, _ = bundles_and_manifest("U1")
    assert a.spec.initial_velocity_m_s == b.spec.initial_velocity_m_s == (.2, 0., 0.)
    assert a.spec.mass_kg != b.spec.mass_kg
    for bundle in (left, right):
        rows = bundle["prescribed_motion"]["samples"]
        assert rows[-1]["position_m"][0] > 0
        assert rows[-1]["position_m"][2] < 0
        assert rows[0]["position_m"] == [0., 0., 0.]


def test_volume_sphere_motion_remains_inside_not_conductor_induction():
    a, b, left, right, _ = bundles_and_manifest("V1")
    for bundle in (left, right):
        rows = bundle["prescribed_motion"]["samples"]
        assert all(hypot(*r["position_m"]) < .1 for r in rows)
    assert left["prescribed_motion"]["samples"][-1]["position_m"] != \
           right["prescribed_motion"]["samples"][-1]["position_m"]
    assert a.spec.field.sources[0].radius_m == b.spec.field.sources[0].radius_m


def test_rod_flyby_and_point_plus_field_truly_2d():
    for case in ("R1", "P1"):
        a, b, left, right, _ = bundles_and_manifest(case)
        initial = a.spec.particle.position_m
        final = right["prescribed_motion"]["samples"][-1]["position_m"]
        assert abs(final[0] - initial[0]) > 1e-6
        assert abs(final[2] - initial[2]) > 1e-6
        assert a.spec.plane_normal == b.spec.plane_normal
        assert final[1] == pytest.approx(0., abs=1e-12)


def test_shell_interior_is_nonconducting_reference_control():
    a, b, left, right, _ = bundles_and_manifest("S1")
    rows_a, rows_b = (x["prescribed_motion"]["samples"] for x in (left, right))
    assert all(x["electric_field_v_m"] == [0., 0., 0.] for x in rows_a + rows_b)
    assert [x["position_m"] for x in rows_a] == [x["position_m"] for x in rows_b]
    assert rows_a[0]["potential_energy_j"] != rows_b[0]["potential_energy_j"]
    assert all(hypot(*x["position_m"]) < .12 for x in rows_a)
    assert a.spec.field.sources[0].charge_c * 2 == b.spec.field.sources[0].charge_c


def test_disc_axis_remains_explicitly_axis_only():
    a, b, left, right, _ = bundles_and_manifest("D1")
    assert a.spec.axis == b.spec.axis == (0., 0., 1.)
    assert all(abs(c) < 1e-12 for x in (left, right)
               for r in x["prescribed_motion"]["samples"]
               for c in r["position_m"][:2])
    assert b.spec.field.sources[0].radius_m == .11


@pytest.mark.parametrize("case", NEGATIVE[:4])
def test_four_refusals_are_real_validation_errors_not_fabricated_results(case):
    record = _grade_refusal(case)
    assert record["card"] == case
    assert record["status"] == "refused"
    assert record["exception"]
    assert record["reason"]


@pytest.mark.parametrize("case", NEGATIVE[4:])
def test_admissible_does_not_imply_comparable_complete_histories(case):
    record = _grade_incomparable(case)
    assert record["status"] == "admitted-not-comparable", record
    assert len(record["terminations"]) == 2
    assert record["physical_revision"]
    assert record["parent_unchanged"] is True
    assert "comparison" in record["reason"] or "scenarios" in record["reason"]


def test_every_preregistered_card_has_result_and_no_hidden_exclusions():
    grade = grade_trial()
    assert grade["accepted"] is True, grade
    assert grade["case_count"] == 12
    assert [r["card"] for r in grade["case_results"]] == list(PAIRED + NEGATIVE)
    assert [r["status"] for r in grade["case_results"]] == (
        ["paired"]*6 + ["refused"]*4 + ["admitted-not-comparable"]*2)
    assert grade["status_counts"] == {
        "paired": 6, "refused": 4, "admitted-not-comparable": 2}
    assert grade["dimensions"]["textbook_answer_completeness"].startswith("Not assessed")


@pytest.mark.parametrize("case", ("U1", "P1", "S1", "D1"))
def test_m20c_offline_html_and_detached_audit(tmp_path, case):
    output = tmp_path / (case + ".html")
    html, audit = render_case(case, output)
    assert html.is_file() and audit.is_file()
    text = html.read_text(encoding="utf-8")
    assert "M20c cross-source " + case in text
    assert "comparison-data" in text and "iframe" in text
    record = json.loads(audit.read_text(encoding="utf-8"))
    assert record["variation"]["parent_revision_id"] == \
        record["baseline"]["revision_id"]
    assert record["independent_physics"]["checked_scalar_observations"] > 0
    assert record["manifest"]["time_interpolation"] is False


def test_out_of_scope_edit_does_not_change_any_declared_parent():
    root = baseline("D1")
    before = root.as_dict()
    with pytest.raises(Exception):
        revise_scenario(root, scenario_id="bad", note="An invalid off-axis launch.",
                        motion_changes={"initial_velocity_m_s": (.1, 0., .01)})
    assert root.as_dict() == before
