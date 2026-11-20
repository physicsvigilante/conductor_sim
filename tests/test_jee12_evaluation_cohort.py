"""Guard that the JEE-12 cohort never silently becomes a fabricated coverage score.

This verifies the source-anchored selection, not actual physical answer completeness.
Evidence status can advance only with separately reviewed case-specific tests.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "docs" / "jee12_evaluation_cohort.json"
HISTORY = ROOT / "docs" / "m36a_jee30_coverage.json"
IDS = (
    "p01-single-08",
    "p02-single-19",
    "p03-single-27",
    "p07-multiple-13",
    "p08-multiple-21",
    "p11-linked-28",
    "p12-linked-40",
    "p16-archive-numerical-01",
    "p26-conductor-potential",
    "p27-cavity-two-shells",
    "p69-cavity-central-vs-shifted",
    "p70-three-cavities",
)
EVALUATION_FIELDS = (
    "source_fact_attribution",
    "interpretation_fidelity",
    "independent_physics",
    "original_answer_completeness",
    "real_browser_observation",
    "teaching_usefulness",
    "refusal_correctness",
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_jee12_historical_exact_roster_and_original_source_provenance() -> None:
    cohort, source = _read(COHORT), _read(HISTORY)
    rows = cohort["cases"]
    assert cohort["schema"] == "conductor_sim.jee12_source_evaluation.v1"
    assert cohort["historical_schema"] == source["schema"]
    assert cohort["case_count"] == 12 == len(rows)
    assert tuple(x["id"] for x in rows) == IDS
    assert len(set(IDS)) == len(IDS)
    original = [x for x in source["cases"] if x["id"] in IDS]
    assert [x["id"] for x in original] == list(IDS)
    for actual, historical in zip(rows, original):
        for key in ("id", "source", "pdf_page", "printed_anchor",
                    "problem_family", "source_provenance"):
            assert actual[key] == historical[key]
        assert actual["historical_candidate_route_not_verified"] == historical["candidate_route"]
        assert actual["historical_qualified_route_not_verified"] == historical["qualified_route"]
        assert actual["historical_exclusions"] == historical["exclusions"]
    assert cohort["frozen_pdf_sha256"] == source["frozen_baseline"]["original_pdf_sha256"]
    assert len(cohort["frozen_pdf_sha256"]) == 2
    assert all(len(digest) == 64 for digest in cohort["frozen_pdf_sha256"].values())


def test_jee12_no_unreviewed_source_or_answer_claims() -> None:
    cohort = _read(COHORT)
    assert "NOT A JEE PERFORMANCE SCORE" in cohort["status"]
    assert "ZERO INDEPENDENT ORIGINAL ANSWER VERDICTS" in cohort["status"]
    assert cohort["verified_original_source_answers_in_this_cohort"] == 0
    assert cohort["source_inspections_performed_in_this_cohort"] == 3
    inspected = {"p26-conductor-potential", "p27-cavity-two-shells",
                 "p69-cavity-central-vs-shifted"}
    assert set(cohort["developer_visually_inspected_case_ids"]) == inspected
    assert cohort["independently_reviewed_original_answer_cases"] == 0
    routes = [x["historical_candidate_route_not_verified"] for x in cohort["cases"]]
    assert routes.count("supported_scoped") == 8
    assert routes.count("partial") == 2
    assert routes.count("unsupported") == 2
    for row in cohort["cases"]:
        evaluation = row["evaluation"]
        assert evaluation["original_pdf_reinspected"] is (row["id"] in inspected)
        if row["id"] in inspected:
            assert "developer_pdf_source_note" in row
            note = row["developer_pdf_source_note"]
            if row["id"] == "p26-conductor-potential":
                assert note["standalone_html_authenticity_not_certified"] is True
            else:
                assert note["original_answer_newly_independently_reviewed"] is False
        assert evaluation["reviewer_evidence"] == "none"
        for key in EVALUATION_FIELDS:
            assert evaluation[key] == "not_evaluated"
        assert row["source_provenance"]
        assert row["printed_anchor"]
        assert row["historical_exclusions"]


def test_jee12_physical_refusal_cases_must_not_be_assigned_spherical_success() -> None:
    cohort = _read(COHORT)
    entries = {x["id"]: x for x in cohort["cases"]}
    assert entries["p11-linked-28"]["historical_candidate_route_not_verified"] == "unsupported"
    assert entries["p70-three-cavities"]["historical_candidate_route_not_verified"] == "unsupported"
    assert "irregular" in entries["p11-linked-28"]["problem_family"]
    assert "three" in entries["p70-three-cavities"]["problem_family"]
    assert entries["p69-cavity-central-vs-shifted"]["evaluation"]["original_answer_completeness"] == "not_evaluated"
    assert entries["p26-conductor-potential"]["evaluation"]["original_answer_completeness"] == "not_evaluated"
    p26=entries["p26-conductor-potential"]
    assert p26["developer_pdf_source_note"]["actual_printed_shape"] == "one solid charged conducting sphere (not a hollow shell)"
    assert p26["developer_pdf_source_note"]["historical_problem_family_is_superseded"] is True
    assert p26["developer_pdf_source_note"]["numerical_values_are_teacher_authored"] is True
    assert p26["developer_pdf_source_note"]["originally_asked_and_printed_solutions"] == [
        "V(R/2) = kQ/R", "V(3R/2) = 2kQ/(3R)"]
    p27=entries["p27-cavity-two-shells"]
    assert p27["evaluation"]["original_answer_completeness"] == "not_evaluated"
    assert p27["developer_pdf_source_note"]["asked"] == ["V(A)", "V(B)"]
    assert p27["developer_pdf_source_note"]["teacher_numerical_values_not_source_givens"] is True
    assert p27["developer_pdf_source_note"]["original_answer_newly_independently_reviewed"] is False
    p69=entries["p69-cavity-central-vs-shifted"]
    assert p69["evaluation"]["original_answer_completeness"] == "not_evaluated"
    assert p69["developer_pdf_source_note"]["source_given_metal_charge"] == "3q"
    assert p69["developer_pdf_source_note"]["source_given_enclosed_charge"] == "q"
    assert p69["developer_pdf_source_note"]["original_printed_qualitative_result"] == [
        "inner induced total -q, uniform if centered",
        "inner induced total -q, nonuniform if displaced",
        "outer total +4q, uniform and unchanged"]
    assert p69["developer_pdf_source_note"]["original_answer_newly_independently_reviewed"] is False


