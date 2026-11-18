"""M36a inventory guard tests: capability routes are not simulation results."""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.m36a_jee30_audit import (
    ROOT, SOURCE_PDF_SHA256, audit_inventory, load_inventory,
    verify_pdf_bytes,
)


@pytest.fixture
def candidate():
    return copy.deepcopy(load_inventory(ROOT / "docs/m36a_jee30_coverage.json"))


def test_original_29_preserved_and_single_official_30th(candidate):
    result = audit_inventory(candidate)
    assert result["frozen_apparatus_anchors"] == 30
    assert result["historic_m21a_anchors_preserved"] == 29
    assert result["case_specific_native_replays_performed_by_this_tool"] == 0
    assert result["independently_verified_source_specific_simulations"] == 0
    assert result["documented_route_hypotheses_not_scores"] == {
        "partial": 3, "supported_scoped": 24, "unsupported": 3,
    }


def test_changed_historical_source_anchor_refused(candidate):
    candidate["cases"][0]["pdf_page"] += 1
    with pytest.raises(ValueError, match="historical anchor changed"):
        audit_inventory(candidate)


def test_duplicate_question_id_refused(candidate):
    candidate["cases"][-1]["id"] = candidate["cases"][0]["id"]
    with pytest.raises(ValueError, match="duplicate"):
        audit_inventory(candidate)


def test_forged_execution_status_refused(candidate):
    candidate["cases"][4]["case_replay_status"] = "pass"
    with pytest.raises(ValueError, match="unexecuted case claimed PASS"):
        audit_inventory(candidate)


def test_forged_physics_pass_refused(candidate):
    candidate["cases"][0]["verification"]["physical_correctness"] = "pass"
    with pytest.raises(ValueError, match="unearned physical_correctness PASS"):
        audit_inventory(candidate)


def test_unsupported_case_cannot_claim_simulation(candidate):
    case = next(x for x in candidate["cases"] if x["candidate_route"] == "unsupported")
    case["verification"]["simulation_coverage"] = "pass"
    with pytest.raises(ValueError, match="unearned simulation_coverage PASS"):
        audit_inventory(candidate)


def test_invalid_document_or_unsafe_traversal_refused(candidate):
    candidate["cases"][5]["document_basis"] = ["docs/../src/conductor_sim/__init__.py"]
    with pytest.raises(ValueError, match="invalid document link"):
        audit_inventory(candidate)


def test_forged_aggregate_evidence_refused(candidate):
    candidate["case_execution_evidence"]["physically_verified_in_m36a"] = 24
    with pytest.raises(ValueError, match="unproven execution claim"):
        audit_inventory(candidate)


def test_forged_route_denominator_refused(candidate):
    candidate["candidate_route_totals_PROVISIONAL_NOT_A_SCORE"]["supported_scoped"] = 30
    with pytest.raises(ValueError, match="candidate route count mismatch"):
        audit_inventory(candidate)


def test_clean_checkout_cli_emits_zero_execution_evidence(tmp_path):
    out = tmp_path / "audit.json"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, "-m", "tools.m36a_jee30_audit", "--output", str(out)],
        cwd=ROOT, env=env, capture_output=True, text=True, check=True,
    )
    emitted = json.loads(proc.stdout)
    assert emitted == json.loads(out.read_text(encoding="utf-8"))
    assert emitted["result"] == "INVENTORY_VALID; ALL_SOURCE_CASE_EXECUTION_PENDING"
    assert emitted["educational_viewers_actually_inspected"] == 0


@pytest.mark.parametrize("field,replacement", [
    ("source", "fabricated-source.pdf"),
    ("printed_anchor", "Paper 1 Questions 9–11 (one two-charge apparatus)"),
    ("source_provenance", "official JEE Advanced 2022 Paper 1"),
    ("pdf_page", 17),
    ("problem_family", "unrelated electric circuit"),
    ("official_apparatus", {
        "paper": "JEE Advanced 2021 Paper 1", "question_ids": ["Q9", "Q11"],
    }),
    ("official_apparatus", {
        "paper": "JEE Advanced 2020 Paper 1", "question_ids": ["Q9", "Q10"],
    }),
])
def test_official_2021_source_and_membership_fail_closed(candidate, field, replacement):
    """Correct-looking text fragments must not disguise a different apparatus."""
    candidate["cases"][-1][field] = replacement
    with pytest.raises(ValueError, match="2021 apparatus identity or grouping changed"):
        audit_inventory(candidate)


@pytest.mark.parametrize("refusal_id", [
    "p11-linked-28", "p70-cavity-offset", "p70-three-cavities",
])
def test_local_3d_refusal_cannot_be_swapped_with_supported_case(candidate, refusal_id):
    """Balanced 24/3/3 totals cannot falsely declare multicavity physics solved."""
    by_id = {c["id"]: c for c in candidate["cases"]}
    assert by_id[refusal_id]["candidate_route"] == "unsupported"
    by_id[refusal_id]["candidate_route"] = "supported_scoped"
    by_id["p01-single-08"]["candidate_route"] = "unsupported"
    assert sum(c["candidate_route"] == "unsupported" for c in candidate["cases"]) == 3
    with pytest.raises(ValueError, match="frozen local-3D refusal reclassified"):
        audit_inventory(candidate)


def test_historical_and_candidate_pdf_digests_cannot_collude(candidate):
    """Same falsified hash in both JSON files is not acceptable evidence."""
    historic = load_inventory(ROOT / "docs/m21a_coverage.json")
    historic = copy.deepcopy(historic)
    filename = "POTENTIAL_CONDUCTORS1.pdf"
    historic["source_sha256"][filename] = "0" * 64
    candidate["frozen_baseline"]["original_pdf_sha256"][filename] = "0" * 64
    with pytest.raises(ValueError, match="historic PDF metadata differs"):
        audit_inventory(candidate, historic)


def test_original_pdf_stream_bytes_and_mutated_copy(tmp_path):
    """A modified PDF with unchanged JSON cannot pass the actual byte verifier."""
    filename = "POTENTIAL_CONDUCTORS1.pdf"
    expected_sha = SOURCE_PDF_SHA256[filename]
    original = ROOT / filename
    verify_pdf_bytes(original, expected_sha)
    edited = tmp_path / filename
    shutil.copyfile(original, edited)
    with edited.open("ab") as stream:
        stream.write(b"\nM36a-deliberate-postscript-tamper\n")
    with pytest.raises(ValueError, match="source PDF bytes changed"):
        verify_pdf_bytes(edited, expected_sha)


def test_historic_chromium_steps_retained_with_bounded_timeout():
    """Do not resolve the original cancellation by silently deleting old tests."""
    content = (ROOT / ".github/workflows/visualization.yml").read_text(encoding="utf-8")
    match = re.search(r"^\s+timeout-minutes:\s+(\d+)\s*$", content, re.MULTILINE)
    assert match and int(match.group(1)) >= 20
    for original_browser_case in (
        "M14b PDF p3 Chromium offline two-state acceptance",
        "M15 nested cavity with shell constraints",
        "M16b Chromium electron nanosecond time and energy acceptance",
        "M17b Chromium central-force vectors, energy and path",
        "M26a structural driven source Chromium acceptance",
        "M27a real Chromium static two-ring off-axis E/V acceptance",
    ):
        assert original_browser_case in content
