"""M36a inventory guard tests: capability routes are not simulation results."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.m36a_jee30_audit import ROOT, audit_inventory, load_inventory


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
