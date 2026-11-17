"""M36a: validate an honest, source-anchored *candidate* JEE-30 inventory.

No physics modules are imported and this tool does not run a simulation.
A documented supported route is not a source-specific executed PASS.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASELINE_SHA = "73e076cf7eaf7fefbae2b24afc1f54ff8f0b6fc6"
BASELINE_TREE = "89363a917e3862d663d934501a1b5cd8a22a84c0"
NEW_ANCHOR = "jee2021-p1-q09-q10-equipotential"
ROUTES = frozenset({"supported_scoped", "partial", "unsupported"})
PRIMARY = (
    "interpretation_fidelity",
    "physical_correctness",
    "simulation_coverage",
    "educational_usefulness",
)
EVIDENCE_FIELDS = (
    "independently_replayed_in_m36a",
    "physically_verified_in_m36a",
    "viewer_checked_in_m36a",
    "source_documents_newly_reinspected_in_m36a",
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_inventory(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    _assert(type(data) is dict, "inventory must be an object")
    return data


def audit_inventory(
    candidate: dict[str, Any] | None = None,
    historic: dict[str, Any] | None = None,
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    """Validate claimed evidence, historic source identity and route denominators.

    Dependency injection permits tampering tests to exercise rejected cases
    without modifying the checked-in source inventory.
    """
    if candidate is None:
        candidate = load_inventory(repo_root / "docs/m36a_jee30_coverage.json")
    if historic is None:
        historic = load_inventory(repo_root / "docs/m21a_coverage.json")
    _assert(type(candidate) is dict and type(historic) is dict, "invalid inventory objects")
    _assert(candidate.get("schema") == "conductor_sim.m36a_jee30_candidate_coverage.v1", "schema")
    baseline = candidate.get("frozen_baseline")
    _assert(type(baseline) is dict, "missing pinned baseline")
    _assert(baseline.get("source_commit") == BASELINE_SHA, "baseline commit")
    _assert(baseline.get("source_git_tree") == BASELINE_TREE, "baseline tree")
    _assert(baseline.get("historical_inventory") == "docs/m21a_coverage.json", "historical source")
    _assert(type(baseline.get("historical_anchor_count")) is int and baseline["historical_anchor_count"] == 29, "historical count")
    _assert(baseline.get("original_pdf_sha256") == historic.get("source_sha256"), "original PDF digests")
    old = historic.get("cases")
    cases = candidate.get("cases")
    _assert(type(old) is list and len(old) == 29, "original historical cohort changed")
    _assert(type(cases) is list and len(cases) == 30, "exactly 30 JEE apparatus anchors required")
    ids = [entry.get("id") if type(entry) is dict else None for entry in cases]
    _assert(len(set(ids)) == 30, "duplicate or malformed case ID")

    keys = (
        ("id", "id"),
        ("source", "source"),
        ("pdf_page", "pdf_page"),
        ("printed_anchor", "printed_anchor"),
        ("problem_family", "problem_family"),
        ("provenance", "source_provenance"),
        ("tier", "original_m21a_tier"),
    )
    for i, old_case in enumerate(old):
        new_case = cases[i]
        _assert(type(old_case) is dict and type(new_case) is dict, f"bad case type {i}")
        for old_key, new_key in keys:
            _assert(old_case.get(old_key) == new_case.get(new_key),
                    f"historical anchor changed at index {i}: {old_key}")

    extra = cases[-1]
    _assert(extra.get("id") == NEW_ANCHOR, "new apparatus ID")
    _assert(extra.get("original_m21a_tier") is None, "do not invent a 2024 M21a tier for new source")
    _assert(extra.get("pdf_page") == 16 and "Questions 9" in extra.get("printed_anchor", "")
            and "2021" in extra.get("source_provenance", ""), "2021 source anchor")
    _assert("official JEE Advanced 2021" in extra.get("source_provenance", ""), "official source provenance")
    _assert(candidate.get("evaluation_contract") == "docs/evaluation_contract.md", "evaluation rubric")
    _assert(type(candidate.get("status")) is str and "PROVISIONAL" in candidate["status"], "candidate warning")

    actual = Counter()
    for c in cases:
        route = c.get("candidate_route")
        _assert(route in ROUTES, f"invalid route for {c.get('id')}")
        actual[route] += 1
        _assert(type(c.get("qualified_route")) is str and bool(c["qualified_route"].strip()),
                f"missing physical qualification for {c.get('id')}")
        _assert(type(c.get("exclusions")) is str and bool(c["exclusions"].strip()),
                f"missing excluded physics for {c.get('id')}")
        paths = c.get("document_basis")
        _assert(type(paths) is list and bool(paths), f"missing document basis {c.get('id')}")
        for path in paths:
            _assert(type(path) is str and path.startswith("docs/")
                    and ".." not in Path(path).parts
                    and (repo_root / path).is_file(), f"invalid document link: {path!r}")
        _assert(c.get("case_replay_status") == "pending", f"unexecuted case claimed PASS: {c.get('id')}")
        _assert(c.get("independent_source_reinspection") == "not_performed_in_m36a",
                f"unobserved source reading claimed: {c.get('id')}")
        ev = c.get("verification")
        _assert(type(ev) is dict and set(ev) == {*PRIMARY, "textbook_answer_completeness"},
                f"missing five-axis assessment: {c.get('id')}")
        for name in PRIMARY:
            _assert(ev[name] == "not_evaluated", f"unearned {name} PASS: {c.get('id')}")
        _assert(ev["textbook_answer_completeness"] == "not_attempted",
                f"unearned textbook answer: {c.get('id')}")

    declared = candidate.get("candidate_route_totals_PROVISIONAL_NOT_A_SCORE")
    _assert(type(declared) is dict and set(declared) == ROUTES, "declared triage categories")
    for status in ROUTES:
        _assert(type(declared[status]) is int and declared[status] == actual[status],
                f"candidate route count mismatch: {status}")
    _assert(actual["supported_scoped"] == 24 and actual["partial"] == 3
            and actual["unsupported"] == 3, "frozen provisional triage cohort changed")
    evidence = candidate.get("case_execution_evidence")
    _assert(type(evidence) is dict and set(evidence) == set(EVIDENCE_FIELDS), "execution evidence schema")
    for key in EVIDENCE_FIELDS:
        _assert(type(evidence[key]) is int and evidence[key] == 0, f"unproven execution claim: {key}")
    return {
        "schema": "conductor_sim.m36a_jee30_audit_report.v1",
        "baseline_commit": BASELINE_SHA,
        "frozen_apparatus_anchors": len(cases),
        "historic_m21a_anchors_preserved": len(old),
        "documented_route_hypotheses_not_scores": dict(sorted(actual.items())),
        "case_specific_native_replays_performed_by_this_tool": 0,
        "independently_verified_source_specific_simulations": 0,
        "source_documents_freshly_reinspected": 0,
        "educational_viewers_actually_inspected": 0,
        "textbook_answers_reproduced": 0,
        "result": "INVENTORY_VALID; ALL_SOURCE_CASE_EXECUTION_PENDING",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = audit_inventory()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
