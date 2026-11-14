"""M21a: validate and report source-anchored JEE coverage without engine edits.

This is an EVIDENCE matrix of preselected 29 questions, not 29 executed
end-to-end simulations or a statistical sample of the JEE Advanced exam.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs" / "m21a_coverage.json"
SOURCE_COUNTS = {"02_conductors_half_size.pdf": 18,
                 "POTENTIAL_CONDUCTORS1.pdf": 11}
SOURCE_PAGES = {"02_conductors_half_size.pdf": 17,
                "POTENTIAL_CONDUCTORS1.pdf": 71}
ALLOWED = {"A", "B", "C", "D"}
RELEVANCE = {"high", "medium", "foundational-control"}
FIDELITY = {"full", "partial", "none"}


def validate(root=ROOT):
    root = Path(root)
    payload = json.loads((root / "docs" / "m21a_coverage.json").read_text(encoding="utf-8"))
    if payload["schema"] != "conductor_sim.m21a_jee_coverage.v1":
        raise ValueError("unknown audit data schema")
    cases = payload["cases"]
    if len(cases) != 29 or payload["anchor_count"] != 29:
        raise ValueError("audit must retain all 29 preregistered source anchors")
    ids = [case["id"] for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate source anchor")
    expected = {"02_conductors_half_size.pdf": 0, "POTENTIAL_CONDUCTORS1.pdf": 0}
    for case in cases:
        filename = case["source"]
        if filename not in expected:
            raise ValueError("unrecognized source PDF")
        expected[filename] += 1
        page = case["pdf_page"]
        if type(page) is not int or not 1 <= page <= SOURCE_PAGES[filename]:
            raise ValueError("invalid physical PDF page")
        if (case["tier"] not in ALLOWED or case["fidelity"] not in FIDELITY
                or case["jee_relevance"] not in RELEVANCE):
            raise ValueError("unknown qualification category")
        for key in ("printed_anchor", "problem_family", "existing_route",
                    "limitation", "provenance", "evidence_tier"):
            if not isinstance(case[key], str) or not case[key].strip():
                raise ValueError(f"missing {key} on {case['id']}")
        if case["tier"] == "A" and case["fidelity"] == "none":
            raise ValueError("full capability cannot be assigned to unsupported model")
        if case["tier"] == "C" and case["fidelity"] == "full":
            raise ValueError("new physics cannot be listed as already complete")
    if expected != SOURCE_COUNTS:
        raise ValueError(f"source anchors differ from preregistered counts: {expected}")
    sources = {}
    for filename in SOURCE_COUNTS:
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256(path.read_bytes()).hexdigest()
        pinned = payload["source_sha256"][filename]
        if digest != pinned:
            raise ValueError("source PDF digest mismatch: " + filename)
        sources[filename] = dict(sha256=digest,size_bytes=path.stat().st_size)
    tiers = dict(sorted(Counter(c["tier"] for c in cases).items()))
    relevance = dict(sorted(Counter(c["jee_relevance"] for c in cases).items()))
    fidelity = dict(sorted(Counter(c["fidelity"] for c in cases).items()))
    result = dict(schema="conductor_sim.m21a_jee_audit_report.v1",
                  classification_is_engineering_assessment=True,
                  cases_examined=len(cases), source_counts=expected,
                  source_integrity=sources, tier_counts=tiers,
                  fidelity_counts=fidelity, relevance_counts=relevance,
                  case_ids=ids,
                  interpretation="Printed workbook/source anchors and original figures checked; archived exam provenance verified only where explicitly indicated",
                  physics="A/B/C are qualifications from accepted API and figure audit, not 29 solved trajectories",
                  usefulness="Prioritize authentic complex JEE-style electrostatic observations and source-faithful presentation",
                  textbook_answer_completeness="Not assessed for all 29 anchors",
                  source_cases=cases)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path,default=Path("output/m21a-coverage-report.json"))
    args = ap.parse_args()
    report = validate()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("M21a",report["cases_examined"],"source anchors,",report["tier_counts"],
          "PDF digests verified. Classification is not end-to-end execution.")


if __name__ == "__main__":
    main()
