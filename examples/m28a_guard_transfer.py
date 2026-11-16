"""R2-disclosed M28a transfer: a baseline terminates at a source guard while
an independently authored gravity variation completes.

This case was proposed in Chat B's PR #65 R2 review. It is now a regression,
not a blind or textbook-source problem. All numerical conditions are explicit
teacher hypotheticals; no charge transfer or exact impact is simulated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from conductor_sim import (
    FixedPointField, PointCharge, PrescribedField, TestChargeMotionSpec,
    UniformGravity, SweepAxis, TrajectorySweepPlan, run_trajectory_sweep,
    write_trajectory_sweep_html,
)
from conductor_sim.scenarios import PrescribedScenario


def run_case():
    baseline = PrescribedScenario(
        "m28:review-gravity-baseline",
        TestChargeMotionSpec(
            PrescribedField((FixedPointField(-1e-10, (.18, 0., .03)),)),
            PointCharge(1e-9, (0., 0., 0.)),
            1e-4, (.8, 0., 2/15), .30, .02,
            plane_normal=(0., 1., 0.), source_guard_m=.006,
            gravity=UniformGravity((0., 0., 0.))),
        (("point:review-held", "Reviewer hypothetical attracting charge"),),
        "Reviewer-proposed held-out teacher setup: one fixed negative point, "
        "one positive moving probe, explicit mass and launch, true source guard.",
    )
    return run_trajectory_sweep(TrajectorySweepPlan(
        "m28:review-gravity", baseline,
        SweepAxis("gravity", "acceleration_m_s2",
                  ((-4., 0., -2/3),)),
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/m28a-gravity-transfer.html"))
    args = parser.parse_args()
    run = run_case()
    path = write_trajectory_sweep_html(run, args.output,
                                        title="M28a · Gravity alters guarded point-charge motion")
    path.with_suffix(".json").write_text(
        json.dumps(run.as_dict(), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({
        "html": str(path),
        "json": str(path.with_suffix(".json")),
        "baseline_termination": run.baseline_trajectory.termination,
        "variant_status": run.entries[0].status,
        "variant_termination": (
            None if run.entries[0].trajectory is None else
            run.entries[0].trajectory.termination),
        "source_provenance": "Teacher hypothetical, suggested by independent reviewer",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
