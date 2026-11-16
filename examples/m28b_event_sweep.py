"""M28b: one source apparatus, many physical hypotheses, the same event questions.

The published ring workbook values are author-attributed source givens;
M27b numeric launch/mass/time and the new detector/sweep are teacher choices.
This does not model a constrained axial textbook answer, charge transfer,
actual contact, electromagnetic switching or browser-side event physics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from conductor_sim import (
    PointCharge, PrescribedField, TestChargeMotionSpec, UniformElectricField,
    SweepAxis, TrajectorySweepPlan,
)
from conductor_sim.scenarios import PrescribedScenario
from conductor_sim.event_sweep import EventSweepPlan, run_event_sweep
from conductor_sim.event_sweep_viewer import write_event_sweep_html
from conductor_sim.motion_events import EventPolicy, PlaneEvent, VelocityReversal
from examples.m27b_offaxis_motion import author

CASES = ("ring-detector", "uniform-mass", "multi-crossing", "gravity-guard")


def make_plan(case="ring-detector"):
    if case == "ring-detector":
        base, questions = author("workbook-xy")
        axis = SweepAxis("ring:B", "charge_c", (-18e-6, -22e-6))
        policy = EventPolicy(.001, 1e-8, 1e-7, 1e-5)
    elif case == "uniform-mass":
        base = PrescribedScenario(
            "m28b:uniform-base",
            TestChargeMotionSpec(
                PrescribedField((UniformElectricField((1000., 0., 0.)),)),
                PointCharge(1e-6, (0., 0., 0.)),
                .001, (0., 0., 0.), .5, .025,
                plane_normal=(0., 0., 1.), relative_tolerance=1e-10),
            (("field:E", "Held uniform E"),),
            "Teacher hypothetical: analytic acceleration/detector comparison.")
        axis = SweepAxis("test-particle", "mass_kg", (.0005, .002))
        questions = (PlaneEvent("x:detector", (.05, 0., 0.), (1., 0., 0.),
                                "Passive crossing at x=.05 m", direction=1),)
        policy = EventPolicy(.01, 1e-8, 1e-9, 1e-7)
    elif case == "multi-crossing":
        base = PrescribedScenario(
            "m28b:double-root-base",
            TestChargeMotionSpec(
                PrescribedField((UniformElectricField((-1000., 0., 0.)),)),
                PointCharge(1e-9, (0., 0., 0.)),
                1e-6, (1., 0., 0.), 2., .5,
                plane_normal=(0., 0., 1.)),
            (("field:E", "Held retarding electric field"),),
            "Teacher hypothetical: one detector crossed forwards and backwards.")
        axis = SweepAxis("test-particle", "mass_kg", (.8e-6,))
        questions = (PlaneEvent("x:cross-twice", (.25, 0., 0.),
                                (1., 0., 0.), "Nonterminal detector at .25m"),)
        policy = EventPolicy(.05, 1e-9, 1e-9, 1e-8)
    elif case == "gravity-guard":
        from conductor_sim import FixedPointField, UniformGravity
        base = PrescribedScenario(
            "m28b:gravity-guard-base",
            TestChargeMotionSpec(
                PrescribedField((FixedPointField(-1e-10, (.18, 0., .03)),)),
                PointCharge(1e-9, (0., 0., 0.)),
                1e-4, (.8, 0., 2/15), .30, .02,
                plane_normal=(0., 1., 0.), source_guard_m=.006,
                gravity=UniformGravity((0., 0., 0.))),
            (("point:review-held", "Reviewer-source attracting point"),),
            "Teacher hypothetical, independently proposed in M28a review-01.")
        axis = SweepAxis("gravity", "acceleration_m_s2", ((-4., 0., -2/3),))
        questions = (PlaneEvent("far:detector", (.25, 0., 0.),
                                (1., 0., 0.), "Passive detector beyond fixed point", 1),)
        policy = EventPolicy(.01, 1e-8, 1e-7, 1e-6)
    else:
        raise ValueError("unknown M28b case")
    sweep = TrajectorySweepPlan("m28b:" + case, base, axis)
    return EventSweepPlan(sweep, questions, policy)


def run_case(case="ring-detector"):
    return run_event_sweep(make_plan(case))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default="ring-detector")
    parser.add_argument("--output", type=Path, default=Path("output/m28b-event-sweep.html"))
    args = parser.parse_args()
    result = run_case(args.case)
    path = write_event_sweep_html(result, args.output,
                                  title="M28b · Typed event sweep · "+args.case)
    record = path.with_suffix(".json")
    record.write_text(json.dumps(result.as_dict(), indent=2, allow_nan=False)+"\n",
                      encoding="utf-8")
    print(json.dumps(dict(case=args.case, html=str(path), json=str(record),
        baseline_termination=result.baseline_run.trajectory.termination,
        variation_statuses=[row.status for row in result.entries],
        question_ids=[q.event_id for q in result.plan.questions],
        provenance="Original model source IDs and author claims; teacher sweep/detector",
    ), sort_keys=True))


if __name__ == "__main__":
    main()
