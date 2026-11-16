"""M31b: export honest native event evidence for atomic coupled hypotheses."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from conductor_sim import (
    CoupledPropertyEdit, CoupledHypothesis, CoupledPhysicalPlan,
    CoupledEventPlan, run_coupled_event_family,
)
from conductor_sim.motion_events import EventPolicy, PlaneEvent
from examples.m29a_structural_family import model


def example(case):
    if case == "added-mass":
        family = model("uniform-edits")
        branch = "add-E"
        changes = (
            CoupledPropertyEdit("field:Enew", "field_v_m", (150., 0., 0.)),
            CoupledPropertyEdit("test-particle", "mass_kg", 2e-5),
        )
        detector_x = .006
    elif case == "joint-shift":
        family = model("point-replace")
        branch = "baseline"
        changes = (
            CoupledPropertyEdit("test-particle", "position_m", (0., .1, .05)),
            CoupledPropertyEdit("point:A", "position_m", (.2, 0., .05)),
        )
        detector_x = .001
    else:
        raise ValueError("unknown M31b authored example")
    physical = CoupledPhysicalPlan("m31b:example", family, (
        CoupledHypothesis("atomic-hypothesis", branch,
                          "Teacher-authored hypothetical physical revision.",
                          changes),
    ))
    return CoupledEventPlan(
        physical, (
            PlaneEvent("x-detector", (detector_x, 0., 0.), (1., 0., 0.),
                       "Passive physical x detector", direction=1),
        ), EventPolicy(.005, 2e-7, 1e-7, 1e-7),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=("added-mass", "joint-shift"),
                   default="added-mass")
    p.add_argument("--output", default="output/m31b-events.json")
    args = p.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = run_coupled_event_family(example(args.case)).as_dict()
    out.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
