"""Independent-reference transfer: one genuine M31a coupled rod/source event.

This program authors a physical model and runs the published native M24a
event observations. It does NOT compute the independent DOP853 reference.
Use tools/m31b_independent_dop853.py separately to challenge its JSON.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from conductor_sim import (
    CoupledPropertyEdit, CoupledHypothesis, CoupledPhysicalPlan,
    FixedPointField, PointCharge, PrescribedField, TestChargeMotionSpec,
    UniformGravity, UniformRodField,
    CoupledEventPlan, run_coupled_event_family,
)
from conductor_sim.motion_events import EventPolicy, PlaneEvent
from conductor_sim.scenarios import PrescribedScenario
from conductor_sim.structural_family import StructuralCase, StructuralFamilyPlan
from conductor_sim.structural_scenarios import NamedSource


def make_plan():
    original = PrescribedScenario(
        "m31b:rod-original",
        TestChargeMotionSpec(
            PrescribedField((
                UniformRodField(3e-9, (.11,-.1,0.), (.13,.12,0.)),
                FixedPointField(2e-9, (.18,.15,0.)),
                FixedPointField(-1e-9, (-.16,.10,0.)),
            )),
            PointCharge(1e-9, (0.,0.,0.)),
            1e-5, (.05,.02,0.), .7, .005,
            plane_normal=(0.,0.,1.), relative_tolerance=1e-10,
            gravity=UniformGravity((0.,-.1,0.))),
        (("rod:slanted","Tilted uniformly charged finite rod"),
         ("point:plus","Positive held point"),
         ("point:minus","Negative held point")),
        "Teacher-authored new rod/point/gravity physical source model.")
    family = StructuralFamilyPlan("m31b:rod-family", original, (
        StructuralCase("new-point", "Add a real held source with its own ID.",
            add_sources=(NamedSource("point:added", "Extra held point",
                FixedPointField(1.7e-9, (.21,-.09,0.))),)),
    ))
    coupled = CoupledPhysicalPlan("m31b:rod-verify", family, (
        CoupledHypothesis(
            "rod-mass", "new-point", "Couple new point charge and probe mass.",
            (
                CoupledPropertyEdit("point:added", "charge_c", 2e-9),
                CoupledPropertyEdit("test-particle", "mass_kg", 9e-6),
            )),
    ))
    return CoupledEventPlan(
        coupled,
        (PlaneEvent("x:left", (-.006,0.,0.), (1.,0.,0.),
                    "Signed x-left passive detector", direction=-1),),
        EventPolicy(.002, 1e-9, 1e-9, 1e-7),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="output/m31b-rod-native.json")
    args = parser.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = run_coupled_event_family(make_plan())
    data = result.as_dict()
    out.write_text(json.dumps(data, allow_nan=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(
        path=str(out), source_head="read git rev-parse HEAD separately",
        parent_termination=result.parents[0].event_run.trajectory.termination,
        child_termination=result.children[0].event_run.trajectory.termination,
        parent_roots=[e.sample.time_s for e in result.parents[0].event_run.events],
        child_roots=[e.sample.time_s for e in result.children[0].event_run.events],
        source_ids=[s["id"] for s in data["coupled_events"][0]["accepted_source_inventory"]],
    ), sort_keys=True))


if __name__ == "__main__":
    main()
