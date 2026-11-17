"""M34a: an actual non-dyadic oblique physical plane survives source revisions.

No new solver: genuine M22a source addition, genuine M20b mass revision,
then native M22a replacement on the true immediate physical source model,
plus a sibling hypothesis using the same ID on its independent branch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from conductor_sim import (
    CoupledPropertyEdit, PhysicalWorldNode, PointCharge, PrescribedField,
    StructuralWorldNode, TestChargeMotionSpec, UniformElectricField,
    UniformGravity, UnifiedObservationPlan, run_unified_observation_graph,
)
from conductor_sim.motion_events import EventPolicy, PlaneEvent
from conductor_sim.scenarios import PrescribedScenario
from conductor_sim.structural_family import StructuralCase
from conductor_sim.structural_scenarios import NamedSource, SourceReplacement


def make_plan():
    original = PrescribedScenario(
        "m34a:oblique-root",
        TestChargeMotionSpec(
            PrescribedField((UniformElectricField((320., -80., 160.)),)),
            PointCharge(2e-9, (.003, .012, .06)), 1.6e-6,
            (.25, -.125, 0.), .6, .01,
            plane_normal=(1., 2., -1.),
            gravity=UniformGravity((.16, -.04, .08))),
        (("field:base", "Original fixed oblique field"),),
        "Teacher hypothetical three-component planar qE+mg model.")

    add = StructuralCase(
        "add-a", "Genuinely introduce another oblique in-plane held source.",
        add_sources=(NamedSource(
            "field:aux", "Introduced held field A",
            UniformElectricField((90., 35., 160.))),))
    replacement = StructuralCase(
        "replace-a", "Retire A and introduce a genuinely different source B.",
        replace_sources=(SourceReplacement(
            "field:aux", NamedSource(
                "field:final", "Replacement held field B",
                UniformElectricField((130., -10., 110.)))),))
    sibling = StructuralCase(
        "sibling", "An independent alternate source world may reuse field:aux.",
        add_sources=(NamedSource(
            "field:aux", "Different sibling held A",
            UniformElectricField((10., 15., 40.))),))
    nodes = (
        StructuralWorldNode("add-a", "baseline", add, "events"),
        PhysicalWorldNode("mass", "add-a",
            "Change physical probe mass without altering actual plane.",
            (CoupledPropertyEdit("test-particle", "mass_kg", 2.2e-6),),
            "events"),
        StructuralWorldNode("replace-a", "mass", replacement, "events"),
        StructuralWorldNode("sibling", "baseline", sibling, "events"),
    )
    detector=(PlaneEvent("x:detector",(.075,0.,0.),(1.,0.,0.),
              "Passive positive x crossing of the actual oblique trajectory",
              direction=1),)
    return UnifiedObservationPlan(
        "m34a:oblique",original,nodes,"events",detector,
        EventPolicy(.006,1e-8,1e-8,1e-6),
        retain_source_ancestry=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,
                        default=Path("output/m34a-oblique-worlds.json"))
    args=parser.parse_args()
    plan=make_plan()
    frozen=plan.original.as_dict()
    run=run_unified_observation_graph(plan)
    data=run.as_dict()
    assert frozen==plan.original.as_dict()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,allow_nan=False,indent=2)+"\n",
                           encoding="utf-8")
    print(json.dumps({
        "output":str(args.output),
        "schema":data["schema"],
        "normal":list(plan.original.spec.plane_normal),
        "statuses":[row.status for row in run.entries],
        "source_paths":[row.source_ancestry.world_path
                        if row.source_ancestry is not None else None
                        for row in run.entries],
        "root_times_s":[next(event.sample.time_s for event in native.events
                             if event.event_id=="x:detector")
                        for native in (run.baseline_native,
                                       *(entry.native for entry in run.entries))],
    },sort_keys=True))


if __name__=="__main__":
    main()
