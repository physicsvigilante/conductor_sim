"""M32a: one graph of native physical source worlds and typed observations.

Export finite evidence JSON; no scene graph, browser-time ODE or imported-model
trust. All parameters are explicitly teacher-authored physical hypotheses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from conductor_sim import (
    CoupledPropertyEdit, PhysicalWorldNode, StructuralWorldNode,
    UnifiedObservationPlan, run_unified_observation_graph,
)
from conductor_sim.motion_events import EventPolicy, PlaneEvent
from examples.m29a_structural_family import model


def make_plan(case="uniform-events"):
    if case == "uniform-events":
        family = model("uniform-edits")
        cases = {s.case_id:s for s in family.cases}
        nodes = (
            StructuralWorldNode("add-E","baseline",cases["add-E"],"events"),
            StructuralWorldNode("remove-Ey","baseline",cases["remove-Ey"],"trajectory"),
            PhysicalWorldNode("source-mass","add-E",
                "Atomic change of a newly introduced field and a genuine probe mass.",
                (CoupledPropertyEdit("field:Enew","field_v_m",(150.,0.,0.)),
                 CoupledPropertyEdit("test-particle","mass_kg",2e-5)),
                "events"),
            PhysicalWorldNode("launch","source-mass",
                "Change actual launch velocity after jointly edited source world.",
                (CoupledPropertyEdit("test-particle","initial_velocity_m_s",(.16,0.,0.)),),
                "events"),
        )
        q = (PlaneEvent("x:detector",(.006,0.,0.),(1.,0.,0.),
                        "Signed original physical x detector", direction=1),)
        return UnifiedObservationPlan(
            "m32a:teacher",family.baseline,nodes,"events",q,
            EventPolicy(.003,1e-8,1e-8,1e-6))
    if case == "joint-translation":
        family = model("point-replace")
        nodes = (
            PhysicalWorldNode("both-z","baseline",
                "Move held point and probe physical initial conditions together.",
                (CoupledPropertyEdit("point:A","position_m",(.2,0.,.05)),
                 CoupledPropertyEdit("test-particle","position_m",(0.,.1,.05)))),
        )
        return UnifiedObservationPlan(
            "m32a:shift",family.baseline,nodes)
    raise ValueError("unsupported M32a example")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("uniform-events","joint-translation"),
                        default="uniform-events")
    parser.add_argument("--output",type=Path,default=Path("output/m32a-graph.json"))
    args=parser.parse_args()
    plan=make_plan(args.case)
    result=run_unified_observation_graph(plan)
    record=result.as_dict()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(
        json.dumps(record,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({
        "file":str(args.output),"original":plan.original.revision_id,
        "statuses":[r.status for r in result.entries],
        "parent_ids":[r.node.parent_id for r in result.entries],
        "schema":record["schema"],
        "provenance":"Teacher hypotheses, actual M22a/M20b source graph, native M24a events",
    },sort_keys=True))


if __name__=="__main__":
    main()
