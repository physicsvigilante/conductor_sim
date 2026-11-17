"""M32b: structural→physical→structural→physical on one native source graph.

All sources and changes are explicit teacher-authored stationary hypotheticals.
No moving held source, conductor induction, source registry or browser ODE.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from conductor_sim import (
    CoupledPropertyEdit, PhysicalWorldNode, StructuralWorldNode,
    UnifiedObservationPlan, UniformElectricField,
    run_unified_observation_graph,
)
from conductor_sim.motion_events import EventPolicy, PlaneEvent
from conductor_sim.structural_family import StructuralCase
from conductor_sim.structural_scenarios import NamedSource, SourceReplacement
from examples.m29a_structural_family import model


def make_plan():
    family=model("uniform-edits")
    initial_add=next(case for case in family.cases if case.case_id=="add-E")
    replacement=StructuralCase(
        "replace-after","Retire edited source Enew, introduce genuinely NEW Eafter.",
        replace_sources=(SourceReplacement(
            "field:Enew",NamedSource(
                "field:Eafter","New stationary x-field source after physical edit",
                UniformElectricField((130.,0.,0.)))),))
    nodes=(
        StructuralWorldNode("add-E","baseline",initial_add,"events"),
        PhysicalWorldNode("source-mass","add-E",
            "Change newly introduced held field and real probe mass simultaneously.",
            (CoupledPropertyEdit("field:Enew","field_v_m",(150.,0.,0.)),
             CoupledPropertyEdit("test-particle","mass_kg",2e-5)), "events"),
        StructuralWorldNode("replace-after","source-mass",replacement,"events"),
        PhysicalWorldNode("final","replace-after",
            "Revise the genuine new held field and physical initial velocity.",
            (CoupledPropertyEdit("field:Eafter","field_v_m",(180.,0.,0.)),
             CoupledPropertyEdit("test-particle","initial_velocity_m_s",(.16,0.,0.))),"events"),
    )
    return UnifiedObservationPlan(
        "m32b:mix",family.baseline,nodes,"events",
        (PlaneEvent("x:detector",(.006,0.,0.),(1.,0.,0.),
                    "Passive signed physical x detector",direction=1),),
        EventPolicy(.003,1e-8,1e-8,1e-6),
        retain_source_ancestry=True,
    )


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=Path("output/m32b-mixed.json"))
    args=p.parse_args()
    out=args.output
    out.parent.mkdir(parents=True,exist_ok=True)
    result=run_unified_observation_graph(make_plan())
    payload=result.as_dict()
    out.write_text(json.dumps(payload,allow_nan=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({
        "file":str(out), "schema":payload["schema"],
        "statuses":[r.status for r in result.entries],
        "parents":[r.node.parent_id for r in result.entries],
        "retired_sources":[r.source_ancestry.retired_source_ids
                           for r in result.entries],
    },sort_keys=True))


if __name__=="__main__":
    main()
