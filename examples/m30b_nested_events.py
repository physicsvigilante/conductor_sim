"""M30b: native event observations along structural→parameter physics branches.

Every source inventory and each parameter variation is a SEPARATE teacher
hypothetical. Numerical events use the existing physical M24a integrator,
not event markers inferred from HTML or saved-trajectory interpolation.

Usage:
  python examples/m30b_nested_events.py --case added-field --output output/m30b.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from conductor_sim import (
    BranchSweep, NestedEventPlan, StructuralParameterPlan, SweepAxis,
    run_nested_event_family,
)
from conductor_sim.motion_events import EventPolicy, PlaneEvent
from examples.m29a_structural_family import model
from examples.m29b_structural_events import make_plan as event_example


CASES=("added-field","gravity","guard","terminal-refusal","double-crossing")


def make_plan(case="added-field"):
    if case=="added-field":
        physical=StructuralParameterPlan("m30b:added-field",model("uniform-edits"),(
            BranchSweep("add-E",SweepAxis(
                "field:Enew","field_v_m",((50.,0.,0.),(160.,0.,0.)))),
            BranchSweep("replace-Ey",SweepAxis(
                "field:Ey-new","field_v_m",((0.,70.,0.),))),
        ))
        questions=(PlaneEvent("x:detector",(.006,0.,0.),(1.,0.,0.),
                              "Same positive-x passive detector",direction=1),)
        policy=EventPolicy(.003,1e-8,1e-8,1e-6)
    elif case=="gravity":
        physical=StructuralParameterPlan("m30b:gravity",model("uniform-edits"),(
            BranchSweep("add-gravity",SweepAxis(
                "gravity","acceleration_m_s2",((0.,-1.,0.),(0.,-4.,0.)))),))
        questions=(PlaneEvent("y:down",(0.,-.003,0.),(0.,1.,0.),
                              "Passive downward detector",direction=-1),)
        policy=EventPolicy(.003,1e-8,1e-8,1e-6)
    elif case=="guard":
        previous=event_example("guard-detector")
        physical=StructuralParameterPlan("m30b:guard",previous.family,(
            BranchSweep("attract-guard",SweepAxis(
                "point:attract","charge_c",(-5e-11,-2e-10))),))
        questions,policy=previous.questions,previous.policy
    elif case=="terminal-refusal":
        previous=event_example("terminal-refusal")
        physical=StructuralParameterPlan("m30b:terminal",previous.family,(
            BranchSweep("reversed-force",SweepAxis(
                "field:Eplus","field_v_m",((100.,0.,0.),(500.,0.,0.)))),))
        questions,policy=previous.questions,previous.policy
    elif case=="double-crossing":
        previous=event_example("double-crossing")
        physical=StructuralParameterPlan("m30b:double",previous.family,(
            BranchSweep("extra-retarding",SweepAxis(
                "field:more","field_v_m",((-100.,0.,0.),(-350.,0.,0.)))),))
        questions,policy=previous.questions,previous.policy
    else:
        raise ValueError("unsupported teacher-authored case")
    return NestedEventPlan(physical,questions,policy)


def run_case(case="added-field"):
    return run_nested_event_family(make_plan(case))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",choices=CASES,default="added-field")
    parser.add_argument("--output",type=Path,default=Path("output/m30b-events.json"))
    args=parser.parse_args()
    result=run_case(args.case)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(
        result.as_dict(),allow_nan=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(
        case=args.case,
        output=str(args.output),
        original_revision_id=result.plan.experiment.family.baseline.revision_id,
        original_termination=result.original_event_run.trajectory.termination,
        branches=[
            dict(branch=entry.physical_entry.request.branch_id,
                 physical_status=entry.physical_entry.status,
                 native_event_status=entry.status,
                 property_native_event_statuses=[x.status for x in entry.variations])
            for entry in result.branches],
        provenance=("Separate teacher-authored source worlds and physical properties; "
                    "original native M24a event clocks, no numerical interpolation."),
    ),sort_keys=True))


if __name__=="__main__":
    main()
