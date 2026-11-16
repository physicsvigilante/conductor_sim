"""M29b: author identical detector/reversal questions over different source inventories.

Usage (offline physical numerical event report):
  python examples/m29b_structural_events.py --case uniform-detector --output output/m29b.html

Every source, detector and launch here is an explicit teacher hypothetical.
No source is an induced conductor, contact surface or moving during a run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from conductor_sim import (
    PointCharge, PrescribedField, TestChargeMotionSpec, UniformElectricField,
)
from conductor_sim.scenarios import PrescribedScenario
from conductor_sim.structural_scenarios import NamedSource
from conductor_sim.structural_family import StructuralCase, StructuralFamilyPlan
from conductor_sim.structural_event_family import (
    StructuralEventFamilyPlan, run_structural_event_family,
)
from conductor_sim.structural_event_viewer import write_structural_event_family_html
from conductor_sim.motion_events import PlaneEvent, EventPolicy
from examples.m29a_structural_family import model


CASES=("uniform-detector","guard-detector","terminal-refusal",
       "double-crossing","coincident-terminal","mirror-detector")


def make_plan(case="uniform-detector"):
    if case=="uniform-detector":
        family=model("uniform-edits")
        questions=(PlaneEvent(
            "x:detector",(.006,0.,0.),(1.,0.,0.),
            "Teacher passive crossing detector x=.006 m",direction=1),)
        policy=EventPolicy(.003,1e-8,1e-8,1e-6)
    elif case=="guard-detector":
        family=model("guard-and-refusal")
        questions=(
            PlaneEvent("x:early",(.05,0.,0.),(1.,0.,0.),
                       "Real early passive detector",direction=1),
            PlaneEvent("x:far",(.22,0.,0.),(1.,0.,0.),
                       "Passive far detector beyond attracting point",direction=1),
        )
        policy=EventPolicy(.007,1e-8,1e-7,1e-6)
    elif case=="mirror-detector":
        family=model("mirror-pair")
        questions=(
            PlaneEvent("x:mirrors",(.001,0.,0.),(1.,0.,0.),
                       "Detector for atomically added off-plane mirror pair",
                       direction=1),)
        policy=EventPolicy(.002,1e-9,1e-8,1e-6)
    elif case=="terminal-refusal":
        baseline=PrescribedScenario(
            "m29b:initial-stop-boundary",
            TestChargeMotionSpec(
                PrescribedField((UniformElectricField((-1000.,0.,0.)),)),
                PointCharge(1e-9,(0.,0.,0.)),1e-6,(0.,0.,0.),
                .25,.025,plane_normal=(0.,0.,1.)),
            (("field:Eneg","Held initial negative x electric field"),),
            "Teacher hypothetical: released at an ideal stop plane with "
            "the physical force initially directed into the permitted region.")
        family=StructuralFamilyPlan("m29b:stop-study",baseline,(
            StructuralCase(
                "reversed-force","New +x electric field reverses initial stop-side acceleration.",
                add_sources=(NamedSource(
                    "field:Eplus","New +2000 x physical electric field",
                    UniformElectricField((2000.,0.,0.))),)),
            StructuralCase(
                "more-negative","New -x electric field preserves admissible initial release.",
                add_sources=(NamedSource(
                    "field:Eminus","Additional negative electric field",
                    UniformElectricField((-1000.,0.,0.))),)),
        ))
        questions=(PlaneEvent(
            "stop:x0",(0.,0.,0.),(1.,0.,0.),
            "Ideal observed stop, no physical collision",direction=1,
            terminal=True,role="ideal-stop-surface"),)
        policy=EventPolicy(.01,1e-8,1e-8,1e-7)
    elif case=="double-crossing":
        baseline=PrescribedScenario(
            "m29b:retarding-original",
            TestChargeMotionSpec(
                PrescribedField((UniformElectricField((-1000.,0.,0.)),)),
                PointCharge(1e-9,(0.,0.,0.)),1e-6,(1.,0.,0.),
                2.,.1,plane_normal=(0.,0.,1.)),
            (("field:retarding","Held signed retarding electric field"),),
            "Teacher hypothetical: particle crosses one passive plane twice.")
        family=StructuralFamilyPlan("m29b:two-crossings",baseline,(
            StructuralCase(
                "extra-retarding","Add separate negative x field, keep both detector crossings.",
                add_sources=(NamedSource(
                    "field:more","New negative x electric field",
                    UniformElectricField((-200.,0.,0.))),)),
        ))
        questions=(PlaneEvent(
            "x:two-crossings",(.25,0.,0.),(1.,0.,0.),
            "Bidirectional passive crossing",direction=0),)
        policy=EventPolicy(.04,1e-9,1e-9,1e-8)
    elif case=="coincident-terminal":
        baseline=PrescribedScenario(
            "m29b:terminal-original",
            TestChargeMotionSpec(
                PrescribedField((UniformElectricField((1000.,0.,0.)),)),
                PointCharge(1e-6,(0.,0.,0.)),.001,(0.,0.,0.),
                .5,.025,plane_normal=(0.,0.,1.),
                relative_tolerance=1e-10),
            (("field:original","Held original positive x E"),),
            "Teacher hypothetical: many close ideal terminal observations.")
        family=StructuralFamilyPlan("m29b:terminal-family",baseline,(
            StructuralCase(
                "more-E","Add a separate positive x source, independently faster trajectory.",
                add_sources=(NamedSource(
                    "field:new","Additional held positive x E",
                    UniformElectricField((1000.,0.,0.))),)),
            StructuralCase(
                "zero-E","Add a named zero field while retaining the baseline trajectory.",
                add_sources=(NamedSource(
                    "field:zero","Physically zero additional source",
                    UniformElectricField((0.,0.,0.))),)),
        ))
        questions=tuple(PlaneEvent(
            f"terminal:{k:02}",(.05+k*1e-11,0.,0.),(1.,0.,0.),
            f"Ideal near-coincident terminal question {k:02}",
            direction=1,terminal=True,role="ideal-stop-surface")
            for k in range(16))
        policy=EventPolicy(.01,1e-8,1e-10,1e-8)
    else:
        raise ValueError("unknown structural event-family example")
    return StructuralEventFamilyPlan(family,questions,policy)


def run_case(case="uniform-detector"):
    return run_structural_event_family(make_plan(case))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",choices=CASES,default="uniform-detector")
    parser.add_argument("--output",type=Path,default=Path("output/m29b-events.html"))
    args=parser.parse_args()
    run=run_case(args.case)
    path=write_structural_event_family_html(
        run,args.output,title="M29b · Structural source events · "+args.case)
    record=path.with_suffix(".json")
    record.write_text(json.dumps(run.as_dict(),allow_nan=False,indent=2)+"\n",
                      encoding="utf-8")
    print(json.dumps(dict(
        case=args.case,html=str(path),json=str(record),
        baseline_termination=run.baseline_run.trajectory.termination,
        child_statuses=[entry.status for entry in run.entries],
        event_ids=[q.event_id for q in run.plan.questions],
        roots=[len(run.baseline_run.events)] +
              [len(x.event_run.events) if x.event_run is not None else None
               for x in run.entries],
        provenance="Each source membership is a teacher-authored, independently "
                   "requalified physical scenario; events are original M24a numerical roots.",
    ),sort_keys=True))


if __name__=="__main__":
    main()
