"""M27b — real particle motion in qualified planes of full charged rings.

The workbook-two-rings case preserves scanned p12 Q40–41's charges,
radius/separation and vertical external E, but its numeric probe mass,
launch position, velocity, duration and numerical guards are explicitly
teacher hypotheses. It is NOT the original symbolic Q41's full solution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from conductor_sim import (
    PointCharge, PrescribedField, TestChargeMotionSpec, UniformElectricField,
    UniformGravity, UniformRingField,
)
from conductor_sim.scenarios import PrescribedScenario
from conductor_sim.motion_events import (
    EventStudy, EventPolicy, PlaneEvent, VelocityReversal, investigate_motion,
)
from conductor_sim.motion_event_viewer import render_event_run
from conductor_sim.driven_motion import (
    DrivenModel, DrivenStudy, FieldPhase, LinearDrag,
    UniformFieldSchedule, investigate_driven_motion,
)
from conductor_sim.driven_viewer import render_driven_run

CASES=("workbook-xy", "single-meridian", "single-equatorial", "driven-switch")

POLICY=EventPolicy(.001, 1e-8, 1e-7, 1e-5)


def author(case: str):
    if case in ("workbook-xy","driven-switch"):
        sources=(
            UniformRingField(10e-6,.3,center_m=(0.,0.,0.),axis=(1.,0.,0.)),
            UniformRingField(-20e-6,.3,center_m=(.4,0.,0.),axis=(1.,0.,0.)),
            UniformElectricField((0.,1e5,0.)),
        )
        annotations=(("ring:A","Ring A · fixed positive ring"),
                     ("ring:B","Ring B · fixed negative ring"),
                     ("field:up","Upward uniform electric field"))
        particle=PointCharge(10e-6,(.2,0.,0.))
        mass=0.01  # teacher-chosen; original Q41 gives a symbolic m
        velocity=(.1,0.,0.)
        duration=.015
        normal=(0.,0.,-1.)  # world +x screen right and +y screen up
        gravity=None
        questions=(PlaneEvent("x:detector",(.275,0.,0.),(1.,0.,0.),
                              "Cross x = 0.275 m, passive detector",1),)
        provenance=(
            "Workbook physical p12 Q40–41: ring A +10 μC, B -20 μC, "
            "both radius .30 m, separation .40 m, +y E=100,000 V/m. "
            "TEST PARTICLE mass 0.01 kg, initial x=.20 m, v_x=.1 m/s, "
            "duration .015 s and motion plane are TEACHER HYPOTHESES. "
            "This is free physical 2D motion, NOT a guided axial Q41 path."
        )
    elif case=="single-meridian":
        sources=(UniformRingField(2e-9,.15),)
        annotations=(("ring:meridian","Full charged ring on world z axis"),)
        particle=PointCharge(8e-10,(.055,0.,.09))
        mass=2e-7
        velocity=(.03,0.,.02)
        duration=.06
        normal=(0.,1.,0.)
        gravity=None
        questions=(PlaneEvent("z:detector",(0.,0.,.097),(0.,0.,1.),
                              "Passive z detector"),)
        provenance="Teacher hypothetical: off-axis xz meridian ring launch, not a sourced JEE numerical problem."
    elif case=="single-equatorial":
        sources=(UniformRingField(2e-9,.15),)
        annotations=(("ring:equatorial","Full charged ring on its own xy plane"),)
        particle=PointCharge(8e-10,(.05,0.,0.))
        mass=2e-7
        velocity=(.0,.14,0.)
        duration=.06
        normal=(0.,0.,-1.)
        gravity=None
        questions=(PlaneEvent("y:detector",(0.,.006,0.),(0.,1.,0.),
                              "Passive equatorial y detector"),)
        provenance="Teacher hypothetical: off-axis ring-plane launch, no guiding wire or contact."
    else:
        raise ValueError("unknown M27b authored case")

    spec=TestChargeMotionSpec(
        PrescribedField(sources),particle,mass,velocity,duration,
        sample_interval_s=.001,plane_normal=normal,
        length_scale_m=.4,source_guard_m=.001,relative_tolerance=1e-9)
    base=PrescribedScenario("m27b:"+case,spec,annotations,provenance)
    return base, questions


def run_case(case):
    if case=="driven-switch":
        base,questions=author("workbook-xy")
        # A separate TEACHER hypothetical perturbation; the source-given
        # +y uniform 100 kV/m field stays in base, never silently replaced.
        model=DrivenModel(
            base, LinearDrag(1e-6),
            UniformFieldSchedule((
                FieldPhase(0.,(0.,0.,0.)),
                FieldPhase(.008,(0.,2e4,0.)),
            )),
            "m27b:driven-switch",
            "Teacher variation: add linear drag and +y 20 kV/m at .008 s. "
            "No impulse, field propagation or contact is implied.")
        result=investigate_driven_motion(DrivenStudy(model,questions,POLICY))
        html=render_driven_run(result,title="M27b · Full rings + drag and applied E switch")
        record=dict(provenance="Teacher scenario relative to original ring geometry",
                    numerical=result.as_dict(),applied_teacher_addition="20 kV/m +y at 0.008 s")
    else:
        base,questions=author(case)
        result=investigate_motion(EventStudy(base,questions,POLICY))
        html=render_event_run(result,title="M27b · "+case+" · actual curved off-axis motion")
        record=dict(provenance=base.provenance,numerical=result.as_dict())
    return html,record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case",choices=CASES,default="workbook-xy")
    p.add_argument("--output",type=Path,default=Path("output/m27b-workbook-xy.html"))
    args=p.parse_args()
    html,record=run_case(args.case)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(html,encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(record,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(dict(case=args.case,html=str(args.output),
        termination=record["numerical"]["trajectory"]["termination"]
            if "trajectory" in record["numerical"] else record["numerical"]["termination"],
        provenance="Original workbook geometry, explicit teacher launch and mass",
        output_json=str(args.output.with_suffix(".json"))),sort_keys=True))


if __name__=="__main__":
    main()
