"""M29a: compare independently solved physical trajectories after SOURCE MEMBERSHIP edits.

Real model changes are M22a atomic structural reassemblies, not edits to a
moving particle during playback or a replacement source under a stolen ID.
All numerical mass, initial conditions, duration and variations are explicit
teacher hypotheses; no source is a responding conductor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from conductor_sim import (
    FixedPointField, PointCharge, PrescribedField, TestChargeMotionSpec,
    UniformElectricField, UniformGravity, UniformSphereField,
)
from conductor_sim.scenarios import PrescribedScenario
from conductor_sim.structural_scenarios import NamedSource, SourceReplacement
from conductor_sim.structural_family import (
    StructuralCase, StructuralFamilyPlan, run_structural_family,
)
from conductor_sim.structural_family_viewer import write_structural_family_html


CASES=("uniform-edits","point-replace","mirror-pair","guard-and-refusal")


def model(case: str):
    if case=="uniform-edits":
        baseline=PrescribedScenario(
            "m29a:uniform",
            TestChargeMotionSpec(
                PrescribedField((UniformElectricField((100.,0.,0.)),
                                 UniformElectricField((0.,50.,0.)))),
                PointCharge(1e-9,(0.,0.,0.)),1e-5,(.1,0.,0.),
                .1,.01,plane_normal=(0.,0.,1.)),
            (("field:E0","Original physical +x field"),
             ("field:Ey","Original physical +y field")),
            "Teacher hypothetical: two fixed uniform electric fields; numerical "
            "mass, probe charge, launch and duration disclosed.")
        cases=(
            StructuralCase("add-E", "Teacher adds a separately named +x uniform E source.",
                           add_sources=(NamedSource(
                               "field:Enew","Independent additional +x field",
                               UniformElectricField((100.,0.,0.))),)),
            StructuralCase("remove-Ey", "Teacher removes the original +y field.",
                           remove_sources=("field:Ey",)),
            StructuralCase("add-gravity","Teacher adds uniform downward gravity.",
                           gravity_action="add",gravity=UniformGravity((0.,-2.,0.))),
            StructuralCase("replace-Ey","Teacher replaces +y field using a NEW source ID.",
                           replace_sources=(SourceReplacement(
                               "field:Ey",NamedSource(
                                   "field:Ey-new","Replacement +y field",
                                   UniformElectricField((0.,90.,0.)))),)),
        )
    elif case=="point-replace":
        baseline=PrescribedScenario(
            "m29a:point",
            TestChargeMotionSpec(
                PrescribedField((FixedPointField(1e-9,(.2,0.,0.)),)),
                PointCharge(1e-9,(0.,.1,0.)),1e-5,(.03,0.,0.),
                .04,.004,plane_normal=(0.,0.,1.),source_guard_m=.005),
            (("point:A","Original named positive fixed point source"),),
            "Teacher hypothetical fixed point and launched positive probe, "
            "not a real conducting sphere.")
        cases=(
            StructuralCase("sphere-new","Retire point A and add a finite volume sphere.",
                replace_sources=(SourceReplacement(
                    "point:A",NamedSource(
                        "sphere:S","NEW uniformly charged volume sphere, not metal",
                        UniformSphereField(1e-9,.35,(.2,0.,0.)))),)),
            StructuralCase("remove-point","Remove the original positive point source.",
                           remove_sources=("point:A",)),
        )
    elif case=="mirror-pair":
        baseline=PrescribedScenario(
            "m29a:mirror",
            TestChargeMotionSpec(
                PrescribedField((FixedPointField(1e-9,(.2,0.,0.)),)),
                PointCharge(1e-9,(0.,.04,0.)),1e-5,(.04,0.,0.),
                .04,.004,plane_normal=(0.,0.,1.),source_guard_m=.006),
            (("point:centre","Original in-plane held point"),),
            "Teacher hypothetical: reflection-qualified x-y motion plane.")
        cases=(
            StructuralCase("mirror-two","Add both reflected off-plane point sources atomically.",
                add_sources=(
                    NamedSource("point:upper","Held +z mirror",
                                FixedPointField(2e-9,(.12,.0,.08))),
                    NamedSource("point:lower","Held -z mirror",
                                FixedPointField(2e-9,(.12,.0,-.08))),
                )),
            StructuralCase("unpaired","Attempt to add only +z mirror source, must refuse.",
                add_sources=(NamedSource(
                    "point:single","Unpaired +z source",
                    FixedPointField(2e-9,(.12,0.,.08))),)),
            StructuralCase("extra-field","Add a separate in-plane upward field.",
                add_sources=(NamedSource(
                    "field:up","Teacher added uniform +y E",
                    UniformElectricField((0.,35.,0.))),)),
        )
    elif case=="guard-and-refusal":
        baseline=PrescribedScenario(
            "m29a:guard",
            TestChargeMotionSpec(
                PrescribedField((UniformElectricField((0.,0.,0.)),)),
                PointCharge(1e-9,(0.,0.,0.)),1e-4,(1.,0.,0.),
                .25,.01,plane_normal=(0.,0.,1.),source_guard_m=.005),
            (("field:original","Teacher neutral reference field"),),
            "Teacher hypothetical: ballistic probe with optional held sources.")
        cases=(
            StructuralCase("attract-guard","Add negative held source that censors the trajectory.",
                add_sources=(NamedSource(
                    "point:attract","New attracting point at x=.2m",
                    FixedPointField(-1e-10,(.2,0.,0.))),)),
            StructuralCase("far-positive","Add a distant held positive source that admits duration.",
                add_sources=(NamedSource(
                    "point:far","Far positive held source",
                    FixedPointField(1e-10,(.8,0.,0.))),)),
            StructuralCase("off-plane-refusal","Unpaired off-plane held point must refuse.",
                add_sources=(NamedSource(
                    "point:unpaired","Unqualified out-of-plane held point",
                    FixedPointField(1e-10,(.2,0.,.07))),)),
        )
    else:
        raise ValueError("unknown structural family case")
    return StructuralFamilyPlan("m29a:"+case,baseline,cases)


def run_case(case="uniform-edits"):
    return run_structural_family(model(case))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",choices=CASES,default="uniform-edits")
    parser.add_argument("--output",type=Path,default=Path("output/m29a-structural-family.html"))
    args=parser.parse_args()
    result=run_case(args.case)
    html=write_structural_family_html(
        result,args.output,title="M29a · Structural source family · "+args.case)
    record=html.with_suffix(".json")
    record.write_text(json.dumps(result.as_dict(),indent=2,allow_nan=False)+"\n",
                      encoding="utf-8")
    print(json.dumps(dict(
        case=args.case,html=str(html),json=str(record),
        baseline_termination=result.baseline_trajectory.termination,
        case_statuses=[x.status for x in result.entries],
        source_ids=[[sid for sid,label in x.assembly.scenario.source_annotations]
                    if x.assembly else None for x in result.entries],
        provenance="Each alternative is an independent teacher structural hypothesis.",
    ),sort_keys=True))


if __name__=="__main__":
    main()
