"""M31a: atomically couple multiple physical changes in a real source world.

Each named hypothesis revises its selected qualified physical parent once;
there are no physically simulated intermediate models or source changes during
a single test particle trajectory.

Usage:
  python examples/m31a_coupled_physical_family.py --case added-mass --output output/m31a.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from conductor_sim import (
    CoupledPropertyEdit, CoupledHypothesis, CoupledPhysicalPlan,
    run_coupled_physical_family,
)
from examples.m29a_structural_family import model


CASES=("added-mass","joint-shift","gravity-electric","mixed-refusals")


def edit(component_id, property_name, value):
    return CoupledPropertyEdit(component_id, property_name, value)


def case(name, branch, note, *edits):
    return CoupledHypothesis(name, branch, note, tuple(edits))


def make_plan(example="added-mass"):
    if example=="added-mass":
        family=model("uniform-edits")
        requests=(
            case("added-mass","add-E",
                 "Use the genuine newly added source, plus a changed probe mass.",
                 edit("field:Enew","field_v_m",(150.,0.,0.)),
                 edit("test-particle","mass_kg",2e-5)),
            case("replaced-launch","replace-Ey",
                 "Revise the replacement source and real initial velocity together.",
                 edit("field:Ey-new","field_v_m",(0.,130.,0.)),
                 edit("test-particle","initial_velocity_m_s",(.2,0.,0.))),
        )
    elif example=="joint-shift":
        family=model("point-replace")
        requests=(
            case("joint-z-shift","baseline",
                 "Shift test-charge initial position and held point together by 0.05 m in z.",
                 edit("test-particle","position_m",(0.,.1,.05)),
                 edit("point:A","position_m",(.2,0.,.05))),
        )
    elif example=="gravity-electric":
        family=model("uniform-edits")
        requests=(
            case("added-gravity-E","add-gravity",
                 "Revise newly added gravity and existing held E without changing membership.",
                 edit("gravity","acceleration_m_s2",(0.,-3.,0.)),
                 edit("field:E0","field_v_m",(180.,0.,0.))),
        )
    elif example=="mixed-refusals":
        family=model("mirror-pair")
        requests=(
            case("source-refusal","unpaired",
                 "Unpaired off-plane held charge cannot be a physical parent.",
                 edit("test-particle","mass_kg",2e-5),
                 edit("point:single","charge_c",3e-9)),
            case("axis-refusal","extra-field",
                 "Do not silently target absent source from a different arrangement.",
                 edit("point:single","charge_c",2e-9),
                 edit("test-particle","mass_kg",2e-5)),
            case("valid-case","extra-field",
                 "A genuine admitted source and mass change.",
                 edit("field:up","field_v_m",(0.,50.,0.)),
                 edit("test-particle","mass_kg",2e-5)),
        )
    else:
        raise ValueError("unknown coupled hypothesis example")
    return CoupledPhysicalPlan("m31a:"+example,family,requests)


def run_case(example="added-mass"):
    return run_coupled_physical_family(make_plan(example))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",choices=CASES,default="added-mass")
    parser.add_argument("--output",type=Path,default=Path("output/m31a-coupled.json"))
    args=parser.parse_args()
    result=run_case(args.case)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result.as_dict(),allow_nan=False,indent=2)+"\n",
                           encoding="utf-8")
    print(json.dumps(dict(
        case=args.case,
        output=str(args.output),
        original_revision_id=result.plan.family.baseline.revision_id,
        results=[
            dict(hypothesis=r.hypothesis.case_id,structural_branch=r.hypothesis.structural_branch,
                 status=r.status,parent=None if r.parent_scenario is None else r.parent_scenario.revision_id,
                 revised=None if r.scenario is None else r.scenario.revision_id,
                 native_termination=None if r.trajectory is None else r.trajectory.termination)
            for r in result.entries],
        provenance="Distinct atomic teacher hypotheticals; source models remain immutable.",
    ),sort_keys=True))


if __name__=="__main__":
    main()
