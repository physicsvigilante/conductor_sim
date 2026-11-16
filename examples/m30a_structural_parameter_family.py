"""M30a: source inventory membership edits followed by property sweeps.

This teacher-authored demonstration reuses M22a/M29a + M20b/M28a without
new electrostatic laws or numerical methods. Every child is a separate
hypothesis; no source changes physically during a particle trajectory.

Usage:
  python examples/m30a_structural_parameter_family.py --case added-field --output output/m30a.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from conductor_sim import (
    BranchSweep, StructuralParameterPlan, SweepAxis,
    run_structural_parameter_family,
)
from examples.m29a_structural_family import model


CASES=("added-field","replaced-field","gravity","mixed-refusal")


def make_plan(case: str = "added-field") -> StructuralParameterPlan:
    if case == "added-field":
        family = model("uniform-edits")
        branches = (
            BranchSweep("add-E", SweepAxis(
                "field:Enew", "field_v_m",
                ((50.,0.,0.),(150.,0.,0.),(250.,0.,0.)))),
            BranchSweep("remove-Ey", SweepAxis(
                "test-particle", "mass_kg", (2e-5,5e-6))),
        )
    elif case == "replaced-field":
        family = model("uniform-edits")
        branches = (
            BranchSweep("replace-Ey", SweepAxis(
                "field:Ey-new", "field_v_m",
                ((0.,20.,0.),(0.,120.,0.)))),
        )
    elif case == "gravity":
        family = model("uniform-edits")
        branches = (
            BranchSweep("add-gravity", SweepAxis(
                "gravity", "acceleration_m_s2",
                ((0.,-1.,0.),(0.,-4.,0.)))),
        )
    elif case == "mixed-refusal":
        family = model("mirror-pair")
        branches = (
            BranchSweep("unpaired", SweepAxis(
                "point:single","charge_c",(1e-9,3e-9))),
            BranchSweep("extra-field", SweepAxis(
                "point:single","charge_c",(1e-9,3e-9))),
            BranchSweep("mirror-two", SweepAxis(
                "test-particle","mass_kg",(2e-5,3e-5))),
        )
    else:
        raise ValueError("unknown teacher example")
    return StructuralParameterPlan("m30a:"+case,family,branches)


def run_case(case: str = "added-field"):
    return run_structural_parameter_family(make_plan(case))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",choices=CASES,default="added-field")
    parser.add_argument("--output",type=Path,default=Path("output/m30a-branches.json"))
    args=parser.parse_args()
    result=run_case(args.case)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result.as_dict(),allow_nan=False,indent=2)+"\n",
                           encoding="utf-8")
    print(json.dumps(dict(
        case=args.case,output=str(args.output),
        baseline_sha=result.plan.family.baseline.revision_id,
        branch_statuses=[e.status for e in result.entries],
        variation_statuses=[
            None if e.sweep_run is None else
            [v.status for v in e.sweep_run.entries]
            for e in result.entries],
        provenance="Teacher hypotheticals: source structure first, physical property second. "
                   "All variations independently validated and integrated.",
    ),sort_keys=True))


if __name__ == "__main__":
    main()
