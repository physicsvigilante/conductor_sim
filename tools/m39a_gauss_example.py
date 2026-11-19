"""M39a reproducible teacher-only 3-cavity Gauss-law integral examples.

NOT workbook p11/p70 answer replays, not an irregular 3D field renderer.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from conductor_sim.model import FixedNetCharge, Grounded
from conductor_sim.cavity_gauss import (
    CavityCharge, SealedCavity, GaussCavityAssembly, solve_cavity_gauss,
)


def sample_assembly():
    """The same immutable topology can be evaluated under distinct constraints."""
    return GaussCavityAssembly(
        "Metal_A",
        (
            SealedCavity("cavity_A",(
                CavityCharge("source_A_positive",+2e-9),
                CavityCharge("source_A_negative",-2e-9),
            )),
            SealedCavity("cavity_B",(
                CavityCharge("source_B",-1e-9),
            )),
            SealedCavity("cavity_C",(
                CavityCharge("source_C",+3e-9),
            )),
        ),
        FixedNetCharge(+1e-9),
    )


def run_all(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    isolated=sample_assembly()
    # Explicitly separate independently authored static teacher scenarios.
    # This is not a wire time course or a numeric initial condition.
    grounded=replace(isolated,constraint=Grounded())
    results={}
    for label,model in (("isolated",isolated),("grounded",grounded)):
        trace=solve_cavity_gauss(model)
        record=trace.as_dict()
        (output_dir/f"m39a-{label}-teacher.json").write_text(
            json.dumps(record,indent=2,allow_nan=False)+"\n",encoding="utf-8"
        )
        results[label]=record
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path("output/m39a"))
    args=parser.parse_args()
    d=run_all(args.output_dir)
    for key,row in d.items():
        print(
            "TEACHER topology-only",key,
            "original-JEE-items:",row["original_workbook_items_newly_solved"],
            "inner faces:",
            [(x["cavity_id"],x["induced_inner_face_total_charge_c"])
             for x in row["cavities"]],
            "outer:",row["outer_face_total_charge_c"],"C",
            "flux:",row["exterior_electric_flux_v_m"],"V*m",
        )


if __name__=="__main__":
    main()
