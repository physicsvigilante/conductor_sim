"""Export physically qualified static source relocation vs baseline, no invented motion."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from conductor_sim.spherical_sigma_relocation import (
    StaticSourceRelocation, compare_static_spherical_relocation,
)
from conductor_sim.spherical_sigma_relocation_viewer import render_static_sigma_comparison
from tools.m40a_spherical_example import teacher_model


def run_all(directory: Path):
    directory.mkdir(parents=True,exist_ok=True)
    original=teacher_model()
    change=StaticSourceRelocation(original, "NeutralButNonuniform","plus_A",(-1.45,.04,.01))
    result=compare_static_spherical_relocation(original,change)
    evidence=result.as_dict()
    (directory/"m40c-static-teacher.json").write_text(
        json.dumps(evidence,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    (directory/"m40c-static-teacher.html").write_text(
        render_static_sigma_comparison(result),encoding="utf-8")
    return evidence


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,default=Path("output/m40c"))
    args=p.parse_args()
    result=run_all(args.output_dir)
    print("M40c static teacher-only relocation:",args.output_dir,
          "cavities:",len(result["cavities"]),"new original JEE solutions: 0")

if __name__=="__main__":main()
