"""Export teacher-only M40b real spherical geometry and native sigma atlas."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from conductor_sim.model import Grounded
from conductor_sim.spherical_multicavity import solve_spherical_sigma
from conductor_sim.spherical_sigma_atlas import (
    surface_atlas_manifest, render_surface_atlas,
)
from tools.m40a_spherical_example import teacher_model


def run_all(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    original = teacher_model()
    grounded = replace(
        original,
        gauss_assembly=replace(original.gauss_assembly, constraint=Grounded()),
    )
    isolated_solution = solve_spherical_sigma(original)
    grounded_solution = solve_spherical_sigma(grounded)
    manifest = surface_atlas_manifest(isolated_solution, grounded_solution)
    html = render_surface_atlas(isolated_solution, grounded_solution)
    (output_dir / "m40b-teacher-atlas.json").write_text(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False,
                   indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    (output_dir / "m40b-teacher-atlas.html").write_text(
        html, encoding="utf-8",
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/m40b"))
    args = parser.parse_args()
    result = run_all(args.output_dir)
    print("M40b genuine offline XY projection + sigma atlas:", args.output_dir)
    print("Cavities:", len(result["cavities"]), "JEE source items solved:",
          result["original_workbook_items_newly_solved"])


if __name__ == "__main__":
    main()
