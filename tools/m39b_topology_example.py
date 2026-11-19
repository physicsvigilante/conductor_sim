"""Generate the actual teacher-only M39b Gauss cavity topology page and evidence.

No original JEE workbook problem is claimed as solved. This is not geometry.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from conductor_sim.cavity_gauss import solve_cavity_gauss
from conductor_sim.cavity_topology_viewer import (
    topology_view_manifest, render_cavity_topology,
)
from conductor_sim.model import Grounded
from tools.m39a_gauss_example import sample_assembly


def run_all(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = sample_assembly()
    isolated = solve_cavity_gauss(baseline)
    grounded = solve_cavity_gauss(replace(baseline, constraint=Grounded()))
    manifest = topology_view_manifest(isolated, grounded)
    html = render_cavity_topology(isolated, grounded)
    (output_dir / "m39b-topology-view.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "m39b-topology-view.html").write_text(html, encoding="utf-8")
    for case, ledger in (("isolated", isolated), ("grounded", grounded)):
        (output_dir / ("m39b-" + case + "-native-gauss.json")).write_text(
            json.dumps(ledger.as_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/m39b"))
    args = parser.parse_args()
    manifest = run_all(args.output_dir)
    print("Teacher topology-only offline view:", args.output_dir / "m39b-topology-view.html")
    print("Cavities:", len(manifest["cavities"]),
          "source-provenance:", manifest["source_role"],
          "new original JEE answers:", manifest["original_workbook_items_newly_solved"])


if __name__ == "__main__":
    main()
