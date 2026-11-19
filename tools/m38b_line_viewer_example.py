"""M38b teacher-only, authentic off-axis Kelvin line and offline viewer export.

Never relabel this off-axis source as the original radial workbook p8 Q21.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from conductor_sim.prescribed_ground_wire import author_prescribed_grounded_sphere
from conductor_sim.prescribed_ground_line import (
    PrescribedSphereLineSpec, sample_prescribed_sphere_line,
)
from conductor_sim.prescribed_line_viewer import (
    line_view_manifest, render_prescribed_line_viewer,
)


def example_trace():
    model = author_prescribed_grounded_sphere(
        1e-9, (-0.4, 0.2, 0.0), 0.1, sphere_name="Sphere A"
    )
    return sample_prescribed_sphere_line(PrescribedSphereLineSpec(
        model, (0.2, 0., 0.), duration_s=4., sample_count=41,
        surface_guard_m=.01,
    ))


def run_all(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    trace = example_trace()
    native = trace.as_dict()
    viewer = line_view_manifest(trace)
    html = render_prescribed_line_viewer(trace)
    if viewer["trace"] != json.loads(json.dumps(native,allow_nan=False)):
        raise ValueError("M38b viewer lost genuine source Kelvin SI state identity")
    (output_dir/"m38b-line-physics.json").write_text(
        json.dumps(native,indent=2,allow_nan=False,ensure_ascii=False)+"\n", encoding="utf-8"
    )
    (output_dir/"m38b-line-viewer.json").write_text(
        json.dumps(viewer,indent=2,allow_nan=False,ensure_ascii=False)+"\n", encoding="utf-8"
    )
    (output_dir/"m38b-line.html").write_text(html,encoding="utf-8")
    return viewer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path("output/m38b"))
    args=parser.parse_args()
    m=run_all(args.output_dir)
    print("Teacher-only off-axis grounded-sphere physical viewer:",args.output_dir)
    print("True Kelvin SI states:",len(m["trace"]["samples"]))
    print("Closest passage:",m["trace"]["closest_approach_time_s"],"s")
    print("Signed conventional current at start/middle/end:",
          *[m["trace"]["samples"][j]["wire_current_sphere_to_ground_a"] for j in (0,20,40)])


if __name__ == "__main__":
    main()
