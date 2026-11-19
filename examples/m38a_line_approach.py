"""M38a: one genuine 3D teacher-prescribed line past a grounded sphere.

The original workbook p8 Q21 is a *radial* maintained-speed source.
This new off-axis case is an explicitly teacher-authored transfer, not an
additional original JEE question or a Newtonian freely moving orbit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from conductor_sim.prescribed_ground_wire import author_prescribed_grounded_sphere
from conductor_sim.prescribed_ground_line import (
    PrescribedSphereLineSpec, sample_prescribed_sphere_line,
)


def example_data():
    problem = author_prescribed_grounded_sphere(
        1e-9, (-0.4, 0.2, 0.0), 0.1, sphere_name="Sphere A"
    )
    # Closest approach at t=2 seconds (d=0.2 m > a+guard=0.11 m).
    # The *radial component* of velocity, and therefore the signed ideal
    # grounding-wire current, changes sign while the Cartesian v is constant.
    spec = PrescribedSphereLineSpec(
        problem, (0.2, 0., 0.), duration_s=4., sample_count=41,
        surface_guard_m=0.01,
    )
    return sample_prescribed_sphere_line(spec).as_dict()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("output/m38a/teacher-off-axis-line.json"))
    args = p.parse_args()
    data = example_data()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print("TEACHER extension, not original p8 question:", args.output)
    print("Real Kelvin observations:", len(data["samples"]))
    print("Continuous closest approach:", data["closest_approach_time_s"],
          "s at", data["closest_approach_distance_m"], "m")
    print("Signed sphere->ground current (start, closest, end):",
          *(data["samples"][i]["wire_current_sphere_to_ground_a"] for i in (0, 20, 40)))


if __name__ == "__main__":
    main()
