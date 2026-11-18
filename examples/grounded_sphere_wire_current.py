"""M37a grounded sphere: teacher-authored instantaneous and actual-motion wire flux.

Produces JSON only. This is not a reproduction of the source workbook's
complete Multiple Correct Q21, which remains source/reviewer pending.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from conductor_sim import (
    ElectrostaticProblem, ExternalGroundedSphereBackend, Grounded,
    Motion1DSpec, PointCharge, SolidSphere, integrate_motion_1d,
)
from conductor_sim.grounded_sphere_wire import (
    observe_grounded_sphere_motion, observe_grounded_sphere_wire,
)


def example_data() -> dict:
    problem = ElectrostaticProblem(
        (PointCharge(1e-9, (0., 0., .2)),),
        (SolidSphere(.1, Grounded(), name="Sphere A"),),
    )
    observations = [
        observe_grounded_sphere_wire(problem, velocity).as_dict()
        for velocity in ((0., 0., -.1), (0., 0., +.1), (0., +.1, 0.))
    ]
    spec = Motion1DSpec(
        problem=problem, charge_index=0, mass_kg=1e-3,
        axis=(0., 0., 1.), initial_velocity_m_s=-.05,
        duration_s=.10, sample_interval_s=.02,
    )
    true_motion = integrate_motion_1d(spec, ExternalGroundedSphereBackend())
    trace = observe_grounded_sphere_motion(true_motion)
    return {
        "schema": "conductor_sim.m37a_teacher_wire_exploration.v1",
        "source_role": "teacher-authored numerical models; original workbook source p8 Q21 not yet individually revalidated",
        "instantaneous_prescribed": observations,
        "actual_elapsed_time": trace.as_dict(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("output/m37a-wire-current.json"))
    args = parser.parse_args()
    data = example_data()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    first = data["instantaneous_prescribed"][0]
    print("Teacher instantaneous current sphere→ground (A):",
          first["wire_current_sphere_to_ground_a"])
    print("Actual elapsed-time samples:",
          len(data["actual_elapsed_time"]["observations"]))
    print("Written:", args.output)


if __name__ == "__main__":
    main()
