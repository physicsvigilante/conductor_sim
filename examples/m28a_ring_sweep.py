"""M28a: independently integrate a bounded parameter sweep of M27b rings.

Usage:
  python examples/m28a_ring_sweep.py --axis ring-b-charge --output output/m28a.html
  python examples/m28a_ring_sweep.py --axis ring-b-separation --output output/m28a-separation.html

The PDF workbook gives both ring charges, radius, separation, E and test q.
The numeric mass, altered launch, duration and sweep variations are explicit
teacher hypotheses. This is NOT the original symbolic Q41 axial-rest solution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from conductor_sim import (SweepAxis, TrajectorySweepPlan, run_trajectory_sweep,
                           write_trajectory_sweep_html)
from examples.m27b_offaxis_motion import author

AXES = {
    "ring-b-charge": ("ring:B", "charge_c", (-18e-6, -22e-6)),
    "ring-b-separation": ("ring:B", "center_m", ((.35, 0., 0.), (.45, 0., 0.))),
    "mass": ("test-particle", "mass_kg", (.008, .012)),
    "launch-vx": ("test-particle", "initial_velocity_m_s",
                  ((.08, 0., 0.), (.12, 0., 0.))),
}


def run_case(axis_name="ring-b-charge"):
    if axis_name not in AXES:
        raise ValueError("unsupported sweep axis")
    baseline, _ = author("workbook-xy")
    component_id, property_name, values = AXES[axis_name]
    plan = TrajectorySweepPlan(
        "m28a:" + axis_name,
        baseline, SweepAxis(component_id, property_name, values))
    return run_trajectory_sweep(plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", choices=AXES, default="ring-b-charge")
    parser.add_argument("--output", type=Path, default=Path("output/m28a-ring-sweep.html"))
    args = parser.parse_args()
    experiment = run_case(args.axis)
    html = write_trajectory_sweep_html(
        experiment, args.output,
        title=f"M28a · Real curved ring trajectories · {args.axis}")
    record_path = html.with_suffix(".json")
    record_path.write_text(
        json.dumps(experiment.as_dict(), indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "axis": args.axis,
        "baseline_termination": experiment.baseline_trajectory.termination,
        "case_statuses": [x.status for x in experiment.entries],
        "html": str(html), "json": str(record_path),
        "source_provenance": "Workbook p12 Q40–41 geometry, field and probe charge; "
                             "teacher numeric mass/launch/duration/sweep changes",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
