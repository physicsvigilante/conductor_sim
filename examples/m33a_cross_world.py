"""M33a: explicit comparisons of separately qualified teacher hypotheses."""
import argparse
import json
from pathlib import Path

from conductor_sim import (
    CoupledPropertyEdit, PhysicalWorldNode, StructuralWorldNode,
    UnifiedObservationPlan, WorldComparison, run_unified_observation_graph,
)
from conductor_sim.motion_events import PlaneEvent, EventPolicy
from conductor_sim.structural_family import StructuralCase
from conductor_sim.structural_scenarios import NamedSource
from conductor_sim.prescribed_fields import UniformElectricField
from examples.m29a_structural_family import model


def make_plan():
    original = model("uniform-edits").baseline
    nodes = (
        StructuralWorldNode("field-low", "baseline", StructuralCase(
            "field-low", "Teacher hypothetical low additional +x field",
            add_sources=(NamedSource("field:added", "Additional low field",
                         UniformElectricField((40., 0., 0.))),)), "events"),
        StructuralWorldNode("field-high", "baseline", StructuralCase(
            "field-high", "Independent high-field hypothesis",
            add_sources=(NamedSource("field:added", "Additional high field",
                         UniformElectricField((120., 0., 0.))),)), "events"),
        PhysicalWorldNode("low-mass", "field-low", "Actual mass property revision",
            (CoupledPropertyEdit("test-particle", "mass_kg", 2e-5),), "events"),
    )
    return UnifiedObservationPlan(
        "m33a:demo", original, nodes, "events",
        (PlaneEvent("x:detector", (.006, 0., 0.), (1., 0., 0.),
                    "Passive x-crossing detector", direction=1),),
        EventPolicy(.003, 1e-8, 1e-8, 1e-6),
        retain_source_ancestry=True,
        comparisons=(
            WorldComparison("siblings", "field-low", "field-high"),
            WorldComparison("cousins", "field-high", "low-mass"),
            WorldComparison("original", "baseline", "field-high"),
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("output/m33a-cross-world.json"))
    args = parser.parse_args()
    run = run_unified_observation_graph(make_plan())
    payload = run.as_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, allow_nan=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"output": str(args.output), "schema": payload["schema"],
        "pairs": [{"id": x["comparison_id"], "status": x["status"],
                   "shared_samples": len(x["exact_time_trajectory_comparisons"]),
                   "event_statuses": [e["status"] for e in x["event_time_comparisons"]]}
                  for x in payload["cross_world_comparisons"]]}, sort_keys=True))


if __name__ == "__main__":
    main()
