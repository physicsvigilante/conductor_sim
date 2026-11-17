"""M33b: ask NEW M24a detector questions after building one immutable graph.

Three actual source worlds are prepared and integrated as ordinary trajectories.
The same admitted physical models are later asked NEW signed x-detector questions
using fresh native EventStudy runs, not old saved-frame postprocessing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from conductor_sim import (
    CoupledPropertyEdit, GraphEventInquiryPlan, PhysicalWorldNode,
    StructuralWorldNode, UnifiedObservationPlan, UniformElectricField,
    WorldComparison, run_graph_event_inquiry, run_unified_observation_graph,
)
from conductor_sim.motion_events import PlaneEvent, EventPolicy
from conductor_sim.structural_family import StructuralCase
from conductor_sim.structural_scenarios import NamedSource
from examples.m29a_structural_family import model


def make_graph():
    original = model("uniform-edits").baseline
    nodes = (
        StructuralWorldNode("alpha", "baseline", StructuralCase(
            "alpha", "Teacher hypothetical A: additional +x held field",
            add_sources=(NamedSource("field:alternate", "Hypothesis A held field",
                UniformElectricField((40., 0., 0.))),)), "trajectory"),
        StructuralWorldNode("beta", "baseline", StructuralCase(
            "beta", "Teacher hypothetical B: different independent held field",
            add_sources=(NamedSource("field:alternate", "Hypothesis B held field",
                UniformElectricField((120., 0., 0.))),)), "trajectory"),
        PhysicalWorldNode("heavier", "alpha", "Vary genuine probe mass in A",
            (CoupledPropertyEdit("test-particle", "mass_kg", 2e-5),),
            "trajectory"),
    )
    return run_unified_observation_graph(
        UnifiedObservationPlan("m33b:original-worlds", original, nodes,
                               retain_source_ancestry=True))


def make_inquiry():
    graph = make_graph()
    return GraphEventInquiryPlan(
        graph, "m33b:new-x-question",
        ("baseline", "alpha", "beta", "heavier"),
        (PlaneEvent("x:followup", (.006, 0., 0.), (1., 0., 0.),
                    "New passive signed detector not asked by source graph",
                    direction=1),),
        EventPolicy(.003, 1e-8, 1e-8, 1e-6),
        comparisons=(
            WorldComparison("siblings", "alpha", "beta"),
            WorldComparison("cousins", "beta", "heavier"),
            WorldComparison("baseline", "baseline", "beta"),
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("output/m33b-inquiry.json"))
    parser.add_argument("--source-graph-output", type=Path, default=None)
    args = parser.parse_args()
    plan = make_inquiry()
    graph_before = plan.graph.as_dict()
    if args.source_graph_output is not None:
        args.source_graph_output.parent.mkdir(parents=True, exist_ok=True)
        args.source_graph_output.write_text(
            json.dumps(graph_before, allow_nan=False, indent=2)+"\n",
            encoding="utf-8")
    result = run_graph_event_inquiry(plan)
    data = result.as_dict()
    assert plan.graph.as_dict() == graph_before
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, allow_nan=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "file": str(args.output),
        "schema": data["schema"],
        "inquiry_statuses": [entry.status for entry in result.entries],
        "pair_statuses": [row["status"] for row in data["cross_world_comparisons"]],
        "event_statuses": [
            [event["status"] for event in row["event_time_comparisons"]]
            for row in data["cross_world_comparisons"]],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
