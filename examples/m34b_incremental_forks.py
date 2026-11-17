"""M34b: append genuine new physical hypotheses to one ALREADY solved source graph.

The old baseline and two sibling M24a studies are built only once.
The fork then constructs/observes five NEW counterfactual worlds (one refused)
using M22a whole source edits, M20b physical mass edits and M32b branch-local
source identities. Old graph output must remain byte-for-byte unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from conductor_sim import (
    CoupledPropertyEdit, GraphWorldForkPlan, PhysicalWorldNode,
    StructuralWorldNode, UnifiedObservationPlan, UniformElectricField,
    WorldComparison, run_graph_world_fork, run_unified_observation_graph,
)
from conductor_sim.motion_events import EventPolicy, PlaneEvent
from conductor_sim.structural_family import StructuralCase
from conductor_sim.structural_scenarios import NamedSource, SourceReplacement
from examples.m29a_structural_family import model


def build_source_graph():
    original = model("uniform-edits").baseline
    questions = (PlaneEvent(
        "x:detector", (.006, 0., 0.), (1., 0., 0.),
        "A passive positive x-plane detector for actual native M24a event times",
        direction=1),)
    policy = EventPolicy(.003, 1e-8, 1e-8, 1e-6)
    nodes = (
        StructuralWorldNode("old-a", "baseline", StructuralCase(
            "old-a", "Teacher first fixed +x field alternative",
            add_sources=(NamedSource(
                "field:alt", "Alternative A new held field",
                UniformElectricField((40., 0., 0.))),)), "events"),
        StructuralWorldNode("old-b", "baseline", StructuralCase(
            "old-b", "Independent sibling allowed to introduce same source ID",
            add_sources=(NamedSource(
                "field:alt", "Alternative B distinct held field",
                UniformElectricField((120., 0., 0.))),)), "events"),
    )
    return run_unified_observation_graph(UnifiedObservationPlan(
        "m34b:original", original, nodes, "events",
        questions, policy, retain_source_ancestry=True))


def make_fork(graph):
    nodes = (
        PhysicalWorldNode(
            "mass-a", "old-a", "Actual changed probe mass in earlier A world",
            (CoupledPropertyEdit("test-particle", "mass_kg", 2e-5),),
            "events"),
        StructuralWorldNode("replace-a", "mass-a", StructuralCase(
            "replace-a", "Retire A's held field and install genuinely new held field",
            replace_sources=(SourceReplacement(
                "field:alt", NamedSource(
                    "field:replacement", "Later held replacement field",
                    UniformElectricField((90., 0., 0.)))),)), "events"),
        StructuralWorldNode("remove-a", "replace-a", StructuralCase(
            "remove-a", "Remove genuinely new held replacement source",
            remove_sources=("field:replacement",)), "events"),
        StructuralWorldNode("resurrect", "remove-a", StructuralCase(
            "resurrect", "Deliberate illegal retired field:alt resurrection",
            add_sources=(NamedSource(
                "field:alt", "Forbidden retired source label",
                UniformElectricField((50., 0., 0.))),)), "events"),
        PhysicalWorldNode(
            "mass-b", "old-b", "Independent valid sibling physical-mass revision",
            (CoupledPropertyEdit("test-particle", "mass_kg", 3e-5),),
            "events"),
    )
    return GraphWorldForkPlan(
        graph, "m34b:incremental", nodes,
        graph.plan.questions, graph.plan.policy,
        comparisons=(
            WorldComparison("old-to-new", "old-a", "mass-a"),
            WorldComparison("cross-branches", "old-b", "replace-a"),
            WorldComparison("new-branches", "remove-a", "mass-b"),
            WorldComparison("refused", "resurrect", "mass-b"),
        ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-output", type=Path,
                        default=Path("output/m34b-original-graph.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("output/m34b-new-world-fork.json"))
    args = parser.parse_args()
    old = build_source_graph()
    before = old.as_dict()
    plan = make_fork(old)
    fork = run_graph_world_fork(plan)
    result = fork.as_dict()
    assert old.as_dict() == before  # old SHA, observations and source histories immutable
    args.original_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.original_output.write_text(
        json.dumps(before, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    args.output.write_text(
        json.dumps(result, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(
        source_graph_file=str(args.original_output),
        fork_file=str(args.output),
        schema=result["schema"], source_graph_sha256=result["source_graph_sha256"],
        old_saved_worlds=1+len(old.entries),
        new_world_statuses=[e.status for e in fork.entries],
        pair_statuses=[x["status"] for x in result["cross_world_comparisons"]],
        event_pair_statuses=[
            [e["status"] for e in pair["event_time_comparisons"]]
            for pair in result["cross_world_comparisons"]],
    ), sort_keys=True))


if __name__ == "__main__":
    main()
