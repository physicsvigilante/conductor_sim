"""Emit five authentic historical outputs; run unchanged under base and candidate.

This tool intentionally uses ONLY APIs present in released M34b. It is not
an importer and does not manufacture old native graph or observation objects.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from conductor_sim import UnifiedObservationPlan, WorldComparison, run_unified_observation_graph
from conductor_sim import GraphEventInquiryPlan, run_graph_event_inquiry, run_graph_world_fork
from conductor_sim.motion_events import PlaneEvent
from examples.m34b_incremental_forks import build_source_graph, make_fork


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = build_source_graph()
    results = {}
    for name, ancestry, comparisons in (
        ('graph-v1', False, ()), ('graph-v2', True, ()),
        ('graph-v3', True, (WorldComparison('siblings', 'old-a', 'old-b'),)),
    ):
        plan = replace(root.plan, plan_id='audit:' + name,
                       retain_source_ancestry=ancestry, comparisons=comparisons)
        results[name] = run_unified_observation_graph(plan).as_dict()
    results['fork-v1'] = run_graph_world_fork(make_fork(root)).as_dict()
    inquiry = GraphEventInquiryPlan(
        root, 'audit:old-inquiry', ('baseline', 'old-a'),
        (PlaneEvent('x:new', (.008, 0., 0.), (1., 0., 0.),
                    'New passive detector', direction=1),), root.plan.policy,
        (WorldComparison('pair', 'baseline', 'old-a'),))
    results['inquiry-v1'] = run_graph_event_inquiry(inquiry).as_dict()
    hashes = {}
    for name, record in results.items():
        data = (json.dumps(record, sort_keys=True, separators=(',', ':'),
                           allow_nan=False) + '\n').encode('utf-8')
        (args.output / (name + '.json')).write_bytes(data)
        hashes[name] = sha256(data).hexdigest()
    print(json.dumps(hashes, sort_keys=True))


if __name__ == '__main__':
    main()
