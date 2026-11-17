"""Three real exploration stages; same physical root, no replay of earlier worlds.

This is native in-process reuse, not loading trusted physics from a JSON file.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from conductor_sim import (
    CoupledPropertyEdit, GraphWorldForkPlan, GraphEventInquiryPlan,
    PhysicalWorldNode, StructuralWorldNode, UnifiedObservationPlan,
    PointCharge, PrescribedField, TestChargeMotionSpec, UniformElectricField,
    UniformGravity, WorldComparison, run_graph_world_fork,
    run_graph_event_inquiry, run_unified_observation_graph,
)
from conductor_sim.motion_events import PlaneEvent, EventPolicy
from conductor_sim.scenarios import PrescribedScenario
from conductor_sim.structural_family import StructuralCase
from conductor_sim.structural_scenarios import NamedSource, SourceReplacement


def add(node_id, parent_id, source_id, vector):
    return StructuralWorldNode(node_id,parent_id,StructuralCase(
        node_id,'Independent fixed-source hypothesis',add_sources=(
            NamedSource(source_id,'Held '+source_id,UniformElectricField(vector)),)),
        'events')


def mass(node_id, parent_id, value):
    return PhysicalWorldNode(node_id,parent_id,'Genuine probe mass change',(
        CoupledPropertyEdit('test-particle','mass_kg',value),),'events')


def source_graph():
    original=PrescribedScenario('m35a:root',TestChargeMotionSpec(
        PrescribedField((UniformElectricField((180.,-30.,210.)),)),
        PointCharge(1.2e-9,(.003,.004,.0125)),2e-6,(.12,.03,.24),.7,.035,
        plane_normal=(3.,4.,-2.),gravity=UniformGravity((.03,-.02,.005))),
        (('field:base','Original held oblique field'),),
        'Teacher hypothetical. A certified plane, not unrestricted 3D motion.')
    question=(PlaneEvent('x:detector',(.05,0.,0.),(1.,0.,0.),
                        'Passive positive x crossing',direction=1),)
    policy=EventPolicy(.007,1e-9,1e-8,1e-8)
    return run_unified_observation_graph(UnifiedObservationPlan(
        'm35a:original',original,(
            add('old-a','baseline','field:shared',(40.,5.,70.)),
            add('old-b','baseline','field:shared',(-20.,25.,20.))),
        'events',question,policy,retain_source_ancestry=True))


def first_plan(graph):
    return GraphWorldForkPlan(graph,'m35a:stage1',(
        mass('mass-a','old-a',3e-6),
        add('extra-a','mass-a','field:temp',(30.,0.,45.))),
        graph.plan.questions,graph.plan.policy,
        (WorldComparison('first','old-a','extra-a'),))


def second_plan(first):
    replacement=StructuralWorldNode('replace-temp','extra-a',StructuralCase(
        'replace-temp','Retire old temporary source and introduce a new one',
        replace_sources=(SourceReplacement('field:temp',NamedSource(
            'field:final','Replacement held field',UniformElectricField((80.,-20.,80.)))),)),
        'events')
    return GraphWorldForkPlan(first,'m35a:stage2',(
        replacement,mass('mass-b','old-b',4e-6)),
        first.plan.questions,first.plan.policy,(
            WorldComparison('root-new','baseline','replace-temp'),
            WorldComparison('branches','replace-temp','mass-b')))


def third_plan(second):
    return GraphWorldForkPlan(second,'m35a:stage3',(
        add('resurrect-temp','replace-temp','field:temp',(30.,0.,45.)),
        StructuralWorldNode('remove-final','replace-temp',StructuralCase(
            'remove-final','Remove the replacement, retaining its retirement history',
            remove_sources=('field:final',)),'events'),
        mass('heavier','mass-b',5e-6),
        add('fresh-temp','heavier','field:temp',(30.,0.,45.))),
        second.plan.questions,second.plan.policy,(
            WorldComparison('old-new','extra-a','remove-final'),
            WorldComparison('new-branches','remove-final','fresh-temp'),
            WorldComparison('refused','resurrect-temp','fresh-temp')))


def inquiry_plan(third):
    questions=(PlaneEvent('x:followup',(.07,0.,0.),(1.,0.,0.),
                         'A genuinely new passive detector',direction=1),)
    return GraphEventInquiryPlan(third,'m35a:followup',
        ('baseline','extra-a','replace-temp','fresh-temp'),questions,
        third.plan.policy,(WorldComparison('later','replace-temp','fresh-temp'),))


def run_example():
    root=source_graph()
    before=root.as_dict()
    first=run_graph_world_fork(first_plan(root))
    first_before=first.as_dict()
    second=run_graph_world_fork(second_plan(first))
    second_before=second.as_dict()
    third=run_graph_world_fork(third_plan(second))
    inquiry=run_graph_event_inquiry(inquiry_plan(third))
    assert root.as_dict()==before and first.as_dict()==first_before
    assert second.as_dict()==second_before
    return root,first,second,third,inquiry


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('output/m35a'))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    results=run_example()
    for name,run in zip(('root','stage1','stage2','stage3','inquiry'),results):
        record=run.as_dict()
        (args.output/(name+'.json')).write_text(
            json.dumps(record,allow_nan=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(dict(name=name,schema=record['schema'],
            statuses=[row.status for row in run.entries]),sort_keys=True))


if __name__=='__main__':
    main()
