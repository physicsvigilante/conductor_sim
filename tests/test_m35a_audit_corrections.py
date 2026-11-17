"""Red-before/green-after released M34b audit defects; no force-law oracle reuse."""
from dataclasses import asdict, replace
from math import hypot, ulp
import random

import pytest

from conductor_sim import (
    CoupledPropertyEdit, GraphEventInquiryPlan, GraphWorldForkPlan,
    PhysicalWorldNode, StructuralWorldNode, UnifiedObservationPlan,
    WorldComparison, run_graph_world_fork, run_unified_observation_graph,
)
from conductor_sim.motion_events import PlaneEvent, EventPolicy, validate_event_requests
from conductor_sim.prescribed_fields import _unit, UniformElectricField, PrescribedField
from conductor_sim.model import PointCharge
from conductor_sim.prescribed_motion import TestChargeMotionSpec as MotionSpec
from conductor_sim.structural_family import StructuralCase
from conductor_sim.structural_scenarios import NamedSource
from examples.m34b_incremental_forks import build_source_graph


@pytest.fixture(scope='module')
def source():
    return build_source_graph()


def node(name='new'):
    return PhysicalWorldNode(name, 'old-a', 'Actual mass alternative',
        (CoupledPropertyEdit('test-particle', 'mass_kg', 2e-5),), 'events')


@pytest.mark.parametrize('kind', ('graph', 'fork', 'inquiry'))
def test_same_world_request_mutated_after_native_construction_refuses(source, kind):
    pair = WorldComparison('p', 'old-a', 'old-b')
    object.__setattr__(pair, 'right_world_id', 'old-a')
    with pytest.raises(ValueError, match='different physical worlds'):
        if kind == 'graph':
            replace(source.plan, comparisons=(pair,))
        elif kind == 'fork':
            GraphWorldForkPlan(source, 'audit:fork', (node(),),
                source.plan.questions, source.plan.policy, (pair,))
        else:
            GraphEventInquiryPlan(source, 'audit:inquiry', ('old-a','old-b'),
                source.plan.questions, source.plan.policy, (pair,))


@pytest.mark.parametrize('timing', ('before-plan', 'after-plan'))
def test_malformed_mode_rejected_before_numerical_work(source, monkeypatch, timing):
    import conductor_sim.graph_world_fork as module
    request = node()
    if timing == 'after-plan':
        plan = GraphWorldForkPlan(source, 'audit:mode', (request,),
            source.plan.questions, source.plan.policy)
    object.__setattr__(request, 'observation', 'typo-event')
    calls=[]
    monkeypatch.setattr(module, '_observe', lambda *a: calls.append(a))
    with pytest.raises(ValueError, match='observation'):
        if timing == 'before-plan':
            GraphWorldForkPlan(source, 'audit:mode', (request,),
                source.plan.questions, source.plan.policy)
        else:
            run_graph_world_fork(plan)
    assert calls == []


def test_base_graph_runner_rechecks_mutated_plan_before_any_solve(source, monkeypatch):
    import conductor_sim.unified_observation_graph as module
    old_plan=replace(source.plan)
    object.__setattr__(old_plan, 'baseline_observation', 'bad-mode')
    calls=[]
    monkeypatch.setattr(module, '_observe', lambda *a: calls.append(a))
    with pytest.raises(ValueError, match='baseline observation'):
        run_unified_observation_graph(old_plan)
    assert calls == []


@pytest.mark.parametrize('field,value', [
    ('node_id','x'*21), ('note','x'*401), ('edits',()), ('parent_id',False),
])
def test_mutated_native_node_nested_bounds_rechecked(source, field, value):
    request=node()
    object.__setattr__(request, field, value)
    with pytest.raises((TypeError, ValueError)):
        GraphWorldForkPlan(source,'audit:node',(request,),
            source.plan.questions,source.plan.policy)


def test_structural_case_and_physical_edit_rechecked(source):
    edit=CoupledPropertyEdit('test-particle','mass_kg',2e-5)
    request=PhysicalWorldNode('edit','old-a','Actual mass',(edit,),'events')
    object.__setattr__(edit,'value',False)
    with pytest.raises((TypeError, ValueError)):
        GraphWorldForkPlan(source,'audit:edit',(request,),
            source.plan.questions,source.plan.policy)
    case=StructuralCase('add','Actual addition',add_sources=(
        NamedSource('field:new','Additional E',UniformElectricField((1.,0.,0.))),))
    request=StructuralWorldNode('add','old-a',case,'events')
    object.__setattr__(case,'case_id','wrong-case')
    with pytest.raises((TypeError, ValueError)):
        GraphWorldForkPlan(source,'audit:case',(request,),
            source.plan.questions,source.plan.policy)


@pytest.mark.parametrize('field,value', [('direction',True),('normal',(2.,0.,0.)),
    ('label','x'*121),('point_m',(False,0.,0.)),('terminal',1),('role','arbitrary')])
def test_event_request_revalidation_is_pure_and_rejects_invalid_native(field,value):
    question=PlaneEvent('q',(.1,0.,0.),(1.,2.,-1.),'Physical plane',direction=1)
    object.__setattr__(question,field,value)
    before=asdict(question)
    with pytest.raises((TypeError,ValueError)):
        validate_event_requests((question,),EventPolicy(.01,1e-8,1e-8,1e-8))
    assert asdict(question)==before


def test_event_revalidation_does_not_renormalize_valid_nondyadic_axis():
    question=PlaneEvent('q',(.1,0.,0.),(1.,2.,-1.),'Oblique plane',direction=1)
    before=asdict(question)
    for _ in range(20):
        validate_event_requests((question,),EventPolicy(.01,1e-8,1e-8,1e-8))
    assert asdict(question)==before


def test_mutated_policy_rejected_before_new_study(source):
    policy=replace(source.plan.policy)
    object.__setattr__(policy,'max_events',True)
    with pytest.raises((TypeError,ValueError)):
        GraphEventInquiryPlan(source,'audit:policy',('baseline',),
            source.plan.questions,policy)


@pytest.mark.parametrize('v', [
    (ulp(0.),ulp(0.),0.),(ulp(0.),2*ulp(0.),-ulp(0.)),
    (-1e-320,3e-320,7e-320),(1e-310,-2e-310,4e-310),
])
def test_subnormal_direction_is_unit_and_normal_motion_reconstruction_idempotent(v):
    normal=_unit(v,'audit normal')
    assert abs(hypot(*normal)-1.) <= ulp(1.)
    spec=MotionSpec(PrescribedField((UniformElectricField((0.,0.,0.)),)),
        PointCharge(1e-9,(0.,0.,0.)),1e-5,(0.,0.,0.),.1,.01,plane_normal=v)
    assert abs(hypot(*spec.plane_normal)-1.) <= ulp(1.)
    assert replace(spec,mass_kg=2e-5).plane_normal == spec.plane_normal


def test_normal_range_arithmetic_unchanged_and_invalid_values_still_refuse():
    rng=random.Random(3501)
    for _ in range(200):
        v=tuple(rng.uniform(-1,1)*10**rng.randint(-100,100) for _ in range(3))
        length=hypot(*v)
        assert _unit(v,'ordinary') == tuple(x/length for x in v)
    for v in ((0.,0.,0.), (float('inf'),0.,0.),(float('nan'),0.,1.),
              (True,0.,0.),(1.,2.),(1.7e308,1.7e308,1.7e308)):
        with pytest.raises((ValueError,TypeError)):
            _unit(v,'invalid')
