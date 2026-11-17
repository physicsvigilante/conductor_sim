"""Scientific and adversarial tests for exact native multi-stage reuse."""
from dataclasses import asdict, replace
from math import sqrt
import json
import pytest
from conductor_sim import (
    CoupledPropertyEdit, GraphWorldForkPlan, GraphWorldForkRun,
    PhysicalWorldNode, WorldComparison, run_graph_world_fork,
    run_graph_event_inquiry,
)
from conductor_sim.motion_events import PlaneEvent, EventPolicy
from conductor_sim.graph_world_fork import _source_context, _receipt
from examples.m35a_multistage_exploration import (
    source_graph,first_plan,second_plan,third_plan,inquiry_plan,run_example,
)


def canonical(run):
    return json.dumps(run.as_dict(),sort_keys=True,separators=(',',':'),allow_nan=False)


def state(spec,t):
    # Independently expressed Cartesian Newtonian equations; no source callback.
    electric=tuple(sum(s.field_v_m[k] for s in spec.field.sources) for k in range(3))
    gravity=spec.gravity.acceleration_m_s2
    force=tuple(spec.particle.charge_c*electric[k]+spec.mass_kg*gravity[k]
                for k in range(3))
    a=tuple(f/spec.mass_kg for f in force)
    p=tuple(spec.particle.position_m[k]+spec.initial_velocity_m_s[k]*t+.5*a[k]*t*t
            for k in range(3))
    v=tuple(spec.initial_velocity_m_s[k]+a[k]*t for k in range(3))
    return p,v,force,a


def root_time(spec,x):
    a=state(spec,0.)[3][0]
    dx=x-spec.particle.position_m[0]
    v=spec.initial_velocity_m_s[0]
    return 2*dx/(v+sqrt(v*v+2*a*dx))


def test_three_native_stages_new_only_solves_and_complete_physical_ancestry(monkeypatch):
    import conductor_sim.graph_world_fork as module
    root=source_graph()
    root_bytes=canonical(root)
    original_objects=(root.baseline_native,*(e.native for e in root.entries))
    calls=[]; observe=module._observe
    def counted(model,mode,questions,policy):
        calls.append(model.scenario_id)
        return observe(model,mode,questions,policy)
    monkeypatch.setattr(module,'_observe',counted)
    one=run_graph_world_fork(first_plan(root)); one_bytes=canonical(one)
    two=run_graph_world_fork(second_plan(one)); two_bytes=canonical(two)
    three=run_graph_world_fork(third_plan(two))
    assert len(calls)==7 and len(set(calls))==7
    assert not any(x.startswith('m35a:original:') or x=='m35a:root' for x in calls)
    assert canonical(root)==root_bytes and canonical(one)==one_bytes
    assert canonical(two)==two_bytes
    assert root.baseline_native is original_objects[0]
    assert all(e.native is n for e,n in zip(root.entries,original_objects[1:]))
    assert one.as_dict()['schema']=='conductor_sim.graph_world_fork.v1'
    assert two.as_dict()['schema']==three.as_dict()['schema']=='conductor_sim.graph_world_fork.v2'
    assert one.entries[1].assembly.parent is one.entries[0].scenario
    assert two.entries[0].assembly.parent is one.entries[1].assembly
    assert three.entries[1].assembly.parent is two.entries[0].assembly
    assert three.entries[0].status=='rejected-structural-model'
    assert three.entries[0].scenario is three.entries[0].native is None
    assert three.entries[0].source_ancestry is None
    assert three.entries[1].source_ancestry.retired_source_ids==('field:temp','field:final')
    assert three.entries[3].source_ancestry.introduced_source_ids==('field:temp',)
    assert three.entries[3].source_ancestry.world_path==(
        'baseline','old-b','mass-b','heavier','fresh-temp')
    record=three.as_dict()
    assert record['source_history_sha256']==_receipt(two.as_dict())
    assert record['original_graph_sha256']==_receipt(root.as_dict())
    assert [p['fork_id'] for p in record['prior_stages']]==['m35a:stage1','m35a:stage2']
    assert record['cumulative_world_count']==10
    assert record['stage_number']==3
    inquiry=run_graph_event_inquiry(inquiry_plan(three))
    assert inquiry.as_dict()['schema']=='conductor_sim.graph_event_inquiry.v2'
    assert inquiry.entries[1].scenario is one.entries[1].scenario
    assert inquiry.entries[2].scenario is two.entries[0].scenario
    assert inquiry.entries[3].scenario is three.entries[3].scenario
    assert all(e.status=='complete' for e in inquiry.entries)
    assert len(calls)==7, 'inquiry has its own genuine event solver, not a fork replay'


def test_independent_oblique_si_states_event_roots_and_comparison_signs():
    original,one,two,three,inquiry=run_example()
    context=_source_context(three)
    checked=0
    for world in context.worlds.values():
        if world.model is None:
            assert world.native is None
            continue
        native=world.native; spec=world.model.spec
        assert native.trajectory.termination=='duration'
        assert native.events[0].sample.time_s==pytest.approx(root_time(spec,.05),abs=1e-7)
        for sample in native.trajectory.samples:
            p,v,f,_=state(spec,sample.time_s)
            assert sample.position_m==pytest.approx(p,abs=5e-8)
            assert sample.velocity_m_s==pytest.approx(v,abs=5e-8)
            assert sample.force_n==pytest.approx(f,abs=2e-12)
            checked+=1
    assert checked>=200
    for run in (one,two,three):
        for comparison in run.as_dict()['cross_world_comparisons']:
            if comparison['status']=='unpaired':
                assert comparison['left_revision_id'] is None
                continue
            lhs=context.worlds[comparison['left_world_id']]
            rhs=context.worlds[comparison['right_world_id']]
            for row in comparison['exact_time_trajectory_comparisons']:
                lp,lv,*_=state(lhs.model.spec,row['time_s'])
                rp,rv,*_=state(rhs.model.spec,row['time_s'])
                assert row['delta_position_m']==pytest.approx(tuple(rp[k]-lp[k] for k in range(3)),abs=1e-7)
                assert row['delta_velocity_m_s']==pytest.approx(tuple(rv[k]-lv[k] for k in range(3)),abs=1e-7)
            pair=comparison['event_time_comparisons'][0]
            assert pair['status']=='one-resolved-pair'
            assert pair['estimated_numerical_shift_s']==pytest.approx(
                root_time(rhs.model.spec,.05)-root_time(lhs.model.spec,.05),abs=2e-7)
    for entry in inquiry.entries:
        assert entry.native.events[0].sample.time_s==pytest.approx(
            root_time(entry.scenario.spec,.07),abs=2e-7)


def test_question_or_policy_difference_does_not_pair_ancestor_event_times():
    root=source_graph(); one=run_graph_world_fork(first_plan(root))
    fresh=(PlaneEvent('x:detector',(.06,0.,0.),(1.,0.,0.),'Changed geometry, same ID',direction=1),)
    for updates in ({'questions':fresh}, {'policy':EventPolicy(.008,1e-9,1e-8,1e-8)}):
        two=run_graph_world_fork(replace(second_plan(one),**updates))
        pair=two.as_dict()['cross_world_comparisons'][0]
        assert pair['status']=='compared' and pair['exact_time_trajectory_comparisons']
        assert pair['event_study_compatible'] is False
        assert pair['event_time_comparisons'][0]['status']=='unpaired'
        assert 'estimated_numerical_shift_s' not in pair['event_time_comparisons'][0]


@pytest.mark.parametrize('which', ['root','first','second'])
def test_any_ancestor_forgery_invalidates_descendant_export_and_new_solve(which,monkeypatch):
    root,one,two,three,_=run_example()
    target={'root':root,'first':one,'second':two}[which]
    altered=replace(target.entries[0],reason='forged ancestor reason')
    object.__setattr__(target,'entries',(altered,)+target.entries[1:])
    import conductor_sim.graph_world_fork as module
    calls=[]
    monkeypatch.setattr(module,'_observe',lambda *args:calls.append(args))
    with pytest.raises(ValueError,match='receipt'):
        three.as_dict()
    with pytest.raises(ValueError,match='receipt'):
        GraphWorldForkPlan(three,'m35a:bad',(PhysicalWorldNode('bad','baseline','mass',(
            CoupledPropertyEdit('test-particle','mass_kg',7e-6),)),))
    assert calls==[]


def test_cycle_is_detected_iteratively_before_solver_or_recursion_error():
    root=source_graph(); first=run_graph_world_fork(first_plan(root))
    second=run_graph_world_fork(second_plan(first))
    object.__setattr__(first.plan,'source_graph',second)
    with pytest.raises(ValueError,match='cycle'):
        second.as_dict()


def test_new_node_or_stage_id_cannot_overwrite_any_ancestor():
    root=source_graph(); first=run_graph_world_fork(first_plan(root))
    with pytest.raises(ValueError,match='fork ID'):
        replace(second_plan(first),fork_id=first.plan.fork_id)
    original=second_plan(first).nodes[1]
    with pytest.raises(ValueError,match='reuse'):
        replace(second_plan(first),nodes=(replace(original,node_id='extra-a'),),comparisons=())
    with pytest.raises(TypeError):
        GraphWorldForkPlan(first.as_dict(),'m35a:json',(original,))


def test_subsequent_fork_of_refused_world_skips_only_its_descendants():
    root,one,two,three,_=run_example()
    plan=GraphWorldForkPlan(three,'m35a:stage4',(
        PhysicalWorldNode('skipped','resurrect-temp','No physical model exists',(
            CoupledPropertyEdit('test-particle','mass_kg',6e-6),)),
        PhysicalWorldNode('good','fresh-temp','Valid different branch',(
            CoupledPropertyEdit('test-particle','mass_kg',6e-6),))))
    result=run_graph_world_fork(plan)
    assert [e.status for e in result.entries]==['skipped-parent-model','complete']
    assert result.entries[0].scenario is result.entries[0].native is None
    assert result.entries[0].source_ancestry is None


def test_false_zero_clock_and_changed_source_digest_rejected_after_issuance():
    root=source_graph(); one=run_graph_world_fork(first_plan(root))
    two=run_graph_world_fork(second_plan(one))
    native=two.entries[0].native
    sample=replace(native.trajectory.samples[0],time_s=False)
    fake=replace(native,trajectory=replace(native.trajectory,
        samples=(sample,)+native.trajectory.samples[1:]))
    with pytest.raises((ValueError,TypeError)):
        replace(two,entries=(replace(two.entries[0],native=fake),)+two.entries[1:])
    exported=two.as_dict()
    exported['source_history_sha256']='FAKE'
    exported['prior_stages'][0]['run_sha256']='FAKE'
    assert two.as_dict()['source_history_sha256']!='FAKE'
    assert two.as_dict()['prior_stages'][0]['run_sha256']!='FAKE'


def test_history_validation_visits_each_prior_stage_once_per_export(monkeypatch):
    import conductor_sim.graph_world_fork as module
    root,one,two,three,_=run_example()
    original=module._validate_run
    calls=[]
    def observe(run,context):
        calls.append(run.plan.fork_id)
        return original(run,context)
    monkeypatch.setattr(module,'_validate_run',observe)
    three.as_dict()
    assert calls==['m35a:stage1','m35a:stage2','m35a:stage3']


def test_eight_stage_budget_cannot_reset():
    previous=source_graph()
    for i in range(8):
        previous=run_graph_world_fork(GraphWorldForkPlan(previous,f'limit:stage{i}',(
            PhysicalWorldNode(f'n{i}','baseline','Independent mass',(
                CoupledPropertyEdit('test-particle','mass_kg',(i+3)*1e-6),)),)))
    assert previous.as_dict()['stage_number']==8
    with pytest.raises(ValueError,match='eight fork stages'):
        GraphWorldForkPlan(previous,'limit:ninth',(
            PhysicalWorldNode('ninth','baseline','Too many stages',(
                CoupledPropertyEdit('test-particle','mass_kg',12e-6),)),))


def test_cumulative_world_budget_counts_refused_worlds_too():
    previous=source_graph()
    for stage in range(6):
        nodes=tuple(PhysicalWorldNode(f's{stage}n{i}','baseline','Rejected negative mass',(
            CoupledPropertyEdit('test-particle','mass_kg',-1.),)) for i in range(5))
        previous=run_graph_world_fork(GraphWorldForkPlan(previous,f'budget:stage{stage}',nodes))
    assert previous.as_dict()['cumulative_world_count']==32
    with pytest.raises(ValueError,match='world-count'):
        GraphWorldForkPlan(previous,'budget:more',(
            PhysicalWorldNode('one-more','baseline','No budget reset',(
                CoupledPropertyEdit('test-particle','mass_kg',3e-6),)),))


def test_cumulative_physical_depth_twelve_is_not_reset_between_stages():
    previous = source_graph()
    parent = 'old-a'  # depth one
    for stage, count in enumerate((5, 5, 1)):
        nodes = []
        for i in range(count):
            name = f'd{stage}n{i}'
            nodes.append(PhysicalWorldNode(name, parent, 'True new mass', (
                CoupledPropertyEdit('test-particle', 'mass_kg', (stage+3)*1e-6),)))
            parent = name
        previous = run_graph_world_fork(GraphWorldForkPlan(
            previous, f'depth:stage{stage}', tuple(nodes)))
    assert len(previous.entries[-1].source_ancestry.world_path) - 1 == 12
    with pytest.raises(ValueError, match='depth'):
        GraphWorldForkPlan(previous, 'depth:too-deep', (
            PhysicalWorldNode('thirteenth', parent, 'Too deep', (
                CoupledPropertyEdit('test-particle', 'mass_kg', 7e-6),)),))


def test_v2_parent_origin_distinguishes_old_graph_from_earlier_fork():
    root = source_graph()
    one = run_graph_world_fork(first_plan(root))
    two = run_graph_world_fork(second_plan(one)).as_dict()
    assert two['nodes'][0]['parent_origin'] == 'prior_fork'
    assert two['nodes'][1]['parent_origin'] == 'source_graph'
