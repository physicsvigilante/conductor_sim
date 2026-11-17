"""M34b: incrementally branch true certified worlds without re-solving the source graph."""
from dataclasses import replace
from math import sqrt

import pytest

from conductor_sim import (
    CoupledPropertyEdit, GraphWorldForkPlan, GraphWorldForkRun,
    PhysicalWorldNode, StructuralWorldNode, UnifiedObservationPlan,
    UniformElectricField, WorldComparison, run_graph_world_fork,
    run_unified_observation_graph,
)
from conductor_sim.motion_events import EventPolicy, PlaneEvent
from conductor_sim.structural_family import StructuralCase
from conductor_sim.structural_scenarios import NamedSource
from examples.m29a_structural_family import model
from examples.m29b_structural_events import make_plan as event_family
from examples.m34a_oblique_worlds import make_plan as oblique_plan
from examples.m34b_incremental_forks import build_source_graph, make_fork


def _exact_state(spec, t):
    ex = tuple(sum(src.field_v_m[k] for src in spec.field.sources)
               for k in range(3))
    g = ((0., 0., 0.) if spec.gravity is None
         else spec.gravity.acceleration_m_s2)
    q, mass = spec.particle.charge_c, spec.mass_kg
    acceleration = tuple(q*ex[k]/mass + g[k] for k in range(3))
    position = tuple(
        spec.particle.position_m[k] + spec.initial_velocity_m_s[k]*t
        + .5*acceleration[k]*t*t for k in range(3))
    velocity = tuple(spec.initial_velocity_m_s[k]+acceleration[k]*t
                     for k in range(3))
    force = tuple(q*ex[k]+mass*g[k] for k in range(3))
    return position, velocity, force, acceleration


def _root_x(spec, x):
    ex = sum(src.field_v_m[0] for src in spec.field.sources)
    grav = 0. if spec.gravity is None else spec.gravity.acceleration_m_s2[0]
    ax = spec.particle.charge_c*ex/spec.mass_kg + grav
    vx = spec.initial_velocity_m_s[0]
    distance = x-spec.particle.position_m[0]
    return 2*distance/(vx+sqrt(vx*vx+2*ax*distance))


def _worlds(graph, fork):
    worlds = {"baseline": (graph.plan.original, graph.baseline_native)}
    worlds.update((e.node.node_id, (e.scenario,e.native)) for e in graph.entries)
    worlds.update((e.node.node_id, (e.scenario,e.native)) for e in fork.entries)
    return worlds


def test_01_new_nodes_only_native_integration_and_3d_analytical_reference():
    graph = build_source_graph()
    initial = graph.as_dict()
    graph_saved_native = (graph.baseline_native, *(e.native for e in graph.entries))
    fork = run_graph_world_fork(make_fork(graph))
    assert type(fork) is GraphWorldForkRun
    assert [e.status for e in fork.entries] == [
        "complete","complete","complete","rejected-structural-model","complete"]
    assert graph.as_dict() == initial
    assert (graph.baseline_native, *(e.native for e in graph.entries)) == graph_saved_native
    worlds = _worlds(graph, fork)
    assert fork.entries[0].parent_scenario is graph.entries[0].scenario
    assert fork.entries[1].parent_scenario is fork.entries[0].scenario
    assert fork.entries[1].assembly.parent is fork.entries[0].scenario
    assert fork.entries[2].assembly.parent is fork.entries[1].assembly
    assert fork.entries[4].parent_scenario is graph.entries[1].scenario
    assert fork.entries[3].scenario is None and fork.entries[3].native is None
    assert fork.entries[3].source_ancestry is None
    assert fork.entries[1].source_ancestry.retired_source_ids == ("field:alt",)
    assert fork.entries[2].source_ancestry.retired_source_ids == (
        "field:alt","field:replacement")
    assert fork.entries[4].source_ancestry.world_path == (
        "baseline","old-b","mass-b")
    assert graph.entries[0].source_ancestry.introduced_source_ids == ("field:alt",)
    assert graph.entries[1].source_ancestry.introduced_source_ids == ("field:alt",)
    for world_id, (model, native) in worlds.items():
        if model is None:
            assert native is None
            continue
        assert native is not None, world_id
        assert native.trajectory.termination == "duration", world_id
        assert len(native.events) == 1
        assert native.events[0].sample.time_s == pytest.approx(
            _root_x(model.spec, .006), abs=2e-7)
        for sample in native.trajectory.samples:
            p,v,f,_ = _exact_state(model.spec,sample.time_s)
            assert sample.position_m == pytest.approx(p,abs=3e-8)
            assert sample.velocity_m_s == pytest.approx(v,abs=3e-8)
            assert sample.force_n == pytest.approx(f,abs=2e-12)
    data = fork.as_dict()
    assert data["schema"] == "conductor_sim.graph_world_fork.v1"
    assert data["plan"]["schema"] == "conductor_sim.graph_world_fork_plan.v1"
    assert data["plan"]["source_graph_schema"] == "conductor_sim.unified_observation_graph.v2"
    assert data["plan"]["source_graph_sha256"] == data["source_graph_sha256"]
    assert data["time_interpolation"] is False and data["postguard_extrapolation"] is False
    assert [p["status"] for p in data["cross_world_comparisons"]] == [
        "compared","compared","compared","unpaired"]
    for pair in data["cross_world_comparisons"][:3]:
        left, ln = worlds[pair["left_world_id"]]
        right, rn = worlds[pair["right_world_id"]]
        assert pair["left_revision_id"] == left.revision_id
        assert pair["right_revision_id"] == right.revision_id
        assert pair["event_study_compatible"] is True
        event = pair["event_time_comparisons"][0]
        assert event["status"] == "one-resolved-pair"
        assert event["estimated_numerical_shift_s"] == pytest.approx(
            _root_x(right.spec,.006)-_root_x(left.spec,.006),abs=3e-7)
        shared = {s.time_s for s in ln.trajectory.samples}
        rows = pair["exact_time_trajectory_comparisons"]
        assert [r["time_s"] for r in rows] == [
            s.time_s for s in rn.trajectory.samples if s.time_s in shared]
        for row in rows:
            t = row["time_s"]
            lp,lv,_,_ = _exact_state(left.spec,t)
            rp,rv,_,_ = _exact_state(right.spec,t)
            assert row["delta_position_m"] == pytest.approx(
                tuple(rp[k]-lp[k] for k in range(3)),abs=4e-8)
            assert row["delta_velocity_m_s"] == pytest.approx(
                tuple(rv[k]-lv[k] for k in range(3)),abs=4e-8)
    assert data["nodes"][3]["status"] == "rejected-structural-model"
    assert data["nodes"][3]["parent_revision_id"] == fork.entries[2].scenario.revision_id
    assert data["nodes"][3]["native_observation"] is None
    assert graph.as_dict() == initial


def test_02_new_question_changes_are_incompatible_for_event_time_not_actual_shared_times():
    graph = build_source_graph()
    new_question = (PlaneEvent(
        "x:detector",(.008,0.,0.),(1.,0.,0.),
        "Same event ID but genuinely different plane",direction=1),)
    original = make_fork(graph)
    plan = replace(original,questions=new_question)
    result = run_graph_world_fork(plan).as_dict()
    pair = result["cross_world_comparisons"][0]
    assert pair["status"]=="compared"
    assert pair["event_study_compatible"] is False
    assert pair["exact_time_trajectory_comparisons"]
    assert pair["event_time_comparisons"][0]["status"]=="unpaired"
    assert "different" in pair["event_time_comparisons"][0]["reason"]
    assert "estimated_numerical_shift_s" not in pair["event_time_comparisons"][0]


def test_03_same_question_new_policy_never_generates_fake_cross_study_event_shift():
    graph=build_source_graph()
    old=make_fork(graph)
    p=replace(old, policy=EventPolicy(.004,1e-8,1e-8,1e-6))
    data=run_graph_world_fork(p).as_dict()
    assert data["cross_world_comparisons"][0]["status"]=="compared"
    assert data["cross_world_comparisons"][0]["event_study_compatible"] is False
    assert data["cross_world_comparisons"][0]["event_time_comparisons"][0][
        "status"]=="unpaired"
    # Same new inquiry policy for two new worlds remains eligible.
    assert data["cross_world_comparisons"][2]["event_study_compatible"] is True


def test_04_trajectory_new_vs_existing_event_mode_explicitly_unpaired():
    graph=build_source_graph()
    new=PhysicalWorldNode("traj","old-a","New physically revised mass",
                          (CoupledPropertyEdit("test-particle","mass_kg",2e-5),),
                          "trajectory")
    p=GraphWorldForkPlan(graph,"m34b:traj",(new,),comparisons=(
        WorldComparison("cross","old-a","traj"),))
    run=run_graph_world_fork(p)
    assert run.entries[0].status=="complete"
    assert run.entries[0].native is not None
    rec=run.as_dict()["cross_world_comparisons"][0]
    assert rec["status"]=="unpaired"
    assert "modes" in rec["reason"]
    assert rec["exact_time_trajectory_comparisons"]==[]
    assert rec["event_time_comparisons"]==[]


def test_05_physically_rejected_parent_skips_child_but_sibling_runs():
    graph=build_source_graph()
    bad=PhysicalWorldNode("bad","old-a","Negative mass cannot qualify",
        (CoupledPropertyEdit("test-particle","mass_kg",-1.),),"trajectory")
    skipped=StructuralWorldNode("child","bad",StructuralCase(
        "child","Source of physically rejected model cannot be reassembled",
        add_sources=(NamedSource("field:new","New field",
                               UniformElectricField((40.,0.,0.))),)),"trajectory")
    good=PhysicalWorldNode("good","old-b","Independent valid mass",
        (CoupledPropertyEdit("test-particle","mass_kg",2.5e-5),),"trajectory")
    result=run_graph_world_fork(GraphWorldForkPlan(
        graph,"m34b:refused",(bad,skipped,good)))
    assert [e.status for e in result.entries]==[
        "rejected-physical-model","skipped-parent-model","complete"]
    assert all(e.scenario is None and e.source_ancestry is None and e.native is None
               for e in result.entries[:2])
    assert result.entries[2].scenario is not None
    assert result.entries[2].native is not None
    assert result.entries[2].source_ancestry.parent_world_id=="old-b"


def test_06_old_admitted_world_with_rejected_previous_event_study_can_be_forked():
    old=event_family("terminal-refusal")
    source=run_unified_observation_graph(UnifiedObservationPlan(
        "m34b:initial-stop",old.family.baseline,
        (StructuralWorldNode("reversed-force","baseline",old.family.cases[0],"events"),),
        "events",old.questions,old.policy,retain_source_ancestry=True))
    assert source.entries[0].scenario is not None
    assert source.entries[0].native is None
    assert source.entries[0].status=="rejected-event-study"
    p=GraphWorldForkPlan(source,"m34b:after-refusal",(
        PhysicalWorldNode("mass","reversed-force",
            "Genuine mass change after original study refused",
            (CoupledPropertyEdit("test-particle","mass_kg",2e-6),),
            "trajectory"),))
    result=run_graph_world_fork(p)
    assert result.entries[0].parent_scenario is source.entries[0].scenario
    assert result.entries[0].scenario is not None
    assert result.entries[0].status=="complete"
    assert result.entries[0].native is not None
    assert result.entries[0].source_ancestry.world_path==(
        "baseline","reversed-force","mass")


def test_07_genuine_source_guard_prefix_does_not_invent_later_event_shifts():
    family=event_family("guard-detector")
    old=run_unified_observation_graph(UnifiedObservationPlan(
        "m34b:guard-source",family.family.baseline,(
            StructuralWorldNode("far-positive","baseline",family.family.cases[1],"events"),
        ),"events",family.questions,family.policy,retain_source_ancestry=True))
    added=StructuralWorldNode(
        "attract","baseline",StructuralCase(
            "attract","New independent negative point field",
            add_sources=family.family.cases[0].add_sources),"events")
    result=run_graph_world_fork(GraphWorldForkPlan(
        old,"m34b:guard-fork",(added,),old.plan.questions,old.plan.policy,
        (WorldComparison("prefix","baseline","attract"),)))
    entry=result.entries[0]
    assert entry.status=="censored-source-guard"
    outcomes=entry.native.as_dict()["outcomes"]
    assert outcomes["x:early"]=="located"
    assert outcomes["x:far"]=="search-truncated"
    pair=result.as_dict()["cross_world_comparisons"][0]
    assert pair["status"]=="compared"
    assert pair["exact_time_trajectory_comparisons"]
    assert all(r["time_s"] <= entry.native.trajectory.samples[-1].time_s
               for r in pair["exact_time_trajectory_comparisons"])
    assert all(e["status"]=="unpaired" and
               "estimated_numerical_shift_s" not in e
               for e in pair["event_time_comparisons"])


def test_08_malformed_duplicate_forward_source_id_and_excess_requests_refused():
    graph=build_source_graph()
    good=PhysicalWorldNode("good","old-a","Actual mass",
        (CoupledPropertyEdit("test-particle","mass_kg",2e-5),),"trajectory")
    with pytest.raises(TypeError):
        GraphWorldForkPlan({"JSON":"not native"},"m34b:bad",(good,))
    with pytest.raises(ValueError):
        GraphWorldForkPlan(graph,"bad ID",(good,))
    with pytest.raises(ValueError):
        GraphWorldForkPlan(graph,"m34b:dup",(good,good))
    with pytest.raises(ValueError):
        GraphWorldForkPlan(graph,"m34b:reuse",(
            PhysicalWorldNode("old-a","baseline","Reuse old world ID",
                (CoupledPropertyEdit("test-particle","mass_kg",2e-5),)),))
    with pytest.raises(ValueError):
        GraphWorldForkPlan(graph,"m34b:forward",(
            PhysicalWorldNode("one","later","Forward model",
                (CoupledPropertyEdit("test-particle","mass_kg",2e-5),)),))
    with pytest.raises(ValueError):
        GraphWorldForkPlan(graph,"m34b:legacy-old-only",(good,),comparisons=(
            WorldComparison("old","baseline","old-a"),))
    with pytest.raises(ValueError):
        GraphWorldForkPlan(graph,"m34b:pair-duplicate",(good,),comparisons=(
            WorldComparison("one","old-a","good"),
            WorldComparison("two","good","old-a")))
    with pytest.raises(ValueError):
        GraphWorldForkPlan(graph,"m34b:pair-unselected",(good,),comparisons=(
            WorldComparison("bad","old-a","unknown"),))
    with pytest.raises(ValueError):
        GraphWorldForkPlan(graph,"m34b:list",[good])
    with pytest.raises(TypeError):
        GraphWorldForkPlan(graph,"m34b:raw",(("not","native"),))
    with pytest.raises(TypeError):
        GraphWorldForkPlan(graph,"m34b:badpolicy",(
            replace(good,observation="events"),),policy=False)
    with pytest.raises(TypeError):
        GraphWorldForkPlan(graph,"m34b:badquestions",(
            replace(good,observation="events"),),
            questions=[PlaneEvent("x:bad",(.008,0.,0.),(1.,0.,0.),"Bad list")],
            policy=graph.plan.policy)
    with pytest.raises(ValueError):
        GraphWorldForkPlan(graph,"m34b:too-many",(good,)*6)
    legacy=run_unified_observation_graph(UnifiedObservationPlan(
        "m34b:legacy",model("uniform-edits").baseline,(
        PhysicalWorldNode("heavy","baseline","New mass",
            (CoupledPropertyEdit("test-particle","mass_kg",2e-5),)),)))
    assert legacy.as_dict()["schema"]=="conductor_sim.unified_observation_graph.v1"
    with pytest.raises(ValueError,match="retained ancestry"):
        GraphWorldForkPlan(legacy,"m34b:noncert",(
            PhysicalWorldNode("other","baseline","New mass",
                (CoupledPropertyEdit("test-particle","mass_kg",2e-5),)),))


def test_09_result_issuance_status_model_and_bool_clock_integrity():
    graph=build_source_graph()
    p=make_fork(graph)
    run=run_graph_world_fork(p)
    with pytest.raises(ValueError,match="receipt"):
        GraphWorldForkRun(p,run.entries)
    with pytest.raises((ValueError,TypeError)):
        replace(run,entries=(replace(run.entries[0],status="forged"),)+run.entries[1:])
    with pytest.raises((ValueError,TypeError)):
        replace(run,entries=(replace(run.entries[0],reason="tampered"),)+run.entries[1:])
    with pytest.raises((ValueError,TypeError)):
        replace(run,plan=replace(p,fork_id="m34b:forged"))
    new=run.entries[0].native
    root=new.events[0]
    fake=replace(root,sample=replace(root.sample,time_s=root.sample.time_s+.03),
                 time_bracket_s=tuple(x+.03 for x in root.time_bracket_s))
    with pytest.raises((ValueError,TypeError)):
        replace(run,entries=(
            replace(run.entries[0],native=replace(new,events=(fake,))),
        )+run.entries[1:])
    native=new.trajectory
    assert native.samples[0].time_s==0.0
    altered=replace(native.samples[0],time_s=False)
    fake_trajectory=replace(native,samples=(altered,)+native.samples[1:])
    with pytest.raises((ValueError,TypeError)):
        replace(run,entries=(
            replace(run.entries[0],native=replace(new,trajectory=fake_trajectory)),
        )+run.entries[1:])
    deltas=run.as_dict()
    deltas["cross_world_comparisons"][0]["status"]="FAKE"
    assert run.as_dict()["cross_world_comparisons"][0]["status"]=="compared"


def test_10_source_receipt_must_remain_authentic_across_new_fork():
    graph=build_source_graph()
    p=make_fork(graph)
    before=graph.as_dict()
    result=run_graph_world_fork(p)
    assert graph.as_dict()==before
    altered=replace(graph.entries[0],status="forged-status")
    object.__setattr__(graph,"entries",(altered,)+graph.entries[1:])
    with pytest.raises((ValueError,TypeError)):
        result.as_dict()
    with pytest.raises((ValueError,TypeError)):
        run_graph_world_fork(p)


def test_11_m34a_non_dyadic_oblique_source_graph_can_genuinely_fork():
    base_plan=oblique_plan()
    graph=run_unified_observation_graph(base_plan)
    assert all(e.status=="complete" for e in graph.entries)
    note="Independent genuine non-dyadic in-plane source on qualified M34a parent"
    field=StructuralWorldNode("new-c","mass",StructuralCase(
        "new-c",note,add_sources=(NamedSource(
            "field:third","New independent held in-plane",
            UniformElectricField((60.,25.,110.))),)),"events")
    child=PhysicalWorldNode("new-mass","new-c",
        "Another independently qualified true probe mass",
        (CoupledPropertyEdit("test-particle","mass_kg",3e-6),),"events")
    p=GraphWorldForkPlan(graph,"m34b:non-dyadic",(field,child),
        graph.plan.questions,graph.plan.policy,
        (WorldComparison("old-vs-new","mass","new-c"),
         WorldComparison("new-vs-new","new-c","new-mass")))
    before=graph.as_dict()
    fork=run_graph_world_fork(p)
    assert [entry.status for entry in fork.entries]==["complete","complete"]
    assert fork.entries[0].parent_scenario is graph.entries[1].scenario
    assert fork.entries[1].parent_scenario is fork.entries[0].scenario
    assert fork.entries[0].scenario.spec.plane_normal == graph.plan.original.spec.plane_normal
    assert fork.entries[1].scenario.spec.plane_normal == graph.plan.original.spec.plane_normal
    assert fork.entries[0].source_ancestry.world_path==(
        "baseline","add-a","mass","new-c")
    assert graph.as_dict()==before
    for entry in fork.entries:
        for sample in entry.native.trajectory.samples:
            p0,v0,f,_=_exact_state(entry.scenario.spec,sample.time_s)
            assert sample.position_m==pytest.approx(p0,abs=3e-8)
            assert sample.velocity_m_s==pytest.approx(v0,abs=3e-8)
            assert sample.force_n==pytest.approx(f,abs=2e-12)
    assert all(item["status"]=="compared"
               for item in fork.as_dict()["cross_world_comparisons"])


def test_12_guard_and_native_refusal_limit_new_world_evidence_and_keep_siblings():
    family=event_family("terminal-refusal")
    source=run_unified_observation_graph(UnifiedObservationPlan(
        "m34b:stop-original",family.family.baseline,(
            StructuralWorldNode("more-negative","baseline",family.family.cases[1],"events"),
        ),"events",family.questions,family.policy,retain_source_ancestry=True))
    # A structurally valid revised +x force yields a new M24a study refusal.
    bad=StructuralWorldNode("force-inward","baseline",StructuralCase(
        "force-inward","New +x field means initial outward stop-plane acceleration",
        add_sources=family.family.cases[0].add_sources),"events")
    valid=PhysicalWorldNode("more-mass","more-negative","Actual valid mass",
        (CoupledPropertyEdit("test-particle","mass_kg",2e-6),),"events")
    fork=run_graph_world_fork(GraphWorldForkPlan(
        source,"m34b:stop-fork",(bad,valid),source.plan.questions,source.plan.policy))
    assert fork.entries[0].scenario is not None
    assert fork.entries[0].native is None
    assert fork.entries[0].status=="rejected-event-study"
    assert fork.entries[0].source_ancestry is not None
    assert fork.entries[1].status in ("complete","terminal-event")
    assert fork.entries[1].native is not None


def test_13_no_legacy_source_graph_mutation_or_export_leakage():
    graph=build_source_graph()
    snapshot=graph.as_dict()
    p=make_fork(graph)
    run=run_graph_world_fork(p)
    exported=run.as_dict()
    exported["nodes"][0]["source_ancestry"]["world_path"]=["FABRICATED"]
    exported["nodes"][0]["native_observation"]["trajectory"]["samples"][0]["time_s"]=False
    exported["plan"]["source_graph_sha256"]="FAKE"
    assert run.as_dict()["plan"]["source_graph_sha256"]!= "FAKE"
    assert run.as_dict()["nodes"][0]["source_ancestry"]["world_path"] != ["FABRICATED"]
    assert type(run.as_dict()["nodes"][0]["native_observation"]["trajectory"]["samples"][0]["time_s"]) is float
    assert graph.as_dict()==snapshot


def test_14_only_NEW_physical_worlds_are_integrated(monkeypatch):
    """Spy on the adapter's actual solve function, not a detached export."""
    import conductor_sim.graph_world_fork as fork_module
    graph = build_source_graph()
    snapshot = graph.as_dict()
    old_models = {graph.plan.original.scenario_id}
    old_models.update(e.scenario.scenario_id for e in graph.entries)
    invoked = []
    original = fork_module._observe

    def checked_observe(scenario, mode, questions, policy):
        assert scenario.scenario_id not in old_models, "fork reran old native source world"
        invoked.append(scenario.scenario_id)
        return original(scenario, mode, questions, policy)

    monkeypatch.setattr(fork_module, "_observe", checked_observe)
    run = run_graph_world_fork(make_fork(graph))
    expected = {e.scenario.scenario_id for e in run.entries if e.scenario is not None}
    assert set(invoked) == expected
    assert len(invoked) == len(expected) == 4
    assert graph.as_dict() == snapshot
    assert [e.status for e in run.entries].count("complete") == 4


def test_15_old_m33a_v3_cross_world_graph_stays_unchanged_after_fork():
    graph = build_source_graph()
    from dataclasses import replace as dc_replace
    source_plan = dc_replace(graph.plan, comparisons=(
        WorldComparison("old", "old-a", "old-b"),))
    source = run_unified_observation_graph(source_plan)
    assert source.as_dict()["schema"] == "conductor_sim.unified_observation_graph.v3"
    previous = source.as_dict()
    fork = run_graph_world_fork(make_fork(source))
    assert fork.as_dict()["plan"]["source_graph_schema"] == (
        "conductor_sim.unified_observation_graph.v3")
    assert fork.entries[1].source_ancestry.retired_source_ids == ("field:alt",)
    assert source.as_dict() == previous
    assert source.as_dict()["cross_world_comparisons"][0]["status"] == "compared"


def test_16_fork_preserves_two_real_roots_unpaired_even_when_saved_positions_compared():
    old = event_family("double-crossing")
    graph = run_unified_observation_graph(UnifiedObservationPlan(
        "m34b:two-old", old.family.baseline,
        (StructuralWorldNode("extra-retarding", "baseline",
                             old.family.cases[0], "events"),),
        "events", old.questions, old.policy, retain_source_ancestry=True))
    fork = run_graph_world_fork(GraphWorldForkPlan(
        graph, "m34b:two-new",
        (PhysicalWorldNode("slightly-lighter", "extra-retarding",
                           "A genuine changed physical probe mass",
                           (CoupledPropertyEdit("test-particle", "mass_kg", 0.9e-6),),
                           "events"),),
        old.questions, old.policy,
        (WorldComparison("two", "extra-retarding", "slightly-lighter"),)))
    assert fork.entries[0].status == "complete"
    assert len([e for e in fork.entries[0].native.events
                if e.event_id == "x:two-crossings"]) == 2
    comparison = fork.as_dict()["cross_world_comparisons"][0]
    assert comparison["status"] == "compared"
    assert comparison["exact_time_trajectory_comparisons"]
    assert comparison["event_time_comparisons"][0]["status"] == "unpaired"
    assert "estimated_numerical_shift_s" not in comparison["event_time_comparisons"][0]


def test_17_in_process_postconstruction_reason_forgery_detected_on_export():
    graph = build_source_graph()
    run = run_graph_world_fork(make_fork(graph))
    clone = replace(run.entries[0], reason="Invented initial condition and native result")
    object.__setattr__(run, "entries", (clone,) + run.entries[1:])
    with pytest.raises(ValueError, match="receipt"):
        run.as_dict()


def test_18_native_fork_depth_is_bounded_without_global_source_id_registry():
    graph = build_source_graph()
    edits = (CoupledPropertyEdit("test-particle", "mass_kg", 2e-5),)
    nodes = (
        PhysicalWorldNode("n1", "old-a", "New mass from old world", edits),
        PhysicalWorldNode("n2", "n1", "New mass from previous fork", edits),
        PhysicalWorldNode("n3", "n2", "New mass from previous fork", edits),
        PhysicalWorldNode("n4", "n3", "New mass from previous fork", edits),
        PhysicalWorldNode("n5", "n4", "New mass from previous fork", edits),
    )
    # old-a is at depth 1, n5 is depth 6: last admissible.
    GraphWorldForkPlan(graph, "m34b:depth-six", nodes)
    too_deep_graph = run_unified_observation_graph(UnifiedObservationPlan(
        "m34b:depth-old", graph.plan.original,
        (PhysicalWorldNode("first", "baseline", "True mass",
             (CoupledPropertyEdit("test-particle", "mass_kg", 2e-5),)),
         PhysicalWorldNode("second", "first", "Mass change again",
             (CoupledPropertyEdit("test-particle", "mass_kg", 3e-5),)),
         PhysicalWorldNode("third", "second", "Mass change once more",
             (CoupledPropertyEdit("test-particle", "mass_kg", 4e-5),)),
         PhysicalWorldNode("fourth", "third", "Fourth generation",
             (CoupledPropertyEdit("test-particle", "mass_kg", 5e-5),))),
        retain_source_ancestry=True))
    nodes2 = (
        PhysicalWorldNode("fifth", "fourth", "Fifth-generation mass",
            (CoupledPropertyEdit("test-particle", "mass_kg", 6e-5),)),
        PhysicalWorldNode("sixth", "fifth", "Sixth-generation mass",
            (CoupledPropertyEdit("test-particle", "mass_kg", 7e-5),)),
        PhysicalWorldNode("seventh", "sixth", "Unsupported seventh generation",
            (CoupledPropertyEdit("test-particle", "mass_kg", 8e-5),)),
    )
    with pytest.raises(ValueError, match="depth"):
        GraphWorldForkPlan(too_deep_graph, "m34b:seven", nodes2)
