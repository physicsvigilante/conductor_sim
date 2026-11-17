"""M33b: new native M24a questions on unchanged qualified source-world graphs."""
from dataclasses import replace
from math import sqrt

import pytest

from conductor_sim import (
    CoupledPropertyEdit, GraphEventInquiryPlan, GraphEventInquiryRun,
    PhysicalWorldNode, StructuralWorldNode, UnifiedObservationPlan,
    UniformElectricField, WorldComparison, run_graph_event_inquiry,
    run_unified_observation_graph,
)
from conductor_sim.motion_events import EventPolicy, EventRun, PlaneEvent
from conductor_sim.structural_family import StructuralCase
from conductor_sim.structural_scenarios import NamedSource
from examples.m29a_structural_family import model
from examples.m29b_structural_events import make_plan
from examples.m33b_event_inquiry import make_graph, make_inquiry


def xquestion(x=.006):
    return PlaneEvent("x:new", (x, 0., 0.), (1., 0., 0.),
                      "New independent passive signed detector", direction=1)


def policy():
    return EventPolicy(.003, 1e-8, 1e-8, 1e-6)


def physical_status_graph():
    original = model("uniform-edits").baseline
    return run_unified_observation_graph(UnifiedObservationPlan(
        "m33b:refusal-source", original,
        (PhysicalWorldNode("bad", "baseline", "Negative mass is not a physical world",
            (CoupledPropertyEdit("test-particle", "mass_kg", -1.),)), 
         PhysicalWorldNode("good", "baseline", "Physically qualified alternate probe",
            (CoupledPropertyEdit("test-particle", "mass_kg", 2e-5),))),
        retain_source_ancestry=True))


def test_01_new_event_question_reuses_source_models_and_computes_real_new_m24a_roots():
    p = make_inquiry()
    g = p.graph
    snapshot = g.as_dict()
    assert g.plan.questions == () and g.plan.comparisons == ()
    assert all(entry.native is not None for entry in g.entries)
    run = run_graph_event_inquiry(p)
    assert type(run) is GraphEventInquiryRun
    assert [e.status for e in run.entries] == ["complete"] * 4
    assert all(type(e.native) is EventRun for e in run.entries)
    byid = {"baseline": g.plan.original}
    byid.update((e.node.node_id, e.scenario) for e in g.entries)
    for entry in run.entries:
        model = byid[entry.world_id]
        assert entry.scenario is model
        assert entry.native.study.scenario is model
        assert entry.native.study.questions is p.questions
        assert entry.native.study.policy is p.policy
        assert entry.source_ancestry.model_revision_id == model.revision_id
        q = model.spec.particle.charge_c
        mass = model.spec.mass_kg
        electric = tuple(sum(s.field_v_m[k] for s in model.spec.field.sources)
                         for k in range(3))
        accel = tuple(q * electric[k] / mass for k in range(3))
        vx = model.spec.initial_velocity_m_s[0]
        dx = .006 - model.spec.particle.position_m[0]
        true_root = 2*dx/(vx+sqrt(vx*vx+2*accel[0]*dx))
        roots = [e for e in entry.native.events if e.event_id=="x:followup"]
        assert len(roots)==1 and roots[0].direction==1
        assert roots[0].sample.time_s == pytest.approx(true_root,abs=1e-7)
        for sample in entry.native.trajectory.samples:
            t=sample.time_s
            pos = tuple(model.spec.particle.position_m[k]+
                        model.spec.initial_velocity_m_s[k]*t+.5*accel[k]*t*t
                        for k in range(3))
            vel = tuple(model.spec.initial_velocity_m_s[k]+accel[k]*t
                        for k in range(3))
            assert sample.position_m == pytest.approx(pos,abs=2e-8)
            assert sample.velocity_m_s == pytest.approx(vel,abs=2e-8)
    d = run.as_dict()
    assert d["schema"] == "conductor_sim.graph_event_inquiry.v1"
    assert d["plan"]["schema"] == "conductor_sim.graph_event_inquiry_plan.v1"
    assert d["plan"]["source_graph_schema"] == "conductor_sim.unified_observation_graph.v2"
    assert d["time_interpolation"] is False
    assert d["postguard_extrapolation"] is False
    assert [row["status"] for row in d["cross_world_comparisons"]] == [
        "compared"]*3
    for pair in d["cross_world_comparisons"]:
        l = next(e for e in run.entries if e.world_id == pair["left_world_id"])
        r = next(e for e in run.entries if e.world_id == pair["right_world_id"])
        assert pair["left_revision_id"] == l.scenario.revision_id
        assert pair["right_revision_id"] == r.scenario.revision_id
        assert pair["event_time_comparisons"][0]["status"] == "one-resolved-pair"
        expected = r.native.events[0].sample.time_s-l.native.events[0].sample.time_s
        assert pair["event_time_comparisons"][0]["estimated_numerical_shift_s"] == pytest.approx(expected)
        shared = set(e.time_s for e in l.native.trajectory.samples)
        assert [row["time_s"] for row in pair["exact_time_trajectory_comparisons"]] == [
            e.time_s for e in r.native.trajectory.samples if e.time_s in shared]
    assert g.as_dict()==snapshot
    d["nodes"][0]["native_event_run"]["events"][0]["sample"]["time_s"] = False
    assert run.as_dict()["nodes"][0]["native_event_run"]["events"][0]["sample"]["time_s"] != False


def test_02_second_new_question_set_never_changes_prior_native_worlds():
    graph = make_graph()
    frozen = graph.as_dict()
    p1 = GraphEventInquiryPlan(graph, "m33b:first", ("baseline","alpha"),
        (xquestion(.006),), policy())
    p2 = GraphEventInquiryPlan(graph, "m33b:second", ("baseline","alpha"),
        (xquestion(.008),), policy())
    first = run_graph_event_inquiry(p1)
    second = run_graph_event_inquiry(p2)
    assert all(e.status=="complete" for e in (*first.entries,*second.entries))
    assert first.entries[0].scenario is second.entries[0].scenario is graph.plan.original
    assert first.entries[0].native is not second.entries[0].native
    assert second.entries[0].native.events[0].sample.time_s > (
        first.entries[0].native.events[0].sample.time_s)
    assert graph.as_dict()==frozen
    assert not graph.plan.questions  # original world graph was ordinary trajectory mode


def test_03_physically_rejected_model_never_fabricates_new_event_run():
    graph = physical_status_graph()
    assert graph.entries[0].scenario is None
    query = GraphEventInquiryPlan(graph, "m33b:refused", ("bad", "good"),
        (xquestion(),), policy(),
        (WorldComparison("bad-good", "bad", "good"),))
    result=run_graph_event_inquiry(query)
    assert [e.status for e in result.entries] == ["skipped-parent-model", "complete"]
    assert result.entries[0].native is None
    row=result.as_dict()["cross_world_comparisons"][0]
    assert row["status"]=="unpaired"
    assert row["left_revision_id"] is None
    assert row["left_source_ancestry"] is None
    assert row["event_time_comparisons"][0]["status"] == "unpaired"
    assert row["exact_time_trajectory_comparisons"] == []


def test_04_old_event_study_refused_but_new_passive_detector_can_succeed():
    old=make_plan("terminal-refusal")
    g=run_unified_observation_graph(UnifiedObservationPlan(
        "m33b:old-study",old.family.baseline,(
            StructuralWorldNode("reversed-force","baseline",old.family.cases[0],
                                "events"),),
        "events",old.questions,old.policy,retain_source_ancestry=True))
    assert g.entries[0].scenario is not None
    assert g.entries[0].native is None
    assert g.entries[0].status=="rejected-event-study"
    inquiry=GraphEventInquiryPlan(g,"m33b:new-passive",
        ("baseline","reversed-force"),
        (PlaneEvent("x:new-passive",(.004,0.,0.),(1.,0.,0.),
                    "New passive detector unrelated to invalid initial ideal stop",
                    direction=1),), old.policy,
        (WorldComparison("alternative", "baseline", "reversed-force"),))
    run=run_graph_event_inquiry(inquiry)
    assert run.entries[0].scenario is g.plan.original
    assert run.entries[1].scenario is g.entries[0].scenario
    assert run.entries[1].status=="complete"
    assert run.entries[1].native is not None
    assert run.entries[1].original_graph_observation_status=="rejected-event-study"
    assert run.entries[0].status=="complete"
    assert run.as_dict()["cross_world_comparisons"][0]["status"]=="compared"


def test_05_new_inquiry_respects_actual_guard_and_unique_root_censorship():
    old=make_plan("guard-detector")
    case=next(c for c in old.family.cases if c.case_id=="attract-guard")
    graph=run_unified_observation_graph(UnifiedObservationPlan(
        "m33b:guard-source",old.family.baseline,
        (StructuralWorldNode("attract-guard","baseline",case),),
        retain_source_ancestry=True))
    p=GraphEventInquiryPlan(graph,"m33b:guard-new",
        ("baseline","attract-guard"),old.questions,old.policy,
        (WorldComparison("guard-pair","baseline","attract-guard"),))
    run=run_graph_event_inquiry(p)
    assert run.entries[1].status=="censored-source-guard"
    assert run.entries[1].native.trajectory.termination=="source-guard"
    assert run.entries[1].native.as_dict()["outcomes"]["x:early"]=="located"
    assert run.entries[1].native.as_dict()["outcomes"]["x:far"]=="search-truncated"
    pair=run.as_dict()["cross_world_comparisons"][0]
    assert pair["status"]=="compared"
    assert all(r["status"]=="unpaired" for r in pair["event_time_comparisons"])
    assert all("estimated_numerical_shift_s" not in r
               for r in pair["event_time_comparisons"])
    assert all(r["time_s"]<=run.entries[1].native.trajectory.samples[-1].time_s
               for r in pair["exact_time_trajectory_comparisons"])


def test_06_new_inquiry_does_not_pair_two_opposite_direction_roots():
    old=make_plan("double-crossing")
    graph=run_unified_observation_graph(UnifiedObservationPlan(
        "m33b:tworoots",old.family.baseline,(
            StructuralWorldNode("extra-retarding","baseline",
                                old.family.cases[0]),),retain_source_ancestry=True))
    run=run_graph_event_inquiry(GraphEventInquiryPlan(
        graph,"m33b:tworoot-study",("baseline","extra-retarding"),
        old.questions,old.policy,
        (WorldComparison("two-roots","baseline","extra-retarding"),)))
    assert all(e.status=="complete" for e in run.entries)
    assert all(len([event for event in e.native.events
                    if event.event_id=="x:two-crossings"])==2
               for e in run.entries)
    pair=run.as_dict()["cross_world_comparisons"][0]
    assert pair["status"]=="compared"
    assert pair["event_time_comparisons"][0]["status"]=="unpaired"
    assert "multiple" in pair["event_time_comparisons"][0]["reason"]


def test_07_new_inquiry_terminal_detectors_are_not_duration_complete_pairs():
    old=make_plan("coincident-terminal")
    case=next(c for c in old.family.cases if c.case_id=="more-E")
    graph=run_unified_observation_graph(UnifiedObservationPlan(
        "m33b:terminal-source",old.family.baseline,
        (StructuralWorldNode("more-E","baseline",case),),
        retain_source_ancestry=True))
    terminal=(PlaneEvent("x:terminal",(.05,0.,0.),(1.,0.,0.),
              "Terminating ideal detector is a boundary, not impact dynamics",
              direction=1,terminal=True,role="ideal-stop-surface"),)
    inquiry=GraphEventInquiryPlan(
        graph,"m33b:terminal-study",("baseline","more-E"),
        terminal,old.policy,
        (WorldComparison("terminal-pair","baseline","more-E"),))
    run=run_graph_event_inquiry(inquiry)
    assert all(e.status=="terminal-event" for e in run.entries)
    pair=run.as_dict()["cross_world_comparisons"][0]
    assert pair["event_time_comparisons"][0]["status"]=="unpaired"
    assert "estimated_numerical_shift_s" not in pair["event_time_comparisons"][0]


def test_08_malformed_native_requests_and_unapproved_graph_refuse():
    graph=make_graph()
    orig=(xquestion(),)
    with pytest.raises(TypeError):
        GraphEventInquiryPlan({"exported":"JSON"},"bad",("baseline",),orig,policy())
    with pytest.raises(ValueError):
        GraphEventInquiryPlan(graph,"bad ID",("baseline",),orig,policy())
    with pytest.raises(ValueError):
        GraphEventInquiryPlan(graph,"m33b:dup",("baseline","baseline"),orig,policy())
    with pytest.raises(ValueError):
        GraphEventInquiryPlan(graph,"m33b:unknown",("absent",),orig,policy())
    with pytest.raises(ValueError):
        GraphEventInquiryPlan(graph,"m33b:list",["baseline"],orig,policy())
    with pytest.raises(TypeError):
        GraphEventInquiryPlan(graph,"m33b:listq",("baseline",),list(orig),policy())
    with pytest.raises(ValueError):
        GraphEventInquiryPlan(graph,"m33b:dupq",("baseline",),orig*2,policy())
    with pytest.raises(TypeError):
        GraphEventInquiryPlan(graph,"m33b:badpolicy",("baseline",),orig,False)
    with pytest.raises(ValueError):
        GraphEventInquiryPlan(graph,"m33b:badpair",("baseline",),orig,policy(),
            (WorldComparison("pair","baseline","alpha"),))
    with pytest.raises(ValueError):
        GraphEventInquiryPlan(graph,"m33b:dup-pairs",("baseline","alpha"),orig,
            policy(),(WorldComparison("one","baseline","alpha"),
                      WorldComparison("two","alpha","baseline")))
    with pytest.raises(TypeError):
        GraphEventInquiryPlan(graph,"m33b:rawpair",("baseline","alpha"),orig,
            policy(),(("baseline","alpha"),))
    legacy=run_unified_observation_graph(UnifiedObservationPlan(
        "m33b:legacy",model("uniform-edits").baseline,
        (PhysicalWorldNode("heavy","baseline","Actual mass",
             (CoupledPropertyEdit("test-particle","mass_kg",2e-5),)),)))
    assert legacy.as_dict()["schema"]=="conductor_sim.unified_observation_graph.v1"
    with pytest.raises(ValueError,match="retained"):
        GraphEventInquiryPlan(legacy,"m33b:unrooted",("baseline",),orig,policy())


def test_09_no_event_import_shortcut_and_issued_run_required():
    graph=make_graph()
    p=GraphEventInquiryPlan(graph,"m33b:issued",("baseline","alpha"),
          (xquestion(),),policy())
    run=run_graph_event_inquiry(p)
    assert all(e.native is not None for e in run.entries)
    with pytest.raises(ValueError,match="receipt"):
        GraphEventInquiryRun(p,run.entries)
    with pytest.raises(TypeError):
        run_graph_event_inquiry({"not":"a native plan"})


def test_10_inquiry_plan_status_source_and_clock_mutations_fail():
    p=make_inquiry()
    run=run_graph_event_inquiry(p)
    with pytest.raises(ValueError,match="receipt"):
        replace(run,entries=(replace(run.entries[0],reason="forged"),)
                            +run.entries[1:])
    with pytest.raises((ValueError,TypeError)):
        replace(run,entries=(replace(run.entries[0],status="falsified"),)
                            +run.entries[1:])
    with pytest.raises((ValueError,TypeError)):
        replace(run,plan=replace(p,inquiry_id="m33b:wrong"))
    alt=(PlaneEvent("other",(.008,0.,0.),(1.,0.,0.),"New false question",
                    direction=1),)
    with pytest.raises((ValueError,TypeError)):
        replace(run,plan=replace(p,questions=alt))
    root=run.entries[0].native.events[0]
    modified=replace(root,sample=replace(root.sample,time_s=root.sample.time_s+.03),
                     time_bracket_s=tuple(t+.03 for t in root.time_bracket_s))
    with pytest.raises((ValueError,TypeError)):
        replace(run,entries=(replace(run.entries[0],
            native=replace(run.entries[0].native,events=(modified,))),)
            +run.entries[1:])
    first_sample=run.entries[0].native.trajectory.samples[0]
    assert first_sample.time_s==0.0
    evil=replace(first_sample,time_s=False)
    motion=replace(run.entries[0].native.trajectory,
                   samples=(evil,)+run.entries[0].native.trajectory.samples[1:])
    with pytest.raises((ValueError,TypeError)):
        replace(run,entries=(replace(run.entries[0],
            native=replace(run.entries[0].native,trajectory=motion)),)
            +run.entries[1:])
    changed=replace(run.entries[0],reason="fake pre-export refusal")
    object.__setattr__(run,"entries",(changed,)+run.entries[1:])
    with pytest.raises(ValueError,match="receipt"):
        run.as_dict()


def test_11_source_graph_mutation_blocks_new_inquiry_and_export():
    p=make_inquiry()
    result=run_graph_event_inquiry(p)
    original=p.graph.entries[0]
    altered=replace(original,status="invented-termination")
    object.__setattr__(p.graph,"entries",(altered,)+p.graph.entries[1:])
    with pytest.raises((ValueError,TypeError)):
        result.as_dict()
    with pytest.raises((ValueError,TypeError)):
        run_graph_event_inquiry(p)


def test_12_detached_output_does_not_mutate_new_or_old_evidence():
    p=make_inquiry()
    graph_before=p.graph.as_dict()
    run=run_graph_event_inquiry(p)
    record=run.as_dict()
    record["plan"]["questions"][0]["label"]="A FORGED DETECTOR"
    record["nodes"][1]["source_inventory"][0]["label"]="Fake source"
    record["nodes"][1]["native_event_outcomes"]["x:followup"]="fake"
    record["cross_world_comparisons"][0]["status"]="forged"
    after=run.as_dict()
    assert after["plan"]["questions"][0]["label"] != "A FORGED DETECTOR"
    assert after["nodes"][1]["source_inventory"][0]["label"] != "Fake source"
    assert after["nodes"][1]["native_event_outcomes"]["x:followup"]=="located"
    assert after["cross_world_comparisons"][0]["status"]=="compared"
    assert p.graph.as_dict()==graph_before


def test_13_new_inquiry_oblique_xz_genuine_three_component_science():
    from conductor_sim import (
        PointCharge, PrescribedField, TestChargeMotionSpec, UniformGravity,
    )
    from conductor_sim.scenarios import PrescribedScenario
    old=PrescribedScenario(
        "m33b:xz-root",
        TestChargeMotionSpec(
            PrescribedField((UniformElectricField((280.,0.,-60.)),)),
            PointCharge(1.4e-9,(0.,0.,.05)),1.3e-6,(.12,0.,.2),
            .5,.005,plane_normal=(0.,1.,0.),
            gravity=UniformGravity((0.,0.,-.14))),
        (("field:base","Unchanged original held x-z field"),),
        "Teacher oblique physical x-z plane with genuine gravity.")
    gplan=UnifiedObservationPlan("m33b:xz-graph",old,(
        StructuralWorldNode("add-a","baseline",StructuralCase(
            "add-a","Independent held x-z field A",
            add_sources=(NamedSource("field:new","New A field",
                UniformElectricField((65.,0.,35.))),))),
        StructuralWorldNode("add-b","baseline",StructuralCase(
            "add-b","Independent held x-z field B",
            add_sources=(NamedSource("field:new","New B field",
                UniformElectricField((-45.,0.,90.))),))),
        PhysicalWorldNode("mass","add-a","Actual physical mass change",
            (CoupledPropertyEdit("test-particle","mass_kg",2e-6),)),
    ),retain_source_ancestry=True)
    graph=run_unified_observation_graph(gplan)
    assert [e.status for e in graph.entries]==["complete"]*3
    questions=(PlaneEvent("new:xz",(.06,0.,0.),(1.,0.,0.),
                "New signed physical x detector",direction=1),)
    p=GraphEventInquiryPlan(graph,"m33b:xz-question",
        ("baseline","add-a","add-b","mass"),questions,
        EventPolicy(.005,1e-8,1e-8,1e-6),
        (WorldComparison("sibling","add-a","add-b"),
         WorldComparison("cousin","add-b","mass")))
    run=run_graph_event_inquiry(p)
    assert all(e.status=="complete" for e in run.entries)
    for entry in run.entries:
        spec=entry.scenario.spec
        charge,mass=spec.particle.charge_c,spec.mass_kg
        E=tuple(sum(src.field_v_m[k] for src in spec.field.sources)
                for k in range(3))
        a=tuple(charge*E[k]/mass+spec.gravity.acceleration_m_s2[k]
                for k in range(3))
        f=tuple(charge*E[k]+mass*spec.gravity.acceleration_m_s2[k]
                for k in range(3))
        vx=spec.initial_velocity_m_s[0]
        expected_t=2*.06/(vx+sqrt(vx*vx+2*a[0]*.06))
        assert entry.native.events[0].sample.time_s==pytest.approx(expected_t,abs=2e-7)
        for sample in entry.native.trajectory.samples:
            t=sample.time_s
            pos=tuple(spec.particle.position_m[k]+spec.initial_velocity_m_s[k]*t+
                      .5*a[k]*t*t for k in range(3))
            vel=tuple(spec.initial_velocity_m_s[k]+a[k]*t for k in range(3))
            assert sample.position_m==pytest.approx(pos,abs=3e-8)
            assert sample.velocity_m_s==pytest.approx(vel,abs=3e-8)
            assert sample.force_n==pytest.approx(f,abs=2e-12)
        assert entry.native.trajectory.samples[-1].position_m[2] != 0.
    pairs=run.as_dict()["cross_world_comparisons"]
    for item in pairs:
        assert item["status"]=="compared"
        assert item["event_time_comparisons"][0]["status"]=="one-resolved-pair"
        l=next(e for e in run.entries if e.world_id==item["left_world_id"])
        r=next(e for e in run.entries if e.world_id==item["right_world_id"])
        assert item["left_revision_id"]==l.scenario.revision_id
        assert item["right_revision_id"]==r.scenario.revision_id
        assert any(abs(row["delta_position_m"][2])>1e-8
                   for row in item["exact_time_trajectory_comparisons"]
                   if row["time_s"]>0.)
