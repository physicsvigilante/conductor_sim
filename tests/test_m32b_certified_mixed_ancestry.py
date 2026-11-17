"""M32b: true mixed structural/physical order with native source-ID ancestry.

All physics uses unchanged M20b/M22a/M24a; every observation is genuinely
integrated, not inferred from a parent trajectory or browser animation.
"""
from dataclasses import replace
from math import sqrt

import pytest

from conductor_sim import (
    CoupledPropertyEdit, PhysicalWorldNode, SourceWorldAncestry,
    StructuralWorldNode, UniformElectricField, UnifiedObservationPlan,
    UnifiedObservationRun, run_unified_observation_graph,
)
from conductor_sim.motion_events import PlaneEvent, EventPolicy
from conductor_sim.structural_family import StructuralCase
from conductor_sim.structural_scenarios import (
    NamedSource, SourceReplacement, reassemble_scenario,
)
from examples.m29a_structural_family import model
from examples.m29b_structural_events import make_plan


def physical(node, parent, *edits, observation="events"):
    return PhysicalWorldNode(
        node, parent, "Explicit atomic teacher physical-source hypothesis.",
        tuple(CoupledPropertyEdit(*e) for e in edits), observation,
    )


def ask():
    return (
        PlaneEvent("x:detector", (.006, 0., 0.), (1., 0., 0.),
                   "Passive signed x crossing", direction=1),
    )


def policy():
    return EventPolicy(.003, 1e-8, 1e-8, 1e-6)


def mixed_plan():
    family = model("uniform-edits")
    added = next(c for c in family.cases if c.case_id == "add-E")
    replaced = StructuralCase(
        "replace-after", "Replace previously edited Enew with NEW field identity.",
        replace_sources=(SourceReplacement(
            "field:Enew", NamedSource("field:Eafter", "Replacement held +x field",
                                       UniformElectricField((130., 0., 0.)))),),
    )
    nodes = (
        StructuralWorldNode("add-E", "baseline", added, "events"),
        physical("source-mass", "add-E",
                 ("field:Enew", "field_v_m", (150., 0., 0.)),
                 ("test-particle", "mass_kg", 2e-5)),
        StructuralWorldNode("replace-after", "source-mass", replaced, "events"),
        physical("final", "replace-after",
                 ("field:Eafter", "field_v_m", (180., 0., 0.)),
                 ("test-particle", "initial_velocity_m_s", (.16, 0., 0.))),
    )
    return UnifiedObservationPlan(
        "m32b:mix", family.baseline, nodes, "events", ask(), policy(),
        retain_source_ancestry=True,
    )


def analytic_root(native):
    spec = native.study.scenario.spec
    ax = spec.particle.charge_c * sum(
        s.field_v_m[0] for s in spec.field.sources) / spec.mass_kg
    vx = spec.initial_velocity_m_s[0]
    return .012 / (vx + sqrt(vx*vx + .012*ax))


def test_01_true_structural_physical_structural_physical_events_and_ledger():
    plan = mixed_plan()
    original = plan.original.as_dict()
    run = run_unified_observation_graph(plan)
    assert run.baseline_status == "complete"
    assert [r.status for r in run.entries] == ["complete"] * 4
    p0, p1, p2, p3 = run.entries
    assert p1.scenario.parent_revision_id == p0.scenario.revision_id
    assert p2.assembly.parent is p1.scenario
    assert p2.assembly.as_dict()["parent_revision_id"] == p1.scenario.revision_id
    assert p3.scenario.parent_revision_id == p2.scenario.revision_id
    assert p1.scenario.spec.field.sources[-1].field_v_m == (150.,0.,0.)
    assert p2.scenario.source_annotations[-1][0] == "field:Eafter"
    assert p3.scenario.spec.field.sources[-1].field_v_m == (180.,0.,0.)

    for native in (run.baseline_native, *(entry.native for entry in run.entries)):
        roots = [e for e in native.events if e.event_id == "x:detector"]
        assert native.trajectory.termination == "duration"
        assert len(roots) == 1 and roots[0].direction == 1
        assert roots[0].sample.time_s == pytest.approx(analytic_root(native), abs=1e-7)
    cert = p2.source_ancestry
    assert type(cert) is SourceWorldAncestry
    assert cert.parent_world_id == "source-mass"
    assert cert.parent_revision_id == p1.scenario.revision_id
    assert cert.revision_path[-1] == p2.scenario.revision_id
    assert cert.world_path == ("baseline", "add-E", "source-mass", "replace-after")
    assert cert.active_source_ids == ("field:E0", "field:Ey", "field:Eafter")
    assert cert.seen_source_ids == ("field:E0", "field:Ey", "field:Enew", "field:Eafter")
    assert cert.retired_source_ids == ("field:Enew",)
    assert cert.introduced_source_ids == ("field:Eafter",)
    assert cert.retired_in_this_step == ("field:Enew",)
    assert p3.source_ancestry.retired_source_ids == ("field:Enew",)
    data = run.as_dict()
    assert data["schema"] == "conductor_sim.unified_observation_graph.v2"
    assert data["plan"]["schema"] == "conductor_sim.unified_observation_plan.v2"
    assert data["plan"]["retain_source_ancestry"] is True
    assert data["original_source_ancestry"]["seen_source_ids"] == ["field:E0","field:Ey"]
    assert data["nodes"][2]["source_ancestry"]["retired_source_ids"] == ["field:Enew"]
    assert data["nodes"][2]["parent_source_ancestry"]["world_id"] == "source-mass"
    assert [r["parent_event_time_comparisons"][0]["status"] for r in data["nodes"]] == [
        "one-resolved-pair"]*4
    assert data["time_interpolation"] is False and data["postguard_extrapolation"] is False
    data["nodes"][2]["source_ancestry"]["retired_source_ids"].clear()
    assert run.as_dict()["nodes"][2]["source_ancestry"]["retired_source_ids"] == ["field:Enew"]
    assert plan.original.as_dict() == original


def test_02_historically_retired_source_reuse_rejected_after_physical_revision():
    original = model("uniform-edits").baseline
    add = next(c for c in model("uniform-edits").cases if c.case_id=="add-E")
    swap = StructuralCase("swap-E","Retire Enew and introduce Eplus.",
        replace_sources=(SourceReplacement("field:Enew",NamedSource(
            "field:Eplus","New field Eplus",UniformElectricField((190.,0.,0.)))),))
    again = NamedSource("field:Enew", "Illegal resurrection", UniformElectricField((10.,0.,0.)))
    bad_add = StructuralCase("reuse-add","Attempt to reintroduce retired Enew.",
                             add_sources=(again,))
    bad_replace = StructuralCase("reuse-replace","Attempt to replace Eplus with retired Enew.",
                                replace_sources=(SourceReplacement("field:Eplus",again),))
    okay = StructuralCase("fresh","Introduce genuinely unseen source.",
        add_sources=(NamedSource("field:fresh","Fresh independent held field",
                                UniformElectricField((10.,0.,0.))),))
    plan = UnifiedObservationPlan("m32b:retired",original,(
        StructuralWorldNode("add-E","baseline",add),
        StructuralWorldNode("swap-E","add-E",swap),
        physical("mass","swap-E",("test-particle","mass_kg",2e-5),
                 observation="trajectory"),
        StructuralWorldNode("reuse-add","mass",bad_add),
        StructuralWorldNode("reuse-replace","mass",bad_replace),
        StructuralWorldNode("fresh","mass",okay),
    ),retain_source_ancestry=True)
    result = run_unified_observation_graph(plan)
    assert [r.status for r in result.entries] == [
        "complete","complete","complete","rejected-structural-model",
        "rejected-structural-model","complete"]
    assert all("retired or existing" in r.reason for r in result.entries[3:5])
    assert all(r.scenario is None and r.native is None and r.source_ancestry is None
               for r in result.entries[3:5])
    assert result.entries[5].assembly.parent is result.entries[2].scenario
    assert result.entries[5].source_ancestry.retired_source_ids == ("field:Enew",)
    assert result.entries[5].source_ancestry.seen_source_ids[-1] == "field:fresh"
    assert result.entries[5].source_ancestry.parent_revision_id == (
        result.entries[2].scenario.revision_id)

    # The direct old M22a call on an intermediate plain M20b scenario is
    # insufficient to know Enew was retired earlier. M32b closes that gap.
    raw = reassemble_scenario(
        result.entries[2].scenario, scenario_id="m32b:old-m22a-only",
        note="Demonstrate missing history in a bare M20b physical scenario.",
        add_sources=(again,))
    assert "field:Enew" in dict(raw.scenario.source_annotations)
    assert raw.parent is result.entries[2].scenario
    evidence=result.as_dict()["nodes"]
    assert evidence[3]["source_ancestry"] is None
    assert evidence[4]["revised_source_inventory"] is None
    assert evidence[5]["source_ancestry"]["retired_source_ids"] == ["field:Enew"]


def test_03_sibling_ancestries_are_independent_not_global_identity_registry():
    old = model("uniform-edits").baseline
    src = NamedSource("field:shared","Shared ID on two alternative branches",
                      UniformElectricField((30.,0.,0.)))
    nodes=tuple(StructuralWorldNode(
        ident,"baseline",StructuralCase(ident,"Independent teacher alternative.",
                                       add_sources=(src,)))
                for ident in ("alternative-A","alternative-B"))
    plan=UnifiedObservationPlan("m32b:siblings",old,nodes,retain_source_ancestry=True)
    run=run_unified_observation_graph(plan)
    assert [e.status for e in run.entries]==["complete","complete"]
    assert run.entries[0].scenario.revision_id != run.entries[1].scenario.revision_id
    assert all(e.source_ancestry.introduced_source_ids == ("field:shared",)
               for e in run.entries)
    assert all(e.source_ancestry.world_path == ("baseline",e.node.node_id)
               for e in run.entries)


def test_04_physically_accepted_event_refusal_then_structural_child_success():
    old=make_plan("terminal-refusal")
    physical_parent=physical(
        "opposite-force","baseline",
        ("field:Eneg","field_v_m",(1000.,0.,0.)),
        observation="events")
    correcting=StructuralCase(
        "fix-force","Add a physically negative field without rewriting Eneg.",
        add_sources=(NamedSource("field:Efix","New negative x field",
                                 UniformElectricField((-3000.,0.,0.))),))
    plan=UnifiedObservationPlan("m32b:stop",old.family.baseline,(
        physical_parent,
        StructuralWorldNode("fix-force","opposite-force",correcting,"events"),
    ),"events",old.questions,old.policy,retain_source_ancestry=True)
    run=run_unified_observation_graph(plan)
    parent,child=run.entries
    assert parent.scenario is not None and parent.native is None
    assert parent.status=="rejected-event-study"
    assert child.scenario is not None and child.native is not None
    assert child.status=="complete"
    assert child.assembly.parent is parent.scenario
    assert child.source_ancestry.parent_revision_id==parent.scenario.revision_id
    comparison=run.as_dict()["nodes"][1]["parent_event_time_comparisons"][0]
    assert comparison["status"]=="unpaired"
    assert "missing native" in comparison["reason"]


def test_05_guard_after_real_physical_edit_retains_early_and_truncated_roots():
    old=make_plan("guard-detector")
    attr=next(c for c in old.family.cases if c.case_id=="attract-guard")
    plan=UnifiedObservationPlan("m32b:guard",old.family.baseline,(
        physical("launch","baseline",
                 ("test-particle","initial_velocity_m_s",(1.1,0.,0.)),
                 observation="events"),
        StructuralWorldNode("attract-guard","launch",attr,"events"),
    ),"events",old.questions,old.policy,retain_source_ancestry=True)
    run=run_unified_observation_graph(plan)
    assert run.entries[0].status=="complete"
    child=run.entries[1]
    assert child.status=="censored-source-guard"
    assert child.native.trajectory.termination=="source-guard"
    assert child.native.as_dict()["outcomes"]["x:early"]=="located"
    assert child.native.as_dict()["outcomes"]["x:far"]=="search-truncated"
    assert all(row["status"]=="unpaired" for row in
               run.as_dict()["nodes"][1]["parent_event_time_comparisons"])


def test_06_two_oppositely_signed_roots_after_physical_then_structural_never_pair():
    old=make_plan("double-crossing")
    c=old.family.cases[0]
    plan=UnifiedObservationPlan("m32b:double",old.family.baseline,(
        physical("mass","baseline",("test-particle","mass_kg",1.2e-6),
                 observation="events"),
        StructuralWorldNode("extra-retarding","mass",c,"events"),
    ),"events",old.questions,old.policy,retain_source_ancestry=True)
    out=run_unified_observation_graph(plan)
    assert [r.status for r in out.entries]==["complete","complete"]
    assert out.entries[1].assembly.parent is out.entries[0].scenario
    for native in (out.baseline_native,out.entries[0].native,out.entries[1].native):
        assert native.trajectory.termination=="duration"
        roots=[e for e in native.events if e.event_id=="x:two-crossings"]
        assert len(roots)==2 and [e.direction for e in roots]==[1,-1]
    assert out.as_dict()["nodes"][1]["parent_event_time_comparisons"][0]["status"]=="unpaired"


def test_07_refused_model_then_child_skipped_but_valid_sibling_preserved():
    old=model("mirror-pair")
    invalid=physical("off-plane","baseline",
        ("point:centre","position_m",(.2,0.,.04)),
        observation="trajectory")
    child=StructuralWorldNode(
        "add-field","off-plane",StructuralCase(
            "add-field","Do not construct from rejected parent.",
            add_sources=(NamedSource("field:extra","Unobserved extra field",
                                     UniformElectricField((10.,0.,0.))),)))
    sibling=StructuralWorldNode(
        "real-field","baseline",StructuralCase(
            "real-field","Separate physically admitted source branch.",
            add_sources=(NamedSource("field:good","Real field",
                                     UniformElectricField((20.,0.,0.))),)))
    plan=UnifiedObservationPlan("m32b:refusals",old.baseline,
                                 (invalid,child,sibling),
                                 retain_source_ancestry=True)
    run=run_unified_observation_graph(plan)
    assert [e.status for e in run.entries]==[
        "rejected-physical-model","skipped-parent-model","complete"]
    assert all(e.source_ancestry is None for e in run.entries[:2])
    assert run.entries[2].source_ancestry is not None
    assert run.as_dict()["nodes"][1]["admitted_scenario"] is None


def test_08_original_v1_policy_and_legacy_json_remain_unchanged_by_default():
    old=model("uniform-edits")
    first=physical("mass","baseline",("test-particle","mass_kg",2e-5),
                   observation="trajectory")
    second=StructuralWorldNode("add-E","mass",old.cases[0])
    with pytest.raises(ValueError,match="structural-after-physical"):
        UnifiedObservationPlan("m32b:legacy",old.baseline,(first,second))
    plan=UnifiedObservationPlan("m32b:legacy",old.baseline,(first,))
    run=run_unified_observation_graph(plan)
    out=run.as_dict()
    assert out["schema"]=="conductor_sim.unified_observation_graph.v1"
    assert out["plan"]["schema"]=="conductor_sim.unified_observation_plan.v1"
    assert "retain_source_ancestry" not in out["plan"]
    assert "source_ancestry" not in out["nodes"][0]
    assert "original_source_ancestry" not in out
    assert run.entries[0].source_ancestry is None
    with pytest.raises(TypeError,match="native Boolean"):
        replace(plan,retain_source_ancestry=0)


def test_09_adversarial_ancestry_and_result_rebindings_refuse():
    result=run_unified_observation_graph(mixed_plan())
    child=result.entries[2]
    cert=child.source_ancestry
    assert cert.retired_source_ids==("field:Enew",)
    forged=replace(cert,gravity_present=True)
    with pytest.raises(ValueError,match="ancestry disagrees"):
        replace(result,entries=(
            *result.entries[:2],
            replace(child,source_ancestry=forged),
            *result.entries[3:]))
    with pytest.raises(ValueError,match="runner-issued receipt"):
        replace(result,plan=replace(result.plan,plan_id="m32b:fake"))
    fake_node=replace(child.node,case=replace(child.node.case,note="Forged author source note"))
    alt_plan=replace(result.plan,nodes=(
        *result.plan.nodes[:2],fake_node,*result.plan.nodes[3:]))
    changed=replace(child,node=fake_node)
    with pytest.raises(ValueError,match="runner-issued receipt"):
        replace(result,plan=alt_plan,entries=(
            *result.entries[:2],changed,*result.entries[3:]))
    native=child.native
    first=native.events[0]
    tampered=replace(first,sample=replace(first.sample,time_s=first.sample.time_s+.03),
                     time_bracket_s=tuple(t+.03 for t in first.time_bracket_s))
    with pytest.raises((ValueError,TypeError)):
        replace(result,entries=(
            *result.entries[:2],replace(child,native=replace(native,events=(tampered,))),
            *result.entries[3:]))
    with pytest.raises((ValueError,TypeError)):
        replace(result,entries=(
            *result.entries[:2],replace(child,native=replace(native,
                events=(replace(first,sample=replace(first.sample,time_s=False)),))),
            *result.entries[3:]))
    destroyed=replace(child,source_ancestry=None)
    object.__setattr__(result,"entries",(
        *result.entries[:2],destroyed,*result.entries[3:]))
    with pytest.raises(ValueError,match="ancestry disagrees"):
        result.as_dict()


def test_10_root_ancestry_scope_and_explicit_retirement_type_guards():
    original=model("uniform-edits").baseline
    plan=UnifiedObservationPlan("m32b:root",original,(
        physical("mass","baseline",("test-particle","mass_kg",2e-5),
                 observation="trajectory"),
    ),retain_source_ancestry=True)
    run=run_unified_observation_graph(plan)
    ancestor=run.entries[0].source_ancestry
    assert ancestor.seen_source_ids==("field:E0","field:Ey")
    assert ancestor.retired_source_ids==()
    assert ancestor.parent_revision_id==original.revision_id
    assert ancestor.world_path==("baseline","mass")
    root=run.as_dict()["original_source_ancestry"]
    assert root["world_path"]==["baseline"]
    assert "supplied graph root" in root["history_scope"]
    with pytest.raises((TypeError,ValueError)):
        replace(ancestor,gravity_present=0)
    with pytest.raises((TypeError,ValueError)):
        replace(ancestor,active_source_ids=("field:E0",False))
