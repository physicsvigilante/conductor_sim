"""M30b: native typed events over real source-structure→property-revised models.

References are independent constant-acceleration formulae, original M24a
real numerical brackets and direct SI Coulomb rod quadrature. The source,
RK, and root service are untouched and all modifications are separate
teacher-authored physical hypotheses.
"""
from dataclasses import replace
from math import hypot, pi, sqrt

import pytest

from conductor_sim import (
    EPSILON_0, BranchSweep, FixedPointField, PointCharge, PrescribedField,
    StructuralParameterPlan, SweepAxis, TestChargeMotionSpec,
    UniformGravity, UniformRodField,
)
from conductor_sim.motion_events import EventPolicy, PlaneEvent
from conductor_sim.nested_event_family import (
    NestedEventPlan, NestedEventRun, run_nested_event_family,
)
from conductor_sim.scenarios import PrescribedScenario
from conductor_sim.structural_family import (
    StructuralCase, StructuralFamilyPlan, run_structural_family,
)
from conductor_sim.structural_scenarios import NamedSource
from examples.m29a_structural_family import model
from examples.m29b_structural_events import make_plan


def _uniform_study():
    experiment = StructuralParameterPlan(
        "m30b:uniform", model("uniform-edits"), (
            BranchSweep("add-E", SweepAxis(
                "field:Enew", "field_v_m", ((50.,0.,0.),(160.,0.,0.)))),
            BranchSweep("replace-Ey", SweepAxis(
                "field:Ey-new", "field_v_m", ((0.,70.,0.),))),
        ))
    return NestedEventPlan(
        experiment,
        (PlaneEvent("x:detector", (.006,0.,0.), (1.,0.,0.),
                    "Passive x detector",direction=1),),
        EventPolicy(.003,1e-8,1e-8,1e-6))


def _roots(run, event_id):
    return [x for x in run.events if x.event_id == event_id]


def test_p1_actual_new_source_and_parameter_native_event_roots_analytic_two_edge():
    study = _uniform_study()
    before = study.experiment.family.baseline.as_dict()
    result = run_nested_event_family(study)
    assert type(result) is NestedEventRun
    assert result.original_event_run.study.scenario is study.experiment.family.baseline
    assert [b.status for b in result.branches] == ["complete", "complete"]
    assert result.original_event_run.trajectory.termination == "duration"
    assert [x.status for x in result.branches[0].variations] == ["complete","complete"]
    physical = result.branches[0].physical_entry
    assert physical.parent_scenario.source_annotations[-1][0] == "field:Enew"
    assert physical.parent_scenario.revision_id == (
        physical.structural_entry.assembly.as_dict()["child_revision_id"])
    assert [v.physical_entry.scenario.parent_revision_id for v in
            result.branches[0].variations] == [physical.parent_scenario.revision_id]*2

    q, m, d, v0 = 1e-9, 1e-5, .006, .1
    native_cases = [
        (result.original_event_run,100.),
        (result.branches[0].event_run,200.),
        (result.branches[0].variations[0].event_run,150.),
        (result.branches[0].variations[1].event_run,260.),
    ]
    for native, ex in native_cases:
        assert native is not None
        assert native.as_dict()["outcomes"]["x:detector"] == "located"
        found = _roots(native,"x:detector")
        assert len(found) == 1 and found[0].direction == 1
        expected = 2*d/(v0+sqrt(v0*v0+2*(q*ex/m)*d))
        assert found[0].sample.time_s == pytest.approx(expected, abs=1e-8)
        lo, hi = found[0].time_bracket_s
        assert lo <= expected <= hi
        assert found[0].sample.position_m[0] == pytest.approx(d,abs=1e-8)
        assert native.trajectory.termination == "duration"

    audit = result.as_dict()
    assert audit["schema"] == "conductor_sim.nested_event_family.v1"
    assert audit["original_revision_id"] == before["revision_id"]
    assert audit["time_interpolation"] is False and audit["postguard_extrapolation"] is False
    edge = audit["branches"][0]
    assert edge["original_to_structural_event_comparisons"][0]["status"]=="one-resolved-pair"
    assert [v["parent_event_comparisons"][0]["status"] for v in edge["variations"]] == [
        "one-resolved-pair", "one-resolved-pair"]
    for v in edge["variations"]:
        assert v["event_status"] == "complete"
        assert len(v["exact_time_event_trajectory_differences"]) > 5
        # Root pairing is relative to actual newly assembled world, not
        # the original E=100 V/m source inventory.
        assert v["model_revision_id"] != physical.parent_scenario.revision_id
    for row in result.branches[0].variations:
        assert row.event_run.study.scenario is row.physical_entry.scenario
    audit["branches"][0]["variations"][0]["event_run"]["events"][0]["sample"]["position_m"][0]=999
    assert result.branches[0].variations[0].event_run.events[0].sample.position_m[0] != 999
    assert study.experiment.family.baseline.as_dict() == before


def test_p2_new_gravity_negative_detector_original_no_root_while_children_pair():
    study = NestedEventPlan(
        StructuralParameterPlan("m30b:gravity",model("uniform-edits"),(
            BranchSweep("add-gravity",SweepAxis(
                "gravity","acceleration_m_s2",((0.,-1.,0.),(0.,-4.,0.)))),)),
        (PlaneEvent("y:down",(0.,-.003,0.),(0.,1.,0.),
                    "Teacher downward crossing",direction=-1),),
        EventPolicy(.003,1e-8,1e-8,1e-6))
    run = run_nested_event_family(study)
    assert len(_roots(run.original_event_run,"y:down")) == 0
    parent = run.branches[0]
    assert parent.status == "complete"
    assert [x.status for x in parent.variations] == ["complete","complete"]
    for native,g in [
        (parent.event_run,-2.),
        (parent.variations[0].event_run,-1.),
        (parent.variations[1].event_run,-4.),
    ]:
        assert native.as_dict()["outcomes"]["y:down"] == "located"
        roots=_roots(native,"y:down")
        assert len(roots)==1 and roots[0].direction==-1
        ay=1e-9*50/1e-5+g
        expected=sqrt(2*(-.003)/ay)
        assert roots[0].sample.time_s==pytest.approx(expected,abs=1e-8)
    record=run.as_dict()["branches"][0]
    assert record["original_to_structural_event_comparisons"][0]["status"]=="unpaired"
    assert [x["parent_event_comparisons"][0]["status"] for x in
            record["variations"]] == ["one-resolved-pair","one-resolved-pair"]
    assert all(x["parent_event_comparisons"][0]["direction"]==-1 for x in record["variations"])


def test_p3_event_refused_parent_does_not_refuse_accepted_property_grandchildren():
    original_study=make_plan("terminal-refusal")
    study=NestedEventPlan(
        StructuralParameterPlan("m30b:terminal-parent",original_study.family,(
            BranchSweep("reversed-force",SweepAxis(
                "field:Eplus","field_v_m",((100.,0.,0.),(500.,0.,0.)))),)),
        original_study.questions,original_study.policy)
    run=run_nested_event_family(study)
    node=run.branches[0]
    assert node.physical_entry.status=="evaluated"
    assert node.physical_entry.parent_scenario is not None
    assert node.status=="rejected-event-study" and node.event_run is None
    assert "initial release" in node.reason
    assert [v.status for v in node.variations]==["complete","complete"]
    for var in node.variations:
        assert var.event_run is not None
        assert var.event_run.study.scenario is var.physical_entry.scenario
        assert var.event_run.trajectory.termination=="duration"
        assert var.physical_entry.scenario.parent_revision_id==(
            node.physical_entry.parent_scenario.revision_id)
    data=run.as_dict()["branches"][0]
    assert data["event_run"] is None
    assert data["physical_branch_status"]=="evaluated"
    assert all(v["parent_event_comparisons"][0]["status"]=="unpaired"
               for v in data["variations"])
    assert all(v["event_run"] is not None for v in data["variations"])


def test_p4_preguard_real_early_root_does_not_pair_censored_parent_or_child():
    study=NestedEventPlan(
        StructuralParameterPlan("m30b:guard",model("guard-and-refusal"),(
            BranchSweep("attract-guard",SweepAxis(
                "point:attract","charge_c",(-5e-11,-2e-10))),)),
        make_plan("guard-detector").questions,
        make_plan("guard-detector").policy)
    run=run_nested_event_family(study)
    parent=run.branches[0]
    assert parent.status=="censored"
    assert parent.event_run.trajectory.termination=="source-guard"
    assert parent.event_run.as_dict()["outcomes"]["x:early"]=="located"
    assert len(_roots(parent.event_run,"x:early"))==1
    assert parent.event_run.as_dict()["outcomes"]["x:far"]=="search-truncated"
    rec=run.as_dict()["branches"][0]
    assert rec["original_to_structural_event_comparisons"][0]["status"]=="unpaired"
    for var in parent.variations:
        if var.event_run is not None:
            assert var.status in ("censored","complete")
    assert all(v["parent_event_comparisons"][0]["status"]=="unpaired"
               for v in rec["variations"])


def test_p5_two_bidirectional_detector_roots_remain_unpaired():
    earlier=make_plan("double-crossing")
    study=NestedEventPlan(
        StructuralParameterPlan("m30b:double",earlier.family,(
            BranchSweep("extra-retarding",SweepAxis(
                "field:more","field_v_m",((-100.,0.,0.),(-350.,0.,0.)))),)),
        earlier.questions,earlier.policy)
    run=run_nested_event_family(study)
    originals=[run.original_event_run,run.branches[0].event_run,
               *[v.event_run for v in run.branches[0].variations]]
    for native in originals:
        assert native.trajectory.termination=="duration"
        roots=_roots(native,"x:two-crossings")
        assert len(roots)==2
        assert [r.direction for r in roots]==[1,-1]
    saved=run.as_dict()["branches"][0]
    assert saved["original_to_structural_event_comparisons"][0]["status"]=="unpaired"
    assert all(v["parent_event_comparisons"][0]["status"]=="unpaired"
               for v in saved["variations"])
    assert all(len(v["event_run"]["events"])==2 for v in saved["variations"])


def test_p7_missing_structural_child_and_bad_axis_keep_true_inventories():
    study=NestedEventPlan(
        StructuralParameterPlan("m30b:refusals",model("mirror-pair"),(
            BranchSweep("unpaired",SweepAxis("point:single","charge_c",(1e-9,3e-9))),
            BranchSweep("extra-field",SweepAxis("point:single","charge_c",(1e-9,3e-9))),
            BranchSweep("mirror-two",SweepAxis("test-particle","mass_kg",(2e-5,))),
        )),
        (PlaneEvent("x:early",(.001,0.,0.),(1.,0.,0.),
                    "Passive early detector",direction=1),),
        EventPolicy(.002,1e-8,1e-8,1e-6))
    run=run_nested_event_family(study)
    assert [b.physical_entry.status for b in run.branches]==[
        "skipped-structural-model","rejected-sweep-axis","evaluated"]
    assert run.branches[0].status=="skipped-structural-model"
    assert run.branches[0].event_run is None
    assert run.branches[1].physical_entry.parent_scenario is not None
    assert run.branches[1].event_run is not None
    assert run.branches[1].variations==()
    assert run.branches[2].variations[0].event_run is not None
    data=run.as_dict()
    assert data["physical_model_family"]["entries"][0]["source_inventory"] is None
    assert data["physical_model_family"]["entries"][1]["source_inventory"] is not None
    assert data["branches"][0]["event_run"] is None
    assert data["branches"][1]["event_run"] is not None
    assert data["branches"][1]["variations"]==[]


def test_n2_baseline_reuse_wrong_model_binding_and_bad_untrusted_input():
    study=NestedEventPlan(
        StructuralParameterPlan("m30b:baseline",model("uniform-edits"),(
            BranchSweep("baseline",SweepAxis("test-particle","mass_kg",(2e-5,))),)),
        (PlaneEvent("x:det",(.006,0.,0.),(1.,0.,0.),
                    "Original baseline signed detector",direction=1),),
        EventPolicy(.003,1e-8,1e-8,1e-6))
    run=run_nested_event_family(study)
    assert run.branches[0].event_run is run.original_event_run
    assert run.branches[0].status=="complete"
    assert run.as_dict()["branches"][0]["original_to_structural_event_comparisons"][0]["status"]==(
        "reference-baseline")
    bad_branch=replace(run.branches[0],event_run=run.branches[0].variations[0].event_run)
    with pytest.raises(ValueError,match="single source of truth"):
        NestedEventRun(study,run.physical_family,run.original_event_run,(bad_branch,))
    with pytest.raises((ValueError,TypeError)):
        run_nested_event_family({"payload":"untrusted"})
    with pytest.raises(TypeError):
        NestedEventPlan(study.experiment,[],study.policy)
    with pytest.raises(ValueError,match="duplicate"):
        NestedEventPlan(study.experiment,(study.questions[0],study.questions[0]),study.policy)


def test_p8_m30a_review_f01_admitted_structural_world_unavailable_native_parent(monkeypatch):
    """Explicit controlled test-side numerical failure; not a physical solver claim."""
    from conductor_sim import integrate_test_charge
    from conductor_sim.structural_family import StructuralFamilyEntry, StructuralFamilyRun
    import conductor_sim.structural_parameter_family as m30a

    all_cases=model("uniform-edits")
    family=StructuralFamilyPlan("m30b:unavailable", all_cases.baseline,
                                (all_cases.cases[0],all_cases.cases[1]))
    experiment=StructuralParameterPlan("m30b:availability",family,(
        BranchSweep("add-E",SweepAxis("field:Enew","field_v_m",((50.,0.,0.),))),
        BranchSweep("remove-Ey",SweepAxis("test-particle","mass_kg",(2e-5,))),
    ))
    original=run_structural_family(family)
    first=original.entries[0]
    assert first.assembly is not None and first.trajectory is not None
    altered=replace(first,trajectory=None,status="integration-failed",
                    reason="Test-injected numerical integration failure on admitted model")
    mock=StructuralFamilyRun(family,original.baseline_trajectory,
                             (altered, original.entries[1]))
    monkeypatch.setattr(m30a,"run_structural_family",lambda chosen: mock)
    result=m30a.run_structural_parameter_family(experiment)
    assert [e.status for e in result.entries]==[
        "unavailable-branch-trajectory","evaluated"]
    assert result.entries[0].parent_scenario is first.assembly.scenario
    assert result.entries[0].sweep_run is None
    assert result.entries[0].structural_entry.assembly is first.assembly
    assert result.as_dict()["entries"][0]["source_inventory"] is not None
    assert result.as_dict()["entries"][0]["structural_assembly"] is not None
    assert result.entries[1].sweep_run.entries[0].status=="complete"


def test_p6_unknown_tilted_rod_and_signed_point_new_charge_event_transfer():
    sources=(
        UniformRodField(3e-9,(.11,-.1,0.),(.13,.12,0.)),
        FixedPointField(2e-9,(.18,.15,0.)),
        FixedPointField(-1e-9,(-.16,.10,0.)),
    )
    original=PrescribedScenario(
        "m30b:rod-original",
        TestChargeMotionSpec(
            PrescribedField(sources),
            PointCharge(1e-9,(0.,0.,0.)),
            1e-5,(.05,.02,0.),.7,.005,
            plane_normal=(0.,0.,1.),
            relative_tolerance=1e-10,
            gravity=UniformGravity((0.,-.1,0.))),
        (("rod:slanted","Tilted charged finite rod"),
         ("point:plus","Positive held point"),
         ("point:minus","Negative held point")),
        "Teacher hypothetical: genuine tilted rod and signed source charges.")
    family=StructuralFamilyPlan("m30b:rod-family",original,(
        StructuralCase("new-point","Add new fixed positive source.",
            add_sources=(NamedSource("point:added","Extra held point",
                FixedPointField(1.7e-9,(.21,-.09,0.))),)),))
    study=NestedEventPlan(
        StructuralParameterPlan("m30b:rod-sweep",family,(
            BranchSweep("new-point",SweepAxis(
                "point:added","charge_c",(1.2e-9,2e-9))),)),
        (PlaneEvent("x:left",(-.006,0.,0.),(1.,0.,0.),
                    "Signed negative-x detector",direction=-1),),
        EventPolicy(.002,1e-9,1e-9,1e-7))
    run=run_nested_event_family(study)
    assert run.original_event_run.trajectory.termination=="duration"
    parent=run.branches[0]
    assert parent.status=="complete"
    assert all(v.status=="complete" for v in parent.variations)
    tests=[run.original_event_run,parent.event_run,*[v.event_run for v in parent.variations]]
    for native in tests:
        assert native.trajectory.termination=="duration"
        roots=_roots(native,"x:left")
        assert len(roots)==1 and roots[0].direction==-1
        assert roots[0].sample.time_s > .4
        assert roots[0].sample.time_s < .7
        assert roots[0].sample.position_m[0]==pytest.approx(-.006,abs=1e-8)
    assert [v["parent_event_comparisons"][0]["status"] for v in
            run.as_dict()["branches"][0]["variations"]]==[
            "one-resolved-pair","one-resolved-pair"]
    # Independent test-side finite rod SI field evaluated by 2500 midpoint
    # Coulomb quadrature, not the production exact UniformRodField._values.
    p=(0.,0.,0.);a=(.11,-.1,0.);b=(.13,.12,0.)
    k=1/(4*pi*EPSILON_0)
    electric=[0.,0.,0.]
    n=2500
    for j in range(n):
        u=(j+.5)/n
        src=tuple(a[i]+u*(b[i]-a[i]) for i in range(3))
        d=tuple(-v for v in src)
        distance=hypot(*d)
        for i in range(3):
            electric[i]+=k*3e-9*d[i]/(n*distance**3)
    for q,site in ((2e-9,(.18,.15,0.)),(-1e-9,(-.16,.10,0.)),
                   (1.7e-9,(.21,-.09,0.))):
        d=tuple(-v for v in site);distance=hypot(*d)
        for i in range(3):
            electric[i]+=k*q*d[i]/distance**3
    reference_force=(
        1e-9*electric[0],
        1e-9*electric[1]-1e-5*.1,
        1e-9*electric[2],
    )
    saved=parent.event_run.trajectory.samples[0].force_n
    assert saved==pytest.approx(reference_force,abs=2e-13)
    assert original.as_dict()==study.experiment.family.baseline.as_dict()
