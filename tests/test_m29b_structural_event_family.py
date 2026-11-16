"""M29b: genuine native event observations over independently assembled inventories.

No extra field law or event interpolator is introduced. Independent analytic
uniform-force roots and rotated direct-Coulomb force references test that
physical changed inventories affect the actual M24a integration.
"""
from dataclasses import replace
from math import hypot, pi, sqrt
from xml.etree import ElementTree as ET
import json

import pytest

from conductor_sim import (
    EPSILON_0, FixedPointField, PointCharge, PrescribedField,
    TestChargeMotionSpec, UniformElectricField,
)
from conductor_sim.scenarios import PrescribedScenario
from conductor_sim.structural_scenarios import NamedSource
from conductor_sim.structural_family import StructuralCase, StructuralFamilyPlan
from conductor_sim.structural_event_family import (
    StructuralEventFamilyPlan, StructuralEventFamilyRun,
    run_structural_event_family,
)
from conductor_sim.structural_event_viewer import (
    render_structural_event_family, write_structural_event_family_html,
)
from conductor_sim.motion_events import PlaneEvent, VelocityReversal, EventPolicy
from examples.m29b_structural_events import make_plan, run_case


def _roots(run, event_id):
    return [event for event in run.events if event.event_id == event_id]


def test_p1_analytic_uniform_source_membership_detector_times_and_shift_intervals():
    plan=make_plan("uniform-detector")
    baseline_before=plan.family.baseline.as_dict()
    out=run_structural_event_family(plan)
    assert out.plan is plan
    assert out.baseline_run.study.scenario is plan.family.baseline
    assert [x.status for x in out.entries]==["complete"]*4
    assert out.plan.family.baseline.as_dict()==baseline_before
    assert [e.assembly.parent for e in out.entries]==[plan.family.baseline]*4
    root_id="x:detector"
    q=1e-9
    m=1e-5
    d=.006
    v0=.1
    for native,ax in [(out.baseline_run,q*100/m)]+[
        (row.event_run,q*(200 if row.case.case_id=="add-E" else 100)/m)
        for row in out.entries]:
        assert native.trajectory.termination=="duration"
        assert native.as_dict()["outcomes"][root_id]=="located"
        root=_roots(native,root_id)
        assert len(root)==1
        expected=2*d/(v0+sqrt(v0*v0+2*ax*d))
        assert root[0].sample.time_s == pytest.approx(expected,abs=2e-8)
        assert root[0].time_bracket_s[0] <= root[0].sample.time_s <= root[0].time_bracket_s[1]
        assert root[0].direction == +1
    records=out.as_dict()
    assert records["schema"]=="conductor_sim.structural_event_family.v1"
    assert records["baseline_revision_id"]==plan.family.baseline.revision_id
    assert records["baseline_outcomes"][root_id]=="located"
    assert records["time_interpolation"] is False and records["postguard_extrapolation"] is False
    assert [row["structural_assembly"]["parent_revision_id"] for row in records["entries"]]==[
        plan.family.baseline.revision_id]*4
    assert [row["status"] for row in records["entries"]]==["complete"]*4
    for row in records["entries"]:
        match=row["event_time_comparisons"][0]
        assert match["status"]=="one-resolved-pair"
        assert match["baseline_termination"]=="duration"
        assert match["variation_termination"]=="duration"
        assert match["direction"]==1
        low,high=match["numerical_bracket_shift_interval_s"]
        assert low<=match["estimated_numerical_shift_s"]<=high
        assert "NOT rigorous" in records["qualification"]
    assert records["entries"][0]["event_time_comparisons"][0]["estimated_numerical_shift_s"]<0
    for row in records["entries"][1:]:
        assert row["event_time_comparisons"][0]["estimated_numerical_shift_s"]==pytest.approx(
            0.,abs=2e-8)
    records["entries"][0]["event_run"]["trajectory"]["samples"][0]["position_m"][0]=999.
    assert out.entries[0].event_run.trajectory.samples[0].position_m[0] != 999.
    assert plan.family.baseline.as_dict()==baseline_before


def test_p2_real_preguard_root_and_unreached_far_detector_are_not_confused():
    out=run_case("guard-detector")
    assert out.baseline_run.trajectory.termination=="duration"
    assert [row.status for row in out.entries]==[
        "censored","complete","rejected-structural-model"]
    early="x:early"
    far="x:far"
    baseline=out.baseline_run
    child=out.entries[0].event_run
    assert child.trajectory.termination=="source-guard"
    assert baseline.as_dict()["outcomes"][early]=="located"
    assert child.as_dict()["outcomes"][early]=="located"
    assert len(_roots(baseline,early))==len(_roots(child,early))==1
    assert baseline.as_dict()["outcomes"][far]=="located"
    assert child.as_dict()["outcomes"][far]=="search-truncated"
    compared=out.entries[0].as_dict(baseline,out.plan.questions)
    assert [x["status"] for x in compared["event_time_comparisons"]]==[
        "unpaired","unpaired"]
    assert "censored" in compared["event_time_comparisons"][0]["reason"]
    assert "estimated_numerical_shift_s" not in compared["event_time_comparisons"][0]
    assert compared["event_time_comparisons"][1]["variation_outcome"]=="search-truncated"
    assert compared["exact_time_trajectory_comparisons"]
    last_time=child.trajectory.samples[-1].time_s
    assert last_time<.25
    assert max(row["time_s"] for row in compared["exact_time_trajectory_comparisons"])<last_time
    assert all(row["time_s"] in {s.time_s for s in baseline.trajectory.samples}
               for row in compared["exact_time_trajectory_comparisons"])
    other=out.entries[1].as_dict(baseline,out.plan.questions)
    assert other["status"]=="complete"
    assert other["event_time_comparisons"][0]["status"]=="one-resolved-pair"
    assert out.entries[2].assembly is None and out.entries[2].event_run is None
    refused=out.entries[2].as_dict(baseline,out.plan.questions)
    assert refused["source_inventory"] is None
    assert refused["event_time_comparisons"]==[]


def test_p3_event_specific_stop_side_refusal_does_not_reject_physical_reassembly():
    out=run_case("terminal-refusal")
    assert out.baseline_run.trajectory.termination=="duration"
    assert [e.status for e in out.entries]==["rejected-event-study","complete"]
    first=out.entries[0]
    assert first.assembly is not None and first.event_run is None
    assert first.assembly.parent is out.plan.family.baseline
    assert first.assembly.scenario.spec.field.sources[1].field_v_m==(2000.,0.,0.)
    assert "initial release" in first.reason
    second=out.entries[1]
    assert second.assembly is not None
    assert second.event_run.trajectory.termination=="duration"
    assert second.event_run.as_dict()["outcomes"]["stop:x0"]=="not-found-at-requested-resolution"
    payload=out.as_dict()
    assert payload["entries"][0]["structural_assembly"] is not None
    assert payload["entries"][0]["event_run"] is None
    assert payload["entries"][0]["status"]=="rejected-event-study"
    assert payload["entries"][0]["source_inventory"] is not None
    assert payload["entries"][0]["event_time_comparisons"]==[]
    assert "No located root" in render_structural_event_family(out)


def test_p4_two_crossings_each_result_unpaired_never_arbitrary_first_root():
    out=run_case("double-crossing")
    id="x:two-crossings"
    assert out.baseline_run.trajectory.termination=="duration"
    assert out.entries[0].status=="complete"
    for r in (out.baseline_run,out.entries[0].event_run):
        assert r.as_dict()["outcomes"][id]=="located"
        located=_roots(r,id)
        assert len(located)==2
        assert [x.direction for x in located]==[+1,-1]
        assert located[0].sample.time_s < located[1].sample.time_s
    record=out.entries[0].as_dict(out.baseline_run,out.plan.questions)
    compare=record["event_time_comparisons"][0]
    assert compare["status"]=="unpaired"
    assert compare["baseline_root_count"]==2
    assert compare["variation_root_count"]==2
    assert compare["baseline_termination"]==compare["variation_termination"]=="duration"
    assert "estimated_numerical_shift_s" not in compare
    html=render_structural_event_family(out)
    assert "signed direction +1" in html and "signed direction -1" in html


def test_p5_dense_16_terminal_candidates_preserved_but_only_count_glyphs():
    out=run_case("coincident-terminal")
    assert len(out.plan.questions)==16
    native=[out.baseline_run]+[e.event_run for e in out.entries]
    assert [x.trajectory.termination for x in native]==["terminal-event"]*3
    assert [len(x.events) for x in native]==[16,16,16]
    assert all(x.ordering=="coincident-or-order-unresolved-within-time-resolution"
               for x in native)
    assert all(e.status=="censored" for e in out.entries)
    evidence=out.as_dict()
    assert sum(len(x["event_run"]["events"]) for x in evidence["entries"])+len(
        evidence["baseline_event_run"]["events"])==48
    assert all(c["status"]=="unpaired" for e in evidence["entries"]
               for c in e["event_time_comparisons"])
    html=render_structural_event_family(out)
    root=ET.fromstring(html[html.index('<svg id="structural-event-timeline"'):
                          html.index("</svg>")+len("</svg>")])
    w,h=map(float,root.attrib["viewBox"].split()[2:])
    markers=[el for el in root.iter() if el.tag in ("circle","rect")
             and el.get("data-root-count") is not None]
    assert sum(int(m.get("data-root-count")) for m in markers)==48
    assert sum(1 for m in markers if int(m.get("data-root-count"))>1)>=3
    for i,run in enumerate(native):
        own=[x for x in markers if x.get("data-lane-index")==str(i)]
        assert sum(int(x.get("data-root-count")) for x in own)==16
        y=38+i*52
        real_stop=run.trajectory.samples[-1].time_s
        stop_x=187+real_stop*(930-187-37)/out.plan.family.baseline.spec.duration_s
        lines=[x for x in root.iter() if x.get("data-observed-lane")==str(i)]
        assert len(lines)==1
        assert float(lines[0].get("x2"))==pytest.approx(stop_x,abs=1e-6)
        for el in own:
            if el.tag=="circle":
                x,ypos,rx=float(el.get("cx")),float(el.get("cy")),float(el.get("r"))
                assert 0<=x-rx and x+rx<=w
                assert abs(ypos-y)+rx<26
            else:
                x,ypos,rx=float(el.get("x")),float(el.get("y")),float(el.get("width"))
                assert 0<=x and x+rx<=w
                assert abs((ypos+9)-y)+9<26
            assert float(el.get("data-stop-x"))==pytest.approx(stop_x,abs=1e-6)
    assert html.count('data-event-id="terminal:')>=48


def test_p6_rotated_point_plus_new_source_event_times_covariant():
    """Novel physically qualified rigid rotation; direct independent point Coulomb reference."""
    def rot(v):
        return (-v[1],v[0],v[2])
    point=(.2,-.07,0.)
    add=(.27,.06,0.)
    launch=(0.,.02,0.)
    velocity=(.05,0.,0.)
    query=(.003,0.,0.)
    def plan(rotated):
        def R(v):return rot(v) if rotated else v
        parent=PrescribedScenario(
            "m29b:rot-original" if rotated else "m29b:unrot-original",
            TestChargeMotionSpec(
                PrescribedField((FixedPointField(.6e-9,R(point)),)),
                PointCharge(-1e-9,R(launch)),1e-5,R(velocity),.1,.01,
                plane_normal=(0.,0.,1.),source_guard_m=.006),
            (("point:original","Held point with non-axis offset"),),
            "Teacher hypothetical: novel rotated physical detector and source insertion.")
        family=StructuralFamilyPlan(
            "m29b:rot" if rotated else "m29b:unrot",parent,(
                StructuralCase("add-point","Add one held point at new in-plane location.",
                               add_sources=(NamedSource(
                                   "point:new","Additional held point",
                                   FixedPointField(.3e-9,R(add))),)),))
        event=(PlaneEvent("detector:rigid",R(query),R((1.,0.,0.)),
                          "Physical rotated passive detector",direction=1),)
        return run_structural_event_family(StructuralEventFamilyPlan(
            family,event,EventPolicy(.005,1e-9,1e-8,1e-7)))
    original,turned=plan(False),plan(True)
    for a,b in ((original.baseline_run,turned.baseline_run),
                (original.entries[0].event_run,turned.entries[0].event_run)):
        assert a.trajectory.termination==b.trajectory.termination=="duration"
        ea,eb=_roots(a,"detector:rigid"),_roots(b,"detector:rigid")
        assert len(ea)==len(eb)==1
        assert ea[0].direction==eb[0].direction==1
        assert ea[0].sample.time_s==pytest.approx(eb[0].sample.time_s,abs=2e-8)
        assert eb[0].sample.position_m==pytest.approx(rot(ea[0].sample.position_m),abs=5e-9)
        assert eb[0].sample.velocity_m_s==pytest.approx(rot(ea[0].sample.velocity_m_s),abs=5e-9)
        assert eb[0].sample.force_n==pytest.approx(rot(ea[0].sample.force_n),abs=5e-14)
        spec=a.study.scenario.spec
        field=[0.,0.,0.]
        for source in spec.field.sources:
            dr=tuple(spec.particle.position_m[i]-source.position_m[i] for i in range(3))
            size=hypot(*dr)
            k=source.charge_c/(4*pi*EPSILON_0*size**3)
            for i in range(3):
                field[i]+=k*dr[i]
        expected=tuple(spec.particle.charge_c*f for f in field)
        assert a.trajectory.samples[0].force_n==pytest.approx(expected,abs=5e-14)
    first=original.entries[0].as_dict(original.baseline_run,original.plan.questions)
    second=turned.entries[0].as_dict(turned.baseline_run,turned.plan.questions)
    assert first["event_time_comparisons"][0]["status"]=="one-resolved-pair"
    assert first["event_time_comparisons"][0]["estimated_numerical_shift_s"]==pytest.approx(
        second["event_time_comparisons"][0]["estimated_numerical_shift_s"],abs=2e-8)


def test_n1_bad_plans_and_refused_structural_siblings():
    plan=make_plan("uniform-detector")
    with pytest.raises(TypeError,match="immutable tuple"):
        StructuralEventFamilyPlan(plan.family,list(plan.questions),plan.policy)
    with pytest.raises(ValueError,match="duplicate"):
        StructuralEventFamilyPlan(
            plan.family,(PlaneEvent("same",(.005,0,0),(1,0,0),"A"),
                         PlaneEvent("same",(.006,0,0),(1,0,0),"B")),plan.policy)
    with pytest.raises(TypeError,match="unsupported event"):
        StructuralEventFamilyPlan(plan.family,("x=.006",),plan.policy)
    with pytest.raises(ValueError,match="1..16"):
        StructuralEventFamilyPlan(plan.family,(),plan.policy)
    with pytest.raises(TypeError,match="StructuralFamilyPlan"):
        StructuralEventFamilyPlan({"not": "a native family"},plan.questions,plan.policy)
    with pytest.raises(Exception):
        StructuralEventFamilyPlan(plan.family,(
            VelocityReversal("not-physical",(0.,0.,1.),"normal only"),),plan.policy)
    with pytest.raises(TypeError,match="native"):
        render_structural_event_family(run_case("uniform-detector").as_dict())


def test_n4_unchanged_qualified_source_revisions_and_no_fabricated_energy_work():
    plan=make_plan("uniform-detector")
    original=plan.family.baseline
    before=original.as_dict()
    result=run_structural_event_family(plan)
    report=result.as_dict()
    assert original.as_dict()==before
    assert report["baseline_scenario"]==before
    assert report["baseline_termination"]=="duration"
    assert report["time_interpolation"] is False
    assert report["postguard_extrapolation"] is False
    assert "not" in report["qualification"].lower()
    assert all(e["structural_assembly"]["parent_model"]["revision_id"]==
               original.revision_id for e in report["entries"])
    assert all(e["structural_assembly"]["child_model"]["revision_id"]==
               e["structural_assembly"]["child_revision_id"] for e in report["entries"])
    report["entries"][0]["event_run"]["trajectory"]["samples"][0]["position_m"][0]=999
    assert result.entries[0].event_run.trajectory.samples[0].position_m[0]!=999


def test_n6_offline_html_strictly_escapes_hostile_source_labels_and_provenance(tmp_path):
    plain=make_plan("uniform-detector")
    source=plain.family.baseline
    hostile=PrescribedScenario(
        "m29b:hostile",source.spec,
        (("field:E0","<img src=x onerror=alert(1)>"),
         ("field:Ey","Another named uniform E source")),
        'Teacher <img src=x onerror=alert(1)> </script><script>window.pwned=true</script>')
    family=StructuralFamilyPlan(
        "m29b:malicious-labels",hostile,(
            StructuralCase("add-new","Teacher adds held field with markup-looking label.",
                add_sources=(NamedSource(
                    "field:extra","<svg onload=window.pwned=true>",
                    UniformElectricField((100.,0.,0.))),)),))
    result=run_structural_event_family(StructuralEventFamilyPlan(
        family,plain.questions,plain.policy))
    text=render_structural_event_family(result,title="<teacher events>")
    assert "&lt;teacher events&gt;" in text
    assert "<img src=x onerror=alert(1)>" not in text
    assert "</script><script>window.pwned=true" not in text
    assert r"\u003cimg" in text
    assert "source-memberships" in text
    assert 'id="m29b-native-evidence"' in text
    assert "EVENT-TIME" in text
    assert "Workbook" not in text and "Q41" not in text
    assert "innerHTML" not in text
    assert len(result.entries)==1
    path=write_structural_event_family_html(result,tmp_path/"event"/"study.html")
    assert path.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert json.dumps(result.as_dict(),allow_nan=False)
