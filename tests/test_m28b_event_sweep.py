"""M28b: independent event-time tests across immutably revised physical models."""
from math import sqrt
import json

import pytest

from conductor_sim import (
    FixedPointField, PointCharge, PrescribedField, TestChargeMotionSpec,
    SweepAxis, TrajectorySweepPlan, UniformElectricField,
)
from conductor_sim.event_sweep import (
    EventSweepPlan, EventSweepRun, run_event_sweep,
)
from conductor_sim.event_sweep_viewer import render_event_sweep, write_event_sweep_html
from conductor_sim.motion_events import PlaneEvent, VelocityReversal, EventPolicy
from conductor_sim.scenarios import PrescribedScenario
from examples.m28b_event_sweep import make_plan, run_case


def test_p1_same_real_ring_model_three_independent_event_investigations():
    plan = make_plan("ring-detector")
    before = plan.sweep.baseline.as_dict()
    run = run_event_sweep(plan)
    assert run.plan is plan
    assert run.baseline_run.study.scenario is plan.sweep.baseline
    assert run.plan.sweep.baseline.as_dict() == before
    assert [e.status for e in run.entries] == ["complete", "complete"]
    assert len(run.plan.questions) == 1
    qid = run.plan.questions[0].event_id
    assert qid == "x:detector"
    for entry, charge in zip(run.entries, plan.sweep.axis.values):
        assert entry.scenario.parent_revision_id == plan.sweep.baseline.revision_id
        assert entry.scenario.source_annotations == plan.sweep.baseline.source_annotations
        assert entry.scenario.spec.field.sources[1].charge_c == charge
        assert entry.run.study.scenario is entry.scenario
        assert entry.run.trajectory.spec is entry.scenario.spec
        assert entry.run.trajectory.samples[-1].time_s <= .015
        assert entry.run.trajectory.termination == "duration"
        assert entry.run.as_dict()["outcomes"][qid] in (
            "located", "not-found-at-requested-resolution")
    record = run.as_dict()
    assert record["schema"] == "conductor_sim.event_sweep.v1"
    assert record["baseline_scenario"]["revision_id"] == plan.sweep.baseline.revision_id
    assert record["baseline_outcomes"][qid] in (
        "located", "not-found-at-requested-resolution")
    assert record["time_interpolation"] is False and record["postguard_extrapolation"] is False
    record["entries"][0]["scenario"]["spec"]["mass_kg"] = 9999.
    assert run.entries[0].scenario.spec.mass_kg != 9999.
    assert plan.sweep.baseline.as_dict() == before


def test_p2_uniform_field_analytic_detector_times_and_numerical_shift_brackets():
    run = run_case("uniform-mass")
    assert run.baseline_run.trajectory.termination == "duration"
    assert [x.status for x in run.entries] == ["complete", "complete"]
    qid = "x:detector"
    base_time = sqrt(2*.05*.001 / (1e-6*1000.))
    cases = [(run.baseline_run, .001)] + [
        (e.run, e.scenario.spec.mass_kg) for e in run.entries]
    for observed, mass in cases:
        event = [e for e in observed.events if e.event_id == qid]
        assert len(event) == 1, observed.as_dict()["outcomes"]
        assert observed.as_dict()["outcomes"][qid] == "located"
        expected = sqrt(2*.05*mass/(1e-6*1000.))
        assert event[0].sample.time_s == pytest.approx(expected, abs=2e-8)
        assert event[0].time_bracket_s[0] <= event[0].sample.time_s <= event[0].time_bracket_s[1]
        assert event[0].time_bracket_s[1]-event[0].time_bracket_s[0] <= run.plan.policy.time_tolerance_s*1.01
        assert event[0].sample.position_m[0] == pytest.approx(.05, abs=1e-9)
    assert run.baseline_run.events[0].sample.time_s == pytest.approx(base_time, abs=2e-8)
    for e in run.entries:
        comparisons = e.as_dict(run.baseline_run, run.plan.questions)["event_time_comparisons"]
        assert len(comparisons) == 1
        shift = comparisons[0]
        assert shift["status"] == "one-resolved-pair"
        lo, hi = shift["numerical_bracket_shift_interval_s"]
        assert lo <= shift["estimated_numerical_shift_s"] <= hi
        assert shift["event_id"] == qid
        assert (shift["estimated_numerical_shift_s"] < 0) == (e.scenario.spec.mass_kg < .001)
    # Initial state and derivative labels are original M24a observations, not
    # approximate time interpolation from M28a's historical saved samples.
    assert run.baseline_run.trajectory.samples[0].time_s == 0.
    assert "time_bracket_s" in json.dumps(run.as_dict())


def test_p3_terminal_stop_side_refusal_is_separate_from_physical_model_rejection():
    orig = make_plan("uniform-mass")
    base = orig.sweep.baseline
    sweep = TrajectorySweepPlan(
        "m28b:terminal-study", base,
        SweepAxis("test-particle", "position_m", ((.06,0.,0.),(.01,0.,0.))))
    q = PlaneEvent("stop:x", (.05,0.,0.), (1.,0.,0.), "Ideal terminal stop",
                   direction=1, terminal=True, role="ideal-stop-surface")
    run = run_event_sweep(EventSweepPlan(sweep, (q,), orig.policy))
    assert run.baseline_run.trajectory.termination == "terminal-event"
    assert run.entries[0].status == "rejected-event-study"
    assert run.entries[0].scenario is not None and run.entries[0].run is None
    assert "beyond" in run.entries[0].reason
    assert run.entries[1].status == "censored"
    assert run.entries[1].run.trajectory.termination == "terminal-event"
    assert run.entries[1].run.events[0].terminal
    assert run.entries[0].as_dict(run.baseline_run, (q,))["event_time_comparisons"] == []
    assert base.spec.particle.position_m == (0.,0.,0.)


def test_p3_physical_oblique_ring_refusal_keeps_other_event_study():
    orig = make_plan("ring-detector")
    out = run_event_sweep(EventSweepPlan(
        TrajectorySweepPlan("m28b:oblique-ring",orig.sweep.baseline,
                            SweepAxis("ring:B", "center_m",
                                      ((.4,0.,.0001),(.35,0.,0.)))),
        orig.questions, orig.policy))
    assert out.entries[0].status == "rejected-physical-model"
    assert out.entries[0].scenario is None and out.entries[0].run is None
    assert out.entries[1].status == "complete"
    assert out.entries[1].scenario.parent_revision_id == orig.sweep.baseline.revision_id


def test_p4_source_guard_does_not_imply_future_detector_absence():
    out = run_case("gravity-guard")
    qid = out.plan.questions[0].event_id
    assert out.baseline_run.trajectory.termination == "source-guard"
    assert out.entries[0].status == "complete"
    assert out.entries[0].run.trajectory.termination == "duration"
    baseline_status = out.baseline_run.as_dict()["outcomes"][qid]
    child_status = out.entries[0].run.as_dict()["outcomes"][qid]
    assert baseline_status == "search-truncated"
    assert child_status == "not-found-at-requested-resolution"
    pair=out.entries[0].as_dict(out.baseline_run,out.plan.questions)["event_time_comparisons"][0]
    assert pair["status"] == "unpaired"
    assert pair["baseline_outcome"]=="search-truncated"
    assert pair["variation_outcome"]=="not-found-at-requested-resolution"
    assert "No located root" in render_event_sweep(out)
    record=out.as_dict()
    assert record["baseline_event_run"]["trajectory"]["termination"]=="source-guard"
    assert record["entries"][0]["event_run"]["trajectory"]["termination"]=="duration"


def test_n4_two_detector_crossings_never_get_paired_arbitrarily():
    out = run_case("multi-crossing")
    qid=out.plan.questions[0].event_id
    assert len([e for e in out.baseline_run.events if e.event_id==qid]) == 2
    assert len([e for e in out.entries[0].run.events if e.event_id==qid]) == 2
    assert out.baseline_run.as_dict()["outcomes"][qid] == "located"
    assert out.entries[0].run.as_dict()["outcomes"][qid] == "located"
    shift=out.entries[0].as_dict(out.baseline_run,out.plan.questions)["event_time_comparisons"][0]
    assert shift["status"]=="unpaired"
    assert shift["baseline_root_count"]==shift["variation_root_count"]==2
    assert "estimated_numerical_shift_s" not in shift


def test_n2_event_baseline_bad_plan_refuses_without_integration():
    base=make_plan("uniform-mass")
    with pytest.raises(ValueError,match="duplicate"):
        EventSweepPlan(base.sweep,
                       (PlaneEvent("same",(.1,0,0),(1,0,0),"A"),
                        PlaneEvent("same",(.2,0,0),(1,0,0),"B")),base.policy)
    with pytest.raises(TypeError,match="immutable tuple"):
        EventSweepPlan(base.sweep, list(base.questions), base.policy)
    with pytest.raises(TypeError,match="event"):
        EventSweepPlan(base.sweep, ("cross .05",),base.policy)
    with pytest.raises(ValueError,match="1..16"):
        EventSweepPlan(base.sweep, (),base.policy)
    with pytest.raises(Exception):
        EventSweepPlan(base.sweep, (
            VelocityReversal("invalid",(0.,0.,1.),"Normal velocity"),),base.policy)


def test_p5_rotated_point_plus_uniform_field_event_times_are_covariant():
    """Novel non-ring field, real plane event and physical source sweep."""
    def rot(p):
        return (-p[1], p[0], p[2])
    origin=(0.,0.,0.)
    point=(.22,.04,0.)
    held=(40.,-20.,0.)
    changed=(100.,-20.,0.)
    vel=(.35,.1,0.)
    detector=(.05,0.,0.)
    def build(transformed):
        def r(v):return rot(v) if transformed else v
        sc=PrescribedScenario(
            "m28b:rotated" if transformed else "m28b:world",
            TestChargeMotionSpec(
                PrescribedField((FixedPointField(-2e-8,r(point)),
                                 UniformElectricField(r(held)))),
                PointCharge(5e-9,r(origin)),1e-5,r(vel),
                .18,.015,plane_normal=(0.,0.,1.),source_guard_m=.008),
            (("point:P","Held P"),("field:E","Uniform held E")),
            "Teacher hypothetical: mixed point/uniform event transfer.")
        question=(PlaneEvent("detector:one",r(detector),r((1.,0.,0.)),
                             "Passive detector",direction=1),)
        return run_event_sweep(EventSweepPlan(
            TrajectorySweepPlan(
                "m28b:rotation1" if transformed else "m28b:rotation0",sc,
                SweepAxis("field:E","field_v_m",(r(changed),))),
            question,EventPolicy(.006,1e-9,1e-8,1e-7)))
    a,b=build(False),build(True)
    assert a.baseline_run.trajectory.termination == b.baseline_run.trajectory.termination == "duration"
    assert a.entries[0].status == b.entries[0].status == "complete"
    for left,right in [(a.baseline_run,b.baseline_run),(a.entries[0].run,b.entries[0].run)]:
        assert [e.event_id for e in left.events] == [e.event_id for e in right.events]
        assert len(left.events)==1
        assert left.events[0].sample.time_s == pytest.approx(right.events[0].sample.time_s,abs=2e-8)
        assert right.events[0].sample.position_m == pytest.approx(
            rot(left.events[0].sample.position_m), abs=2e-7)
        assert right.events[0].sample.force_n == pytest.approx(
            rot(left.events[0].sample.force_n), abs=2e-11)
    assert a.entries[0].scenario.parent_revision_id == a.plan.sweep.baseline.revision_id
    assert b.entries[0].scenario.parent_revision_id == b.plan.sweep.baseline.revision_id


def test_p6_html_escapes_source_and_all_event_brackets(tmp_path):
    result=run_case("uniform-mass")
    page=render_event_sweep(result,title="M28b <untrusted>")
    assert "&lt;untrusted&gt;" in page
    assert "field:E" in page
    assert "M28b" in page
    assert 'id="m28b-evidence"' in page
    assert "time_bracket_s" in page and "numerical_bracket_shift_interval_s" in page
    assert 'id="event-timeline"' in page
    assert "Workbook" not in page and "Q41" not in page
    assert "Unpaired" not in page or "one-resolved-pair" in page
    with pytest.raises(TypeError,match="native"):
        render_event_sweep(result.as_dict())
    path=write_event_sweep_html(result,tmp_path/"m28b"/"events.html")
    assert path.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert json.dumps(result.as_dict(),allow_nan=False)
    base=result.plan.sweep.baseline
    hostile=PrescribedScenario(
        "m28b:hostile",base.spec,base.source_annotations,
        'Teacher <img src=x onerror=alert(1)> </script><script>alert("X")</script>')
    sweep=TrajectorySweepPlan(
        "m28b:hostile-sweep",hostile,SweepAxis("test-particle","mass_kg",(.00075,)))
    x=run_event_sweep(EventSweepPlan(sweep,result.plan.questions,result.plan.policy))
    html=render_event_sweep(x)
    assert "<img src=x onerror=alert(1)>" not in html
    assert "</script><script>alert" not in html
    assert r"\u003cimg" in html
