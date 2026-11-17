"""M35b: real old M28a/M28b result integrity and honest physical evidence.

No changes to historical field laws or numerical integrators. Includes the
literal previously released issue #80 complete-without-native exploit.
"""
from dataclasses import replace

import pytest

from conductor_sim import (
    EventSweepPlan, EventSweepRun, GraphWorldForkPlan, PhysicalWorldNode,
    SweepAxis, TrajectorySweepPlan, TrajectorySweepRun,
    run_event_sweep, run_trajectory_sweep,
)
from conductor_sim.event_sweep import EventSweepEntry
from conductor_sim.motion_events import EventPolicy, PlaneEvent
from conductor_sim.trajectory_sweep import SweepEntry
from examples.m28b_event_sweep import make_plan


@pytest.fixture(scope="module")
def events():
    result = run_event_sweep(make_plan("uniform-mass"))
    assert result.entries[0].status == "complete"
    return result


@pytest.fixture(scope="module")
def motion():
    result = run_trajectory_sweep(make_plan("uniform-mass").sweep)
    assert result.entries[0].status == "complete"
    return result


def _changed(run, item):
    return (item,) + run.entries[1:]


def test_01_original_issue_80_event_complete_without_model_or_native_fails_at_construction(events):
    bad = replace(events.entries[0], scenario=None, run=None, status="complete", reason="forged")
    with pytest.raises(ValueError, match="complete/censored"):
        replace(events, entries=_changed(events, bad))
    with pytest.raises(ValueError, match="complete/censored"):
        bad.as_dict(events.baseline_run, events.plan.questions)


def test_02_original_issue_80_motion_complete_without_model_or_native_fails(motion):
    bad = replace(motion.entries[0], scenario=None, trajectory=None,
                  status="complete", explanation="forged")
    with pytest.raises(ValueError, match="complete/censored"):
        replace(motion, entries=_changed(motion, bad))
    with pytest.raises(ValueError, match="complete/censored"):
        bad.as_dict(motion.baseline_trajectory)


def test_03_event_status_cannot_claim_complete_for_genuine_censored_run():
    p = make_plan("gravity-guard")
    r = run_event_sweep(p)
    assert r.baseline_run.trajectory.termination == "source-guard"
    native = r.baseline_run
    bad = replace(r.entries[0], run=native, status="complete",
                  scenario=r.plan.sweep.baseline)
    with pytest.raises(ValueError):
        replace(r, entries=_changed(r,bad))


def test_04_event_rejection_must_not_fabricate_model_or_native(events):
    good = events.entries[0]
    for status, scenario, native in (
        ("rejected-physical-model", good.scenario, None),
        ("rejected-event-study", None, None),
        ("integration-failed", good.scenario, good.run),
        ("censored", good.scenario, None),
        ("other", good.scenario, good.run),
    ):
        with pytest.raises(ValueError):
            replace(events, entries=_changed(events, replace(
                good, status=status, scenario=scenario, run=native)))


def test_05_motion_rejection_and_guard_status_must_not_fabricate_data(motion):
    good = motion.entries[0]
    for status,scenario,native in (
        ("rejected-physical-model",good.scenario,None),
        ("integration-failed",good.scenario,good.trajectory),
        ("censored",good.scenario,None),
        ("unknown",good.scenario,good.trajectory),
    ):
        with pytest.raises(ValueError):
            replace(motion, entries=_changed(motion,replace(
                good,status=status,scenario=scenario,trajectory=native)))


def test_06_honest_censor_and_refusal_native_runs_remain_exportable():
    e=run_event_sweep(make_plan("gravity-guard"))
    assert e.baseline_run.trajectory.termination=="source-guard"
    assert e.as_dict()["baseline_termination"]=="source-guard"
    m=run_trajectory_sweep(make_plan("gravity-guard").sweep)
    assert m.as_dict()["baseline_termination"] in ("duration","source-guard")
    p=make_plan("ring-detector")
    altered=TrajectorySweepPlan("m35b:oblique",p.sweep.baseline,
                               SweepAxis("ring:B","center_m",(
                                   (.4,0.,.0001),(.35,0.,0.))))
    rejected=run_event_sweep(EventSweepPlan(altered,p.questions,p.policy))
    assert rejected.entries[0].status=="rejected-physical-model"
    assert rejected.entries[0].scenario is None and rejected.entries[0].run is None
    assert rejected.entries[1].status=="complete"
    assert rejected.as_dict()["entries"][0]["event_run"] is None


def test_07_valid_terminal_event_refusal_survives_native_integrity():
    p=make_plan("uniform-mass")
    stop=PlaneEvent("stop:x",(.05,0.,0.),(1.,0.,0.),
                    "Terminal static ideal stop plane", direction=1,
                    terminal=True,role="ideal-stop-surface")
    altered=TrajectorySweepPlan("m35b:terminal",p.sweep.baseline,
        SweepAxis("test-particle","position_m",((.06,0.,0.),(.01,0.,0.))))
    r=run_event_sweep(EventSweepPlan(altered,(stop,),p.policy))
    assert r.baseline_run.trajectory.termination=="terminal-event"
    assert r.entries[0].status=="rejected-event-study"
    assert r.entries[0].scenario is not None and r.entries[0].run is None
    assert r.entries[1].status=="censored"
    assert r.entries[1].run.trajectory.termination=="terminal-event"
    assert r.as_dict()["entries"][0]["event_time_comparisons"]==[]


def test_08_native_event_questions_policy_and_parent_must_match(events):
    good=events.entries[0]
    other=events.entries[1]
    with pytest.raises(ValueError):
        replace(events,entries=_changed(events,
            replace(good,scenario=other.scenario)))
    with pytest.raises(ValueError):
        replace(events,entries=_changed(events,
            replace(good,run=other.run)))
    wrong=replace(events.plan, policy=EventPolicy(.02,1e-8,1e-8,1e-6))
    with pytest.raises(ValueError):
        replace(events,plan=wrong)


def test_09_native_model_revision_and_source_identity_alteration_refused(motion):
    good=motion.entries[0]
    sibling=motion.entries[1]
    with pytest.raises(ValueError):
        replace(motion,entries=_changed(motion,
            replace(good, scenario=sibling.scenario)))
    forged=replace(good, explanation="FICTIONAL COMPLETE EXPERIMENT")
    with pytest.raises(ValueError,match="receipt"):
        replace(motion,entries=_changed(motion,forged))


def test_10_in_process_issued_receipt_detects_after_construction_change(events,motion):
    for run in (events,motion):
        snapshot=run.as_dict()
        original=run.entries
        try:
            altered=replace(original[0],status="censored")
            object.__setattr__(run,"entries",(altered,)+original[1:])
            with pytest.raises(ValueError):
                run.as_dict()
        finally:
            object.__setattr__(run,"entries",original)
        assert run.as_dict()==snapshot


def test_11_bool_zero_timestamp_tamper_detected_before_export(motion):
    entry=motion.entries[0]
    native=entry.trajectory
    assert native.samples[0].time_s==0.0
    bad0=replace(native.samples[0],time_s=False)
    forged=replace(native,samples=(bad0,)+native.samples[1:])
    with pytest.raises(ValueError):
        replace(motion,entries=_changed(motion,replace(entry,trajectory=forged)))
    with pytest.raises(ValueError):
        replace(entry,trajectory=forged).as_dict(motion.baseline_trajectory)


def test_12_bool_zero_event_trajectory_tamper_detected(events):
    entry=events.entries[0]
    native=entry.run
    trajectory=native.trajectory
    bad0=replace(trajectory.samples[0],time_s=False)
    forged=replace(native,trajectory=replace(trajectory,samples=(
        bad0,)+trajectory.samples[1:]))
    with pytest.raises(ValueError):
        replace(events,entries=_changed(events,replace(entry,run=forged)))


def test_13_forged_event_root_and_bracket_rejected(events):
    entry=events.entries[0]
    observed=entry.run
    assert observed.events
    root=observed.events[0]
    forged=replace(root,sample=replace(root.sample,time_s=root.sample.time_s+.03),
                   time_bracket_s=tuple(t+.03 for t in root.time_bracket_s))
    bad_run=replace(observed,events=(forged,)+observed.events[1:])
    with pytest.raises(ValueError):
        replace(events,entries=_changed(events,replace(entry,run=bad_run)))


def test_14_native_termination_must_match_complete_or_guard_status(motion):
    entry=motion.entries[0]
    assert entry.trajectory.termination=="duration"
    altered=replace(entry.trajectory,termination="source-guard")
    with pytest.raises(ValueError):
        replace(motion,entries=_changed(motion,replace(entry,trajectory=altered)))


def test_15_direct_unauthenticated_construction_has_no_runner_issued_receipt(events,motion):
    with pytest.raises(ValueError,match="issued"):
        EventSweepRun(events.plan,events.baseline_run,events.entries)
    with pytest.raises(ValueError,match="issued"):
        TrajectorySweepRun(motion.plan,motion.baseline_trajectory,motion.entries)
    with pytest.raises(ValueError,match="issued"):
        replace(events,_issued_digest="0"*64)


def test_16_detached_json_edits_do_not_change_original_native_exports(events,motion):
    for run in (events,motion):
        before=run.as_dict()
        fake=run.as_dict()
        fake["entries"][0]["status"]="FAKE"
        fake["baseline_scenario"]["spec"]["mass_kg"]=999.
        assert run.as_dict()==before


def test_17_degenerate_wrong_native_entry_and_bool_axis_value_refused(events,motion):
    with pytest.raises(ValueError):
        replace(events,entries=_changed(events,replace(events.entries[0],index=3)))
    with pytest.raises(ValueError):
        replace(motion,entries=_changed(motion,
            replace(motion.entries[0],requested_value=False)))


def test_18_m35a_multi_stage_source_refuses_unauthenticated_legacy_sweeps(events):
    from conductor_sim import GraphWorldForkPlan
    from conductor_sim.graph_world_fork import GraphWorldForkPlan as NativeForkPlan
    with pytest.raises(TypeError):
        NativeForkPlan(events, "legacy-cannot-fork", ())
    with pytest.raises(TypeError):
        NativeForkPlan(motion, "legacy-cannot-fork", ())


def test_19_historical_valid_run_serialization_preserves_v1_schema(events,motion):
    ed=events.as_dict()
    td=motion.as_dict()
    assert ed["schema"]=="conductor_sim.event_sweep.v1"
    assert td["schema"]=="conductor_sim.trajectory_sweep.v1"
    assert [row["status"] for row in ed["entries"]]==["complete","complete"]
    assert [row["status"] for row in td["entries"]]==["complete","complete"]
