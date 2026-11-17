"""M35b: legacy sweep status/native consistency, model binding and issuance.

Source M35a issue #80 red repro is preserved. No new field/ODE/event law.
"""
from dataclasses import replace
import json

import pytest

from conductor_sim import run_trajectory_sweep, run_event_sweep
from conductor_sim.trajectory_sweep import TrajectorySweepRun, SweepEntry
from conductor_sim.event_sweep import EventSweepRun, EventSweepEntry
from examples.m28b_event_sweep import make_plan


def _runs():
    plan=make_plan("uniform-mass")
    return run_trajectory_sweep(plan.sweep),run_event_sweep(plan)


def test_issue80_original_complete_without_model_and_native_rejected_at_entry_construction():
    trajectory,event=_runs()
    t=trajectory.entries[0]
    e=event.entries[0]
    with pytest.raises((TypeError,ValueError)):
        replace(t, scenario=None, trajectory=None,
                status="complete", explanation="forged")
    with pytest.raises((TypeError,ValueError)):
        replace(e, scenario=None, run=None,
                status="complete", reason="forged")


def test_issue80_in_place_native_status_mutation_refused_before_export():
    trajectory,event=_runs()
    t=trajectory.entries[0]
    e=event.entries[0]
    object.__setattr__(t,"status","censored")
    with pytest.raises((TypeError,ValueError)):
        trajectory.as_dict()
    object.__setattr__(e,"status","rejected-physical-model")
    with pytest.raises((TypeError,ValueError)):
        event.as_dict()


def test_reconstructing_runs_or_mutating_reason_cannot_import_old_native_as_new_evidence():
    trajectory,event=_runs()
    with pytest.raises((TypeError,ValueError)):
        TrajectorySweepRun(trajectory.plan,trajectory.baseline_trajectory,trajectory.entries)
    with pytest.raises((TypeError,ValueError)):
        EventSweepRun(event.plan,event.baseline_run,event.entries)
    with pytest.raises((TypeError,ValueError)):
        replace(trajectory,entries=(replace(trajectory.entries[0],
            explanation="fabricated"),)+trajectory.entries[1:])
    with pytest.raises((TypeError,ValueError)):
        replace(event,entries=(replace(event.entries[0],
            reason="fabricated"),)+event.entries[1:])


def test_legitimate_event_refusal_accepted_without_fake_trajectory_or_missing_model():
    from conductor_sim import SweepAxis,TrajectorySweepPlan
    from conductor_sim.motion_events import PlaneEvent
    from conductor_sim.event_sweep import EventSweepPlan
    original=make_plan("uniform-mass")
    sweep=TrajectorySweepPlan("m35b:stop",original.sweep.baseline,
        SweepAxis("test-particle","position_m",((.06,0.,0.),(.01,0.,0.))))
    event=run_event_sweep(EventSweepPlan(
        sweep,(PlaneEvent("stop:x",(.05,0.,0.),(1.,0.,0.),"Stop",
                          direction=1,terminal=True,role="ideal-stop-surface"),),
        original.policy))
    assert event.entries[0].status=="rejected-event-study"
    assert event.entries[0].scenario is not None
    assert event.entries[0].run is None
    assert event.entries[1].status=="censored"
    assert event.entries[1].run.trajectory.termination=="terminal-event"
    assert event.as_dict()["entries"][0]["event_run"] is None
    assert event.as_dict()["entries"][1]["status"]=="censored"


def test_legitimate_physical_rejection_and_unchanged_sibling_and_native():
    from conductor_sim import SweepAxis,TrajectorySweepPlan
    from examples.m27b_offaxis_motion import author
    base,_=author("workbook-xy")
    plan=TrajectorySweepPlan("m35b:refused",base,
        SweepAxis("ring:B","radius_m",(-.2,.31)))
    out=run_trajectory_sweep(plan)
    assert [r.status for r in out.entries]==["rejected-physical-model","complete"]
    assert out.entries[0].scenario is None
    assert out.entries[0].trajectory is None
    assert out.entries[1].trajectory.spec is out.entries[1].scenario.spec
    assert out.as_dict()["entries"][0]["trajectory"] is None


def test_native_saved_float_boolean_tampering_rejected_on_export():
    trajectory,event=_runs()
    orig=trajectory.entries[0].trajectory.samples[0]
    object.__setattr__(orig,"time_s",False)
    with pytest.raises((ValueError,TypeError)):
        trajectory.as_dict()
    native=event.entries[0].run.events[0]
    object.__setattr__(native,"time_bracket_s",(False,native.time_bracket_s[1]))
    with pytest.raises((TypeError,ValueError)):
        event.as_dict()


def test_forged_parent_revision_and_plan_value_fail_closed():
    trajectory,event=_runs()
    model=trajectory.entries[0].scenario
    object.__setattr__(model,"parent_revision_id","fake-sha")
    with pytest.raises((ValueError,TypeError)):
        trajectory.as_dict()
    object.__setattr__(event.entries[0],"requested_value",True)
    with pytest.raises((ValueError,TypeError)):
        event.as_dict()


def test_censored_trajectory_without_actual_guard_is_rejected():
    trajectory,event=_runs()
    tr=trajectory.entries[0]
    with pytest.raises((ValueError,TypeError)):
        replace(tr,status="censored",explanation="source-guard")
    er=event.entries[0]
    with pytest.raises((ValueError,TypeError)):
        replace(er,status="censored",reason="source-guard")


def test_issued_exports_are_detached_and_repeatably_identical():
    trajectory,event=_runs()
    for run in (trajectory,event):
        before=json.dumps(run.as_dict(),sort_keys=True,allow_nan=False)
        exported=run.as_dict()
        exported["entries"][0]["status"]="forged"
        assert json.dumps(run.as_dict(),sort_keys=True,allow_nan=False)==before


def test_actual_multi_root_and_guard_outcome_conservatively_unpaired():
    for name in ("multi-crossing","gravity-guard"):
        run=run_event_sweep(make_plan(name))
        record=run.as_dict()
        for entry in record["entries"]:
            if entry["event_time_comparisons"]:
                for comparison in entry["event_time_comparisons"]:
                    assert comparison["status"]!="one-resolved-pair"
                    assert "estimated_numerical_shift_s" not in comparison


def test_native_source_and_policy_cannot_be_swapped_after_issuance():
    _,event=_runs()
    scenario=event.entries[0].scenario
    object.__setattr__(event.entries[0].run.study,"scenario",event.plan.sweep.baseline)
    with pytest.raises((ValueError,TypeError)):
        event.as_dict()


def test_event_native_root_pinned_and_rejection_statuses_are_exact():
    _,event=_runs()
    e=event.entries[0].run.events[0]
    old=e.sample.time_s
    object.__setattr__(e.sample,"time_s",old+.03)
    with pytest.raises((ValueError,TypeError)):
        event.as_dict()
