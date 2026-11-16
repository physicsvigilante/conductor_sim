"""M28b: answer identical typed M24a event questions across M28a physical sweeps.

No new equations or event locator: each model is revised by M20b/M28a and
runs the original M24a EventStudy/investigate_motion independently. Event
brackets localize the numerical trajectory's roots, not exact-ODE uncertainty.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from math import isfinite

from .motion_1d import MotionIntegrationError
from .motion_events import (
    EventPolicy, EventStudy, EventRun, PlaneEvent, VelocityReversal,
    investigate_motion, validate_event_run,
)
from .solver import UnsupportedProblem
from .trajectory_sweep import TrajectorySweepPlan, _revision, _shared_rows


@dataclass(frozen=True, slots=True)
class EventSweepPlan:
    """One immutable physical sweep, the SAME typed questions and policy."""
    sweep: TrajectorySweepPlan
    questions: tuple[PlaneEvent | VelocityReversal, ...]
    policy: EventPolicy

    def __post_init__(self):
        if type(self.sweep) is not TrajectorySweepPlan or type(self.policy) is not EventPolicy:
            raise TypeError("event sweep requires an M28a plan and M24a numerical policy")
        if type(self.questions) is not tuple:
            raise TypeError("event questions must be an immutable tuple")
        # This also checks finite-resolution clock, 1..16 typed questions,
        # unique IDs and any initial ideal-stop constraints, before execution.
        EventStudy(self.sweep.baseline, self.questions, self.policy)


def _outcomes(run: EventRun) -> dict:
    # The existing M24a validator owns scientific interpretation:
    # located, unresolved, truncated, not-found-at-requested-resolution.
    return run.as_dict()["outcomes"]


def _root_pair(base: EventRun, child: EventRun, event_id: str) -> dict:
    """Pair only same-direction, unique resolved events over COMPLETE horizons.

    A detector may have one *observed* root before source guard or a terminal
    stop, yet later roots are unknowable. This API compares uniqueness across
    the requested physical horizon, not first observed crossings. For a
    bidirectional query, opposite directions are distinct physical events.
    Actual native event roots and brackets remain in both EventRun records.
    """
    left, right = _outcomes(base), _outcomes(child)
    b = [e for e in base.events if e.event_id == event_id]
    v = [e for e in child.events if e.event_id == event_id]
    metadata = dict(
        event_id=event_id,
        baseline_outcome=left[event_id], variation_outcome=right[event_id],
        baseline_termination=base.trajectory.termination,
        variation_termination=child.trajectory.termination,
        baseline_root_count=len(b), variation_root_count=len(v),
        baseline_directions=[e.direction for e in b],
        variation_directions=[e.direction for e in v],
    )
    if left[event_id] != "located" or right[event_id] != "located":
        return dict(**metadata, status="unpaired",
                    reason="not two independently resolved event outcomes")
    if len(b) != 1 or len(v) != 1:
        return dict(**metadata, status="unpaired",
                    reason="multiple crossings/reversals cannot be silently paired")
    if (base.trajectory.termination != "duration" or
            child.trajectory.termination != "duration"):
        return dict(**metadata, status="unpaired",
                    reason="censored observation horizon: later roots are unknown; "
                           "cannot assert a unique whole-duration event")
    if b[0].direction != v[0].direction:
        return dict(**metadata, status="unpaired",
                    reason="opposite signed crossing/reversal directions are distinct "
                           "even for a bidirectional event question")
    lower = v[0].time_bracket_s[0] - b[0].time_bracket_s[1]
    upper = v[0].time_bracket_s[1] - b[0].time_bracket_s[0]
    diff = v[0].sample.time_s - b[0].sample.time_s
    if not all(isfinite(x) for x in (lower, upper, diff)):
        raise ValueError("nonfinite numerical event-shift interval")
    if not lower <= diff <= upper:
        raise ValueError("numerical event-shift point estimate escapes bracket")
    return dict(**metadata, status="one-resolved-pair",
                direction=b[0].direction,
                baseline_time_bracket_s=b[0].time_bracket_s,
                variation_time_bracket_s=v[0].time_bracket_s,
                estimated_numerical_shift_s=diff,
                numerical_bracket_shift_interval_s=(lower, upper),
                qualification=("Same-direction unique numerical roots in two "
                               "duration-complete studies. Difference of localized "
                               "numerical roots; no bound on global ODE error"))

@dataclass(frozen=True, slots=True)
class EventSweepEntry:
    index: int
    requested_value: float | tuple
    scenario: object | None
    run: EventRun | None
    status: str
    reason: str

    def as_dict(self, baseline: EventRun, questions: tuple):
        data = dict(index=self.index, requested_value=self.requested_value,
                    status=self.status, reason=self.reason,
                    scenario=None if self.scenario is None else self.scenario.as_dict(),
                    event_run=None if self.run is None else self.run.as_dict(),
                    exact_time_trajectory_comparisons=[] if self.run is None else
                    _shared_rows(baseline.trajectory, self.run.trajectory),
                    event_time_comparisons=[] if self.run is None else
                    [_root_pair(baseline, self.run, q.event_id) for q in questions])
        return data


@dataclass(frozen=True, slots=True)
class EventSweepRun:
    plan: EventSweepPlan
    baseline_run: EventRun
    entries: tuple[EventSweepEntry, ...]

    def __post_init__(self):
        if (type(self.plan) is not EventSweepPlan or
                type(self.baseline_run) is not EventRun or
                type(self.entries) is not tuple):
            raise TypeError("native event sweep requires typed plan and M24a run")
        if self.baseline_run.study.scenario is not self.plan.sweep.baseline:
            raise ValueError("baseline event study is not bound to the sweep's baseline")
        if len(self.entries) != len(self.plan.sweep.axis.values):
            raise ValueError("missing event sweep variants")
        for index, entry in enumerate(self.entries, 1):
            if type(entry) is not EventSweepEntry or entry.index != index:
                raise ValueError("invalid event sweep entry identity/order")
            if entry.requested_value != self.plan.sweep.axis.values[index-1]:
                raise ValueError("event sweep lost requested axis values")
            if entry.run is not None and (entry.scenario is None or
                                          entry.run.study.scenario is not entry.scenario):
                raise ValueError("event run and physical model differ")

    def as_dict(self):
        """Detach typed event evidence into finite JSON; never accept it back as physics."""
        base = self.baseline_run.as_dict()
        result = dict(
            schema="conductor_sim.event_sweep.v1",
            mode="M28a independently revised physical cases; M24a new native event solve per case",
            provenance=("Authored source facts are assertions, not authenticated PDFs; "
                        "sweep changes and detectors are teacher hypotheses"),
            plan_id=self.plan.sweep.plan_id,
            axis=dict(component_id=self.plan.sweep.axis.component_id,
                      property_name=self.plan.sweep.axis.property_name,
                      values=self.plan.sweep.axis.values),
            baseline_revision_id=self.plan.sweep.baseline.revision_id,
            baseline_scenario=self.plan.sweep.baseline.as_dict(),
            questions=[dict(kind=type(q).__name__, **asdict(q))
                       for q in self.plan.questions],
            numerical_policy=asdict(self.plan.policy),
            baseline_event_run=base,
            baseline_termination=self.baseline_run.trajectory.termination,
            baseline_outcomes=base["outcomes"],
            entries=[entry.as_dict(self.baseline_run, self.plan.questions)
                     for entry in self.entries],
            time_alignment="exactly identical actual saved physical timestamps only",
            event_shift_rule="only one separately resolved root per case and event ID",
            time_interpolation=False, postguard_extrapolation=False,
            qualification=("Event brackets are finite-resolution localization in a "
                           "numerical trajectory, NOT rigorous exact-ODE error bounds. "
                           "No unlocated event is asserted absent; no terminal collision law."),
        )
        return json.loads(json.dumps(result, allow_nan=False))


def run_event_sweep(plan: EventSweepPlan) -> EventSweepRun:
    """Each case is requalified and re-integrated by the existing M24a event engine.

    Events, not the M28a no-event integration, are the primary physical runs.
    Only expected admission and numerical-domain failures are classified.
    Unexpected implementation exceptions must propagate.
    """
    if type(plan) is not EventSweepPlan:
        raise TypeError("require an EventSweepPlan")
    baseline_sha = plan.sweep.baseline.revision_id
    baseline_study = EventStudy(plan.sweep.baseline, plan.questions, plan.policy)
    baseline = investigate_motion(baseline_study)
    if baseline.study.scenario is not plan.sweep.baseline:
        raise RuntimeError("event service replaced the baseline physical model")
    validate_event_run(baseline)
    rows = []
    for index, value in enumerate(plan.sweep.axis.values, 1):
        try:
            scenario = _revision(plan.sweep, index, value)
        except (ValueError, TypeError, UnsupportedProblem) as exc:
            rows.append(EventSweepEntry(index, value, None, None,
                                        "rejected-physical-model", f"{type(exc).__name__}: {exc}"))
            continue
        if scenario.parent_revision_id != baseline_sha:
            raise RuntimeError("physical revision lost parent lineage")
        try:
            study = EventStudy(scenario, plan.questions, plan.policy)
        except (ValueError, TypeError, UnsupportedProblem) as exc:
            rows.append(EventSweepEntry(index, value, scenario, None,
                                        "rejected-event-study", f"{type(exc).__name__}: {exc}"))
            continue
        try:
            observed = investigate_motion(study)
        except (UnsupportedProblem, MotionIntegrationError) as exc:
            rows.append(EventSweepEntry(index, value, scenario, None,
                                        "integration-failed", f"{type(exc).__name__}: {exc}"))
            continue
        if observed.study.scenario is not scenario:
            raise RuntimeError("event service replaced a revised physical model")
        validate_event_run(observed)
        status = "complete" if observed.trajectory.termination == "duration" else "censored"
        rows.append(EventSweepEntry(index, value, scenario, observed,
                                    status, observed.trajectory.termination))
    if plan.sweep.baseline.revision_id != baseline_sha:
        raise RuntimeError("baseline physical model was mutated")
    return EventSweepRun(plan, baseline, tuple(rows))
