"""M31b: genuine M24a events on M31a atomic coupled physical source worlds.

One native M31a physical family, independently integrated M24a event studies,
and unchanged M28b duration-complete unique-root comparisons. Source worlds
are immutable hypotheses, not phases of one moving-source trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from hmac import compare_digest
import json

from .coupled_physical_family import (
    CoupledPhysicalEntry, CoupledPhysicalPlan, CoupledPhysicalRun,
    _inventory, run_coupled_physical_family,
)
from .event_sweep import _root_pair
from .motion_1d import MotionIntegrationError
from .motion_events import (
    EventPolicy, EventRun, EventStudy, PlaneEvent, VelocityReversal,
    investigate_motion, validate_event_run,
)
from .prescribed_motion import TestChargeMotionResult
from .scenarios import PrescribedScenario
from .solver import UnsupportedProblem
from .trajectory_sweep import _shared_rows


_TERMINATIONS = {
    "duration": "complete",
    "source-guard": "censored-source-guard",
    "terminal-event": "terminal-event",
    "event-unresolved": "event-unresolved",
}


def _status(native: EventRun) -> str:
    try:
        return _TERMINATIONS[native.trajectory.termination]
    except KeyError as exc:
        raise ValueError("unknown native coupled event termination") from exc


def _validate_native(native, scenario, questions, policy):
    """Validate the exact native observation and its actual scenario binding."""
    if (type(native) is not EventRun or type(native.study) is not EventStudy or
            type(native.trajectory) is not TestChargeMotionResult or
            native.study.scenario is not scenario or
            native.trajectory.spec is not scenario.spec or
            native.study.questions is not questions or
            native.study.policy is not policy):
        raise ValueError("coupled event is not a native observation of its actual physical model")
    validate_event_run(native)
    return _status(native)


def _observe(scenario, questions, policy):
    try:
        study = EventStudy(scenario, questions, policy)
    except (UnsupportedProblem, TypeError, ValueError) as exc:
        return None, "rejected-event-study", f"{type(exc).__name__}: {exc}"
    try:
        result = investigate_motion(study)
    except (UnsupportedProblem, MotionIntegrationError) as exc:
        return None, "integration-failed", f"{type(exc).__name__}: {exc}"
    state = _validate_native(result, scenario, questions, policy)
    return result, state, result.trajectory.termination


def _unpaired(event_id, reason):
    return dict(event_id=event_id, status="unpaired", reason=reason)


# Only the native runner issues authentic in-process event observations.
# This is not cryptographic security against arbitrary Python private access.
_RUNNER_ISSUER = object()


@dataclass(frozen=True, slots=True)
class CoupledEventPlan:
    """Same immutable M24a questions on each actual M31a coupled source world."""
    physical_plan: CoupledPhysicalPlan
    questions: tuple[PlaneEvent | VelocityReversal, ...]
    policy: EventPolicy

    def __post_init__(self):
        if type(self.physical_plan) is not CoupledPhysicalPlan:
            raise TypeError("coupled events require one native M31a physical plan")
        if type(self.questions) is not tuple or type(self.policy) is not EventPolicy:
            raise TypeError("coupled events require a tuple and native M24a numerical policy")
        EventStudy(self.physical_plan.family.baseline, self.questions, self.policy)


@dataclass(frozen=True, slots=True)
class CoupledParentEvent:
    branch_id: str
    parent_scenario: PrescribedScenario | None
    event_run: EventRun | None
    status: str
    reason: str


@dataclass(frozen=True, slots=True)
class CoupledChildEvent:
    index: int
    physical_entry: CoupledPhysicalEntry
    event_run: EventRun | None
    status: str
    reason: str


def _observation_receipt(original, parents, children):
    """Snapshot genuine native event states, root clocks and exception stages.

    Hashing only detects post-observation alterations; the receipt must be
    minted in the trusted runner, never from a supplied typed result.
    """
    record = dict(
        original=original.as_dict(),
        parents=[dict(
            branch_id=p.branch_id, status=p.status, reason=p.reason,
            model_revision_id=(None if p.parent_scenario is None
                               else p.parent_scenario.revision_id),
            native_event=(None if p.event_run is None else p.event_run.as_dict()),
        ) for p in parents],
        children=[dict(
            index=c.index, case_id=c.physical_entry.hypothesis.case_id,
            physical_status=c.physical_entry.status,
            model_revision_id=(None if c.physical_entry.scenario is None
                               else c.physical_entry.scenario.revision_id),
            status=c.status, reason=c.reason,
            native_event=(None if c.event_run is None else c.event_run.as_dict()),
        ) for c in children],
    )
    payload = json.dumps(record, allow_nan=False, sort_keys=True,
                         separators=(",", ":"), ensure_ascii=True)
    return sha256(payload.encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class CoupledEventRun:
    plan: CoupledEventPlan
    physical_family: CoupledPhysicalRun
    original_event_run: EventRun
    parents: tuple[CoupledParentEvent, ...]
    children: tuple[CoupledChildEvent, ...]
    _observation_digest: str = field(default="", repr=False, compare=False)
    _issuer: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if (type(self.plan) is not CoupledEventPlan or
                type(self.physical_family) is not CoupledPhysicalRun or
                self.physical_family.plan is not self.plan.physical_plan or
                type(self.parents) is not tuple or type(self.children) is not tuple or
                len(self.children) != len(self.physical_family.entries)):
            raise ValueError("coupled event result is not bound to its original native family")
        # This also catches a mutation of frozen M31a entries' mutable comparison rows.
        self.physical_family.__post_init__()
        original = self.plan.physical_plan.family.baseline
        _validate_native(self.original_event_run, original,
                         self.plan.questions, self.plan.policy)
        required = tuple(dict.fromkeys(
            entry.hypothesis.structural_branch for entry in self.physical_family.entries))
        if (len(self.parents) != len(required) or
                tuple(p.branch_id for p in self.parents) != required):
            raise ValueError("coupled parent source worlds are missing, duplicated or reordered")
        parent_map = {}
        for branch_id, p in zip(required, self.parents):
            if type(p) is not CoupledParentEvent or type(p.status) is not str or type(p.reason) is not str:
                raise TypeError("coupled parent requires typed native observation status")
            actual = next(e for e in self.physical_family.entries
                          if e.hypothesis.structural_branch == branch_id)
            if p.parent_scenario is not actual.parent_scenario:
                raise ValueError("coupled event parent refers to another physical source world")
            if p.parent_scenario is None:
                if p.event_run is not None or p.status != "skipped-structural-model":
                    raise ValueError("refused structural world has fabricated parent event")
            elif branch_id == "baseline":
                if p.event_run is not self.original_event_run or p.status != _status(self.original_event_run):
                    raise ValueError("baseline event was not reused as the original native observation")
            elif actual.structural_entry.trajectory is None:
                if p.event_run is not None or p.status != "unavailable-branch-trajectory":
                    raise ValueError("unavailable structural trajectory has fabricated parent event")
            elif p.event_run is None:
                if p.status not in ("rejected-event-study", "integration-failed"):
                    raise ValueError("admitted structural parent has unexplained missing event study")
            elif p.status != _validate_native(
                    p.event_run, p.parent_scenario, self.plan.questions, self.plan.policy):
                raise ValueError("coupled parent event status mismatches native termination")
            parent_map[branch_id] = p
        for idx, child in enumerate(self.children, 1):
            physical = self.physical_family.entries[idx-1]
            if (type(child) is not CoupledChildEvent or child.index != idx or
                    child.physical_entry is not physical or
                    type(child.status) is not str or type(child.reason) is not str):
                raise ValueError("coupled event child lost its original physical case identity")
            parent = parent_map[physical.hypothesis.structural_branch]
            if physical.parent_scenario is not parent.parent_scenario:
                raise ValueError("coupled child event lost its immediate physical parent")
            if physical.scenario is None:
                if child.event_run is not None or child.status != physical.status:
                    raise ValueError("unqualified coupled child has fabricated native event evidence")
            else:
                if physical.scenario.parent_revision_id != parent.parent_scenario.revision_id:
                    raise ValueError("coupled child event lost its real source-model revision parent")
                if physical.trajectory is None:
                    if (physical.status != "integration-failed" or
                            child.event_run is not None or child.status != physical.status):
                        raise ValueError("failed M31a physical integration has fabricated M24a event")
                elif child.event_run is None:
                    if child.status not in ("rejected-event-study", "integration-failed"):
                        raise ValueError("admitted child has unexplained missing native event")
                elif child.status != _validate_native(
                        child.event_run, physical.scenario, self.plan.questions, self.plan.policy):
                    raise ValueError("coupled child event status mismatches native termination")
                if child.event_run is not None and child.event_run is parent.event_run:
                    raise ValueError("coupled revision reused its parent's event integration")
        if (self._issuer is not _RUNNER_ISSUER or
                type(self._observation_digest) is not str or
                not compare_digest(
                    self._observation_digest,
                    _observation_receipt(
                        self.original_event_run, self.parents, self.children))):
            raise ValueError("coupled event differs from runner-issued observation receipt")

    def as_dict(self):
        """Detached evidence; event shifts are derived anew from checked native runs."""
        self.__post_init__()
        question_ids = [q.event_id for q in self.plan.questions]
        parents = {}
        for p in self.parents:
            parents[p.branch_id] = p
        parent_records = []
        for p in self.parents:
            parent_records.append(dict(
                branch_id=p.branch_id, parent_revision_id=(
                    None if p.parent_scenario is None else p.parent_scenario.revision_id),
                accepted_parent_inventory=_inventory(p.parent_scenario),
                event_status=p.status, reason=p.reason,
                reuses_original_event=(p.branch_id == "baseline"),
                event_run=(None if p.event_run is None or p.branch_id == "baseline"
                           else p.event_run.as_dict()),
                event_outcomes=(None if p.event_run is None else p.event_run.as_dict()["outcomes"]),
            ))
        child_records = []
        for c in self.children:
            physical = c.physical_entry
            parent = parents[physical.hypothesis.structural_branch]
            observed = c.event_run
            base = parent.event_run
            comparisons = [
                (_root_pair(base, observed, event_id)
                 if base is not None and observed is not None else
                 _unpaired(event_id, "immediate parent or child lacks a native M24a event observation"))
                for event_id in question_ids
            ]
            child_records.append(dict(
                index=c.index,
                case_id=physical.hypothesis.case_id,
                structural_branch=physical.hypothesis.structural_branch,
                parent_revision_id=(None if parent.parent_scenario is None
                                    else parent.parent_scenario.revision_id),
                revised_revision_id=(None if physical.scenario is None
                                     else physical.scenario.revision_id),
                accepted_source_inventory=_inventory(physical.scenario),
                physical_status=physical.status, event_status=c.status, reason=c.reason,
                event_run=None if observed is None else observed.as_dict(),
                event_outcomes=None if observed is None else observed.as_dict()["outcomes"],
                parent_event_time_comparisons=comparisons,
                exact_time_event_trajectory_differences=(
                    [] if base is None or observed is None else
                    _shared_rows(base.trajectory, observed.trajectory)),
            ))
        record = dict(
            schema="conductor_sim.coupled_event_family.v1",
            provenance=("Teacher-authored atomic physical hypotheses on independently "
                        "qualified named source worlds, not source motion"),
            plan_id=self.plan.physical_plan.plan_id,
            original_revision_id=self.plan.physical_plan.family.baseline.revision_id,
            physical_model_family=self.physical_family.as_dict(),
            questions=[dict(event_id=q.event_id, kind=type(q).__name__)
                       for q in self.plan.questions],
            original_event_run=self.original_event_run.as_dict(),
            original_outcomes=self.original_event_run.as_dict()["outcomes"],
            parent_events=parent_records,
            coupled_events=child_records,
            pairing_rule="unique same-direction located root in two duration-complete M24a runs",
            time_alignment="actual identical saved M24a SI times only",
            time_interpolation=False, postguard_extrapolation=False,
            qualification=("Numerical event brackets do not bound global ODE error; "
                           "guard/terminal/unresolved roots have no complete-horizon pairing; "
                           "no cross-inventory electrostatic energy-as-work or induction."),
        )
        return json.loads(json.dumps(record, allow_nan=False))


def run_coupled_event_family(plan: CoupledEventPlan) -> CoupledEventRun:
    if type(plan) is not CoupledEventPlan:
        raise TypeError("require native CoupledEventPlan")
    original = plan.physical_plan.family.baseline
    before = original.as_dict()
    original_sha = original.revision_id
    physical = run_coupled_physical_family(plan.physical_plan)
    study = EventStudy(original, plan.questions, plan.policy)
    original_run = investigate_motion(study)
    _validate_native(original_run, original, plan.questions, plan.policy)
    parents_by_id = {}
    for e in physical.entries:
        branch_id = e.hypothesis.structural_branch
        if branch_id in parents_by_id:
            if e.parent_scenario is not parents_by_id[branch_id].parent_scenario:
                raise RuntimeError("one structural branch resolved to two different native parents")
            continue
        if e.parent_scenario is None:
            native, state, reason = None, "skipped-structural-model", e.reason
        elif branch_id == "baseline":
            native, state, reason = (
                original_run, _status(original_run), "Original M24a baseline observation reused")
        elif e.structural_entry.trajectory is None:
            native, state, reason = (
                None, "unavailable-branch-trajectory", e.structural_entry.reason)
        else:
            native, state, reason = _observe(e.parent_scenario, plan.questions, plan.policy)
        parents_by_id[branch_id] = CoupledParentEvent(
            branch_id, e.parent_scenario, native, state, reason)
    children = []
    for idx, entry in enumerate(physical.entries, 1):
        if entry.scenario is None or entry.trajectory is None:
            # An admitted but physically unsolved M31a child is NOT eligible
            # for another solver to fabricate a successful M24a observation.
            native, state, reason = None, entry.status, entry.reason
        else:
            native, state, reason = _observe(entry.scenario, plan.questions, plan.policy)
        children.append(CoupledChildEvent(idx, entry, native, state, reason))
    if original.revision_id != original_sha or original.as_dict() != before:
        raise RuntimeError("coupled event exploration changed its immutable original world")
    parents, children = tuple(parents_by_id.values()), tuple(children)
    return CoupledEventRun(
        plan, physical, original_run, parents, children,
        _observation_receipt(original_run, parents, children), _RUNNER_ISSUER)
