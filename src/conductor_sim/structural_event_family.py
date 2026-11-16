"""M29b: same M24a typed events across genuinely different fixed-source inventories.

This is a bounded orchestration of M22a/M29a atomic source membership and
M24a event integration. No new Coulomb law, force, numerical integrator,
event root locator, plane qualification, guard or browser-time physics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json

from .event_sweep import _root_pair
from .motion_1d import MotionIntegrationError
from .motion_events import (
    EventPolicy, EventRun, EventStudy, PlaneEvent, VelocityReversal,
    investigate_motion, validate_event_run,
)
from .solver import UnsupportedProblem
from .structural_family import StructuralCase, StructuralFamilyPlan, _assembly
from .structural_scenarios import ScenarioReassembly
from .trajectory_sweep import _shared_rows


@dataclass(frozen=True, slots=True)
class StructuralEventFamilyPlan:
    """One original apparatus, 1..5 atomic structural alternatives, SAME questions."""
    family: StructuralFamilyPlan
    questions: tuple[PlaneEvent | VelocityReversal, ...]
    policy: EventPolicy

    def __post_init__(self):
        if type(self.family) is not StructuralFamilyPlan:
            raise TypeError("event family requires an authored M29a StructuralFamilyPlan")
        if type(self.questions) is not tuple:
            raise TypeError("event questions must be an immutable tuple")
        if type(self.policy) is not EventPolicy:
            raise TypeError("event family requires a built-in M24a EventPolicy")
        # Existing M24a handles 1..16 typed questions, distinct event IDs,
        # finite numerical clocks, reversal-axis qualification, and the
        # terminal stop surface's actual ORIGINAL physical initial side.
        # Reject baseline-inadmissible questions BEFORE numerical integration.
        EventStudy(self.family.baseline, self.questions, self.policy)


@dataclass(frozen=True, slots=True)
class StructuralEventEntry:
    index: int
    case: StructuralCase
    assembly: ScenarioReassembly | None
    event_run: EventRun | None
    status: str
    reason: str

    def as_dict(self, baseline: EventRun, questions: tuple):
        assembly=self.assembly
        observed=self.event_run
        return dict(
            index=self.index,
            case=self.case.as_dict(),
            status=self.status, reason=self.reason,
            source_inventory=(None if assembly is None else [
                dict(id=sid, label=label, kind=type(source).__name__)
                for (sid, label), source in zip(
                    assembly.scenario.source_annotations,
                    assembly.scenario.spec.field.sources)]),
            structural_assembly=None if assembly is None else assembly.as_dict(),
            event_run=None if observed is None else observed.as_dict(),
            event_outcomes=None if observed is None else observed.as_dict()["outcomes"],
            exact_time_trajectory_comparisons=[] if observed is None else
            _shared_rows(baseline.trajectory, observed.trajectory),
            event_time_comparisons=[] if observed is None else [
                _root_pair(baseline, observed, question.event_id)
                for question in questions],
        )


@dataclass(frozen=True, slots=True)
class StructuralEventFamilyRun:
    plan: StructuralEventFamilyPlan
    baseline_run: EventRun
    entries: tuple[StructuralEventEntry, ...]

    def __post_init__(self):
        if (type(self.plan) is not StructuralEventFamilyPlan or
                type(self.baseline_run) is not EventRun or
                type(self.entries) is not tuple):
            raise TypeError("requires a native M29b event-family plan and M24a outcomes")
        baseline=self.plan.family.baseline
        if (self.baseline_run.study.scenario is not baseline or
                self.baseline_run.trajectory.spec is not baseline.spec or
                self.baseline_run.study.questions != self.plan.questions or
                self.baseline_run.study.policy != self.plan.policy):
            raise ValueError("baseline event observation does not match the authored study")
        if len(self.entries) != len(self.plan.family.cases):
            raise ValueError("event family is missing structural alternatives")
        for index, entry in enumerate(self.entries, 1):
            if type(entry) is not StructuralEventEntry or entry.index != index:
                raise ValueError("event family entry index or type mismatch")
            if entry.case is not self.plan.family.cases[index-1]:
                raise ValueError("event family lost authored structural case identity")
            if entry.assembly is None:
                if entry.event_run is not None or entry.status != "rejected-structural-model":
                    raise ValueError("unqualified structural variant has a fabricated event result")
                continue
            if (type(entry.assembly) is not ScenarioReassembly or
                    entry.assembly.parent is not baseline):
                raise ValueError("a structural sibling was not authored from the original parent")
            if entry.assembly.scenario.origin != "teacher-hypothetical":
                raise ValueError("structural child cannot silently claim source-given provenance")
            if entry.event_run is None:
                if entry.status not in ("rejected-event-study", "integration-failed"):
                    raise ValueError("accepted structural model has an unexplained absent event result")
                continue
            if (entry.event_run.study.scenario is not entry.assembly.scenario or
                    entry.event_run.trajectory.spec is not entry.assembly.scenario.spec or
                    entry.event_run.study.questions != self.plan.questions or
                    entry.event_run.study.policy != self.plan.policy):
                raise ValueError("structural event result was bound to another model or study")
            if entry.status != (
                    "complete" if entry.event_run.trajectory.termination == "duration"
                    else "censored"):
                raise ValueError("event family status misstates native observation termination")

    def as_dict(self):
        """Detached, finite original structural and event evidence, never trusted import."""
        baseline=self.plan.family.baseline
        result=dict(
            schema="conductor_sim.structural_event_family.v1",
            mode="Independent M22a whole-model reassembly and native M24a event solving",
            plan_id=self.plan.family.plan_id,
            provenance=("Source-given status is an author assertion, not scanned-PDF "
                        "authentication; every structural edit and detector is a teacher "
                        "hypothesis, not a time-dependent physical change."),
            baseline_revision_id=baseline.revision_id,
            baseline_scenario=baseline.as_dict(),
            baseline_source_inventory=[
                dict(id=sid,label=label,kind=type(source).__name__)
                for (sid,label),source in zip(
                    baseline.source_annotations, baseline.spec.field.sources)],
            questions=[dict(kind=type(question).__name__, **asdict(question))
                       for question in self.plan.questions],
            numerical_policy=asdict(self.plan.policy),
            baseline_event_run=self.baseline_run.as_dict(),
            baseline_outcomes=self.baseline_run.as_dict()["outcomes"],
            baseline_termination=self.baseline_run.trajectory.termination,
            entries=[entry.as_dict(self.baseline_run, self.plan.questions)
                     for entry in self.entries],
            alignment="identical actually saved SI physical timestamps only",
            unique_root_pairing=("Same event ID and signed direction, exactly one "
                                 "resolved numerical root per DURATION-COMPLETE study; "
                                 "all censored, multiple and unresolved studies unpaired"),
            time_interpolation=False,
            postguard_extrapolation=False,
            qualification=("Root time brackets and their difference intervals refer to "
                           "numerically integrated trajectories, NOT rigorous error "
                           "bounds for exact ODE roots. A source guard or terminal "
                           "event ends observation, not a physical collision; different "
                           "source inventories' absolute potential energies are NOT work."),
        )
        return json.loads(json.dumps(result, allow_nan=False))


def run_structural_event_family(
        plan: StructuralEventFamilyPlan) -> StructuralEventFamilyRun:
    """Independently assemble, qualify and locate native events for every sibling.

    Expected structural refusal, event-study-only refusal and known numerical
    failures remain separate. Coding errors propagate, never become a fake
    physics refusal or a completed trajectory. The source baseline stays immutable.
    """
    if type(plan) is not StructuralEventFamilyPlan:
        raise TypeError("require a StructuralEventFamilyPlan")
    base=plan.family.baseline
    base_sha=base.revision_id
    before=base.as_dict()
    bstudy=EventStudy(base,plan.questions,plan.policy)
    br=investigate_motion(bstudy)
    if br.study.scenario is not base:
        raise RuntimeError("event solver changed the baseline physical model")
    validate_event_run(br)
    entries=[]
    for index,case in enumerate(plan.family.cases,1):
        try:
            assembly=_assembly(plan.family,case)
        except (ValueError,TypeError,UnsupportedProblem) as exc:
            entries.append(StructuralEventEntry(
                index,case,None,None,"rejected-structural-model",
                f"{type(exc).__name__}: {exc}"))
            continue
        if (assembly.parent is not base or
                assembly.as_dict()["parent_revision_id"] != base_sha):
            raise RuntimeError("structural event case lost its original physical parent")
        try:
            study=EventStudy(assembly.scenario,plan.questions,plan.policy)
        except (ValueError,TypeError,UnsupportedProblem) as exc:
            entries.append(StructuralEventEntry(
                index,case,assembly,None,"rejected-event-study",
                f"{type(exc).__name__}: {exc}"))
            continue
        try:
            observed=investigate_motion(study)
        except (UnsupportedProblem,MotionIntegrationError) as exc:
            entries.append(StructuralEventEntry(
                index,case,assembly,None,"integration-failed",
                f"{type(exc).__name__}: {exc}"))
            continue
        if observed.study.scenario is not assembly.scenario:
            raise RuntimeError("event service changed a structurally assembled physical model")
        validate_event_run(observed)
        status=("complete" if observed.trajectory.termination=="duration"
                else "censored")
        entries.append(StructuralEventEntry(
            index,case,assembly,observed,status,observed.trajectory.termination))
    if base.revision_id!=base_sha or base.as_dict()!=before:
        raise RuntimeError("original physical model was mutated during event family")
    return StructuralEventFamilyRun(plan,br,tuple(entries))
