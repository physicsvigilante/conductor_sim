"""M30b: typed M24a events over real two-stage structural→property models.

The accepted M30a physical family is independently requalified and integrated
as before. Every requested observable is a fresh original M24a EventStudy on
the actual model; every numerical event pairing is the unchanged M28b rule.
No event interpolation, new ODE, physical law, guard or source response.
"""
from __future__ import annotations

from dataclasses import dataclass
import json

from .event_sweep import _root_pair
from .motion_1d import MotionIntegrationError
from .motion_events import (
    EventPolicy, EventRun, EventStudy, PlaneEvent, VelocityReversal,
    investigate_motion, validate_event_run,
)
from .scenarios import PrescribedScenario
from .solver import UnsupportedProblem
from .structural_parameter_family import (
    BranchSweepEntry, StructuralParameterPlan, StructuralParameterRun,
    run_structural_parameter_family,
)
from .trajectory_sweep import SweepEntry, _shared_rows


def _unpaired(event_id: str, reason: str):
    return dict(event_id=event_id, status="unpaired", reason=reason)


def _observe(scenario, questions, policy):
    """Separate model-accepted, study-refused and numerical-failed stages."""
    try:
        study = EventStudy(scenario, questions, policy)
    except (UnsupportedProblem, TypeError, ValueError) as exc:
        return None, "rejected-event-study", f"{type(exc).__name__}: {exc}"
    try:
        native = investigate_motion(study)
    except (UnsupportedProblem, MotionIntegrationError) as exc:
        return None, "integration-failed", f"{type(exc).__name__}: {exc}"
    if native.study.scenario is not scenario or native.trajectory.spec is not scenario.spec:
        raise RuntimeError("native event solver returned an incorrectly bound physical model")
    validate_event_run(native)
    return native, (
        "complete" if native.trajectory.termination == "duration" else "censored"
    ), native.trajectory.termination


@dataclass(frozen=True, slots=True)
class NestedEventPlan:
    """One immutable M30a family, the SAME genuine M24a questions everywhere."""
    experiment: StructuralParameterPlan
    questions: tuple[PlaneEvent | VelocityReversal, ...]
    policy: EventPolicy

    def __post_init__(self):
        if type(self.experiment) is not StructuralParameterPlan:
            raise TypeError("nested events require an actual M30a structural-parameter plan")
        if type(self.questions) is not tuple:
            raise TypeError("nested event questions must be an immutable tuple")
        if type(self.policy) is not EventPolicy:
            raise TypeError("nested events require a native M24a EventPolicy")
        # Validate the original question inventory, event IDs, clock and
        # initial ideal-stop boundary BEFORE any M30a physical execution.
        EventStudy(self.experiment.family.baseline, self.questions, self.policy)


@dataclass(frozen=True, slots=True)
class NestedVariationEvent:
    index: int
    physical_entry: SweepEntry
    event_run: EventRun | None
    status: str
    reason: str

    def as_dict(self, parent: EventRun | None, questions: tuple):
        native = self.event_run
        source = self.physical_entry.scenario
        if native is not None and source is None:
            raise ValueError("event outcome without an admitted property grandchild")
        comparisons = [
            (_root_pair(parent, native, q.event_id)
             if parent is not None and native is not None
             else _unpaired(q.event_id, "parent or property child has no native event outcome"))
            for q in questions
        ]
        return dict(
            index=self.index,
            requested_value=self.physical_entry.requested_value,
            physical_status=self.physical_entry.status,
            event_status=self.status,
            reason=self.reason,
            model_revision_id=None if source is None else source.revision_id,
            source_inventory=None if source is None else [
                dict(id=i, label=label, kind=type(src).__name__)
                for (i, label), src in zip(
                    source.source_annotations, source.spec.field.sources)],
            event_run=None if native is None else native.as_dict(),
            event_outcomes=None if native is None else native.as_dict()["outcomes"],
            parent_event_comparisons=comparisons,
            exact_time_event_trajectory_differences=(
                [] if parent is None or native is None else
                _shared_rows(parent.trajectory, native.trajectory)),
        )


@dataclass(frozen=True, slots=True)
class NestedBranchEvent:
    index: int
    physical_entry: BranchSweepEntry
    event_run: EventRun | None
    status: str
    reason: str
    variations: tuple[NestedVariationEvent, ...]

    def as_dict(self, original: EventRun, questions: tuple):
        model = self.physical_entry.parent_scenario
        if self.physical_entry.request.branch_id == "baseline":
            comparisons = [
                dict(event_id=q.event_id, status="reference-baseline",
                     reason="original physical world is not a structural change")
                for q in questions
            ]
        else:
            comparisons = [
                (_root_pair(original, self.event_run, q.event_id)
                 if self.event_run is not None
                 else _unpaired(q.event_id, "structural parent has no native event outcome"))
                for q in questions
            ]
        return dict(
            index=self.index,
            branch_id=self.physical_entry.request.branch_id,
            physical_branch_status=self.physical_entry.status,
            event_status=self.status,
            reason=self.reason,
            model_revision_id=None if model is None else model.revision_id,
            event_run=None if self.event_run is None else self.event_run.as_dict(),
            event_outcomes=None if self.event_run is None else self.event_run.as_dict()["outcomes"],
            original_to_structural_event_comparisons=comparisons,
            exact_time_original_to_structural_event_differences=(
                [] if self.event_run is None or
                self.physical_entry.request.branch_id == "baseline"
                else _shared_rows(original.trajectory, self.event_run.trajectory)),
            variations=[v.as_dict(self.event_run, questions) for v in self.variations],
        )


@dataclass(frozen=True, slots=True)
class NestedEventRun:
    plan: NestedEventPlan
    physical_family: StructuralParameterRun
    original_event_run: EventRun
    branches: tuple[NestedBranchEvent, ...]

    def __post_init__(self):
        if (type(self.plan) is not NestedEventPlan or
                type(self.physical_family) is not StructuralParameterRun or
                self.physical_family.plan is not self.plan.experiment or
                type(self.original_event_run) is not EventRun or
                self.original_event_run.study.scenario is not self.plan.experiment.family.baseline or
                self.original_event_run.study.questions != self.plan.questions or
                self.original_event_run.study.policy != self.plan.policy or
                type(self.branches) is not tuple or
                len(self.branches) != len(self.physical_family.entries)):
            raise ValueError("nested event result does not match its original typed physical family")
        for i, branch in enumerate(self.branches, 1):
            if (type(branch) is not NestedBranchEvent or branch.index != i or
                    branch.physical_entry is not self.physical_family.entries[i-1] or
                    type(branch.variations) is not tuple):
                raise ValueError("nested structural branch identity or ordering mismatch")
            selected = branch.physical_entry
            parent = selected.parent_scenario
            if parent is None:
                if (branch.event_run is not None or
                        branch.status != "skipped-structural-model"):
                    raise ValueError("refused structural model has an invented event observation")
            elif branch.event_run is not None:
                if (branch.event_run.study.scenario is not parent or
                        branch.event_run.study.questions != self.plan.questions or
                        branch.event_run.study.policy != self.plan.policy):
                    raise ValueError("structural event observation bound to wrong physical parent")
            elif branch.status not in (
                    "unavailable-branch-trajectory", "rejected-event-study", "integration-failed"):
                raise ValueError("accepted structural world has an unexplained missing event run")
            if selected.request.branch_id == "baseline" and (
                    branch.event_run is not self.original_event_run):
                raise ValueError("original baseline event must be the single source of truth")
            original_variations = (
                () if selected.sweep_run is None else selected.sweep_run.entries)
            if len(branch.variations) != len(original_variations):
                raise ValueError("missing property-grandchild native event observations")
            for j, (variation, physical) in enumerate(
                    zip(branch.variations, original_variations), 1):
                if (type(variation) is not NestedVariationEvent or
                        variation.index != j or variation.physical_entry is not physical):
                    raise ValueError("grandchild event or physical variation identity mismatch")
                source = physical.scenario
                if source is None:
                    if variation.event_run is not None or variation.status != physical.status:
                        raise ValueError("unqualified physical grandchild has invented native event")
                elif variation.event_run is not None:
                    if (variation.event_run.study.scenario is not source or
                            variation.event_run.study.questions != self.plan.questions or
                            variation.event_run.study.policy != self.plan.policy):
                        raise ValueError("property event history bound to another physical model")
                    if source.parent_revision_id != parent.revision_id:
                        raise ValueError("property event lost its immediate structural source parent")
                elif variation.status not in ("rejected-event-study", "integration-failed"):
                    raise ValueError("admitted physical grandchild has unexplained absent event study")

    def as_dict(self):
        """Finite, detached, source-pinned observational evidence; never trusted import."""
        original = self.plan.experiment.family.baseline
        result = dict(
            schema="conductor_sim.nested_event_family.v1",
            mode=("M30a independently assembled structural/parameter families "
                  "with original M24a event integrations and M28b root pairing"),
            provenance=("All edits and detectors are separate authored teacher hypotheticals; "
                        "source-given origin is an author assertion, not scanned-PDF authentication."),
            original_revision_id=original.revision_id,
            plan_id=self.plan.experiment.plan_id,
            original_model=original.as_dict(),
            physical_model_family=self.physical_family.as_dict(),
            questions=[
                dict(event_id=q.event_id, kind=type(q).__name__) for q in self.plan.questions
            ],
            original_event_run=self.original_event_run.as_dict(),
            original_outcomes=self.original_event_run.as_dict()["outcomes"],
            branches=[
                b.as_dict(self.original_event_run, self.plan.questions) for b in self.branches
            ],
            event_shift_rule=("Only unique resolved same-direction roots in TWO "
                              "duration-complete native M24a histories; edges pair "
                              "original to structural and structural to property grandchild."),
            time_alignment="literally equal ORIGINAL saved M24a physical times only",
            time_interpolation=False,
            postguard_extrapolation=False,
            qualification=("Original numerical event brackets are not rigorous bounds "
                           "for the exact ODE root; no unobserved postguard motion "
                           "or arbitrary first crossing. No cross-inventory potential "
                           "energy difference is interpreted as work or induction."),
        )
        return json.loads(json.dumps(result, allow_nan=False))


def run_nested_event_family(plan: NestedEventPlan) -> NestedEventRun:
    """Run a finite original→structural→property family of genuine typed events."""
    if type(plan) is not NestedEventPlan:
        raise TypeError("require native NestedEventPlan")
    original = plan.experiment.family.baseline
    before = original.as_dict()
    original_revision = original.revision_id
    physical = run_structural_parameter_family(plan.experiment)
    original_study = EventStudy(original, plan.questions, plan.policy)
    original_run = investigate_motion(original_study)
    if original_run.study.scenario is not original:
        raise RuntimeError("M24a baseline event returned different physical model")
    validate_event_run(original_run)
    entries = []
    for index, branch in enumerate(physical.entries, 1):
        parent = branch.parent_scenario
        if parent is None:
            observed, status, reason = (
                None, "skipped-structural-model", branch.reason)
        elif branch.status == "unavailable-branch-trajectory":
            observed, status, reason = (
                None, "unavailable-branch-trajectory", branch.reason)
        elif branch.request.branch_id == "baseline":
            observed = original_run
            status = ("complete" if original_run.trajectory.termination == "duration"
                      else "censored")
            reason = "Original M24a native event observation reused as baseline"
        else:
            observed, status, reason = _observe(parent, plan.questions, plan.policy)
        variations = []
        if branch.sweep_run is not None:
            for j, value in enumerate(branch.sweep_run.entries, 1):
                child = value.scenario
                if child is None:
                    native, state, why = None, value.status, value.explanation
                else:
                    native, state, why = _observe(child, plan.questions, plan.policy)
                variations.append(NestedVariationEvent(j, value, native, state, why))
        entries.append(NestedBranchEvent(
            index, branch, observed, status, reason, tuple(variations)))
    if original.revision_id != original_revision or original.as_dict() != before:
        raise RuntimeError("nested event authoring altered original immutable source world")
    return NestedEventRun(plan, physical, original_run, tuple(entries))
