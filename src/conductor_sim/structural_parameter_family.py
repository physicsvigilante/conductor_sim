"""M30a: compose true source-inventory reassembly with physical parameter sweeps.

One frozen structural parent produces independent, qualified M22a source
assemblies; within each selected assembly the unmodified M28a runner owns
all M20b property revisions and native integrator evaluations. There is NO
new field law, RK, source qualifier, event search, or source-change-in-time.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re

from .motion_1d import MotionIntegrationError
from .scenarios import PrescribedScenario
from .solver import UnsupportedProblem
from .structural_family import (
    StructuralFamilyEntry, StructuralFamilyPlan, StructuralFamilyRun,
    run_structural_family,
)
from .trajectory_sweep import (
    SweepAxis, TrajectorySweepPlan, TrajectorySweepRun,
    run_trajectory_sweep,
)

_PLAN = re.compile(r"[A-Za-z][A-Za-z0-9:_-]{0,51}\Z")
_MAX_BRANCHES = 3
_MAX_TOTAL_VALUES = 12


@dataclass(frozen=True, slots=True)
class BranchSweep:
    """Exactly one M28a physical sweep on one authored structural branch.

    "baseline" is the unchanged original. All other branch IDs refer to an
    explicitly declared StructuralCase.case_id, not source list positions.
    """
    branch_id: str
    axis: SweepAxis

    def __post_init__(self):
        if type(self.branch_id) is not str or (
                self.branch_id != "baseline" and
                re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,19}", self.branch_id) is None):
            raise ValueError("branch ID must be 'baseline' or an M29a structural case ID")
        if type(self.axis) is not SweepAxis:
            raise TypeError("each branch must provide an existing native M28a SweepAxis")

    def as_dict(self):
        return dict(branch_id=self.branch_id,
                    axis=dict(component_id=self.axis.component_id,
                              property_name=self.axis.property_name,
                              values=self.axis.values))


@dataclass(frozen=True, slots=True)
class StructuralParameterPlan:
    """Finite directed source-structure → physical-parameter exploration."""

    plan_id: str
    family: StructuralFamilyPlan
    branches: tuple[BranchSweep, ...]

    def __post_init__(self):
        if type(self.plan_id) is not str or _PLAN.fullmatch(self.plan_id) is None:
            raise ValueError("M30a plan ID must be stable, ASCII and at most 52 characters")
        if type(self.family) is not StructuralFamilyPlan:
            raise TypeError("M30a requires one existing M29a StructuralFamilyPlan")
        if type(self.branches) is not tuple or not 1 <= len(self.branches) <= _MAX_BRANCHES:
            raise ValueError("M30a requires 1..3 immutable branch-sweep requests")
        if any(type(b) is not BranchSweep for b in self.branches):
            raise TypeError("branch sweeps must have exact BranchSweep type")
        keys = [b.branch_id for b in self.branches]
        if len(set(keys)) != len(keys):
            raise ValueError("a physical structural branch can have only one sweep")
        # "baseline" is a reserved selector, not an alternative case ID.
        # Otherwise a teacher might silently sweep the untouched original
        # instead of a separately authored structural branch with that name.
        if any(c.case_id == "baseline" for c in self.family.cases):
            raise ValueError("M30a reserves baseline as the original-world selector")
        existing = {"baseline", *(c.case_id for c in self.family.cases)}
        if any(k not in existing for k in keys):
            raise ValueError("sweep branch does not identify an authored structural case")
        if sum(len(b.axis.values) for b in self.branches) > _MAX_TOTAL_VALUES:
            raise ValueError("M30a permits at most 12 parameter variations in total")


def _inventory(model: PrescribedScenario | None):
    if model is None:
        return None
    return [
        dict(id=identity, label=label, kind=type(source).__name__)
        for (identity, label), source in zip(
            model.source_annotations, model.spec.field.sources)
    ]


@dataclass(frozen=True, slots=True)
class BranchSweepEntry:
    index: int
    request: BranchSweep
    structural_entry: StructuralFamilyEntry | None
    parent_scenario: PrescribedScenario | None
    sweep_run: TrajectorySweepRun | None
    status: str
    reason: str

    def as_dict(self):
        parent = self.parent_scenario
        assembly = (None if self.structural_entry is None or
                    self.structural_entry.assembly is None else
                    self.structural_entry.assembly.as_dict())
        return dict(
            index=self.index,
            request=self.request.as_dict(),
            status=self.status,
            reason=self.reason,
            source_inventory=_inventory(parent),
            structural_assembly=assembly,
            branch_scenario=None if parent is None else parent.as_dict(),
            branch_revision_id=None if parent is None else parent.revision_id,
            trajectory_sweep=None if self.sweep_run is None else self.sweep_run.as_dict(),
            parameter_revision_parents=[] if self.sweep_run is None else [
                None if row.scenario is None else row.scenario.parent_revision_id
                for row in self.sweep_run.entries
            ],
        )


@dataclass(frozen=True, slots=True)
class StructuralParameterRun:
    plan: StructuralParameterPlan
    structural_run: StructuralFamilyRun
    entries: tuple[BranchSweepEntry, ...]

    def __post_init__(self):
        if (type(self.plan) is not StructuralParameterPlan or
                type(self.structural_run) is not StructuralFamilyRun or
                self.structural_run.plan is not self.plan.family or
                type(self.entries) is not tuple or
                len(self.entries) != len(self.plan.branches)):
            raise ValueError("M30a result must be bound to one original native structural family")
        structural_by_id = {row.case.case_id: row for row in self.structural_run.entries}
        for i, entry in enumerate(self.entries, 1):
            if (type(entry) is not BranchSweepEntry or entry.index != i or
                    entry.request is not self.plan.branches[i-1]):
                raise ValueError("M30a branch result ordering or identity changed")
            if entry.request.branch_id == "baseline":
                if (entry.structural_entry is not None or
                        entry.parent_scenario is not self.plan.family.baseline):
                    raise ValueError("baseline branch not bound to the original world")
            else:
                actual = structural_by_id[entry.request.branch_id]
                if entry.structural_entry is not actual:
                    raise ValueError("branch result lost its actual structural case")
                expected = None if actual.assembly is None else actual.assembly.scenario
                if entry.parent_scenario is not expected:
                    raise ValueError("branch source inventory does not match reassembly")
            if entry.parent_scenario is None:
                if entry.status != "skipped-structural-model" or entry.sweep_run is not None:
                    raise ValueError("refused structural branch has a fabricated physical model")
            elif entry.sweep_run is not None:
                if (entry.status != "evaluated" or
                        entry.sweep_run.plan.baseline is not entry.parent_scenario or
                        entry.sweep_run.plan.axis is not entry.request.axis):
                    raise ValueError("native physical sweep not bound to selected branch")
                for child in entry.sweep_run.entries:
                    if child.scenario is not None:
                        if child.scenario.parent_revision_id != entry.parent_scenario.revision_id:
                            raise ValueError("parameter variation lost its immediate structural parent")
                        if child.scenario.source_annotations != entry.parent_scenario.source_annotations:
                            raise ValueError("parameter revision changed source ID/name membership")
            elif entry.status not in ("rejected-sweep-axis", "unavailable-branch-trajectory"):
                raise ValueError("accepted physical branch lacks a classified sweep result")

    def as_dict(self):
        """Detached complete two-layer lineage; not a physical model importer."""
        result = dict(
            schema="conductor_sim.structural_parameter_family.v1",
            mode=("Independently reassemble physical source inventories, then "
                  "independently revise one physically admissible property per branch"),
            plan_id=self.plan.plan_id,
            provenance=("Original source-given facts are author assertions, not "
                        "authenticated scanned-PDF facts. All structural and "
                        "parameter changes are distinct teacher hypotheticals."),
            original_revision_id=self.plan.family.baseline.revision_id,
            original_scenario=self.plan.family.baseline.as_dict(),
            original_inventory=_inventory(self.plan.family.baseline),
            structural_family=self.structural_run.as_dict(),
            entries=[entry.as_dict() for entry in self.entries],
            exact_time_alignment="M29a and M28a literal saved physical times only, within their own parents",
            time_interpolation=False,
            postguard_extrapolation=False,
            qualification=("No automatic cross-lattice work/sensitivity/event comparison. "
                           "Every different inventory and every parameter variation "
                           "is a separate immutable physical hypothetical; a source "
                           "guard censors observation rather than proving later absence. "
                           "No induced conductor response or time-moving sources."),
        )
        return json.loads(json.dumps(result, allow_nan=False))


def run_structural_parameter_family(plan: StructuralParameterPlan) -> StructuralParameterRun:
    """Reuse M29a and M28a as-is; no new numerical or physical interpretation."""
    if type(plan) is not StructuralParameterPlan:
        raise TypeError("require a native StructuralParameterPlan")
    baseline = plan.family.baseline
    before = baseline.as_dict()
    original_sha = baseline.revision_id
    structural = run_structural_family(plan.family)
    cases = {entry.case.case_id: entry for entry in structural.entries}
    entries = []
    for index, branch in enumerate(plan.branches, 1):
        case = None if branch.branch_id == "baseline" else cases[branch.branch_id]
        if case is not None and case.assembly is None:
            entries.append(BranchSweepEntry(
                index, branch, case, None, None,
                "skipped-structural-model", case.reason))
            continue

        parent = baseline if case is None else case.assembly.scenario
        if case is not None and (
                case.assembly.parent is not baseline or
                case.assembly.as_dict()["parent_revision_id"] != original_sha):
            raise RuntimeError("M30a structural branch has an incorrect original source parent")
        if case is not None and case.trajectory is None:
            # Structural admission succeeded but its original M29a numerical
            # integration did not. Do not turn this into a physical refusal.
            entries.append(BranchSweepEntry(
                index, branch, case, parent, None,
                "unavailable-branch-trajectory",
                "Accepted structural model had no completed native parent integration: " + case.reason))
            continue

        try:
            sweep = TrajectorySweepPlan(
                f"{plan.plan_id}:b{index:02d}", parent, branch.axis)
        except (UnsupportedProblem, ValueError, TypeError) as exc:
            entries.append(BranchSweepEntry(
                index, branch, case, parent, None,
                "rejected-sweep-axis", f"{type(exc).__name__}: {exc}"))
            continue
        try:
            run = run_trajectory_sweep(sweep)
        except (UnsupportedProblem, MotionIntegrationError) as exc:
            entries.append(BranchSweepEntry(
                index, branch, case, parent, None,
                "unavailable-branch-trajectory", f"{type(exc).__name__}: {exc}"))
            continue

        structural_native = (structural.baseline_trajectory if case is None
                             else case.trajectory)
        if (run.baseline_trajectory.spec is not parent.spec or
                run.baseline_trajectory.as_dict() != structural_native.as_dict()):
            raise RuntimeError("two executions of the same physical parent disagreed")
        if run.plan.baseline.revision_id != parent.revision_id:
            raise RuntimeError("parameter runner modified original structural source identity")
        entries.append(BranchSweepEntry(
            index, branch, case, parent, run, "evaluated",
            "Independently requalified M28a property sweep from accepted physical branch"))

    if baseline.revision_id != original_sha or baseline.as_dict() != before:
        raise RuntimeError("M30a altered its immutable physical source baseline")
    return StructuralParameterRun(plan, structural, tuple(entries))
