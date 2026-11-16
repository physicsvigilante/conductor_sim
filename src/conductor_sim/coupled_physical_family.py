"""M31a: atomic coupled physical revisions on original or reassembled source worlds.

Each hypothesis is one complete M20b revise_scenario call, not a sequence of
potentially invalid intermediate physical worlds. M22a/M29a owns genuine
source membership and parent lineage, M28a owns physical target allowlists,
and the original M19/M27 adaptive numerical motion law remains unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re

from .motion_1d import MotionIntegrationError
from .prescribed_motion import TestChargeMotionResult, integrate_test_charge
from .scenarios import PrescribedScenario, revise_scenario, _text
from .solver import UnsupportedProblem
from .structural_family import (
    StructuralFamilyEntry, StructuralFamilyPlan, StructuralFamilyRun,
    run_structural_family,
)
from .trajectory_sweep import _number, _shared_rows, _target, _value_shape


_PLAN_ID = re.compile(r"[A-Za-z][A-Za-z0-9:_-]{0,39}\Z")
_CASE_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,19}\Z")
_TARGET_ID = re.compile(r"[A-Za-z][A-Za-z0-9:_-]{0,63}\Z")
_PROPERTY = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")


@dataclass(frozen=True, slots=True)
class CoupledPropertyEdit:
    """One explicit whole scalar or whole 3-vector SI physical property."""
    component_id: str
    property_name: str
    value: float | tuple[float, float, float]

    def __post_init__(self):
        if type(self.component_id) is not str or _TARGET_ID.fullmatch(self.component_id) is None:
            raise ValueError("coupled edit needs a stable native physical component ID")
        if type(self.property_name) is not str or _PROPERTY.fullmatch(self.property_name) is None:
            raise ValueError("coupled edit needs one named physical property")
        if not (_number(self.value) or
                (type(self.value) is tuple and len(self.value) == 3 and
                 all(_number(v) for v in self.value))):
            raise TypeError("coupled edit must use a finite scalar or complete 3-vector SI value")

    def as_dict(self):
        return dict(component_id=self.component_id, property_name=self.property_name,
                    value=self.value)


@dataclass(frozen=True, slots=True)
class CoupledHypothesis:
    """All changed properties are applied atomically to one real source world."""
    case_id: str
    structural_branch: str
    note: str
    edits: tuple[CoupledPropertyEdit, ...]

    def __post_init__(self):
        if type(self.case_id) is not str or _CASE_ID.fullmatch(self.case_id) is None:
            raise ValueError("hypothesis case requires a stable bounded ID")
        if (type(self.structural_branch) is not str or
                (self.structural_branch != "baseline" and
                 _CASE_ID.fullmatch(self.structural_branch) is None)):
            raise ValueError("hypothesis branch must identify baseline or a structural case")
        _text(self.note, "coupled teacher hypothetical", 400)
        if type(self.edits) is not tuple or not 2 <= len(self.edits) <= 4:
            raise ValueError("one atomic coupled hypothesis requires 2..4 immutable physical edits")
        if any(type(edit) is not CoupledPropertyEdit for edit in self.edits):
            raise TypeError("coupled edits must use exact built-in CoupledPropertyEdit")
        targets = [(edit.component_id, edit.property_name) for edit in self.edits]
        if len(set(targets)) != len(targets):
            raise ValueError("duplicate physical property target within one coupled hypothesis")

    def as_dict(self):
        return dict(case_id=self.case_id, structural_branch=self.structural_branch,
                    note=self.note, edits=[edit.as_dict() for edit in self.edits])


@dataclass(frozen=True, slots=True)
class CoupledPhysicalPlan:
    plan_id: str
    family: StructuralFamilyPlan
    hypotheses: tuple[CoupledHypothesis, ...]

    def __post_init__(self):
        if type(self.plan_id) is not str or _PLAN_ID.fullmatch(self.plan_id) is None:
            raise ValueError("coupled plan requires stable ASCII identifier at most 40 characters")
        if type(self.family) is not StructuralFamilyPlan:
            raise TypeError("coupled plan requires existing M29a structural family")
        if type(self.hypotheses) is not tuple or not 1 <= len(self.hypotheses) <= 5:
            raise ValueError("coupled plan requires 1..5 immutable hypotheses")
        if any(type(c) is not CoupledHypothesis for c in self.hypotheses):
            raise TypeError("coupled plan contains nonnative hypotheses")
        if any(s.case_id == "baseline" for s in self.family.cases):
            raise ValueError("baseline is reserved for the original source model")
        if len(set(c.case_id for c in self.hypotheses)) != len(self.hypotheses):
            raise ValueError("coupled hypothesis IDs must be unique")
        known = {"baseline", *(s.case_id for s in self.family.cases)}
        if any(c.structural_branch not in known for c in self.hypotheses):
            raise ValueError("coupled hypothesis targets an unknown source-structure branch")
        if sum(len(c.edits) for c in self.hypotheses) > 16:
            raise ValueError("coupled plan has more than 16 requested physical changes")
        if any(f"{self.plan_id}:{c.case_id}" == self.family.baseline.scenario_id
               for c in self.hypotheses):
            raise ValueError("coupled revision cannot reuse the original scenario ID")


def _inventory(scenario: PrescribedScenario | None):
    if scenario is None:
        return None
    return [dict(id=identity, label=label, kind=type(s).__name__)
            for (identity, label), s in zip(
                scenario.source_annotations, scenario.spec.field.sources)]


def _physical_changes(parent: PrescribedScenario, edits: tuple[CoupledPropertyEdit, ...]):
    """Check all target shapes against the assembled parent before any patch."""
    patches = dict(source_changes={}, particle_changes={}, motion_changes={},
                   gravity_changes={}, medium_changes={})
    for edit in edits:
        reference = _target(parent, edit.component_id, edit.property_name)
        if not _value_shape(edit.value, reference):
            raise TypeError("coupled physical property shape mismatches native M28a target")
        if edit.value == reference:
            raise ValueError("coupled edit repeats its physical parent's unchanged value")
        if edit.component_id == "test-particle":
            kind = ("particle_changes" if edit.property_name in
                    ("charge_c", "position_m") else "motion_changes")
            patches[kind][edit.property_name] = edit.value
        elif edit.component_id == "medium":
            patches["medium_changes"][edit.property_name] = edit.value
        elif edit.component_id == "gravity":
            patches["gravity_changes"][edit.property_name] = edit.value
        else:
            patches["source_changes"].setdefault(edit.component_id, {})[
                edit.property_name] = edit.value
    return {key: value for key, value in patches.items() if value}


@dataclass(frozen=True, slots=True)
class CoupledPhysicalEntry:
    index: int
    hypothesis: CoupledHypothesis
    structural_entry: StructuralFamilyEntry | None
    parent_scenario: PrescribedScenario | None
    scenario: PrescribedScenario | None
    trajectory: TestChargeMotionResult | None
    status: str
    reason: str
    comparisons: tuple[dict, ...]

    def as_dict(self):
        assembly = (None if self.structural_entry is None or
                    self.structural_entry.assembly is None else
                    self.structural_entry.assembly.as_dict())
        return dict(
            index=self.index,
            hypothesis=self.hypothesis.as_dict(),
            status=self.status, reason=self.reason,
            accepted_parent_inventory=_inventory(self.parent_scenario),
            structural_assembly=assembly,
            parent_revision_id=None if self.parent_scenario is None
                               else self.parent_scenario.revision_id,
            parent_scenario=None if self.parent_scenario is None
                            else self.parent_scenario.as_dict(),
            revised_scenario=None if self.scenario is None else self.scenario.as_dict(),
            revised_source_inventory=_inventory(self.scenario),
            physical_trajectory=None if self.trajectory is None else self.trajectory.as_dict(),
            exact_time_comparisons=list(self.comparisons),
        )


@dataclass(frozen=True, slots=True)
class CoupledPhysicalRun:
    plan: CoupledPhysicalPlan
    structural_run: StructuralFamilyRun
    entries: tuple[CoupledPhysicalEntry, ...]

    def __post_init__(self):
        if (type(self.plan) is not CoupledPhysicalPlan or
                type(self.structural_run) is not StructuralFamilyRun or
                self.structural_run.plan is not self.plan.family or
                type(self.entries) is not tuple or
                len(self.entries) != len(self.plan.hypotheses)):
            raise ValueError("coupled result not bound to its original physical source-family")
        structural = {entry.case.case_id: entry for entry in self.structural_run.entries}
        for idx, entry in enumerate(self.entries, 1):
            if (type(entry) is not CoupledPhysicalEntry or entry.index != idx or
                    entry.hypothesis is not self.plan.hypotheses[idx-1] or
                    type(entry.comparisons) is not tuple):
                raise ValueError("coupled result has invalid hypothesis order or identity")
            if entry.hypothesis.structural_branch == "baseline":
                expected_parent, expected_entry = self.plan.family.baseline, None
            else:
                expected_entry = structural[entry.hypothesis.structural_branch]
                expected_parent = (None if expected_entry.assembly is None else
                                   expected_entry.assembly.scenario)
            if entry.structural_entry is not expected_entry or entry.parent_scenario is not expected_parent:
                raise ValueError("coupled result bound to wrong structural source world")
            if expected_parent is None:
                if (entry.status != "skipped-structural-model" or
                        entry.scenario is not None or entry.trajectory is not None or entry.comparisons):
                    raise ValueError("refused physical source world has fabricated revisions")
                continue
            if entry.scenario is None:
                if (entry.trajectory is not None or entry.comparisons or
                        entry.status not in ("unavailable-branch-trajectory",
                                            "rejected-coupled-edit", "rejected-physical-model")):
                    raise ValueError("unsolved coupled hypothesis has fabricated native evidence")
                continue
            if (entry.scenario.parent_revision_id != expected_parent.revision_id or
                    entry.scenario.source_annotations != expected_parent.source_annotations):
                raise ValueError("coupled physical revision lost immediate source parent or membership")
            if entry.trajectory is None:
                if entry.status != "integration-failed" or entry.comparisons:
                    raise ValueError("coupled numerical failure mislabeled as accepted observation")
                continue
            if (entry.trajectory.spec is not entry.scenario.spec or
                    entry.status != ("complete" if entry.trajectory.termination == "duration"
                                     else "censored")):
                raise ValueError("coupled native trajectory does not match its physical revision")

    def as_dict(self):
        record = dict(
            schema="conductor_sim.coupled_physical_family.v1",
            mode="Atomic full-system physical edits after qualified M22a source assembly",
            plan_id=self.plan.plan_id,
            provenance=("Author-asserted physical source data; coupled hypotheses are "
                        "separate teacher scenarios, not source transitions within a trajectory"),
            original_revision_id=self.plan.family.baseline.revision_id,
            original_scenario=self.plan.family.baseline.as_dict(),
            original_inventory=_inventory(self.plan.family.baseline),
            structural_family=self.structural_run.as_dict(),
            entries=[entry.as_dict() for entry in self.entries],
            comparison_rule="original actual equal saved SI times against immediate parent only",
            time_interpolation=False,
            postguard_extrapolation=False,
            qualification=("Native numerical trajectories are not exact ODE proofs. "
                           "Different source inventories and gauge conventions do not "
                           "permit cross-model potential-energy differences as work. "
                           "No conductor response, source motion or collision law."),
        )
        return json.loads(json.dumps(record, allow_nan=False))


def run_coupled_physical_family(plan: CoupledPhysicalPlan) -> CoupledPhysicalRun:
    if type(plan) is not CoupledPhysicalPlan:
        raise TypeError("require native CoupledPhysicalPlan, not an imported JSON record")
    original = plan.family.baseline
    original_before = original.as_dict()
    original_sha = original.revision_id
    native_family = run_structural_family(plan.family)
    structural = {entry.case.case_id: entry for entry in native_family.entries}
    entries = []
    for idx, hypo in enumerate(plan.hypotheses, 1):
        branch = (None if hypo.structural_branch == "baseline" else
                  structural[hypo.structural_branch])
        if branch is not None and branch.assembly is None:
            entries.append(CoupledPhysicalEntry(
                idx,hypo,branch,None,None,None,
                "skipped-structural-model",branch.reason,()))
            continue
        parent = original if branch is None else branch.assembly.scenario
        if branch is not None and (
                branch.assembly.parent is not original or
                branch.assembly.as_dict()["parent_revision_id"] != original_sha):
            raise RuntimeError("coupled hypothesis lost exact original M22a source parent")
        parent_native = (native_family.baseline_trajectory if branch is None
                         else branch.trajectory)
        if parent_native is None:
            entries.append(CoupledPhysicalEntry(
                idx,hypo,branch,parent,None,None,
                "unavailable-branch-trajectory",
                "Accepted physical source arrangement has no native parent trajectory: "
                +branch.reason,()))
            continue
        try:
            patches = _physical_changes(parent,hypo.edits)
        except (UnsupportedProblem, TypeError, ValueError) as exc:
            entries.append(CoupledPhysicalEntry(
                idx,hypo,branch,parent,None,None,
                "rejected-coupled-edit",f"{type(exc).__name__}: {exc}",()))
            continue
        try:
            candidate = revise_scenario(
                parent,scenario_id=f"{plan.plan_id}:{hypo.case_id}",
                note=hypo.note,**patches)
        except (UnsupportedProblem, TypeError, ValueError) as exc:
            entries.append(CoupledPhysicalEntry(
                idx,hypo,branch,parent,None,None,
                "rejected-physical-model",f"{type(exc).__name__}: {exc}",()))
            continue
        if (candidate.parent_revision_id != parent.revision_id or
                candidate.source_annotations != parent.source_annotations):
            raise RuntimeError("M20b joint revision has lost its actual immediate parent")
        try:
            solved = integrate_test_charge(candidate.spec)
        except (UnsupportedProblem, MotionIntegrationError) as exc:
            entries.append(CoupledPhysicalEntry(
                idx,hypo,branch,parent,candidate,None,
                "integration-failed",f"{type(exc).__name__}: {exc}",()))
            continue
        if solved.spec is not candidate.spec:
            raise RuntimeError("coupled native integration returned another model's results")
        status = "complete" if solved.termination == "duration" else "censored"
        aligned = tuple(_shared_rows(parent_native, solved))
        entries.append(CoupledPhysicalEntry(
            idx,hypo,branch,parent,candidate,solved,status,solved.termination,aligned))
    if original.revision_id != original_sha or original.as_dict() != original_before:
        raise RuntimeError("coupled source exploration mutated immutable original model")
    return CoupledPhysicalRun(plan,native_family,tuple(entries))
