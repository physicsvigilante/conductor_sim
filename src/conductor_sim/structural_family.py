"""M29a: bounded, independent structural source-family trajectory comparisons.

An authored structural change is *not* a source property revision (M20b/M28a)
and not a physical transition within one trajectory. Reuse M22a's atomic
whole-system reassembly and the unchanged M27b vector particle integrator.
No source E/V, force law, RK, guard, plane qualifier or browser physics lives
here. SHA identities attest authored records, not workbook authentication.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re

from .conservative_forces import UniformGravity
from .motion_1d import MotionIntegrationError
from .prescribed_motion import TestChargeMotionResult, integrate_test_charge
from .scenarios import PrescribedScenario, _text
from .solver import UnsupportedProblem
from .structural_scenarios import (
    NamedSource, SourceReplacement, ScenarioReassembly, reassemble_scenario,
)
from .trajectory_sweep import _shared_rows


_CASE_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,19}\Z")
_PLAN_ID = re.compile(r"[A-Za-z][A-Za-z0-9:_-]{0,39}\Z")
_MAX_ALTERNATIVES = 5
_MAX_STRUCTURAL_OPS_PER_CASE = 12


def _typed_tuple(value, cls, name):
    if type(value) is not tuple or any(type(x) is not cls for x in value):
        raise TypeError(f"{name} must be an immutable tuple of {cls.__name__}")
    return value


@dataclass(frozen=True, slots=True)
class StructuralCase:
    """One alternative whole-source-inventory authoring operation.

    All operations form a single M22a atomic reassembly. A source replacement
    *retires* the old identity; a property edit under the same ID belongs to
    M20b and is not masqueraded as a structural change.
    """
    case_id: str
    note: str
    add_sources: tuple[NamedSource, ...] = ()
    remove_sources: tuple[str, ...] = ()
    replace_sources: tuple[SourceReplacement, ...] = ()
    gravity_action: str = "keep"
    gravity: UniformGravity | None = None

    def __post_init__(self):
        if type(self.case_id) is not str or _CASE_ID.fullmatch(self.case_id) is None:
            raise ValueError("structural case requires a stable bounded case ID")
        _text(self.note, "structural family teacher note", 400)
        _typed_tuple(self.add_sources, NamedSource, "add_sources")
        _typed_tuple(self.remove_sources, str, "remove_sources")
        _typed_tuple(self.replace_sources, SourceReplacement, "replace_sources")
        if type(self.gravity_action) is not str or self.gravity_action not in (
                "keep", "add", "remove"):
            raise ValueError("gravity_action must be keep, add, or remove")
        if self.gravity_action == "add":
            if type(self.gravity) is not UniformGravity:
                raise TypeError("adding gravity requires an existing built-in UniformGravity")
        elif self.gravity is not None:
            raise ValueError("gravity parameter supplied without gravity_action='add'")
        count = (len(self.add_sources) + len(self.remove_sources) +
                 len(self.replace_sources) +
                 (0 if self.gravity_action == "keep" else 1))
        if not 1 <= count <= _MAX_STRUCTURAL_OPS_PER_CASE:
            raise ValueError("structural alternative needs 1..12 explicit membership operations")
        if any(type(x) is not str or not x or len(x) > 64 for x in self.remove_sources):
            raise ValueError("removed source IDs must be bounded nonblank strings")
        # Duplicate/unknown identities are still passed to the normal M22a
        # reassembly to be classified per-case before any variant integration.

    def as_dict(self):
        return dict(case_id=self.case_id, note=self.note,
                    add_sources=[x.as_dict() for x in self.add_sources],
                    remove_sources=self.remove_sources,
                    replace_sources=[dict(old_id=r.old_id, new=r.new.as_dict())
                                     for r in self.replace_sources],
                    gravity_action=self.gravity_action,
                    gravity=(self.gravity.as_dict() if self.gravity is not None else None))


@dataclass(frozen=True, slots=True)
class StructuralFamilyPlan:
    plan_id: str
    baseline: PrescribedScenario
    cases: tuple[StructuralCase, ...]

    def __post_init__(self):
        if type(self.plan_id) is not str or _PLAN_ID.fullmatch(self.plan_id) is None:
            raise ValueError("structural family needs stable bounded plan ID (max 40)")
        if type(self.baseline) is not PrescribedScenario:
            raise TypeError("structural family baseline must be PrescribedScenario")
        _typed_tuple(self.cases, StructuralCase, "cases")
        if not 1 <= len(self.cases) <= _MAX_ALTERNATIVES:
            raise ValueError("structural family requires 1..5 independent alternatives")
        if len({c.case_id for c in self.cases}) != len(self.cases):
            raise ValueError("structural alternatives must have distinct stable case IDs")
        if any(f"{self.plan_id}:{c.case_id}" == self.baseline.scenario_id
               for c in self.cases):
            raise ValueError("structural child cannot reuse baseline scenario identity")


def _assembly(plan, case):
    kw = dict(
        scenario_id=f"{plan.plan_id}:{case.case_id}",
        note=case.note, add_sources=case.add_sources,
        remove_sources=case.remove_sources,
        replace_sources=case.replace_sources)
    if case.gravity_action == "add":
        kw["gravity"] = case.gravity
    elif case.gravity_action == "remove":
        kw["gravity"] = None
    return reassemble_scenario(plan.baseline, **kw)


@dataclass(frozen=True, slots=True)
class StructuralFamilyEntry:
    index: int
    case: StructuralCase
    assembly: ScenarioReassembly | None
    trajectory: TestChargeMotionResult | None
    status: str
    reason: str

    def as_dict(self, baseline_trajectory):
        native = self.trajectory
        assembly = self.assembly
        inventory = (None if assembly is None else
                     [dict(id=identity, label=label)
                      for identity, label in assembly.scenario.source_annotations])
        return dict(
            index=self.index, case=self.case.as_dict(),
            status=self.status, reason=self.reason,
            source_inventory=inventory,
            structural_assembly=None if assembly is None else assembly.as_dict(),
            trajectory=None if native is None else native.as_dict(),
            comparisons=[] if native is None else
            _shared_rows(baseline_trajectory, native))


@dataclass(frozen=True, slots=True)
class StructuralFamilyRun:
    plan: StructuralFamilyPlan
    baseline_trajectory: TestChargeMotionResult
    entries: tuple[StructuralFamilyEntry, ...]

    def __post_init__(self):
        if (type(self.plan) is not StructuralFamilyPlan or
                type(self.baseline_trajectory) is not TestChargeMotionResult or
                type(self.entries) is not tuple):
            raise TypeError("requires a native StructuralFamilyPlan, trajectory and entries")
        if self.baseline_trajectory.spec is not self.plan.baseline.spec:
            raise ValueError("baseline physical model and native trajectory differ")
        if len(self.entries) != len(self.plan.cases):
            raise ValueError("missing structural alternative")
        for i, entry in enumerate(self.entries, 1):
            if type(entry) is not StructuralFamilyEntry or entry.index != i:
                raise ValueError("structural alternative index/order invalid")
            if entry.case is not self.plan.cases[i-1]:
                raise ValueError("structural alternative lost authored identity")
            if entry.assembly is not None:
                if (type(entry.assembly) is not ScenarioReassembly or
                        entry.assembly.parent is not self.plan.baseline):
                    raise ValueError("structural sibling was not assembled from unchanged original")
                if (entry.trajectory is not None and
                        entry.trajectory.spec is not entry.assembly.scenario.spec):
                    raise ValueError("native structural trajectory does not belong to assembled model")
            elif entry.trajectory is not None:
                raise ValueError("trajectory exists without qualified structural model")

    def as_dict(self):
        """Detached, finite actual SI evidence, not arbitrary data import."""
        result = dict(
            schema="conductor_sim.structural_motion_family.v1",
            mode="Independent whole-model structural reassembly and unchanged native RK",
            plan_id=self.plan.plan_id,
            provenance=("Parent source facts are author assertions, not scanned-PDF "
                        "authentication. Each membership change is a separate teacher "
                        "hypothetical, not a transition within a particle trajectory."),
            baseline_scenario=self.plan.baseline.as_dict(),
            baseline_revision_id=self.plan.baseline.revision_id,
            baseline_source_inventory=[
                dict(id=i, label=label)
                for i, label in self.plan.baseline.source_annotations],
            baseline_trajectory=self.baseline_trajectory.as_dict(),
            baseline_termination=self.baseline_trajectory.termination,
            entries=[e.as_dict(self.baseline_trajectory) for e in self.entries],
            alignment="literal matching actual saved SI timestamps only",
            time_interpolation=False, postguard_extrapolation=False,
            qualification=("Rejected structural models are not solved; source-guard "
                           "histories are censored, never extrapolated. Cross-inventory "
                           "potential energy differences are not work or induced energy; "
                           "no moving source/contact/conductor response is inferred."),
        )
        return json.loads(json.dumps(result, allow_nan=False))


def run_structural_family(plan: StructuralFamilyPlan) -> StructuralFamilyRun:
    """Freshly qualify and solve each structural sibling from ONE unchanged baseline.

    Valid model admission does not imply numerical completion. Only expected
    whole-model admission errors and recognized integration-domain failures
    are recorded. Unexpected coding errors must propagate.
    """
    if type(plan) is not StructuralFamilyPlan:
        raise TypeError("require StructuralFamilyPlan")
    original_revision = plan.baseline.revision_id
    original_record = plan.baseline.as_dict()
    baseline = integrate_test_charge(plan.baseline.spec)
    if baseline.spec is not plan.baseline.spec:
        raise RuntimeError("integrator replaced baseline physical model")
    items = []
    for i, case in enumerate(plan.cases, 1):
        try:
            assembly = _assembly(plan, case)
        except (UnsupportedProblem, ValueError, TypeError) as exc:
            items.append(StructuralFamilyEntry(
                i, case, None, None, "rejected-structural-model",
                f"{type(exc).__name__}: {exc}"))
            continue
        if assembly.parent is not plan.baseline:
            raise RuntimeError("structural child lost original source-model lineage")
        try:
            native = integrate_test_charge(assembly.scenario.spec)
        except (UnsupportedProblem, MotionIntegrationError) as exc:
            items.append(StructuralFamilyEntry(
                i, case, assembly, None, "integration-failed",
                f"{type(exc).__name__}: {exc}"))
            continue
        if native.spec is not assembly.scenario.spec:
            raise RuntimeError("integrator replaced structurally authored model")
        status = "complete" if native.termination == "duration" else "censored"
        items.append(StructuralFamilyEntry(
            i, case, assembly, native, status, native.termination))
    if (plan.baseline.revision_id != original_revision or
            plan.baseline.as_dict() != original_record):
        raise RuntimeError("baseline changed during structural-family study")
    return StructuralFamilyRun(plan, baseline, tuple(items))
