"""M28a: bounded, provenance-preserving sweeps of existing prescribed trajectories.

Every variant uses M20b's immutable revision path and the unchanged M19/M27
physical qualification and RK integrator. JSON contains numerical evidence,
not a proof of the exact ODE or an authenticated transcription of a workbook.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from math import dist, isfinite
import re

from .prescribed_fields import (
    UniformElectricField, FixedPointField, UniformSphereField, UniformShellField,
    UniformRingAxisField, UniformRingField, UniformDiskAxisField, UniformRodField,
)
from .motion_1d import MotionIntegrationError
from .prescribed_motion import TestChargeMotionResult, integrate_test_charge
from .scenarios import PrescribedScenario, revise_scenario
from .solver import UnsupportedProblem

_ID = re.compile(r"[A-Za-z][A-Za-z0-9:_-]{0,63}\Z")
_MAX_VARIATIONS = 6
_PARTICLE = frozenset(("charge_c", "position_m",
                       "mass_kg", "initial_velocity_m_s"))
_GRAVITY = frozenset(("acceleration_m_s2",))

# M20b deliberately permits ordinary gauge/reference-only revisions. M28a's
# physically varying-trajectory interface is narrower. Explicit allowlists
# prevent a potential-reference shift from being mislabeled a physical force
# or geometry change. Keep new source types unsupported until classified.
_PHYSICAL_SOURCE_PROPERTIES = {
    UniformElectricField: frozenset(("field_v_m",)),
    FixedPointField: frozenset(("charge_c", "position_m")),
    UniformSphereField: frozenset(("charge_c", "radius_m", "center_m")),
    UniformShellField: frozenset(("charge_c", "radius_m", "center_m")),
    UniformRingAxisField: frozenset(("charge_c", "radius_m", "center_m", "axis")),
    UniformRingField: frozenset(("charge_c", "radius_m", "center_m", "axis")),
    UniformDiskAxisField: frozenset(("charge_c", "radius_m", "center_m", "axis")),
    UniformRodField: frozenset(("charge_c", "start_m", "end_m")),
}


def _number(value):
    return type(value) in (int, float) and isfinite(value)


def _value_shape(value, reference):
    if isinstance(reference, tuple):
        return (type(value) is tuple and len(value) == len(reference) and
                all(_number(x) for x in value))
    return _number(value)


def _target(baseline, component_id, property_name):
    if component_id == "test-particle":
        if property_name not in _PARTICLE:
            raise ValueError("unsupported test-particle physical parameter")
        if property_name in ("charge_c", "position_m"):
            return getattr(baseline.spec.particle, property_name)
        return getattr(baseline.spec, property_name)
    if component_id == "medium":
        if property_name != "permittivity_f_m":
            raise ValueError("medium permits only permittivity_f_m")
        return baseline.spec.field.permittivity_f_m
    if component_id == "gravity":
        if baseline.spec.gravity is None or property_name not in _GRAVITY:
            raise ValueError("gravity sweep requires an existing component and "
                             "a physical acceleration parameter; reference position is an energy gauge")
        return getattr(baseline.spec.gravity, property_name)
    inventory = dict(zip((identity for identity, _ in baseline.source_annotations),
                         baseline.spec.field.sources))
    if component_id not in inventory:
        raise ValueError("unknown physical source identity; source indices are not IDs")
    source = inventory[component_id]
    if property_name not in _PHYSICAL_SOURCE_PROPERTIES.get(type(source), ()):
        raise ValueError(
            "unsupported physical sweep property on named source; "
            "gauge/reference-only settings belong to the ordinary M20b revision API")
    return getattr(source, property_name)


@dataclass(frozen=True, slots=True)
class SweepAxis:
    """One physical property, whole scalar or vector values; no implicit grids."""
    component_id: str
    property_name: str
    values: tuple

    def __post_init__(self):
        if type(self.component_id) is not str or _ID.fullmatch(self.component_id) is None:
            raise ValueError("axis requires a stable physical component identifier")
        if (type(self.property_name) is not str or
                re.fullmatch(r"[a-z][a-z0-9_]{0,63}", self.property_name) is None):
            raise ValueError("axis requires a named physical property")
        if type(self.values) is not tuple or not 1 <= len(self.values) <= _MAX_VARIATIONS:
            raise ValueError("axis must contain 1..6 distinct immutable values")
        for value in self.values:
            if not (_number(value) or
                    (type(value) is tuple and len(value) == 3 and
                     all(_number(x) for x in value))):
                raise TypeError("axis values must be finite scalar or complete 3-vector SI values")
        if any(a == b for i, a in enumerate(self.values)
               for b in self.values[:i]):
            raise ValueError("duplicate sweep values")


@dataclass(frozen=True, slots=True)
class TrajectorySweepPlan:
    plan_id: str
    baseline: PrescribedScenario
    axis: SweepAxis

    def __post_init__(self):
        if (type(self.plan_id) is not str or len(self.plan_id) > 56 or
                _ID.fullmatch(self.plan_id) is None):
            raise ValueError("plan_id must be a bounded stable identifier (max 56)")
        if type(self.baseline) is not PrescribedScenario or type(self.axis) is not SweepAxis:
            raise TypeError("sweep needs a PrescribedScenario and one SweepAxis")
        reference = _target(self.baseline, self.axis.component_id,
                            self.axis.property_name)
        for value in self.axis.values:
            if not _value_shape(value, reference):
                raise TypeError("sweep value has wrong physical parameter shape or nonfinite data")
            if value == reference:
                raise ValueError("baseline value must not be repeated as a variation")
        # Validate the full new models at execution; a failed physical
        # qualification remains an observable sweep refusal rather than
        # aborting otherwise admissible sibling variations.


def _revision(plan, number, value):
    axis = plan.axis
    props = {axis.property_name: value}
    target = axis.component_id
    kw = {}
    if target == "test-particle":
        if axis.property_name in ("charge_c", "position_m"):
            kw["particle_changes"] = props
        else:
            kw["motion_changes"] = props
    elif target == "medium":
        kw["medium_changes"] = props
    elif target == "gravity":
        kw["gravity_changes"] = props
    else:
        kw["source_changes"] = {target: props}
    return revise_scenario(
        plan.baseline,
        scenario_id=f"{plan.plan_id}:v{number:02d}",
        note=f"Teacher hypothetical {target}.{axis.property_name}={value!r}; "
             f"all other authored conditions retained.",
        **kw,
    )


def _shared_rows(left, right):
    """No time interpolation or index equality assumption: literal times only."""
    times = {r.time_s: r for r in left.samples}
    matched = []
    for sample in right.samples:
        old = times.get(sample.time_s)
        if old is None:
            continue
        dpos = tuple(sample.position_m[k] - old.position_m[k] for k in range(3))
        dvel = tuple(sample.velocity_m_s[k] - old.velocity_m_s[k] for k in range(3))
        matched.append(dict(time_s=sample.time_s,
                            delta_position_m=dpos,
                            delta_velocity_m_s=dvel,
                            position_distance_m=dist(sample.position_m, old.position_m),
                            velocity_distance_m_s=dist(sample.velocity_m_s, old.velocity_m_s)))
    return matched


@dataclass(frozen=True, slots=True)
class SweepEntry:
    index: int
    requested_value: float | tuple
    scenario: PrescribedScenario | None
    trajectory: TestChargeMotionResult | None
    status: str
    explanation: str

    def as_dict(self, baseline_trajectory):
        return dict(index=self.index, requested_value=self.requested_value,
                    status=self.status, explanation=self.explanation,
                    scenario=None if self.scenario is None else self.scenario.as_dict(),
                    trajectory=None if self.trajectory is None else self.trajectory.as_dict(),
                    comparisons=[] if self.trajectory is None else
                    _shared_rows(baseline_trajectory, self.trajectory))


@dataclass(frozen=True, slots=True)
class TrajectorySweepRun:
    plan: TrajectorySweepPlan
    baseline_trajectory: TestChargeMotionResult
    entries: tuple[SweepEntry, ...]

    def as_dict(self):
        """Detached SI audit JSON, with all actual samples and explicit refusals."""
        result = dict(schema="conductor_sim.trajectory_sweep.v1",
                      mode="Independently requalified physical scenarios; original RK results",
                      provenance="Parameter revisions are teacher hypotheticals; "
                                 "source-given status is an author assertion, not PDF authentication.",
                      plan_id=self.plan.plan_id,
                      axis=dict(component_id=self.plan.axis.component_id,
                                property_name=self.plan.axis.property_name,
                                values=self.plan.axis.values),
                      baseline_scenario=self.plan.baseline.as_dict(),
                      baseline_trajectory=self.baseline_trajectory.as_dict(),
                      baseline_termination=self.baseline_trajectory.termination,
                      entries=[row.as_dict(self.baseline_trajectory) for row in self.entries],
                      alignment="identical saved physical timestamps only",
                      time_interpolation=False, extrapolation=False,
                      qualification="Early source-guard results are censored; no impact or "
                                    "later trajectory is inferred; cross-scenario potential "
                                    "energies may use different reference gauges.")
        return json.loads(json.dumps(result, allow_nan=False))


def run_trajectory_sweep(plan: TrajectorySweepPlan) -> TrajectorySweepRun:
    """Run baseline plus each independent physical hypothetical from scratch.

    Only expected model-admission failures become 'rejected' and known
    integration-domain/numerical failures become 'integration-failed'.
    Unexpected implementation exceptions propagate instead of being hidden.
    """
    if type(plan) is not TrajectorySweepPlan:
        raise TypeError("require TrajectorySweepPlan")
    baseline_revision = plan.baseline.revision_id
    baseline_run = integrate_test_charge(plan.baseline.spec)
    if baseline_run.spec is not plan.baseline.spec:
        raise RuntimeError("baseline solver returned a different physical model")
    entries = []
    for i, value in enumerate(plan.axis.values, 1):
        try:
            child = _revision(plan, i, value)
        except (UnsupportedProblem, ValueError, TypeError) as exc:
            entries.append(SweepEntry(i, value, None, None,
                                      "rejected-physical-model", f"{type(exc).__name__}: {exc}"))
            continue
        if child.parent_revision_id != baseline_revision:
            raise RuntimeError("child scenario lost its exact baseline lineage")
        try:
            result = integrate_test_charge(child.spec)
        except (UnsupportedProblem, MotionIntegrationError) as exc:
            entries.append(SweepEntry(i, value, child, None,
                                      "integration-failed", f"{type(exc).__name__}: {exc}"))
            continue
        if result.spec is not child.spec:
            raise RuntimeError("sweep solver returned a different physical model")
        status = "complete" if result.termination == "duration" else "censored"
        entries.append(SweepEntry(i, value, child, result, status, result.termination))
    if plan.baseline.revision_id != baseline_revision:
        raise RuntimeError("baseline was mutated during the sweep")
    return TrajectorySweepRun(plan, baseline_run, tuple(entries))
