"""M35b: strictly in-process verification helpers for genuine legacy sweep evidence.

This is NOT an external cryptographic signature or an independent ODE proof.
Only runner-issued native sweep results can carry a valid private receipt;
exported JSON remains detached and cannot be imported as trusted physics.
"""
from __future__ import annotations

from hashlib import sha256
import json
from math import isfinite

from .prescribed_motion import TestChargeMotionResult, TestChargeSample
from .scenarios import PrescribedScenario

_ISSUER = object()


def _receipt(record: dict) -> str:
    return sha256(json.dumps(
        record, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("ascii")).hexdigest()


def _issue(result_type, **parts):
    """Construct *only* a fully validated issued result; no fake public ctor."""
    raw = object.__new__(result_type)
    for name, value in parts.items():
        object.__setattr__(raw, name, value)
    object.__setattr__(raw, "_issuer", _ISSUER)
    object.__setattr__(raw, "_issued_digest", _receipt(raw._record()))
    raw.__post_init__()
    return raw


def _verify_issued(run):
    if (run._issuer is not _ISSUER or type(run._issued_digest) is not str
            or len(run._issued_digest) != 64
            or run._issued_digest != _receipt(run._record())):
        raise ValueError(
            "legacy native sweep evidence was never issued by its runner, "
            "or its model, history, status or numerical evidence was altered")


def _native_trajectory(trajectory, scenario):
    """Validate model binding, physical clock, type fidelity and guard horizon."""
    if (type(scenario) is not PrescribedScenario or
            type(trajectory) is not TestChargeMotionResult or
            trajectory.spec is not scenario.spec):
        raise ValueError("native sweep trajectory does not belong to its physical model")
    if trajectory.termination not in ("duration", "source-guard"):
        raise ValueError("unknown actual trajectory termination")
    rows = trajectory.samples
    if type(rows) is not tuple or not 1 <= len(rows) <= 2001:
        raise ValueError("native trajectory requires bounded actual saved physical samples")
    if (type(trajectory.max_abs_energy_drift_j) not in (int, float) or
            not isfinite(trajectory.max_abs_energy_drift_j) or
            trajectory.max_abs_energy_drift_j < 0):
        raise ValueError("native trajectory energy-drift summary is invalid")
    previous = None
    for row in rows:
        if type(row) is not TestChargeSample or type(row.time_s) is not float:
            raise ValueError("saved native SI samples and times must have exact genuine types")
        t = row.time_s
        if (not isfinite(t) or t < 0 or t > scenario.spec.duration_s or
                previous is not None and t <= previous):
            raise ValueError("native SI trajectory has invalid saved physical time sequence")
        previous = t
        for name in ("position_m", "displacement_m", "velocity_m_s",
                     "acceleration_m_s2", "electric_field_v_m", "force_n"):
            vec = getattr(row, name)
            if (type(vec) is not tuple or len(vec) != 3 or
                    any(type(x) not in (int, float) or not isfinite(x) for x in vec)):
                raise ValueError("native trajectory contains invalid three-component SI vector")
        for name in ("potential_v", "kinetic_energy_j", "potential_energy_j",
                     "total_energy_j", "energy_drift_j"):
            v = getattr(row, name)
            if type(v) not in (int, float) or not isfinite(v):
                raise ValueError("native trajectory contains invalid SI energy or potential")
    if (rows[0].time_s != 0.0 or
            rows[0].position_m != scenario.spec.particle.position_m or
            rows[0].velocity_m_s != scenario.spec.initial_velocity_m_s):
        raise ValueError("native sweep initial physical time, position or velocity differs")
    if trajectory.termination == "duration" and rows[-1].time_s != scenario.spec.duration_s:
        raise ValueError("duration-complete native sweep is missing end of observation")
    if trajectory.termination == "source-guard" and rows[-1].time_s >= scenario.spec.duration_s:
        raise ValueError("source-guard-censored sweep wrongly claims full observation")
    return "complete" if trajectory.termination == "duration" else "censored"


def _validate_entry_status(status, model, native, *, event: bool):
    if type(status) is not str:
        raise ValueError("legacy sweep entry status must be exact typed native text")
    if status == "rejected-physical-model":
        if model is not None or native is not None:
            raise ValueError("physically rejected sweep cannot have a fabricated model or run")
    elif status == "rejected-event-study" and event:
        if type(model) is not PrescribedScenario or native is not None:
            raise ValueError("rejected native event study has no recorded event-run evidence")
    elif status == "integration-failed":
        if type(model) is not PrescribedScenario or native is not None:
            raise ValueError("integration-failed sweep retains only admitted physical model")
    elif status in ("complete", "censored"):
        if type(model) is not PrescribedScenario or native is None:
            raise ValueError("complete/censored sweep must hold actual model and native run")
    else:
        raise ValueError("unknown legacy sweep result status")
