"""M35b bounded legacy-sweep native evidence validation.

A runner's in-process receipt binds original native observations and status to
the serialized record. It is not a digital signature or an external trust root.
"""
from __future__ import annotations

from hashlib import sha256
from hmac import compare_digest
import json
from math import isfinite, ulp

from .prescribed_motion import TestChargeMotionResult, TestChargeSample
from .scenarios import PrescribedScenario
from .solver import UnsupportedProblem

ISSUER = object()


def digest(record):
    return sha256(json.dumps(record, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")).hexdigest()


def verify_receipt(issuer, received, record):
    if issuer is not ISSUER or type(received) is not str or not compare_digest(
            received, digest(record)):
        raise ValueError("legacy sweep original runner-issued evidence receipt mismatch")


def _number(value, name):
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError(name + " must be finite ordinary SI number (not Boolean)")
    return value


def validate_native_trajectory(native, scenario):
    if (type(scenario) is not PrescribedScenario or
            type(native) is not TestChargeMotionResult or
            native.spec is not scenario.spec):
        raise ValueError("legacy sweep requires real same-model native trajectory")
    if native.termination not in ("duration", "source-guard"):
        raise ValueError("legacy ordinary trajectory has unsupported termination")
    rows = native.samples
    if type(rows) is not tuple or not 1 <= len(rows) <= 2001:
        raise ValueError("legacy trajectory requires bounded actual native samples")
    spec = scenario.spec
    last = -1.
    for index, sample in enumerate(rows):
        if type(sample) is not TestChargeSample:
            raise ValueError("legacy trajectory contains nonnative sample")
        t = _number(sample.time_s, "native saved time")
        if not 0 <= t <= spec.duration_s or t <= last:
            raise ValueError("legacy sweep physical SI times must increase within horizon")
        last = t
        for vec in ("position_m", "displacement_m", "velocity_m_s",
                    "acceleration_m_s2", "electric_field_v_m", "force_n"):
            value = getattr(sample, vec)
            if type(value) is not tuple or len(value) != 3:
                raise ValueError("legacy trajectory vector shape invalid: "+vec)
            for x in value:
                _number(x, vec)
        for attr in ("potential_v", "kinetic_energy_j", "potential_energy_j",
                     "total_energy_j", "energy_drift_j"):
            _number(getattr(sample, attr), attr)
        if index == 0 and (t != 0 or
                sample.position_m != spec.particle.position_m or
                sample.velocity_m_s != spec.initial_velocity_m_s):
            raise ValueError("legacy trajectory original physical initial state changed")
        if sample.displacement_m != tuple(
                sample.position_m[k]-spec.particle.position_m[k]
                for k in range(3)):
            raise ValueError("legacy trajectory physical displacement changed")
        # Validate the source's actual field and force independently of
        # serialized result fields; no new numerical integration.
        observed_E = spec.field.electric_field_at(sample.position_m)
        for actual, correct in zip(sample.electric_field_v_m, observed_E):
            if abs(actual-correct)>256*ulp(max(abs(actual),abs(correct),1e-300)):
                raise ValueError("legacy trajectory field differs from actual source")
        force = spec.forces.observe(
            sample.position_m,charge_c=spec.particle.charge_c,mass_kg=spec.mass_kg)
        for actual, expected in zip(sample.force_n,force.force_n):
            if abs(actual-expected)>256*ulp(max(abs(actual),abs(expected),1e-300)):
                raise ValueError("legacy trajectory force differs from actual model")
        if any(abs(sample.acceleration_m_s2[k]-force.force_n[k]/spec.mass_kg) >
                256*ulp(max(abs(sample.acceleration_m_s2[k]),
                            abs(force.force_n[k]/spec.mass_kg),1e-300))
                for k in range(3)):
            raise ValueError("legacy trajectory acceleration differs from qE/m+g")
        _number(sample.max_abs_energy_drift_j,"energy") if hasattr(sample, "max_abs_energy_drift_j") else None
    if native.termination == "duration" and rows[-1].time_s != spec.duration_s:
        raise ValueError("legacy complete result does not reach real duration")
    _number(native.max_abs_energy_drift_j,"maximum energy drift")
    if type(native.events) is not tuple:
        raise ValueError("legacy trajectory native source guard events not immutable")
    if native.termination == "source-guard" and rows[-1].time_s >= spec.duration_s:
        raise ValueError("guard-censored native trajectory cannot claim full horizon")


def validate_axis_value(current, expected):
    if type(current) is not type(expected):
        # Native floats, unlike Boolean numeric aliases, must retain identity.
        raise ValueError("legacy sweep requested parameter changed type")
    if isinstance(expected, tuple):
        if len(current) != len(expected) or any(
                type(a) is not type(b) or a != b for a,b in zip(current,expected)):
            raise ValueError("legacy sweep requested physical vector changed")
    elif current != expected:
        raise ValueError("legacy sweep requested parameter no longer equals authored value")
