"""Independent SI oracle for M37a output; NO conductor_sim imports.

Checks genuine saved Kelvin equilibria and the signed ground-wire current.
Does NOT re-integrate M16a or certify the source workbook figure.
"""
from __future__ import annotations

import argparse
import json
from math import fsum, hypot, isclose, isfinite, pi, ulp
from pathlib import Path

EPSILON_0 = 8.8541878128e-12


def _number(v, label):
    if type(v) not in (int, float) or not isfinite(v):
        raise ValueError(f"{label}: expected finite SI number, not Boolean")
    return float(v)


def _vector(v, label):
    if type(v) not in (list, tuple) or len(v) != 3:
        raise ValueError(f"{label}: expected three-vector")
    t = tuple(_number(x, label) for x in v)
    if not isfinite(hypot(*t)):
        raise ValueError(f"{label}: vector overflow")
    return t


def _same(actual, expected, label):
    value = _number(actual, label)
    if not isclose(value, expected, rel_tol=3e-9, abs_tol=128*ulp(abs(expected))):
        raise ValueError(f"{label}: saved {value} differs from independent SI {expected}")


def check_instant(record):
    if not isinstance(record, dict) or record.get("schema") != "conductor_sim.grounded_sphere_wire_instant.v1":
        raise ValueError("invalid instantaneous Kelvin observation schema")
    if record.get("mode") != "instantaneous Kelvin observation; supplied velocity does not define a trajectory":
        raise ValueError("prescribed instant falsely claims time evolution")
    if record.get("current_sign") != "positive sphere_to_ground is conventional +charge flux":
        raise ValueError("ground-wire current sign convention missing or false")
    if not isinstance(record.get("sphere_name"), str) or not record["sphere_name"]:
        raise ValueError("physical sphere name missing")
    c = _vector(record.get("sphere_center_m"), "center")
    x = _vector(record.get("source_position_m"), "source position")
    v = _vector(record.get("source_velocity_m_s"), "source velocity")
    a = _number(record.get("sphere_radius_m"), "radius")
    q = _number(record.get("point_charge_c"), "source charge")
    eps = _number(record.get("permittivity_f_per_m"), "permittivity")
    if not 0 < a or not 0 < eps or q == 0 or hypot(*v) >= .1*299_792_458.:
        raise ValueError("outside qualified current/kinematic domain")
    dr = tuple(x[i]-c[i] for i in range(3))
    d = hypot(*dr)
    if not isfinite(d) or not d > a:
        raise ValueError("source inside/touching conductor")
    vr = fsum(dr[i]*v[i]/d for i in range(3))
    image_q = -q*a/d
    rate = q*a*vr/(d*d)
    current = -rate
    k = 1/(4*pi*eps)
    force_r = -k*q*q*a*d/((d-a)**2*(d+a)**2)
    force = tuple(force_r*dr[i]/d for i in range(3))
    energy = -k*q*q*a/(2*(d-a)*(d+a))
    for key, expectation in (
        ("distance_from_center_m", d),
        ("radial_velocity_m_s", vr),
        ("induced_sphere_net_charge_c", image_q),
        ("induced_net_charge_rate_c_per_s", rate),
        ("wire_current_sphere_to_ground_a", current),
        ("wire_current_ground_to_sphere_a", rate),
        ("induced_mechanical_energy_j", energy),
    ):
        _same(record.get(key), expectation, key)
    force_saved = _vector(record.get("induced_force_on_source_n"), "induced force")
    for i in range(3):
        _same(force_saved[i], force[i], f"physical induced force axis {i}")
    return d, vr, energy, force


def validate_examples(document):
    if not isinstance(document, dict) or document.get("schema") != "conductor_sim.m37a_teacher_wire_exploration.v1":
        raise ValueError("unrecognized M37a teacher example JSON")
    if document.get("source_role") != ("teacher-authored numerical models; original workbook source p8 Q21 "
                                       "not yet individually revalidated"):
        raise ValueError("original JEE question provenance must not be fabricated")
    states = document.get("instantaneous_prescribed")
    if not isinstance(states, list) or len(states) != 3:
        raise ValueError("three distinct teacher-prescribed instantaneous states required")
    for state in states:
        check_instant(state)
    if not (states[0]["wire_current_sphere_to_ground_a"] > 0
            and states[1]["wire_current_sphere_to_ground_a"] < 0
            and states[2]["wire_current_sphere_to_ground_a"] == 0):
        raise ValueError("teacher inward/outward/tangential current directions contradict model")
    trace = document.get("actual_elapsed_time")
    if not isinstance(trace, dict) or trace.get("schema") != "conductor_sim.grounded_sphere_wire_motion.v1":
        raise ValueError("missing qualified actual-motion trace")
    if trace.get("mode") != "actual elapsed-time M16a motion; instant Kelvin response":
        raise ValueError("saved teacher time trace does not declare actual integrated origin")
    motion = trace.get("motion")
    if not isinstance(motion, dict) or motion.get("schema") != "conductor_sim.motion_1d.v1":
        raise ValueError("M16a motion schema missing")
    rows, observations = motion.get("samples"), trace.get("observations")
    if not isinstance(rows, list) or not isinstance(observations, list) or len(rows) != len(observations) or len(rows) < 2:
        raise ValueError("actual motion time/current sample lengths disagree")
    axis = _vector(motion.get("axis"), "authored true-motion axis")
    x0 = _vector(motion.get("initial_position_m"), "authored initial position")
    q0 = _number(motion.get("charge_c"), "original moving point charge")
    mass = _number(motion.get("mass_kg"), "moving source mass")
    if mass <= 0 or not isclose(hypot(*axis), 1., rel_tol=2e-12):
        raise ValueError("invalid physical mass or radial motion axis")
    prev_time = -1.
    firstK = firstU = None
    for row, record in zip(rows, observations):
        if not isinstance(row, dict) or not isinstance(record, dict):
            raise ValueError("malformed physical time sample")
        t = _number(row.get("time_s"), "elapsed time")
        if t <= prev_time:
            raise ValueError("actual time stamps must increase")
        prev_time = t
        _same(record.get("time_s"), t, "current snapshot elapsed time")
        d, vr, U, force = check_instant(record)
        delta = _number(row.get("displacement_m"), "actual motion displacement")
        vscalar = _number(row.get("velocity_m_s"), "signed particle velocity")
        position = _vector(row.get("position_m"), "actual saved position")
        for i in range(3):
            _same(position[i], x0[i] + delta*axis[i], "actual trajectory position")
            _same(record["source_position_m"][i], position[i], "instant/current position")
            _same(record["source_velocity_m_s"][i], vscalar*axis[i], "instant/current velocity")
        _same(record["point_charge_c"], q0, "retained moving source Q")
        if d <= record["sphere_radius_m"]:
            raise ValueError("motion crosses metal")
        _same(row.get("force_n"), fsum(force[i]*axis[i] for i in range(3)), "actual force")
        _same(row.get("acceleration_m_s2"), row["force_n"]/mass, "actual acceleration")
        K = .5*mass*vscalar*vscalar
        _same(row.get("kinetic_energy_j"), K, "actual kinetic energy")
        _same(row.get("interaction_energy_j"), U, "actual induced mechanical energy")
        # The independent SI U above is checked on its physical scale. Drift
        # itself is a subtraction of near-cancelling small energies: recompute
        # using the stored *individually SI-validated* U, with identical M16a
        # summation order. Comparing two independently rounded U differences
        # at the drift scale would manufacture a false failure.
        saved_U = _number(row.get("interaction_energy_j"), "saved interaction energy")
        _same(row.get("total_energy_j"), K+saved_U, "actual total mechanical energy")
        if firstK is None:
            firstK, firstU = K, saved_U
            if t != 0 or delta != 0:
                raise ValueError("actual motion lacks original physical initial state")
        _same(row.get("energy_drift_j"), fsum((K-firstK, saved_U-firstU)), "energy drift")
    if motion.get("termination") not in ("duration", "surface-guard"):
        raise ValueError("unsupported motion termination")
    if motion.get("termination") == "surface-guard":
        events = motion.get("events")
        if not isinstance(events, list) or not events or events[-1].get("kind") != "surface-guard":
            raise ValueError("guarded motion has no honest approach event")
    return {"prescribed_states": len(states), "actual_time_samples": len(rows),
            "motion_termination": motion["termination"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    print(json.dumps(validate_examples(data), sort_keys=True))


if __name__ == "__main__":
    main()
