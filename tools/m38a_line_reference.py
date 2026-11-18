"""Independent standard-library-only physical oracle for the M38a JSON output.

NO conductor_sim imports: reconstructs the full kinematics and Coulomb/Kelvin
force, energy, induced sphere charge and both signed ground terminal currents.
This authenticates necessary SI mathematics, not the origin of arbitrary JSON.
"""
from __future__ import annotations

import argparse
import json
from math import fsum, hypot, isclose, isfinite, pi, ulp
from pathlib import Path

SCHEMA = "conductor_sim.prescribed_ground_line.v1"
PROTOCOL = "external prescribed CONSTANT 3D Cartesian velocity; static Kelvin observations; NOT Newtonian dynamics"
ROLE = "teacher_hypothetical_off_axis_transfer_not_original_workbook_p8_Q21"
ORIGINAL_NOTE = "p8 Multiple Correct Q21 is RADIAL; this path is a teacher extension"
UNITS = {
    "time": "s", "distance": "m", "position": "m",
    "velocity": "m/s", "radial_velocity": "m/s",
    "charge": "C", "current": "A", "force": "N",
    "energy": "J", "permittivity": "F/m",
}


def _number(value, label):
    if type(value) not in (float, int) or not isfinite(value):
        raise ValueError(f"{label}: expected finite non-Boolean SI number")
    return float(value)


def _vector(values, label):
    if type(values) not in (list, tuple) or len(values) != 3:
        raise ValueError(f"{label}: expected three SI coordinates")
    v = tuple(_number(x, label) for x in values)
    if not isfinite(hypot(*v)):
        raise ValueError(f"{label}: vector magnitude overflows")
    return v


def _close(actual, expected, label):
    value = _number(actual, label)
    if not isclose(value, expected, rel_tol=3e-9,
                   abs_tol=128 * ulp(abs(expected))):
        raise ValueError(f"{label}: claimed {value!r} differs from independent SI {expected!r}")


def _required(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"{label}: incorrect evidence keys")


def check_line_record(data):
    headers = {
        "schema", "protocol", "source_role", "original_workbook_case",
        "units", "sphere_name", "sphere_constraint", "sphere_is_solid",
        "sphere_center_m", "sphere_radius_m", "permittivity_f_per_m",
        "source_charge_c", "initial_source_position_m",
        "constant_cartesian_velocity_m_s", "duration_s", "surface_guard_m",
        "sample_count", "closest_approach_time_s", "closest_approach_distance_m",
        "samples",
    }
    _required(data, headers, "line export")
    if data["schema"] != SCHEMA or data["protocol"] != PROTOCOL:
        raise ValueError("unsupported or misleading path/physics protocol")
    if data["source_role"] != ROLE or data["original_workbook_case"] != ORIGINAL_NOTE:
        raise ValueError("teacher-only off-axis extension falsely attributed to original workbook")
    if type(data["units"]) is not dict or data["units"] != UNITS:
        raise ValueError("saved SI unit declarations are false")
    if type(data["sphere_name"]) is not str or not data["sphere_name"].strip():
        raise ValueError("authored grounded sphere identity is missing")
    if data["sphere_constraint"] != "Grounded" or data["sphere_is_solid"] is not True:
        raise ValueError("line-current law requires a genuinely grounded solid sphere")
    center = _vector(data["sphere_center_m"], "sphere center")
    x0 = _vector(data["initial_source_position_m"], "source initial position")
    velocity = _vector(data["constant_cartesian_velocity_m_s"], "source Cartesian velocity")
    q = _number(data["source_charge_c"], "source charge")
    a = _number(data["sphere_radius_m"], "solid sphere radius")
    eps = _number(data["permittivity_f_per_m"], "medium permittivity")
    guard = _number(data["surface_guard_m"], "surface approach guard")
    duration = _number(data["duration_s"], "prescribed duration")
    n = data["sample_count"]
    speed = hypot(*velocity)
    if (q == 0 or a <= 0 or eps <= 0 or guard <= 0 or duration <= 0
            or type(n) is not int or not 3 <= n <= 201
            or not 0 < speed < 0.1 * 299_792_458):
        raise ValueError("original physical model or path is outside qualified domain")
    if not isfinite(a + guard):
        raise ValueError("unrepresentable physical surface clearance")
    endpoint = tuple(x0[i] + velocity[i] * duration for i in range(3))
    if not all(isfinite(z) for z in endpoint) or endpoint == x0:
        raise ValueError("physical line endpoint is not numerically resolved")
    resolution = max(ulp(abs(z)) for z in (*x0, *center, *endpoint))
    if 64 * resolution >= min(a, guard):
        raise ValueError("world-space geometry cannot resolve guarded solid sphere")
    relative = tuple(x0[i] - center[i] for i in range(3))
    if not isfinite(hypot(*relative)):
        raise ValueError("unrepresentable source-to-metal separation")
    # Analytic derivative of |r0 + vt|^2: 2(r0.v + |v|^2*t)=0.
    # An independent calculation; never infer safety from sampled snapshots.
    closest_unbounded = fsum(relative[i] * (velocity[i] / speed)
                             for i in range(3)) / -speed
    if not isfinite(closest_unbounded):
        raise ValueError("closest passage time is not representable")
    closest_t = min(duration, max(0., closest_unbounded))
    closest_point = tuple(x0[i] + velocity[i]*closest_t for i in range(3))
    if not all(isfinite(x) for x in closest_point):
        raise ValueError("continuous closest passage position overflows")
    closest_d = hypot(*(closest_point[i]-center[i] for i in range(3)))
    if not isfinite(closest_d) or closest_d <= a + guard + 128*ulp(max(closest_d, a+guard)):
        raise ValueError("path crosses or cannot resolve guarded metal between samples")
    _close(data["closest_approach_time_s"], closest_t, "closest-approach physical time")
    _close(data["closest_approach_distance_m"], closest_d, "closest-approach distance")
    rows = data["samples"]
    if type(rows) is not list or len(rows) != n:
        raise ValueError("actual elapsed-time saved sample count is wrong")
    sample_keys = {
        "index", "time_s", "radial_distance_m", "radial_velocity_m_s",
        "source_position_m", "source_velocity_m_s",
        "induced_sphere_net_charge_c", "wire_current_sphere_to_ground_a",
        "wire_current_ground_to_sphere_a", "induced_force_on_source_n",
        "induced_mechanical_energy_j",
    }
    k = 1/(4*pi*eps)
    previous = None
    for i, row in enumerate(rows):
        _required(row, sample_keys, f"sample {i}")
        if type(row["index"]) is not int or row["index"] != i:
            raise ValueError("saved physical sample index/order was forged")
        t = duration*i/(n-1)
        _close(row["time_s"], t, f"sample {i} actual prescribed time")
        point = tuple(x0[j] + velocity[j]*t for j in range(3))
        if not all(isfinite(x) for x in point) or (previous is not None and previous == point):
            raise ValueError("source world-space position is not numerically resolvable")
        actual = _vector(row["source_position_m"], f"sample {i} source position")
        saved_v = _vector(row["source_velocity_m_s"], f"sample {i} source velocity")
        for j in range(3):
            _close(actual[j], point[j], f"sample {i} physical x{j}")
            _close(saved_v[j], velocity[j], f"sample {i} physical v{j}")
        radius_vector = tuple(point[j]-center[j] for j in range(3))
        d = hypot(*radius_vector)
        if not isfinite(d) or d <= a + guard:
            raise ValueError("saved point entered the physical metal/surface guard")
        vr = fsum(radius_vector[j]*velocity[j]/d for j in range(3))
        induced_q = -q*a/d
        wire = -q*a*vr/(d*d)
        force_r = -k*q*q*a*d/((d-a)**2 * (d+a)**2)
        induced_force = tuple(force_r*radius_vector[j]/d for j in range(3))
        induced_u = -k*q*q*a/(2*(d-a)*(d+a))
        for key, expected in (
            ("radial_distance_m", d),
            ("radial_velocity_m_s", vr),
            ("induced_sphere_net_charge_c", induced_q),
            ("wire_current_sphere_to_ground_a", wire),
            ("wire_current_ground_to_sphere_a", -wire),
            ("induced_mechanical_energy_j", induced_u),
        ):
            _close(row[key], expected, f"sample {i} {key}")
        force_saved = _vector(row["induced_force_on_source_n"], f"sample {i} force")
        for j in range(3):
            _close(force_saved[j], induced_force[j], f"sample {i} induced force axis {j}")
        previous = point
    return {
        "schema": "conductor_sim.m38a_independent_reference.v1",
        "teacher_hypothetical_paths": 1,
        "original_workbook_items_newly_solved": 0,
        "independently_checked_kelvin_states": n,
        "closest_approach_time_s": closest_t,
        "closest_approach_distance_m": closest_d,
        "signed_ground_wire_current": "verified independently for every saved physical frame",
        "continuous_between_frame_guard": "verified analytically, not sampled",
    }


def main():
    p = argparse.ArgumentParser(description="Independently rederive M38a Kelvin and line-kinematic physics")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    record = json.loads(args.input.read_text(encoding="utf-8"))
    answer = check_line_record(record)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(answer, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(answer, indent=2))


if __name__ == "__main__":
    main()
