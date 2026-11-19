"""Independent M38b visual/physical SI reference: standard library + M38a oracle.

Do NOT import conductor_sim, production viewer, its projection helpers, or
the example generator. Re-derive basis, projection, graph and display rows.
"""
from __future__ import annotations

import argparse
import json
from math import fsum, hypot, isclose, isfinite, ulp
from pathlib import Path

from tools.m38a_line_reference import check_line_record

SCHEMA = "conductor_sim.prescribed_ground_line_viewer.v1"
HEADING = "TEACHER HYPOTHETICAL · 3D off-axis grounded-sphere passage"
ORIGINAL = (
    "Original workbook p8 Multiple Correct Q21 is RADIAL. This off-axis "
    "straight-line 3D passage is TEACHER-AUTHORED, not a printed workbook "
    "apparatus or an additional solved original JEE problem."
)
TEACHER = (
    "ALL numbers, source positions, 3D velocity and duration are "
    "TEACHER-CHOSEN SI values, NOT printed workbook numerical givens."
)
PROTOCOL = (
    "Externally maintained constant CARTESIAN velocity; changing radial speed "
    "and instantaneous Kelvin equilibria. NOT a free Newtonian orbit, "
    "real ground-wire microphysics, radiation or a Maxwell transient."
)
PROJECTION = (
    "Fixed physical local u-w plane through the sphere centre and the actual "
    "3D straight line; NOT world x-y, not a perspective 3D scene."
)
DISPLAY = (
    "Only actual precomputed M38a Kelvin states, physical local u-w "
    "projection, saved current/force/energy and teacher-prescribed time. "
    "Polyline connections are DISPLAY interpolation, never a physical solver."
)


def _shape(data, keys, label):
    if type(data) is not dict or set(data) != set(keys):
        raise ValueError(label + ": unexpected or missing scientific evidence keys")


def _number(v, label):
    if type(v) not in (int, float) or not isfinite(v):
        raise ValueError(label + ": non-Boolean finite SI number required")
    return float(v)


def _same(v, expected, label):
    actual = _number(v, label)
    if not isclose(actual, expected, rel_tol=2e-10,
                   abs_tol=128 * ulp(abs(expected))):
        raise ValueError(label + ": differs from independent physical geometry/current")


def _vector(x, label):
    if type(x) not in (tuple, list) or len(x) != 3:
        raise ValueError(label + ": physical world vector expected")
    return tuple(_number(x[i], label) for i in range(3))


def _dot(a, b):
    return fsum(a[i]*b[i] for i in range(3))


def _normed(x):
    h = hypot(*x)
    if h <= 0 or not isfinite(h):
        raise ValueError("independent local projection is mathematically degenerate")
    return tuple(k/h for k in x)


def _format(v):
    clean = _number(v, "formatted value")
    # Canonical SI readouts show 0, never misleading signed negative zero.
    if clean == 0:
        return "0"
    return format(clean, ".11g")


def _triple(x, units):
    return "[" + ", ".join(_format(k) for k in x) + "] " + units


def _check_vec(actual, expected, label):
    a = _vector(actual, label)
    for i in range(3):
        _same(a[i], expected[i], label + " Cartesian axis " + str(i))


def check_line_view(doc):
    """Recompute everything from saved independently verified source SI."""
    _shape(doc, ("schema", "source", "trace", "layout", "readouts", "display_contract"), "viewer")
    if doc["schema"] != SCHEMA or doc["display_contract"] != DISPLAY:
        raise ValueError("unsupported or misleading M38b display contract")
    source = doc["source"]
    _shape(source, ("role","heading","original_warning","teacher_notice","protocol_notice",
                    "newly_solved_original_workbook_items"), "visible source/teacher")
    if (source["role"] != "teacher_hypothetical" or source["heading"] != HEADING
            or source["original_warning"] != ORIGINAL or source["teacher_notice"] != TEACHER
            or source["protocol_notice"] != PROTOCOL
            or type(source["newly_solved_original_workbook_items"]) is not int
            or source["newly_solved_original_workbook_items"] != 0):
        raise ValueError("false visible original-versus-teacher provenance or physics mode")

    trace = doc["trace"]
    physical = check_line_record(trace)
    if physical["original_workbook_items_newly_solved"] != 0:
        raise ValueError("false new original JEE question claim")
    samples = trace["samples"]
    n = len(samples)
    rows = doc["readouts"]
    if type(rows) is not list or len(rows) != n:
        raise ValueError("display snapshots do not match independently verified Kelvin states")

    c = _vector(trace["sphere_center_m"], "sphere centre")
    x0 = _vector(trace["initial_source_position_m"], "charge initial world position")
    v = _vector(trace["constant_cartesian_velocity_m_s"], "Cartesian velocity")
    rad = _number(trace["sphere_radius_m"], "real metal radius")
    u = _normed(v)
    r0 = tuple(x0[i]-c[i] for i in range(3))
    projected = _dot(r0, u)
    trans = tuple(r0[i]-projected*u[i] for i in range(3))
    if hypot(*trans) <= 128*ulp(max(hypot(*r0), rad)):
        k = min(range(3), key=lambda i: abs(u[i]))
        test_axis = tuple(1. if i==k else 0. for i in range(3))
        projected_axis = _dot(test_axis, u)
        trans = tuple(test_axis[i]-projected_axis*u[i] for i in range(3))
    w = _normed(trans)
    nvec = _normed((
        u[1]*w[2]-u[2]*w[1],
        u[2]*w[0]-u[0]*w[2],
        u[0]*w[1]-u[1]*w[0],
    ))
    if abs(_dot(u,w)) > 2e-12:
        raise ValueError("local axes not orthogonal")

    positions = []
    for row in samples:
        p = _vector(row["source_position_m"], "saved source 3D position")
        relative = tuple(p[i]-c[i] for i in range(3))
        if abs(_dot(relative, nvec)) > 3e-10*max(row["radial_distance_m"],rad):
            raise ValueError("saved world path does not lie in source-centre physical plane")
        positions.append((_dot(relative,u), _dot(relative,w)))

    scale = min(
        275/max(rad, *(abs(a) for a,b in positions)),
        122/max(rad, *(abs(b) for a,b in positions)),
        106/rad
    )
    peak = max(abs(row["wire_current_sphere_to_ground_a"]) for row in samples)
    if not isfinite(scale) or scale<=0 or not isfinite(peak) or peak<=0:
        raise ValueError("not an independently resolved actual scene or current chart")
    ly = doc["layout"]
    keys = (
        "projection","local_u_world","local_w_world","local_normal_world",
        "u_label","w_label","normal_label","sphere_cx_px","sphere_cy_px",
        "sphere_r_px","pixels_per_meter","path_start_x_px","path_start_y_px",
        "path_end_x_px","path_end_y_px","chart_zero_y_px","chart_peak_abs_a"
    )
    _shape(ly, keys, "fixed physical scene geometry")
    if ly["projection"] != PROJECTION:
        raise ValueError("wrong physically projected axes label or world-xy attribution")
    for field, expected in (
        ("local_u_world",u), ("local_w_world",w), ("local_normal_world",nvec)
    ):
        _check_vec(ly[field],expected,field)
    for field, actual in (("u_label",u),("w_label",w),("normal_label",nvec)):
        if ly[field] != _triple(actual,"unit"):
            raise ValueError("display basis vector label falsified")
    for field, expected in (
        ("sphere_cx_px",360.), ("sphere_cy_px",190.),
        ("sphere_r_px",rad*scale), ("pixels_per_meter",scale),
        ("path_start_x_px",360+positions[0][0]*scale),
        ("path_start_y_px",190-positions[0][1]*scale),
        ("path_end_x_px",360+positions[-1][0]*scale),
        ("path_end_y_px",190-positions[-1][1]*scale),
        ("chart_zero_y_px",151.), ("chart_peak_abs_a",peak),
    ):
        _same(ly[field],expected,"physical layout "+field)

    readout_keys = (
        "index","scene_x_px","scene_y_px","sign_x_px","sign_y_px",
        "chart_x_px","chart_y_px","time_label","distance_label",
        "radial_velocity_label","induced_charge_label","current_label",
        "opposite_label","world_position_label","world_velocity_label",
        "force_label","energy_label"
    )
    for i,(row,s,(ax,wy)) in enumerate(zip(rows,samples,positions)):
        _shape(row,readout_keys,"viewer saved frame "+str(i))
        if type(row["index"]) is not int or row["index"] != i:
            raise ValueError("visual frame order/index forged")
        sx,sy=360+ax*scale,190-wy*scale
        mx=64+506*s["time_s"]/trace["duration_s"]
        my=151-97*s["wire_current_sphere_to_ground_a"]/peak
        for k,exp in (
            ("scene_x_px",sx), ("scene_y_px",sy),
            ("sign_x_px",sx),("sign_y_px",sy-20.),
            ("chart_x_px",mx), ("chart_y_px",my)
        ):
            _same(row[k],exp,"visual frame "+str(i)+" "+k)
        for k,exp in (
            ("time_label",_format(s["time_s"])+" s"),
            ("distance_label",_format(s["radial_distance_m"])+" m"),
            ("radial_velocity_label",_format(s["radial_velocity_m_s"])+" m/s"),
            ("induced_charge_label",_format(s["induced_sphere_net_charge_c"])+" C"),
            ("current_label",_format(s["wire_current_sphere_to_ground_a"])+" A"),
            ("opposite_label",_format(s["wire_current_ground_to_sphere_a"])+" A"),
            ("world_position_label",_triple(s["source_position_m"],"m")),
            ("world_velocity_label",_triple(s["source_velocity_m_s"],"m/s")),
            ("force_label",_triple(s["induced_force_on_source_n"],"N")),
            ("energy_label",_format(s["induced_mechanical_energy_j"])+" J"),
        ):
            if type(row[k]) is not str or row[k] != exp:
                raise ValueError("visual frame "+str(i)+" "+k+" falsifies verified physical SI")
    return {
        "schema": "conductor_sim.m38b_independent_display_reference.v1",
        "teacher_hypothetical_paths": 1,
        "additional_original_jee_items": 0,
        "independently_checked_kelvin_states": n,
        "independently_checked_display_rows": n,
        "physical_projection": "fixed exact local plane",
        "native_source_checker": physical["schema"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    result = check_line_view(json.loads(args.input.read_text(encoding="utf-8")))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
