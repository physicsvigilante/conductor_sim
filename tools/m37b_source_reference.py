"""M37b source-specific independent SI oracle, using Python standard library only.

Recompute every static Kelvin observation for the *teacher numerical*
constant-speed real-world protocol anchored to workbook physical p8 Q21.
Never import conductor_sim, the example runner, or browser visual code.
This checker does not prove source-archive authenticity or retarded EM response.
"""
from __future__ import annotations

import argparse
import json
from math import fsum, hypot, isclose, isfinite, pi, ulp
from pathlib import Path

EPS = 8.8541878128e-12
SOURCE_SCHEMA = "conductor_sim.prescribed_ground_wire.v1"
VIEW_SCHEMA = "conductor_sim.prescribed_ground_wire_viewer.v1"
ORIGINAL_STATEMENT = (
    "Positive point charge q approaches a grounded metallic sphere of "
    "radius R at externally maintained constant small speed v; "
    "questions ask conventional ground-wire current direction "
    "and its increasing magnitude."
)
TEACHER_LABEL = (
    "R=0.10m, q=+1nC, d0=0.30m, speed=0.04m/s, "
    "duration=3.0s and all additional numeric quantities "
    "are TEACHER-CHOSEN; NONE is a printed numerical given."
)
UNITS = {
    "time": "s", "distance": "m", "position": "m",
    "speed": "m/s", "charge": "C", "current": "A",
    "force": "N", "energy": "J", "permittivity": "F/m",
}
PROTOCOL = (
    "EXTERNALLY MAINTAINED constant radial speed; real prescribed elapsed-time "
    "snapshots of static Kelvin equilibria; NOT free Newtonian motion"
)


def number(v, title):
    if type(v) not in (int,float) or not isfinite(v):
        raise ValueError(f"{title} must be finite numeric SI, not Boolean")
    return float(v)


def vector(v, title):
    if type(v) not in (list,tuple) or len(v) != 3:
        raise ValueError(f"{title} must have three SI coordinates")
    result=tuple(number(c,title) for c in v)
    if not isfinite(hypot(*result)):
        raise ValueError(f"{title} overflows world coordinates")
    return result


def same(actual,expected,title):
    x=number(actual,title)
    if not isclose(x,expected,rel_tol=3e-9,abs_tol=128*ulp(abs(expected))):
        raise ValueError(f"{title} SI mismatch: {x} versus independent expected {expected}")


def fmt(v):
    return format(v,".11g")


def check(manifest):
    if type(manifest) is not dict or manifest.get("schema") != VIEW_SCHEMA:
        raise ValueError("M37b visual dataset schema missing")
    if set(manifest) != {"schema","source","trace","layout","readouts","display_contract"}:
        raise ValueError("scientific viewer export contains unknown or missing top-level claims")
    meta=manifest.get("source")
    if type(meta) is not dict or set(meta) != {
        "role","source_file","physical_pdf_page","printed_anchor","case_id",
        "source_kind","statement","original_symbolic_givens",
        "original_correct_option_indices","numeric_attribution",
        "original_question_apparatus_count","teacher_hypotheticals_counted_as_original",
    }:
        raise ValueError("source metadata schema missing or contains unverified claims")
    if (
        meta.get("role")!="original_workbook_teacher_example" or
        meta.get("source_file")!="02_conductors_half_size.pdf" or
        meta.get("physical_pdf_page")!=8 or
        meta.get("printed_anchor")!="Multiple Correct Q21" or
        meta.get("case_id")!="p08-multiple-21" or
        meta.get("source_kind")!="workbook exercise, NOT an asserted official JEE exam question" or
        meta.get("original_symbolic_givens")!=["q > 0","R > 0","grounded metal sphere","constant small inward v > 0"] or
        type(meta.get("original_correct_option_indices")) is not list or
        [(type(x),x) for x in meta["original_correct_option_indices"]] != [(int,1),(int,3)] or
        meta.get("numeric_attribution")!=TEACHER_LABEL or
        meta.get("statement")!=ORIGINAL_STATEMENT or
        type(meta.get("original_question_apparatus_count")) is not int or
        meta["original_question_apparatus_count"] != 1 or
        type(meta.get("teacher_hypotheticals_counted_as_original")) is not int or
        meta["teacher_hypotheticals_counted_as_original"] != 0
    ):
        raise ValueError("false workbook given, correct-option, or teacher-numerical attribution")
    if manifest.get("display_contract")!=(
        "Only precomputed Python Kelvin equilibrium observations and externally prescribed "
        "constant-speed timestamps; display is a fixed-radial schematic, NOT an M16a "
        "trajectory or browser electromagnetic calculation."
    ):
        raise ValueError("false display-physics contract")
    data=manifest.get("trace")
    if type(data) is not dict or set(data) != {
        "schema","protocol","actual_time_origin","units","source_name",
        "sphere_constraint","sphere_is_solid","sphere_center_m","sphere_radius_m",
        "permittivity_f_per_m","source_charge_c","initial_source_position_m",
        "radial_axis","signed_constant_radial_speed_m_s","duration_s",
        "sample_count","surface_guard_m","samples",
    } or data.get("schema")!=SOURCE_SCHEMA or data.get("protocol")!=PROTOCOL:
        raise ValueError("qualified physical protocol or source fields falsely labelled")
    if data.get("actual_time_origin")!=(
        "teacher-prescribed constant-speed kinematics; NOT M16a integration"
    ):
        raise ValueError("teacher-driven time falsely mislabelled as Newtonian motion")
    if type(data.get("units")) is not dict or data["units"]!=UNITS:
        raise ValueError("incorrect physical SI units or missing unit labels")
    if (data.get("source_name")!="Sphere A" or
        data.get("sphere_constraint")!="Grounded" or
        data.get("sphere_is_solid") is not True):
        raise ValueError("false physical source identity, ground constraint or solid geometry")
    center=vector(data.get("sphere_center_m"),"sphere centre")
    a=number(data.get("sphere_radius_m"),"sphere radius")
    q=number(data.get("source_charge_c"),"source charge")
    eps=number(data.get("permittivity_f_per_m"),"permittivity")
    p0=vector(data.get("initial_source_position_m"),"initial physical charge")
    axis=vector(data.get("radial_axis"),"world radial axis")
    speed=number(data.get("signed_constant_radial_speed_m_s"),"signed constant speed")
    duration=number(data.get("duration_s"),"prescribed duration")
    guard=number(data.get("surface_guard_m"),"physical surface guard")
    count=data.get("sample_count")
    if type(count) is not int or not 3 <= count <= 201:
        raise ValueError("sample count must be valid exact integer")
    if not all(x>0 for x in (a,eps,duration,guard)) or q==0 or speed==0 or abs(speed)>=.1*299_792_458:
        raise ValueError("invalid charged-source physical parameter domain")
    for label,value,expected in (
        ("teacher sphere R",a,.1),("teacher q",q,1e-9),
        ("teacher epsilon",eps,EPS),("teacher speed",speed,-.04),
        ("teacher duration",duration,3.),("teacher guard",guard,1e-4)
    ): same(value,expected,label)
    if count!=31 or center!=(0.,0.,0.) or p0!=(0.,0.,.3):
        raise ValueError("teacher numerical geometry/count differs from declared example")
    d0=hypot(*(p0[i]-center[i] for i in range(3)))
    if d0<=a+guard or d0+speed*duration<=a+guard:
        raise ValueError("source must stay outside genuine guarded sphere")
    if not isclose(hypot(*axis),1.,rel_tol=2e-12):
        raise ValueError("radial axis does not have unit magnitude")
    for j in range(3):
        same(axis[j],(p0[j]-center[j])/d0,"radial world axis")
    rows=data.get("samples")
    readouts=manifest.get("readouts")
    if type(rows) is not list or type(readouts) is not list or len(rows)!=count or len(readouts)!=count:
        raise ValueError("sample count/readout count contradict authored protocol")
    k=1/(4*pi*eps)
    current_values=[]
    distances=[]
    for i,(row,visible) in enumerate(zip(rows,readouts)):
        if type(row) is not dict or type(visible) is not dict:
            raise ValueError("malformed numerical row or viewer readout")
        if set(visible) != {
            "index","charge_x_px","chart_x_px","chart_y_px",
            "time_label","distance_label","induced_charge_label",
            "wire_current_label","opposite_current_label",
        }:
            raise ValueError("viewer readout has missing or extraneous unverified SI labels")
        if set(row)!={
            "index","time_s","radial_distance_m","source_position_m",
            "source_velocity_m_s","induced_sphere_net_charge_c",
            "wire_current_sphere_to_ground_a","wire_current_ground_to_sphere_a",
            "induced_force_on_source_n","induced_mechanical_energy_j",
        }:
            raise ValueError("physical sample schema mutated")
        t=duration*i/(count-1)
        d=d0+speed*t
        if not d>a+guard:
            raise ValueError("sample crosses ideal metal/surface guard")
        expected_Q=-q*a/d
        expected_I=-q*a*speed/(d*d)
        f_radial=-k*q*q*a*d/((d-a)**2*(d+a)**2)
        induced_U=-k*q*q*a/(2*(d-a)*(d+a))
        if type(row.get("index")) is not int or row["index"]!=i:
            raise ValueError("out-of-order or forged physical snapshot index")
        for title,expected in (
            ("saved prescribed time",t),
            ("saved physical separation",d),
            ("native induced conductor charge",expected_Q),
            ("conventional sphere-to-ground current",expected_I),
            ("opposite wire current",-expected_I),
            ("physical induced mechanical energy",induced_U),
        ):
            key={
                "saved prescribed time":"time_s",
                "saved physical separation":"radial_distance_m",
                "native induced conductor charge":"induced_sphere_net_charge_c",
                "conventional sphere-to-ground current":"wire_current_sphere_to_ground_a",
                "opposite wire current":"wire_current_ground_to_sphere_a",
                "physical induced mechanical energy":"induced_mechanical_energy_j",
            }[title]
            same(row.get(key),expected,f"frame {i} {title}")
        actual_p=vector(row.get("source_position_m"),"saved physical source")
        actual_v=vector(row.get("source_velocity_m_s"),"saved physical velocity")
        actual_force=vector(row.get("induced_force_on_source_n"),"native induced force")
        for j in range(3):
            same(actual_p[j],center[j]+axis[j]*d,f"frame {i} physical source x{j}")
            same(actual_v[j],axis[j]*speed,f"frame {i} velocity axis {j}")
            same(actual_force[j],axis[j]*f_radial,f"frame {i} induced self-free force axis {j}")
        if type(visible.get("index")) is not int or visible["index"]!=i:
            raise ValueError("viewer frame index conflicts with source SI")
        for label,expected in (
            ("time_label",fmt(t)+" s"),
            ("distance_label",fmt(d)+" m"),
            ("induced_charge_label",fmt(row["induced_sphere_net_charge_c"])+" C"),
            ("wire_current_label",fmt(row["wire_current_sphere_to_ground_a"])+" A"),
            ("opposite_current_label",fmt(row["wire_current_ground_to_sphere_a"])+" A"),
        ):
            if visible.get(label)!=expected:
                raise ValueError(f"visible frame {i} {label} contradicts source physics")
        current_values.append(expected_I)
        distances.append(d)
    if not (all(current>0 for current in current_values) and
            all(y>x for x,y in zip(current_values,current_values[1:])) and
            all(y<x for x,y in zip(distances,distances[1:]))):
        raise ValueError("source q>0 constant-speed approach has incorrect direction/trend")
    layout=manifest.get("layout")
    if type(layout) is not dict or set(layout) != {
        "fixed_scene","radial_schematic","sphere_cx_px","sphere_cy_px",
        "sphere_radius_px","charge_cy_px","px_per_meter",
        "chart_center_y_px","chart_current_peak_abs_a","scene_positions_precomputed",
    } or layout.get("fixed_scene") is not True or layout.get("radial_schematic") is not True or layout.get("scene_positions_precomputed") is not True:
        raise ValueError("misleading physical scene claim")
    scale=min(96./a,515./max(distances))
    x0,y0=129.,169.
    peak=max(abs(x) for x in current_values)
    for key,expected in (
        ("sphere_cx_px",x0),("sphere_cy_px",y0),
        ("sphere_radius_px",scale*a),("charge_cy_px",y0),
        ("px_per_meter",scale),("chart_center_y_px",139.),
        ("chart_current_peak_abs_a",peak)
    ):
        same(layout.get(key),expected,key)
    for i,visible in enumerate(readouts):
        current=current_values[i]
        for key,value in (
            ("charge_x_px",x0+scale*distances[i]),
            ("chart_x_px",61.+487*i/(count-1)),
            ("chart_y_px",139.-current/peak*100.),
        ):
            same(visible.get(key),value,f"viewer frame {i} {key}")
    # This is a source-specific proof of 1 workbook apparatus / options (1),(3)
    # rather than a blanket promise for a 30-problem JEE dataset.
    return {
        "physical_source_apparatus":1,
        "teacher_numeric_examples":1,
        "verified_python_kelvin_states":count,
        "original_answer_options":[1,3],
        "ground_wire_current_increases":True,
        "kinematic_protocol":"teacher_prescribed_constant_speed_not_M16a",
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True,type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    manifest=json.loads(args.input.read_text(encoding="utf-8"))
    report=check(manifest)
    if args.output is not None:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,sort_keys=True))


if __name__=="__main__":
    main()
