"""M36d: independent SI model for a true standalone SOLID conductor.

Standard library ONLY: absolutely no imports from conductor_sim or example
modules. These are separate analytic Gauss-law expressions against native
saved solver/visualization measurements, with semantic source provenance.
"""
from __future__ import annotations

import argparse
import json
from math import isclose, isfinite, pi
from pathlib import Path

EPSILON_0_SI = 8.8541878128e-12
K = 1/(4*pi*EPSILON_0_SI)

CASES=(
    ("source_original_shape_teacher_numeric_example",.1,+1e-9,(0.,0.,0.),"FixedNetCharge"),
    ("teacher_changed_radius_charge_and_centre",.12,-2e-9,(.025,0.,0.),"FixedNetCharge"),
    ("teacher_grounded_no_external_sources",.08,0.,(0.,0.,0.),"Grounded"),
)


def demand(test: bool, message: str)->None:
    if not test:
        raise AssertionError(message)


def close(actual, expected, *, label: str, rtol=2e-11, atol=1e-16)->None:
    demand(type(actual) in (float,int) and isfinite(actual),
           "nonnumeric or nonfinite "+label)
    demand(isclose(actual, expected, rel_tol=rtol, abs_tol=atol),
           f"{label}: native {actual!r} differs from independent {expected!r}")


def vector(actual, expected, *, label, rtol=2e-11, atol=1e-15):
    demand(isinstance(actual,list) and len(actual)==3,"missing true 3D vector "+label)
    for axis,(x,y) in enumerate(zip(actual,expected)):
        close(x,y,label=f"{label}[{axis}]",rtol=rtol,atol=atol)


def check(data: dict)->dict:
    demand(data.get("schema")=="conductor_sim.m36d_real_solid_p26.v1",
           "wrong native source evidence schema")
    source=data.get("printed_source")
    demand(source=={
        "filename":"POTENTIAL_CONDUCTORS1.pdf",
        "page":26,
        "question":"Illustration 1",
        "original_shape":"solid conducting sphere",
        "original_parameters":"symbolic radius R and net charge Q",
        "original_asked":["V(R/2)","V(3R/2)"],
    },"actual printed solid-source question/provenance changed")
    demand(data["original_source_apparatus_count"]==1
           and data["teacher_hypothetical_worlds"]==2
           and data["physical_time_evolution"] is False
           and data["historical_m6_benchmark_mutated"] is False,
           "original one-case count, teacher variations or static status fabricated")
    demand(data["backend"]=="MixedRadialBackend","genuine released physical backend expected")
    demand(data["viewer_file"]=="p26-true-solid-sphere.html","expected source-authentic viewer")
    demand(data["independent_si_verification"]=="PENDING separate production-free checker",
           "native runner must not self-certify physics")
    demand(data["actual_chromium"]=="PENDING real browser visual gate",
           "native runner must not self-certify a browser test")
    states=data["native_states"]
    drawings=data["actual_rendered_geometry"]
    demand(isinstance(states,list) and len(states)==3
           and isinstance(drawings,list) and len(drawings)==3,
           "required source and true changed-parameter teacher SOLID worlds")
    observed={}
    for index,((role,R,q,c,constraint),state,draw) in enumerate(zip(CASES,states,drawings)):
        demand(state["index"]==index and state["case_role"]==role,
               "original/teacher scientific role lost")
        demand(state["physical_source_id"]=="Sphere A" and
               state["physical_shape_type"]=="SolidSphere" and
               state["source_time_axis"] is False,
               "stable SOURCE solid physical identity/motion scope")
        title=state["state_title"]
        demand(("PRINTED p26 SOLID sphere" in title if index==0
                else "TEACHER HYPOTHETICAL" in title),
               "source-given vs teacher-authored world title indistinguishable")
        geometry=state["physical_geometry"]
        demand(geometry["is_physically_solid"] is True,
               "source solid must not be drawn as hollow")
        close(geometry["radius_m"],R,label=f"physical radius {index}")
        close(geometry["inner_radius_m"],0,label=f"nonexistent inner face radius {index}")
        close(geometry["outer_radius_m"],R,label=f"outer solid face radius {index}")
        vector(geometry["center_m"],c,label=f"world centre {index}")
        demand(state["constraint_type"]==constraint,
               "source/teacher grounded-versus-isolated constraint changed")
        if constraint=="Grounded":
            demand(state["authored_charge_c"] is None,
                   "grounded zero response must not be reported fixed-charge")
        else:
            close(state["authored_charge_c"],q,
                  label=f"true input charge {index}",atol=1e-21)
        conductors=state["observed_conductor_results"]
        demand(len(conductors)==1 and conductors[0]["name"]=="Sphere A",
               "actual named solid conductor result missing")
        cs=conductors[0]
        close(cs["net_charge_c"],q,label=f"physical metal net Q {index}",atol=1e-21)
        close(cs["outer_surface_charge_c"],q,label=f"only physical outer charge {index}",atol=1e-21)
        close(cs["inner_surface_charge_c"],0,
              label=f"zero bookkeeping inner charge, NO actual face {index}")
        expected_inside=K*q/R
        expected_out=K*q/(1.5*R)
        field_out=K*q/(1.5*R)**2
        field_neg=-K*q/(2*R)**2
        sigma=q/(4*pi*R*R)
        close(cs["potential_v"],expected_inside,
              label=f"true metal equipotential {index}",atol=1e-10)
        sample=state["native_samples"]
        demand(set(sample)=={
            "center_potential_v",
            "half_radius_potential_v",
            "three_half_radius_potential_v",
            "metal_center_e_v_m",
            "metal_half_radius_e_v_m",
            "external_pos_x_e_v_m",
            "external_neg_x_e_v_m",
            "outer_sigma_c_m2",
        },"real native-observer schema completeness")
        for key,val in (
            ("center_potential_v",expected_inside),
            ("half_radius_potential_v",expected_inside),
            ("three_half_radius_potential_v",expected_out),
            ("outer_sigma_c_m2",sigma),
        ):
            close(sample[key],val,
                  label=f"independent Gauss {index} {key}",atol=1e-10 if "potential" in key else 1e-21)
        vector(sample["metal_center_e_v_m"],(0.,0.,0.),
               label=f"inside actual metal centre E {index}")
        vector(sample["metal_half_radius_e_v_m"],(0.,0.,0.),
               label=f"inside metal V(R/2) E {index}")
        vector(sample["external_pos_x_e_v_m"],(field_out,0.,0.),
               label=f"signed true exterior E at 3R/2 {index}",atol=1e-9)
        vector(sample["external_neg_x_e_v_m"],(field_neg,0.,0.),
               label=f"signed true exterior E at -2R {index}",atol=1e-9)
        demand(draw["kind"]=="sphere" and draw["name"]=="Sphere A"
               and draw["visible"] is True,
               f"drawn world shape not a physically real sphere {index}")
        close(draw["inner_radius_m"],0,label=f"rendered solid disk has no hollow ring {index}")
        close(draw["outer_radius_m"],R,label=f"rendered solid outer radius {index}")
        close(draw["distance_to_plane_m"],0,label=f"exact section through sphere centre {index}")
        # The meridian u axis is x, v axis is z, and all centre y=0.
        vector2=draw["center_m"]
        demand(len(vector2)==2,"2D meridian center missing")
        close(vector2[0],c[0],label=f"visual centre x {index}")
        close(vector2[1],0,label=f"visual centre z {index}")
        observed[role]={
            "independent_V_half_R_v":expected_inside,
            "independent_V_three_half_R_v":expected_out,
            "independent_signed_E_three_half_R_v_m":field_out,
            "independent_outer_sigma_c_m2":sigma,
            "independently_derived_external_self_energy_j":K*q*q/(2*R),
            "source_solid_face_count":1,
            "has_time_evolution":False,
        }
    demand(observed[CASES[0][0]]["independent_V_half_R_v"]>0
           and observed[CASES[1][0]]["independent_V_half_R_v"]<0
           and observed[CASES[2][0]]["independent_V_half_R_v"]==0,
           "genuine signed/grounded source revisions not distinguished")
    return {
        "schema":"conductor_sim.m36d_true_solid_independent_si.v1",
        "qualified_original_jee_solid_sphere_apparatus":1,
        "teacher_real_physical_revisions":2,
        "actual_solid_source_states_independently_checked":len(observed),
        "physics_results":observed,
        "unsupported_external_induction_claimed":False,
        "browser_legibility":"SEPARATE actual offline Chromium",
        "extrapolated_jee_30_success":"NOT measured",
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    native=json.loads(args.input.read_text(encoding="utf-8"))
    result=check(native)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,allow_nan=False,indent=2)+"\n",encoding="utf-8")
    print("M36d physically authentic SOLID source original JEE apparatus:",
          result["qualified_original_jee_solid_sphere_apparatus"])
    print("M36d actually solved signed/grounded teacher SOLID variants:",
          result["teacher_real_physical_revisions"])
    print("M36d source-free verified physical states:",
          result["actual_solid_source_states_independently_checked"])


if __name__=="__main__":
    main()
