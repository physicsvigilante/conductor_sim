"""M36d true SOLID conducting-sphere JEE source viewer and native observations.

The printed p26 Illustration 1 asks symbolic V(R/2), V(3R/2).
We intentionally use a TRUE SolidSphere and existing public MixedRadial
solver/visualization, not M6's radial-observer-equivalent hollow shell.
Teacher numeric values and hypothetical models are NEVER source-given facts.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from math import hypot
import json
from pathlib import Path

from conductor_sim import (
    ElectrostaticProblem, FixedNetCharge, Grounded, MixedRadialBackend,
    SolidSphere,
)
from conductor_sim.visualization import SliceSpec, sample_sweep, write_visualization
from tools.m36a_jee30_audit import ROOT, audit_inventory, load_inventory

PDF = "POTENTIAL_CONDUCTORS1.pdf"
SOURCE_ID = "p26-conductor-potential"
SOURCE_ANCHOR = "Illustration 1"
PHYSICAL_NAME = "Sphere A"


def make_models():
    """One original-shape model and TWO conspicuously teacher-authored worlds."""
    source=ElectrostaticProblem(conductors=(
        SolidSphere(.10,FixedNetCharge(+1e-9),name=PHYSICAL_NAME),
    ))
    signed_revision=ElectrostaticProblem(conductors=(
        SolidSphere(.12,FixedNetCharge(-2e-9),center_m=(.025,0.,0.),
                    name=PHYSICAL_NAME),
    ))
    grounded_revision=ElectrostaticProblem(conductors=(
        SolidSphere(.08,Grounded(),name=PHYSICAL_NAME),
    ))
    return (
        ("PRINTED p26 SOLID sphere · teacher example R=0.10m, Q=+1nC",
         source,"source_original_shape_teacher_numeric_example"),
        ("TEACHER HYPOTHETICAL · same named SOLID, R=0.12m, Q=-2nC, shifted centre",
         signed_revision,"teacher_changed_radius_charge_and_centre"),
        ("TEACHER HYPOTHETICAL · same named SOLID, R=0.08m, grounded in empty space",
         grounded_revision,"teacher_grounded_no_external_sources"),
    )


def _sample(solution, sphere):
    c=sphere.center_m
    R=sphere.radius_m
    # Absolute world coordinates are real native observations. Never pre-fill
    # source fields with expected kQ/R or kQ/r mathematical references.
    probes={
        "center_potential_v":solution.potential_at(c),
        "half_radius_potential_v":solution.potential_at((c[0]+R/2,c[1],c[2])),
        "three_half_radius_potential_v":solution.potential_at((c[0]+1.5*R,c[1],c[2])),
        "metal_center_e_v_m":list(solution.electric_field_at(c)),
        "metal_half_radius_e_v_m":list(solution.electric_field_at((c[0]+R/2,c[1],c[2]))),
        "external_pos_x_e_v_m":list(solution.electric_field_at((c[0]+1.5*R,c[1],c[2]))),
        "external_neg_x_e_v_m":list(solution.electric_field_at((c[0]-2*R,c[1],c[2]))),
        "outer_sigma_c_m2":solution.surface_charge_density_at(
            sphere.name,"outer",(c[0]+R,c[1],c[2])),
    }
    return probes


def _render_with_truth(bundle,output):
    write_visualization(
        bundle,output,
        title="JEE original p26 Illustration 1 · TRUE SOLID conducting sphere",
    )
    source_banner=(
        '<section id="m36d-source-authentic-solid" '
        'style="margin-top:12px;padding:12px;background:#e8f7f5;'
        'color:#123638;border:2px solid #27857f">'
        '<strong>SOURCE GEOMETRY: genuinely SOLID conducting sphere.</strong> '
        'The original p26 Illustration 1 asks V(R/2) and V(3R/2) '
        'of a charged SOLID conductor, and zero E inside its metal. '
        'The first stored state uses REAL SolidSphere physics and a solid '
        'disk cross-section, NOT the historical hollow-shell proxy. '
        'Printed values R and Q are SYMBOLIC. The R=0.10m, Q=+1nC '
        'numbers are TEACHER illustration choices. '
        'Further stored states are labelled TEACHER HYPOTHETICAL: '
        'a changed signed-charge/radius/centre world and a grounded '
        'source-free world. The slider changes PRECOMPUTED STATIC states, '
        'NOT elapsed-time charge transfer.'
        '</section>'
    )
    html=output.read_text(encoding="utf-8")
    if "</header>" not in html:
        raise ValueError("could not insert genuinely solid-source educational label")
    output.write_text(html.replace("</header>",source_banner+"</header>",1),
                      encoding="utf-8")


def run_all(directory: Path) -> dict:
    audit_inventory()  # Historical actual original PDF byte/digest integrity.
    inventory=load_inventory(ROOT/"docs/m36a_jee30_coverage.json")
    source=next(case for case in inventory["cases"] if case["id"]==SOURCE_ID)
    if (source["pdf_page"]!=26 or source["source"]!=PDF
            or source["candidate_route"]!="supported_scoped"
            or source["printed_anchor"]!=SOURCE_ANCHOR):
        raise ValueError("original p26 source identity no longer admitted")
    directory.mkdir(parents=True,exist_ok=True)
    backend=MixedRadialBackend()
    models=make_models()
    source_cases=[]
    states=[]
    for index,(label,problem,role) in enumerate(models):
        if not backend.supports(problem):
            raise ValueError("real standalone SOLID source model is not admitted")
        sphere=problem.conductors[0]
        solution=backend.solve(problem)
        if sphere.inner_radius_m!=0 or len(problem.conductors)!=1 or problem.charges:
            raise ValueError("source solid geometry or physical admission changed")
        state={
            "index":index,
            "case_role":role,
            "state_title":label,
            "physical_source_id":sphere.name,
            "physical_shape_type":type(sphere).__name__,
            "physical_geometry":{
                "radius_m":sphere.radius_m,
                "center_m":list(sphere.center_m),
                "inner_radius_m":sphere.inner_radius_m,
                "outer_radius_m":sphere.outer_radius_m,
                "is_physically_solid":True,
            },
            "constraint_type":type(sphere.constraint).__name__,
            "authored_charge_c": (
                sphere.constraint.charge_c
                if isinstance(sphere.constraint,FixedNetCharge) else None
            ),
            "observed_conductor_results":[asdict(c) for c in solution.conductor_results],
            "native_samples":_sample(solution,sphere),
            "source_time_axis":False,
        }
        source_cases.append((label,problem))
        states.append(state)
    plane=SliceSpec.meridian(models[0][1],extent_m=.31,resolution=37)
    bundle=sample_sweep(source_cases,backend,plane=plane,line_steps=85,profile_samples=33)
    if len(bundle["frames"])!=3:
        raise AssertionError("real static source and two teacher state slices absent")
    geometry=[f["geometry"][0] for f in bundle["frames"]]
    for i,item in enumerate(geometry):
        if (item["kind"]!="sphere" or item["inner_radius_m"]!=0
                or not item["visible"]):
            raise AssertionError("the drawn source is not genuinely SOLID")
    output=directory/"p26-true-solid-sphere.html"
    _render_with_truth(bundle,output)
    report={
        "schema":"conductor_sim.m36d_real_solid_p26.v1",
        "printed_source":{"filename":PDF,"page":26,
                          "question":SOURCE_ANCHOR,
                          "original_shape":"solid conducting sphere",
                          "original_parameters":"symbolic radius R and net charge Q",
                          "original_asked":["V(R/2)","V(3R/2)"]},
        "original_source_apparatus_count":1,
        "teacher_hypothetical_worlds":2,
        "physical_time_evolution":False,
        "viewer_file":output.name,
        "backend":"MixedRadialBackend",
        "native_states":states,
        "actual_rendered_geometry":geometry,
        "independent_si_verification":"PENDING separate production-free checker",
        "actual_chromium":"PENDING real browser visual gate",
        "historical_m6_benchmark_mutated":False,
    }
    (directory/"native-solid-sphere.json").write_text(
        json.dumps(report,allow_nan=False,indent=2)+"\n",encoding="utf-8"
    )
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path("output/m36d"))
    args=parser.parse_args()
    data=run_all(args.output_dir)
    print("M36d original JEE source apparatus:",data["original_source_apparatus_count"])
    print("M36d different true physical teacher worlds:",data["teacher_hypothetical_worlds"])
    print("M36d authentic SOLID source geometry states:",len(data["native_states"]))
    print("NATIVE SOURCE ONLY; independent physical and browser checks are separate")


if __name__=="__main__":
    main()
