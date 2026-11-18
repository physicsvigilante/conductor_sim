"""M36d true SOLID conducting-sphere JEE source viewer and native observations.

The printed p26 Illustration 1 asks symbolic V(R/2), V(3R/2).
We intentionally use a TRUE SolidSphere and existing public MixedRadial
solver/visualization, not M6's radial-observer-equivalent hollow shell.
Teacher numeric values and hypothetical models are NEVER source-given facts.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from html import escape
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


def _source_answer_panel(states):
    """Render true NATIVE observation numbers, never prefilled answer formulas.

    Exactly one section is displayed at a time. The original-source section
    remains readable when JavaScript is disabled. The existing M5 state
    slider drives the source-scoped panel; it does not integrate physics.
    """
    def fmt(x):
        return format(x, ".11g")

    sections=[]
    for index,state in enumerate(states):
        samples=state["native_samples"]
        geom=state["physical_geometry"]
        net=state["observed_conductor_results"][0]["net_charge_c"]
        grounded=state["constraint_type"]=="Grounded"
        role=(
            "PRINTED ORIGINAL GEOMETRY · TEACHER numerical example"
            if index==0 else
            "TEACHER HYPOTHETICAL · NOT printed source data"
        )
        charge=(
            "Grounded · actual solved Q=0 C in empty space"
            if grounded else
            f"Isolated net Q={fmt(net*1e9)} nC (authored teacher value)"
        )
        number_values=(
            ("V(R/2)","V_half_R",samples["half_radius_potential_v"],"V"),
            ("V(3R/2)","V_three_half_R",samples["three_half_radius_potential_v"],"V"),
            ("|E(R/2)|","E_inside",
             sum(v*v for v in samples["metal_half_radius_e_v_m"])**.5,"V/m"),
            ("Outer-face sigma","sigma_outer",samples["outer_sigma_c_m2"],"C/m²"),
        )
        elements=[]
        for label,key,value,unit in number_values:
            elements.append(
                '<div class="m36e-answer-observation" '
                'style="padding:7px 10px;border:1px solid #cbd9e0;border-radius:6px">'
                f'<div style="font-size:13px;color:#345">{escape(label)}</div>'
                f'<strong data-m36e-number="{escape(key,quote=True)}" '
                f'data-native-value="{format(value,".17g")}">'
                f'{escape(fmt(value))} {escape(unit)}</strong>'
                '</div>'
            )
        hidden=" hidden" if index else ""
        cx,cy,cz=geom["center_m"]
        sections.append(
            f'<section data-m36e-state="{index}" data-m36e-role="{escape(state["case_role"],quote=True)}"{hidden}>'
            f'<p style="font-weight:650;margin:6px 0">{escape(role)}</p>'
            f'<p style="margin:5px 0">TRUE SOLID Sphere A: '
            f'R={escape(fmt(geom["radius_m"]))} m; '
            f'{escape(charge)}; '
            f'center=({escape(fmt(cx))}, {escape(fmt(cy))}, {escape(fmt(cz))}) m. '
            'Only one physical OUTER charge-density face exists.</p>'
            '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:7px">'
            + "".join(elements) + '</div></section>'
        )
    return (
        '<section id="m36e-source-observations" aria-live="polite" '
        'style="max-width:1220px;margin:16px auto;padding:14px 18px;'
        'background:#fff;border:1px solid #b4cad0;border-radius:9px;'
        'color:#142334;box-shadow:0 2px 12px #14233412">'
        '<h2 style="font-size:19px;margin:0">Original JEE p26: native potential observations</h2>'
        '<p style="margin:5px 0;color:#3d5462">The printed source supplies SYMBOLIC R and Q. '
        'All numerical inputs are teacher-chosen. Switching the stored-state slider '
        'updates these values to the selected genuine static solution, '
        'NOT a time trajectory.</p>'
        + "".join(sections) + '</section>'
    )


def _source_answer_sync_script():
    """Bind additional source-only readouts after the existing M5 JS starts."""
    return (
        '<script id="m36e-static-answer-synchronizer">'
        '(function(){'
        'const frameInput=document.getElementById("frame");'
        'const panels=[...document.querySelectorAll("[data-m36e-state]")];'
        'function update(){'
        'const index=Number(frameInput.value);'
        'panels.forEach((panel)=>{'
        'panel.hidden=Number(panel.dataset.m36eState)!==index;'
        '});'
        '}'
        'frameInput.addEventListener("input",update);'
        'update();'
        '})();'
        '</script>'
    )


def _render_with_truth(bundle,output,states):
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
        'A TRUE SOLID has ONE outer charged face and NO physical inner face; '
        'an inner charge of zero is only a result bookkeeping value. '
        'Printed values R and Q are SYMBOLIC. The R=0.10m, Q=+1nC '
        'numbers are TEACHER illustration choices. '
        'Further stored states are labelled TEACHER HYPOTHETICAL: '
        'a changed signed-charge/radius/centre world and a grounded '
        'source-free world. The slider changes PRECOMPUTED STATIC states, '
        'NOT elapsed-time charge transfer.'
        '</section>'
    )
    html=output.read_text(encoding="utf-8")
    if "</header>" not in html or "</body>" not in html:
        raise ValueError("source viewer is missing required HTML regions")
    if "<h2>Induced surface-charge profile</h2>" not in html:
        raise ValueError("cannot truthfully label isolated outer surface-charge density")
    if 'aria-label="Induced charge density along a great circle"' not in html:
        raise ValueError("isolated charge-density viewer label unexpectedly changed")
    # This refinement is scoped ONLY to this source-authentic single solid
    # sphere. Induced-charge density headings in other source viewers remain.
    html=html.replace(
        "<h2>Induced surface-charge profile</h2>",
        "<h2>Surface-charge density (one real outer face)</h2>",1,
    ).replace(
        'aria-label="Induced charge density along a great circle"',
        'aria-label="Surface-charge density along a great circle"',1,
    )
    html=html.replace(
        "</header>",source_banner+"</header>"+_source_answer_panel(states),1,
    ).replace(
        "</body>",_source_answer_sync_script()+"</body>",1,
    )
    output.write_text(html,encoding="utf-8")


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
    # Denser ACTUAL stored physical field samples improve circular edge
    # display. No invented interpolation or unverified subpixel surface E.
    plane=SliceSpec.meridian(models[0][1],extent_m=.31,resolution=65)
    bundle=sample_sweep(source_cases,backend,plane=plane,line_steps=85,profile_samples=33)
    if len(bundle["frames"])!=3:
        raise AssertionError("real static source and two teacher state slices absent")
    geometry=[f["geometry"][0] for f in bundle["frames"]]
    for i,item in enumerate(geometry):
        if (item["kind"]!="sphere" or item["inner_radius_m"]!=0
                or not item["visible"]):
            raise AssertionError("the drawn source is not genuinely SOLID")
    output=directory/"p26-true-solid-sphere.html"
    _render_with_truth(bundle,output,states)
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
