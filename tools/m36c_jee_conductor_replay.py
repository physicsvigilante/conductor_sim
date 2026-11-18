"""M36c: seven *real* original-source conductor apparatus, eight question anchors.

Each output is a native analytic conductor-solution observation. This is
static electrostatics: frame switches are NEVER wire-time or particle motion.
A second, source-free SI reference checker and Chromium run are separate gates.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from hashlib import sha256
from math import pi
from pathlib import Path

from conductor_sim import (
    ConnectedShellBackend, NestedShellBackend, ExternalSphereBackend,
    ElectrostaticProblem, FixedNetCharge, PointCharge, SphericalShell,
)
from conductor_sim.solver import SingularField
from conductor_sim.visualization import SliceSpec, sample_sweep, write_visualization
from conductor_sim.pdf_benchmark import load_case, measure, export_case
from conductor_sim.pdf_benchmark import PDF_NAME, PDF_SHA256
from examples.solid_sphere_connected import scenario as solid_connection
from examples.connected_shells import scenario as shell_connection
from m13c.pdf_cases import authored as external_case, export_pdf as export_external
from tools.m36a_jee30_audit import audit_inventory, load_inventory, ROOT

SOURCE_CASES = (
    "p03-single-27",
    "p10-linked-13+p10-linked-14",
    "p14-numerical-07",
    "p26-conductor-potential",
    "p27-cavity-two-shells",
    "p69-cavity-central-vs-shifted",
    "p09-multiple-27",
)
M36A_ANCHORS = (
    "p03-single-27", "p10-linked-13", "p10-linked-14",
    "p14-numerical-07", "p26-conductor-potential",
    "p27-cavity-two-shells", "p69-cavity-central-vs-shifted",
    "p09-multiple-27",
)
NOT_FULLY_MODELED = ("p08-multiple-21","p70-three-cavities")


def _results(solution):
    return [asdict(c) for c in solution.conductor_results]


def _connected(case_id: str, authored_case: str, directory: Path) -> dict:
    if case_id == "p14-numerical-07":
        plan = shell_connection(authored_case)
    else:
        plan = solid_connection(authored_case)
    native=ConnectedShellBackend().solve(plan)
    output=directory/(case_id+".html")
    bundle=native.visualization(resolution=19,line_steps=55,profile_samples=15)
    if len(bundle["frames"])!=2:
        raise AssertionError("conductor connection must show two actual equilibria")
    bundle["source_replay_case_id"]=case_id
    write_visualization(bundle,output,title="JEE conductor replay: "+case_id)
    bodies=[]
    for body in plan.initial.conductors:
        bodies.append({
            "name":body.name,
            "kind":type(body).__name__,
            "inner_radius_m":getattr(body,"inner_radius_m",None),
            "outer_radius_m":body.outer_radius_m,
        })
    sample_r=.15
    return {
        "source_case_id":case_id,
        "kind":"two_physical_static_equilibria_not_wire_current_time",
        "native":True,
        "viewer_file":output.name,
        "source_authored_case":authored_case,
        "source_input_bodies":bodies,
        "links":[{"mode":link.mode,"members":list(link.members)} for link in plan.links],
        "before":_results(native.before),
        "after":_results(native.after),
        "transfers":[asdict(x) for x in native.transfers],
        "radial_energy_before_j":native.radial_field_energy_before_j,
        "radial_energy_after_j":native.radial_field_energy_after_j,
        "released_energy_j":native.released_field_energy_j,
        "gap_radius_m":sample_r,
        "gap_field_before_v_m":list(native.before.electric_field_at((sample_r,0.,0.))),
        "gap_field_after_v_m":list(native.after.electric_field_at((sample_r,0.,0.))),
    }


def _notes_case(id_: str, name: str, directory: Path) -> dict:
    case=load_case(name)
    if not case.runnable or case.author is None:
        raise ValueError("selected JEE original source author unavailable: "+name)
    original=case.author()
    backend=NestedShellBackend()
    states=[]
    for state_index,(title,problem) in enumerate(original.states):
        if not backend.supports(problem):
            raise AssertionError("source-constrained nested conductor unsupported: "+name)
        sol=backend.solve(problem)
        states.append({
            "index":state_index,
            "title":title,
            "conductors":[{
                "name":c.name,"inner_radius_m":c.inner_radius_m,
                "outer_radius_m":c.outer_radius_m,
                "initial_constraint":type(c.constraint).__name__,
            } for c in problem.conductors],
            "charges":[{"position_m":list(q.position_m),"charge_c":q.charge_c}
                       for q in problem.charges],
            "native_conductor_results":_results(sol),
        })
    observations=[]
    for check in original.checks:
        label,problem=original.states[check.state]
        sol=backend.solve(problem)
        # Do NOT copy check.expected into recorded measured output.
        # It is an old exercise reference, not a fresh physical observation.
        observed=measure(sol,check,problem)
        observations.append({
            "state_index":check.state,
            "label":check.label,
            "kind":check.kind,
            "point_m":list(check.point_m) if check.point_m is not None else None,
            "conductor":check.conductor,
            "other_conductor":check.other_conductor,
            "theta_rad":check.theta_rad,
            "native_value":observed,
        })
    output=directory/(id_+".html")
    export_case(name,output,backend_name="analytical",resolution=19)
    if id_ == "p26-conductor-potential":
        # Original Illustration 1 shows a SOLID conducting sphere. The
        # historical M6 authoring example represents ONLY V(R/2), V(3R/2)
        # and E=0 by a finite-thickness shell proxy. Its physical interior
        # is hollow, and that MUST be conspicuous in the actual visible HTML.
        notice = (
            '<p id="original-solid-sphere-proxy-disclosure" style="margin:12px 0;'
            'padding:10px;background:#fff1d6;color:#342800;font-weight:600">'
            'SOURCE-FIDELITY LIMIT: Printed Illustration 1 is a SOLID conducting '
            'sphere. This display uses a HOLLOW SHELL PROXY (0.09–0.10 m) '
            'ONLY for the identical sampled radial V(R/2), V(3R/2) and '
            'zero interior E. The drawn hollow cavity is NOT in the source.'
            '</p>'
        )
        html = output.read_text(encoding="utf-8")
        if "</header>" not in html:
            raise ValueError("cannot disclose source-shape proxy in viewer header")
        output.write_text(html.replace("</header>",notice+"</header>",1),encoding="utf-8")
    if not output.is_file():
        raise AssertionError("existing JEE source viewer did not render")
    return {
        "source_case_id":id_,"source_authoring_id":name,
        "kind":"true_static_conductor_field_and_surface_observations",
        "native":True,"viewer_file":output.name,
        "source_pdf_page":case.page,"source_anchor":case.source_anchor,
        "source_shape_fidelity": (
            "SOURCE SOLID sphere; HOLLOW shell is a scoped radial V/E PROXY, NOT the drawn source geometry"
            if id_ == "p26-conductor-potential" else
            "qualified real spherical conductor geometry"
        ),
        "authored_states":states,"native_observations":observations,
    }


def _p27_original_A_B(directory: Path) -> dict:
    """Original Illustration 4: central q, potential at A and B.

    Historical M6's off-centre charge is a legitimate TEACHER extension, but
    it must not silently replace the printed central-charge A/B question.
    The source gives symbolic q, q1, q2, a, b, r and r1. The chosen numeric
    scale/probe radii and 1-micrometre shell widths are teacher examples.
    """
    original_id="p27-cavity-two-shells"
    source=load_case(original_id)
    if source.source_anchor != "Illustration 4" or source.page != 27:
        raise ValueError("actual source question identity changed")
    q,q1,q2=1e-9,2e-9,3e-9
    a,b=.10,.20
    metal_width=1e-6
    r_A,r_B=.04,.15
    shells=(
        SphericalShell(a-metal_width,a,FixedNetCharge(q1),name="inner"),
        SphericalShell(b-metal_width,b,FixedNetCharge(q2),name="outer"),
    )
    original=ElectrostaticProblem(
        charges=(PointCharge(q,(0.,0.,0.)),),conductors=shells)
    teacher=ElectrostaticProblem(
        charges=(PointCharge(q,(0.,0.,.02)),),conductors=shells)
    backend=NestedShellBackend()
    if not backend.supports(original) or not backend.supports(teacher):
        raise ValueError("real central/teacher variant outside qualified cavity regime")
    o=backend.solve(original)
    displaced=backend.solve(teacher)
    A=(r_A,0.,0.);B=(r_B,0.,0.)
    observations=[
        dict(state_index=0,label="Printed original V(A), central q in cavity",
             kind="potential",point_m=list(A),conductor=None,
             other_conductor=None,theta_rad=None,
             native_value=o.potential_at(A)),
        dict(state_index=0,label="Printed original V(B), intershell gap",
             kind="potential",point_m=list(B),conductor=None,
             other_conductor=None,theta_rad=None,
             native_value=o.potential_at(B)),
        dict(state_index=0,label="Printed original inner shell total",
             kind="shell_net_charge",point_m=None,conductor="inner",
             other_conductor=None,theta_rad=None,
             native_value=o.conductor_by_name("inner").net_charge_c),
        dict(state_index=0,label="Printed original outer shell total",
             kind="shell_net_charge",point_m=None,conductor="outer",
             other_conductor=None,theta_rad=None,
             native_value=o.conductor_by_name("outer").net_charge_c),
        dict(state_index=1,label="TEACHER off-centre extension, inner face -q",
             kind="inner_total",point_m=None,conductor="inner",
             other_conductor=None,theta_rad=None,
             native_value=displaced.conductor_by_name("inner").inner_surface_charge_c),
    ]
    states=[]
    for i,(label,p,solution) in enumerate((
        ("PRINTED ORIGINAL: central q; evaluate V(A), V(B)",original,o),
        ("TEACHER EXTENSION ONLY: shifted q, no original A/B formula",teacher,displaced),
    )):
        states.append(dict(
            index=i,title=label,
            conductors=[
                {"name":c.name,"inner_radius_m":c.inner_radius_m,
                 "outer_radius_m":c.outer_radius_m,
                 "initial_constraint":type(c.constraint).__name__}
                for c in p.conductors
            ],
            charges=[{"position_m":list(c.position_m),"charge_c":c.charge_c}
                     for c in p.charges],
            native_conductor_results=_results(solution),
        ))
    plane=SliceSpec.meridian(original,extent_m=.32,resolution=25)
    bundle=sample_sweep(
        [(states[0]["title"],original),(states[1]["title"],teacher)],
        backend,plane=plane,line_steps=65,profile_samples=17,
    )
    bundle["frames"][0]["notes"].append(
        "SOURCE ORIGINAL: central q. A=(0.04,0,0)m cavity; "
        "B=(0.15,0,0)m intershell gap are TEACHER numerical probe choices."
    )
    bundle["frames"][1]["notes"].append(
        "TEACHER off-centre extension ONLY; not original V(A)/V(B) formula."
    )
    output=directory/(original_id+".html")
    write_visualization(
        bundle,output,
        title="PDF p.27 Illustration 4 · original central V(A), V(B) + teacher extension",
    )
    # The source-asked observables must be visible to the teacher, not
    # concealed in a JSON sidecar or replaced by unrelated conductor totals.
    # The supplied textbook relationships are ideal ZERO-thickness limits;
    # finite thickness in the actually rendered geometry is disclosed.
    k=1/(4*pi*original.permittivity_f_per_m)
    ideal_A=k*(q/r_A+q1/a+q2/b)
    ideal_B=k*((q+q1)/r_B+q2/b)
    original_readout=(
        '<section id="p27-original-A-B" style="margin:12px 0;padding:12px;'
        'border:2px solid #116b6b;background:#e9f8f6;color:#142f36">'
        '<strong>PRINTED ORIGINAL — CENTRAL charge q, TWO STATIC conductor shells:</strong> '
        f'Actual finite-metal V(A) = {o.potential_at(A):.9g} V, '
        f'V(B) = {o.potential_at(B):.9g} V. '
        f'Printed ideal-thin formula comparison: V(A) = {ideal_A:.9g} V, '
        f'V(B) = {ideal_B:.9g} V. '
        'A is an illustrative cavity probe at r=0.04m; B is an illustrative '
        'gap probe at r1=0.15m. Source original radii/charges are symbolic; '
        'these numerical values and a 1-micrometre metal thickness are TEACHER choices. '
        'The second stored state moves the charge by 2cm as a TEACHER EXTENSION ONLY; '
        'it does NOT answer the original central-charge V(A)/V(B) formula.'
        '</section>'
    )
    html=output.read_text(encoding="utf-8")
    if "</header>" not in html:
        raise ValueError("cannot render original p27 A/B observables visibly")
    output.write_text(
        html.replace("</header>",original_readout+"</header>",1),
        encoding="utf-8",
    )
    # Record actual finite-thickness results separately from the zero-metal-
    # width PRINTED symbolic formulas; no old numeric benchmark expected values.
    return {
        "source_case_id":original_id,
        "source_authoring_id":"p27-central-original-and-disclosed-teacher-variant",
        "kind":"true_static_conductor_field_and_surface_observations",
        "native":True,
        "viewer_file":output.name,
        "source_pdf_page":27,
        "source_anchor":"Illustration 4",
        "source_shape_fidelity":"source central point q, two concentric thin conducting shells; separate shifted teacher extension",
        "original_asked_observables":["V(A)","V(B)"],
        "source_givens":"CENTRAL q, two nested thin spherical shells q1/q2, A in cavity at r, B in gap at r1; source is symbolic",
        "teacher_numerical_values":{
            "q_c":q,"q1_c":q1,"q2_c":q2,
            "radius_a_m":a,"radius_b_m":b,
            "A_world_m":list(A),"B_world_m":list(B),
            "authored_metal_thickness_m":metal_width,
            "displaced_teacher_source_m":[0.,0.,.02],
        },
        "original_formula_limit":{
            "V_A_zero_thickness_v": (
                (1/(4*pi*original.permittivity_f_per_m))*
                (q/r_A+q1/a+q2/b)
            ),
            "V_B_zero_thickness_v": (
                (1/(4*pi*original.permittivity_f_per_m))*
                ((q+q1)/r_B+q2/b)
            ),
            "scope":"printed symbolic zero-thickness limit; NOT the saved numerical observation",
        },
        "authored_states":states,
        "native_observations":observations,
    }


def _neutral_tangent(directory: Path) -> dict:
    id_="p09-multiple-27"
    spec=external_case(id_)
    solution=ExternalSphereBackend().solve(spec.problem)
    output=directory/(id_+".html")
    export_external(id_,output,resolution=15)
    sphere=spec.problem.conductors[0]
    source=spec.problem.charges[0]
    tangent=(3.,0.,0.)
    direct_distance=4.
    direct_k=1/(4*pi*spec.problem.permittivity_f_per_m)
    induced=solution.potential_at(tangent)-direct_k*source.charge_c/direct_distance
    return {
        "source_case_id":id_,
        "kind":"static_external_neutral_sphere_kelvin_induction_not_approach_time",
        "native":True,
        "viewer_file":output.name,
        "source_page":9,"source_question":"Multiple Correct Q27",
        "source_givens":{
            "sphere_radius_m":sphere.radius_m,
            "sphere_net_charge_c":solution.conductor_results[0].net_charge_c,
            "point_source_charge_c":source.charge_c,
            "point_source_position_m":list(source.position_m),
            "tangent_segment_ab_m":4.,
        },
        "native_sphere_potential_v":solution.conductor_results[0].potential_v,
        "native_total_potential_at_tangent_v":solution.potential_at(tangent),
        "native_induced_only_at_tangent_v":induced,
        "teacher_position_sweep":"Additional 5-point static sweep, not in original source",
    }


def run_all(directory: Path) -> dict:
    audit_inventory()   # Real original PDFs and 30 source identities admitted.
    inventory=load_inventory(ROOT/"docs/m36a_jee30_coverage.json")
    by_id={x["id"]:x for x in inventory["cases"]}
    if any(by_id[id_]["candidate_route"]!="supported_scoped" for id_ in M36A_ANCHORS):
        raise ValueError("original conductor case is not scoped-supported")
    if by_id[NOT_FULLY_MODELED[0]]["candidate_route"]!="partial":
        raise ValueError("unspecified current history cannot become a solved conductor case")
    if by_id[NOT_FULLY_MODELED[1]]["candidate_route"]!="unsupported":
        raise ValueError("unqualified irregular 3D multicavity cannot become a solved case")
    directory.mkdir(parents=True,exist_ok=True)
    recorded={
        "p03-single-27":_connected("p03-single-27","pdf-p03-q27",directory),
        "p10-linked-13+p10-linked-14":_connected(
            "p10-linked-13+p10-linked-14","pdf-p10-q13-q14",directory),
        "p14-numerical-07":_connected(
            "p14-numerical-07","pdf-p14-q7",directory),
        "p26-conductor-potential":_notes_case(
            "p26-conductor-potential","p26-conductor-potential",directory),
        "p27-cavity-two-shells":_p27_original_A_B(directory),
        "p69-cavity-central-vs-shifted":_notes_case(
            "p69-cavity-central-vs-shifted","p69-cavity-central-vs-shifted",directory),
        "p09-multiple-27":_neutral_tangent(directory),
    }
    if tuple(recorded)!=SOURCE_CASES:
        raise AssertionError("unexpected 7-apparatus pilot composition")
    for case in recorded.values():
        if not case["native"] or not (directory/case["viewer_file"]).is_file():
            raise AssertionError("missing real native original conductor source viewer")
    refusals=[
        {"id":"p08-multiple-21","kind":"PARTIAL","native":False,
         "reason":"Static grounded sphere position sweep is NOT a source-given approach-time current dQ/dt."},
        {"id":"p70-three-cavities","kind":"UNSUPPORTED","native":False,
         "reason":"Nonconcentric three-cavity LOCAL 3D E and surface sigma remain outside certified physical domain."},
    ]
    manifest={
        "schema":"conductor_sim.m36c_conductor_source_replays.v1",
        "source_cohort":"M36a original 29+1 PDF apparatus",
        "source_case_count":7,
        "historical_anchors_used":list(M36A_ANCHORS),
        "native_source_apparatus":recorded,
        "non_success_controls":refusals,
        "physical_reference":"PENDING separate standard-library SI reference checker",
        "actual_browser_inspection":"PENDING actual offline Chromium",
        "source_givens_vs_teacher":"Example authoring values must be explicit; two linked p10 questions count as ONE physical apparatus.",
        "textbook_answer_completeness":"NOT attempted",
    }
    (directory/"native-conductors.json").write_text(
        json.dumps(manifest,indent=2,allow_nan=False,ensure_ascii=False)+"\n",encoding="utf-8")
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path("output/m36c"))
    args=parser.parse_args()
    data=run_all(args.output_dir)
    print("M36c actual unique conductor apparatus:",data["source_case_count"])
    print("M36c historical source-question anchors:",len(data["historical_anchors_used"]))
    print("M36c truthfully limited/unsupported controls:",len(data["non_success_controls"]))
    print("NATIVE ONLY; independent physics and real Chromium not self-certified")


if __name__=="__main__":
    main()
