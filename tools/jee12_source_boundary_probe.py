"""Developer inspection: workbook p11 Q28 and notes p70 Illustration 8/9.

These are deliberately different physical boundaries. Source p11 has an
irregular conductor with an exterior point q, not represented by the sealed
cavity Gauss model's local-field API. Notes p70 Illustration 8 has genuinely
irregular boundaries, while Illustration 9 draws three circular cross-sections
and gives uniform/nonuniform patterns. The latter is checked only under an
EXPLICIT teacher-authorized 3D spherical interpretation, not an inference
that original source specified full 3D radii and charge positions.
"""
from __future__ import annotations

from hashlib import sha256
from math import isclose, pi
from pathlib import Path
import json

from conductor_sim.cavity_gauss import (
    CavityCharge, GaussCavityAssembly, SealedCavity, solve_cavity_gauss,
)
from conductor_sim.model import EPSILON_0, FixedNetCharge
from conductor_sim.solver import UnsupportedProblem
from conductor_sim.spherical_multicavity import (
    NamedSourcePosition, SphericalVoid, SphericalMetalGeometry,
    SphericalSigmaModel, solve_spherical_sigma,
)
from tools.jee12_p69_source_probe import _independent_image_field_sigma, DIRECTIONS

WORKBOOK_SHA = "9dff3c02a9f1848d2c3d99cb1a6e7a44dc7f5a51f7a3a2b8cc92d10ce1fbca5d"
NOTES_SHA = "8d0d9b5cf821219058cfc4b7685a39b777e460944df44c3f9cffcb68157a60e5"


def _p11_partial_gauss_boundary():
    """Qualify ONLY complete-face charges; refuse original local E/sigma."""
    q1,q2,metal = +1e-9,-.4e-9,+.7e-9  # teacher SI; original source symbolic
    assembly = GaussCavityAssembly(
        "IrregularMetalQ", (
            SealedCavity("Cavity1", (CavityCharge("q1",q1),)),
            SealedCavity("Cavity2", (CavityCharge("q2",q2),)),
        ), FixedNetCharge(metal),EPSILON_0,
    )
    ledger = solve_cavity_gauss(assembly)
    record = ledger.as_dict()
    expected_outer = metal + q1 + q2
    if not isclose(record["outer_face_total_charge_c"],expected_outer,rel_tol=1e-13,abs_tol=1e-20):
        raise AssertionError("p11 conditional Gauss face total wrong")
    try:
        ledger.surface_charge_density_at("outer",(1.,0.,0.))
    except UnsupportedProblem:
        refused_local=True
    else:
        raise AssertionError("topology-only M39a generated an unqualified local exterior distribution")
    return dict(
        original_case_id="p11-linked-28", original_question=28,
        original_page=11, original_source_shape="irregular conductor with TWO cavity point charges and an EXTERNAL point charge q",
        source_given_metal_charge_symbol="Q", original_source_external_q_present=True,
        supported_subset="conditional whole-face Q_outer = Q + q1 + q2 from Gauss law",
        teacher_q1_c=q1,teacher_q2_c=q2,teacher_metal_q_c=metal,
        native_outer_face_total_c=record["outer_face_total_charge_c"],
        local_exterior_sigma_physically_refused=refused_local,
        exterior_q_effect_on_distribution_not_modelled=True,
        original_answer_completeness="partial Gauss total only, original exterior distribution unqualified",
    )


def _p70_spherical_interpretation():
    """Qualified teacher 3D spheres consistent with, not deduced from, drawing."""
    q1,q2,q3=1e-9,2e-9,3e-9
    data = (
        ("S1",(-1.7,0.,0.),.60,"q1",q1,(-1.7,0.,0.)),
        ("S2",(1.6,.9,0.),.72,"q2",q2,(1.82,1.00,.10)),
        ("S3",(.0,-1.9,0.),.58,"q3",q3,(.10,-2.15,.13)),
    )
    cavs=[];voids=[]
    for cid,centre,radius,sid,q,position in data:
        cavs.append(SealedCavity(cid,(CavityCharge(sid,q),)))
        voids.append(SphericalVoid(cid,centre,radius,(NamedSourcePosition(sid,position),)))
    assembly=GaussCavityAssembly("S4",tuple(cavs),FixedNetCharge(0.),EPSILON_0)
    geom=SphericalMetalGeometry((0.,0.,0.),4.0,tuple(voids))
    solution=solve_spherical_sigma(SphericalSigmaModel(assembly,geom))
    record=solution.as_dict([(cid,DIRECTIONS[0]) for cid,*_ in data] + [("outer",DIRECTIONS[0])])
    totals=record["gauss"]["cavities"]
    assert len(totals)==3
    sampling=[]
    for i,(cid,centre,radius,sid,q,position) in enumerate(data):
        gauss=totals[i]["induced_inner_face_total_charge_c"]
        if not isclose(gauss,-q,rel_tol=1e-13,abs_tol=1e-20):
            raise AssertionError("original p70 conditional -qi Gauss theorem not satisfied")
        rel=tuple(position[k]-centre[k] for k in range(3))
        cases=[]
        for n in DIRECTIONS:
            actual=solution.inner_surface_charge_density(cid,n)
            reference=_independent_image_field_sigma(q,radius,rel,n)
            if not isclose(actual,reference,rel_tol=3e-13,abs_tol=3e-23):
                raise AssertionError("real source+Kelvin image surface sigma disagreement")
            cases.append(dict(raw_ray=list(n),actual_sigma_c_per_m2=actual,independent_sigma_c_per_m2=reference))
        vals=[x["actual_sigma_c_per_m2"] for x in cases]
        actual_uniform = (max(vals)-min(vals) <= abs(q)/(4*pi*radius*radius)*1e-13)
        expected_uniform = (i==0)
        if actual_uniform != expected_uniform:
            raise AssertionError(f"S{i+1} original-source qualitative profile incorrect")
        sampling.append(dict(face_id=cid,source_id=sid,teacher_q_c=q,
                             teacher_center_m=list(centre),teacher_radius_m=radius,
                             teacher_source_position_m=list(position),
                             native_gauss_inner_face_c=gauss,
                             profile_uniform_at_sampled_rays=actual_uniform,
                             discrete_real_plus_kelvin_samples=cases))
    outer_expected=(q1+q2+q3)/(4*pi*4.0**2)
    outer_actual=[solution.outer_surface_charge_density(n) for n in DIRECTIONS]
    if any(not isclose(x,outer_expected,rel_tol=1e-13,abs_tol=3e-23) for x in outer_actual):
        raise AssertionError("unchanged S4 exterior uniform density wrong")
    return dict(original_page=70, printed_anchor="Illustration 9",
                original_source_face_totals=["S1=-q1", "S2=-q2", "S3=-q3", "S4=q1+q2+q3"],
                original_qualitative_pattern=["S1 inner uniform", "S2 inner nonuniform",
                                              "S3 inner nonuniform", "S4 outer uniform"],
                actual_3d_qualified_geometry="teacher-created nonoverlapping spherical voids and spherical outer metal",
                original_3d_shape_determined_from_2d_sketch=False,
                original_numeric_radii_and_charges_given=False,
                teacher_metal_net_charge_c=0.,
                original_answer_completeness="spherical case is a physically verified teacher instantiation, NOT a proven unique source geometry",
                actual_cavities=sampling,outer_sigma_c_per_m2=outer_actual[0])


def check(root:Path)->dict:
    pdfs=(("02_conductors_half_size.pdf",WORKBOOK_SHA),("POTENTIAL_CONDUCTORS1.pdf",NOTES_SHA))
    for name,digest in pdfs:
        if sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise AssertionError("original PDF byte identity wrong: "+name)
    history=json.loads((root/"docs/m36a_jee30_coverage.json").read_text(encoding="utf-8"))
    cases={x["id"]:x for x in history["cases"]}
    if (cases["p11-linked-28"]["pdf_page"]!=11 or
            cases["p11-linked-28"]["printed_anchor"]!="Linked Comprehension Q28" or
            cases["p70-three-cavities"]["pdf_page"]!=70 or
            cases["p70-three-cavities"]["printed_anchor"]!="Illustration 9"):
        raise AssertionError("historic source IDs changed")
    return dict(schema="conductor_sim.jee12_p11_p70_developer_boundary_probe.v1",
                source_status="DEVELOPER FROZEN PDF PAGE INSPECTIONS; NO NEW INDEPENDENT ORIGINAL-ANSWER VERDICT",
                workbook_p11=_p11_partial_gauss_boundary(),
                notes_p70_illustration8={"original_shapes":"irregular outer or inner physical boundary",
                    "local_sigma_qualified_by_current_spherical_M40a":False,
                    "source_answer_claimed":False},
                notes_p70_illustration9=_p70_spherical_interpretation(),
                original_workbook_source_items_newly_certified=0,
                no_unqualified_irregular_3d_solver_claim=True,
                no_fake_time_or_external_current=True)


if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path)
    args=ap.parse_args()
    doc=check(Path(__file__).resolve().parents[1])
    text=json.dumps(doc,indent=2,allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text,encoding="utf-8")
    else:
        print(text)
