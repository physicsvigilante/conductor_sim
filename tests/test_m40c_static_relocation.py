"""M40c: genuine static-source models, independent images, refusal and honest display."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from conductor_sim.cavity_gauss import GaussCavityAssembly, SealedCavity, CavityCharge
from conductor_sim.model import FixedNetCharge, Grounded, EPSILON_0
from conductor_sim.solver import UnsupportedProblem
from conductor_sim.spherical_multicavity import (
    SphericalSigmaModel, SphericalMetalGeometry, SphericalVoid,
    NamedSourcePosition,
)
from conductor_sim.spherical_sigma_relocation import (
    StaticSourceRelocation, StaticSigmaComparison,
    relocate_spherical_source, compare_static_spherical_relocation,
)
from conductor_sim.spherical_sigma_relocation_viewer import render_static_sigma_comparison
from tools.m40a_spherical_example import teacher_model
from tools.m40a_spherical_reference import check_record
from tools.m40c_static_reference import check_static_comparison
from tools.m40c_static_example import run_all


def scenario():
    return teacher_model(), StaticSourceRelocation("NeutralButNonuniform", "plus_A",(-1.45,.04,.01))


def test_teacher_examples_recomputed_and_oracle(tmp_path: Path):
    out=run_all(tmp_path)
    assert check_static_comparison(out)["inner_samples_checked"]==36
    assert len(out["baseline_voids"])==3
    assert out["relocation"]["source_id"]=="plus_A"
    assert out["original_workbook_items_newly_solved"]==0
    assert (tmp_path/"m40c-static-teacher.html").read_text()==render_static_sigma_comparison(
        compare_static_spherical_relocation(*scenario()))
    assert json.loads((tmp_path/"m40c-static-teacher.json").read_text())==out
    assert all(check_record(r)["cavities_checked"]==3
               for record in out["m40a_v2_witness_records"].values() for r in record)


def test_original_model_immutable_and_gauss_faces_unchanged():
    model,change=scenario()
    original=model.geometry
    result=compare_static_spherical_relocation(model,change)
    d=result.as_dict()
    assert model.geometry==original
    assert model.geometry != result.revised.geometry
    assert d["gauss"]["cavities"][0]["induced_inner_face_total_charge_c"]==0.0
    assert d["outer_uniform_sigma_c_per_m2"] !=0
    assert d["cavities"][0]["samples"][0]["sigma_baseline_c_per_m2"] != d["cavities"][0]["samples"][0]["sigma_revised_c_per_m2"]
    for cav in d["cavities"][1:]:
        assert all(x["delta_status"]=="unchanged_exact" and x["delta_c_per_m2"]==0.0
                   for x in cav["samples"])
    assert d["baseline_voids"][0]["source_positions"][0]["position_m"]==[-1.4,0.,0.]
    assert d["revised_voids"][0]["source_positions"][0]["position_m"]==[-1.45,.04,.01]


def test_distinct_static_electrical_constraints_requalified_separately():
    base,change=scenario()
    iso=compare_static_spherical_relocation(base,change).as_dict()
    gr=compare_static_spherical_relocation(
        replace(base,gauss_assembly=replace(base.gauss_assembly,constraint=Grounded())),change).as_dict()
    assert iso["outer_uniform_sigma_c_per_m2"]!=0
    assert gr["outer_uniform_sigma_c_per_m2"]==0
    for left,right in zip(iso["cavities"],gr["cavities"]):
        assert left["samples"]==right["samples"]
    assert check_static_comparison(gr)["one_source_relocated"]


def test_genuinely_near_wall_raw_direction_preserved():
    radius=.85; gap=1.5e-8
    model=SphericalSigmaModel(
        GaussCavityAssembly("Metal",(SealedCavity("A",(CavityCharge("q",+3.1e-9),)),),FixedNetCharge(+1.1e-9),EPSILON_0),
        SphericalMetalGeometry((0.,0.,0.),5.,(
            SphericalVoid("A",(0.,0.,0.),radius,(
                NamedSourcePosition("q",(radius*(1.-gap),0.,0.)),)),)))
    change=StaticSourceRelocation("A","q",(radius*(1.-2.0e-8),0.,0.))
    rays=((1.+9e-13,0.,0.),(0.,1.,0.))
    doc=compare_static_spherical_relocation(model,change,rays=rays).as_dict()
    assert doc["cavities"][0]["samples"][0]["input_direction"][0]==1.+9e-13
    assert doc["cavities"][0]["samples"][0]["unit_normal_display_only"]==[1.,0.,0.]
    assert doc["cavities"][0]["samples"][0]["delta_status"]=="resolved_difference"
    assert check_static_comparison(doc)["inner_samples_checked"]==4


def test_equal_density_at_one_ray_is_unresolved_not_theorem_zero():
    m=SphericalSigmaModel(
        GaussCavityAssembly("M",(SealedCavity("A",(CavityCharge("q",+1e-9),)),),FixedNetCharge(1e-9),EPSILON_0),
        SphericalMetalGeometry((0.,0.,0.),5.,(SphericalVoid("A",(0.,0.,0.),1.,(
            NamedSourcePosition("q",(.3,0.,0.)),)),)))
    out=compare_static_spherical_relocation(m,StaticSourceRelocation("A","q",(-.3,0.,0.)),rays=((0.,1.,0.),(1.,0.,0.))).as_dict()
    assert out["cavities"][0]["samples"][0]["delta_status"]=="unresolved_numerical_difference"
    assert out["cavities"][0]["samples"][0]["delta_c_per_m2"] is None
    assert out["cavities"][0]["samples"][1]["delta_status"]=="resolved_difference"
    assert check_static_comparison(out)["one_source_relocated"]


@pytest.mark.parametrize("cavity,ident,coords",[
    ("Unknown","plus_A",(-1.45,.04,.01)),
    ("NeutralButNonuniform","missing",(-1.45,.04,.01)),
    ("NegativeCavity","plus_A",(-1.45,.04,.01)),
    ("NeutralButNonuniform","plus_A",(-1.4,0.,0.)),
    ("NeutralButNonuniform","plus_A",(-1.05,0.,0.)), # nearly on assigned wall
    ("NeutralButNonuniform","plus_A",(5000.,0.,0.)),
    ("NeutralButNonuniform","plus_A",(False,0.,0.)),
    ("NeutralButNonuniform","plus_A",(float("nan"),0.,0.)),
    ("NeutralButNonuniform","plus_A",(float("inf"),0.,0.)),
    ("NeutralButNonuniform","plus_A",(0.,0.)),
    ("NeutralButNonuniform",False,(-1.45,.04,.01)),
    (False,"plus_A",(-1.45,.04,.01)),
])
def test_invalid_or_stale_source_relocations_refused(cavity,ident,coords):
    m=teacher_model()
    with pytest.raises((ValueError,TypeError,UnsupportedProblem)):
        relocate_spherical_source(m,StaticSourceRelocation(cavity,ident,coords))


@pytest.mark.parametrize("rays",[(),((0.,0.,0.),),((2.,0.,0.),),((False,0.,0.),),
                                  ((float("nan"),0.,0.),),((float("inf"),0.,0.),),
                                  ((1.,0.),),tuple((1.,0.,0.) for _ in range(25))])
def test_invalid_surface_rays_refused(rays):
    with pytest.raises((ValueError,TypeError,UnsupportedProblem)):
        compare_static_spherical_relocation(*scenario(),rays=rays)


def test_tampered_saved_comparison_cannot_reuse_native_evidence():
    result=compare_static_spherical_relocation(*scenario())
    altered=replace(result,revised=result.baseline)
    with pytest.raises(ValueError,match="requested static relocation"):
        altered.as_dict()
    wrong=replace(result,instruction=StaticSourceRelocation(
        "NeutralButNonuniform","plus_A",(-1.48,.03,.01)))
    with pytest.raises(ValueError,match="requested static relocation"):
        wrong.as_dict()


@pytest.mark.parametrize("path,wrong",[
    (("source_role",),"source_jee_answer"),
    (("time_evolution_defined",),True),
    (("no_chronology",),"STATIC but actually elapsed time is simulated with a current"),
    (("original_workbook_items_newly_solved",),1),
    (("relocation","source_id"),"minus_A"),
    (("relocation","original_position_m",0),True),
    (("relocation","revised_position_m",0),-1.4),
    (("cavities",0,"samples",0,"delta_status"),"unchanged_exact"),
    (("cavities",0,"samples",0,"delta_c_per_m2"),False),
    (("cavities",0,"samples",0,"sigma_revised_c_per_m2"),0),
    (("cavities",0,"whole_face_induced_charge_c"),1e-9),
    (("m40a_v2_witness_records","revised",0,"samples",0,"sigma_c_per_m2"),False),
    (("m40a_v2_witness_records","revised",0,"samples",0,"input_direction",0),False),
])
def test_forged_record_claims_are_refused(path,wrong):
    out=deepcopy(compare_static_spherical_relocation(*scenario()).as_dict())
    node=out
    for key in path[:-1]:node=node[key]
    node[path[-1]]=wrong
    with pytest.raises((ValueError,TypeError,KeyError,OverflowError)):
        check_static_comparison(out)


def test_unsupported_extra_claim_refused():
    out=compare_static_spherical_relocation(*scenario()).as_dict()
    out["source_force_newly_solved"]=True
    with pytest.raises(ValueError,match="invented additional physical capabilities"):
        check_static_comparison(out)


def test_witness_omission_of_empty_cavity_refused():
    data=deepcopy(compare_static_spherical_relocation(*scenario()).as_dict())
    data["cavities"].pop()
    data["m40a_v2_witness_records"]["baseline"].pop(-2)
    data["m40a_v2_witness_records"]["revised"].pop(-2)
    with pytest.raises(ValueError,match="all original cavity"):
        check_static_comparison(data)


def test_viewer_shows_two_static_scenes_true_source_xyz_and_no_network():
    page=render_static_sigma_comparison(compare_static_spherical_relocation(*scenario()))
    assert page.count('role="img"')==2
    assert page.count('id="m40c-source"')==1
    assert "STATIC" in page and "not motion" in page
    assert "revised (-1.45, 0.04, 0.01) m" in page
    assert "Unresolved numerical difference" not in page or "Unresolved" in page
    assert "not area" not in page.lower() or "area" in page.lower()
    assert "default-src &#39;none&#39;" in page
    assert "script-src &#39;none&#39;" in page
    assert "<script" not in page.lower()
    assert "0 C" in page
    assert "ZERO NEW ORIGINAL JEE" in page
