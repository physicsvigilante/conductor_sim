"""M36c: authentic new original-JEE conductor solves and falsification tests."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from tools.m36c_jee_conductor_replay import run_all
from tools.m36c_jee_conductor_reference import verify


@pytest.fixture(scope="module")
def actual(tmp_path_factory):
    root=tmp_path_factory.mktemp("m36c-real-conductor-apparatus")
    manifest=run_all(root)
    return root,manifest


def test_original_static_conductor_apparatus_and_grouping(actual):
    folder,manifest=actual
    assert manifest["source_case_count"]==7
    assert len(manifest["native_source_apparatus"])==7
    assert len(manifest["historical_anchors_used"])==8
    assert len(manifest["non_success_controls"])==2
    assert all(row["native"] is True for row in manifest["native_source_apparatus"].values())
    assert json.loads((folder/"native-conductors.json").read_text())==manifest
    assert len([x for x in manifest["historical_anchors_used"]
                if x.startswith("p10-linked-")])==2
    assert "p10-linked-13+p10-linked-14" in manifest["native_source_apparatus"]


def test_all_seven_real_conductor_observations_match_independent_SI_reference(actual):
    _folder,manifest=actual
    result=verify(manifest)
    assert result["checked_distinct_source_apparatus"]==7
    assert result["source_question_anchors"]==8
    assert len(result["source_independent_reference_results"])==7
    assert result["honest_partial_or_refused_controls"]==2


def test_all_real_offline_conductor_visualizations_exist(actual):
    folder,manifest=actual
    for key,row in manifest["native_source_apparatus"].items():
        html=(folder/row["viewer_file"]).read_text(encoding="utf-8")
        assert "<html" in html.lower() and len(html)>1000,key
        assert "canvas" in html or "<svg" in html,key
    assert not (folder/"p70-three-cavities.html").exists()
    assert not (folder/"p08-multiple-21.html").exists()


@pytest.mark.parametrize("case,path,bad",[
    ("p03-single-27",("before",1,"inner_surface_charge_c"),1e-6),
    ("p03-single-27",("released_energy_j",),0.),
    ("p10-linked-13+p10-linked-14",("radial_energy_before_j",),99.),
    ("p10-linked-13+p10-linked-14",("before",1,"outer_surface_charge_c"),3e-6),
    ("p14-numerical-07",("after",1,"potential_v"),-500.),
    ("p26-conductor-potential",("native_observations",0,"native_value"),1.),
    ("p27-cavity-two-shells",("authored_states",0,"native_conductor_results",0,"inner_surface_charge_c"),1e-9),
    ("p69-cavity-central-vs-shifted",("native_observations",5,"native_value"),0.),
    ("p09-multiple-27",("native_induced_only_at_tangent_v",),0.),
    ("p27-cavity-two-shells",("native_observations",0,"native_value"),17.),
    ("p27-cavity-two-shells",("authored_states",0,"charges",0,"position_m"),[0.,0.,.02]),
    ("p27-cavity-two-shells",("native_observations",2,"conductor"),"outer"),
    ("p27-cavity-two-shells",("source_anchor",),"Unrelated illustration 8"),
    ("p69-cavity-central-vs-shifted",("native_observations",5,"theta_rad"),3.141592653589793),
    ("p69-cavity-central-vs-shifted",("native_observations",6,"theta_rad"),0.),
    ("p69-cavity-central-vs-shifted",("source_anchor",),"Completely different source"),
    ("p26-conductor-potential",("source_shape_fidelity",),"Literal solid sphere with no proxy"),
])
def test_refuse_fabricated_real_conductor_energies_charges_and_field(
    actual,case,path,bad
):
    _folder,manifest=actual
    forged=deepcopy(manifest)
    selected=forged["native_source_apparatus"][case]
    for key in path[:-1]:
        selected=selected[key]
    selected[path[-1]]=bad
    with pytest.raises((AssertionError,TypeError,ValueError,KeyError)):
        verify(forged)


def test_refuse_fake_irregular_local_3d_field_success(actual):
    _folder,manifest=actual
    forged=deepcopy(manifest)
    forged["non_success_controls"][1]["native"]=True
    with pytest.raises(AssertionError,match="cannot be fabricated"):
        verify(forged)


def test_refuse_fake_grounded_sphere_time_current_from_static_sweep(actual):
    _folder,manifest=actual
    forged=deepcopy(manifest)
    forged["non_success_controls"][0]["native"]=True
    with pytest.raises(AssertionError,match="NOT a prescribed approach-time electric current"):
        verify(forged)


def test_refuse_double_counting_linked_questions(actual):
    _folder,manifest=actual
    forged=deepcopy(manifest)
    row=forged["native_source_apparatus"].pop("p10-linked-13+p10-linked-14")
    forged["native_source_apparatus"]["p10-linked-13"]=row
    forged["native_source_apparatus"]["p10-linked-14"]=deepcopy(row)
    forged["source_case_count"]=8
    with pytest.raises(AssertionError,match="seven fixed physical apparatus"):
        verify(forged)


def test_static_connection_is_not_fake_time_domain(actual):
    _folder,manifest=actual
    forged=deepcopy(manifest)
    forged["native_source_apparatus"]["p03-single-27"]["kind"]="wire_discharge_time_series"
    with pytest.raises(AssertionError,match="not a time-dependent current"):
        verify(forged)


def test_true_p27_original_central_source_AB_not_a_displaced_source_substitute(actual):
    _folder,manifest=actual
    row=manifest["native_source_apparatus"]["p27-cavity-two-shells"]
    assert row["original_asked_observables"]==["V(A)","V(B)"]
    assert row["authored_states"][0]["charges"][0]["position_m"]==[0.,0.,0.]
    assert row["authored_states"][1]["charges"][0]["position_m"]==[0.,0.,.02]
    assert "TEACHER EXTENSION ONLY" in row["authored_states"][1]["title"]
    # The printed mathematical ideal-shell formula and the finite-thickness
    # numerical model are distinct; an actual source observer is evaluated.
    from tools.m36c_jee_conductor_reference import K
    q=1e-9;q1=2e-9;q2=3e-9
    ref_A=K*(q/.04+q1/.10+q2/.20)
    ref_B=K*((q+q1)/.15+q2/.20)
    assert row["original_formula_limit"]["V_A_zero_thickness_v"]==pytest.approx(ref_A)
    assert row["original_formula_limit"]["V_B_zero_thickness_v"]==pytest.approx(ref_B)
    actual_A=row["native_observations"][0]["native_value"]
    actual_B=row["native_observations"][1]["native_value"]
    assert actual_A==pytest.approx(ref_A,abs=.002)
    assert actual_B==pytest.approx(ref_B,abs=.002)
    assert actual_A!=ref_A and actual_B!=ref_B
    evidence=verify(manifest)["source_independent_reference_results"]["p27-cavity-two-shells"]
    assert evidence["kind"]=="genuine_source_central_A_B_plus_disclosed_teacher_extension"


def test_true_p26_source_solid_sphere_explicit_hollow_proxy_visible(actual):
    folder,manifest=actual
    case=manifest["native_source_apparatus"]["p26-conductor-potential"]
    assert "SOURCE SOLID sphere; HOLLOW shell" in case["source_shape_fidelity"]
    text=(folder/case["viewer_file"]).read_text(encoding="utf-8")
    assert 'id="original-solid-sphere-proxy-disclosure"' in text
    assert "Printed Illustration 1 is a SOLID conducting sphere" in text
    assert "HOLLOW SHELL PROXY" in text
    assert "The drawn hollow cavity is NOT in the source" in text


@pytest.mark.parametrize("case,path,bad", [
    ("p69-cavity-central-vs-shifted",("native_observations",5,"theta_rad"),3.141592653589793),
    ("p69-cavity-central-vs-shifted",("native_observations",6,"theta_rad"),0.),
    ("p27-cavity-two-shells",("native_observations",2,"conductor"),"outer"),
    ("p27-cavity-two-shells",("source_anchor",),"Not Illustration 4"),
])
def test_reviewer_exact_three_attribution_falsifications_fail_closed(
    actual,case,path,bad,
):
    _folder,manifest=actual
    forged=deepcopy(manifest)
    row=forged["native_source_apparatus"][case]
    for key in path[:-1]:
        row=row[key]
    row[path[-1]]=bad
    with pytest.raises(AssertionError):
        verify(forged)
