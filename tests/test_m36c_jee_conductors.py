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
