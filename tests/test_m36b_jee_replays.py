"""M36b: real source-anchored native observables vs no-production-import physics."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.m36b_jee_replay import run_all
from tools.m36b_jee_reference import check_native


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    directory=tmp_path_factory.mktemp("m36b-source-replays")
    data=run_all(directory)
    return directory,data


def test_seven_real_source_apparatus_and_separate_3d_refusal(native):
    directory, data=native
    assert data["source_apparatus_count"]==7
    assert data["unsupported_refusal_count"]==1
    assert data["explicit_refusal"]["native"] is False
    assert data["explicit_refusal"]["viewer_file"] is None
    assert len(data["source_apparatus"])==7
    assert json.loads((directory/"native-replays.json").read_text(encoding="utf-8"))==data


def test_independent_reference_checks_every_actual_native_case(native):
    _directory,data=native
    report=check_native(data)
    assert report["independent_math_verified_scoped_apparatus"]==7
    assert len(report["independently_checked_cases"])==7
    assert report["justified_local_3d_refusals_not_simulations"]==1
    assert report["browser_inspection"]=="separate real Chromium gate required"


def test_actual_html_written_for_every_admissible_source(native):
    directory,data=native
    for name,case in data["source_apparatus"].items():
        doc=directory/case["viewer_file"]
        assert doc.is_file(),name
        html=doc.read_text(encoding="utf-8")
        assert len(html)>1000 and "<html" in html.lower()
    assert not (directory/"p70-three-cavities.html").exists()


@pytest.mark.parametrize("id_,path,value", [
    ("p02-single-19",("events",0,"time_s"),5e-8),
    ("p03-single-30",("events",0,"kinetic_energy_j"),2e-14),
    ("p05-single-47",("observed_delta_u_j",0),1.),
    ("p12-linked-40",("observed_field_work_j",),1.),
    ("p15-archive-02",("saved_samples",5,"position_m",0),100.),
    ("p02-single-15",("after",0,"charge_c"),1e-7),
    ("jee2021-p1-q09-q10-equipotential",("sampled_contour_world_xy_m",0,0),300.),
])
def test_independent_reference_rejects_altered_saved_physics(native,id_,path,value):
    _directory,data=native
    bad=copy.deepcopy(data)
    obj=bad["source_apparatus"][id_]
    for key in path[:-1]:
        obj=obj[key]
    obj[path[-1]]=value
    with pytest.raises((AssertionError,KeyError,TypeError,ValueError)):
        check_native(bad)


def test_fake_multicavity_simulation_cannot_count(native):
    _directory,data=native
    bad=copy.deepcopy(data)
    bad["explicit_refusal"]["native"]=True
    with pytest.raises(AssertionError,match="must explicitly refuse"):
        check_native(bad)


def test_witnessed_static_dipole_is_not_fake_motion(native):
    _directory,data=native
    bad=copy.deepcopy(data)
    bad["source_apparatus"]["p05-single-47"]["kind"]="time_resolved_free_dipole_rotation"
    with pytest.raises(AssertionError,match="static dipole work"):
        check_native(bad)


def test_observed_ring_transverse_field_must_not_be_dropped(native):
    _directory,data=native
    bad=copy.deepcopy(data)
    bad["source_apparatus"]["p12-linked-40"]["transverse_profile_max_abs_v_m"]=0.
    with pytest.raises(AssertionError,match="nonzero transverse force"):
        check_native(bad)


def test_unsupported_source_cannot_be_silent_extra_success(native):
    _directory,data=native
    bad=copy.deepcopy(data)
    bad["source_apparatus"]["p70-three-cavities"]=copy.deepcopy(
        bad["source_apparatus"]["p02-single-15"])
    bad["source_apparatus_count"]=8
    with pytest.raises(AssertionError,match="must preserve all seven"):
        check_native(bad)


@pytest.mark.parametrize("case_id,path,forgery,expected_message", [
    ("p02-single-19",("events",0,"kinetic_energy_j"),1e-18,
     "stopping kinetic energy"),
    ("p02-single-19",("events",0,"velocity_m_s",1),1000.,
     "full velocity vector"),
    ("p02-single-19",("early_surface_variant","event_time_s"),1.19429886696e-10,
     "FIRST metal-surface root time"),
    ("p12-linked-40",("observed_source_work_components",0,"field_work_j"),999.,
     "independent signed work from ring A"),
    ("p12-linked-40",("start_electric_field_v_m",0),-1e9,
     "independent E vector at ring centre A"),
    ("jee2021-p1-q09-q10-equipotential",("source_masks",0,"center_m",0),.5,
     "source mask centre 0"),
    ("p02-single-15",("teacher_transfer_variant","charge_transfers_c",0),0.,
     "independent signed Q transfer 0"),
    ("p02-single-15",("teacher_transfer_variant","released_field_energy_j"),0.,
     "independent positive released field energy"),
])
def test_reviewer_original_forged_observations_rejected(
    native,case_id,path,forgery,expected_message,
):
    """Preserve red-before/green-after coverage of actual review #01 F01–F05.

    These adversarial states were accepted by the previous reference checker
    on genuine archived native data; production physics itself was correct.
    """
    _directory,data=native
    bad=copy.deepcopy(data)
    selected=bad["source_apparatus"][case_id]
    for key in path[:-1]:
        selected=selected[key]
    selected[path[-1]]=forgery
    with pytest.raises(AssertionError,match=expected_message):
        check_native(bad)


def test_first_contact_real_event_has_full_native_state(native):
    _directory,data=native
    e=data["source_apparatus"]["p02-single-19"]["early_surface_variant"]
    assert e["actual_native_contact_event"]["id"]=="sheet"
    assert e["actual_native_contact_event"]["full_stop"] is False
    assert len(e["actual_native_contact_event"]["position_m"])==3
    assert len(e["actual_native_contact_event"]["velocity_m_s"])==3
    assert e["event_time_s"]==e["actual_native_contact_event"]["time_s"]


def test_far_sphere_teacher_variation_moves_actual_charge(native):
    _directory,data=native
    card=data["source_apparatus"]["p02-single-15"]
    v=card["teacher_transfer_variant"]
    assert "teacher-authored" in v["origin"]
    assert [row["charge_c"] for row in v["before"]]==[5e-9,1e-9]
    assert [row["charge_c"] for row in v["after"]]==pytest.approx([2e-9,4e-9])
    assert v["charge_transfers_c"]==pytest.approx([-3e-9,3e-9])
    assert v["released_field_energy_j"]>0
    assert data["source_apparatus_count"]==7  # Variation != new original question.
