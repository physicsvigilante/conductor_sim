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
    assert json.loads((directory/"native-replays.json").read_text())==data


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
