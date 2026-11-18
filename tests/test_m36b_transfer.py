"""M36b source-faithful open transfer: three NEW teacher-authored native worlds.

These do not replace or inflate the original seven source-apparatus denominator.
"""
from __future__ import annotations

import json
import copy

import pytest
from math import isclose

from tools.m36b_transfer import run_transfers
from tools.m36b_transfer_reference import verify


def test_three_actual_new_source_preserving_transfer_families():
    report=run_transfers()
    assert report["original_JEE_case_denominator_increment"]==0
    assert report["source_original_values_changed"] is False
    assert set(report["teacher_variations"])=={
        "charged_sheet","rigid_dipole","electric_plus_gravity",
    }
    assert all(row["reference_verified"] is True
               for row in report["teacher_variations"].values())
    assert "developer-executed" in report["reviewer_independence"]
    assert json.loads(json.dumps(report,allow_nan=False))==report


def test_changed_charged_sheet_locates_true_new_native_turning():
    case=run_transfers()["teacher_variations"]["charged_sheet"]
    assert case["kinetic_eV"]==175
    assert case["sigma_c_m2"]==-3.4e-6
    assert case["native_termination"]=="terminal-event"
    assert isclose(case["native_event_t_s"],case["independently_derived_t_s"],
                   rel_tol=0,abs_tol=2e-12)
    assert case["native_event_position_m"][0]>0


def test_changed_dipole_preserves_static_not_rotational_meaning():
    case=run_transfers()["teacher_variations"]["rigid_dipole"]
    assert case["angle_deg"]==65
    assert "NOT rotational dynamics" in case["kind"]
    assert isclose(case["native_work_j"],case["independent_delta_u_j"],
                   rel_tol=1e-10)
    assert isclose(case["native_torque_n_m"][2],case["independent_torque_z_n_m"],
                   rel_tol=1e-10)


def test_changed_proton_full_native_2d_electric_and_gravity():
    case=run_transfers()["teacher_variations"]["electric_plus_gravity"]
    assert case["duration_s"]==.017
    assert case["real_sample_count"]>=15
    for x,y in zip(case["native_end_position_m"],
                   case["independent_end_position_m"]):
        assert isclose(x,y,abs_tol=3e-14,rel_tol=1e-12)
    assert case["native_end_position_m"][0]>0
    assert case["native_end_position_m"][1]<0


def test_no_production_import_reference_rechecks_actual_teacher_variations():
    report=run_transfers()
    independently_recomputed=verify(report)
    assert independently_recomputed["genuinely_recomputed_teacher_physical_families"]==3
    assert independently_recomputed["original_source_apparatus_count_added"]==0


@pytest.mark.parametrize("family,field,change", [
    ("charged_sheet","native_event_t_s",1e-8),
    ("rigid_dipole","native_torque_n_m",[0.,0.,4e-9]),
    ("electric_plus_gravity","native_end_position_m",[0.08,0.,0.]),
])
def test_no_production_import_reference_rejects_forged_transfer_evidence(
    family,field,change,
):
    report=copy.deepcopy(run_transfers())
    report["teacher_variations"][family][field]=change
    with pytest.raises(AssertionError):
        verify(report)
