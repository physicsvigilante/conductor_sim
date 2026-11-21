"""Original workbook p1/p2/p3 source-given SI, with released physical references."""
from math import isclose
from pathlib import Path

from tools.jee12_workbook_numeric_probe import check


def test_original_workbook_Q8_Q19_Q27_real_source_answer_sign_and_physics():
    data=check(Path(__file__).resolve().parents[1])
    assert data["original_numeric_source_anchors_reinspected_by_developer"]==3
    assert data["original_answers_newly_independently_reviewed"]==0
    p1=data["p01_single_08"]
    assert p1["derived_horizontal_delta_m"] == 4.
    assert p1["original_multiple_choice_40eV"] is True
    assert p1["real_field_work_eV"] == -40.
    assert p1["physical_path_trajectory_or_time_claimed"] is False
    p2=data["p02_single_19"]
    assert isclose(p2["independent_stopping_distance_m"], .00044270939064,rel_tol=1e-13)
    assert p2["source_given_metal_sigma_c_per_m2"]==-2e-6
    assert p2["native_real_electron_stopping_event"] is True
    assert p2["original_multiple_choice_option"]=="0.44 mm"
    p3=data["p03_single_27"]
    assert 8.98 < p3["native_finite_metal_release_j"] < 9.
    assert p3["source_multiple_choice_option"]=="9 J"
    assert p3["finite_shell_thickness_is_teacher_authored"] is True
    assert p3["no_physical_wire_current_or_dissipation_time"] is True
