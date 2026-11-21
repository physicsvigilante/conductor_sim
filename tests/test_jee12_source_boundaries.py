"""Source p11 external q and p70 spherical/irregular physics boundaries."""
from pathlib import Path
from math import isclose

from tools.jee12_source_boundary_probe import check


def test_original_p11_external_q_cannot_acquire_fictional_local_gauss_density():
    data=check(Path(__file__).resolve().parents[1])
    p11=data["workbook_p11"]
    assert p11["original_page"]==11
    assert p11["original_source_external_q_present"] is True
    assert p11["local_exterior_sigma_physically_refused"] is True
    assert p11["exterior_q_effect_on_distribution_not_modelled"] is True
    expected=p11["teacher_metal_q_c"]+p11["teacher_q1_c"]+p11["teacher_q2_c"]
    assert isclose(p11["native_outer_face_total_c"],expected,rel_tol=1e-13)
    assert data["no_unqualified_irregular_3d_solver_claim"] is True


def test_original_p70_three_cavities_spherical_teacher_assumption_is_explicit():
    data=check(Path(__file__).resolve().parents[1])
    p70=data["notes_p70_illustration9"]
    assert p70["original_page"]==70
    assert p70["original_3d_shape_determined_from_2d_sketch"] is False
    assert p70["original_numeric_radii_and_charges_given"] is False
    assert p70["teacher_metal_net_charge_c"]==0.
    assert p70["original_source_face_totals"] == ["S1=-q1","S2=-q2","S3=-q3","S4=q1+q2+q3"]
    assert [c["profile_uniform_at_sampled_rays"] for c in p70["actual_cavities"]] == [True,False,False]
    assert all(len(c["discrete_real_plus_kelvin_samples"])==7 for c in p70["actual_cavities"])
    assert data["notes_p70_illustration8"]["local_sigma_qualified_by_current_spherical_M40a"] is False
    assert data["original_workbook_source_items_newly_certified"] == 0
