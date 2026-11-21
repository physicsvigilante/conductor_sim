from fractions import Fraction

import pytest

from conductor_sim.parallel_sheets import ParallelSheetField
from conductor_sim.solver import UnsupportedProblem


def test_q49_left_sheet_has_larger_positive_density():
    # Teacher example with the printed qualitative slopes. Left is sigma_2.
    profile = ((-2, 0), (-1, 3), (1, 1), (2, -2))
    model = ParallelSheetField.from_potential_segments(profile, permittivity_f_m=2)
    assert model.sheets == ((-1, 8), (1, 4))
    assert model.bias_v_m == 0
    for x, v in profile:
        assert model.potential_at((x,0,0)) == v
    assert model.gap_graph(-0.5, 0.5).axial_field_v_m(0) == 1


def test_reconstruction_preserves_gauge_and_signed_work_with_external_bias():
    base = 2**80
    offset = 2**54
    xs = [offset+i for i in (0, 1, 3, 4)]
    values = [base+i for i in (0, -2, 4, 1)]
    model = ParallelSheetField.from_potential_segments(zip(xs, values), permittivity_f_m=3)
    assert model.sheets == ((xs[1], -15), (xs[2], 18))
    assert model.bias_v_m == Fraction(5, 2)
    assert model.field_work_j(-1, (xs[0],0,0), (xs[-1],0,0)) == 1
    assert model.electric_field_at((xs[-1]+1,0,0))[0] == 3
    with pytest.raises(UnsupportedProblem, match="jumps"):
        model.electric_field_at((xs[2],0,0))


def test_collinear_knots_do_not_introduce_spurious_sheets():
    model = ParallelSheetField.from_potential_segments(((0,3), (1,1), (2,-1)))
    assert model.sheets == ()
    assert model.electric_field_at((1,0,0)) == (2,0,0)
    for invalid in ((), ((0,0),), ((1,0),(1,1)), ((2,0),(1,0))):
        with pytest.raises(ValueError):
            ParallelSheetField.from_potential_segments(invalid)
