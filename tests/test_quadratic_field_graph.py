from fractions import Fraction
from math import sqrt

import pytest

from conductor_sim.axial_field_graph import AxialFieldGraph
from conductor_sim.radial_field_graph import RadialFieldGraph
from conductor_sim.solver import UnsupportedProblem


def test_q55_equal_drops_in_the_increasing_positive_field():
    graph = AxialFieldGraph(((1, 2), (3, 18)), quadratic_v_m3=(2,))
    third = 15**(1/3)
    assert graph.field_work_j(1, 1, 2) == pytest.approx(14/3)
    assert graph.field_work_j(1, 2, third) == pytest.approx(14/3)
    assert third - 2 < 2 - 1
    assert graph.field_work_j(-3, 1, 2) == -14


def test_mixed_curvatures_and_unequal_grids_preserve_cancellation_and_gauge():
    offset = 2**56
    a = AxialFieldGraph(((offset, 0), (offset+2, 4), (offset+4, 0)),
                        2**100, (1, -1))
    b = AxialFieldGraph(((offset+1, -1), (offset+3, -1)), -(2**100), (-1,))
    total = a.superpose(b)
    assert total.quadratic_v_m3 == (0, -2)
    for dx in (1, Fraction(3, 2), 2, Fraction(5, 2), 3):
        x = offset+dx
        assert total.axial_field_v_m(x) == a.axial_field_v_m(x) + b.axial_field_v_m(x)
        # Independently integrate E_b = -(x-offset-2)^2 on [1,3].
        u = Fraction(dx)
        area_b = -((u-2)**3 + 1)/3
        area_a = u**3/3 if u <= 2 else Fraction(8, 3) + 4*(u-2) - (u-2)**3/3
        assert total.potential_v(x) == pytest.approx(float(-area_a-area_b))
    with pytest.raises(UnsupportedProblem):
        total.potential_v(offset)


def test_radial_quadratic_segment_at_irrational_radius_and_at_origin():
    profile = AxialFieldGraph(((0, 0), (1, 1), (3, 9)), 12, (1, 1))
    radial = RadialFieldGraph(profile, (1, 2, 3))
    assert radial.electric_field_at((1, 2, 3)) == (0, 0, 0)
    assert radial.electric_field_at((2, 3, 3)) == pytest.approx((sqrt(2), sqrt(2), 0))
    assert radial.potential_at((2, 3, 3)) == pytest.approx(12-2*sqrt(2)/3)
    assert radial.superpose(radial).potential_at((2, 3, 3)) == pytest.approx(24-4*sqrt(2)/3)


@pytest.mark.parametrize("coefficients", [(), (1, 2), (float("nan"),), (True,)])
def test_invalid_quadratic_segments(coefficients):
    with pytest.raises(ValueError):
        AxialFieldGraph(((0, 0), (1, 1)), quadratic_v_m3=coefficients)


def test_linear_graph_can_be_reauthored_with_a_different_number_of_knots():
    from dataclasses import replace

    original = AxialFieldGraph(((0, 1), (3, 2)))
    total = original.superpose(original)
    revised = replace(total, knots=((0, 2), (1, 4), (2, 1)))
    assert revised.field_work_j(1, 0, 2) == 5.5
