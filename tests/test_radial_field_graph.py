from math import nextafter, sqrt

import pytest

from conductor_sim.axial_field_graph import AxialFieldGraph
from conductor_sim.radial_field_graph import RadialFieldGraph
from conductor_sim.solver import UnsupportedProblem


def test_q38_gradient_of_ten_times_radius():
    field = RadialFieldGraph(AxialFieldGraph(((0, -10), (8, -10))))
    point = (3, 4, -5)
    assert field.potential_at(point) == pytest.approx(10*sqrt(50))
    assert field.electric_field_at(point) == pytest.approx(tuple(-sqrt(2)*x for x in point))
    assert field.potential_at((0,0,0)) == 0
    with pytest.raises(UnsupportedProblem, match="no direction"):
        field.electric_field_at((0,0,0))


def test_shared_radial_bias_with_shifted_centre_and_inner_domain_boundary():
    graph = AxialFieldGraph(((0, 0), (2, 4), (5, 1)), 12)
    bias = AxialFieldGraph(((1, -1), (4, -1)), 3)
    baseline = RadialFieldGraph(graph, (5, -2, 1))
    field = baseline.superpose(RadialFieldGraph(bias, (5, -2, 1)))
    with pytest.raises(UnsupportedProblem, match="common centre"):
        baseline.superpose(RadialFieldGraph(bias))
    # At r=1: V=11+3; at r=3: subtract the trapezoids of E_sum on [1,2] and [2,3].
    assert field.potential_at((5,1,1)) == 14-2-2.5
    assert field.electric_field_at((5,1,1)) == (0,2,0)
    with pytest.raises(UnsupportedProblem, match="interval"):
        field.potential_at((5,-2,1))


def test_unit_radius_combines_constant_and_inverse_radius_at_large_translation():
    origin = 2**54
    field = RadialFieldGraph(AxialFieldGraph(((0, 3), (2, 1)), 9), (origin, 0, 0))
    assert field.electric_field_at((origin+1,0,0)) == (2,0,0)
    assert field.potential_at((origin+1,0,0)) == 6.5


def test_regular_centre_and_exact_irrational_radius_admission():
    field = RadialFieldGraph(AxialFieldGraph(((0, 0), (1, 2), (5, 10)), 9))
    assert field.electric_field_at((0,0,0)) == (0,0,0)
    assert field.electric_field_at((0,1,0)) == (0,2,0)
    assert field.electric_field_at((3,4,0)) == (6,8,0)
    assert field.potential_at((1,1,0)) == 7
    with pytest.raises(UnsupportedProblem, match="interval"):
        field.electric_field_at((3,nextafter(4., 5.),0))
