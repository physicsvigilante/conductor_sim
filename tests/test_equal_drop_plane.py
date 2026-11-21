from fractions import Fraction

import pytest

from conductor_sim.axial_field_graph import AxialFieldGraph
from conductor_sim.solver import UnsupportedProblem


def test_q55_locates_equal_drops_without_a_given_middle_plane():
    graph = AxialFieldGraph(((1, 2), (3, 18)), 1e100, (2,))
    tolerance = Fraction(1, 10**12)
    left, right = graph.equal_drop_bracket_m(3, 1, tolerance_m=tolerance)
    assert left**3 <= 14 <= right**3
    assert 0 <= right-left <= tolerance
    assert left-1 > 3-left


def test_negative_field_after_superposition_is_inverted_on_shared_domain():
    curved = AxialFieldGraph(((0, 0), (2, 4)), quadratic_v_m3=(1,))
    opposite = AxialFieldGraph(((0, -6), (1, -6), (2, -6)))
    model = curved.superpose(opposite)
    left, right = model.equal_drop_bracket_m(0, 2, tolerance_m=Fraction(1, 10**9))
    # Integral of x²-6 from zero is x³/3 - 6x. Half its total is -14/3.
    assert left**3 - 18*left >= -14 >= right**3 - 18*right


def test_interior_reversal_qualifies_only_on_clipped_monotone_interval():
    graph = AxialFieldGraph(((-2, 3), (2, 3)), quadratic_v_m3=(1,))
    with pytest.raises(UnsupportedProblem, match="reverses"):
        graph.equal_drop_bracket_m(-2, 2, tolerance_m=0.01)
    lo, hi = graph.equal_drop_bracket_m(1, 2, tolerance_m=0.001)
    assert 1 < lo <= hi < 2


def test_flat_target_refused_and_irrelevant_plateau_allowed():
    graph = AxialFieldGraph(((0, 2), (1, 0), (2, 0), (3, 2)))
    with pytest.raises(UnsupportedProblem, match="plateau"):
        graph.equal_drop_bracket_m(0, 3, tolerance_m=4)
    left, right = graph.equal_drop_bracket_m(1, 3, tolerance_m=Fraction(1, 1024))
    assert (left-2)**2 <= Fraction(1, 2) <= (right-2)**2
    with pytest.raises(UnsupportedProblem, match="outside"):
        graph.equal_drop_bracket_m(-1, 3, tolerance_m=1)
    with pytest.raises(ValueError):
        graph.equal_drop_bracket_m(0, 3, tolerance_m=0)


def test_exact_middle_plane_after_huge_translation():
    start = 2**64
    graph = AxialFieldGraph(((start, -1), (start+2, -1)))
    assert graph.equal_drop_bracket_m(start, start+2, tolerance_m=Fraction(1, 10)) == (start+1, start+1)
