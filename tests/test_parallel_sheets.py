from fractions import Fraction

import pytest

from conductor_sim.axial_field_graph import AxialFieldGraph
from conductor_sim.parallel_sheets import ParallelSheetField
from conductor_sim.solver import UnsupportedProblem


def test_q46_equal_area_oppositely_charged_plates():
    # Teacher normalization d = 1 m, Q/(A epsilon) = 1 V/m and V0 = 8 V.
    field = ParallelSheetField(((1, 2), (2, -10)), permittivity_f_m=2, reference_potential_v=8)
    assert [field.potential_at((x,0,0)) for x in range(4)] == [8,6,3,5]
    assert [field.electric_field_at((x,0,0))[0] for x in (0,1.5,3)] == [2,3,-2]
    with pytest.raises(UnsupportedProblem, match="jumps"):
        field.electric_field_at((2,8,9))
    assert field.field_work_j(-1, (0,2,5), (3,4,-2)) == -3


def test_q48_motion_parallel_to_sheet_leaves_energy_unchanged():
    field = ParallelSheetField(((0, 8e-9),), reference_potential_v=7)
    a, b, c = (10,2,0), (10,0,2), (10,0,0)
    assert field.potential_at(a) == field.potential_at(b) == field.potential_at(c)
    assert field.field_work_j(3e-9, a, b) == field.field_work_j(3e-9, b, c) == 0


def test_composable_gap_and_refusal_at_discontinuities():
    field = ParallelSheetField(((0, 2), (3,-1)), permittivity_f_m=1, bias_v_m=2, reference_x_m=2)
    graph = field.gap_graph(Fraction(1,2), 2)
    assert graph.potential_v(1) == field.potential_at((1,0,0))
    assert graph.axial_field_v_m(1) == 3.5
    total = graph.superpose(AxialFieldGraph(((1,-1),(4,2))))
    assert total.field_work_j(2, 1, 2) == 6
    for ends in ((0,1),(-1,1),(1,3)):
        with pytest.raises(UnsupportedProblem, match="charged sheet"):
            field.gap_graph(*ends)


def test_coincident_cancellation_and_gauge_independent_work():
    x = 2**54
    field = ParallelSheetField(((x, 3), (x,-3)), reference_potential_v=1e100, bias_v_m=0.25)
    assert field.electric_field_at((x,0,0))[0] == 0.25
    assert field.potential_at((x,0,0)) == field.potential_at((x+1,0,0))
    assert field.field_work_j(-2, (x,0,0), (x+1,0,0)) == -0.5
