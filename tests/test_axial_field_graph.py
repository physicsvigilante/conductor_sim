from fractions import Fraction

import pytest

from conductor_sim.axial_field_graph import AxialFieldGraph
from conductor_sim.solver import UnsupportedProblem


def test_workbook_q40_original_graph_and_signed_area():
    graph = AxialFieldGraph(((0, 0), (2, -20), (4, 20), (5, 20)), 10)
    assert graph.potential_v(2) == 30
    assert graph.potential_v(3) == 40
    assert graph.potential_v(5) == 10
    assert graph.axial_field_v_m(3) == 0
    assert graph.field_work_j(-2, 0, 3) == 60
    assert graph.field_work_j(-2, 3, 0) == -60


def test_superposition_preserves_breakpoints_and_reference_in_common_domain():
    a = AxialFieldGraph(((0, 0), (2, -4), (4, 4)), 10)
    b = AxialFieldGraph(((1, 2), (3, 2), (5, -2)), 3)
    total = a.superpose(b)
    assert [x for x, e in total.knots] == [1, 2, 3, 4]
    for x in (1, Fraction(3, 2), 2, 3, Fraction(7, 2), 4):
        assert total.axial_field_v_m(x) == a.axial_field_v_m(x) + b.axial_field_v_m(x)
        assert total.potential_v(x) == a.potential_v(x) + b.potential_v(x)
    with pytest.raises(UnsupportedProblem):
        total.potential_v(0)


def test_restrict_existing_coupled_affine_field_and_keep_its_shifted_reference():
    from conductor_sim.affine_fields import AffineFieldAssembly, ConservativeAffineField
    source = ConservativeAffineField(((2, 1, 0), (1, -1, 0), (0, 0, 0)),
                                    (1, 2, 3), (0, 3, 1), 7)
    assembly = AffineFieldAssembly((source,))
    graph = AxialFieldGraph.from_affine(assembly, -1, 3)
    for x in (-1, 0, 2, 3):
        assert graph.axial_field_v_m(x) == assembly.electric_field_at((x, 0, 0))[0]
        assert graph.potential_v(x) == assembly.potential_at((x, 0, 0))
    assert graph.field_work_j(-2, -1, 3) == assembly.field_work_j(-2, (-1, 0, 0), (3, 0, 0))


def test_large_gauge_preserves_small_work_and_shared_end_alone_is_refused():
    base = 2**55
    a = AxialFieldGraph(((base, 1), (base+1, 2)), 1e100)
    assert a.potential_v(base) == a.potential_v(base+1)
    assert a.field_work_j(1, base, base+1) == 1.5
    with pytest.raises(UnsupportedProblem, match="overlapping interval"):
        a.superpose(AxialFieldGraph(((base+1, 0), (base+2, 1))))
    with pytest.raises(ValueError, match="strictly increase"):
        AxialFieldGraph(((0, 1), (0, 3)))
