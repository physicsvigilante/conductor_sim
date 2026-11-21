import pytest

from conductor_sim.affine_fields import AffineFieldAssembly, ConservativeAffineField
from conductor_sim.prescribed_fields import FixedPointField, UniformElectricField
from conductor_sim.solver import UnsupportedProblem


def test_workbook_q28_signed_potential_and_gauge():
    field = ConservativeAffineField(((0, 3, 0), (3, 0, 0), (0, 0, 0)), reference_potential_v=7)
    assert field.electric_field_at((2, -4, 10)) == (-12, 6, 0)
    assert field.potential_at((2, -4, 10)) == 31
    assert field.potential_at((0, 0, -8)) == 7


def test_shifted_three_dimensional_field_plus_uniform_bias():
    field = ConservativeAffineField(((2, 0, 1), (0, -2, 0), (1, 0, 0)),
                                   (1, 2, 3), (1, 0, 2), 8)
    assembly = AffineFieldAssembly((field, UniformElectricField((0, 1, 0), (0, 1, 0), 5)))
    start, end = (1, 0, 2), (2, 2, 4)
    assert assembly.electric_field_at(end) == (5, -1, 4)
    assert assembly.potential_at(start) == 14
    assert assembly.potential_at(end) == 2
    assert assembly.field_work_j(-2, start, end) == -24
    assert assembly.field_work_j(-2, end, start) == 24


def test_work_survives_gauge_rounding_and_integer_geometry():
    x = 2**54
    field = ConservativeAffineField(((2, 0, 0), (0, 0, 0), (0, 0, 0)),
                                   reference_point_m=(x, 0, 0), reference_potential_v=1e100)
    assembly = AffineFieldAssembly((field,))
    assert assembly.potential_at((x, 0, 0)) == assembly.potential_at((x+1, 0, 0))
    assert assembly.field_work_j(1, (x, 0, 0), (x+1, 0, 0)) == 1


def test_nonzero_observations_below_binary64_range_are_refused():
    scale = 2.0**-600
    source = ConservativeAffineField(((scale, 0, 0), (0, 0, 0), (0, 0, 0)))
    assembly = AffineFieldAssembly((source,))
    point = (scale, 0, 0)
    for sample in (assembly.potential_at, assembly.electric_field_at):
        with pytest.raises(ValueError, match="underflows"):
            sample(point)
    with pytest.raises(ValueError, match="underflows"):
        assembly.field_work_j(1, (0, 0, 0), point)


def test_reject_curl_and_nonaffine_component():
    with pytest.raises(UnsupportedProblem, match="symmetric"):
        ConservativeAffineField(((0, 2, 0), (1, 0, 0), (0, 0, 0)))
    with pytest.raises(UnsupportedProblem, match="affine"):
        AffineFieldAssembly((FixedPointField(1),))
