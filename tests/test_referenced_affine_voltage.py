from conductor_sim.affine_fields import AffineFieldAssembly, ConservativeAffineField
from conductor_sim.prescribed_fields import UniformElectricField


def test_q56_voltage_adds_across_sources_and_path_legs():
    gradient = ConservativeAffineField(((10, 0, 0), (0, 0, 0), (0, 0, 0)))
    total = AffineFieldAssembly((gradient, UniformElectricField((0, -20, 0))))
    a, corner, b = (30, -20, 0), (-20, -20, 0), (-20, 30, 0)
    assert total.potential_difference_v(a, corner) == 2500
    assert total.potential_difference_v(corner, b) == 1000
    assert total.potential_difference_v(a, b) == 3500
    assert total.potential_difference_v(b, a) == -3500
    assert total.field_work_j(-2, a, b) == 7000


def test_q54_voltage_difference_survives_unresolvable_absolute_difference():
    model = AffineFieldAssembly((UniformElectricField((4, 5, 0), reference_potential_v=1e100),))
    a, b = (0, 0, 0), (2, -1, 0)
    assert model.potential_at(a) == model.potential_at(b)
    assert model.potential_difference_v(a, b) == -3
    assert model.potential_difference_v(b, b) == 0
