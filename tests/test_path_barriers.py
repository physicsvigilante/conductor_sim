from fractions import Fraction
from math import pi, sqrt

import pytest

from conductor_sim.affine_fields import AffineFieldAssembly, ConservativeAffineField
from conductor_sim.model import EPSILON_0
from conductor_sim.path_barriers import straight_path_energy_barrier
from conductor_sim.prescribed_fields import UniformElectricField, UniformShellField, UniformSphereField
from conductor_sim.solver import UnsupportedProblem


def test_q35_equal_endpoint_potentials_hide_internal_barrier():
    radius, density, mass = 2., 1e-9, 0.5
    sphere = UniformSphereField(4*pi*radius**3*density/3, radius)
    half_chord = sqrt(3)*radius/2
    start, end = (-half_chord, radius/2, 0), (half_chord, radius/2, 0)
    result = straight_path_energy_barrier((sphere,), 1, start, end)
    assert sphere.potential_at(start) == sphere.potential_at(end)
    assert result.peak_fraction == Fraction(1, 2)
    assert result.rise_j == pytest.approx(sphere.potential_at((0, radius/2, 0)) - sphere.potential_at(start))
    assert sqrt(2*result.rise_j/mass) == pytest.approx(sqrt(density*radius**2/(4*mass*EPSILON_0)))
    assert straight_path_energy_barrier((sphere,), -1, start, end).rise_j == 0


def test_superposed_bias_moves_peak_off_center_and_outside_segment():
    sphere = UniformSphereField(4*pi, 1)
    start, end = (-1, 0, 0), (1, 0, 0)
    def barrier(bias):
        field = AffineFieldAssembly((UniformElectricField((bias, 0, 0)),))
        return straight_path_energy_barrier((field, sphere), 2, start, end, permittivity_f_m=1)
    assert barrier(0.5).peak_fraction == Fraction(1, 4)
    assert barrier(0.5).rise_j == 0.25
    assert barrier(2).peak_fraction == 0
    assert barrier(-2).peak_fraction == 1
    assert barrier(-2).rise_j == 8


def test_three_dimensional_guide_and_large_gauge():
    field = ConservativeAffineField(((0, 2, 0), (2, 0, 0), (0, 0, 0)), reference_potential_v=1e100)
    assembly = AffineFieldAssembly((field,))
    result = straight_path_energy_barrier((assembly,), 1, (-2, -2, -2), (1, 1, 1))
    assert result.peak_fraction == Fraction(2, 3)
    assert result.rise_j == 8


def test_refuses_exterior_segment_shell_and_collapsed_guide():
    with pytest.raises(UnsupportedProblem, match="within"):
        straight_path_energy_barrier((UniformSphereField(1, 1),), 1, (0, 0, 0), (2, 0, 0))
    with pytest.raises(UnsupportedProblem, match="volume sphere"):
        straight_path_energy_barrier((UniformShellField(1, 1),), 1, (0, 0, 0), (0.5, 0, 0))
    with pytest.raises(ValueError, match="distinct"):
        straight_path_energy_barrier((), 1, (0, 0, 0), (0, 0, 0))
