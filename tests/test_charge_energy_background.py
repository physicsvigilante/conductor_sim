from math import pi

import pytest

from conductor_sim import ElectrostaticProblem, FixedNetCharge, PointCharge, SolidSphere, UniformElectricField
from conductor_sim.affine_fields import AffineFieldAssembly, ConservativeAffineField
from conductor_sim.point_charge_work import configuration_energy_j, interaction_energy_j
from conductor_sim.solver import UnsupportedProblem


def test_q61_pair_counted_once_and_external_potentials_counted_in_full():
    problem = ElectrostaticProblem(charges=(PointCharge(2e-9, (0, 0, 0)), PointCharge(-3e-9, (3, -4, 0))))
    background = AffineFieldAssembly((UniformElectricField((1, 0, 0), reference_potential_v=7),))
    pair = -6e-18 / (4*pi*problem.permittivity_f_per_m*5)
    expected = 2e-9*7 - 3e-9*4 + pair
    assert configuration_energy_j(problem, background) == pytest.approx(expected, rel=1e-13, abs=0)
    assert interaction_energy_j(problem) == pytest.approx(pair, rel=1e-13, abs=0)


def test_pair_background_cancellation_happens_before_rounding():
    problem = ElectrostaticProblem(charges=(PointCharge(2**60), PointCharge(4*pi, (1, 0, 0))),
                                   permittivity_f_per_m=1)
    constant = AffineFieldAssembly((UniformElectricField((0, 0, 0), reference_potential_v=-1),))
    assert configuration_energy_j(problem, constant) == -4*pi


def test_gauge_shift_and_nonuniform_composition_for_a_single_charge():
    gradient = ConservativeAffineField(((2, 0, 0), (0, 0, 0), (0, 0, 0)))
    shifted = UniformElectricField((0, 0, 0), reference_potential_v=5)
    charge = ElectrostaticProblem(charges=(PointCharge(-3, (2, 0, 0)),))
    assert configuration_energy_j(charge, AffineFieldAssembly((gradient,))) == 12
    assert configuration_energy_j(charge, AffineFieldAssembly((gradient, shifted))) == -3
    assert configuration_energy_j(ElectrostaticProblem(), AffineFieldAssembly((gradient,))) == 0


def test_qualification_rejects_singular_sites_and_responding_conductors():
    background = AffineFieldAssembly((UniformElectricField((1, 0, 0)),))
    with pytest.raises(UnsupportedProblem, match="coincident"):
        configuration_energy_j(ElectrostaticProblem(charges=(PointCharge(1), PointCharge(-1))), background)
    with pytest.raises(UnsupportedProblem, match="conductors"):
        configuration_energy_j(ElectrostaticProblem(conductors=(SolidSphere(1, FixedNetCharge(0)),)), background)
