from math import inf, nextafter, pi, sqrt

import pytest

from conductor_sim.model import ElectrostaticProblem, FixedNetCharge, PointCharge, SolidSphere
from conductor_sim import insert_point_charge, interaction_energy_j, rearrange_point_charges
from conductor_sim.solver import UnsupportedProblem


def at(q, x, y=0, z=0):
    return PointCharge(q, (x, y, z))


def test_q37_shrinking_square_reduces_energy_and_requires_negative_external_work():
    initial = ElectrostaticProblem(charges=(at(1, 0, 2), at(1, 2, 2), at(-1, 0), at(-1, 2)))
    change = rearrange_point_charges(initial, ((0,1,0),(1,1,0),(0,0,0),(1,0,0)))
    k = 1/(4*pi*initial.permittivity_f_per_m)
    assert interaction_energy_j(initial) == pytest.approx(-sqrt(2)*k/2)
    assert interaction_energy_j(change.final) == pytest.approx(-sqrt(2)*k)
    assert change.external_work_j == pytest.approx(-sqrt(2)*k/2)


def test_q44_bisector_section_and_axial_crossing_of_full_locus():
    initial = ElectrostaticProblem(charges=(at(1, -1), at(1, 1)))
    def energy(x, y):
        return interaction_energy_j(insert_point_charge(initial, at(-1, x, y)).final)
    for root, axial in ((sqrt(15), False), (2+sqrt(5), True)):
        below, above = nextafter(root, -inf), nextafter(root, inf)
        assert (energy(below, 0) if axial else energy(0, below)) < 0
        assert (energy(above, 0) if axial else energy(0, above)) > 0
    assert energy(20, sqrt(15)) > 0


def test_staged_insertion_rearrangement_and_exact_zero_total():
    initial = ElectrostaticProblem(charges=(at(9, 0), at(9, 3)))
    inserted = insert_point_charge(initial, at(-2, 1))
    k = 1/(4*pi*initial.permittivity_f_per_m)
    assert inserted.external_work_j == pytest.approx(-27*k)
    assert interaction_energy_j(initial) == pytest.approx(27*k)
    assert interaction_energy_j(inserted.final) == 0
    moved = rearrange_point_charges(inserted.final, ((0,0,0),(3,0,0),(5,0,0)))
    assert moved.external_work_j == interaction_energy_j(moved.final)
    assert inserted.final.charges[:2] == initial.charges


def test_insertion_work_survives_rounding_of_large_existing_energy():
    initial = ElectrostaticProblem(charges=(at(1e40, 0), at(1e40, 1)))
    inserted = insert_point_charge(initial, at(-1, 2))
    assert interaction_energy_j(initial) == interaction_energy_j(inserted.final)
    assert inserted.external_work_j == pytest.approx(-1.5e40/(4*pi*initial.permittivity_f_per_m))


def test_integer_translation_cannot_silently_change_pair_separation():
    origin = 2**53
    with pytest.raises(ValueError, match="coordinate that loses precision"):
        at(1, origin+1)
    pair = ElectrostaticProblem(charges=(at(1, origin), at(1, origin+4)), permittivity_f_per_m=1)
    assert interaction_energy_j(pair) == pytest.approx(1/(16*pi))


def test_registered_integral_coordinate_uses_its_exact_integer_value():
    from numbers import Integral

    @Integral.register
    class IntegerScalar:
        def __index__(self):
            return 2**53 + 1

    with pytest.raises(ValueError, match="coordinate that loses precision"):
        at(1, IntegerScalar())


def test_singular_sources_and_responding_metal_refused():
    initial = ElectrostaticProblem(charges=(at(1, 0),))
    with pytest.raises(UnsupportedProblem, match="singular"):
        insert_point_charge(initial, at(-1, 0))
    assert interaction_energy_j(insert_point_charge(initial, at(0, 0)).final) == 0
    metal = ElectrostaticProblem(conductors=(SolidSphere(1, FixedNetCharge(0)),))
    with pytest.raises(UnsupportedProblem, match="conductors"):
        interaction_energy_j(metal)
