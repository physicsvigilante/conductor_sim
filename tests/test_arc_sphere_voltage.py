from math import pi

import pytest

from conductor_sim import (ElectrostaticProblem, ExternalMultiSphereBackend,
                           FixedNetCharge, Grounded, PointCharge, SolidSphere)
from conductor_sim.coaxial_arcs import CircularArcCharge, CoaxialArcAssembly
from conductor_sim.solver import UnsupportedProblem


def test_q51_nonuniform_ring_voltage_depends_only_on_total_charge():
    sphere = SolidSphere(4, FixedNetCharge(0))
    whole = CoaxialArcAssembly((CircularArcCharge(1e-6, 4, 0, 360, 3),))
    split = CoaxialArcAssembly((CircularArcCharge(1.5e-6, 4, 0, 90, 3),
                                CircularArcCharge(-0.5e-6, 4, 90, 270, 3)))
    for assembly in (whole, split):
        assert assembly.isolated_sphere_voltage_v(sphere) == pytest.approx(1797.51, abs=0.01)


def test_distinct_arc_planes_and_charged_sphere_match_kelvin_solution():
    sphere = SolidSphere(1, FixedNetCharge(-1e-9), center_m=(0, 0, 2))
    arcs = CoaxialArcAssembly((CircularArcCharge(2e-9, 3, 0, 60, 6),
                              CircularArcCharge(-3e-9, 2, 90, 50, 2)))
    # Each point concentrates its arc charge at the start angle. This preserves
    # centre distance, hence conductor voltage, even though polarization changes.
    points = (PointCharge(2e-9, (3, 0, 6)), PointCharge(-3e-9, (0, 2, 2)))
    solution = ExternalMultiSphereBackend().solve(ElectrostaticProblem(
        charges=points, conductors=(sphere,)))
    assert arcs.isolated_sphere_voltage_v(sphere) == pytest.approx(solution.conductor_potential_v)
    expected = (-1e-9 + 2e-9/5 - 3e-9/2)/(4*pi*arcs.permittivity_f_per_m)
    assert arcs.isolated_sphere_voltage_v(sphere) == pytest.approx(expected)


def test_qualification_checks_the_complete_geometry_and_constraint():
    external = CircularArcCharge(1, 3, 0, 90)
    assembly = CoaxialArcAssembly((external, CircularArcCharge(-1, 1, 10, 30)))
    for radius in (1, 2):
        with pytest.raises(UnsupportedProblem, match="strictly outside"):
            assembly.isolated_sphere_voltage_v(SolidSphere(radius, FixedNetCharge(0)))
    with pytest.raises(UnsupportedProblem, match="axis"):
        assembly.isolated_sphere_voltage_v(SolidSphere(1, FixedNetCharge(0), center_m=(1, 0, 0)))
    with pytest.raises(UnsupportedProblem, match="fixed net charge"):
        assembly.isolated_sphere_voltage_v(SolidSphere(1, Grounded()))


@pytest.mark.parametrize(('radius', 'charge'), [(True, 1), (1, True)])
def test_boolean_sphere_quantities_are_rejected(radius, charge):
    sphere = SolidSphere(radius, FixedNetCharge(charge))
    with pytest.raises(ValueError, match="sphere"):
        CoaxialArcAssembly(()).isolated_sphere_voltage_v(sphere)
