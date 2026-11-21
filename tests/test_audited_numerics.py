"""Regression probes for the seven findings on 0b39e424."""
from dataclasses import replace
from math import pi, sqrt, ulp

import pytest

from conductor_sim import (
    ConnectedShellBackend, ConnectionPlan, ElectricalLink, ElectrostaticProblem,
    FixedNetCharge, Grounded, MultiImageShellBackend, PointCharge, SolidSphere,
    SphericalShell, integrate_test_charge,
)
from conductor_sim.jee_potential_explorer import endpoint_field_work
from conductor_sim.model import EPSILON_0
from conductor_sim.prescribed_fields import (
    FixedPointField, PrescribedField, UniformElectricField, UniformRingAxisField, UniformRingField,
)
from conductor_sim.prescribed_motion import TestChargeMotionSpec as MotionSpec
from conductor_sim.prescribed_viewer import sample_test_charge_viewer
from conductor_sim.prescribed_viewer_validation import validate_prescribed_viewer
from tools.jee12_walkthrough_reference import verify_work
from tools.jee12_walkthroughs import work_record

K = 1 / (4 * pi * EPSILON_0)
ELECTRON_CHARGE = -1.602176634e-19


def known_issue(number):
    return pytest.mark.xfail(strict=True, raises=AssertionError, reason=f"Open audit finding #{number}")


@pytest.mark.parametrize("x, speed", [(0., 1e-4), (1e9, 1e-4), (1e9, -1e-8)])
def test_free_drift_after_translation(x, speed):
    field = PrescribedField((UniformElectricField((0., 0., 0.)),))
    result = integrate_test_charge(MotionSpec(field, PointCharge(1e-9, (x, 0., 0.)),
                                              1., (speed, 0., 0.), 1., .001))
    final = result.samples[-1]
    assert result.termination == "duration"
    assert final.time_s == pytest.approx(1.)
    assert final.velocity_m_s[0] == pytest.approx(speed)
    for row in result.samples:
        expected = speed * row.time_s
        assert row.displacement_m[0] == pytest.approx(expected, rel=1e-12, abs=1e-24)
        assert row.position_m[0] - x == pytest.approx(expected, abs=max(ulp(x), 1e-16))


def test_oblique_translated_drift_keeps_the_certified_plane_in_saved_samples():
    spec = MotionSpec(PrescribedField((UniformElectricField((0., 0., 0.)),)),
                      PointCharge(1e-9, (1e9, 0., 0.)), 1., (1e-4, 1e-4, 0.),
                      1., .01, plane_normal=(1., -1., 0.))
    result = integrate_test_charge(spec)
    final = result.samples[-1]
    assert final.displacement_m == pytest.approx((1e-4, 1e-4, 0.), abs=1e-16)
    bundle = sample_test_charge_viewer(spec, result=result)
    validate_prescribed_viewer(bundle)
    # This transverse perturbation still rounds to the same world position.
    forged = replace(final, displacement_m=(final.displacement_m[0]+3e-8,
                                             final.displacement_m[1], 0.))
    assert spec.particle.position_m[0] + forged.displacement_m[0] == final.position_m[0]
    with pytest.raises(ValueError, match="plane"):
        sample_test_charge_viewer(spec, result=replace(result, samples=(*result.samples[:-1], forged)))


def test_inner_density_at_accepted_surface_direction():
    d = .99999999
    problem = ElectrostaticProblem((PointCharge(1e-9, (d, 0., 0.)),),
                                   (SphericalShell(1., 1.1, Grounded()),))
    solution = MultiImageShellBackend().solve(problem)
    expected = -1e-9 * (1 + d) / (4 * pi * (1 - d)**2)
    assert solution.surface_charge_density_at("inner", (1., 0., 0.)) == pytest.approx(expected)
    density = solution.surface_charge_density_at("inner", (1.00000000009, 0., 0.))
    assert density == pytest.approx(expected)


def test_translated_cavity_projects_an_oblique_probe_onto_the_inner_wall():
    center = (8., -16., 4.)
    radius = 5.
    gap = radius * 2**-27
    direction = (3., 4., 0.)

    def point(scale):
        return tuple(c + scale * x for c, x in zip(center, direction))

    charge = -2e-9
    shell = SphericalShell(radius, 6., Grounded(), center)
    problem = ElectrostaticProblem((PointCharge(charge, point(1 - 2**-27)),), (shell,))
    solution = MultiImageShellBackend().solve(problem)
    expected = -charge * (2 * radius - gap) / (4 * pi * radius * gap**2)
    for scale in (1 - 2**-35, 1., 1 + 2**-35):
        density = solution.surface_charge_density_at("inner", point(scale))
        assert density == pytest.approx(expected, rel=2e-14)
    with pytest.raises(ValueError, match="not on"):
        solution.surface_charge_density_at("inner", point(1.001))


@pytest.mark.parametrize("outer_charge", [0., 1e-3])
def test_connection_releases_the_gap_field_energy(outer_charge):
    q = 1e-12
    solid = SolidSphere(.1, FixedNetCharge(q), name="solid")
    shell = SphericalShell(.199999, .2, FixedNetCharge(outer_charge), name="shell")
    plan = ConnectionPlan(ElectrostaticProblem(conductors=(solid, shell)),
                          (ElectricalLink(("solid", "shell")),))
    result = ConnectedShellBackend().solve(plan)
    expected = K * q**2 / 2 * (1 / .1 - 1 / .199999)
    assert result.released_field_energy_j == pytest.approx(expected, rel=1e-12, abs=0.)


@pytest.mark.parametrize("reference", [0., 1e18])
def test_uniform_field_work_under_reference_shift(reference):
    field = PrescribedField((UniformElectricField((10., 0., 0.), reference_potential_v=reference),))
    result = endpoint_field_work(field, source_annotations=(("u", "uniform"),),
                                 probe_charge_c=ELECTRON_CHARGE, start_m=(0., 0., 0.), end_m=(4., 0., 0.))
    expected = ELECTRON_CHARGE * 40
    for value in (result["field_work_j"], result["source_work_components"][0]["field_work_j"],
                  -result["external_quasistatic_work_j"]):
        assert value == pytest.approx(expected, rel=1e-14, abs=0.)


def test_shifted_uniform_field_composes_with_point_source_work():
    uniform = UniformElectricField((10., 0., 0.), reference_potential_v=1e18)
    point = FixedPointField(1e-9)
    result = endpoint_field_work(PrescribedField((point, uniform)),
                                 source_annotations=(("p", "point"), ("u", "uniform")),
                                 probe_charge_c=ELECTRON_CHARGE, start_m=(1., 0., 0.), end_m=(5., 0., 0.))
    expected = ELECTRON_CHARGE * (K * 1e-9 * (1 - 1/5) + 40)
    assert result["field_work_j"] == pytest.approx(expected, rel=1e-14, abs=0.)


@known_issue(5)
@pytest.mark.parametrize("mutate_page", [True, False])
def test_boolean_is_not_source_or_geometry_evidence(mutate_page):
    record = work_record()
    if mutate_page:
        record["source"]["pdf_page_index"] = True
    else:
        record["derived"]["start_m"][0] = False
    try:
        verify_work(record)
    except AssertionError:
        return
    assert False, "verifier accepted a Boolean in a numeric evidence field"


@known_issue(6)
def test_integer_source_separation_is_preserved_or_refused():
    try:
        source = FixedPointField(1e-9, (2**53 + 1, 0, 0))
    except ValueError:
        return
    point = (2**53 + 2, 0, 0)
    assert source.potential_at(point) == pytest.approx(K * 1e-9)
    assert source.electric_field_at(point)[0] == pytest.approx(K * 1e-9)


@known_issue(7)
def test_full_ring_agrees_with_axial_solution_at_large_scale():
    radius, charge = 1e160, 1e200
    expected = K * (charge / radius) / radius / (2 * sqrt(2))
    for provider in (UniformRingAxisField, UniformRingField):
        field = provider(charge, radius).electric_field_at((0., 0., radius))
        assert field[2] == pytest.approx(expected, rel=1e-12, abs=0.)
