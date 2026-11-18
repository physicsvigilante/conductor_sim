"""M37a: independently derived grounded-sphere wire charge, force and current."""
from dataclasses import replace
from math import hypot, pi

import pytest

from conductor_sim import (
    EPSILON_0, ElectrostaticProblem, ExternalGroundedSphereBackend,
    FixedNetCharge, Grounded, Motion1DSpec, PointCharge, SolidSphere,
    SphericalShell, integrate_motion_1d,
)
from conductor_sim.grounded_sphere_wire import (
    observe_grounded_sphere_wire, observe_grounded_sphere_motion,
)
from conductor_sim.solver import UnsupportedProblem


K = 1 / (4 * pi * EPSILON_0)


def model(q=1e-9, r=.2, *, a=.1, c=(0., 0., 0.), constraint=None):
    center = c
    p = (c[0], c[1], c[2] + r)
    return ElectrostaticProblem(
        (PointCharge(q, p),),
        (SolidSphere(a, Grounded() if constraint is None else constraint, center, "Sphere A"),),
    )


@pytest.mark.parametrize("q,velocity,expected_sign", [
    (1e-9, -.1, 1),
    (1e-9, +.1, -1),
    (-2e-9, -.1, -1),
    (-2e-9, +.1, 1),
])
def test_independent_kelvin_gauss_force_energy_and_signed_current(q, velocity, expected_sign):
    a, d = .1, .2
    state = observe_grounded_sphere_wire(model(q), (0., 0., velocity))
    expected_Q = -q*a/d
    expected_current = -q*a*velocity/d**2
    expected_force_r = -K*q*q*a*d/(d*d-a*a)**2
    expected_U = -K*q*q*a/(2*(d*d-a*a))
    assert state.induced_sphere_net_charge_c == pytest.approx(expected_Q, rel=1e-12)
    assert state.wire_current_sphere_to_ground_a == pytest.approx(expected_current, rel=1e-12)
    assert state.wire_current_ground_to_sphere_a == pytest.approx(-expected_current, rel=1e-12)
    assert state.induced_net_charge_rate_c_per_s == pytest.approx(-expected_current, rel=1e-12)
    assert state.induced_force_on_source_n == pytest.approx((0., 0., expected_force_r), rel=1e-12)
    assert state.induced_mechanical_energy_j == pytest.approx(expected_U, rel=1e-12)
    assert state.wire_current_sphere_to_ground_a*expected_sign > 0
    assert state.as_dict()["mode"].startswith("instantaneous")
    assert state.as_dict()["units"]["current"] == "A"


def test_signed_wire_current_is_native_charge_derivative_not_potential_energy_derivative():
    p = model(1e-9, .28)
    velocity = (0., 0., -.07)
    current = observe_grounded_sphere_wire(p, velocity).wire_current_sphere_to_ground_a
    dt = 1e-4
    backend = ExternalGroundedSphereBackend()
    def native_Q(offset):
        moved = replace(p, charges=(PointCharge(1e-9, (0., 0., .28 - .07*offset)),))
        return backend.solve(moved).conductor_results[0].net_charge_c
    independent_derivative = (native_Q(dt)-native_Q(-dt))/(2*dt)
    assert current == pytest.approx(-independent_derivative, rel=1e-8)
    assert independent_derivative < 0


def test_translated_and_rotated_conductors_and_tangential_motion_have_qualified_current():
    for center in ((0., 0., 0.), (1., -2., 3.)):
        c = center
        p = ElectrostaticProblem(
            (PointCharge(-1.2e-9, (c[0]+.2, c[1], c[2])),),
            (SolidSphere(.1, Grounded(), c, "Sphere A"),),
        )
        tangent = observe_grounded_sphere_wire(p, (0., .4, 0.))
        assert tangent.radial_velocity_m_s == pytest.approx(0, abs=1e-15)
        assert tangent.wire_current_sphere_to_ground_a == pytest.approx(0, abs=1e-22)
        assert tangent.induced_sphere_net_charge_c > 0
        incoming = observe_grounded_sphere_wire(p, (-.05, .4, 0.))
        assert incoming.radial_velocity_m_s < 0
        assert incoming.wire_current_sphere_to_ground_a < 0
        assert incoming.induced_force_on_source_n[0] < 0
    # A pure 3D radial direction need not align with any canonical coordinate.
    p = ElectrostaticProblem(
        (PointCharge(2e-9, (.3, .4, 0.)),),
        (SolidSphere(.1, Grounded(), name="Sphere A"),),
    )
    radial = observe_grounded_sphere_wire(p, (-.06, -.08, 0.))
    assert radial.radial_velocity_m_s == pytest.approx(-.1, rel=1e-12)


def test_actual_m16a_radial_motion_produces_real_elapsed_time_charge_flux():
    p = model(1e-9, .2)
    spec = Motion1DSpec(p, 0, 1e-3, (0, 0, 1), -.05, .1, .02)
    motion = integrate_motion_1d(spec, ExternalGroundedSphereBackend())
    trace = observe_grounded_sphere_motion(motion)
    assert motion.termination == "duration"
    assert len(motion.samples) == len(trace.observations) == 6
    assert trace.as_dict()["mode"].startswith("actual elapsed-time")
    assert trace.as_dict()["motion"]["schema"] == "conductor_sim.motion_1d.v1"
    for row, obs in zip(motion.samples, trace.observations):
        d = hypot(*row.position_m)
        vr = row.velocity_m_s  # actual authored axis points radially outward
        assert obs.induced_sphere_net_charge_c == pytest.approx(-1e-9*.1/d)
        assert obs.wire_current_sphere_to_ground_a == pytest.approx(-1e-9*.1*vr/d**2)
        assert obs.induced_force_on_source_n[2] == pytest.approx(row.force_n)
        assert obs.induced_mechanical_energy_j == pytest.approx(row.interaction_energy_j)
        assert obs.wire_current_sphere_to_ground_a > 0
    assert trace.observations[-1].induced_sphere_net_charge_c < trace.observations[0].induced_sphere_net_charge_c


def test_nonlinear_real_motion_on_negative_side_has_correct_current_and_guard():
    p = ElectrostaticProblem(
        (PointCharge(-1e-9, (0., 0., -.14)),),
        (SolidSphere(.1, Grounded(), name="Sphere A"),),
    )
    spec = Motion1DSpec(p, 0, 1e-6, (0, 0, 1), +.35, .25, .01, surface_guard_m=2e-4)
    motion = integrate_motion_1d(spec, ExternalGroundedSphereBackend())
    trace = observe_grounded_sphere_motion(motion)
    assert trace.motion.termination == "surface-guard"
    assert trace.motion.events[-1].kind == "surface-guard"
    assert all(h.distance_from_center_m > .1002 for h in trace.observations)
    # The negative charge becomes less positive? q<0 => induced Q>0;
    # inward motion increases it: positive ground→sphere, negative sphere→ground.
    assert all(x.wire_current_sphere_to_ground_a < 0 for x in trace.observations)
    assert trace.observations[-1].induced_sphere_net_charge_c > trace.observations[0].induced_sphere_net_charge_c


@pytest.mark.parametrize("invalid", [
    (True, 0., 0.),
    (float("nan"), 0., 0.),
    (float("inf"), 0., 0.),
    ("0", 0., 0.),
    (1., 2.),
    1.,
])
def test_malformed_prescribed_velocity_is_refused(invalid):
    with pytest.raises((ValueError, TypeError)):
        observe_grounded_sphere_wire(model(), invalid)


def test_unsupported_kinematics_and_boundary_conditions_refused():
    zero = model(q=0)
    isolated = model(constraint=FixedNetCharge(0))
    inside = model(r=.08)
    touching = model(r=.1)
    shell = ElectrostaticProblem(
        (PointCharge(1e-9, (0., 0., .2)),),
        (SphericalShell(.1,.12,Grounded()),),
    )
    multi = replace(model(), charges=(PointCharge(1e-9,(0.,0.,.2)), PointCharge(1e-9,(0.,0.,.3))))
    two_spheres = replace(model(), conductors=(model().conductors[0],
                     SolidSphere(.05, Grounded(), (0., 0., 2.), "Sphere B")))
    for p in (zero, isolated, inside, touching, shell, multi, two_spheres):
        with pytest.raises(UnsupportedProblem):
            observe_grounded_sphere_wire(p, (0., 0., -.05))
    with pytest.raises(UnsupportedProblem, match="speed cutoff"):
        observe_grounded_sphere_wire(model(), (0., 0., 3e7))


def test_motion_rejects_wrong_constraint_and_nonmotion_input():
    with pytest.raises(TypeError, match="Motion1DResult"):
        observe_grounded_sphere_motion(object())
    p = model(constraint=FixedNetCharge(0))
    spec = Motion1DSpec(p, 0, 1e-3, (0, 0, 1), 0., .01, .01)
    from conductor_sim.motion_1d import Motion1DResult
    fake = Motion1DResult(spec, (), (), "duration", 0.)
    with pytest.raises(UnsupportedProblem):
        observe_grounded_sphere_motion(fake)


def test_motion_falsified_sample_fields_and_history_are_refused():
    p = model()
    spec = Motion1DSpec(p, 0, 1e-3, (0, 0, 1), -.05, .06, .02)
    genuine = integrate_motion_1d(spec, ExternalGroundedSphereBackend())
    assert observe_grounded_sphere_motion(genuine)
    from conductor_sim.motion_1d import MotionSample
    first, second, *rest = genuine.samples
    altered = (
        replace(second, position_m=(0.,0.,.5)),
        replace(second, force_n=-second.force_n),
        replace(second, interaction_energy_j=second.interaction_energy_j*2),
        replace(second, velocity_m_s=second.velocity_m_s+.03),
        replace(second, time_s=first.time_s),
        replace(second, time_s=True),
        replace(second, displacement_m=False),
        replace(second, kinetic_energy_j=bool(second.kinetic_energy_j)),
    )
    for forged in altered:
        bad = replace(genuine, samples=(first, forged, *rest))
        with pytest.raises(ValueError):
            observe_grounded_sphere_motion(bad)
    with pytest.raises(ValueError, match="termination"):
        observe_grounded_sphere_motion(replace(genuine, termination="nonsense"))
    with pytest.raises(ValueError, match="maximum"):
        observe_grounded_sphere_motion(replace(genuine, max_abs_energy_drift_j=1e3))
    with pytest.raises(ValueError, match="method"):
        observe_grounded_sphere_motion(replace(genuine, method="teacher-prescribed animation"))
    from conductor_sim.motion_1d import MotionEvent
    false_event = MotionEvent("turning-point", .02, .03, (.0,.0,.2),
                              "a fabricated position inconsistent with physical axis")
    with pytest.raises(ValueError, match="event location"):
        observe_grounded_sphere_motion(replace(genuine, events=(false_event,)))


def test_permittivity_changes_native_force_and_energy_but_not_induced_wire_current():
    standard = model()
    doubled = replace(standard, permittivity_f_per_m=2*EPSILON_0)
    a = observe_grounded_sphere_wire(standard, (0., 0., -.1))
    b = observe_grounded_sphere_wire(doubled, (0., 0., -.1))
    assert b.permittivity_f_per_m == pytest.approx(2*EPSILON_0)
    assert b.induced_sphere_net_charge_c == pytest.approx(a.induced_sphere_net_charge_c)
    assert b.wire_current_sphere_to_ground_a == pytest.approx(a.wire_current_sphere_to_ground_a)
    assert b.induced_force_on_source_n[2] == pytest.approx(a.induced_force_on_source_n[2]/2)
    assert b.induced_mechanical_energy_j == pytest.approx(a.induced_mechanical_energy_j/2)


def test_source_free_independent_si_reference_accepts_genuine_and_rejects_forged_json():
    from copy import deepcopy
    from examples.grounded_sphere_wire_current import example_data
    from tools.m37a_ground_wire_reference import validate_examples
    genuine = example_data()
    assert validate_examples(genuine)["actual_time_samples"] == 6
    modifications = (
        lambda d: d["instantaneous_prescribed"][0].__setitem__("wire_current_sphere_to_ground_a", -1.),
        lambda d: d["instantaneous_prescribed"][0].__setitem__("induced_sphere_net_charge_c", 2e-9),
        lambda d: d["instantaneous_prescribed"][0].__setitem__("point_charge_c", False),
        lambda d: d["instantaneous_prescribed"][0].__setitem__("mode", "simulated trajectory"),
        lambda d: d["actual_elapsed_time"]["observations"][1].__setitem__("wire_current_ground_to_sphere_a", 1.),
        lambda d: d["actual_elapsed_time"]["motion"]["samples"][1].__setitem__("velocity_m_s", 3.),
        lambda d: d["actual_elapsed_time"]["motion"]["samples"][1].__setitem__("time_s", 0.),
        lambda d: d.__setitem__("source_role", "original problem given R=0.1m Q=1nC"),
    )
    for edit in modifications:
        hostile = deepcopy(genuine)
        edit(hostile)
        with pytest.raises(ValueError):
            validate_examples(hostile)
