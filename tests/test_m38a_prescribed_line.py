"""M38a: actual 3D linear-path Kelvin physics, continuous guards, integrity."""
from copy import deepcopy
from dataclasses import replace
from math import hypot, isclose

import pytest

from conductor_sim import (
    ExternalGroundedSphereBackend, FixedNetCharge, Grounded,
    PointCharge, SolidSphere,
)
from conductor_sim.prescribed_ground_line import (
    PrescribedSphereLineSpec, sample_prescribed_sphere_line,
    validate_prescribed_sphere_line,
)
from conductor_sim.prescribed_ground_wire import (
    PrescribedSphereRadialSpec, author_prescribed_grounded_sphere,
    sample_prescribed_sphere_radial,
)
from conductor_sim.solver import UnsupportedProblem
from examples.m38a_line_approach import example_data
from tools.m38a_line_reference import check_line_record


def make(q=1e-9, p=(-.4, .2, 0.), center=(0., 0., 0.), a=.1):
    return author_prescribed_grounded_sphere(
        q, p, a, sphere_center_m=center, sphere_name="Sphere A"
    )


def line(problem=None, *, velocity=(.2, 0., 0.), duration=4.,
         count=41, guard=.01):
    if problem is None:
        problem = make()
    return PrescribedSphereLineSpec(
        problem, velocity_m_s=velocity, duration_s=duration,
        sample_count=count, surface_guard_m=guard,
    )


def test_true_offaxis_line_has_changing_ground_current_and_independent_si():
    doc = example_data()
    evidence = check_line_record(doc)
    assert evidence["teacher_hypothetical_paths"] == 1
    assert evidence["original_workbook_items_newly_solved"] == 0
    assert evidence["independently_checked_kelvin_states"] == 41
    assert doc["source_role"].startswith("teacher_hypothetical")
    assert doc["closest_approach_time_s"] == pytest.approx(2.)
    assert doc["closest_approach_distance_m"] == pytest.approx(.2)
    assert doc["samples"][0]["wire_current_sphere_to_ground_a"] > 0
    assert doc["samples"][20]["wire_current_sphere_to_ground_a"] == pytest.approx(0, abs=1e-20)
    assert doc["samples"][40]["wire_current_sphere_to_ground_a"] < 0
    assert doc["samples"][20]["radial_velocity_m_s"] == pytest.approx(0, abs=1e-15)
    assert tuple(doc["samples"][0]["source_velocity_m_s"]) == (0.2, 0., 0.)
    assert doc["original_workbook_case"].endswith("teacher extension")


@pytest.mark.parametrize("q,signed_speed", [
    (1e-9, -.05), (1e-9, .05), (-2e-9, -.05), (-2e-9, .05)
])
def test_radial_is_genuine_subset_of_existing_m37b(q, signed_speed):
    p = make(q=q, p=(0., 0., .4))
    old = sample_prescribed_sphere_radial(PrescribedSphereRadialSpec(
        p, signed_speed, 2., sample_count=11, surface_guard_m=.01))
    new = sample_prescribed_sphere_line(line(
        p, velocity=(0., 0., signed_speed), duration=2., count=11))
    for prev, now in zip(old.samples, new.samples):
        assert now.time_s == pytest.approx(prev.time_s)
        assert now.radial_distance_m == pytest.approx(prev.radial_distance_m, rel=1e-13)
        assert now.induced_sphere_net_charge_c == pytest.approx(prev.induced_sphere_net_charge_c, rel=1e-13)
        assert now.wire_current_sphere_to_ground_a == pytest.approx(prev.wire_current_sphere_to_ground_a, rel=1e-13)
        assert now.wire_current_ground_to_sphere_a == pytest.approx(prev.wire_current_ground_to_sphere_a, rel=1e-13)
        assert now.induced_force_on_source_n == pytest.approx(prev.induced_force_on_source_n, rel=1e-13)
        assert now.induced_mechanical_energy_j == pytest.approx(prev.induced_mechanical_energy_j, rel=1e-13)
    check_line_record(new.as_dict())


def test_negative_charge_reverses_both_terminal_current_signs_not_force():
    pos = sample_prescribed_sphere_line(line())
    neg = sample_prescribed_sphere_line(line(make(q=-1e-9)))
    for a, b in zip(pos.samples, neg.samples):
        assert b.induced_sphere_net_charge_c == pytest.approx(-a.induced_sphere_net_charge_c)
        assert b.wire_current_sphere_to_ground_a == pytest.approx(-a.wire_current_sphere_to_ground_a)
        assert b.wire_current_ground_to_sphere_a == pytest.approx(-a.wire_current_ground_to_sphere_a)
        assert b.induced_force_on_source_n == pytest.approx(a.induced_force_on_source_n)
        assert b.induced_mechanical_energy_j == pytest.approx(a.induced_mechanical_energy_j)
    check_line_record(neg.as_dict())


def test_between_sample_metal_intersection_cannot_hide_behind_safe_frames():
    # At t=4/3 and 8/3 the source is outside the 0.11m guarded sphere,
    # but it crosses the conductor at t=2, which is NOT a saved frame.
    p = make(p=(-.4, .05, 0.))
    sample_times = (0., 4/3, 8/3, 4.)
    assert all(hypot(-.4+.2*t, .05) > .11 for t in sample_times)
    with pytest.raises(UnsupportedProblem, match="continuous path"):
        line(p, count=4)


def test_near_tangent_guard_and_unseen_authentic_closest_passage():
    with pytest.raises(UnsupportedProblem, match="continuous path"):
        line(make(p=(-.4, .11, 0.)), guard=.01)
    valid = sample_prescribed_sphere_line(line(make(p=(-.4, .12, 0.)), count=4))
    assert valid.spec.closest_approach[0] == pytest.approx(2.)
    assert all(s.time_s != 2 for s in valid.samples)
    check_line_record(valid.as_dict())


def test_world_rotation_and_translation_preserve_physical_scalar_current():
    a = sample_prescribed_sphere_line(line())
    # 90-degree xy rotation, then + (1,-2,3) world translation.
    b = sample_prescribed_sphere_line(line(
        make(p=(.8, -2.4, 3.), center=(1., -2., 3.)),
        velocity=(0., .2, 0.),
    ))
    assert b.spec.closest_approach == pytest.approx(a.spec.closest_approach)
    for x, y in zip(a.samples, b.samples):
        assert y.radial_distance_m == pytest.approx(x.radial_distance_m)
        assert y.wire_current_sphere_to_ground_a == pytest.approx(x.wire_current_sphere_to_ground_a)
        assert y.induced_sphere_net_charge_c == pytest.approx(x.induced_sphere_net_charge_c)
        assert y.induced_mechanical_energy_j == pytest.approx(x.induced_mechanical_energy_j)
        fx, fy, fz = x.induced_force_on_source_n
        assert y.induced_force_on_source_n == pytest.approx((-fy, fx, fz))
    check_line_record(b.as_dict())


def test_inward_current_is_negative_derivative_of_fresh_native_kelvin_charge():
    spec = line()
    trace = sample_prescribed_sphere_line(spec)
    t, dt = 1., 1e-4
    start = spec.problem.charges[0].position_m
    v = spec.velocity_m_s
    def native_q(time):
        point = tuple(start[j] + v[j]*time for j in range(3))
        moved = replace(spec.problem, charges=(PointCharge(1e-9, point),))
        return ExternalGroundedSphereBackend().solve(moved).conductor_results[0].net_charge_c
    derivative = (native_q(t+dt)-native_q(t-dt))/(2*dt)
    current = trace.samples[10].wire_current_sphere_to_ground_a
    assert current == pytest.approx(-derivative, rel=1e-8)


@pytest.mark.parametrize("change", [
    {"velocity_m_s": (True, 0., 0.)},
    {"velocity_m_s": (float("nan"), 0., 0.)},
    {"velocity_m_s": (float("inf"), 0., 0.)},
    {"velocity_m_s": (0., 0., 0.)},
    {"velocity_m_s": (.1*299_792_458., 0., 0.)},
    {"velocity_m_s": (1., 2.)},
    {"duration_s": False},
    {"duration_s": 0.},
    {"duration_s": float("nan")},
    {"surface_guard_m": False},
    {"surface_guard_m": -0.01},
    {"sample_count": True},
    {"sample_count": 2},
    {"sample_count": 202},
])
def test_malformed_or_unsupported_physical_authoring_is_refused(change):
    base = dict(problem=make(), velocity_m_s=(.2, 0., 0.),
                duration_s=4., sample_count=41, surface_guard_m=.01)
    base.update(change)
    with pytest.raises((ValueError, TypeError, UnsupportedProblem)):
        PrescribedSphereLineSpec(**base)


def test_non_grounded_or_multi_source_conductors_refused_before_simulation():
    p = make()
    bad = replace(p, conductors=(SolidSphere(.1, FixedNetCharge(0.),
                                            (0.,0.,0.), "Sphere A"),))
    with pytest.raises(UnsupportedProblem, match="grounded solid sphere"):
        line(bad)
    bad2 = replace(p, charges=p.charges+(PointCharge(-1e-9, (0.,0.,1.)),))
    with pytest.raises(UnsupportedProblem, match="one charge"):
        line(bad2)
    with pytest.raises(ValueError):
        author_prescribed_grounded_sphere(True, (-.4,.2,0), .1)


@pytest.mark.parametrize("key,value", [
    ("time_s", False),
    ("time_s", 1.01),
    ("radial_distance_m", .4),
    ("radial_velocity_m_s", .7),
    ("source_position_m", (1., 0., 0.)),
    ("source_velocity_m_s", (0., 0., .2)),
    ("induced_sphere_net_charge_c", 0.),
    ("wire_current_sphere_to_ground_a", -1.),
    ("wire_current_ground_to_sphere_a", 1.),
    ("induced_force_on_source_n", (0., 0., 0.)),
    ("induced_mechanical_energy_j", 0.),
    ("index", False),
])
def test_native_export_rejects_replaced_rows_even_boolean_zero(key, value):
    result = sample_prescribed_sphere_line(line())
    old = result.samples[0]
    fake_row = replace(old, **{key: value})
    forged = replace(result, samples=(fake_row,)+result.samples[1:])
    with pytest.raises(ValueError, match="native re-execution"):
        validate_prescribed_sphere_line(forged)
    with pytest.raises(ValueError, match="native re-execution"):
        forged.as_dict()


@pytest.mark.parametrize("path,value", [
    (("source_role",), "original_jee_q21"),
    (("original_workbook_case",), "This was printed as off-axis"),
    (("sphere_constraint",), "FixedNetCharge"),
    (("sphere_is_solid",), 1),
    (("units", "current"), "mA"),
    (("sphere_name",), ""),
    (("source_charge_c",), -1e-9),
    (("sphere_radius_m",), .2),
    (("constant_cartesian_velocity_m_s", 0), -.2),
    (("closest_approach_time_s",), 1.1),
    (("closest_approach_distance_m",), .1),
    (("samples", 0, "time_s"), False),
    (("samples", 0, "index"), False),
    (("samples", 20, "source_position_m", 1), 0.),
    (("samples", 20, "wire_current_sphere_to_ground_a"), 5.),
    (("samples", 20, "induced_mechanical_energy_j"), False),
    (("samples", 5, "induced_force_on_source_n", 0), 3.),
])
def test_independent_stdlib_refuses_forged_kinematics_and_evidence(path, value):
    authentic = example_data()
    forged = deepcopy(authentic)
    node = forged
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError):
        check_line_record(forged)


def test_frozen_spec_mutation_does_not_bypass_physical_qualification():
    spec = line()
    object.__setattr__(spec, "duration_s", True)
    with pytest.raises(ValueError):
        sample_prescribed_sphere_line(spec)


def test_spherical_source_kinks_are_not_silent_static-orbit_claims():
    doc = example_data()
    assert "NOT Newtonian" in doc["protocol"]
    assert "teacher" in doc["source_role"]
    assert len({tuple(x["source_velocity_m_s"]) for x in doc["samples"]}) == 1
    assert all(x["wire_current_sphere_to_ground_a"]
               == -x["wire_current_ground_to_sphere_a"] for x in doc["samples"])
