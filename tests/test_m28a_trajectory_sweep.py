"""M28a: physical parameter sweeps, analytic kinematics, censoring, audit and HTML.

These validate new orchestration; independent M27b Coulomb/Verlet/DOP853
references retain responsibility for the unchanged charged-ring physics.
"""
from dataclasses import replace
import json
from math import isclose

import pytest

from conductor_sim import (
    FixedPointField, PointCharge, PrescribedField, TestChargeMotionSpec,
    UniformElectricField, SweepAxis, TrajectorySweepPlan,
    UniformRingField, render_trajectory_sweep, run_trajectory_sweep,
    write_trajectory_sweep_html,
)
from conductor_sim.scenarios import PrescribedScenario
from examples.m27b_offaxis_motion import author


def uniform_base():
    return PrescribedScenario(
        "m28:uniform",
        TestChargeMotionSpec(
            PrescribedField((UniformElectricField((0., 300., 0.)),)),
            PointCharge(1e-6, (0., 0., 0.)), .001, (.5, 0., 0.),
            .10, .01, plane_normal=(0., 0., 1.)),
        (("field:E", "Physical held electric field"),),
        "Teacher scenario; analytic constant acceleration reference.")


def test_p1_ring_charge_sweep_reuses_real_solver_and_immutable_source_model():
    base, _ = author("workbook-xy")
    original = base.as_dict()
    plan = TrajectorySweepPlan("m28:ring-charge", base,
                               SweepAxis("ring:B", "charge_c", (-18e-6, -22e-6)))
    result = run_trajectory_sweep(plan)
    assert base.as_dict() == original
    assert result.baseline_trajectory.spec is base.spec
    assert [e.status for e in result.entries] == ["complete", "complete"]
    for item, charge in zip(result.entries, plan.axis.values):
        assert item.scenario.parent_revision_id == base.revision_id
        assert item.scenario.spec.field.sources[1].charge_c == charge
        assert item.scenario.source_annotations == base.source_annotations
        assert item.trajectory.spec is item.scenario.spec
        assert item.trajectory.samples[-1].time_s == base.spec.duration_s
        assert item.trajectory.samples[-1].position_m[1] > 0
        assert item.scenario.origin == "teacher-hypothetical"
        assert len(item.scenario.changes) == 1
        assert item.scenario.changes[0].component_id == "ring:B"
        assert item.scenario.changes[0].property == "charge_c"
        assert item.trajectory.as_dict()["field"]["sources"][1]["charge_c"] == charge
    record = result.as_dict()
    assert record["schema"] == "conductor_sim.trajectory_sweep.v1"
    assert record["time_interpolation"] is False
    assert record["extrapolation"] is False
    assert [x["requested_value"] for x in record["entries"]] == [-18e-6, -22e-6]
    assert all(len(x["comparisons"]) == len(result.baseline_trajectory.samples)
               for x in record["entries"])
    record["entries"][0]["trajectory"]["samples"][0]["position_m"][0] = 900
    assert result.entries[0].trajectory.samples[0].position_m[0] != 900
    assert base.as_dict() == original


def test_p1_p2_ring_separation_and_particle_launch_requalified():
    base, _ = author("workbook-xy")
    center = run_trajectory_sweep(TrajectorySweepPlan(
        "m28:ring-separation", base, SweepAxis(
            "ring:B", "center_m", ((.35, 0., 0.), (.45, 0., 0.)))))
    assert all(e.status == "complete" for e in center.entries)
    assert [e.scenario.spec.field.sources[1].center_m[0] for e in center.entries] == [.35, .45]
    assert all(e.scenario.spec.particle.charge_c == 10e-6 for e in center.entries)
    launch = run_trajectory_sweep(TrajectorySweepPlan(
        "m28:ring-launch", base, SweepAxis(
            "test-particle", "initial_velocity_m_s",
            ((.08, 0., 0.), (.12, 0., 0.)))))
    assert all(e.status == "complete" for e in launch.entries)
    assert all(e.scenario.spec.field.sources == base.spec.field.sources for e in launch.entries)
    assert all(e.scenario.spec.relative_tolerance == base.spec.relative_tolerance
               for e in launch.entries)


def test_p3_analytic_constant_acceleration_multiple_masses_and_exact_times():
    base = uniform_base()
    result = run_trajectory_sweep(TrajectorySweepPlan(
        "m28:analytic", base, SweepAxis("test-particle", "mass_kg", (.0005, .002))))
    assert [x.status for x in result.entries] == ["complete", "complete"]
    for spec, trajectory in [(base.spec, result.baseline_trajectory)] + [
            (x.scenario.spec, x.trajectory) for x in result.entries]:
        acceleration = spec.particle.charge_c * 300. / spec.mass_kg
        for sample in trajectory.samples:
            t = sample.time_s
            assert sample.position_m[0] == pytest.approx(.5*t, abs=1e-11)
            assert sample.position_m[1] == pytest.approx(.5*acceleration*t*t, abs=1e-11)
            assert sample.velocity_m_s == pytest.approx((.5, acceleration*t, 0.), abs=1e-10)
            assert sample.acceleration_m_s2[1] == pytest.approx(acceleration, rel=1e-10)
    for item in result.entries:
        comparisons = item.as_dict(result.baseline_trajectory)["comparisons"]
        assert [x["time_s"] for x in comparisons] == [
            s.time_s for s in result.baseline_trajectory.samples]
        for row in comparisons:
            assert row["position_distance_m"] >= 0
            assert row["velocity_distance_m_s"] >= 0
            assert row["delta_position_m"][0] == pytest.approx(0, abs=1e-12)
            assert row["delta_position_m"][2] == pytest.approx(0, abs=1e-12)


def test_p4_source_guard_censor_and_no_extrapolated_shared_times():
    base = PrescribedScenario(
        "m28:guard-base",
        TestChargeMotionSpec(
            PrescribedField((FixedPointField(-1e-10, (.2, 0., 0.)),)),
            PointCharge(1e-9, (0., 0., 0.)), 1e-4, (0., 0., 0.),
            .25, .01, plane_normal=(0., 0., 1.), source_guard_m=.005),
        (("point:held", "Fixed attracting point"),),
        "Teacher hypothetical: ballistic approach into a fixed-source guard.")
    run = run_trajectory_sweep(TrajectorySweepPlan(
        "m28:guard", base,
        SweepAxis("test-particle", "initial_velocity_m_s", ((1., 0., 0.),))))
    entry = run.entries[0]
    assert run.baseline_trajectory.termination == "duration"
    assert entry.status == "censored", entry.explanation
    assert entry.trajectory.termination == "source-guard"
    assert entry.trajectory.samples[-1].time_s < base.spec.duration_s
    comparisons = entry.as_dict(run.baseline_trajectory)["comparisons"]
    assert comparisons
    assert max(row["time_s"] for row in comparisons) < entry.trajectory.samples[-1].time_s
    assert max(row["time_s"] for row in comparisons) < base.spec.duration_s
    assert "censored" in render_trajectory_sweep(run)


def test_p4_invalid_ring_model_refusal_preserves_other_variations():
    base, _ = author("workbook-xy")
    before = base.as_dict()
    out = run_trajectory_sweep(TrajectorySweepPlan(
        "m28:refusal", base, SweepAxis(
            "ring:B", "center_m", ((.4, 0., 1e-4), (.35, 0., 0.)))))
    assert out.entries[0].status == "rejected-physical-model"
    assert out.entries[0].scenario is None and out.entries[0].trajectory is None
    assert "full-ring centre" in out.entries[0].explanation
    assert out.entries[1].status == "complete"
    assert out.entries[1].scenario.parent_revision_id == base.revision_id
    assert out.as_dict()["entries"][0]["comparisons"] == []
    assert base.as_dict() == before


def test_n2_oblique_ring_axis_and_invalid_radius_fail_closed():
    base, _ = author("workbook-xy")
    axis = run_trajectory_sweep(TrajectorySweepPlan(
        "m28:oblique", base, SweepAxis(
            "ring:B", "axis", ((1., 0., 1e-4),))))
    assert axis.entries[0].status == "rejected-physical-model"
    assert "oblique" in axis.entries[0].explanation
    radii = run_trajectory_sweep(TrajectorySweepPlan(
        "m28:radii", base, SweepAxis("ring:A", "radius_m", (-.2, .31))))
    assert radii.entries[0].status == "rejected-physical-model"
    assert radii.entries[1].status == "complete"


@pytest.mark.parametrize("axis", [
    SweepAxis("test-particle", "mass_kg", (2e-3, 3e-3)),
    SweepAxis("test-particle", "initial_velocity_m_s", ((.3, 0., 0.),)),
    SweepAxis("field:E", "field_v_m", ((0., 500., 0.),)),
    SweepAxis("medium", "permittivity_f_m", (9e-12,)),
])
def test_generic_supported_targets_reuse_built_in_m20b_revision(axis):
    base = uniform_base()
    out = run_trajectory_sweep(TrajectorySweepPlan("m28:generic", base, axis))
    assert all(x.status == "complete" for x in out.entries)
    assert all(x.scenario.parent_revision_id == base.revision_id for x in out.entries)


def test_n1_invalid_targets_values_numerical_controls_and_duplicate_baseline():
    base = uniform_base()
    with pytest.raises(ValueError, match="baseline value"):
        TrajectorySweepPlan("m28:dup", base, SweepAxis("test-particle", "mass_kg", (.001,)))
    with pytest.raises(ValueError, match="physical"):
        TrajectorySweepPlan("m28:numerical", base, SweepAxis(
            "test-particle", "source_guard_m", (.002,)))
    with pytest.raises(ValueError, match="unknown"):
        TrajectorySweepPlan("m28:unknown", base, SweepAxis("source:1", "charge_c", (2e-6,)))
    with pytest.raises(TypeError, match="shape"):
        TrajectorySweepPlan("m28:shape", base, SweepAxis(
            "field:E", "field_v_m", (3.,)))
    with pytest.raises(ValueError, match="duplicate"):
        SweepAxis("test-particle", "mass_kg", (2e-3, 2e-3))
    with pytest.raises(ValueError, match="1..6"):
        SweepAxis("test-particle", "mass_kg", ())
    with pytest.raises(ValueError, match="1..6"):
        SweepAxis("test-particle", "mass_kg", (1., 2., 3., 4., 5., 6., 7.))
    with pytest.raises(TypeError, match="finite"):
        SweepAxis("test-particle", "mass_kg", (True,))
    with pytest.raises(TypeError, match="finite"):
        SweepAxis("test-particle", "mass_kg", (float("nan"),))


def test_p5_p6_offline_html_displays_native_states_and_escaped_titles(tmp_path):
    base, _ = author("workbook-xy")
    run = run_trajectory_sweep(TrajectorySweepPlan(
        "m28:html", base, SweepAxis("ring:B", "charge_c", (-18e-6,))))
    html = render_trajectory_sweep(run, title="<M28a trajectory comparison>")
    assert "&lt;M28a trajectory comparison&gt;" in html
    assert '<script type="application/json" id="m28-sweep-data">' in html
    assert "no browser physics" in html
    assert "source-guard" in html
    assert "probe q=+10 µC" in html
    assert 'window.m28SweepViewer=viewer;' in html
    assert "frame_times_s" in html
    assert "time_interpolation" in html
    assert "ring:A" in html and "ring:B" in html
    with pytest.raises(TypeError, match="native"):
        render_trajectory_sweep(run.as_dict())
    file = write_trajectory_sweep_html(run, tmp_path/"sweeps"/"view.html")
    assert file.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert json.dumps(run.as_dict(), allow_nan=False)


def test_p3_unfamiliar_fixed_point_and_uniform_field_rotate_covariantly():
    """New M28a transfer: same actual physical experiment after a rigid 90-degree rotation.

    It is deliberately not one of the M27b reserved ring cards. No new
    production solver or field formula is introduced for this case.
    """
    def rotate(v):
        return (-v[1], v[0], v[2])

    source=(.075, 0., -.045)
    electric=(1.4, 0., -.8)
    particle=(.025, 0., .11)
    velocity=(.062, 0., -.042)
    normal=(0., 1., 0.)

    def case(rotated):
        position=rotate(particle) if rotated else particle
        speed=rotate(velocity) if rotated else velocity
        held=rotate(source) if rotated else source
        held_e=rotate(electric) if rotated else electric
        plane=rotate(normal) if rotated else normal
        spec=TestChargeMotionSpec(
            PrescribedField((FixedPointField(2e-10, held),
                             UniformElectricField(held_e))),
            PointCharge(-5.2e-10, position), 4.2e-7, speed,
            .085, .0085, plane_normal=plane, source_guard_m=.006,
            relative_tolerance=1e-10)
        baseline=PrescribedScenario(
            "m28:rotated" if rotated else "m28:unrotated", spec,
            (("point:held", "Held point P"), ("field:held", "Held E")),
            "Teacher hypothetical: unfamiliar point and mixed uniform field.")
        return run_trajectory_sweep(TrajectorySweepPlan(
            "m28:rot90" if rotated else "m28:original",
            baseline, SweepAxis("point:held", "charge_c", (2.4e-10, 2.8e-10))))

    original, turned=case(False), case(True)
    assert [x.status for x in original.entries] == ["complete", "complete"]
    assert [x.status for x in turned.entries] == ["complete", "complete"]
    for native, rotated in [(original.baseline_trajectory, turned.baseline_trajectory)] + [
        (a.trajectory, b.trajectory) for a, b in zip(original.entries, turned.entries)]:
        assert len(native.samples)==len(rotated.samples)
        for a, b in zip(native.samples, rotated.samples):
            assert b.time_s == a.time_s
            assert b.position_m == pytest.approx(rotate(a.position_m), abs=2e-11)
            assert b.velocity_m_s == pytest.approx(rotate(a.velocity_m_s), abs=2e-10)
            assert b.force_n == pytest.approx(rotate(a.force_n), abs=2e-15)
            assert b.total_energy_j == pytest.approx(a.total_energy_j, abs=2e-14)
    assert original.plan.baseline.revision_id != turned.plan.baseline.revision_id
