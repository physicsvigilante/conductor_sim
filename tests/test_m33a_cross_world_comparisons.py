"""M33a: independent world-pair comparisons, original physics references and refusals."""
from dataclasses import replace
from math import sqrt

import pytest

from conductor_sim import (
    CoupledPropertyEdit, PhysicalWorldNode, StructuralWorldNode,
    UnifiedObservationPlan, WorldComparison, run_unified_observation_graph,
)
from conductor_sim.motion_events import EventPolicy, PlaneEvent
from conductor_sim.structural_family import StructuralCase
from conductor_sim.structural_scenarios import NamedSource
from conductor_sim.prescribed_fields import UniformElectricField
from examples.m29a_structural_family import model


def plan(*, comparisons=(), modes=("events", "events"), ancestry=True):
    original = model("uniform-edits").baseline
    nodes = (
        StructuralWorldNode("left", "baseline", StructuralCase(
            "left", "Teacher alternative +x field",
            add_sources=(NamedSource("field:extra", "New +x field",
                                     UniformElectricField((40., 0., 0.))),)), modes[0]),
        StructuralWorldNode("right", "baseline", StructuralCase(
            "right", "Independent teacher alternative +x field",
            add_sources=(NamedSource("field:extra", "Another hypothesis +x field",
                                     UniformElectricField((120., 0., 0.))),)), modes[1]),
        PhysicalWorldNode("descendant", "left", "Vary actual probe mass",
            (CoupledPropertyEdit("test-particle", "mass_kg", 2e-5),), "events"),
    )
    return UnifiedObservationPlan(
        "m33a:alternatives", original, nodes, "events",
        (PlaneEvent("x:detector", (.006, 0., 0.), (1., 0., 0.),
                    "Passive signed x crossing", direction=1),),
        EventPolicy(.003, 1e-8, 1e-8, 1e-6),
        retain_source_ancestry=ancestry, comparisons=comparisons)


def comparisons():
    return (
        WorldComparison("siblings", "left", "right"),
        WorldComparison("cousins", "right", "descendant"),
        WorldComparison("reference", "baseline", "right"),
    )


def analytic_position_and_velocity(spec, t):
    # Independent Newtonian uniform-field expression in SI: NOT production _shared_rows.
    ex = tuple(sum(src.field_v_m[k] for src in spec.field.sources)
               for k in range(3))
    g = (0., 0., 0.) if spec.gravity is None else spec.gravity.acceleration_m_s2
    a = tuple(spec.particle.charge_c * ex[k] / spec.mass_kg + g[k]
              for k in range(3))
    pos = tuple(spec.particle.position_m[k] +
                spec.initial_velocity_m_s[k] * t + .5 * a[k] * t*t
                for k in range(3))
    vel = tuple(spec.initial_velocity_m_s[k] + a[k]*t for k in range(3))
    return pos, vel


def analytic_positive_detector_root(spec):
    ex = sum(s.field_v_m[0] for s in spec.field.sources)
    a = spec.particle.charge_c * ex / spec.mass_kg
    v = spec.initial_velocity_m_s[0]
    dx = .006 - spec.particle.position_m[0]
    return 2*dx/(v+sqrt(v*v+2*a*dx))


def test_cross_branch_reference_and_real_event_roots():
    p = plan(comparisons=comparisons())
    before = p.original.as_dict()
    run = run_unified_observation_graph(p)
    native = run.as_dict()
    assert native["schema"] == "conductor_sim.unified_observation_graph.v3"
    assert native["plan"]["schema"] == "conductor_sim.unified_observation_plan.v3"
    pairs = native["cross_world_comparisons"]
    assert [x["status"] for x in pairs] == ["compared"]*3
    assert [x["comparison_id"] for x in pairs] == ["siblings", "cousins", "reference"]
    worlds = {"baseline": (p.original, run.baseline_native)}
    worlds.update((e.node.node_id, (e.scenario, e.native)) for e in run.entries)
    for pair in pairs:
        left, left_native = worlds[pair["left_world_id"]]
        right, right_native = worlds[pair["right_world_id"]]
        assert pair["left_revision_id"] == left.revision_id
        assert pair["right_revision_id"] == right.revision_id
        assert pair["left_source_ancestry"]["model_revision_id"] == left.revision_id
        assert pair["right_source_ancestry"]["model_revision_id"] == right.revision_id
        assert pair["exact_time_trajectory_comparisons"]
        left_times = {s.time_s for s in left_native.trajectory.samples}
        right_times = {s.time_s for s in right_native.trajectory.samples}
        assert [row["time_s"] for row in pair["exact_time_trajectory_comparisons"]] == [
            s.time_s for s in right_native.trajectory.samples if s.time_s in left_times]
        for row in pair["exact_time_trajectory_comparisons"]:
            t = row["time_s"]
            lp, lv = analytic_position_and_velocity(left.spec, t)
            rp, rv = analytic_position_and_velocity(right.spec, t)
            assert row["delta_position_m"] == pytest.approx(
                tuple(rp[k]-lp[k] for k in range(3)), abs=1e-7)
            assert row["delta_velocity_m_s"] == pytest.approx(
                tuple(rv[k]-lv[k] for k in range(3)), abs=1e-7)
        root = pair["event_time_comparisons"][0]
        assert root["status"] == "one-resolved-pair"
        expected = (analytic_positive_detector_root(right.spec) -
                    analytic_positive_detector_root(left.spec))
        assert root["estimated_numerical_shift_s"] == pytest.approx(expected, abs=1e-7)
    assert worlds["left"][0].revision_id != worlds["right"][0].revision_id
    assert pairs[0]["left_source_ancestry"]["introduced_source_ids"] == ["field:extra"]
    assert pairs[0]["right_source_ancestry"]["introduced_source_ids"] == ["field:extra"]
    native["cross_world_comparisons"][0]["status"] = "fabricated"
    assert run.as_dict()["cross_world_comparisons"][0]["status"] == "compared"
    assert p.original.as_dict() == before


def test_missing_mode_and_physically_rejected_world_never_pair():
    p = plan(comparisons=(WorldComparison("different", "left", "right"),),
             modes=("events", "trajectory"))
    rec = run_unified_observation_graph(p).as_dict()["cross_world_comparisons"][0]
    assert rec["status"] == "unpaired"
    assert "different" in rec["reason"]
    assert not rec["exact_time_trajectory_comparisons"]
    assert not rec["event_time_comparisons"]

    original = model("uniform-edits").baseline
    nodes = (
        PhysicalWorldNode("bad", "baseline", "Physically invalid negative mass",
                          (CoupledPropertyEdit("test-particle", "mass_kg", -1.),)),
        PhysicalWorldNode("good", "baseline", "Valid mass",
                          (CoupledPropertyEdit("test-particle", "mass_kg", 2e-5),)),
    )
    rejected = UnifiedObservationPlan("m33a:reject", original, nodes,
        retain_source_ancestry=True,
        comparisons=(WorldComparison("missing", "bad", "good"),))
    output = run_unified_observation_graph(rejected).as_dict()
    assert output["nodes"][0]["admitted_scenario"] is None
    c = output["cross_world_comparisons"][0]
    assert c["status"] == "unpaired"
    assert c["left_revision_id"] is None
    assert c["left_source_ancestry"] is None
    assert c["right_revision_id"] is not None
    assert c["exact_time_trajectory_comparisons"] == []


@pytest.mark.parametrize("bad", [
    (WorldComparison("x", "left", "right"), WorldComparison("x", "baseline", "left")),
    (WorldComparison("x", "left", "right"), WorldComparison("y", "right", "left")),
    (WorldComparison("unknown", "left", "absent"),),
])
def test_duplicate_or_unknown_requests_refused(bad):
    with pytest.raises((ValueError, TypeError)):
        plan(comparisons=bad)


def test_invalid_requests_and_legacy_v1_v2_shapes():
    with pytest.raises(ValueError):
        WorldComparison("same", "baseline", "baseline")
    with pytest.raises(ValueError):
        WorldComparison("bool", True, "baseline")
    with pytest.raises(ValueError):
        WorldComparison("bad space", "baseline", "left")
    with pytest.raises(TypeError):
        plan(comparisons=(("left", "right"),))
    with pytest.raises(ValueError, match="retained source ancestry"):
        plan(comparisons=(WorldComparison("pair", "left", "right"),), ancestry=False)
    without = plan()
    assert without.as_dict()["schema"] == "conductor_sim.unified_observation_plan.v2"
    data = run_unified_observation_graph(without).as_dict()
    assert data["schema"] == "conductor_sim.unified_observation_graph.v2"
    assert "cross_world_comparisons" not in data
    assert "comparisons" not in data["plan"]
    legacy = plan(ancestry=False)
    assert legacy.as_dict()["schema"] == "conductor_sim.unified_observation_plan.v1"
    assert run_unified_observation_graph(legacy).as_dict()["schema"] == (
        "conductor_sim.unified_observation_graph.v1")


def test_native_result_and_request_tampering_refused():
    p = plan(comparisons=(WorldComparison("siblings", "left", "right"),))
    run = run_unified_observation_graph(p)
    with pytest.raises(ValueError, match="receipt"):
        replace(run, baseline_reason="forged")
    with pytest.raises((ValueError, TypeError)):
        replace(run, plan=replace(p, comparisons=(
            WorldComparison("reverse", "right", "left"),)))
    with pytest.raises((ValueError, TypeError)):
        replace(run, entries=(replace(run.entries[0], status="forged"),)+run.entries[1:])
