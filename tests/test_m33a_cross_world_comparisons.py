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


def test_genuine_two_axis_field_difference_not_projected_into_x():
    original = model("uniform-edits").baseline
    nodes = tuple(StructuralWorldNode(
        name, "baseline", StructuralCase(name, "Independent planar vector field",
            add_sources=(NamedSource("field:vector", "Held planar vector",
                UniformElectricField(vector)),)), "trajectory")
        for name, vector in (
            ("planar-a", (25., 17., 0.)),
            ("planar-b", (-15., -23., 0.))))
    p = UnifiedObservationPlan("m33a:planar", original, nodes,
        retain_source_ancestry=True,
        comparisons=(WorldComparison("planar", "planar-a", "planar-b"),))
    run = run_unified_observation_graph(p)
    assert [e.status for e in run.entries] == ["complete", "complete"]
    pair = run.as_dict()["cross_world_comparisons"][0]
    assert pair["status"] == "compared"
    assert pair["event_time_comparisons"] == []
    for row in pair["exact_time_trajectory_comparisons"]:
        t = row["time_s"]
        expected = tuple(
            .5 * original.spec.particle.charge_c / original.spec.mass_kg *
            (delta) * t * t
            for delta in (-40., -40., 0.))
        assert row["delta_position_m"] == pytest.approx(expected, abs=1e-8)
        assert row["delta_position_m"][1] != 0 or t == 0
        assert row["delta_position_m"][2] == pytest.approx(0., abs=1e-10)


def test_cross_branch_source_guard_does_not_fabricate_complete_event_pair():
    from examples.m29b_structural_events import make_plan
    authored = make_plan("guard-detector")
    attr = next(c for c in authored.family.cases if c.case_id == "attract-guard")
    proposed = UnifiedObservationPlan(
        "m33a:guard", authored.family.baseline,
        (StructuralWorldNode("attract-guard", "baseline", attr, "events"),),
        "events", authored.questions, authored.policy,
        retain_source_ancestry=True,
        comparisons=(WorldComparison("guard", "baseline", "attract-guard"),))
    run = run_unified_observation_graph(proposed)
    assert run.entries[0].status == "censored-source-guard"
    comp = run.as_dict()["cross_world_comparisons"][0]
    assert comp["status"] == "compared"
    assert comp["exact_time_trajectory_comparisons"]
    assert all(row["time_s"] <= run.entries[0].native.trajectory.samples[-1].time_s
               for row in comp["exact_time_trajectory_comparisons"])
    assert all(e["status"] == "unpaired" and
               "estimated_numerical_shift_s" not in e
               for e in comp["event_time_comparisons"])


def test_cross_branch_admitted_but_event_refused_keeps_model_and_no_fake_observation():
    from examples.m29b_structural_events import make_plan
    old = make_plan("terminal-refusal")
    bad_study = StructuralWorldNode(
        "reversed-force", "baseline", old.family.cases[0], "events")
    correction = PhysicalWorldNode(
        "correct-stop", "reversed-force", "Real physical correction",
        (CoupledPropertyEdit("field:Eplus", "field_v_m", (500., 0., 0.)),
         CoupledPropertyEdit("test-particle", "mass_kg", 2e-6)), "events")
    proposed = UnifiedObservationPlan(
        "m33a:study-refusal", old.family.baseline,
        (bad_study, correction), "events", old.questions, old.policy,
        retain_source_ancestry=True,
        comparisons=(WorldComparison("refusal", "reversed-force", "correct-stop"),))
    run = run_unified_observation_graph(proposed)
    assert run.entries[0].status == "rejected-event-study"
    assert run.entries[0].scenario is not None
    assert run.entries[0].native is None
    assert run.entries[1].status == "complete"
    record = run.as_dict()["cross_world_comparisons"][0]
    assert record["status"] == "unpaired"
    assert "no native observation" in record["reason"]
    assert record["left_revision_id"] == run.entries[0].scenario.revision_id
    assert record["left_source_ancestry"]["world_id"] == "reversed-force"
    assert record["right_source_ancestry"]["world_id"] == "correct-stop"
    assert record["event_time_comparisons"][0]["status"] == "unpaired"
    assert record["exact_time_trajectory_comparisons"] == []


def test_cross_branch_two_real_opposite_signed_roots_never_unique_event_pair():
    from examples.m29b_structural_events import make_plan
    old = make_plan("double-crossing")
    p = UnifiedObservationPlan(
        "m33a:bidirectional", old.family.baseline,
        (StructuralWorldNode("extra-retarding", "baseline",
                             old.family.cases[0], "events"),),
        "events", old.questions, old.policy,
        retain_source_ancestry=True,
        comparisons=(WorldComparison("two-roots", "baseline", "extra-retarding"),))
    run = run_unified_observation_graph(p)
    comp = run.as_dict()["cross_world_comparisons"][0]
    assert comp["status"] == "compared"
    assert comp["event_time_comparisons"][0]["status"] == "unpaired"
    assert "multiple crossings" in comp["event_time_comparisons"][0]["reason"]
    assert all(len([e for e in native.events if e.event_id == "x:two-crossings"]) == 2
               for native in (run.baseline_native, run.entries[0].native))


def test_cross_branch_single_opposite_direction_events_are_distinct():
    original = model("uniform-edits").baseline
    question = PlaneEvent("x:both", (.006, 0., 0.), (1., 0., 0.),
                          "Bidirectional crossing of the same physical plane",
                          direction=0)
    p = UnifiedObservationPlan(
        "m33a:opposite", original,
        (PhysicalWorldNode("reverse", "baseline",
            "Real changed initial position and velocity", (
                CoupledPropertyEdit("test-particle", "position_m", (.012, 0., 0.)),
                CoupledPropertyEdit("test-particle", "initial_velocity_m_s",
                                    (-.1, 0., 0.))), "events"),),
        "events", (question,), EventPolicy(.003, 1e-8, 1e-8, 1e-6),
        retain_source_ancestry=True,
        comparisons=(WorldComparison("opposite", "baseline", "reverse"),))
    run = run_unified_observation_graph(p)
    assert run.entries[0].status == "complete"
    assert len(run.baseline_native.events) == 1
    assert len(run.entries[0].native.events) == 1
    assert run.baseline_native.events[0].direction == 1
    assert run.entries[0].native.events[0].direction == -1
    pair = run.as_dict()["cross_world_comparisons"][0]
    assert pair["status"] == "compared"
    assert pair["event_time_comparisons"][0]["status"] == "unpaired"
    assert "opposite signed" in pair["event_time_comparisons"][0]["reason"]
    assert "estimated_numerical_shift_s" not in pair["event_time_comparisons"][0]


def test_unfamiliar_xz_world_3d_native_forces_and_cross_branch_event_roots():
    """Independent new x-z qE+mg reference; no production force callback oracle."""
    from conductor_sim import (
        PointCharge, PrescribedField, TestChargeMotionSpec, UniformGravity,
    )
    from conductor_sim.scenarios import PrescribedScenario

    original = PrescribedScenario(
        "m33a:xz-root",
        TestChargeMotionSpec(
            PrescribedField((UniformElectricField((210., 0., -70.)),)),
            PointCharge(1.3e-9, (0., 0., .05)),
            1.2e-6, (.11, 0., .24), .5, .005,
            plane_normal=(0., 1., 0.),
            gravity=UniformGravity((0., 0., -.16))),
        (("field:base", "Original tilted x-z electric field"),),
        "Teacher-authored physical x-z launch with held field and uniform gravity.")
    nodes = (
        StructuralWorldNode("tilted-a", "baseline", StructuralCase(
            "tilted-a", "Introduce first independent x-z field hypothesis.",
            add_sources=(NamedSource("field:tilt", "New first tilted field",
                UniformElectricField((80., 0., 40.))),)), "events"),
        StructuralWorldNode("tilted-b", "baseline", StructuralCase(
            "tilted-b", "Introduce second independent x-z field hypothesis.",
            add_sources=(NamedSource("field:tilt", "New second tilted field",
                UniformElectricField((-60., 0., 110.))),)), "events"),
        PhysicalWorldNode("more-mass", "tilted-a", "Independently qualified heavy probe",
            (CoupledPropertyEdit("test-particle", "mass_kg", 2e-6),), "events"),
    )
    p = UnifiedObservationPlan(
        "m33a:xz", original, nodes, "events",
        (PlaneEvent("x:positive", (.06, 0., 0.), (1., 0., 0.),
                    "Passive signed x-z plane crossing", direction=1),),
        EventPolicy(.005, 1e-8, 1e-8, 1e-6),
        retain_source_ancestry=True,
        comparisons=(
            WorldComparison("xz-siblings", "tilted-a", "tilted-b"),
            WorldComparison("xz-cousins", "tilted-b", "more-mass"),
        ))
    before = original.as_dict()
    result = run_unified_observation_graph(p)
    assert result.baseline_status == "complete"
    assert all(entry.status == "complete" for entry in result.entries)
    worlds = {"baseline": (original, result.baseline_native)}
    worlds.update((x.node.node_id, (x.scenario, x.native)) for x in result.entries)
    def exact_state(spec, t):
        electric = tuple(sum(source.field_v_m[k] for source in spec.field.sources)
                         for k in range(3))
        gravity = spec.gravity.acceleration_m_s2
        q, mass = spec.particle.charge_c, spec.mass_kg
        a = tuple(q*electric[k]/mass+gravity[k] for k in range(3))
        pos = tuple(spec.particle.position_m[k]+spec.initial_velocity_m_s[k]*t+
                    .5*a[k]*t*t for k in range(3))
        vel = tuple(spec.initial_velocity_m_s[k]+a[k]*t for k in range(3))
        force = tuple(q*electric[k]+mass*gravity[k] for k in range(3))
        return pos,vel,force
    def analytic_root(spec):
        electric_x = sum(source.field_v_m[0] for source in spec.field.sources)
        ax = spec.particle.charge_c*electric_x/spec.mass_kg
        vx = spec.initial_velocity_m_s[0]
        d = .06-spec.particle.position_m[0]
        return 2*d/(vx+sqrt(vx*vx+2*ax*d))
    for name, (scenario, native) in worlds.items():
        assert native.trajectory.termination == "duration", name
        assert native.events[0].direction == 1
        assert native.events[0].sample.time_s == pytest.approx(
            analytic_root(scenario.spec), abs=2e-6)
        assert native.trajectory.samples[-1].position_m[2] != 0
        for sample in native.trajectory.samples:
            pos,vel,force=exact_state(scenario.spec,sample.time_s)
            assert sample.position_m == pytest.approx(pos,abs=2e-7)
            assert sample.velocity_m_s == pytest.approx(vel,abs=2e-7)
            assert sample.force_n == pytest.approx(force,abs=2e-12)
            assert sample.position_m[1] == pytest.approx(0.,abs=1e-10)
    output = result.as_dict()
    assert [x["status"] for x in output["cross_world_comparisons"]] == [
        "compared","compared"]
    for comparison in output["cross_world_comparisons"]:
        left = worlds[comparison["left_world_id"]][0].spec
        right = worlds[comparison["right_world_id"]][0].spec
        rows = comparison["exact_time_trajectory_comparisons"]
        assert rows
        assert any(abs(row["delta_position_m"][2])>1e-8
                   for row in rows if row["time_s"]>0)
        for row in rows:
            t = row["time_s"]
            lp,lv,_ = exact_state(left,t)
            rp,rv,_ = exact_state(right,t)
            assert row["delta_position_m"] == pytest.approx(
                tuple(rp[k]-lp[k] for k in range(3)),abs=4e-7)
            assert row["delta_velocity_m_s"] == pytest.approx(
                tuple(rv[k]-lv[k] for k in range(3)),abs=4e-7)
        signed = comparison["event_time_comparisons"][0]
        assert signed["status"] == "one-resolved-pair"
        assert signed["direction"] == 1
        assert signed["estimated_numerical_shift_s"] == pytest.approx(
            analytic_root(right)-analytic_root(left),abs=4e-6)
    assert output["cross_world_comparisons"][0]["left_source_ancestry"][
        "introduced_source_ids"] == ["field:tilt"]
    assert output["cross_world_comparisons"][0]["right_source_ancestry"][
        "introduced_source_ids"] == ["field:tilt"]
    assert original.as_dict() == before
