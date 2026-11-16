"""M31b event orchestration checks use real M24a native event integrations."""
from dataclasses import replace
from math import sqrt

import pytest

from conductor_sim import (
    CoupledPropertyEdit, CoupledHypothesis, CoupledPhysicalPlan,
    CoupledEventPlan, CoupledEventRun, EventPolicy, PlaneEvent,
    run_coupled_event_family,
)
from examples.m29a_structural_family import model


def _plan(*hypotheses, family=None):
    return CoupledPhysicalPlan(
        "m31b:events", model("uniform-edits") if family is None else family,
        tuple(hypotheses),
    )


def _hypothesis(name, branch, *edits):
    return CoupledHypothesis(name, branch, "Teacher-authored atomic physical hypothesis.", tuple(
        CoupledPropertyEdit(identity, prop, value) for identity, prop, value in edits))


def _policy():
    return EventPolicy(.005, 2e-7, 1e-7, 1e-7)


def _detector(x=.006, direction=1):
    return PlaneEvent("x-detector", (x, 0., 0.), (1., 0., 0.),
                      "Passive x detector", direction=direction)


def _run(*hypotheses, family=None, x=.006):
    return run_coupled_event_family(CoupledEventPlan(
        _plan(*hypotheses, family=family), (_detector(x),), _policy()))


def _analytic_positive_time(x, vx, accel):
    return (2*x)/(vx+sqrt(vx*vx+2*accel*x))


def test_m31b_true_new_source_plus_mass_event_studies_and_analytic_crossing():
    run = _run(_hypothesis(
        "changed-source-and-mass", "add-E",
        ("field:Enew", "field_v_m", (150., 0., 0.)),
        ("test-particle", "mass_kg", 2e-5)))
    parent = run.parents[0]
    child = run.children[0]
    assert parent.branch_id == "add-E"
    assert parent.parent_scenario is child.physical_entry.parent_scenario
    assert child.event_run.study.scenario is child.physical_entry.scenario
    assert child.event_run is not parent.event_run
    assert parent.status == "complete"
    assert child.status == "complete"
    assert child.physical_entry.status == "complete"
    assert child.physical_entry.scenario.parent_revision_id == parent.parent_scenario.revision_id
    assert parent.event_run.study.scenario.spec.field.sources[-1].field_v_m == (100., 0., 0.)
    assert child.event_run.study.scenario.spec.field.sources[-1].field_v_m == (150., 0., 0.)
    q = child.physical_entry.scenario.spec.particle.charge_c
    for event_run in (run.original_event_run, parent.event_run, child.event_run):
        spec = event_run.study.scenario.spec
        ex = sum(field.field_v_m[0] for field in spec.field.sources)
        expected = _analytic_positive_time(.006, spec.initial_velocity_m_s[0],
                                            q*ex/spec.mass_kg)
        roots = [ev for ev in event_run.events if ev.event_id == "x-detector"]
        assert len(roots) == 1
        assert roots[0].direction == 1
        assert roots[0].time_bracket_s[0]-2e-7 <= expected <= roots[0].time_bracket_s[1]+2e-7
    detached = run.as_dict()
    assert detached["schema"] == "conductor_sim.coupled_event_family.v1"
    comparison = detached["coupled_events"][0]["parent_event_time_comparisons"][0]
    assert comparison["status"] == "one-resolved-pair"
    assert detached["time_interpolation"] is False
    assert detached["postguard_extrapolation"] is False


def test_m31b_jointly_admissible_point_and_probe_shift_has_genuine_events():
    run = _run(_hypothesis(
        "shift", "baseline",
        ("test-particle", "position_m", (0., .1, .05)),
        ("point:A", "position_m", (.2, 0., .05))),
        family=model("point-replace"), x=.001)
    parent = run.parents[0]
    child = run.children[0]
    assert parent.event_run is run.original_event_run
    assert child.physical_entry.status == "complete"
    assert child.event_run.study.scenario is child.physical_entry.scenario
    assert child.event_run.trajectory.samples[0].position_m[2] == pytest.approx(.05)
    assert len(child.event_run.trajectory.samples) > 1
    assert run.as_dict()["coupled_events"][0]["event_run"] is not None


def test_m31b_refused_edit_does_not_fabricate_event_or_suppress_good_sibling():
    result = _run(
        _hypothesis("gauge-refusal", "add-E",
                    ("field:Enew", "reference_potential_v", 1.),
                    ("test-particle", "mass_kg", 2e-5)),
        _hypothesis("good-sibling", "add-E",
                    ("field:Enew", "field_v_m", (150., 0., 0.)),
                    ("test-particle", "mass_kg", 2e-5)))
    assert [x.physical_entry.status for x in result.children] == [
        "rejected-coupled-edit", "complete"]
    assert result.children[0].event_run is None
    assert result.children[0].status == "rejected-coupled-edit"
    assert result.children[1].status == "complete"
    assert len(result.parents) == 1
    detached = result.as_dict()
    assert detached["coupled_events"][0]["parent_event_time_comparisons"][0]["status"] == "unpaired"


def test_m31b_rejects_fabricated_events_wrong_source_and_unknown_termination():
    result = _run(_hypothesis(
        "added-mass", "add-E",
        ("field:Enew", "field_v_m", (150., 0., 0.)),
        ("test-particle", "mass_kg", 2e-5)))
    child = result.children[0]
    bad_source = replace(child, event_run=result.original_event_run)
    with pytest.raises(ValueError, match="native observation"):
        CoupledEventRun(result.plan, result.physical_family, result.original_event_run,
                        result.parents, (bad_source,))
    altered = replace(child.event_run.trajectory, termination="not-physical")
    bad_term = replace(child, event_run=replace(child.event_run, trajectory=altered))
    with pytest.raises(ValueError, match="unknown event history termination"):
        CoupledEventRun(result.plan, result.physical_family, result.original_event_run,
                        result.parents, (bad_term,))
    class FakeEventRun:
        def as_dict(self):
            return {"schema": "forged", "events": []}
    bad = replace(child, event_run=FakeEventRun())
    with pytest.raises(ValueError, match="native observation"):
        CoupledEventRun(result.plan, result.physical_family, result.original_event_run,
                        result.parents, (bad,))


def test_m31b_plan_rejects_untyped_or_duplicate_detector_questions_before_run():
    phys = _plan(_hypothesis(
        "added-mass", "add-E",
        ("field:Enew", "field_v_m", (150., 0., 0.)),
        ("test-particle", "mass_kg", 2e-5)))
    with pytest.raises(TypeError):
        CoupledEventPlan(phys, [_detector()], _policy())
    with pytest.raises(ValueError, match="duplicate"):
        CoupledEventPlan(phys, (_detector(), _detector()), _policy())
    with pytest.raises(TypeError):
        run_coupled_event_family({"not": "a native plan"})
