"""M31b event orchestration checks use real M24a native event integrations."""
from dataclasses import replace
from math import sqrt

import pytest

from conductor_sim import (
    CoupledPropertyEdit, CoupledHypothesis, CoupledPhysicalPlan,
    CoupledEventPlan, CoupledEventRun,
    run_coupled_event_family,
)
from conductor_sim.motion_events import EventPolicy, PlaneEvent
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
        "source-mass", "add-E",
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


def test_m31b_true_source_guard_early_detector_and_unobserved_late_event():
    from examples.m29b_structural_events import make_plan
    old = make_plan("guard-detector")
    result = run_coupled_event_family(CoupledEventPlan(
        _plan(_hypothesis(
            "guard-charge-mass", "attract-guard",
            ("point:attract", "charge_c", -2e-10),
            ("test-particle", "mass_kg", 2e-4)),
            family=old.family), old.questions, old.policy))
    parent = result.parents[0]
    child = result.children[0]
    assert parent.event_run.trajectory.termination == "source-guard"
    assert parent.status == "censored-source-guard"
    assert child.event_run is not None
    observed = parent.event_run.as_dict()["outcomes"]
    assert observed["x:early"] == "located"
    assert observed["x:far"] == "search-truncated"
    evidence = result.as_dict()["coupled_events"][0]
    assert [c["status"] for c in evidence["parent_event_time_comparisons"]] == [
        "unpaired", "unpaired"]
    assert "estimated_numerical_shift_s" not in evidence["parent_event_time_comparisons"][0]
    assert evidence["event_run"]["trajectory"]["samples"][-1]["time_s"] <= (
        child.event_run.study.scenario.spec.duration_s)


def test_m31b_two_genuine_bidirectional_crossings_never_pair_first_arbitrarily():
    from examples.m29b_structural_events import make_plan
    old = make_plan("double-crossing")
    result = run_coupled_event_family(CoupledEventPlan(
        _plan(_hypothesis(
            "retard-mass", "extra-retarding",
            ("field:more", "field_v_m", (-250., 0., 0.)),
            ("test-particle", "mass_kg", 1.2e-6)),
            family=old.family), old.questions, old.policy))
    for native in (result.original_event_run, result.parents[0].event_run,
                   result.children[0].event_run):
        assert native.trajectory.termination == "duration"
        roots = [r for r in native.events if r.event_id == "x:two-crossings"]
        assert len(roots) == 2
        assert [r.direction for r in roots] == [1, -1]
    pair = result.as_dict()["coupled_events"][0]["parent_event_time_comparisons"][0]
    assert pair["status"] == "unpaired"
    assert pair["baseline_root_count"] == pair["variation_root_count"] == 2
    assert "estimated_numerical_shift_s" not in pair


def test_m31b_event_study_refused_on_valid_parent_but_coupled_child_observed():
    from examples.m29b_structural_events import make_plan
    old = make_plan("terminal-refusal")
    result = run_coupled_event_family(CoupledEventPlan(
        _plan(_hypothesis(
            "correct-stop", "reversed-force",
            ("field:Eplus", "field_v_m", (500., 0., 0.)),
            ("test-particle", "mass_kg", 2e-6)),
            family=old.family), old.questions, old.policy))
    parent = result.parents[0]
    child = result.children[0]
    assert parent.parent_scenario is not None
    assert parent.status == "rejected-event-study"
    assert parent.event_run is None
    assert "initial release" in parent.reason
    assert child.physical_entry.status == "complete"
    assert child.event_run is not None
    assert child.status == "complete"
    assert child.event_run.trajectory.termination == "duration"
    evidence = result.as_dict()
    assert evidence["parent_events"][0]["accepted_parent_inventory"] is not None
    assert evidence["coupled_events"][0]["parent_event_time_comparisons"][0]["status"] == "unpaired"


def test_m31b_rechecks_original_m31a_mutable_rows_and_native_boolean_time_before_export():
    result = _run(_hypothesis(
        "added-mass", "add-E",
        ("field:Enew", "field_v_m", (150., 0., 0.)),
        ("test-particle", "mass_kg", 2e-5)))
    child = result.children[0]
    event = child.event_run
    forged_time = replace(event.trajectory.samples[0], time_s=False)
    forged_motion = replace(event.trajectory, samples=(
        forged_time, *event.trajectory.samples[1:]))
    forged = replace(child, event_run=replace(event, trajectory=forged_motion))
    with pytest.raises(ValueError, match="invalid saved physical time"):
        CoupledEventRun(result.plan, result.physical_family, result.original_event_run,
                        result.parents, (forged,))
    assert result.as_dict()["coupled_events"][0]["event_run"] is not None
    result.physical_family.entries[0].comparisons[0]["time_s"] = False
    with pytest.raises(ValueError, match="comparison rows"):
        result.as_dict()


def test_m31b_structural_refusal_does_not_invent_parent_or_child_events():
    family = model("mirror-pair")
    result = run_coupled_event_family(CoupledEventPlan(
        _plan(_hypothesis(
            "unpaired-case", "unpaired",
            ("point:single", "charge_c", 3e-9),
            ("test-particle", "mass_kg", 2e-5)),
            family=family),
        (PlaneEvent("x:early", (.001,0.,0.), (1.,0.,0.),
                    "Passive early detector", direction=1),),
        EventPolicy(.002, 1e-8, 1e-8, 1e-6)))
    assert result.children[0].physical_entry.status == "skipped-structural-model"
    assert result.children[0].event_run is None
    assert result.parents[0].parent_scenario is None
    assert result.parents[0].event_run is None
    data = result.as_dict()
    assert data["parent_events"][0]["accepted_parent_inventory"] is None
    assert data["coupled_events"][0]["accepted_source_inventory"] is None
    assert data["coupled_events"][0]["parent_event_time_comparisons"][0]["status"] == "unpaired"


def test_m31b_unfamiliar_rod_and_signed_points_coupled_event_and_independent_force():
    from math import pi, hypot
    from conductor_sim import (
        EPSILON_0, FixedPointField, PointCharge, PrescribedField,
        TestChargeMotionSpec, UniformGravity, UniformRodField,
    )
    from conductor_sim.scenarios import PrescribedScenario
    from conductor_sim.structural_scenarios import NamedSource
    from conductor_sim.structural_family import StructuralCase, StructuralFamilyPlan

    original = PrescribedScenario(
        "m31b:rod-original",
        TestChargeMotionSpec(
            PrescribedField((
                UniformRodField(3e-9, (.11,-.1,0.), (.13,.12,0.)),
                FixedPointField(2e-9, (.18,.15,0.)),
                FixedPointField(-1e-9, (-.16,.10,0.)),
            )),
            PointCharge(1e-9, (0.,0.,0.)),
            1e-5, (.05,.02,0.), .7, .005,
            plane_normal=(0.,0.,1.), relative_tolerance=1e-10,
            gravity=UniformGravity((0.,-.1,0.))),
        (("rod:slanted","Tilted uniformly charged finite rod"),
         ("point:plus","Positive held point"),
         ("point:minus","Negative held point")),
        "Teacher-authored unknown charged-rod and signed-point world.")
    family = StructuralFamilyPlan("m31b:rod-family", original, (
        StructuralCase("new-point", "Add a new separate fixed point.",
            add_sources=(NamedSource("point:added", "Extra held point",
                FixedPointField(1.7e-9, (.21,-.09,0.))),)),
    ))
    study = CoupledEventPlan(
        _plan(_hypothesis(
            "rod-mass", "new-point",
            ("point:added", "charge_c", 2e-9),
            ("test-particle", "mass_kg", 9e-6)),
            family=family),
        (PlaneEvent("x:left", (-.006,0.,0.), (1.,0.,0.),
                    "Negative-x passive detector", direction=-1),),
        EventPolicy(.002, 1e-9, 1e-9, 1e-7))
    output = run_coupled_event_family(study)
    parent, child = output.parents[0], output.children[0]
    assert parent.event_run is not None and child.event_run is not None
    assert child.physical_entry.status == "complete"
    assert child.physical_entry.scenario.parent_revision_id == parent.parent_scenario.revision_id
    assert child.physical_entry.scenario.source_annotations[-1][0] == "point:added"
    for native in (parent.event_run, child.event_run):
        assert native.trajectory.termination == "duration"
        roots = [e for e in native.events if e.event_id == "x:left"]
        assert len(roots) == 1
        assert roots[0].direction == -1
    assert output.as_dict()["coupled_events"][0]["parent_event_time_comparisons"][0]["status"] == "one-resolved-pair"
    # Independent SI force at t=0 via midpoint Coulomb quadrature, not
    # the production finite-rod formula or its integrator.
    k = 1/(4*pi*EPSILON_0)
    e = [0., 0., 0.]
    a,b = (.11,-.1,0.), (.13,.12,0.)
    n = 2500
    for j in range(n):
        u = (j+.5)/n
        site = tuple(a[i]+u*(b[i]-a[i]) for i in range(3))
        d = tuple(-x for x in site)
        dist = hypot(*d)
        for i in range(3):
            e[i] += k*3e-9*d[i]/(n*dist**3)
    for q,site in ((2e-9,(.18,.15,0.)),(-1e-9,(-.16,.10,0.)),
                   (2e-9,(.21,-.09,0.))):
        d = tuple(-x for x in site)
        dist = hypot(*d)
        for i in range(3):
            e[i] += k*q*d[i]/dist**3
    force = (1e-9*e[0], 1e-9*e[1]-9e-6*.1, 1e-9*e[2])
    assert child.event_run.trajectory.samples[0].force_n == pytest.approx(force, abs=2e-13)
    assert original.as_dict() == family.baseline.as_dict()


def test_review01_f01_invented_root_timestamp_rejected_at_creation_and_export():
    """The real numerical root's original state cannot be assigned a new SI time."""
    from examples.m31b_rod_reference import make_plan

    authentic = run_coupled_event_family(make_plan())
    child = authentic.children[0]
    root = child.event_run.events[0]
    late = replace(root,
        sample=replace(root.sample, time_s=root.sample.time_s + .03),
        time_bracket_s=tuple(t + .03 for t in root.time_bracket_s))
    doctored = replace(child, event_run=replace(child.event_run, events=(late,)))
    with pytest.raises(ValueError, match="runner-issued observation receipt"):
        CoupledEventRun(authentic.plan, authentic.physical_family,
                        authentic.original_event_run, authentic.parents, (doctored,))
    with pytest.raises(ValueError, match="runner-issued observation receipt"):
        replace(authentic, children=(doctored,))
    assert authentic.as_dict()["coupled_events"][0]["event_run"]["events"][0]["sample"]["time_s"] == (
        root.sample.time_s)

    # Even a frozen dataclass can be mutated through object.__setattr__:
    # pre-export receipt verification must detect that same forgery.
    restored = run_coupled_event_family(make_plan())
    original = restored.children[0]
    changed = replace(original.event_run.events[0],
        sample=replace(original.event_run.events[0].sample,
                       time_s=original.event_run.events[0].sample.time_s + .03),
        time_bracket_s=tuple(t + .03 for t in original.event_run.events[0].time_bracket_s))
    object.__setattr__(restored, "children", (replace(original,
        event_run=replace(original.event_run, events=(changed,))),))
    with pytest.raises(ValueError, match="runner-issued observation receipt"):
        restored.as_dict()


def test_review01_f02_successful_parent_and_child_cannot_claim_fabricated_refusal():
    from examples.m31b_rod_reference import make_plan

    native = run_coupled_event_family(make_plan())
    forged_parent = replace(native.parents[0], event_run=None,
                            status="rejected-event-study", reason="fabricated parent refusal")
    forged_child = replace(native.children[0], event_run=None,
                           status="rejected-event-study", reason="fabricated child refusal")
    with pytest.raises(ValueError, match="runner-issued observation receipt"):
        CoupledEventRun(native.plan, native.physical_family,
                        native.original_event_run, (forged_parent,), native.children)
    with pytest.raises(ValueError, match="runner-issued observation receipt"):
        CoupledEventRun(native.plan, native.physical_family,
                        native.original_event_run, native.parents, (forged_child,))
    with pytest.raises(ValueError, match="runner-issued observation receipt"):
        replace(native, parents=(forged_parent,))
    with pytest.raises(ValueError, match="runner-issued observation receipt"):
        replace(native, children=(forged_child,))
    assert native.as_dict()["coupled_events"][0]["parent_event_time_comparisons"][0]["status"] == (
        "one-resolved-pair")


def test_review01_f03_failed_m31a_physical_child_must_not_be_resolved_as_event(monkeypatch):
    """Test-injected numerical failure; the original physical solver is unchanged."""
    import conductor_sim.coupled_event_family as service
    from conductor_sim.coupled_physical_family import (
        CoupledPhysicalRun, run_coupled_physical_family,
    )
    from examples.m31b_coupled_events import example

    plan = example("added-mass")
    valid = run_coupled_physical_family(plan.physical_plan)
    failure = replace(valid.entries[0], trajectory=None, comparisons=(),
                      status="integration-failed",
                      reason="Test-injected physical integration failure")
    failed = CoupledPhysicalRun(valid.plan, valid.structural_run, (failure,))
    monkeypatch.setattr(service, "run_coupled_physical_family", lambda chosen: failed)
    observed = service.run_coupled_event_family(plan)
    assert observed.children[0].physical_entry.status == "integration-failed"
    assert observed.children[0].status == "integration-failed"
    assert observed.children[0].event_run is None
    record = observed.as_dict()["coupled_events"][0]
    assert record["physical_status"] == record["event_status"] == "integration-failed"
    assert record["event_run"] is None
    assert record["parent_event_time_comparisons"][0]["status"] == "unpaired"

    with pytest.raises(ValueError, match="failed M31a physical integration"):
        replace(observed, children=(replace(observed.children[0],
            event_run=observed.parents[0].event_run, status="complete"),))


def test_review01_novel_y_gravity_field_two_property_case_matches_analytic_crossings():
    """Chat B's previously unexecuted novel case is now a source-pinned native regression."""
    from math import sqrt

    plan = CoupledPhysicalPlan("review:m31b-gravity", model("uniform-edits"), (
        CoupledHypothesis("gravity-field", "add-gravity", "Joint y-field and gravity.", (
            CoupledPropertyEdit("gravity", "acceleration_m_s2", (0.,-3.3,0.)),
            CoupledPropertyEdit("field:Ey", "field_v_m", (0.,1750.,0.)),
        )),
    ))
    asked = CoupledEventPlan(
        plan, (PlaneEvent("y-neg", (0.,-.003,0.), (0.,1.,0.),
                          "Signed downward detector", direction=-1),),
        EventPolicy(.001, 1e-8, 1e-9, 1e-7))
    actual = run_coupled_event_family(asked)
    parent, child = actual.parents[0], actual.children[0]
    assert parent.status == child.status == "complete"
    assert child.event_run.study.scenario is child.physical_entry.scenario
    assert child.physical_entry.scenario.parent_revision_id == parent.parent_scenario.revision_id
    for event, reference in ((parent.event_run, sqrt(.006/1.995)),
                             (child.event_run, sqrt(.006/3.125))):
        roots = [e for e in event.events if e.event_id == "y-neg"]
        assert len(roots) == 1
        assert roots[0].direction == -1
        assert roots[0].sample.time_s == pytest.approx(reference,abs=1e-7)
    assert actual.original_event_run.as_dict()["outcomes"]["y-neg"] == (
        "not-found-at-requested-resolution")
    assert actual.as_dict()["coupled_events"][0]["parent_event_time_comparisons"][0]["status"] == (
        "one-resolved-pair")
