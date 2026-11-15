"""M26a: driven structural reuse, independent physics and refusal boundaries."""
from dataclasses import FrozenInstanceError, replace
from math import exp, hypot, pi
import json

import pytest

from conductor_sim import (
    EPSILON_0, FixedPointField, PointCharge, PrescribedField,
    TestChargeMotionSpec, UniformElectricField, UniformGravity,
    UniformRingAxisField, UniformRodField, UniformSphereField, UnsupportedProblem,
)
from conductor_sim.driven_motion import (
    DrivenModel, DrivenStudy, EventPolicy, FieldPhase, LinearDrag, PlaneEvent,
    UniformFieldSchedule, investigate_driven_motion, revise_driven_model,
    validate_driven_run,
)
from conductor_sim.driven_reassembly import DrivenReassembly, reassemble_driven_model
from conductor_sim.scenarios import PrescribedScenario
from conductor_sim.structural_scenarios import NamedSource, SourceReplacement


def make_model(*, sources=(), gravity=None, phases=None, position=(0, 0, .1),
               velocity=(.01, 0, 0), duration=.04, drag=1e-8, normal=(0, 1, 0)):
    spec = TestChargeMotionSpec(
        PrescribedField(tuple(sources)), PointCharge(1e-9, position),
        1e-6, velocity, duration, .01, plane_normal=normal,
        relative_tolerance=1e-9, source_guard_m=1e-5, gravity=gravity,
    )
    base = PrescribedScenario(
        "m26a:base", spec,
        tuple((f"fixed:{i}", f"Fixed source {i}") for i in range(len(sources))),
        "Teacher hypothetical baseline; no conductor response.",
    )
    return DrivenModel(
        base, LinearDrag(drag),
        UniformFieldSchedule(phases or (FieldPhase(0, (.1, 0, 0)),)),
        "m26a:parent", "Teacher hypothetical; SI input.",
    )


def run(model, questions=()):
    return investigate_driven_motion(
        DrivenStudy(model, questions, EventPolicy(.005, 1e-9, 1e-9, 1e-8))
    )


def edit(parent, *, model_id="m26a:child", scenario_id="m26a:child-base", **kwargs):
    return reassemble_driven_model(
        parent, model_id=model_id, scenario_id=scenario_id,
        note="Teacher hypothetical: explicit structural change.", **kwargs
    )


def test_p1_point_added_force_and_switched_schedule_unmodified():
    phases = (FieldPhase(0, (.1, 0, 0)), FieldPhase(.02, (-.1, 0, .03)))
    parent = make_model(phases=phases)
    before = parent.as_dict()
    named = NamedSource("point:A", "Held point A", FixedPointField(1e-10, (.2, 0, .1)))
    child = edit(parent, add_sources=(named,))
    assert parent.as_dict() == before
    assert child.model.drag is parent.drag and child.model.applied is parent.applied
    assert child.model.base.spec.field.sources[0] is named.source
    assert child.model.base.source_annotations == (("point:A", "Held point A"),)
    observed = run(child.model)
    assert observed.termination == "duration" and len(observed.switches) == 1
    k = 1 / (4 * pi * EPSILON_0)
    expected = (-k * 1e-9 * 1e-10 / .2**2, 0, 0)
    row = observed.samples[0]
    assert row.fixed_electric_force_n == pytest.approx(expected, rel=2e-14, abs=1e-24)
    assert row.applied_electric_force_n == pytest.approx((1e-10, 0, 0))
    assert observed.switches[0].time_s == .02
    assert observed.switches[0].left.position_m == observed.switches[0].right.position_m
    assert validate_driven_run(observed)
    record = child.as_dict()
    assert record["base_assembly"]["changes"][0]["operation"] == "add-source"
    assert record["parent_model_revision_id"] == parent.revision_id
    assert record["child_model_revision_id"] == child.model.revision_id
    assert record["revision_id"] == child.revision_id
    record["parent_model"]["drag"]["coefficient_kg_s"] = -1
    assert parent.drag.coefficient_kg_s > 0


def test_p2_atomic_mirror_pair_cancelled_normal_force():
    parent = make_model()
    p1 = FixedPointField(1e-10, (.12, .04, .1))
    p2 = FixedPointField(1e-10, (.12, -.04, .1))
    child = edit(parent, add_sources=(
        NamedSource("mirror:up", "Upper held charge", p1),
        NamedSource("mirror:down", "Lower held charge", p2),
    ))
    fields = [source.electric_field_at(parent.spec.particle.position_m)
              for source in (p1, p2)]
    assert fields[0][1] == pytest.approx(-fields[1][1], abs=1e-12)
    assert sum(f[1] for f in fields) == pytest.approx(0, abs=1e-12)
    observed = run(child.model)
    assert observed.termination == "duration"
    assert observed.samples[0].fixed_electric_force_n[1] == pytest.approx(0, abs=1e-23)


def test_p3_gravity_added_exact_linear_drag_exponential():
    parent = make_model(sources=(), drag=1e-8)
    child = edit(parent, gravity=UniformGravity((0, 0, -.2)))
    result = run(child.model)
    assert result.termination == "duration"
    mass = parent.spec.mass_kg
    lam = parent.drag.coefficient_kg_s / mass
    t = result.samples[-1].time_s
    q_over_m = parent.spec.particle.charge_c / parent.spec.mass_kg
    applied = parent.applied.phases[0].field_v_m
    a = tuple(q_over_m * field + gravity for field, gravity in
              zip(applied, (0., 0., -.2)))
    v0 = parent.spec.initial_velocity_m_s
    r0 = parent.spec.particle.position_m
    for i in range(3):
        v_inf = a[i] / lam
        assert result.samples[-1].velocity_m_s[i] == pytest.approx(
            v_inf + (v0[i] - v_inf) * exp(-lam * t), abs=1e-9)
        assert result.samples[-1].position_m[i] == pytest.approx(
            r0[i] + v_inf * t + (v0[i] - v_inf) * (1 - exp(-lam * t)) / lam,
            abs=1e-9)
    assert "add-gravity" in [x["operation"] for x in child.changes]
    assert abs(result.samples[-1].work_balance_error_j) < 1e-13


def test_p4_remove_point_and_gravity_and_retain_driven_law():
    point = FixedPointField(1e-10, (.2, 0, .1))
    parent = make_model(sources=(point,), gravity=UniformGravity((0, 0, -.2)))
    child = edit(parent, remove_sources=("fixed:0",), gravity=None)
    assert child.model.base.spec.gravity is None
    assert child.model.base.spec.field.sources == ()
    assert [x["operation"] for x in child.changes] == ["remove-source", "remove-gravity"]
    observed = run(child.model)
    assert observed.termination == "duration"
    assert observed.samples[0].fixed_electric_force_n == (0, 0, 0)
    assert observed.samples[0].gravitational_force_n == (0, 0, 0)
    assert observed.samples[0].applied_electric_force_n == pytest.approx((1e-10, 0, 0))
    assert observed.samples[-1].dissipated_energy_j >= 0


def test_p5_explicit_point_replacement_gauss_sphere_field():
    old = FixedPointField(1e-10, (.2, 0, .1))
    parent = make_model(sources=(old,))
    sphere = UniformSphereField(1e-10, .2, (.04, 0, .1))
    child = edit(parent, replace_sources=(
        SourceReplacement("fixed:0", NamedSource("sphere:new", "Volume sphere", sphere)),
    ))
    assert child.model.base.source_annotations == (("sphere:new", "Volume sphere"),)
    probe = parent.spec.particle.position_m
    d = tuple(probe[i] - sphere.center_m[i] for i in range(3))
    expected = tuple(1e-9 * 1e-10 / (4*pi*EPSILON_0) * x / sphere.radius_m**3 for x in d)
    observed = run(child.model)
    assert observed.samples[0].fixed_electric_force_n == pytest.approx(expected, rel=1e-13)
    assert child.changes[0]["operation"] == "replace-source"
    assert child.changes[0]["before"]["id"] == "fixed:0"
    assert child.changes[0]["after"]["id"] == "sphere:new"


def test_u1_axial_ring_qualifies_both_phases_and_independent_law():
    phases = (FieldPhase(0, (.06, 0, 0)), FieldPhase(.02, (-.03, 0, 0)))
    parent = make_model(
        position=(.1, 0, 0), velocity=(.01, 0, 0),
        phases=phases, duration=.04,
    )
    ring = UniformRingAxisField(1e-10, .15, center_m=(0, 0, 0), axis=(1, 0, 0))
    child = edit(parent, add_sources=(NamedSource("ring:A", "Axial ring A", ring),))
    z, a = .1, .15
    k = 1 / (4*pi*EPSILON_0)
    assert child.model.base.spec.field.potential_at((z, 0, 0)) == pytest.approx(
        k * ring.charge_c / hypot(a, z), rel=1e-13)
    row = run(child.model)
    assert row.termination == "duration" and len(row.switches) == 1
    expected_e = k * ring.charge_c * z / (a*a + z*z)**1.5
    assert row.samples[0].fixed_electric_force_n[0] == pytest.approx(
        1e-9 * expected_e, rel=1e-13)
    assert row.samples[0].fixed_electric_force_n[1:] == (0, 0)


@pytest.mark.parametrize("added", [
    NamedSource("bad:p", "Unpaired source", FixedPointField(1e-10, (.12, .04, .1))),
    NamedSource("bad:r", "Unpaired rod", UniformRodField(1e-10, (.1, .04, .05), (.15, .04, .05))),
])
def test_n1_unpaired_out_of_plane_refused_atomically(added):
    parent = make_model()
    before = parent.as_dict()
    with pytest.raises(UnsupportedProblem):
        edit(parent, add_sources=(added,))
    assert parent.as_dict() == before


def test_n2_later_schedule_phase_axial_conflict_refused():
    parent = make_model(
        position=(.1, 0, 0), velocity=(.01, 0, 0),
        phases=(FieldPhase(0, (.05, 0, 0)), FieldPhase(.02, (0, 0, .1))),
    )
    ring = NamedSource("ring:later", "Ring rejected later",
                       UniformRingAxisField(1e-10, .15, axis=(1, 0, 0)))
    with pytest.raises(UnsupportedProblem):
        edit(parent, add_sources=(ring,))


@pytest.mark.parametrize("source_id", [
    "field:scheduled", "drag:medium", "test-particle", "gravity", "medium",
])
def test_n3_reserved_or_driven_identity_conflict(source_id):
    parent = make_model()
    if source_id in ("test-particle", "gravity", "medium"):
        with pytest.raises(ValueError):
            NamedSource(source_id, "Collision", FixedPointField(1e-10))
    else:
        with pytest.raises(ValueError, match="identities"):
            edit(parent, add_sources=(
                NamedSource(source_id, "Collision", FixedPointField(1e-10, (.2, 0, .1))),
            ))


def test_n3_retired_source_and_model_id_cannot_be_reused():
    parent = make_model()
    a = edit(parent, add_sources=(
        NamedSource("point:retired", "Temporary source", FixedPointField(1e-10, (.2, 0, .1))),
    ))
    b = edit(a, model_id="m26a:grandchild", scenario_id="m26a:base-grandchild",
             remove_sources=("point:retired",))
    with pytest.raises(ValueError, match="reuse"):
        edit(b, model_id="m26a:greatgrandchild", scenario_id="m26a:base-greatgrandchild",
             add_sources=(
                 NamedSource("point:retired", "Recycled", FixedPointField(2e-10, (.2, 0, .1))),
             ))
    with pytest.raises(ValueError, match="fresh model"):
        edit(b, model_id=parent.model_id, scenario_id="m26a:unused",
             gravity=UniformGravity((0, 0, -.1)))
    assert b.base_assembly.parent is a.base_assembly
    assert b.parent is a


@pytest.mark.parametrize("edit_args", [
    {}, {"remove_sources": ("unknown",)}, {"remove_sources": ("fixed:0", "fixed:0")},
    {"add_sources": ("invalid",)},
])
def test_n4_bad_or_noop_edits_are_atomic(edit_args):
    parent = make_model(sources=(FixedPointField(1e-10, (.2, 0, .1)),))
    old = parent.as_dict()
    with pytest.raises((ValueError, TypeError)):
        edit(parent, **edit_args)
    assert parent.as_dict() == old


def test_n5_forged_record_or_hidden_physics_edit_refused():
    parent = make_model()
    child = edit(parent, add_sources=(
        NamedSource("point:new", "New source", FixedPointField(1e-10, (.2, 0, .1))),
    ))
    with pytest.raises(ValueError, match="exact drag"):
        DrivenReassembly(parent, replace(child.model, drag=LinearDrag(2e-8),
                                          parent_revision_id=parent.revision_id), child.base_assembly)
    with pytest.raises(ValueError, match="parent revision"):
        DrivenReassembly(parent, replace(child.model, parent_revision_id="f"*64), child.base_assembly)
    another = replace(parent, base=replace(parent.base))
    with pytest.raises(ValueError, match="retained parent"):
        DrivenReassembly(another, child.model, child.base_assembly)
    with pytest.raises(FrozenInstanceError):
        child.model = parent
    assert child.model.base.spec.initial_velocity_m_s == parent.spec.initial_velocity_m_s
    assert child.model.base.spec.duration_s == parent.spec.duration_s


def test_n6_old_parameter_only_revision_still_refuses_membership_change():
    parent = make_model()
    child = edit(parent, add_sources=(
        NamedSource("point:new", "New source", FixedPointField(1e-10, (.2, 0, .1))),
    ))
    with pytest.raises(ValueError, match="preserve physical source identities"):
        revise_driven_model(
            parent, model_id="m26a:bad-param", note="Cannot hide addition",
            base=child.model.base,
        )
    assert parent.base.spec.field.sources == ()


def test_n7_source_guard_can_censor_child_trajectory():
    parent = make_model(
        position=(0, 0, .1), velocity=(.2, 0, 0),
        duration=.04, drag=1e-8,
    )
    # A genuine guard is reached if launch is initially close to a point
    # source and the finite uniform input drives the probe toward it.
    child = edit(parent, add_sources=(
        NamedSource("point:guard", "Approach source",
                    FixedPointField(-1e-12, (.002, 0, .1))),
    ))
    observed = run(child.model)
    assert observed.termination == "source-guard"
    assert observed.guard_events
    assert validate_driven_run(observed)


def test_no_implicit_change_to_questions_numerical_policy_or_baseline():
    parent = make_model()
    detector = PlaneEvent("detector", (.15, 0, 0), (1, 0, 0), "Passive detector")
    baseline = run(parent, (detector,))
    child = edit(parent, add_sources=(
        NamedSource("field:new", "Small fixed uniform E", UniformElectricField((.02, 0, 0))),
    ))
    second = run(child.model, (detector,))
    assert baseline.study.questions == second.study.questions
    assert baseline.study.policy == second.study.policy
    assert second.study.model is child.model
    assert baseline.study.model is parent
    assert json.dumps(parent.as_dict(), sort_keys=True) == json.dumps(child.parent.as_dict(), sort_keys=True)
