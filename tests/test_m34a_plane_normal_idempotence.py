"""M34a: true oblique physical-plane reassembly should preserve exact native normal bits.

Independent constant-acceleration physics is computed directly from teacher
SI field/gravity components, not by importing the production force callback.
"""
from dataclasses import replace
from math import hypot, isclose, isfinite, sqrt, ulp

import pytest

from conductor_sim import (
    CoupledPropertyEdit, PhysicalWorldNode, PointCharge, PrescribedField,
    StructuralWorldNode, TestChargeMotionSpec, UniformElectricField,
    UniformGravity, UnifiedObservationPlan, run_unified_observation_graph,
)
from conductor_sim.motion_events import EventPolicy, PlaneEvent
from conductor_sim.scenarios import PrescribedScenario
from conductor_sim.structural_family import StructuralCase
from conductor_sim.structural_scenarios import (
    NamedSource, SourceReplacement, reassemble_scenario,
)
from conductor_sim.solver import UnsupportedProblem


ROOT_NORMAL = (1., 2., -1.)


def root():
    return PrescribedScenario(
        "m34a:source-root",
        TestChargeMotionSpec(
            PrescribedField((UniformElectricField((320., -80., 160.)),)),
            PointCharge(2e-9, (.003, .012, .06)), 1.6e-6,
            (.25, -.125, 0.), .6, .01,
            plane_normal=ROOT_NORMAL,
            gravity=UniformGravity((.16, -.04, .08))),
        (("field:base", "Original certified held oblique E"),),
        "Teacher-owned three-component planar constant acceleration example.")


def detector():
    return (PlaneEvent("x:after", (.075, 0., 0.), (1., 0., 0.),
                       "Passive signed detector on the world x plane",
                       direction=1),)


def policy():
    return EventPolicy(.006, 1e-8, 1e-8, 1e-6)


def graph():
    original = root()
    added = StructuralCase(
        "add-a", "Introduce an in-plane fixed held uniform field A.",
        add_sources=(NamedSource("field:A", "First new oblique field",
                    UniformElectricField((90., 35., 160.))),))
    replaced = StructuralCase(
        "replace-a", "Retire added source A and introduce genuinely new B.",
        replace_sources=(SourceReplacement(
            "field:A", NamedSource(
                "field:B", "Genuinely new held oblique B",
                UniformElectricField((130., -10., 110.)))),))
    removed = StructuralCase(
        "remove-b", "Remove newly introduced B without moving sources.",
        remove_sources=("field:B",))
    illegal = StructuralCase(
        "resurrect-a", "Attempt to resurrect a historically retired A ID.",
        add_sources=(NamedSource(
            "field:A", "Forbidden reused field",
            UniformElectricField((20., 10., 40.))),))
    sibling = StructuralCase(
        "sibling", "Alternative branch uses the same name A legitimately.",
        add_sources=(NamedSource(
            "field:A", "Independent sibling A",
            UniformElectricField((10., 15., 40.))),))
    nodes = (
        StructuralWorldNode("add-a", "baseline", added, "events"),
        PhysicalWorldNode("mass", "add-a",
            "Change physical probe mass without changing the plane normal",
            (CoupledPropertyEdit("test-particle", "mass_kg", 2.2e-6),), "events"),
        StructuralWorldNode("replace-a", "mass", replaced, "events"),
        StructuralWorldNode("remove-b", "replace-a", removed, "events"),
        StructuralWorldNode("resurrect-a", "replace-a", illegal, "events"),
        StructuralWorldNode("sibling", "baseline", sibling, "events"),
    )
    return UnifiedObservationPlan(
        "m34a:normal-chain", original, nodes, "events",
        detector(), policy(), retain_source_ancestry=True)


def direct_qe_m_g(spec):
    e = tuple(sum(source.field_v_m[k] for source in spec.field.sources)
              for k in range(3))
    q, mass = spec.particle.charge_c, spec.mass_kg
    g = spec.gravity.acceleration_m_s2
    acceleration = tuple(q * e[k]/mass + g[k] for k in range(3))
    force = tuple(q * e[k] + mass*g[k] for k in range(3))
    return e, acceleration, force


def exact_state(spec, time_s):
    _, acceleration, force = direct_qe_m_g(spec)
    position = tuple(
        spec.particle.position_m[k] + spec.initial_velocity_m_s[k]*time_s +
        .5 * acceleration[k]*time_s*time_s for k in range(3))
    velocity = tuple(spec.initial_velocity_m_s[k] + acceleration[k]*time_s
                     for k in range(3))
    return position, velocity, force


def positive_x_root(spec):
    _, accel, _ = direct_qe_m_g(spec)
    dx = .075 - spec.particle.position_m[0]
    v = spec.initial_velocity_m_s[0]
    return 2*dx/(v+sqrt(v*v+2*accel[0]*dx))


def test_01_original_non_dyadic_normal_reassembles_and_preserves_exact_bits():
    old = root()
    assert old.spec.plane_normal != ROOT_NORMAL
    assert hypot(*old.spec.plane_normal) == 1.0000000000000002
    original = old.as_dict()
    case = StructuralCase(
        "once", "One truly independent held source addition.",
        add_sources=(NamedSource(
            "field:fresh", "New planar fixed field",
            UniformElectricField((90., 35., 160.))),))
    result = reassemble_scenario(
        old, scenario_id="m34a:direct-reassembly",
        note=case.note, add_sources=case.add_sources)
    assert result.parent is old
    assert result.scenario.spec.plane_normal == old.spec.plane_normal
    assert tuple(a.hex() for a in result.scenario.spec.plane_normal) == tuple(
        a.hex() for a in old.spec.plane_normal)
    assert result.as_dict()["parent_revision_id"] == old.revision_id
    assert result.scenario.spec.field.sources[-1].field_v_m == (90.,35.,160.)
    assert original == old.as_dict()


def test_02_true_multi_generation_mixed_structural_physical_graph_and_root_events():
    plan = graph()
    frozen = plan.original.as_dict()
    result = run_unified_observation_graph(plan)
    assert result.baseline_status == "complete"
    assert [e.status for e in result.entries] == [
        "complete", "complete", "complete", "complete",
        "rejected-structural-model", "complete"]
    assert result.entries[2].assembly.parent is result.entries[1].scenario
    assert result.entries[1].scenario.parent_revision_id == result.entries[0].scenario.revision_id
    assert result.entries[2].source_ancestry.parent_revision_id == result.entries[1].scenario.revision_id
    assert result.entries[2].source_ancestry.retired_source_ids == ("field:A",)
    assert result.entries[3].source_ancestry.retired_source_ids == ("field:A", "field:B")
    assert result.entries[4].scenario is None
    assert result.entries[4].native is None
    assert result.entries[4].source_ancestry is None
    assert result.entries[5].source_ancestry.world_path == ("baseline", "sibling")
    assert result.entries[5].source_ancestry.introduced_source_ids == ("field:A",)
    for i,(scenario,native) in enumerate([
        (plan.original,result.baseline_native),
        *((entry.scenario,entry.native) for entry in result.entries
          if entry.scenario is not None)]):
        assert scenario.spec.plane_normal == plan.original.spec.plane_normal, i
        assert native.trajectory.termination == "duration"
        assert native.events[0].direction == 1
        assert native.events[0].sample.time_s == pytest.approx(
            positive_x_root(scenario.spec), abs=3e-7)
        for sample in native.trajectory.samples:
            position,velocity,force = exact_state(scenario.spec,sample.time_s)
            assert sample.position_m == pytest.approx(position,abs=4e-8)
            assert sample.velocity_m_s == pytest.approx(velocity,abs=4e-8)
            assert sample.force_n == pytest.approx(force,abs=2e-12)
            assert abs(position[0])+abs(position[1])+abs(position[2])>0
    assert result.as_dict()["schema"]=="conductor_sim.unified_observation_graph.v2"
    assert plan.original.as_dict() == frozen


def test_03_sizing_one_ulp_window_is_bounded_and_idempotent():
    base = root()
    spec = base.spec
    for value in (1., 1. + ulp(1.), 1. - ulp(1.)):
        n = (value, 0., 0.)
        # This auxiliary input uses a zero electric field and y-only launch,
        # so a plane normal along world x is physically compatible.
        trial = replace(spec,
            field=PrescribedField((UniformElectricField((0.,0.,0.)),)),
            gravity=None, initial_velocity_m_s=(0.,.1,0.),
            plane_normal=n)
        assert trial.plane_normal == n
        assert replace(trial, duration_s=.5).plane_normal == trial.plane_normal
    normal=(1.+2*ulp(1.),0.,0.)
    large = replace(spec,
        field=PrescribedField((UniformElectricField((0.,0.,0.)),)),
        gravity=None,initial_velocity_m_s=(0.,.1,0.),
        plane_normal=normal)
    assert large.plane_normal==(1.,0.,0.)
    for original in ((1.,1.,1.), (1.,2.,-1.), (-1.,-2.,1.), (3.,5.,7.)):
        first = replace(spec,
            field=PrescribedField((UniformElectricField((0.,0.,0.)),)),
            gravity=None, initial_velocity_m_s=(0.,0.,0.),
            plane_normal=original)
        assert replace(first, duration_s=.5).plane_normal == first.plane_normal


@pytest.mark.parametrize("wrong", [
    (0.,0.,0.),
    (float("nan"),0.,1.),
    (float("inf"),0.,1.),
    (1.,2.),
    ("1",2.,3.),
])
def test_04_wrong_physical_normals_still_refuse(wrong):
    with pytest.raises((ValueError, TypeError, UnsupportedProblem)):
        replace(root().spec, plane_normal=wrong)


def test_05_real_out_of_plane_conditions_still_refused_not_projected():
    base=root()
    with pytest.raises(UnsupportedProblem):
        replace(base.spec, initial_velocity_m_s=(1.,0.,0.))
    with pytest.raises(UnsupportedProblem):
        replace(base.spec, gravity=UniformGravity((0.,0.,.1)))
    false=StructuralCase(
        "out-of-plane", "A held source with true transverse field is not admissible.",
        add_sources=(NamedSource(
            "field:wrong","False out-of-plane source",
            UniformElectricField((50.,0.,0.))),))
    plan=UnifiedObservationPlan("m34a:reject",base,(
        StructuralWorldNode("out-of-plane","baseline",false,"events"),
    ),"events",detector(),policy(),retain_source_ancestry=True)
    run=run_unified_observation_graph(plan)
    assert run.entries[0].status=="rejected-structural-model"
    assert run.entries[0].scenario is None and run.entries[0].native is None


def test_06_v1_v2_v3_graph_and_inquiry_boundaries_remain_typed():
    from conductor_sim import WorldComparison
    from conductor_sim import GraphEventInquiryPlan, run_graph_event_inquiry
    original=root()
    case=StructuralCase(
        "inplane","In-plane source addition",
        add_sources=(NamedSource(
            "field:add","New qualified source",
            UniformElectricField((90.,35.,160.))),))
    node=StructuralWorldNode("inplane","baseline",case)
    v1=run_unified_observation_graph(UnifiedObservationPlan(
        "m34a:v1",original,(node,)))
    v2=run_unified_observation_graph(UnifiedObservationPlan(
        "m34a:v2",original,(node,),retain_source_ancestry=True))
    v3=run_unified_observation_graph(UnifiedObservationPlan(
        "m34a:v3",original,(node,),retain_source_ancestry=True,
        comparisons=(WorldComparison("pair","baseline","inplane"),)))
    assert [x.as_dict()["schema"] for x in (v1,v2,v3)]==[
        "conductor_sim.unified_observation_graph.v1",
        "conductor_sim.unified_observation_graph.v2",
        "conductor_sim.unified_observation_graph.v3"]
    for r in (v1,v2,v3):
        assert r.entries[0].status=="complete"
        assert r.entries[0].scenario.spec.plane_normal==original.spec.plane_normal
    inquiry=GraphEventInquiryPlan(v3,"m34a:query",("baseline","inplane"),
        detector(),policy(),(WorldComparison("newpair","baseline","inplane"),))
    q=run_graph_event_inquiry(inquiry)
    assert all(entry.status=="complete" for entry in q.entries)
    assert q.as_dict()["schema"]=="conductor_sim.graph_event_inquiry.v1"
    assert v3.as_dict()["schema"]=="conductor_sim.unified_observation_graph.v3"
