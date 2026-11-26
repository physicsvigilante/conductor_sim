"""M22a prospective composability acceptance; no per-card production physics."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, replace
import json
from math import hypot
from pathlib import Path

import pytest

from conductor_sim import (
    FixedPointField, PointCharge, PrescribedField, UniformElectricField,
    UniformRingAxisField, UniformGravity, SolidSphere, FixedNetCharge,
    TestChargeMotionSpec as MotionSpec, integrate_test_charge,
)
from conductor_sim.scenarios import PrescribedScenario, revise_scenario, sample_scenario_viewer
from conductor_sim.structural_scenarios import (
    NamedSource, SourceReplacement, ScenarioReassembly, reassemble_scenario,
)
from conductor_sim.scenario_comparison import prepare_scenario_comparison
from conductor_sim.solver import UnsupportedProblem
from examples.structural_scenarios import (
    POSITIVE, NEGATIVE, _base, build_card, check_physics, reference,
    grade_trial, negative_card, export_case,
)


@pytest.fixture(scope="module")
def cards():
    return {case: build_card(case) for case in POSITIVE}


@pytest.mark.parametrize("case", POSITIVE)
def test_all_preselected_assemblies_independently_solve_with_existing_laws(cards, case):
    assembly = cards[case]
    before = assembly.parent.as_dict()
    result = integrate_test_charge(assembly.scenario.spec)
    observations = check_physics(assembly, result)
    assert result.termination == "duration"
    assert observations["checked_samples"] == 25
    assert observations["max_relative_energy_change"] < 5e-5
    assert assembly.parent.as_dict() == before
    assert assembly.scenario.origin == "teacher-hypothetical"
    assert type(assembly.scenario) is PrescribedScenario
    assert assembly.required_action == "independently-integrate-qualified-new-assembly"


@pytest.mark.parametrize("case", POSITIVE)
def test_changes_derived_from_real_inventories_and_preserve_other_contracts(cards, case):
    assembly = cards[case]
    old = assembly.parent.scenario if type(assembly.parent) is ScenarioReassembly else assembly.parent
    new = assembly.scenario
    for f in fields(old.spec):
        if f.name not in ("field", "gravity"):
            assert getattr(old.spec, f.name) == getattr(new.spec, f.name)
    assert old.spec.field.permittivity_f_m == new.spec.field.permittivity_f_m
    oldmap = {i: (label, s) for (i, label), s in zip(old.source_annotations, old.spec.field.sources)}
    newmap = {i: (label, s) for (i, label), s in zip(new.source_annotations, new.spec.field.sources)}
    for identity in oldmap.keys() & newmap.keys():
        assert oldmap[identity][0] == newmap[identity][0]
        assert oldmap[identity][1] is newmap[identity][1]
    record = assembly.as_dict()
    assert record["parent_model"] == old.as_dict()
    assert record["child_model"] == new.as_dict()
    assert record["parent_revision_id"] == old.revision_id
    assert record["child_revision_id"] == new.revision_id
    assert record["changes"]
    assert record["revision_id"] == assembly.revision_id
    # The legacy scenario is a fresh model; the structural record holds lineage.
    assert new.parent_revision_id is None and new.changes == ()
    record["child_model"]["spec"]["mass_kg"] = -100.
    assert new.spec.mass_kg > 0
    changes = assembly.changes
    changes.clear()
    assert assembly.changes


def test_new_relationships_not_just_more_parameter_variants(cards):
    expected = {"P1": 2, "P2": 1, "P3": 2, "U1": 3, "U2": 2,
                "U3": 4, "R1": 1, "R2": 1, "A1": 2}
    assert {k: len(a.scenario.spec.field.sources) for k, a in cards.items()} == expected
    assert cards["P2"].scenario.spec.gravity is not None
    assert cards["U2"].scenario.spec.gravity is not None
    assert cards["R2"].scenario.spec.gravity is None
    assert cards["R1"].changes[0]["operation"] == "replace-source"
    assert cards["R1"].changes[0]["before"]["id"] == "point:A"
    assert cards["R1"].changes[0]["after"]["id"] == "sphere:new"
    assert len(cards["R2"].changes) == 2


def test_retained_structural_chain_digests_and_parent_objects(cards):
    assembly = cards["R2"]
    assert type(assembly.parent) is ScenarioReassembly
    record = assembly.as_dict()
    assert record["parent_assembly"] == assembly.parent.as_dict()
    assert record["parent_revision_id"] == record["parent_assembly"]["child_revision_id"]
    assert assembly.parent.scenario.spec.gravity is not None
    assert assembly.scenario.spec.gravity is None
    with pytest.raises(FrozenInstanceError):
        assembly.scenario = assembly.parent.scenario
    with pytest.raises(FrozenInstanceError):
        assembly.scenario.spec.mass_kg = 123.


def test_retired_id_reuse_refused_in_retained_lineage(cards):
    assembly = cards["R2"]  # point:B was removed, but is retained in structural ancestry.
    with pytest.raises(ValueError, match="reuse"):
        reassemble_scenario(assembly, scenario_id="recycled-source", note="Do not reuse retired identities.",
                            add_sources=(NamedSource("point:B", "Recycled point", FixedPointField(1e-9)),))
    with pytest.raises(ValueError, match="new scenario"):
        reassemble_scenario(assembly, scenario_id=assembly.parent.scenario.scenario_id,
                            note="Do not reuse scenario identity.", gravity=UniformGravity((0., -1., 0.)))


def test_atomic_mirror_addition_qualifies_only_final_inventory(monkeypatch):
    import conductor_sim.prescribed_motion as motion
    base = _base("atomic", p=(0.,0.,0.), v=(.02,0.,.05), normal=(0.,1.,0.))
    sources = (
        NamedSource("point:up", "Upper point", FixedPointField(1e-9, (.2,.12,0.))),
        NamedSource("point:down", "Lower point", FixedPointField(1e-9, (.2,-.12,0.))),
    )
    seen = []
    original = motion._certify
    def observe(spec):
        seen.append(len(spec.field.sources))
        return original(spec)
    monkeypatch.setattr(motion, "_certify", observe)
    reassemble_scenario(base, scenario_id="atomic-both", note="Both mirror members together.",
                        add_sources=sources)
    assert seen == [2]
    with pytest.raises(UnsupportedProblem):
        reassemble_scenario(base, scenario_id="atomic-one", note="Single unsupported member.",
                            add_sources=sources[:1])


@pytest.mark.parametrize("case", NEGATIVE)
def test_preselected_boundary_outcomes_are_honest(case):
    record = negative_card(case)
    assert record["outcome"] == "expected-refusal"
    assert record["parent_unchanged"] is True
    assert record["refusal_stage"] and record["reason"]


@pytest.mark.parametrize("identity", ["test-particle", "medium", "gravity", "", "two words", "x"*65])
def test_invalid_or_reserved_new_source_id_rejected(identity):
    with pytest.raises(ValueError):
        NamedSource(identity, "Some source", FixedPointField(1e-9))


@pytest.mark.parametrize("label", ["", " spaced", "line\nbreak", "x"*49])
def test_source_label_contract_not_relaxed(label):
    with pytest.raises(ValueError):
        NamedSource("valid:id", label, FixedPointField(1e-9))


def test_source_count_overflow_is_existing_numerical_policy():
    parent = _base("capacity")
    additions = tuple(NamedSource("point:"+str(i), "Point "+str(i),
                                  FixedPointField(1e-12, (.2+i*.01, 0., 0.))) for i in range(65))
    before = parent.as_dict()
    with pytest.raises(UnsupportedProblem, match="64"):
        reassemble_scenario(parent, scenario_id="overflow", note="Exceed source cap.", add_sources=additions)
    assert parent.as_dict() == before


def test_custom_or_conducting_source_refused_before_assembly():
    class CustomPoint(FixedPointField):
        pass
    for obj in (CustomPoint(1e-9), SolidSphere(.1, FixedNetCharge(1e-9))):
        with pytest.raises(TypeError, match="built-in fixed source"):
            NamedSource("not:accepted", "Unsupported source", obj)


@pytest.mark.parametrize("edit", [
    {"remove_sources": ("point:A", "point:A")},
    {"remove_sources": ("missing",)},
    {"remove_sources": {"point:A"}},
    {"add_sources": ("not-a-NamedSource",)},
    {"replace_sources": (None,)},
    {"gravity": False},
])
def test_malformed_edits_are_atomic(cards, edit):
    parent = cards["P1"].scenario
    before = parent.as_dict()
    with pytest.raises((ValueError, TypeError)):
        reassemble_scenario(parent, scenario_id="bad-edit", note="Invalid edit.", **edit)
    assert parent.as_dict() == before


def test_duplicate_incoming_id_label_and_overlapping_targets_rejected(cards):
    parent = cards["P1"].scenario
    new = NamedSource("new:A", "Distinct label", FixedPointField(1e-9, (.4,0.,0.)))
    for kwargs in (
        dict(add_sources=(new, new)),
        dict(add_sources=(NamedSource("new:B", "Held point A", FixedPointField(1e-9)),)),
        dict(remove_sources=("point:A",), replace_sources=(SourceReplacement("point:A", new),)),
        dict(add_sources=(new,), replace_sources=(SourceReplacement("point:A", new),)),
    ):
        with pytest.raises(ValueError):
            reassemble_scenario(parent, scenario_id="ambiguous", note="Conflicting edits.", **kwargs)


def test_source_replacement_is_not_an_implicit_property_patch(cards):
    parent = cards["P1"].scenario
    with pytest.raises(ValueError, match="NEW"):
        SourceReplacement("point:A", NamedSource("point:A", "Held point A", FixedPointField(2e-9)))
    replacement = SourceReplacement(
        "point:A", NamedSource("point:replacement", "Replacement point", FixedPointField(2e-9, (.3,.3,0.))))
    assembly = reassemble_scenario(parent, scenario_id="replace-explicit", note="New identity.",
                                   replace_sources=(replacement,))
    assert assembly.scenario.source_annotations[0][0] == "point:replacement"
    assert assembly.scenario.source_annotations[1] == parent.source_annotations[1]
    assert assembly.changes[0]["before"]["source"] == json.loads(json.dumps(parent.spec.field.sources[0].as_dict()))


def test_existing_gravity_properties_use_old_api_and_membership_uses_new(cards):
    old = cards["P2"].scenario
    before = old.as_dict()
    with pytest.raises(ValueError, match="revise_scenario"):
        reassemble_scenario(old, scenario_id="patch-not-structural", note="Change gravity.",
                            gravity=UniformGravity((0., -5., 0.)))
    lower = revise_scenario(old, scenario_id="lower-g", note="Parameter revision.",
                            gravity_changes={"acceleration_m_s2": (0., -5., 0.)})
    assert lower.parent_revision_id == old.revision_id
    removed = reassemble_scenario(old, scenario_id="remove-g", note="Remove gravity.", gravity=None)
    assert removed.scenario.spec.gravity is None
    assert removed.changes[0]["operation"] == "remove-gravity"
    restored = reassemble_scenario(removed, scenario_id="restore-g", note="Explicit new gravity.",
                                  gravity=UniformGravity((0., -10., 0.)))
    assert restored.changes[0]["operation"] == "add-gravity"
    assert old.as_dict() == before


def test_old_parameter_and_comparison_contracts_remain_strict(cards):
    plain = cards["P1"].parent
    with pytest.raises(ValueError, match="unknown"):
        revise_scenario(plain, scenario_id="bad-param-add", note="Not a property.",
                        source_changes={"point:unknown": {"charge_c": 1e-9}})
    with pytest.raises(ValueError, match="authored gravity"):
        revise_scenario(plain, scenario_id="bad-param-g", note="Not a property.",
                        gravity_changes={"acceleration_m_s2": (0., -1., 0.)})
    new = reassemble_scenario(plain, scenario_id="only-gravity-added", note="Add gravity explicitly.",
                             gravity=UniformGravity((0., -1., 0.)))
    left, right = [sample_scenario_viewer(s, visual_frames=5, resolution=9, profile_samples=21, extent_m=.6)
                   for s in (plain, new.scenario)]
    with pytest.raises(ValueError):
        prepare_scenario_comparison(left, right)
    # This is an observation-contract refusal, not an invalid physical model.
    assert left["prescribed_motion"]["termination"] == right["prescribed_motion"]["termination"] == "duration"


def test_structurally_authored_model_accepts_normal_parameter_revisions(cards):
    structural = cards["U2"]
    heavier = revise_scenario(structural.scenario, scenario_id="heavier-composite",
                              note="Double only mass.", motion_changes={"mass_kg": 2e-6})
    assert heavier.parent_revision_id == structural.scenario.revision_id
    assert heavier.source_annotations == structural.scenario.source_annotations
    assert heavier.spec.gravity == structural.scenario.spec.gravity
    assert integrate_test_charge(heavier.spec).termination == "duration"
    before = structural.as_dict()
    sample_scenario_viewer(heavier, visual_frames=5, resolution=9, profile_samples=21, extent_m=.6)
    assert structural.as_dict() == before


@pytest.mark.parametrize("case", ("P1", "P2"))
def test_noop_structural_request_rejected(cards, case):
    old = cards[case].scenario
    with pytest.raises(ValueError, match="no actual structural"):
        reassemble_scenario(old, scenario_id="no-op", note="Nothing changed.")


def test_result_constructor_rejects_fabricated_lineage_and_nonstructural_edits(cards):
    assembly = cards["P1"]
    parent = assembly.parent
    badspec = replace(assembly.scenario.spec, mass_kg=4e-6)
    badchild = PrescribedScenario("not-structural", badspec, assembly.scenario.source_annotations,
                                  "Secret mass change.")
    with pytest.raises(ValueError, match="particle, analysis"):
        ScenarioReassembly(parent, badchild)
    with pytest.raises(ValueError, match="ledger disagrees"):
        ScenarioReassembly(parent, assembly.scenario,
            (SourceReplacement("missing", NamedSource("point:B", "Held point B",
                               assembly.scenario.spec.field.sources[-1])),))
    child = replace(assembly.scenario, origin="source-given")
    with pytest.raises(ValueError, match="teacher-hypothetical"):
        ScenarioReassembly(parent, child)


def test_preserved_tilted_sources_are_not_reconstructed():
    ring = UniformRingAxisField(1e-9, .1, axis=(1., 2., 3.))
    parent = PrescribedScenario(
        "tilted-root", MotionSpec(PrescribedField((ring,)), PointCharge(1e-9), 1e-6,
            (0.,0.,0.), .01, .005, length_scale_m=.1), (("ring:tilted", "Tilted ring"),), "Tilted ring.")
    new = reassemble_scenario(parent, scenario_id="tilted-add", note="Add compatible zero E.",
                             add_sources=(NamedSource("field:zero", "Zero E", UniformElectricField((0.,0.,0.))),))
    assert new.scenario.spec.field.sources[0] is ring
    assert new.scenario.spec.field.sources[0].axis == parent.spec.field.sources[0].axis


def test_legacy_scenario_records_and_digests_match_preimplementation_fixture():
    expected = json.loads((Path(__file__).parent/"fixtures/m22a_legacy_scenarios.json").read_text(encoding="utf-8"))
    actual = []
    for with_g in (False, True):
        spec = MotionSpec(
            PrescribedField((UniformElectricField((3.,0.,0.)),)),
            PointCharge(1e-6), 2e-6, (.1,.2,0.), .1, .01,
            plane_normal=(0.,0.,1.), length_scale_m=.2,
            gravity=UniformGravity((0.,-10.,0.)) if with_g else None)
        base = PrescribedScenario("legacy-gravity" if with_g else "legacy-electric", spec,
                                  (("field:applied","Applied E"),), "M22a pre-change characterization.")
        child = revise_scenario(base, scenario_id=base.scenario_id+"-heavy", note="Double only mass.",
                                motion_changes={"mass_kg":4e-6})
        actual.append(dict(baseline=base.as_dict(), revision=child.as_dict()))
    assert actual == expected


def test_nonuniform_unfamiliar_trajectory_against_independent_velocity_verlet(cards):
    spec = cards["U1"].scenario.spec
    p, v = list(spec.particle.position_m), list(spec.initial_velocity_m_s)
    dt = spec.duration_s/2400
    def acceleration(x):
        force = reference(spec, x)[2]
        return [f/spec.mass_kg for f in force]
    a = acceleration(p)
    for _ in range(2400):
        p = [x+w*dt+.5*b*dt*dt for x,w,b in zip(p,v,a)]
        next_a = acceleration(p)
        v = [w+.5*(b+c)*dt for w,b,c in zip(v,a,next_a)]
        a = next_a
    result = integrate_test_charge(spec)
    assert hypot(*(x-y for x,y in zip(p,result.samples[-1].position_m))) < 1e-8
    assert hypot(*(x-y for x,y in zip(v,result.samples[-1].velocity_m_s))) < 1e-8


def test_every_card_and_layer_is_recorded_with_distinct_denominators():
    grade = grade_trial()
    assert grade["accepted"] is True, grade
    assert [c["card"] for c in grade["case_results"]] == list(POSITIVE + NEGATIVE)
    assert grade["counts"] == {"verified_authored":9, "expected_refusals":7}
    assert grade["denominators"]["unfamiliar_positive"] == 3
    for entry in grade["case_results"][:9]:
        assert entry["authoring"] == entry["qualification"] == "pass"
        assert entry["observation_data"] == "pass"
        assert entry["numerical_completion"] == "duration"
        assert entry["textbook_answer_completeness"]["status"] == "not_attempted"
        assert entry["production_case_branch"] is False


@pytest.mark.parametrize("case", ("U1", "U2", "U3", "A1"))
def test_offline_export_keeps_structural_sidecar_and_no_invented_comparison(tmp_path, case):
    path = tmp_path/(case+".html")
    audit = export_case(case, path)
    assert path.is_file()
    assert path.with_suffix(".json").is_file()
    html = path.read_text(encoding="utf-8")
    assert "Structural reassembly" in html
    assert "teacher hypothetical" in html
    assert "comparison-data" not in html
    assert audit["assembly"]["child_revision_id"]
    assert audit["numerical_result"]["termination"] == "duration"


def test_admitted_structural_model_can_end_at_a_guard_without_completing():
    spec = MotionSpec(PrescribedField(()), PointCharge(1e-9), 1.,
                      (1.,0.,0.), .12, .005, plane_normal=(0.,0.,1.),
                      length_scale_m=.1, source_guard_m=.01)
    root = PrescribedScenario("guard-root", spec, (), "Explicit approach boundary.")
    changed = reassemble_scenario(root, scenario_id="guard-child", note="Add attracting source.",
        add_sources=(NamedSource("point:target", "Attracting point", FixedPointField(-1e-9,(.06,0.,0.))),))
    result = integrate_test_charge(changed.scenario.spec)
    assert result.termination == "source-guard"
    assert result.samples[-1].time_s < spec.duration_s
    assert changed.as_dict()["qualification"].endswith("not a numerical run")


def test_structural_child_does_not_inherit_source_given_provenance(cards):
    root = replace(cards["P1"].parent, origin="source-given")
    altered = reassemble_scenario(root, scenario_id="new-hypothetical", note="New physical source.",
        add_sources=(NamedSource("point:extra", "Extra point", FixedPointField(1e-9,(.4,0.,0.))),))
    assert altered.parent.origin == "source-given"
    assert altered.scenario.origin == "teacher-hypothetical"
    assert altered.as_dict()["origin"] == "teacher-hypothetical"
