"""M31a: atomic coupled full-world physical revisions with source provenance.

Test references are direct constant-acceleration and independent Coulomb
midpoint quadrature, never callbacks installed in the production solver.
"""
from dataclasses import replace
import json
from math import hypot, pi

import pytest

from conductor_sim import (
    EPSILON_0, FixedPointField, PointCharge, PrescribedField,
    TestChargeMotionSpec, UniformRodField, UniformGravity,
    UniformElectricField, integrate_test_charge,
)
from conductor_sim.scenarios import PrescribedScenario, revise_scenario
from conductor_sim.solver import UnsupportedProblem
from conductor_sim.structural_family import StructuralCase, StructuralFamilyPlan
from conductor_sim.structural_scenarios import NamedSource
from conductor_sim.trajectory_sweep import _shared_rows
from conductor_sim.coupled_physical_family import (
    CoupledPropertyEdit, CoupledHypothesis, CoupledPhysicalPlan,
    CoupledPhysicalRun, run_coupled_physical_family,
)
from examples.m29a_structural_family import model


def edit(identity, prop, value):
    return CoupledPropertyEdit(identity,prop,value)


def hypothesis(case, world, *edits):
    return CoupledHypothesis(case,world,"Teacher-authored atomic physical changes.",edits)


def make(*hypotheses, family=None):
    return CoupledPhysicalPlan(
        "m31a:coupled",
        model("uniform-edits") if family is None else family,
        tuple(hypotheses))


def test_p1_atomic_added_source_and_probe_mass_with_separate_analytic_acceleration():
    plan=make(hypothesis(
        "added-mass","add-E",
        edit("field:Enew","field_v_m",(150.,0.,0.)),
        edit("test-particle","mass_kg",2e-5)))
    original=plan.family.baseline.as_dict()
    result=run_coupled_physical_family(plan)
    entry=result.entries[0]
    assert entry.status=="complete"
    assert entry.structural_entry.case.case_id=="add-E"
    assert entry.structural_entry.assembly.parent is plan.family.baseline
    assert entry.structural_entry.assembly.scenario.source_annotations[-1][0]=="field:Enew"
    assert entry.parent_scenario is entry.structural_entry.assembly.scenario
    assert entry.scenario.parent_revision_id==entry.parent_scenario.revision_id
    assert entry.scenario.source_annotations==entry.parent_scenario.source_annotations
    assert entry.parent_scenario.spec.field.sources[-1].field_v_m==(100.,0.,0.)
    assert entry.scenario.spec.field.sources[-1].field_v_m==(150.,0.,0.)
    assert entry.scenario.spec.mass_kg==2e-5
    assert {(x.component_id,x.property) for x in entry.scenario.changes}=={
        ("field:Enew","field_v_m"),("test-particle","mass_kg")}
    ax=1e-9*(100+150)/2e-5
    ay=1e-9*50/2e-5
    for sample in entry.trajectory.samples:
        t=sample.time_s
        assert sample.position_m==pytest.approx(
            (.1*t+.5*ax*t*t,.5*ay*t*t,0.),abs=2e-10)
        assert sample.velocity_m_s==pytest.approx(
            (.1+ax*t,ay*t,0.),abs=5e-10)
        assert sample.acceleration_m_s2==pytest.approx((ax,ay,0.),abs=5e-10)
    assert len(entry.comparisons)==11
    evidence=result.as_dict()
    assert evidence["schema"]=="conductor_sim.coupled_physical_family.v1"
    assert evidence["entries"][0]["structural_assembly"]["parent_revision_id"]==original["revision_id"]
    assert evidence["entries"][0]["revised_scenario"]["parent_revision_id"]==entry.parent_scenario.revision_id
    assert evidence["time_interpolation"] is False and evidence["postguard_extrapolation"] is False
    evidence["entries"][0]["physical_trajectory"]["samples"][0]["position_m"][0]=999
    assert entry.trajectory.samples[0].position_m[0]!=999
    assert plan.family.baseline.as_dict()==original


def test_p2_joint_translation_admitted_while_either_individual_physical_edit_refuses():
    family=model("point-replace")
    original=family.baseline
    initial=original.spec.particle.position_m
    source=original.spec.field.sources[0].position_m
    proposed_particle=(initial[0],initial[1],.05)
    proposed_source=(source[0],source[1],.05)
    with pytest.raises(UnsupportedProblem,match="fixed point-source inventory is not reflection symmetric"):
        revise_scenario(
            original,scenario_id="m31a:bad-single-particle",
            note="Isolated z translation does not preserve plane",
            particle_changes={"position_m":proposed_particle})
    with pytest.raises(UnsupportedProblem,match="fixed point-source inventory is not reflection symmetric"):
        revise_scenario(
            original,scenario_id="m31a:bad-single-source",
            note="Isolated z translation does not preserve plane",
            source_changes={"point:A":{"position_m":proposed_source}})
    plan=make(hypothesis(
        "whole-translation","baseline",
        edit("test-particle","position_m",proposed_particle),
        edit("point:A","position_m",proposed_source)),family=family)
    result=run_coupled_physical_family(plan)
    entry=result.entries[0]
    assert entry.status=="complete"
    assert entry.parent_scenario is original and entry.structural_entry is None
    assert entry.scenario.parent_revision_id==original.revision_id
    assert entry.scenario.spec.particle.position_m==proposed_particle
    assert entry.scenario.spec.field.sources[0].position_m==proposed_source
    for base,shifted in zip(result.structural_run.baseline_trajectory.samples,
                            entry.trajectory.samples):
        assert shifted.time_s==base.time_s
        assert shifted.position_m==pytest.approx(
            (base.position_m[0],base.position_m[1],base.position_m[2]+.05),
            abs=5e-11)
        assert shifted.velocity_m_s==pytest.approx(base.velocity_m_s,abs=5e-11)
        assert shifted.force_n==pytest.approx(base.force_n,abs=1e-18)
    assert original.spec.particle.position_m==initial
    assert original.spec.field.sources[0].position_m==source


def test_p3_replaced_source_keeps_new_id_and_no_resurrection_under_coupling():
    result=run_coupled_physical_family(make(hypothesis(
        "replace-mass","replace-Ey",
        edit("field:Ey-new","field_v_m",(0.,130.,0.)),
        edit("test-particle","mass_kg",2e-5))))
    case=result.entries[0]
    assert case.status=="complete"
    assert case.structural_entry.assembly.changes[0]["operation"]=="replace-source"
    expected=("field:E0","field:Ey-new")
    assert tuple(k for k,_ in case.parent_scenario.source_annotations)==expected
    assert tuple(k for k,_ in case.scenario.source_annotations)==expected
    assert "field:Ey" not in dict(case.scenario.source_annotations)
    assert "field:Ey" in dict(result.plan.family.baseline.source_annotations)
    assert case.scenario.parent_revision_id==case.parent_scenario.revision_id
    assert len(case.scenario.changes)==2
    assert case.trajectory.termination=="duration"


def test_p4_gravity_new_member_and_existing_electric_field_coupled_analytically():
    run=run_coupled_physical_family(make(hypothesis(
        "gravity-field","add-gravity",
        edit("gravity","acceleration_m_s2",(0.,-3.,0.)),
        edit("field:E0","field_v_m",(180.,0.,0.)))))
    case=run.entries[0]
    assert case.status=="complete"
    assert case.structural_entry.assembly.changes[0]["operation"]=="add-gravity"
    assert case.parent_scenario.spec.gravity.acceleration_m_s2==(0.,-2.,0.)
    assert case.scenario.spec.gravity.acceleration_m_s2==(0.,-3.,0.)
    assert case.scenario.spec.field.sources[0].field_v_m==(180.,0.,0.)
    ax=1e-9*180/1e-5
    ay=1e-9*50/1e-5-3
    for s in case.trajectory.samples:
        t=s.time_s
        assert s.position_m==pytest.approx(
            (.1*t+.5*ax*t*t,.5*ay*t*t,0.),abs=5e-10)
        assert s.acceleration_m_s2==pytest.approx((ax,ay,0.),abs=1e-9)
        assert s.gravitational_force_n==pytest.approx((0.,-3e-5,0.),abs=1e-19)
        assert s.electric_force_n==pytest.approx((180e-9,50e-9,0.),abs=1e-19)


def test_p5_refusal_stages_real_parent_inventory_and_independent_sibling():
    family=model("mirror-pair")
    plan=make(
        hypothesis("physical-refused","unpaired",
                   edit("point:single","charge_c",(3e-9)),
                   edit("test-particle","mass_kg",2e-5)),
        hypothesis("missing-target","extra-field",
                   edit("point:single","charge_c",2e-9),
                   edit("test-particle","mass_kg",2e-5)),
        hypothesis("bad-combined-plane","extra-field",
                   edit("field:up","field_v_m",(0.,0.,35.)),
                   edit("test-particle","mass_kg",2e-5)),
        hypothesis("valid-continue","extra-field",
                   edit("field:up","field_v_m",(0.,50.,0.)),
                   edit("test-particle","mass_kg",2e-5)),
        family=family)
    result=run_coupled_physical_family(plan)
    assert [row.status for row in result.entries]==[
        "skipped-structural-model","rejected-coupled-edit",
        "rejected-physical-model","complete"]
    assert result.entries[0].parent_scenario is None
    assert result.entries[0].scenario is None
    assert result.entries[1].parent_scenario is not None
    assert result.entries[1].scenario is None
    assert result.entries[2].parent_scenario is not None
    assert result.entries[2].scenario is None
    assert result.entries[3].trajectory.termination=="duration"
    audit=result.as_dict()["entries"]
    assert audit[0]["accepted_parent_inventory"] is None
    assert audit[1]["accepted_parent_inventory"] is not None
    assert audit[1]["structural_assembly"] is not None
    assert audit[2]["accepted_parent_inventory"] is not None
    assert audit[2]["revised_scenario"] is None
    assert audit[3]["revised_scenario"]["parent_revision_id"]==(
        result.entries[3].parent_scenario.revision_id)


def test_p5_unmodified_edit_and_gauge_rejected_at_physical_target_stage():
    plan=make(
        hypothesis("noop","add-E",
                   edit("field:Enew","field_v_m",(100.,0.,0.)),
                   edit("test-particle","mass_kg",2e-5)),
        hypothesis("gauge","add-E",
                   edit("field:Enew","reference_potential_v",1.),
                   edit("test-particle","mass_kg",2e-5)))
    result=run_coupled_physical_family(plan)
    assert [e.status for e in result.entries]==[
        "rejected-coupled-edit","rejected-coupled-edit"]
    assert "unchanged value" in result.entries[0].reason
    assert "unsupported physical sweep property" in result.entries[1].reason
    assert all(e.parent_scenario is not None and e.trajectory is None
               for e in result.entries)


def test_p6_source_guard_and_actual_timestamps_only():
    plan=make(hypothesis(
        "guard-charge-mass","attract-guard",
        edit("point:attract","charge_c",-2e-10),
        edit("test-particle","mass_kg",2e-4)),
        family=model("guard-and-refusal"))
    run=run_coupled_physical_family(plan)
    case=run.entries[0]
    assert case.status in ("complete","censored")
    assert case.structural_entry.trajectory.termination=="source-guard"
    assert case.scenario.parent_revision_id==case.parent_scenario.revision_id
    original_times={s.time_s for s in case.structural_entry.trajectory.samples}
    child_times={s.time_s for s in case.trajectory.samples}
    assert all(row["time_s"] in original_times and row["time_s"] in child_times
               for row in case.comparisons)
    if case.status=="censored":
        assert case.trajectory.samples[-1].time_s<case.scenario.spec.duration_s
    assert run.as_dict()["postguard_extrapolation"] is False


def test_p7_unfamiliar_rod_new_point_plus_mass_and_direct_coulomb_force_reference():
    original=PrescribedScenario(
        "m31a:unfamiliar-original",
        TestChargeMotionSpec(
            PrescribedField((
                UniformRodField(2.8e-9,(.095,-.08,0.),(.14,.11,0.)),
                FixedPointField(1.3e-9,(.19,.14,0.)),
                FixedPointField(-.8e-9,(-.17,.09,0.)),
            )),
            PointCharge(.8e-9,(0.,0.,0.)),9e-6,(.045,.016,0.),
            .42,.0042,plane_normal=(0.,0.,1.),
            relative_tolerance=2e-9,source_guard_m=.001,
            gravity=UniformGravity((0.,-.08,0.))),
        (("rod:slant","Tilted charged rod"),
         ("point:upper","Held positive point"),
         ("point:lower","Held negative point")),
        "Teacher hypothetical nonresponding held sources.")
    family=StructuralFamilyPlan("m31a:transfer-world",original,(
        StructuralCase("new-point","Introduce distinct held point.",
            add_sources=(NamedSource(
                "point:added","New adjustable held point",
                FixedPointField(1.5e-9,(.18,-.085,0.))),)),))
    plan=make(hypothesis(
        "couple-new-source","new-point",
        edit("point:added","charge_c",2.4e-9),
        edit("test-particle","mass_kg",1.5e-5)),family=family)
    result=run_coupled_physical_family(plan)
    case=result.entries[0]
    assert case.status=="complete"
    assert case.scenario.parent_revision_id==case.parent_scenario.revision_id
    assert case.scenario.spec.field.sources[-1].charge_c==2.4e-9
    assert case.scenario.spec.mass_kg==1.5e-5
    assert case.scenario.source_annotations[-1][0]=="point:added"
    # Independent Coulomb quadrature at original release, not production
    # UniformRodField._values or the M19/M27 numerical integrator.
    n=2500
    a=(.095,-.08,0.);b=(.14,.11,0.)
    const=1/(4*pi*EPSILON_0)
    e=[0.,0.,0.]
    for j in range(n):
        u=(j+.5)/n
        site=tuple(a[i]+u*(b[i]-a[i]) for i in range(3))
        d=tuple(-x for x in site)
        dist=hypot(*d)
        for k in range(3): e[k]+=const*2.8e-9*d[k]/(n*dist**3)
    for q,site in ((1.3e-9,(.19,.14,0.)),
                   (-.8e-9,(-.17,.09,0.)),
                   (2.4e-9,(.18,-.085,0.))):
        d=tuple(-x for x in site);dist=hypot(*d)
        for k in range(3):e[k]+=const*q*d[k]/dist**3
    force=(.8e-9*e[0],.8e-9*e[1]-1.5e-5*.08,.8e-9*e[2])
    assert case.trajectory.samples[0].force_n==pytest.approx(force,abs=3e-12)
    assert original.as_dict()==result.plan.family.baseline.as_dict()


def test_n1_invalid_typed_hypotheses_and_wrong_source_world_refused():
    native=model("uniform-edits")
    mass=edit("test-particle","mass_kg",2e-5)
    field=edit("field:E0","field_v_m",(200.,0.,0.))
    with pytest.raises(ValueError,match="2..4"):
        hypothesis("single","baseline",mass)
    with pytest.raises(ValueError,match="duplicate"):
        hypothesis("duplicate","baseline",mass,mass)
    with pytest.raises(TypeError):
        CoupledPropertyEdit("test-particle","mass_kg",True)
    with pytest.raises(TypeError):
        CoupledPropertyEdit("field:E0","field_v_m",(True,0.,0.))
    with pytest.raises(ValueError,match="unknown"):
        make(hypothesis("bad","phantom",mass,field))
    with pytest.raises(ValueError,match="unique"):
        make(hypothesis("x","baseline",mass,field),
             hypothesis("x","add-E",mass,field))
    with pytest.raises(ValueError,match="16"):
        make(*[hypothesis(
            "v"+str(i),"baseline",
            mass,field,
            edit("field:Ey","field_v_m",(0.,20.,0.)),
            edit("test-particle","initial_velocity_m_s",(.2,0.,0.)))
            for i in range(5)])
    with pytest.raises((TypeError,ValueError)):
        CoupledPhysicalPlan("m31a:bad",native,[
            hypothesis("x","baseline",mass,field)])
    with pytest.raises((TypeError,ValueError)):
        run_coupled_physical_family({"fake":"native"})


def test_n2_forged_trajectory_or_parent_result_binding_refused():
    plan=make(
        hypothesis("p1","baseline",
                   edit("field:E0","field_v_m",(200.,0.,0.)),
                   edit("test-particle","mass_kg",2e-5)),
        hypothesis("p2","replace-Ey",
                   edit("field:Ey-new","field_v_m",(0.,120.,0.)),
                   edit("test-particle","mass_kg",2e-5)))
    run=run_coupled_physical_family(plan)
    assert [x.status for x in run.entries]==["complete","complete"]
    bad=replace(run.entries[0],trajectory=run.entries[1].trajectory)
    with pytest.raises(ValueError,match="trajectory does not match"):
        CoupledPhysicalRun(plan,run.structural_run,(bad,run.entries[1]))
    bad=replace(run.entries[1],parent_scenario=run.plan.family.baseline)
    with pytest.raises(ValueError,match="wrong structural source world"):
        CoupledPhysicalRun(plan,run.structural_run,(run.entries[0],bad))



def _actual_complete_coupled_family():
    return run_coupled_physical_family(make(hypothesis(
        "genuine-native", "add-E",
        edit("field:Enew", "field_v_m", (150., 0., 0.)),
        edit("test-particle", "mass_kg", 2e-5))))


def test_f01_reject_non_native_trajectory_despite_matching_spec_and_fake_as_dict():
    run = _actual_complete_coupled_family()
    valid = run.entries[0]

    class FakeTrajectory:
        def __init__(self, spec):
            self.spec = spec
            self.termination = "duration"

        def as_dict(self):
            return {"schema": "forged", "samples": [{"time_s": 9999.}]}

    imposter = replace(valid, trajectory=FakeTrajectory(valid.scenario.spec),
                      comparisons=())
    with pytest.raises(TypeError, match="native TestChargeMotionResult"):
        CoupledPhysicalRun(run.plan, run.structural_run, (imposter,))


def test_f01_unknown_termination_is_not_renamed_to_censored_in_result_or_runner(monkeypatch):
    import conductor_sim.coupled_physical_family as service

    run = _actual_complete_coupled_family()
    valid = run.entries[0]
    invalid = replace(valid, trajectory=replace(valid.trajectory, termination="unknown"),
                      status="censored", comparisons=())
    with pytest.raises(ValueError, match="unsupported native test-charge termination"):
        CoupledPhysicalRun(run.plan, run.structural_run, (invalid,))

    original_integrator = service.integrate_test_charge

    def unsupported_integrator(spec):
        return replace(original_integrator(spec), termination="unknown")

    monkeypatch.setattr(service, "integrate_test_charge", unsupported_integrator)
    with pytest.raises(ValueError, match="unsupported native test-charge termination"):
        service.run_coupled_physical_family(run.plan)


def test_f01_reject_invented_comparison_timestamps_and_missing_saved_time_rows():
    run = _actual_complete_coupled_family()
    valid = run.entries[0]
    assert valid.comparisons
    forged = replace(valid, comparisons=(
        {"time_s": 9999., "delta_position_m": (999., 0., 0.)},))
    with pytest.raises(ValueError, match="comparison rows"):
        CoupledPhysicalRun(run.plan, run.structural_run, (forged,))
    missing = replace(valid, comparisons=())
    with pytest.raises(ValueError, match="comparison rows"):
        CoupledPhysicalRun(run.plan, run.structural_run, (missing,))


def test_f01_preexport_recheck_refuses_mutable_comparison_row_tampering():
    run = _actual_complete_coupled_family()
    assert run.as_dict()["entries"][0]["exact_time_comparisons"]
    run.entries[0].comparisons[0]["time_s"] = 9999.
    with pytest.raises(ValueError, match="comparison rows"):
        run.as_dict()


def test_f01_real_source_guard_remains_censored_and_joint_translation_still_passes():
    guarded = run_coupled_physical_family(make(hypothesis(
        "guard-ratio", "attract-guard",
        edit("point:attract", "charge_c", -2e-10),
        edit("test-particle", "mass_kg", 2e-4)),
        family=model("guard-and-refusal")))
    node = guarded.entries[0]
    assert node.status == "censored"
    assert node.trajectory.termination == "source-guard"
    assert node.comparisons == tuple(_shared_rows(
        node.structural_entry.trajectory, node.trajectory))
    assert guarded.as_dict()["entries"][0]["physical_trajectory"]["termination"] == "source-guard"

    translated = run_coupled_physical_family(make(hypothesis(
        "translated-pair", "baseline",
        edit("test-particle", "position_m", (0., .1, .05)),
        edit("point:A", "position_m", (.2, 0., .05))),
        family=model("point-replace")))
    child = translated.entries[0]
    assert child.status == "complete"
    assert child.trajectory.termination == "duration"
    assert len(child.comparisons) == len(child.trajectory.samples)
    assert translated.as_dict()["entries"][0]["exact_time_comparisons"]


def test_f01r_boolean_zero_timestamp_rejected_at_construction_and_export():
    # Python's ordinary False == 0.0 must not authenticate an SI timestamp
    # or silently serialize a number as the JSON Boolean false.
    run = _actual_complete_coupled_family()
    real = run.entries[0]
    assert type(real.comparisons[0]["time_s"]) is float
    assert real.comparisons[0]["time_s"] == 0.0
    assert False == 0.0  # characterize the historical validator bypass

    doctored = dict(real.comparisons[0])
    doctored["time_s"] = False
    changed = replace(real, comparisons=(doctored, *real.comparisons[1:]))
    with pytest.raises(ValueError, match="comparison rows"):
        CoupledPhysicalRun(run.plan, run.structural_run, (changed,))

    # Frozen container, mutable row: export must check AGAIN.
    second = _actual_complete_coupled_family()
    second.entries[0].comparisons[0]["time_s"] = False
    with pytest.raises(ValueError, match="comparison rows"):
        second.as_dict()


def test_f01r_custom_equality_float_and_dict_subclasses_do_not_forge_comparisons():
    run = _actual_complete_coupled_family()
    good = run.entries[0]

    class AlwaysEqualFloat(float):
        def __eq__(self, _other):
            return True

    class AlwaysEqualDict(dict):
        def __eq__(self, _other):
            return True

    doctored_time = dict(good.comparisons[0])
    doctored_time["time_s"] = AlwaysEqualFloat(9999.0)
    assert doctored_time == good.comparisons[0]  # original equality is unsafe
    with pytest.raises(ValueError, match="comparison rows"):
        CoupledPhysicalRun(run.plan, run.structural_run, (
            replace(good, comparisons=(doctored_time, *good.comparisons[1:])),))

    doctored_row = AlwaysEqualDict({"time_s": 9999.0})
    assert doctored_row == good.comparisons[0]
    with pytest.raises(ValueError, match="comparison rows"):
        CoupledPhysicalRun(run.plan, run.structural_run, (
            replace(good, comparisons=(doctored_row, *good.comparisons[1:])),))

    # +0.0 and -0.0 have different representations in JSON.
    doctored_sign = dict(good.comparisons[0])
    doctored_sign["time_s"] = -0.0
    assert doctored_sign == good.comparisons[0]
    with pytest.raises(ValueError, match="comparison rows"):
        CoupledPhysicalRun(run.plan, run.structural_run, (
            replace(good, comparisons=(doctored_sign, *good.comparisons[1:])),))


def test_f01r_native_literal_time_and_censored_controls_remain_accepted():
    original = _actual_complete_coupled_family()
    assert original.as_dict()["entries"][0]["exact_time_comparisons"][0]["time_s"] == 0.0
    assert type(original.as_dict()["entries"][0]["exact_time_comparisons"][0]["time_s"]) is float

    guard = run_coupled_physical_family(make(hypothesis(
        "guard-literal", "attract-guard",
        edit("point:attract", "charge_c", -2e-10),
        edit("test-particle", "mass_kg", 2e-4)),
        family=model("guard-and-refusal")))
    row = guard.entries[0]
    assert row.status == "censored"
    assert row.trajectory.termination == "source-guard"
    # Detached JSON turns the genuine in-memory 3-vector tuples into arrays.
    # Compare the same canonical JSON shape, not a tuple to a JSON list.
    assert guard.as_dict()["entries"][0]["exact_time_comparisons"] == json.loads(
        json.dumps(row.comparisons, allow_nan=False))
    assert all(type(record["time_s"]) is float
               for record in guard.as_dict()["entries"][0]["exact_time_comparisons"])
