"""M30a: real source-structure → real physical-property composition cards.

Only the accepted M22a/M29a/M20b/M28a public/native physics functions may
perform reassembly, numerical integration and time alignment in production.
The independent references below are test-side analytic/SI evaluations.
"""
from dataclasses import FrozenInstanceError
from math import hypot, pi

import pytest

from conductor_sim import (
    BranchSweep, EPSILON_0, FixedPointField, PointCharge, PrescribedField,
    StructuralParameterPlan, StructuralParameterRun, SweepAxis,
    TestChargeMotionSpec, UniformElectricField, UniformGravity,
    UniformRodField, run_structural_parameter_family,
)
from conductor_sim.scenarios import PrescribedScenario
from conductor_sim.structural_family import StructuralCase, StructuralFamilyPlan
from conductor_sim.structural_scenarios import NamedSource
from examples.m29a_structural_family import model


def _study(*branches, family=None):
    return StructuralParameterPlan(
        "m30a:analytic",
        model("uniform-edits") if family is None else family,
        tuple(branches))


def test_p1_newly_added_source_is_real_parameter_target_and_analytic_trajectory():
    plan=_study(BranchSweep(
        "add-E", SweepAxis(
            "field:Enew", "field_v_m", ((50., 0., 0.), (160., 0., 0.)))))
    original=plan.family.baseline.as_dict()
    run=run_structural_parameter_family(plan)
    assert type(run) is StructuralParameterRun
    branch=run.entries[0]
    assert branch.status=="evaluated"
    assert branch.structural_entry.case.case_id=="add-E"
    structural=branch.parent_scenario
    assert structural.source_annotations[-1][0]=="field:Enew"
    assert structural.spec.field.sources[-1].field_v_m==(100.,0.,0.)
    assert branch.structural_entry.assembly.parent is plan.family.baseline
    assert branch.structural_entry.assembly.as_dict()["parent_revision_id"]==original["revision_id"]
    assert [v.status for v in branch.sweep_run.entries]==["complete","complete"]
    assert plan.family.baseline.as_dict()==original
    q=1e-9; mass=1e-5
    for source_e, entry in zip((50.,160.),branch.sweep_run.entries):
        scenario=entry.scenario
        assert scenario.parent_revision_id==structural.revision_id
        assert scenario.source_annotations==structural.source_annotations
        assert scenario.spec.field.sources[-1].field_v_m==(source_e,0.,0.)
        assert [c.component_id for c in scenario.changes]==["field:Enew"]
        for sample in entry.trajectory.samples:
            t=sample.time_s
            ax=q*(100.+source_e)/mass
            ay=q*50./mass
            assert sample.position_m == pytest.approx(
                (.1*t+.5*ax*t*t,.5*ay*t*t,0.),abs=2e-10)
            assert sample.velocity_m_s == pytest.approx(
                (.1+ax*t,ay*t,0.),abs=2e-9)
    audit=run.as_dict()
    assert audit["schema"]=="conductor_sim.structural_parameter_family.v1"
    assert audit["original_revision_id"]==original["revision_id"]
    assert audit["entries"][0]["structural_assembly"]["parent_revision_id"]==original["revision_id"]
    assert audit["entries"][0]["parameter_revision_parents"]==[structural.revision_id]*2
    assert audit["time_interpolation"] is False and audit["postguard_extrapolation"] is False
    assert len(audit["entries"][0]["trajectory_sweep"]["entries"][0]["comparisons"])==11
    audit["entries"][0]["trajectory_sweep"]["entries"][0]["trajectory"]["samples"][0]["position_m"][0]=999
    assert branch.sweep_run.entries[0].trajectory.samples[0].position_m[0]!=999
    assert plan.family.baseline.as_dict()==original


def test_p2_retired_identity_is_not_reintroduced_by_parameter_revision():
    plan=_study(BranchSweep(
        "replace-Ey", SweepAxis(
            "field:Ey-new", "field_v_m", ((0.,60.,0.),(0.,130.,0.)))))
    run=run_structural_parameter_family(plan)
    row=run.entries[0]
    assert row.status=="evaluated"
    assert row.structural_entry.assembly.changes[0]["operation"]=="replace-source"
    expected=("field:E0","field:Ey-new")
    assert tuple(i for i,label in row.parent_scenario.source_annotations)==expected
    for entry in row.sweep_run.entries:
        assert tuple(i for i,label in entry.scenario.source_annotations)==expected
        assert entry.scenario.parent_revision_id==row.parent_scenario.revision_id
        assert all(change.component_id=="field:Ey-new" for change in entry.scenario.changes)
        assert "field:Ey" not in dict(entry.scenario.source_annotations)
        assert entry.trajectory.spec is entry.scenario.spec
    assert tuple(i for i,label in plan.family.baseline.source_annotations)==(
        "field:E0","field:Ey")


def test_p3_new_gravity_membership_can_be_swept_in_new_world():
    plan=_study(BranchSweep(
        "add-gravity", SweepAxis(
            "gravity","acceleration_m_s2",((0.,-1.,0.),(0.,-4.,0.)))))
    result=run_structural_parameter_family(plan).entries[0]
    assert result.status=="evaluated"
    assert result.parent_scenario.spec.gravity.acceleration_m_s2==(0.,-2.,0.)
    assert result.structural_entry.assembly.changes[0]["operation"]=="add-gravity"
    for g, entry in zip((-1.,-4.),result.sweep_run.entries):
        assert entry.status=="complete"
        assert entry.scenario.spec.gravity.acceleration_m_s2==(0.,g,0.)
        assert [c.component_id for c in entry.scenario.changes]==["gravity"]
        for sample in entry.trajectory.samples:
            t=sample.time_s
            ax=1e-9*100/1e-5
            ay=1e-9*50/1e-5+g
            assert sample.position_m==pytest.approx(
                (.1*t+.5*ax*t*t,.5*ay*t*t,0.),abs=3e-10)
            assert sample.gravitational_force_n==pytest.approx(
                (0.,1e-5*g,0.),abs=1e-20)


def test_p4_structural_refusal_and_wrong_branch_axis_are_separate():
    plan=_study(
        BranchSweep("unpaired",SweepAxis("point:single","charge_c",(1e-9,3e-9))),
        BranchSweep("extra-field",SweepAxis("point:single","charge_c",(1e-9,3e-9))),
        BranchSweep("mirror-two",SweepAxis("test-particle","mass_kg",(2e-5,3e-5))),
        family=model("mirror-pair"))
    run=run_structural_parameter_family(plan)
    assert [e.status for e in run.entries]==[
        "skipped-structural-model","rejected-sweep-axis","evaluated"]
    assert run.entries[0].parent_scenario is None
    assert run.entries[0].structural_entry.assembly is None
    assert run.entries[1].parent_scenario is not None
    assert run.entries[1].sweep_run is None
    assert "unknown physical source identity" in run.entries[1].reason
    assert run.entries[1].structural_entry.assembly is not None
    assert all(e.status=="complete" for e in run.entries[2].sweep_run.entries)
    record=run.as_dict()
    assert record["entries"][0]["source_inventory"] is None
    assert record["entries"][0]["structural_assembly"] is None
    assert record["entries"][1]["source_inventory"] is not None
    assert record["entries"][1]["structural_assembly"] is not None
    assert record["entries"][1]["trajectory_sweep"] is None
    assert record["entries"][2]["parameter_revision_parents"]==[
        run.entries[2].parent_scenario.revision_id]*2


def test_p5_bad_parameter_refused_without_losing_valid_sibling_or_parent():
    plan=_study(
        BranchSweep("add-E",SweepAxis("field:Enew","field_v_m",
                 ((0.,0.,100.),(150.,0.,0.)))),
        BranchSweep("add-gravity",SweepAxis("gravity","acceleration_m_s2",
                 ((0.,-1.,0.),))))
    run=run_structural_parameter_family(plan)
    assert [e.status for e in run.entries]==["evaluated","evaluated"]
    assert [e.status for e in run.entries[0].sweep_run.entries]==[
        "rejected-physical-model","complete"]
    bad, good=run.entries[0].sweep_run.entries
    assert bad.scenario is None and bad.trajectory is None
    assert good.trajectory.termination=="duration"
    assert run.entries[0].parent_scenario.spec.field.sources[-1].field_v_m==(100.,0.,0.)
    assert run.entries[1].sweep_run.entries[0].status=="complete"
    assert run.as_dict()["entries"][0]["parameter_revision_parents"][0] is None


def test_p5_source_guard_remains_native_censored_and_no_postguard_alignment():
    plan=_study(BranchSweep(
        "attract-guard",SweepAxis("point:attract","charge_c",
                                 (-5e-11,-2e-10))),
        family=model("guard-and-refusal"))
    run=run_structural_parameter_family(plan)
    parent=run.entries[0]
    assert parent.status=="evaluated"
    assert parent.structural_entry.trajectory.termination=="source-guard"
    assert parent.sweep_run.baseline_trajectory.termination=="source-guard"
    for child in parent.sweep_run.entries:
        if child.trajectory is None:
            assert child.status in ("rejected-physical-model","integration-failed")
            continue
        assert child.status in ("complete","censored")
        if child.trajectory.termination=="source-guard":
            assert child.trajectory.samples[-1].time_s<.25
        assert all(
            row["time_s"] in {x.time_s for x in parent.sweep_run.baseline_trajectory.samples}
            for row in child.as_dict(parent.sweep_run.baseline_trajectory)["comparisons"])
    assert run.as_dict()["postguard_extrapolation"] is False


def test_p6_unfamiliar_tilted_rod_and_added_held_point_charge_initial_force():
    sources=(
        UniformRodField(3e-9,(.11,-.10,0.),(.13,.12,0.)),
        FixedPointField(2e-9,(.18,.15,0.)),
        FixedPointField(-1e-9,(-.16,.10,0.)),
    )
    original=PrescribedScenario(
        "m30a:novel:original",
        TestChargeMotionSpec(
            PrescribedField(sources),PointCharge(1e-9,(0.,0.,0.)),
            1e-5,(.05,.02,0.),.7,.005,plane_normal=(0.,0.,1.),
            relative_tolerance=1e-10,gravity=UniformGravity((0.,-.1,0.))),
        (("rod:tilted","Tilted charged rod"),
         ("point:plus","Original positive held point"),
         ("point:minus","Original negative held point")),
        "Teacher hypothesis: unfamiliar 2D real Coulomb source arrangement.")
    family=StructuralFamilyPlan("m30a:novel",original,(
        StructuralCase("new-point","Add held point for property experimentation.",
            add_sources=(NamedSource("point:new","Added signed held charge",
                FixedPointField(1.7e-9,(.21,-.09,0.))),)),))
    plan=StructuralParameterPlan("m30a:novel:sweep",family,(
        BranchSweep("new-point",SweepAxis("point:new","charge_c",(1.2e-9,2e-9))),))
    result=run_structural_parameter_family(plan)
    entry=result.entries[0]
    assert entry.status=="evaluated"
    assert all(child.status=="complete" for child in entry.sweep_run.entries)
    assert [x["id"] for x in result.as_dict()["entries"][0]["source_inventory"]]==[
        "rod:tilted","point:plus","point:minus","point:new"]
    k=1/(4*pi*EPSILON_0)
    a=(.11,-.10,0.);b=(.13,.12,0.)
    for target_q,child in zip((1.2e-9,2e-9),entry.sweep_run.entries):
        origin=child.trajectory.samples[0]
        # Independently integrate Coulomb contribution of rod by a midpoint
        # Riemann sum. This does not use production UniformRodField E().
        N=3000
        e=[0.,0.,0.]
        for j in range(N):
            u=(j+.5)/N
            rp=tuple(a[i]+u*(b[i]-a[i]) for i in range(3))
            d=tuple(-x for x in rp)
            r=hypot(*d)
            for i in range(3):
                e[i]+=k*3e-9*d[i]/(r*r*r*N)
        for q,pos in ((2e-9,(.18,.15,0.)),(-1e-9,(-.16,.10,0.)),
                       (target_q,(.21,-.09,0.))):
            d=tuple(-x for x in pos);r=hypot(*d)
            for i in range(3):e[i]+=k*q*d[i]/r**3
        force=tuple(1e-9*e[i]+1e-5*(0.,-.1,0.)[i] for i in range(3))
        assert origin.force_n==pytest.approx(force,abs=1.5e-12)
        assert origin.acceleration_m_s2==pytest.approx(
            tuple(x/1e-5 for x in force),abs=2e-7)
        assert child.scenario.parent_revision_id==entry.parent_scenario.revision_id
        assert child.scenario.spec.field.sources[-1].charge_c==target_q
    assert original.as_dict()==result.plan.family.baseline.as_dict()


def test_n1_invalid_branch_or_variations_and_unknown_lattice_path_rejected():
    fam=model("uniform-edits")
    axis=SweepAxis("test-particle","mass_kg",(2e-5,))
    with pytest.raises(ValueError,match="1..3"):
        StructuralParameterPlan("m30a:x",fam,())
    with pytest.raises(ValueError,match="structural case"):
        StructuralParameterPlan("m30a:x",fam,(BranchSweep("unknown",axis),))
    with pytest.raises(ValueError,match="only one sweep"):
        StructuralParameterPlan("m30a:x",fam,(
            BranchSweep("baseline",axis),BranchSweep("baseline",axis)))
    with pytest.raises(ValueError,match="at most 12"):
        StructuralParameterPlan("m30a:x",fam,(
            BranchSweep("baseline",SweepAxis("test-particle","mass_kg",
                                             (2e-5,3e-5,4e-5,5e-5,6e-5))),
            BranchSweep("add-E",SweepAxis("test-particle","mass_kg",
                                          (2e-5,3e-5,4e-5,5e-5,6e-5))),
            BranchSweep("remove-Ey",SweepAxis("test-particle","mass_kg",
                                              (2e-5,3e-5,4e-5,5e-5,6e-5))))))
    with pytest.raises((TypeError,ValueError)):
        BranchSweep("baseline",object())
    with pytest.raises((TypeError,ValueError)):
        StructuralParameterPlan("m30a:bad",fam,[BranchSweep("baseline",axis)])
    with pytest.raises((TypeError,ValueError)):
        run_structural_parameter_family({"untrusted":"json"})
    with pytest.raises((TypeError,ValueError)):
        SweepAxis("test-particle","mass_kg",(True,))


def test_n2_branch_axis_no_op_and_gauge_not_silently_swept():
    plan=_study(
        BranchSweep("add-E",SweepAxis("field:Enew","field_v_m",
                                      ((100.,0.,0.),))),
        BranchSweep("replace-Ey",SweepAxis("field:Ey","field_v_m",
                                      ((0.,120.,0.),))))
    run=run_structural_parameter_family(plan)
    assert [e.status for e in run.entries]==[
        "rejected-sweep-axis","rejected-sweep-axis"]
    assert "baseline value" in run.entries[0].reason
    assert "unknown physical source identity" in run.entries[1].reason
    gauge=StructuralParameterPlan("m30a:gauge",model("uniform-edits"),(
        BranchSweep("baseline",SweepAxis("field:E0","reference_potential_v",(1.,))),))
    denied=run_structural_parameter_family(gauge)
    assert denied.entries[0].status=="rejected-sweep-axis"
    assert "unsupported physical sweep property" in denied.entries[0].reason
    # Physical shape errors detected by the existing M28a plan on the selected parent.
    wrong=_study(BranchSweep("add-E",SweepAxis("field:Enew","field_v_m",(3.0,))))
    response=run_structural_parameter_family(wrong)
    assert response.entries[0].status=="rejected-sweep-axis"
