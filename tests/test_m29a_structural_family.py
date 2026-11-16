"""M29a: native real-time comparison across source-membership structural reassembly.

Source laws, qualifications and adaptive RK are historical code. These tests
challenge structural lineage/identity, analytic force, real motion, source
guards, world-plane refusal and honest source-specific browser evidence.
"""
from dataclasses import FrozenInstanceError, replace
import json
from math import hypot, pi

import pytest

from conductor_sim import (
    EPSILON_0, FixedPointField, PointCharge, PrescribedField,
    TestChargeMotionSpec, UniformElectricField, UniformGravity,
    UniformSphereField, integrate_test_charge,
)
from conductor_sim.scenarios import PrescribedScenario
from conductor_sim.structural_scenarios import (
    NamedSource, SourceReplacement, reassemble_scenario,
)
from conductor_sim.structural_family import (
    StructuralCase, StructuralFamilyPlan, StructuralFamilyRun,
    run_structural_family,
)
from conductor_sim.structural_family_viewer import (
    render_structural_family, write_structural_family_html,
)
from examples.m29a_structural_family import CASES, model, run_case


def test_p1_uniform_structural_family_real_fresh_analytic_kinematics():
    plan=model("uniform-edits")
    original=plan.baseline.as_dict()
    run=run_structural_family(plan)
    assert run.plan is plan
    assert run.baseline_trajectory.spec is plan.baseline.spec
    assert [e.status for e in run.entries]==["complete"]*4
    assert [e.case.case_id for e in run.entries]==[
        "add-E","remove-Ey","add-gravity","replace-Ey"]
    assert all(e.assembly.parent is plan.baseline for e in run.entries)
    assert all(e.assembly.scenario.spec.duration_s == plan.baseline.spec.duration_s
               for e in run.entries)
    assert all(e.assembly.scenario.spec.particle == plan.baseline.spec.particle
               for e in run.entries)
    assert plan.baseline.as_dict()==original
    assert len(run.baseline_trajectory.samples)==11
    assert [x.time_s for x in run.baseline_trajectory.samples]==[
        x.time_s for x in run.entries[0].trajectory.samples]
    q=plan.baseline.spec.particle.charge_c
    m=plan.baseline.spec.mass_kg
    base_ax=q*100/m
    added_ax=q*200/m
    base_ay=q*50/m
    for spec,track,ax,ay in (
        (plan.baseline.spec,run.baseline_trajectory,base_ax,base_ay),
        (run.entries[0].assembly.scenario.spec,run.entries[0].trajectory,added_ax,base_ay),
        (run.entries[1].assembly.scenario.spec,run.entries[1].trajectory,base_ax,0.),
        (run.entries[2].assembly.scenario.spec,run.entries[2].trajectory,base_ax,base_ay-2.),
        (run.entries[3].assembly.scenario.spec,run.entries[3].trajectory,base_ax,q*90/m),
    ):
        assert track.spec is spec
        for row in track.samples:
            t=row.time_s
            assert row.position_m == pytest.approx(
                (.1*t + .5*ax*t*t, .5*ay*t*t, 0.), abs=5e-12)
            assert row.velocity_m_s == pytest.approx(
                (.1+ax*t,ay*t,0.), abs=5e-11)
    delta=run.entries[0].as_dict(run.baseline_trajectory)["comparisons"]
    assert len(delta)==11
    for item in delta:
        t=item["time_s"]
        assert item["delta_position_m"]==pytest.approx(
            (.5*base_ax*t*t,0.,0.), abs=5e-12)
        assert item["delta_velocity_m_s"]==pytest.approx(
            (base_ax*t,0.,0.), abs=5e-11)
    assert run.entries[0].assembly.changes[0]["operation"]=="add-source"
    assert run.entries[1].assembly.changes[0]["operation"]=="remove-source"
    assert run.entries[2].assembly.changes[0]["operation"]=="add-gravity"
    assert run.entries[3].assembly.changes[0]["operation"]=="replace-source"
    assert run.entries[3].assembly.scenario.source_annotations==(
        ("field:E0","Original physical +x field"),
        ("field:Ey-new","Replacement +y field"),
    )
    # Replacement identity is not a modified old source pretending to be new.
    assert "field:Ey" not in dict(run.entries[3].assembly.scenario.source_annotations)
    payload=run.as_dict()
    assert payload["schema"]=="conductor_sim.structural_motion_family.v1"
    assert payload["time_interpolation"] is False
    assert payload["postguard_extrapolation"] is False
    assert payload["baseline_revision_id"]==plan.baseline.revision_id
    assert all(e["structural_assembly"]["parent_revision_id"]==
               plan.baseline.revision_id for e in payload["entries"])
    assert all(e["structural_assembly"]["child_revision_id"]==
               e["structural_assembly"]["child_model"]["revision_id"]
               for e in payload["entries"])
    payload["entries"][0]["trajectory"]["samples"][0]["position_m"][0]=999.
    assert run.entries[0].trajectory.samples[0].position_m[0]!=999.
    assert plan.baseline.as_dict()==original


def test_p3_point_to_volume_sphere_retires_source_id_and_changes_force():
    run=run_case("point-replace")
    assert len(run.entries)==2
    assert [e.status for e in run.entries]==["complete","complete"]
    replacement=run.entries[0]
    assert replacement.assembly.scenario.spec.field.sources[0].__class__ is UniformSphereField
    assert replacement.assembly.scenario.source_annotations[0][0]=="sphere:S"
    assert replacement.assembly.changes[0]["operation"]=="replace-source"
    assert replacement.assembly.changes[0]["before"]["id"]=="point:A"
    assert replacement.assembly.changes[0]["after"]["id"]=="sphere:S"
    assert "point:A" not in dict(replacement.assembly.scenario.source_annotations)
    assert replacement.assembly.scenario.spec.particle == run.plan.baseline.spec.particle
    p=run.plan.baseline.spec.particle.position_m
    old=run.baseline_trajectory.samples[0].acceleration_m_s2
    new=replacement.trajectory.samples[0].acceleration_m_s2
    # Distinct physical laws: inside a transparent uniform volume sphere
    # E=kQ(r-center)/R^3; positive held point has E=kQd/|d|^3.
    source=run.plan.baseline.spec.field.sources[0]
    r=tuple(p[k]-source.position_m[k] for k in range(3))
    k=1/(4*pi*EPSILON_0)
    length=hypot(*r)
    q=run.plan.baseline.spec.particle.charge_c
    mass=run.plan.baseline.spec.mass_kg
    expected_old=tuple(q*k*source.charge_c*v/(length**3*mass) for v in r)
    R=.35
    expected_new=tuple(q*k*source.charge_c*v/(R**3*mass) for v in r)
    assert old==pytest.approx(expected_old,abs=1e-10)
    assert new==pytest.approx(expected_new,abs=1e-10)
    assert old!=pytest.approx(new,abs=1e-7)
    deleted=run.entries[1]
    assert deleted.assembly.scenario.spec.field.sources==()
    assert deleted.assembly.scenario.source_annotations==()
    assert deleted.assembly.changes[0]["operation"]=="remove-source"


def test_p4_atomic_off_plane_mirror_pair_qualified_and_single_refused():
    plan=model("mirror-pair")
    root=plan.baseline.as_dict()
    run=run_structural_family(plan)
    assert [x.status for x in run.entries]==[
        "complete","rejected-structural-model","complete"]
    assert run.entries[1].assembly is None and run.entries[1].trajectory is None
    both=run.entries[0]
    assert [c.source_id for c in plan.cases[0].add_sources]==[
        "point:upper","point:lower"]
    assert [op["operation"] for op in both.assembly.changes]==[
        "add-source","add-source"]
    assert both.assembly.scenario.source_annotations[0][0]=="point:centre"
    assert both.trajectory.spec is both.assembly.scenario.spec
    assert both.trajectory.termination=="duration"
    # New physical geometry must yield actual independent direct Coulomb
    # reference acceleration, never merely a projected 2D calculation.
    charges=both.assembly.scenario.spec.field.sources
    launch=both.assembly.scenario.spec.particle.position_m
    const=both.assembly.scenario.spec.particle.charge_c/(4*pi*EPSILON_0*both.assembly.scenario.spec.mass_kg)
    ref=[0.,0.,0.]
    for src in charges:
        d=[launch[k]-src.position_m[k] for k in range(3)]
        l=hypot(*d)
        for k in range(3):
            ref[k]+=const*src.charge_c*d[k]/l**3
    acc=both.trajectory.samples[0].acceleration_m_s2
    assert acc==pytest.approx(tuple(ref),abs=5e-9)
    assert acc[2]==pytest.approx(0.,abs=5e-14)
    assert run.plan.baseline.as_dict()==root


def test_p5_runtime_source_guard_retains_censored_run_and_valid_siblings():
    run=run_case("guard-and-refusal")
    assert run.baseline_trajectory.termination=="duration"
    assert [e.status for e in run.entries]==[
        "censored","complete","rejected-structural-model"]
    first=run.entries[0]
    assert first.trajectory.termination=="source-guard"
    t_guard=first.trajectory.samples[-1].time_s
    assert .15 < t_guard < .25
    assert t_guard < run.plan.baseline.spec.duration_s
    assert first.trajectory.samples[-1].time_s not in [
        s.time_s for s in run.baseline_trajectory.samples]
    comparisons=first.as_dict(run.baseline_trajectory)["comparisons"]
    assert comparisons
    assert all(row["time_s"] <= t_guard for row in comparisons)
    assert max(row["time_s"] for row in comparisons) < t_guard
    assert max(row["time_s"] for row in comparisons) < .25
    assert all(row["time_s"] in {s.time_s for s in run.baseline_trajectory.samples}
               for row in comparisons)
    second=run.entries[1]
    assert second.trajectory.termination=="duration"
    assert second.trajectory.samples[-1].time_s==pytest.approx(.25)
    assert len(second.as_dict(run.baseline_trajectory)["comparisons"])==26
    reject=run.entries[2]
    assert reject.assembly is None and reject.trajectory is None
    assert reject.as_dict(run.baseline_trajectory)["comparisons"]==[]
    assert run.plan.baseline.spec.field.sources[0] == UniformElectricField((0.,0.,0.))
    record=run.as_dict()
    assert record["entries"][0]["status"]=="censored"
    assert record["entries"][2]["status"]=="rejected-structural-model"


def test_n1_malformed_family_and_immutable_operation_type():
    baseline=model("uniform-edits").baseline
    with pytest.raises(ValueError,match="1..5"):
        StructuralFamilyPlan("m29a:empty",baseline,())
    with pytest.raises(ValueError,match="1..5"):
        StructuralFamilyPlan("m29a:excess",baseline,tuple(
            StructuralCase(f"case{i}",f"Remove Ey in case {i}",
                           remove_sources=("field:Ey",)) for i in range(6)))
    with pytest.raises(ValueError,match="distinct"):
        StructuralFamilyPlan("m29a:dups",baseline,(
            StructuralCase("same","One option.",remove_sources=("field:Ey",)),
            StructuralCase("same","Another option.",remove_sources=("field:E0",)),
        ))
    with pytest.raises(TypeError,match="immutable tuple"):
        StructuralFamilyPlan("m29a:list",baseline,[
            StructuralCase("case","Remove Ey.",remove_sources=("field:Ey",))])
    with pytest.raises(ValueError,match="1..12"):
        StructuralCase("noop","No operation.")
    with pytest.raises(ValueError,match="gravity"):
        StructuralCase("bad-gravity","Conflicting gravity.",gravity_action="remove",
                       gravity=UniformGravity((0.,-2.,0.)))
    with pytest.raises(TypeError,match="UniformGravity"):
        StructuralCase("bad-add","Not gravity.",gravity_action="add",gravity=123)
    with pytest.raises(ValueError,match="bounded"):
        StructuralCase("x"*22,"Too long.",remove_sources=("field:E0",))
    with pytest.raises(ValueError,match="NEW source identity"):
        SourceReplacement("field:E0",
                          NamedSource("field:E0","Same identity",
                                      UniformElectricField((3.,0.,0.))))
    with pytest.raises(Exception):
        StructuralCase("untyped","Not a typed source.",
                       add_sources=("field:foo",))
    with pytest.raises(TypeError,match="native"):
        render_structural_family(run_case("uniform-edits").as_dict())


def test_n2_rejected_structural_siblings_do_not_mutate_baseline_or_others():
    base=model("uniform-edits").baseline
    before=base.as_dict()
    run=run_structural_family(StructuralFamilyPlan(
        "m29a:negative",base,(
            StructuralCase("unknown-remove","Unknown existing ID.",
                           remove_sources=("field:missing",)),
            StructuralCase("collision","Duplicate source ID.",
                           add_sources=(NamedSource("field:E0","Conflicting same ID",
                               UniformElectricField((20.,0.,0.))),)),
            StructuralCase("valid-add","Legal third source.",
                           add_sources=(NamedSource("field:good","Physical source",
                               UniformElectricField((5.,0.,0.))),)),
        )))
    assert [x.status for x in run.entries]==[
        "rejected-structural-model","rejected-structural-model","complete"]
    assert run.entries[0].assembly is None
    assert run.entries[1].assembly is None
    assert run.entries[2].assembly.parent is base
    assert base.as_dict()==before
    assert run.entries[2].assembly.scenario.spec.field.sources[0] is base.spec.field.sources[0]


def test_p6_offline_html_source_provenance_rejects_forgery_and_escapes(tmp_path):
    run=run_case("uniform-edits")
    html=render_structural_family(run,title="<source add/remove/replace>")
    assert "&lt;source add/remove/replace&gt;" in html
    assert "conductor_sim.structural_family_view.v1" in html
    assert 'id="m29a-native-data"' in html
    assert "field:E0" in html and "field:Enew" in html
    assert "source_inventory" in html
    assert "structural_changes" in html
    assert "time_interpolation" in html
    assert "postguard_extrapolation" not in html or "extrapolation" in html
    assert "source_geometry_rendered" in html
    assert "TRAJECTORY SPACE ONLY" in html
    assert "window.m29aStructuralViewer=view;" in html
    assert "inv.map(s=>s.id+' — '+s.label+' ('+s.kind+')')" in html
    assert 'id="table-scroll-hint"' in html
    assert 'aria-label="Structural source inventory and native trajectory table; scroll horizontally to read all columns"' in html
    assert 'min-width:1060px' in html
    assert 'position:sticky' in html
    assert "Workbook" not in html and "Q41" not in html
    with pytest.raises(TypeError,match="native"):
        render_structural_family(run.as_dict())
    p=write_structural_family_html(run,tmp_path/"compare"/"family.html")
    assert p.exists() and p.read_text(encoding="utf-8").startswith("<!doctype html>")
    hostile=PrescribedScenario("m29a:hostile",run.plan.baseline.spec,
        run.plan.baseline.source_annotations,
        'Teacher <img src=x onerror=alert(1)> </script><script>alert("X")</script>')
    case=StructuralCase("new-field","Teacher's illustrative new source.",
        add_sources=(NamedSource("field:E3","New hypothetical E",
                                UniformElectricField((0.,4.,0.))),))
    other=run_structural_family(StructuralFamilyPlan("m29a:escape",hostile,(case,)))
    escaped=render_structural_family(other)
    assert "<img src=x onerror=alert(1)>" not in escaped
    assert "</script><script>alert" not in escaped
    assert r"\u003cimg" in escaped
    assert json.dumps(other.as_dict(),allow_nan=False)


def test_n5_structural_case_and_run_are_immutable():
    c=StructuralCase("simple","Remove optional source.",
                     remove_sources=("field:Ey",))
    with pytest.raises(FrozenInstanceError):
        c.case_id="different"
    run=run_case("uniform-edits")
    with pytest.raises(FrozenInstanceError):
        run.entries[0].status="fake"
    with pytest.raises(ValueError,match="missing structural alternative"):
        replace(run,entries=())
