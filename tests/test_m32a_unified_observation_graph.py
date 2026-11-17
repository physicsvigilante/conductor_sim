"""M32a: genuine composed physical worlds and independently solved observations.

All reference accelerations/roots are direct SI analytic formulae. These tests
never replace a native event integration by presentation interpolation.
"""
from dataclasses import replace
from math import sqrt

import pytest

from conductor_sim import (
    CoupledPropertyEdit, FixedPointField, PhysicalWorldNode,
    StructuralWorldNode, UnifiedObservationPlan, UnifiedObservationRun,
    UniformElectricField, run_unified_observation_graph,
)
from conductor_sim.motion_events import PlaneEvent, EventPolicy
from conductor_sim.structural_family import StructuralCase
from conductor_sim.structural_scenarios import NamedSource, SourceReplacement
from conductor_sim.scenarios import revise_scenario
from conductor_sim.solver import UnsupportedProblem
from examples.m29a_structural_family import model
from examples.m29b_structural_events import make_plan


def physical(node, parent, *values, observation="trajectory"):
    return PhysicalWorldNode(node, parent, "Teacher-authored atomic physical scenario.",
                             tuple(CoupledPropertyEdit(*row) for row in values),
                             observation)


def detector():
    return PlaneEvent("x:detector", (.006, 0., 0.), (1., 0., 0.),
                      "Positive x physical detector", direction=1)


def policy():
    return EventPolicy(.003, 1e-8, 1e-8, 1e-6)


def events(nodes, *, original=None, questions=None, event_policy=None,
           baseline="events"):
    return UnifiedObservationPlan(
        "m32a:world-graph", model("uniform-edits").baseline if original is None else original,
        tuple(nodes), baseline, (detector(),) if questions is None else questions,
        policy() if event_policy is None else event_policy)


def _positive_root(distance, v0, a):
    return 2*distance/(v0+sqrt(v0*v0+2*a*distance))


def test_p1_graph_branches_and_true_nested_joint_physical_event_roots():
    family=model("uniform-edits")
    cases={c.case_id:c for c in family.cases}
    p=events((
        StructuralWorldNode("add-E","baseline",cases["add-E"],"events"),
        StructuralWorldNode("remove-Ey","baseline",cases["remove-Ey"],"trajectory"),
        physical("source-mass","add-E",
                 ("field:Enew","field_v_m",(150.,0.,0.)),
                 ("test-particle","mass_kg",2e-5),observation="events"),
        physical("launch","source-mass",
                 ("test-particle","initial_velocity_m_s",(.16,0.,0.)),
                 observation="events"),
    ),original=family.baseline)
    original=p.original.as_dict()
    out=run_unified_observation_graph(p)
    assert type(out) is UnifiedObservationRun
    assert out.baseline_status=="complete"
    assert [x.status for x in out.entries]==["complete"]*4
    assert [x.node.node_id for x in out.entries]==[
        "add-E","remove-Ey","source-mass","launch"]
    assert [x.node.parent_id for x in out.entries]==[
        "baseline","baseline","add-E","source-mass"]
    assert out.entries[2].scenario.parent_revision_id==out.entries[0].scenario.revision_id
    assert out.entries[3].scenario.parent_revision_id==out.entries[2].scenario.revision_id
    assert out.entries[2].scenario.spec.field.sources[-1].field_v_m==(150.,0.,0.)
    assert out.entries[2].scenario.spec.mass_kg==2e-5

    for native in (out.baseline_native,out.entries[0].native,
                   out.entries[2].native,out.entries[3].native):
        assert native.trajectory.termination=="duration"
        x0=native.study.scenario.spec
        ex=sum(s.field_v_m[0] for s in x0.field.sources)
        a=x0.particle.charge_c*ex/x0.mass_kg
        root=[e for e in native.events if e.event_id=="x:detector"]
        assert len(root)==1 and root[0].direction==1
        assert root[0].sample.time_s==pytest.approx(
            _positive_root(.006,x0.initial_velocity_m_s[0],a),abs=1e-7)
    evidence=out.as_dict()
    assert evidence["schema"]=="conductor_sim.unified_observation_graph.v1"
    assert evidence["plan"]["original_revision_id"]==p.original.revision_id
    assert evidence["time_interpolation"] is False
    assert evidence["postguard_extrapolation"] is False
    assert evidence["nodes"][2]["parent_revision_id"]==out.entries[0].scenario.revision_id
    assert evidence["nodes"][3]["parent_revision_id"]==out.entries[2].scenario.revision_id
    assert evidence["nodes"][2]["node"]["edits"][0]["component_id"]=="field:Enew"
    assert evidence["nodes"][2]["revised_source_inventory"][-1]["id"]=="field:Enew"
    assert [evidence["nodes"][i]["parent_event_time_comparisons"][0]["status"]
            for i in (0,2,3)]==["one-resolved-pair"]*3
    assert evidence["nodes"][1]["parent_event_time_comparisons"]==[]
    # Detached output does not mutate native records or original model.
    evidence["nodes"][2]["admitted_scenario"]["spec"]["mass_kg"] = 999.
    assert out.entries[2].scenario.spec.mass_kg==2e-5
    assert p.original.as_dict()==original


def test_p2_multiple_structural_generations_retain_retired_source_identity():
    family=model("uniform-edits")
    add=next(c for c in family.cases if c.case_id=="add-E")
    swap=StructuralCase(
        "swap-E", "Replace the new held source by an actual different identity.",
        replace_sources=(SourceReplacement(
            "field:Enew",NamedSource(
                "field:Eplus","Replacement named +x source",
                UniformElectricField((190.,0.,0.)))),))
    reuse=StructuralCase(
        "reuse-E","Attempt to resurrect retired Enew source ID.",
        add_sources=(NamedSource(
            "field:Enew","Illegally resurrected source",
            UniformElectricField((10.,0.,0.))),))
    plan=UnifiedObservationPlan("m32a:ancestry",family.baseline,(
        StructuralWorldNode("add-E","baseline",add),
        StructuralWorldNode("swap-E","add-E",swap),
        StructuralWorldNode("reuse-E","swap-E",reuse),
        physical("tune","swap-E",
                 ("field:Eplus","field_v_m",(180.,0.,0.))),
    ))
    run=run_unified_observation_graph(plan)
    assert [x.status for x in run.entries]==[
        "complete","complete","rejected-structural-model","complete"]
    assert run.entries[1].assembly.parent is run.entries[0].assembly
    assert run.entries[1].scenario.source_annotations[-1][0]=="field:Eplus"
    assert "field:Enew" not in dict(run.entries[1].scenario.source_annotations)
    assert run.entries[2].scenario is None and run.entries[2].native is None
    assert "retired" in run.entries[2].reason or "reuse" in run.entries[2].reason
    assert run.entries[3].scenario.parent_revision_id==run.entries[1].scenario.revision_id
    evidence=run.as_dict()["nodes"]
    assert evidence[2]["admitted_scenario"] is None
    assert evidence[1]["structural_assembly"]["parent_revision_id"]==(
        run.entries[0].scenario.revision_id)
    assert evidence[3]["node"]["edits"][0]["component_id"]=="field:Eplus"
    assert evidence[3]["exact_time_parent_comparisons"]


def test_p3_physical_after_physical_allowed_but_structural_after_physical_refuses():
    family=model("uniform-edits")
    first=physical("mass","baseline",("test-particle","mass_kg",2e-5))
    second=physical("faster","mass",("test-particle","initial_velocity_m_s",(.2,0.,0.)))
    plan=UnifiedObservationPlan("m32a:sequence",family.baseline,(first,second))
    run=run_unified_observation_graph(plan)
    assert [e.status for e in run.entries]==["complete","complete"]
    assert run.entries[0].scenario.spec.mass_kg==2e-5
    assert run.entries[1].scenario.spec.initial_velocity_m_s==(.2,0.,0.)
    assert run.entries[1].scenario.parent_revision_id==run.entries[0].scenario.revision_id
    with pytest.raises(ValueError,match="structural-after-physical"):
        UnifiedObservationPlan("m32a:invalid",family.baseline,(
            first, StructuralWorldNode(
                "add-E","mass",family.cases[0])))
    assert all(x["exact_time_parent_comparisons"] for x in run.as_dict()["nodes"])


def test_p4_joint_translation_is_admissible_in_single_physical_graph_revision():
    family=model("point-replace")
    with pytest.raises(UnsupportedProblem):
        revise_scenario(family.baseline,scenario_id="m32a:bad-probe",
                        note="Only the probe moved away from its physical plane.",
                        particle_changes={"position_m":(0.,.1,.05)})
    with pytest.raises(UnsupportedProblem):
        revise_scenario(family.baseline,scenario_id="m32a:bad-source",
                        note="Only the source moved away from its physical plane.",
                        source_changes={"point:A":{"position_m":(.2,0.,.05)}})
    plan=UnifiedObservationPlan("m32a:translate",family.baseline,(
        physical("both-z","baseline",
                 ("point:A","position_m",(.2,0.,.05)),
                 ("test-particle","position_m",(0.,.1,.05))),
    ))
    run=run_unified_observation_graph(plan)
    node=run.entries[0]
    assert node.status=="complete"
    assert node.scenario.spec.field.sources[0].position_m==(.2,0.,.05)
    assert node.scenario.spec.particle.position_m==(0.,.1,.05)
    assert all(x.position_m[2] == pytest.approx(.05) for x in node.trajectory.samples)
    assert all(a.time_s==b.time_s for a,b in zip(
        run.baseline_native.samples,node.native.samples))
    assert node.scenario.parent_revision_id==family.baseline.revision_id


def test_p5_genuine_refused_parent_event_study_does_not_hide_valid_physical_child():
    old=make_plan("terminal-refusal")
    world=UnifiedObservationPlan(
        "m32a:stop-study",old.family.baseline,(
            StructuralWorldNode(
                "reversed-force","baseline",old.family.cases[0],"events"),
            physical("correct-stop","reversed-force",
                     ("field:Eplus","field_v_m",(500.,0.,0.)),
                     ("test-particle","mass_kg",2e-6),
                     observation="events"),
        ),"events",old.questions,old.policy)
    run=run_unified_observation_graph(world)
    parent,child=run.entries
    assert parent.scenario is not None and parent.native is None
    assert parent.status=="rejected-event-study"
    assert "initial release" in parent.reason
    assert child.status=="complete" and child.native is not None
    assert child.scenario.parent_revision_id==parent.scenario.revision_id
    report=run.as_dict()["nodes"]
    assert report[0]["accepted_parent_inventory"] is not None
    assert report[0]["admitted_scenario"] is not None
    assert report[0]["native_observation"] is None
    assert report[1]["parent_event_time_comparisons"][0]["status"]=="unpaired"


def test_p6_source_guard_and_multiple_real_crossings_never_fake_unique_pair():
    guard=make_plan("guard-detector")
    c=next(x for x in guard.family.cases if x.case_id=="attract-guard")
    p=UnifiedObservationPlan(
        "m32a:guard",guard.family.baseline,
        (StructuralWorldNode("attract-guard","baseline",c,"events"),),
        "events",guard.questions,guard.policy)
    run=run_unified_observation_graph(p)
    row=run.entries[0]
    assert row.status=="censored-source-guard"
    assert row.native.trajectory.termination=="source-guard"
    outcomes=row.native.as_dict()["outcomes"]
    assert outcomes["x:early"]=="located"
    assert outcomes["x:far"]=="search-truncated"
    comp=run.as_dict()["nodes"][0]["parent_event_time_comparisons"]
    assert [x["status"] for x in comp]==["unpaired","unpaired"]
    assert "estimated_numerical_shift_s" not in comp[0]

    double=make_plan("double-crossing")
    dp=UnifiedObservationPlan("m32a:double",double.family.baseline,(
        StructuralWorldNode("extra-retarding","baseline",double.family.cases[0],"events"),
    ),"events",double.questions,double.policy)
    dr=run_unified_observation_graph(dp)
    assert dr.entries[0].status=="complete"
    for native in (dr.baseline_native,dr.entries[0].native):
        roots=[r for r in native.events if r.event_id=="x:two-crossings"]
        assert len(roots)==2 and [r.direction for r in roots]==[1,-1]
    assert dr.as_dict()["nodes"][0]["parent_event_time_comparisons"][0]["status"]=="unpaired"


def test_p7_structural_refusal_does_not_suppress_sibling_and_mixed_mode_unpaired():
    family=model("mirror-pair")
    c={x.case_id:x for x in family.cases}
    q=(PlaneEvent("x:early",(.001,0.,0.),(1.,0.,0.),
                  "Same signed event",direction=1),)
    plan=UnifiedObservationPlan("m32a:refused",family.baseline,(
        StructuralWorldNode("unpaired","baseline",c["unpaired"],"events"),
        physical("absent-child","unpaired",
                 ("test-particle","mass_kg",2e-5),observation="events"),
        StructuralWorldNode("extra-field","baseline",c["extra-field"],"trajectory"),
        physical("observed","extra-field",
                 ("test-particle","mass_kg",2e-5),observation="events"),
    ),"trajectory",q,EventPolicy(.002,1e-8,1e-8,1e-6))
    run=run_unified_observation_graph(plan)
    assert [x.status for x in run.entries]==[
        "rejected-structural-model","skipped-parent-model","complete","complete"]
    assert run.entries[0].scenario is None and run.entries[1].scenario is None
    assert run.entries[2].native is not None and run.entries[3].native is not None
    evidence=run.as_dict()["nodes"]
    assert evidence[0]["revised_source_inventory"] is None
    assert evidence[1]["native_observation"] is None
    assert evidence[3]["parent_event_time_comparisons"][0]["status"]=="unpaired"
    assert "different native" in evidence[3]["parent_event_time_comparisons"][0]["reason"]
    assert evidence[3]["exact_time_parent_comparisons"]==[]


def test_p8_unfamiliar_real_gravity_and_field_joint_event_analytic_reference():
    family=model("uniform-edits")
    gravity=next(c for c in family.cases if c.case_id=="add-gravity")
    detector_y=PlaneEvent("y:down",(0.,-.003,0.),(0.,1.,0.),
                          "Negative y detector",direction=-1)
    plan=UnifiedObservationPlan("m32a:new-y",family.baseline,(
        StructuralWorldNode("add-gravity","baseline",gravity,"events"),
        physical("joint-y","add-gravity",
                 ("gravity","acceleration_m_s2",(0.,-3.3,0.)),
                 ("field:Ey","field_v_m",(0.,1750.,0.)),
                 observation="events"),
    ),"events",(detector_y,),EventPolicy(.001,1e-8,1e-9,1e-7))
    result=run_unified_observation_graph(plan)
    assert [entry.status for entry in result.entries]==["complete","complete"]
    q=family.baseline.spec.particle.charge_c
    for entry,ay in ((result.entries[0], -2 + q*50/1e-5),
                     (result.entries[1], -3.3 + q*1750/1e-5)):
        roots=[x for x in entry.native.events if x.event_id=="y:down"]
        assert len(roots)==1 and roots[0].direction==-1
        assert roots[0].sample.time_s==pytest.approx(sqrt(.006/abs(ay)),abs=1e-7)
    assert result.baseline_native.as_dict()["outcomes"]["y:down"] == (
        "not-found-at-requested-resolution")
    row=result.as_dict()["nodes"][1]
    assert row["parent_event_time_comparisons"][0]["status"]=="one-resolved-pair"
    assert row["admitted_scenario"]["parent_revision_id"]==result.entries[0].scenario.revision_id


def test_negative_typed_graph_bounds_and_unsupported_properties_refuse():
    family=model("uniform-edits")
    good=physical("mass","baseline",("test-particle","mass_kg",2e-5))
    with pytest.raises(ValueError,match="duplicate"):
        UnifiedObservationPlan("m32a:dup",family.baseline,(good,good))
    with pytest.raises(ValueError,match="previous world"):
        UnifiedObservationPlan("m32a:forward",family.baseline,(
            physical("late","not-yet",("test-particle","mass_kg",2e-5)),))
    with pytest.raises(ValueError,match="graph requires"):
        UnifiedObservationPlan("m32a:empty",family.baseline,())
    with pytest.raises(ValueError,match="immutable physical worlds"):
        UnifiedObservationPlan("m32a:wrong",family.baseline,([good]))
    with pytest.raises(TypeError):
        run_unified_observation_graph({"untrusted":"JSON"})
    with pytest.raises(ValueError,match="baseline"):
        physical("baseline","baseline",("test-particle","mass_kg",2e-5))
    with pytest.raises(TypeError):
        CoupledPropertyEdit("test-particle","mass_kg",True)
    with pytest.raises(TypeError):
        StructuralWorldNode("different","baseline",family.cases[0])
    with pytest.raises(ValueError,match="duplicate"):
        PhysicalWorldNode("twice","baseline","duplicate edits",(
            CoupledPropertyEdit("test-particle","mass_kg",2e-5),
            CoupledPropertyEdit("test-particle","mass_kg",3e-5)))
    with pytest.raises(TypeError,match="event worlds require"):
        UnifiedObservationPlan("m32a:noq",family.baseline,(
            physical("mass","baseline",("test-particle","mass_kg",2e-5),
                     observation="events"),))
    with pytest.raises(ValueError,match="duplicate"):
        UnifiedObservationPlan("m32a:two",family.baseline,(good,),
            "events",(detector(),detector()),policy())

    no_op=UnifiedObservationPlan("m32a:noop",family.baseline,(
        physical("unmodified","baseline",("test-particle","mass_kg",1e-5)),
        physical("gauge","baseline",("field:E0","reference_potential_v",1.)),
        physical("absent","baseline",("field:fake","field_v_m",(100.,0.,0.))),
        good,
    ))
    out=run_unified_observation_graph(no_op)
    assert [x.status for x in out.entries]==[
        "rejected-physical-edit"]*3+["complete"]
    assert all(x.scenario is None and x.native is None for x in out.entries[:3])
    assert out.entries[3].scenario is not None


def test_graph_source_and_native_result_forgery_rejected_without_runner_receipt():
    family=model("uniform-edits")
    q=(detector(),)
    plan=UnifiedObservationPlan("m32a:integrity",family.baseline,(
        physical("mass","baseline",("test-particle","mass_kg",2e-5),
                 observation="events"),
    ),"events",q,policy())
    genuine=run_unified_observation_graph(plan)
    node=genuine.entries[0]
    assert node.native is not None
    with pytest.raises(ValueError,match="runner-issued receipt"):
        UnifiedObservationRun(
            plan,genuine.baseline_native,genuine.baseline_status,
            genuine.baseline_reason,genuine.entries)
    forged=replace(node,status="integration-failed",native=None,
                   reason="invented event numerical failure")
    with pytest.raises(ValueError,match="runner-issued receipt"):
        replace(genuine,entries=(forged,))
    root=node.native.events[0]
    bad_event=replace(root, sample=replace(root.sample,time_s=root.sample.time_s+.02),
                      time_bracket_s=tuple(t+.02 for t in root.time_bracket_s))
    with pytest.raises(ValueError,match="runner-issued receipt"):
        replace(genuine,entries=(replace(node,
            native=replace(node.native,events=(bad_event,))),))
    alternate=replace(plan,plan_id="m32a:false-provenance")
    with pytest.raises(ValueError,match="runner-issued receipt"):
        replace(genuine,plan=alternate)
    changed=replace(node.node,edits=(
        replace(node.node.edits[0],value=3e-5),))
    changed_plan=replace(plan,nodes=(changed,))
    wrong_node=replace(node,node=changed)
    with pytest.raises(ValueError,match="runner-issued receipt"):
        replace(genuine,plan=changed_plan,entries=(wrong_node,))
    object.__setattr__(genuine,"entries",(forged,))
    with pytest.raises(ValueError,match="runner-issued receipt"):
        genuine.as_dict()


def test_graph_depth_and_node_count_limits_are_enforced():
    original=model("uniform-edits").baseline
    items=[physical("p"+str(i),"baseline" if i==1 else "p"+str(i-1),
                    ("test-particle","mass_kg",(i+1)*1e-5)) for i in range(1,6)]
    with pytest.raises(ValueError,match="depth exceeds"):
        UnifiedObservationPlan("m32a:deep",original,tuple(items))
    many=tuple(physical("n"+str(i),"baseline",
                        ("test-particle","mass_kg",(i+2)*1e-5))
               for i in range(9))
    with pytest.raises(ValueError,match="1..8"):
        UnifiedObservationPlan("m32a:too-many",original,many)


def test_p9_unfamiliar_tilted_rod_signed_points_gravity_and_new_held_source_transfer():
    """Independent test-side Coulomb quadrature, not production rod field callbacks."""
    from math import pi, hypot
    from conductor_sim import (
        EPSILON_0, FixedPointField, PointCharge, PrescribedField,
        TestChargeMotionSpec, UniformGravity, UniformRodField,
    )
    from conductor_sim.scenarios import PrescribedScenario
    from conductor_sim.structural_scenarios import NamedSource

    start=(.090,-.090,0.)
    end=(.135,.105,0.)
    rod_charge=2.6e-9
    sources=(
        UniformRodField(rod_charge,start,end),
        FixedPointField(1.2e-9,(.18,.13,0.)),
        FixedPointField(-.8e-9,(-.17,.07,0.)),
    )
    original=PrescribedScenario(
        "m32a:unfamiliar-root",
        TestChargeMotionSpec(
            PrescribedField(sources),
            PointCharge(.9e-9,(0.,0.,0.)),
            1e-5,(.048,.012,0.), .42,.0042,
            plane_normal=(0.,0.,1.),source_guard_m=.001,
            relative_tolerance=2e-9,
            gravity=UniformGravity((0.,-.075,0.))),
        (("rod:slanted","Teacher tilted uniform rod"),
         ("point:pos","Held positive point"),
         ("point:neg","Held negative point")),
        "Unfamiliar teacher hypothetical charged rod, two signed point sources and gravity.")
    before=original.as_dict()
    new_point=NamedSource(
        "point:added","New independent held point at lower right",
        FixedPointField(1.2e-9,(.18,-.08,0.)))
    plan=UnifiedObservationPlan("m32a:unknown",original,(
        StructuralWorldNode("new-point","baseline",StructuralCase(
            "new-point","Introduce another real held positive source.",
            add_sources=(new_point,))),
        physical("new-q-mass","new-point",
                 ("point:added","charge_c",2.1e-9),
                 ("test-particle","mass_kg",1.3e-5)),
    ))
    result=run_unified_observation_graph(plan)
    parent,child=result.entries
    assert result.baseline_status==parent.status==child.status=="complete"
    assert child.scenario.parent_revision_id==parent.scenario.revision_id
    assert parent.assembly.parent is original
    assert child.scenario.source_annotations[-1][0]=="point:added"
    assert parent.scenario.spec.field.sources[-1].charge_c==1.2e-9
    assert child.scenario.spec.field.sources[-1].charge_c==2.1e-9
    assert child.scenario.spec.mass_kg==1.3e-5
    assert len(result.as_dict()["nodes"][1]["exact_time_parent_comparisons"])>10

    # Genuine separate 3D SI Coulomb midpoint quadrature on the original
    # uniform charged rod, not UniformRodField._values or production RK.
    k=1/(4*pi*EPSILON_0)
    electric=[0.,0.,0.]
    steps=3200
    for j in range(steps):
        u=(j+.5)/steps
        site=tuple(start[i]+u*(end[i]-start[i]) for i in range(3))
        d=tuple(-site[i] for i in range(3))
        radius=hypot(*d)
        for i in range(3):
            electric[i] += k*rod_charge*d[i]/(steps*radius**3)
    for charge,site in ((1.2e-9,(.18,.13,0.)),
                        (-.8e-9,(-.17,.07,0.)),
                        (2.1e-9,(.18,-.08,0.))):
        d=tuple(-x for x in site)
        radius=hypot(*d)
        for i in range(3):
            electric[i] += k*charge*d[i]/radius**3
    q=.9e-9
    mass=1.3e-5
    direct_force=(q*electric[0], q*electric[1]-mass*.075, q*electric[2])
    measured=child.trajectory.samples[0].force_n
    assert measured==pytest.approx(direct_force,abs=2e-12)
    assert abs(measured[0])+abs(measured[1])>1e-9
    assert all(len(row.position_m)==3 and len(row.velocity_m_s)==3
               for row in child.trajectory.samples)
    assert original.as_dict()==before
