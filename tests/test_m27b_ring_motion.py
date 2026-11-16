"""M27b: qualified real off-axis ring motion, reference ODE and guards.

All inputs are explicit teacher-hypotheticals except the listed workbook
source geometry. The saved phase/time/energy models are not browser physics.
"""
from copy import deepcopy
from dataclasses import replace
from math import cos, fsum, hypot, pi, sin, sqrt
import json

import pytest

from conductor_sim import (EPSILON_0, FixedPointField, PointCharge,
    PrescribedField, TestChargeMotionSpec as MotionSpec,
    UniformElectricField, UniformGravity, UniformRingAxisField,
    UniformRingField, integrate_test_charge)
from conductor_sim.solver import UnsupportedProblem
from conductor_sim.scenarios import PrescribedScenario
from conductor_sim.motion_events import (EventPolicy, EventStudy, PlaneEvent,
    VelocityReversal, investigate_motion, validate_event_run)
from conductor_sim.motion_event_viewer import render_event_run
from conductor_sim.driven_motion import (DrivenModel, DrivenStudy, LinearDrag,
    FieldPhase, UniformFieldSchedule, investigate_driven_motion,
    validate_driven_run)
from conductor_sim.driven_viewer import render_driven_run
from conductor_sim.prescribed_motion import _FixedDomain
from conductor_sim.prescribed_viewer import sample_test_charge_viewer
from conductor_sim.prescribed_viewer_validation import validate_prescribed_viewer
from conductor_sim.source_geometry import source_geometry
from conductor_sim.ring_motion_symmetry import certify_ring_motion_plane
from conductor_sim.structural_scenarios import NamedSource
from conductor_sim.driven_reassembly import reassemble_driven_model
from conductor_sim.ring_field_observer import sample_ring_field_slice
from conductor_sim.visualization import SliceSpec
from examples.m27b_offaxis_motion import author, run_case, POLICY

K=1/(4*pi*EPSILON_0)


def make_spec(*, sources=None, position=(.055,0.,.09), velocity=(.03,0.,.02),
              normal=(0.,1.,0.), charge=8e-10, mass=2e-7,
              duration=.06, interval=.005, gravity=None, guard=.001):
    sources=(UniformRingField(2e-9,.15),) if sources is None else sources
    return MotionSpec(PrescribedField(tuple(sources)),PointCharge(charge,position),
        mass,velocity,duration,interval,plane_normal=normal,
        length_scale_m=.3,relative_tolerance=1e-9,
        source_guard_m=guard,gravity=gravity)


def direct_sites(charge, radius, *, n=384):
    """Fixed independent line-element sample; no Carlson/production E/V."""
    return [(radius*cos(2*pi*(j+.5)/n),radius*sin(2*pi*(j+.5)/n),0.)
            for j in range(n)], K*charge/n


def direct_e(position, points, factor):
    x,y,z=position
    ex=ey=ez=0.
    for sx,sy,sz in points:
        dx,dy,dz=x-sx,y-sy,z-sz
        r=hypot(dx,dy,dz)
        inv=1/(r*r*r)
        ex+=dx*inv;ey+=dy*inv;ez+=dz*inv
    return (factor*ex,factor*ey,factor*ez)


def alternative_verlet(spec, *, n=800):
    """Independent velocity-Verlet using sampled Coulomb-line acceleration."""
    ring=spec.field.sources[0]
    assert type(ring) is UniformRingField and ring.axis==(0.,0.,1.)
    sites,weight=direct_sites(ring.charge_c,ring.radius_m)
    dt=spec.duration_s/n
    position=list(spec.particle.position_m)
    v=list(spec.initial_velocity_m_s)
    qmass=spec.particle.charge_c/spec.mass_kg
    def acceleration(p):
        return tuple(qmass*e for e in direct_e(p,sites,weight))
    a=acceleration(position)
    for _ in range(n):
        new_position=[position[i]+v[i]*dt+.5*a[i]*dt*dt for i in range(3)]
        anew=acceleration(new_position)
        v=[v[i]+.5*(a[i]+anew[i])*dt for i in range(3)]
        position,a=new_position,anew
    return tuple(position),tuple(v)


def test_p1_meridian_oblique_full_field_matches_direct_quadrature_and_verlet():
    spec=make_spec()
    original=spec.field.as_dict()
    assert certify_ring_motion_plane(spec.field.sources[0],
        origin_m=spec.particle.position_m,normal=spec.plane_normal)=="meridian"
    result=integrate_test_charge(spec)
    assert result.termination=="duration"
    assert any(abs(s.position_m[0]-spec.particle.position_m[0])>1e-4 for s in result.samples)
    assert all(abs(s.position_m[1])<2e-11 for s in result.samples)
    assert all(abs(s.force_n[1])<1e-15 for s in result.samples)
    sites,weight=direct_sites(2e-9,.15,n=2048)
    for sample in (result.samples[0],result.samples[len(result.samples)//2],result.samples[-1]):
        expected=direct_e(sample.position_m,sites,weight)
        assert sample.electric_field_v_m==pytest.approx(expected,rel=2e-9,abs=1e-7)
        assert sample.force_n==pytest.approx(tuple(spec.particle.charge_c*x for x in expected),
                                             rel=2e-9,abs=1e-16)
        assert sample.potential_energy_j==pytest.approx(
            spec.particle.charge_c*spec.field.potential_at(sample.position_m),rel=1e-12)
    ref_p,ref_v=alternative_verlet(spec)
    assert result.samples[-1].position_m==pytest.approx(ref_p,abs=3e-6)
    assert result.samples[-1].velocity_m_s==pytest.approx(ref_v,abs=2e-5)
    assert result.max_abs_energy_drift_j<1e-12
    assert original==spec.field.as_dict()


def test_p2_equatorial_is_invariant_and_genuinely_curved():
    spec=make_spec(position=(.05,0.,0.),velocity=(0.,.14,0.),normal=(0.,0.,-1.))
    assert certify_ring_motion_plane(spec.field.sources[0],
        origin_m=spec.particle.position_m,normal=spec.plane_normal)=="equatorial"
    run=integrate_test_charge(spec)
    assert run.termination=="duration"
    assert all(abs(s.position_m[2])<1e-12 and abs(s.velocity_m_s[2])<1e-12 for s in run.samples)
    assert all(abs(s.electric_field_v_m[2])<1e-12 for s in run.samples)
    assert run.samples[-1].position_m[0]!=pytest.approx(.05,abs=2e-8)
    assert abs(run.max_abs_energy_drift_j)<1e-12


def test_p3_w1_real_world_xy_transverse_field_curved_workbook_geometry():
    base,questions=author("workbook-xy")
    spec=base.spec
    assert spec.plane_normal==(0.,0.,-1.)
    assert [x[0] for x in base.source_annotations]==["ring:A","ring:B","field:up"]
    sources=spec.field.sources
    assert sources[0].charge_c==10e-6 and sources[1].charge_c==-20e-6
    assert sources[0].radius_m==sources[1].radius_m==.3
    assert sources[1].center_m==(.4,0.,0.)
    assert sources[2].field_v_m==(0.,100000.,0.)
    assert spec.particle.charge_c==10e-6
    assert spec.mass_kg==.01 and "TEACHER HYPOTHESES" in base.provenance
    assert all(certify_ring_motion_plane(s,origin_m=spec.particle.position_m,
                normal=spec.plane_normal)=="meridian" for s in sources[:2])
    x=.2;a=.3
    axial = K*10e-6*x/(a*a+x*x)**1.5 + K*(-20e-6)*(x-.4)/(a*a+(x-.4)**2)**1.5
    e0=spec.field.electric_field_at(spec.particle.position_m)
    assert e0==pytest.approx((axial,1e5,0.),rel=2e-13)
    r=integrate_test_charge(spec)
    assert r.termination=="duration"
    assert r.samples[-1].position_m[1]>1e-4
    assert r.samples[-1].velocity_m_s[1]>0
    assert r.samples[-1].position_m[0]>.2
    assert max(abs(s.position_m[2]) for s in r.samples)<1e-12
    assert abs(r.max_abs_energy_drift_j)<2e-8
    study=EventStudy(base,questions,POLICY)
    investigated=investigate_motion(study)
    assert investigated.trajectory.termination=="duration"
    assert validate_event_run(investigated)
    geom=source_geometry(spec,base.source_annotations,trusted_scenario=base)
    assert [(e["source_id"],e["kind"]) for e in geom["entries"][:2]]==[
        ("ring:A","UniformRingField"),("ring:B","UniformRingField")]
    assert geom["entries"][0]["motion_symmetry_mode"]=="meridian"
    assert geom["entries"][1]["shape"]=="ring-projection"
    assert geom["u"]==(1.,0.,0.)
    assert geom["v"]==(0.,1.,0.)
    html=render_event_run(investigated,title="M27b physical curved workbook motion")
    assert "ring:A" in html and "ring:B" in html
    assert "full 3D off-axis E/V" in html
    assert 'id="event-data"' in html
    assert "No conductor response" in html


def test_p4_gravity_real_force_energy_kept_separate():
    spec=make_spec(gravity=UniformGravity((.0,0.,-.4)))
    run=integrate_test_charge(spec)
    assert run.termination=="duration"
    for sample in run.samples:
        assert sample.gravitational_force_n==pytest.approx((0.,0.,-.4*spec.mass_kg))
        assert sample.force_n==pytest.approx(tuple(
            sample.electric_force_n[i]+sample.gravitational_force_n[i] for i in range(3)))
        assert sample.potential_energy_j==pytest.approx(
            sample.electric_potential_energy_j+sample.gravitational_potential_energy_j,
            rel=1e-11,abs=1e-17)
    assert run.max_abs_energy_drift_j<1e-12


def test_p5_one_model_two_event_questions_not_scene_edits():
    base,_=author("single-meridian")
    start=base.as_dict()
    questions=(PlaneEvent("z:detection",(0.,0.,.095),(0.,0.,1.),"Observe passage"),
               VelocityReversal("v:x",(1.,0.,0.),"Horizontal speed reversal"))
    study=EventStudy(base,questions,POLICY)
    result=investigate_motion(study)
    assert validate_event_run(result)
    assert result.study.scenario is base
    assert result.trajectory.samples[0].time_s==0.
    assert base.as_dict()==start
    assert set(result.as_dict()["outcomes"])=={"z:detection","v:x"}
    assert all(e.sample.time_s <= result.trajectory.samples[-1].time_s for e in result.events)


def test_p6_driven_phase_validation_switch_work_and_real_viewer():
    base,questions=author("workbook-xy")
    model=DrivenModel(base,LinearDrag(1e-6),UniformFieldSchedule((
        FieldPhase(0.,(0.,0.,0.)),FieldPhase(.008,(0.,2e4,0.)))),
        "m27b:driven-test","Teacher hypothetical switch added to workbook geometry")
    earlier=model.as_dict()
    r=investigate_driven_motion(DrivenStudy(model,questions,POLICY))
    assert r.termination=="duration" and validate_driven_run(r)
    assert [s.time_s for s in r.switches]==[.008]
    sw=r.switches[0]
    assert sw.left.position_m==sw.right.position_m
    assert sw.left.velocity_m_s==sw.right.velocity_m_s
    assert sw.right.acceleration_m_s2[1]-sw.left.acceleration_m_s2[1]==pytest.approx(
        base.spec.particle.charge_c*2e4/base.spec.mass_kg,rel=1e-12)
    assert r.samples[-1].dissipated_energy_j>=0
    assert abs(r.samples[-1].work_balance_error_j)<1e-7
    assert model.as_dict()==earlier
    html=render_driven_run(r,title="M27b driven rings")
    assert "ring:A" in html and "ring:B" in html
    assert "motion_symmetry_mode" in html


def test_p7_existing_M26a_structural_bridge_admits_full_ring_and_retains_ancestry():
    base=PrescribedScenario("m27b:structural-root",make_spec(
        sources=(UniformElectricField((0.,15000.,0.)),),
        position=(.2,0.,0.), velocity=(.01,0.,0.),normal=(0.,0.,-1.),
        charge=10e-6,mass=.01,duration=.015),
        (("field:held","Held uniform E"),),"Teacher hypothetical baseline without rings")
    parent=DrivenModel(base,LinearDrag(0.),
         UniformFieldSchedule((FieldPhase(0.,(0.,0.,0.)),
                               FieldPhase(.009,(0.,2e4,0.)))),
         "m27b:parent","Teacher original scheduled experiment")
    source=NamedSource("ring:added","Physical ring",
                      UniformRingField(10e-6,.3,axis=(1.,0.,0.)))
    a=reassemble_driven_model(parent,model_id="m27b:child",scenario_id="m27b:base-child",
         note="Teacher hypothetical: add one full ring",add_sources=(source,))
    assert a.model.base.source_annotations[-1]==("ring:added","Physical ring")
    assert a.model.applied is parent.applied and a.model.drag is parent.drag
    assert a.parent is parent and a.base_assembly.parent is base
    r=investigate_driven_motion(DrivenStudy(a.model,
         (PlaneEvent("detector",(.23,0.,0.),(1.,0.,0.),"Passive detector"),),POLICY))
    assert validate_driven_run(r)
    with pytest.raises(UnsupportedProblem,match="ring axis|meridian|equatorial|oblique"):
        reassemble_driven_model(parent,model_id="m27b:bad-child",
            scenario_id="m27b:bad-base",note="Teacher invalid oblique new ring",
            add_sources=(NamedSource("ring:oblique","Oblique",
                UniformRingField(1e-9,.1,axis=(1.,1.,1.))),))


def test_u1_rotated_world_frame_preserves_actual_motion_no_projection():
    source=UniformRingField(2e-9,.15)
    original=make_spec(sources=(source,))
    rot=lambda a:(a[2],a[1],-a[0])
    shifted=(.4,-.05,.2)
    def move(a):
        return tuple(rot(a)[i]+shifted[i] for i in range(3))
    moved=make_spec(
        sources=(UniformRingField(2e-9,.15,center_m=shifted,axis=rot(source.axis)),),
        position=move(original.particle.position_m),
        velocity=rot(original.initial_velocity_m_s),
        normal=rot(original.plane_normal))
    a=integrate_test_charge(original)
    b=integrate_test_charge(moved)
    assert a.termination==b.termination=="duration"
    for row, rotated in zip(a.samples,b.samples):
        assert rotated.position_m==pytest.approx(move(row.position_m),abs=2e-8)
        assert rotated.velocity_m_s==pytest.approx(rot(row.velocity_m_s),abs=2e-8)
        assert rotated.force_n==pytest.approx(rot(row.force_n),abs=2e-15)
        assert rotated.total_energy_j==pytest.approx(row.total_energy_j,rel=1e-10,abs=2e-17)


@pytest.mark.parametrize("bad_source,normal",[
    (UniformRingField(1e-9,.15,axis=(1.,1.,1.)),(0.,1.,0.)),
    (UniformRingField(1e-9,.15,center_m=(0.,.003,0.)),(0.,1.,0.)),
])
def test_n1_offplane_or_oblique_ring_refused_without_local_force_shortcut(bad_source,normal):
    with pytest.raises(UnsupportedProblem,match="plane|oblique"):
        make_spec(sources=(bad_source,),normal=normal)


def test_n1_length_scale_may_not_falsify_physical_symmetry():
    source=UniformRingField(1e-9,.1,center_m=(0.,1e-6,0.))
    with pytest.raises(UnsupportedProblem,match="centre"):
        MotionSpec(PrescribedField((source,)),PointCharge(1e-9,(0,0,.2)),
            1e-6,(0.,0.,.1),.1,.01,
            plane_normal=(0.,1.,0.),length_scale_m=1e6)


def test_n2_full_phase_and_gravity_transverse_refusals():
    with pytest.raises(UnsupportedProblem):
        make_spec(gravity=UniformGravity((0.,1.,0.)))
    with pytest.raises(UnsupportedProblem):
        make_spec(sources=(UniformRingField(1e-9,.15),
                           UniformElectricField((0.,1.,0.))))
    base=PrescribedScenario("m27b:phase-refusal",make_spec(),
        (("ring:fixed","Full ring"),),"Teacher phase refusal test")
    with pytest.raises(UnsupportedProblem,match="physical"):
        DrivenModel(base,LinearDrag(1e-8),UniformFieldSchedule(
            (FieldPhase(0.,(0.,0.,0.)),FieldPhase(.02,(0.,1.,0.)))),
            "m27b:unqualified","Teacher invalid later phase")


def test_n3_mixed_axis_only_source_refused_and_legacy_still_solved():
    full=UniformRingField(1e-9,.1)
    legacy=UniformRingAxisField(1e-9,.1)
    with pytest.raises(UnsupportedProblem,match="legacy axis-only"):
        make_spec(sources=(legacy,full))
    old=make_spec(sources=(legacy,),position=(0,0,.2),velocity=(0,0,-.1))
    result=integrate_test_charge(old)
    assert result.termination=="duration"
    assert old.axis==(0.,0.,1.)
    with pytest.raises(UnsupportedProblem,match="symmetry axis"):
        legacy.electric_field_at((.01,0.,.2))


def test_n4_ideal_ring_initial_wire_guard_and_actual_approach_censor():
    with pytest.raises(ValueError,match="initial position"):
        make_spec(position=(.15,0.,0.),velocity=(.0,0.,.1),guard=.002)
    spec=make_spec(
        sources=(UniformRingField(1e-10,.1),),
        position=(.065,0.,0.),velocity=(.35,0.,0.),
        charge=-1e-9,mass=1e-6, duration=.18, interval=.01,
        normal=(0.,0.,-1.),guard=.002)
    r=integrate_test_charge(spec)
    assert r.termination=="source-guard"
    assert r.events and r.events[0].kind=="source-guard"
    assert r.samples[-1].time_s < spec.duration_s
    assert r.samples[-1].position_m[0] < .1
    assert r.samples[-1].position_m[0] > .065
    assert all(spec.field.sources[0].clearance_at(s.position_m) > spec.source_guard_m
               for s in r.samples)


def test_n5_conservative_full_chord_wire_distance_lower_bound():
    spec=make_spec(
        sources=(UniformRingField(1e-9,.15),),
        position=(0.,0.,0.),velocity=(.1,0,0),normal=(0.,0.,-1.))
    domain=_FixedDomain(spec)
    a,b=(-.20,0.,0.),(.20,0.,0.)
    assert domain.clearance(a)>0 and domain.clearance(b)>0
    assert domain.charge_segment_clearance(a,b)<=0  # chord crosses two rim points
    assert domain.charge_segment_clearance((0.,0.,0.),(.001,0.,0.))>0
    # A high-speed segment crosses even if its endpoint samples are safe.
    assert domain.charge_segment_clearance((.05,0.,0.),(.25,0.,0.))<=0


def test_n6_source_geometry_trust_binding_and_readback():
    base,questions=author("workbook-xy")
    g=source_geometry(base.spec,base.source_annotations,trusted_scenario=base)
    assert g["entries"][0]["motion_symmetry_mode"]=="meridian"
    assert g["entries"][0]["charge_c"]==10e-6
    assert g["entries"][1]["charge_c"]==-20e-6
    wrong=PrescribedScenario("m27b:other",replace(base.spec),base.source_annotations,
        "Unrelated author record")
    with pytest.raises(ValueError,match="exact retained scenario"):
        source_geometry(base.spec,base.source_annotations,trusted_scenario=wrong)
    assert source_geometry(base.spec,base.source_annotations)["entries"]==g["entries"]


def test_n7_old_m19b_renderer_explicitly_refuses_new_full_ring():
    spec=make_spec()
    with pytest.raises(UnsupportedProblem,match="M27b event/driven viewer"):
        sample_test_charge_viewer(spec,visual_frames=3)
    # The separate static M27a world-slice observer remains permitted.
    plane=SliceSpec(origin_m=(0,0,0),u=(1,0,0),v=(0,0,1),
                    x_range_m=(-.25,.25),y_range_m=(-.25,.25),resolution=13)
    r=sample_ring_field_slice(spec.field,plane,
        source_annotations=(("ring:static","Static full ring"),),
        exclusion_width_m=.005)
    assert r["motion_qualified"] is False
    assert r["time_evolution"] is False


def test_existing_motion_viewer_data_forgery_cannot_change_physics():
    base,queries=author("single-meridian")
    result=investigate_motion(EventStudy(base,queries,POLICY))
    assert validate_event_run(result)
    saved=deepcopy(result.as_dict())
    assert saved["study"]["model"]["spec"]["field"]["sources"][0]["kind"]=="UniformRingField"
    saved["trajectory"]["samples"][0]["electric_field_v_m"][0]+=1.
    assert saved["trajectory"]["samples"][0]["electric_field_v_m"]!=result.as_dict()["trajectory"]["samples"][0]["electric_field_v_m"]
    # Detached JSON is not a trusted import format; no mutation reaches the
    # independently validated typed result.
    assert validate_event_run(result)


def test_unsupported_equatorial_full_ring_with_wrong_phase_refuses():
    spec=make_spec(position=(.05,0.,0.),velocity=(0.,.14,0.),
                   normal=(0.,0.,1.))
    base=PrescribedScenario("m27b:equatorial",spec,
           (("ring:equatorial","Full ring"),),"Teacher true equatorial launch")
    with pytest.raises(UnsupportedProblem):
        DrivenModel(base,LinearDrag(0.),UniformFieldSchedule(
            (FieldPhase(0.,(0.,.3,0.)),FieldPhase(.02,(0.,0.,2.)))),
            "m27b:wrong-phase","Teacher bad out-of-plane phase")
