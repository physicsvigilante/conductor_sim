"""M37b: source-faithful prescribed-speed current, no fake free dynamics."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from math import hypot, pi

import pytest

from conductor_sim import (
    EPSILON_0, ElectrostaticProblem, FixedNetCharge, Grounded, PointCharge,
    SolidSphere, SphericalShell, PrescribedSphereRadialSpec,
    sample_prescribed_sphere_radial,
)
from conductor_sim.solver import UnsupportedProblem
from tools.m37b_source_approach import SOURCE, make_example, run_all
from tools.m37b_source_reference import check as independent_si_check


def scenario(q=1e-9,v=-.03,centre=(0.,0.,0.),axis=(0.,0.,1.),a=.1,d=.25,
             eps=EPSILON_0,duration=.8):
    initial=tuple(centre[i]+d*axis[i] for i in range(3))
    problem=ElectrostaticProblem(
        charges=(PointCharge(q,initial),),
        conductors=(SolidSphere(a,Grounded(),center_m=centre,name="Sphere A"),),
        permittivity_f_per_m=eps,
    )
    return PrescribedSphereRadialSpec(problem,v,duration,7,1e-5)


@pytest.mark.parametrize("q,v",[
    (+2e-9,-.035), (+2e-9,+.035), (-2e-9,-.035), (-2e-9,+.035),
])
def test_native_signed_constant_speed_can_be_authored_and_preserves_induction(q,v):
    spec=scenario(q,v)
    trace=sample_prescribed_sphere_radial(spec)
    assert trace.spec is spec
    assert len(trace.samples)==7
    k=1/(4*pi*EPSILON_0)
    for i,row in enumerate(trace.samples):
        t=spec.duration_s*i/6
        d=.25+v*t
        current=-q*.1*v/d**2
        f_radial=-k*q*q*.1*d/(d*d-.1**2)**2
        induced_u=-k*q*q*.1/(2*(d*d-.1**2))
        assert row.time_s==pytest.approx(t,rel=2e-14,abs=1e-20)
        assert row.radial_distance_m==pytest.approx(d,rel=3e-14)
        assert row.induced_sphere_net_charge_c==pytest.approx(-q*.1/d,rel=2e-13)
        assert row.wire_current_sphere_to_ground_a==pytest.approx(current,rel=2e-13)
        assert row.wire_current_ground_to_sphere_a==pytest.approx(-current,rel=2e-13)
        assert row.induced_force_on_source_n[2]==pytest.approx(f_radial,rel=2e-13)
        assert row.induced_mechanical_energy_j==pytest.approx(induced_u,rel=2e-13)
        assert row.source_velocity_m_s==(0.,0.,v)
    assert trace.as_dict()["actual_time_origin"].endswith("NOT M16a integration")


def test_translated_oblique_world_unchanged_signed_radial_response():
    from math import sqrt
    axis=(2/7,3/7,6/7)
    c=(.75,-.5,.2)
    p=scenario(q=-1.3e-9,v=-.02,centre=c,axis=axis,eps=3*EPSILON_0)
    result=sample_prescribed_sphere_radial(p)
    assert result.samples[0].source_position_m == p.problem.charges[0].position_m
    for row in result.samples:
        n=tuple((row.source_position_m[i]-c[i])/row.radial_distance_m for i in range(3))
        assert n==pytest.approx(axis,abs=2e-14)
        assert row.source_velocity_m_s==pytest.approx(tuple(-.02*x for x in axis))
        assert row.wire_current_sphere_to_ground_a < 0  # q<0 moves inward
    standard=scenario(q=-1.3e-9,v=-.02,eps=EPSILON_0)
    old=sample_prescribed_sphere_radial(standard)
    assert result.samples[0].wire_current_sphere_to_ground_a == pytest.approx(old.samples[0].wire_current_sphere_to_ground_a)
    assert result.samples[0].induced_mechanical_energy_j == pytest.approx(old.samples[0].induced_mechanical_energy_j/3)
    assert result.samples[0].induced_force_on_source_n == pytest.approx(
        tuple(axis[i]*old.samples[0].induced_force_on_source_n[2]/3 for i in range(3)),
        rel=2e-12,abs=1e-24,
    )


def test_source_original_constant_speed_is_not_relabelled_newtonian():
    spec=make_example()
    assert spec.radial_speed_m_s == -.04
    assert spec.duration_s == 3.
    assert spec.sample_count == 31
    result=sample_prescribed_sphere_radial(spec)
    assert len(result.samples)==31
    assert result.samples[0].radial_distance_m==pytest.approx(.3)
    assert result.samples[-1].radial_distance_m==pytest.approx(.18)
    assert result.samples[-1].time_s==pytest.approx(3.)
    assert all(y.wire_current_sphere_to_ground_a>x.wire_current_sphere_to_ground_a
               for x,y in zip(result.samples,result.samples[1:]))
    assert all(s.wire_current_sphere_to_ground_a>0 for s in result.samples)
    assert SOURCE["original_correct_option_indices"] == [1,3]
    assert SOURCE["original_question_apparatus_count"] == 1


@pytest.mark.parametrize("key,bad",[
    ("radial_speed_m_s",True),("radial_speed_m_s",0),
    ("radial_speed_m_s",float("inf")),("radial_speed_m_s",3e7),
    ("duration_s",True),("duration_s",0),("duration_s",float("nan")),
    ("sample_count",True),("sample_count",2),("sample_count",202),
    ("surface_guard_m",True),("surface_guard_m",-1),("surface_guard_m",float("nan")),
])
def test_unqualified_authored_protocol_inputs_refused(key,bad):
    spec=make_example()
    with pytest.raises((ValueError,UnsupportedProblem)):
        replace(spec,**{key:bad})


def test_invalid_physical_conductor_and_metal_crossings_refused():
    original=make_example()
    problem=original.problem
    source=problem.charges[0]
    bad=(
        replace(problem,charges=(PointCharge(0,source.position_m),)),
        replace(problem,charges=(PointCharge(True,source.position_m),)),
        replace(problem,permittivity_f_per_m=True),
        replace(problem,conductors=(SolidSphere(True,Grounded(),name="Sphere A"),)),
        replace(problem,conductors=(SolidSphere(.1,FixedNetCharge(0.),name="Sphere A"),)),
        replace(problem,conductors=(SphericalShell(.08,.1,Grounded(),name="Sphere A"),)),
        replace(problem,charges=(source,PointCharge(-1e-9,(0,0,.31)))),
        replace(problem,conductors=(problem.conductors[0],
                                    SolidSphere(.05,Grounded(),center_m=(0,0,1.),name="Sphere B"))),
        replace(problem,charges=(PointCharge(1e-9,(0.,0.,.05)),)),
    )
    for p in bad:
        with pytest.raises((ValueError,UnsupportedProblem)):
            PrescribedSphereRadialSpec(p,-.04,1.,7)
    with pytest.raises(UnsupportedProblem,match="guard"):
        replace(original,radial_speed_m_s=-.10,duration_s=3.)
    with pytest.raises(UnsupportedProblem,match="guard"):
        replace(original,surface_guard_m=.3)


@pytest.fixture(scope="module")
def genuine(tmp_path_factory):
    dest=tmp_path_factory.mktemp("m37b-source-p8")
    manifest=run_all(dest)
    return dest,manifest


def test_genuine_original_source_example_has_independent_si_and_real_offline_html(genuine):
    dest,m=genuine
    assert independent_si_check(m)=={
        "physical_source_apparatus":1,
        "teacher_numeric_examples":1,
        "verified_python_kelvin_states":31,
        "original_answer_options":[1,3],
        "ground_wire_current_increases":True,
        "kinematic_protocol":"teacher_prescribed_constant_speed_not_M16a",
    }
    from json import loads
    assert loads((dest/"p8-ground-current.json").read_text(encoding="utf-8"))==m
    html=(dest/"p8-ground-current.html").read_text(encoding="utf-8")
    assert '<svg id="m37b-scene"' in html
    assert '<svg id="m37b-current-chart"' in html
    assert '<script id="m37b-data" type="application/json">' in html
    assert 'id="m37b-nojs"' in html
    assert "ALL numbers" in html
    assert "SOURCE-GIVEN" in html
    assert "NOT free Newtonian motion" in html
    assert "window.m37bPrescribedViewer" in html
    assert "<script src=" not in html and "http://" not in html


@pytest.mark.parametrize("loc,bad",[
    (("trace","source_charge_c"),True),
    (("trace","source_charge_c"),-1e-9),
    (("trace","sphere_radius_m"),.2),
    (("trace","sphere_constraint"),"FixedNetCharge"),
    (("trace","sphere_is_solid"),False),
    (("trace","radial_axis",2),-1.),
    (("trace","signed_constant_radial_speed_m_s"),+.04),
    (("trace","duration_s"),2.),
    (("trace","sample_count"),30),
    (("trace","surface_guard_m"),.3),
    (("trace","units","current"),"mA"),
    (("trace","actual_time_origin"),"M16a Newtonian orbit"),
    (("source","physical_pdf_page"),12),
    (("source","original_correct_option_indices"),[2,4]),
    (("source","original_question_apparatus_count"),3),
    (("source","numeric_attribution"),"q=1nC was printed"),
    (("readouts",5,"wire_current_label"),"0.5 A"),
    (("readouts",5,"charge_x_px"),999.),
    (("layout","sphere_radius_px"),5.),
    (("trace","samples",2,"time_s"),False),
    (("trace","samples",3,"time_s"),1.),
    (("trace","samples",1,"radial_distance_m"),0.),
    (("trace","samples",4,"wire_current_sphere_to_ground_a"),-1e-8),
    (("trace","samples",4,"wire_current_ground_to_sphere_a"),+1e-8),
    (("trace","samples",4,"induced_sphere_net_charge_c"),1e-9),
    (("trace","samples",4,"source_velocity_m_s",2),0.1),
    (("trace","samples",4,"induced_force_on_source_n",2),2.),
    (("trace","samples",4,"induced_mechanical_energy_j"),0.),
])
def test_independent_source_specific_refuses_forged_physics_and_provenance(genuine,loc,bad):
    _dest,manifest=genuine
    forged=deepcopy(manifest)
    p=forged
    for key in loc[:-1]:
        p=p[key]
    p[loc[-1]]=bad
    with pytest.raises(ValueError):
        independent_si_check(forged)


def test_unchanged_source_fixture_is_teacher_not_fabricated_original_numerics(genuine):
    _,m=genuine
    from conductor_sim.prescribed_wire_viewer import viewer_manifest
    assert m["source"]["role"]=="original_workbook_teacher_example"
    assert m["source"]["source_kind"].startswith("workbook")
    assert m["source"]["teacher_hypotheticals_counted_as_original"]==0
    assert "TEACHER-CHOSEN" in m["source"]["numeric_attribution"]
    assert "constant" in m["source"]["statement"]



def test_viewer_refuses_constructed_forged_physics_even_with_valid_dataclass_types():
    from conductor_sim.prescribed_ground_wire import PrescribedSphereTrace
    from conductor_sim.prescribed_wire_viewer import viewer_manifest
    valid = sample_prescribed_sphere_radial(make_example())
    forged = replace(
        valid,
        samples=(replace(valid.samples[0],
                         wire_current_sphere_to_ground_a=-1e-8),*valid.samples[1:]),
    )
    assert type(forged) is PrescribedSphereTrace
    with pytest.raises(ValueError, match="viewer refuses altered prescribed Kelvin"):
        viewer_manifest(forged,SOURCE)
    assert viewer_manifest(valid,SOURCE)["source"]["original_correct_option_indices"]==[1,3]



def test_m37b_refuses_boolean_si_geometry_at_public_authoring_boundary():
    """Provenance safety is required even for permissive legacy geometry types."""
    from conductor_sim import PrescribedSphereRadialSpec
    for position in ((True, 0., .25), (.0, False, .25)):
        with pytest.raises((ValueError, TypeError)):
            p=ElectrostaticProblem(
                (PointCharge(1e-9,position),),
                (SolidSphere(.1,Grounded(),name="Sphere A"),),
            )
            PrescribedSphereRadialSpec(p,-.03,.5,7)
    for centre in ((True,0.,0.), (0.,False,0.)):
        with pytest.raises((ValueError, TypeError)):
            p=ElectrostaticProblem(
                (PointCharge(1e-9,(0.,0.,.25)),),
                (SolidSphere(.1,Grounded(),center_m=centre,name="Sphere A"),),
            )
            PrescribedSphereRadialSpec(p,-.03,.5,7)


def test_original_workbook_role_cannot_be_assigned_to_teacher_negative_or_outward_world():
    from conductor_sim.prescribed_wire_viewer import viewer_manifest
    for q,v in ((-1e-9,-.02),(+1e-9,+.02),(-1e-9,+.02)):
        hypothesis=sample_prescribed_sphere_radial(scenario(q=q,v=v))
        with pytest.raises(ValueError,match="original workbook q>0 inward"):
            viewer_manifest(hypothesis,SOURCE)
        teacher={"role":"teacher_hypothetical",
                 "statement":"Teacher-authored changed sign or outward radial constant speed",
                 "numeric_attribution":"All values TEACHER-CHOSEN; NONE is a printed numerical given."}
        result=viewer_manifest(hypothesis,teacher)
        assert result["source"]["role"]=="teacher_hypothetical"
    original=sample_prescribed_sphere_radial(make_example())
    assert viewer_manifest(original,SOURCE)["source"]["role"]=="original_workbook_teacher_example"


@pytest.mark.parametrize("field,value",[
    ("original_correct_option_indices",[True,3]),
    ("original_question_apparatus_count",True),
    ("teacher_hypotheticals_counted_as_original",False),
    ("source_file","invented.pdf"),
    ("statement","q approaches a sphere with invented current"),
])
def test_public_viewer_refuses_falsified_printed_source_metadata(field,value):
    from conductor_sim.prescribed_wire_viewer import viewer_manifest
    source=deepcopy(SOURCE)
    source[field]=value
    with pytest.raises(ValueError,match="original workbook identity"):
        viewer_manifest(sample_prescribed_sphere_radial(make_example()),source)


@pytest.mark.parametrize("field,value",[
    ("original_correct_option_indices",[True,3]),
    ("original_question_apparatus_count",True),
    ("teacher_hypotheticals_counted_as_original",False),
])
def test_independent_stdlib_reference_rejects_bool_for_int_source_metadata(genuine,field,value):
    m=deepcopy(genuine[1])
    m["source"][field]=value
    with pytest.raises(ValueError):
        independent_si_check(m)


def test_independent_si_checker_rejects_unknown_scientific_claim_fields(genuine):
    original=genuine[1]
    for parent,key,value in (
        ((), "claims_full_maxwell_simulation",True),
        (("source",),"claims_printed_numeric",True),
        (("trace",),"claims_genuine_newtonian_orbit",True),
        (("layout",),"real_current_density_wires",True),
        (("readouts",0),"electron_drift_m_s",1e7),
    ):
        mutated=deepcopy(original)
        cursor=mutated
        for step in parent:
            cursor=cursor[step]
        cursor[key]=value
        with pytest.raises(ValueError):
            independent_si_check(mutated)
