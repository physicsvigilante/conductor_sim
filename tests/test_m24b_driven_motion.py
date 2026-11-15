"""Preselected M24b physical, protocol, event and preservation checks.

Reserved R is deliberately absent: run only after the candidate is frozen.
"""
from dataclasses import replace
from math import exp, expm1, fsum, hypot, isclose, sin, cos
import json
import pytest
from conductor_sim import (PrescribedField, UniformElectricField, UniformGravity,
    UniformRingAxisField, FixedPointField, PointCharge, TestChargeMotionSpec as Spec,
    integrate_test_charge, UnsupportedProblem)
from conductor_sim.scenarios import PrescribedScenario, revise_scenario
from conductor_sim.motion_1d import MotionIntegrationError
from conductor_sim.driven_motion import (LinearDrag, FieldPhase, UniformFieldSchedule,
    DrivenModel, DrivenStudy, PlaneEvent, VelocityReversal, EventPolicy,
    investigate_driven_motion, revise_driven_model, validate_driven_run)


def model(*, b=2e-9, phases=None, sources=None, gravity=(0,0,-.3),
          position=(.1,0,.3), velocity=(.3,0,.2), duration=1.,interval=.1, steps=100000):
    sources=(UniformElectricField((.4,0,-.2)),) if sources is None else sources
    base=PrescribedScenario('base:test',Spec(PrescribedField(sources),PointCharge(1e-9,position),
        1e-9,velocity,duration,interval,plane_normal=(0,1,0),relative_tolerance=1e-9,
        gravity=UniformGravity(gravity),max_internal_steps=steps),
        tuple((f'fixed:{i}',f'Fixed source {i}') for i in range(len(sources))),'Teacher hypothetical')
    return DrivenModel(base,LinearDrag(b),UniformFieldSchedule(phases or (FieldPhase(0,(0,0,0)),)),
        'driven:test','Teacher hypothetical')


def policy():return EventPolicy(.025,1e-9,1e-10,1e-9)


def integrate(m,qs=()):return investigate_driven_motion(DrivenStudy(m,qs,policy()))


def exponential(r,v,a,rate,t):
    if rate==0:
        return tuple(r[i]+v[i]*t+.5*a[i]*t*t for i in range(3)),tuple(v[i]+a[i]*t for i in range(3))
    vinf=tuple(x/rate for x in a);decay=exp(-rate*t)
    return (tuple(r[i]+vinf[i]*t+(v[i]-vinf[i])*(-expm1(-rate*t))/rate for i in range(3)),
            tuple(vinf[i]+(v[i]-vinf[i])*decay for i in range(3)))


def dissipation(v,a,b,m,t):
    if b==0:return 0.
    rate=b/m;vinf=tuple(x/rate for x in a);dv=tuple(x-y for x,y in zip(v,vinf))
    return b*(sum(x*x for x in vinf)*t+2*sum(x*y for x,y in zip(vinf,dv))*(-expm1(-rate*t))/rate+
              sum(x*x for x in dv)*(-expm1(-2*rate*t))/(2*rate))


@pytest.mark.parametrize('b',[0.,.2e-9,2e-9])
def test_constant_force_exact_vector_exponential_and_independent_work(b):
    m=model(b=b);run=integrate(m);a=(.4,0,-.5);mass=m.spec.mass_kg
    for row in run.samples:
        r,v=exponential(m.spec.particle.position_m,m.spec.initial_velocity_m_s,a,b/mass,row.time_s)
        assert row.position_m==pytest.approx(r,abs=2e-9)
        assert row.velocity_m_s==pytest.approx(v,abs=3e-9)
        delta=tuple(x-y for x,y in zip(r,m.spec.particle.position_m))
        assert row.fixed_work_j==pytest.approx(mass*(.4*delta[0]-.2*delta[2]),abs=3e-18)
        assert row.gravitational_work_j==pytest.approx(-mass*.3*delta[2],abs=2e-18)
        assert row.dissipated_energy_j==pytest.approx(dissipation(m.spec.initial_velocity_m_s,a,b,mass,row.time_s),abs=3e-18)
        assert row.applied_work_j==0
        assert abs(row.work_balance_error_j)<2e-17
    assert run.termination=='duration' and validate_driven_run(run)


def switched_model():
    return model(b=1e-9,sources=(),gravity=(0,0,-.2),position=(0,0,.4),velocity=(.4,0,.15),duration=1.5,
        phases=(FieldPhase(0,(.2,0,0)),FieldPhase(.37,(-1.2,0,.5))))


def switch_reference(t):
    r=(0,0,.4);v=(.4,0,.15)
    r,v=exponential(r,v,(.2,0,-.2),1.,min(t,.37))
    return exponential(r,v,(-1.2,0,.3),1.,t-.37) if t>.37 else (r,v)


def test_switch_exact_clock_continuity_and_left_right_acceleration():
    run=integrate(switched_model());assert len(run.switches)==1
    s=run.switches[0];assert s.time_s==.37
    assert s.left.position_m==s.right.position_m and s.left.velocity_m_s==s.right.velocity_m_s
    assert s.left.applied_work_j==s.right.applied_work_j
    assert tuple(b-a for a,b in zip(s.left.acceleration_m_s2,s.right.acceleration_m_s2))==pytest.approx((-1.4,0,.5))
    assert s.right in run.samples
    for row in run.samples:
        r,v=switch_reference(row.time_s)
        assert row.position_m==pytest.approx(r,abs=3e-9)
        assert row.velocity_m_s==pytest.approx(v,abs=3e-9)
        assert row.phase_index==(0 if row.time_s<.37 else 1)


def bisect_reference(fun,lo,hi):
    flo=fun(lo)
    for _ in range(80):
        mid=(lo+hi)/2
        if (fun(mid)>0)==(flo>0):lo=mid
        else:hi=mid
    return (lo+hi)/2


def test_same_model_multiple_questions_and_event_work():
    m=switched_model();snapshot=m.as_dict()
    qs=(VelocityReversal('turn',(1,0,0),'Horizontal reversal'),
        PlaneEvent('detector',(0,0,.5),(0,0,1),'Upper detector',direction=1,terminal=True))
    run=integrate(m,qs);one=integrate(m,qs[:1])
    assert run.study.model is one.study.model is m and m.as_dict()==snapshot
    assert run.termination=='terminal-event' and len(run.events)==2
    expected=(bisect_reference(lambda t:switch_reference(t)[1][0],.37,1.),
              bisect_reference(lambda t:switch_reference(t)[0][2]-.5,.37,1.5))
    for e,t in zip(run.events,expected):
        assert e.sample.time_s==pytest.approx(t,abs=3e-8)
        r,v=switch_reference(t)
        assert e.sample.position_m==pytest.approx(r,abs=2e-9)
        assert not e.full_stop_within_tolerance
        assert abs(e.sample.work_balance_error_j)<3e-18
    assert one.termination=='duration'
    assert one.events[0].sample.time_s==pytest.approx(run.events[0].sample.time_s,abs=1e-9)


def test_zero_drag_and_constant_input_legacy_limit():
    m=model(b=0);old=integrate_test_charge(m.spec);new=integrate(m)
    for a,b in zip(old.samples,new.samples):
        assert a.time_s==b.time_s
        assert a.position_m==pytest.approx(b.position_m,abs=1e-12)
        assert a.velocity_m_s==pytest.approx(b.velocity_m_s,abs=1e-12)
        assert b.dissipated_energy_j==0
    assert 'energy_drift_j' not in new.samples[-1].as_dict()


@pytest.mark.parametrize('bad',[-1.,float('nan'),float('inf'),True,'1'])
def test_invalid_drag_refused(bad):
    with pytest.raises((ValueError,TypeError)):LinearDrag(bad)


@pytest.mark.parametrize('phases',[
    (), (FieldPhase(.1,(0,0,0)),),
    (FieldPhase(0,(0,0,0)),FieldPhase(0,(1,0,0))),
    (FieldPhase(0,(0,0,0)),FieldPhase(.5,(1,0,0)),FieldPhase(.4,(0,0,0))),
    (False,),
])
def test_invalid_schedule_structure_refused(phases):
    with pytest.raises((ValueError,TypeError)):UniformFieldSchedule(phases)


@pytest.mark.parametrize('time,field',[(True,(0,0,0)),(-1,(0,0,0)),(float('inf'),(0,0,0)),(0,(0,1)),(0,(0,True,0)),(0,(0,float('nan'),0))])
def test_invalid_phase_values(time,field):
    with pytest.raises((ValueError,TypeError)):FieldPhase(time,field)


def test_every_phase_is_qualified_not_only_initial():
    with pytest.raises(UnsupportedProblem):
        model(phases=(FieldPhase(0,(0,0,0)),FieldPhase(.3,(0,1,0))))
    ring=UniformRingAxisField(1e-9,.2,axis=(1,0,0))
    with pytest.raises(UnsupportedProblem):
        model(sources=(ring,),gravity=(0,0,0),position=(.1,0,0),velocity=(.2,0,0),
              phases=(FieldPhase(0,(1,0,0)),FieldPhase(.3,(0,0,1))))
    good=model(sources=(ring,),gravity=(0,0,0),position=(.1,0,0),velocity=(.02,0,0),duration=.1,interval=.01,
               phases=(FieldPhase(0,(.1,0,0)),))
    assert integrate(good).termination=='duration'


def test_unresolved_clock_and_identity_aliases_refused():
    with pytest.raises(ValueError,match='clock'):model(phases=(FieldPhase(0,(0,0,0)),FieldPhase(1e-16,(.1,0,0))))
    m=model()
    with pytest.raises(ValueError,match='identities'):replace(m,applied=replace(m.applied,source_id='fixed:0'))
    with pytest.raises(ValueError,match='identities'):replace(m,drag=replace(m.drag,component_id='field:scheduled'))


def test_no_fabricated_finite_stop_in_balanced_relaxation():
    m=model(b=1e-9,sources=(),gravity=(0,0,-.2),velocity=(0,0,-.1),duration=10.,interval=.5,
             phases=(FieldPhase(0,(0,0,.2)),))
    run=integrate(m,(VelocityReversal('stop',(0,0,1),'Potential reversal'),))
    assert not run.events
    assert all(s.velocity_m_s[2]<0 for s in run.samples)
    assert run.samples[-1].velocity_m_s[2]==pytest.approx(-.1*exp(-10),rel=2e-6)
    assert run.as_dict()['outcomes']['stop']=='not-found-at-requested-resolution'
    assert run.samples[-1].dissipated_energy_j>0


def test_equilibrium_initial_rest_not_a_later_stop():
    m=model(b=1e-9,sources=(),gravity=(0,0,0),velocity=(0,0,0))
    run=integrate(m,(VelocityReversal('rest',(1,0,0),'Rest'),))
    assert ('rest','initial-rest') in run.initial_conditions and not run.events
    assert all(s.velocity_m_s==(0,0,0) and s.dissipated_energy_j==0 for s in run.samples)


def test_event_at_switch_does_not_claim_false_velocity_reversal():
    m=model(b=0,sources=(),gravity=(0,0,0),velocity=(.2,0,0),
        phases=(FieldPhase(0,(-1,0,0)),FieldPhase(.2,(1,0,0))))
    run=integrate(m,(VelocityReversal('touch',(1,0,0),'Touches zero at switch',terminal=True),))
    assert run.termination=='event-unresolved' and not run.events
    assert any('switch' in u[3] for u in run.unresolved)
    assert run.samples[-1].time_s<.2


def test_initial_stop_region_checks_actual_scheduled_force():
    m=model(b=1e-9,sources=(),gravity=(0,0,-1),position=(0,0,0),velocity=(0,0,0),
        phases=(FieldPhase(0,(0,0,2)),))
    lower=PlaneEvent('lower',(0,0,0),(0,0,1),'Lower stop',direction=-1,terminal=True,role='ideal-stop-surface')
    run=integrate(m,(lower,));assert not run.events and run.samples[-1].position_m[2]>0
    with pytest.raises(UnsupportedProblem):integrate(m,(replace(lower,direction=1),))


def test_source_guard_is_not_a_field_switch_or_impact():
    m=model(b=1e-9,sources=(FixedPointField(-1e-9,(0,0,0)),),gravity=(0,0,0),
            position=(.03,0,0),velocity=(-.1,0,0),duration=.1,interval=.01)
    run=integrate(m,(PlaneEvent('beyond',(0,0,0),(1,0,0),'Beyond singularity',direction=-1,terminal=True),))
    assert run.termination=='source-guard' and not run.events
    assert run.as_dict()['outcomes']['beyond']=='search-truncated'
    assert run.samples[-1].position_m[0]>m.spec.source_guard_m


def test_stiff_or_insufficient_integration_budget_fails_explicitly():
    m=model(b=1.,steps=10)
    with pytest.raises(MotionIntegrationError):integrate(m)


def test_revisions_are_physical_immutable_and_requalified():
    m=switched_model();before=m.as_dict()
    child,record=revise_driven_model(m,model_id='child:drag',note='Double drag',drag=replace(m.drag,coefficient_kg_s=2e-9))
    assert m.as_dict()==before and record['parent']==before
    assert child.parent_revision_id==m.revision_id
    assert integrate(child).samples[-1].velocity_m_s!=integrate(m).samples[-1].velocity_m_s
    base=revise_scenario(m.base,scenario_id='base:launch',note='New actual launch',motion_changes={'initial_velocity_m_s':(.2,0,.1)})
    launch,log=revise_driven_model(m,model_id='child:launch',note='Revise retained initial state',base=base)
    assert launch.base is base and launch.applied is m.applied
    with pytest.raises(ValueError):revise_driven_model(m,model_id='noop',note='No change')
    with pytest.raises(UnsupportedProblem):revise_driven_model(m,model_id='bad',note='Bad direction',
        applied=replace(m.applied,phases=(FieldPhase(0,(0,2,0)),)))


@pytest.fixture(scope='module')
def saved_run():return integrate(switched_model(),(VelocityReversal('x',(1,0,0),'X reversal'),))


@pytest.mark.parametrize('changes',[
    {'force_n':()}, {'force_n':(0,0,0)}, {'phase_index':False}, {'dissipated_energy_j':-1},
    {'work_balance_error_j':1}, {'kinetic_energy_j':float('nan')},
    {'electric_field_v_m':(0,0,0)}, {'drag_force_n':(0,0,0)},
])
def test_saved_data_tampering_rejected(saved_run,changes):
    rows=list(saved_run.samples);rows[-1]=replace(rows[-1],**changes)
    with pytest.raises((ValueError,TypeError)):validate_driven_run(replace(saved_run,samples=tuple(rows)))


def test_switch_and_event_tampering_rejected(saved_run):
    with pytest.raises(ValueError):validate_driven_run(replace(saved_run,switches=()))
    e=replace(saved_run.events[0],full_stop_within_tolerance=True)
    with pytest.raises(ValueError):validate_driven_run(replace(saved_run,events=(e,)))
    with pytest.raises(ValueError):validate_driven_run(replace(saved_run,max_abs_work_balance_error_j=1))
    json.dumps(saved_run.as_dict(),allow_nan=False)


def test_covariance_3d_rotated_plane_and_translated_sources():
    m=switched_model();angle=.61;c,s=cos(angle),sin(angle)
    def rot(v):return (c*v[0]-s*v[1],s*v[0]+c*v[1],v[2])
    offset=(.7,-.3,.5)
    spec=replace(m.spec,particle=PointCharge(m.spec.particle.charge_c,tuple(a+b for a,b in zip(rot(m.spec.particle.position_m),offset))),
        initial_velocity_m_s=rot(m.spec.initial_velocity_m_s),plane_normal=rot(m.spec.plane_normal),
        gravity=replace(m.spec.gravity,acceleration_m_s2=rot(m.spec.gravity.acceleration_m_s2)))
    base=replace(m.base,spec=spec)
    transformed=replace(m,base=base,applied=replace(m.applied,phases=tuple(replace(p,field_v_m=rot(p.field_v_m)) for p in m.applied.phases)))
    first,second=integrate(m),integrate(transformed)
    for a,b in zip(first.samples,second.samples):
        assert b.position_m==pytest.approx(tuple(x+y for x,y in zip(rot(a.position_m),offset)),abs=2e-9)
        assert b.velocity_m_s==pytest.approx(rot(a.velocity_m_s),abs=2e-9)
        assert b.dissipated_energy_j==pytest.approx(a.dissipated_energy_j,rel=2e-8,abs=1e-21)


def test_switch_time_and_mass_revision_dont_rebuild_static_sources():
    parent=switched_model()
    schedule=replace(parent.applied,phases=(FieldPhase(0,(.2,0,0)),FieldPhase(.27,(-1.2,0,.5))))
    child,_=revise_driven_model(parent,model_id='child:time',note='Earlier switch',applied=schedule)
    assert child.base is parent.base and child.drag is parent.drag
    assert integrate(child).switches[0].time_s==.27
    base=revise_scenario(parent.base,scenario_id='base:mass',note='Double mass',motion_changes={'mass_kg':2e-9})
    heavier,_=revise_driven_model(parent,model_id='child:mass',note='Heavier charged probe',base=base)
    assert heavier.applied is parent.applied and heavier.drag is parent.drag
    # Gravity acceleration unchanged; electric acceleration and relaxation change.
    a=integrate(parent).samples[0];b=integrate(heavier).samples[0]
    assert tuple(f/heavier.spec.mass_kg for f in b.gravitational_force_n)==pytest.approx(parent.spec.gravity.acceleration_m_s2)
    assert b.applied_electric_force_n==a.applied_electric_force_n
    assert b.drag_force_n==a.drag_force_n


def test_multiple_switches_and_output_times_have_no_state_impulses():
    m=model(b=.5e-9,phases=(FieldPhase(0,(.1,0,0)),FieldPhase(.13,(-.5,0,.1)),FieldPhase(.3,(.2,0,-.1)),FieldPhase(.81,(0,0,0))))
    run=integrate(m)
    assert tuple(s.time_s for s in run.switches)==(.13,.3,.81)
    for sw in run.switches:
        for k in ('position_m','velocity_m_s','kinetic_energy_j','fixed_work_j','applied_work_j','gravitational_work_j','dissipated_energy_j'):
            assert getattr(sw.left,k)==getattr(sw.right,k)
    assert all(a.time_s<b.time_s for a,b in zip(run.samples,run.samples[1:]))


def test_gauge_changes_cannot_change_dissipation_or_event_time():
    a=model();source=a.spec.field.sources[0]
    base=replace(a.base,spec=replace(a.spec,field=PrescribedField((replace(source,reference_potential_v=123.),))))
    b=replace(a,base=base)
    qa=(VelocityReversal('apex',(0,0,1),'Apex'),)
    x,y=integrate(a,qa),integrate(b,qa)
    for s,t in zip(x.samples,y.samples):
        assert s.position_m==t.position_m and s.velocity_m_s==t.velocity_m_s
        assert s.dissipated_energy_j==t.dissipated_energy_j and s.fixed_work_j==t.fixed_work_j
    assert x.events[0].sample.time_s==y.events[0].sample.time_s


def test_query_order_does_not_change_resolved_physics():
    m=switched_model();q=(VelocityReversal('x',(1,0,0),'X'),PlaneEvent('z',(0,0,.5),(0,0,1),'Z',1,True))
    a,b=integrate(m,q),integrate(m,q[::-1])
    assert a.samples==b.samples
    assert sorted((x.event_id,x.sample.time_s) for x in a.events)==sorted((x.event_id,x.sample.time_s) for x in b.events)


def test_dissipation_without_external_work_equals_lost_kinetic_energy():
    run=integrate(model(b=2e-9,sources=(),gravity=(0,0,0)))
    initial=run.samples[0].kinetic_energy_j
    for s in run.samples:
        assert s.fixed_work_j==s.applied_work_j==s.gravitational_work_j==0
        assert s.kinetic_energy_j+s.dissipated_energy_j==pytest.approx(initial,rel=2e-8)


def test_driver_event_evaluation_budget_is_not_reported_no_event():
    m=switched_model();p=replace(policy(),max_evaluations=100,max_step_s=.001)
    with pytest.raises(MotionIntegrationError,match='budget'):
        investigate_driven_motion(DrivenStudy(m,(VelocityReversal('x',(1,0,0),'X'),),p))


def test_uniform_force_superposition_not_an_equivalent_drag_field():
    m=switched_model();a=integrate(m)
    extra=UniformElectricField((.3,0,.1))
    base=replace(m.base,spec=replace(m.spec,field=PrescribedField((extra,))),source_annotations=(('new:fixed','Added fixed E'),))
    applied=replace(m.applied,phases=tuple(replace(p,field_v_m=tuple(x-y for x,y in zip(p.field_v_m,extra.field_v_m))) for p in m.applied.phases))
    b=integrate(replace(m,base=base,applied=applied))
    for s,t in zip(a.samples,b.samples):
        assert s.position_m==pytest.approx(t.position_m,abs=2e-9)
        assert s.velocity_m_s==pytest.approx(t.velocity_m_s,abs=2e-9)
        assert s.applied_work_j==pytest.approx(t.applied_work_j+t.fixed_work_j,abs=2e-18)
        assert s.dissipated_energy_j==pytest.approx(t.dissipated_energy_j,rel=2e-8,abs=1e-22)


def test_unrepresentable_relaxation_rate_refused():
    m=model();base=replace(m.base,spec=replace(m.spec,mass_kg=1e300))
    with pytest.raises(ValueError,match='relaxation'):
        replace(m,base=base,drag=LinearDrag(1e-300))


def test_oil_source_metadata_and_hypothetical_are_distinct():
    from examples.drag_and_field_schedule import make_case
    study,meta=make_case('oil-hypothetical')
    assert meta['original_source']['spacing_m']==.01 and meta['original_source']['voltage_v']==200
    assert meta['derived_equilibrium']['rounded_integer_count']==6
    assert meta['teacher_additions']['continuous_balancing_charge_not_exact_integer'] is True
    run=investigate_driven_motion(study)
    sw=run.switches[0]
    assert sw.time_s==.013 and sw.left.velocity_m_s[2]==pytest.approx(-.02)
    assert sw.left.velocity_m_s==sw.right.velocity_m_s
    assert run.termination=='duration' and not run.events
    assert run.samples[-1].velocity_m_s[2]<0
    assert run.samples[-1].velocity_m_s[2]==pytest.approx(-.02*exp(-(.035-.013)/.002),rel=3e-5)
    assert run.max_abs_work_balance_error_j<5e-26


def test_viewer_has_truthful_schema_work_terms_and_escaped_text():
    from conductor_sim.driven_viewer import render_driven_run
    m=model();run=integrate(replace(m,provenance='Teacher text </script><img src=x onerror=oops>'))
    html=render_driven_run(run,title='<test>')
    assert '<title>&lt;test&gt;</title>' in html
    assert '<img src=x' not in html and '\\u003c/script' in html
    assert 'driven_run.v1' in html and 'Work-balance residual' in html
    assert 'Numerical energy drift:' not in html and 'browser physics' not in html.lower()
