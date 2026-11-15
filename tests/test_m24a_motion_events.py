"""Preselected M24a event investigations; reserved R is NOT executed here."""
from dataclasses import replace, FrozenInstanceError
from math import sqrt, pi, sin, cos, atan2
import json

import pytest
from conductor_sim import (PrescribedField, UniformElectricField, UniformGravity,
    PointCharge, TestChargeMotionSpec as MotionSpec, UniformSphereField,
    FixedPointField, UniformRingAxisField, EPSILON_0, integrate_test_charge)
from conductor_sim.motion_events import (PlaneEvent, VelocityReversal, EventPolicy,
    EventStudy, EventRun, investigate_motion, validate_event_run)
from conductor_sim.motion_1d import MotionIntegrationError
from conductor_sim.scenarios import PrescribedScenario, revise_scenario
from conductor_sim.solver import UnsupportedProblem


def model(*,field=None,position=(0.,0.,0.),velocity=(1.,0.,0.),duration=2.,interval=.5,gravity=None):
    field=field or PrescribedField((UniformElectricField((-1000.,0.,0.)),))
    return PrescribedScenario('events:original',MotionSpec(field,PointCharge(1e-9,position),
        1e-6,velocity,duration,interval,plane_normal=(0.,1.,0.),length_scale_m=1.,
        source_guard_m=.001,gravity=gravity),
        tuple((f'source:{i}',f'Fixed source {i}') for i in range(len(field.sources))),
        'Teacher hypothetical for reusable event qualification.')


def policy(**kw):
    return replace(EventPolicy(.05,1e-9,1e-9,1e-8),**kw)


def run(m,queries,**kw):
    return investigate_motion(EventStudy(m,queries,policy(**kw)))


def test_unchanged_model_answers_different_questions():
    m=model();saved=m.as_dict()
    turn=VelocityReversal('turn:x',(1,0,0),'Turn in x')
    plane=PlaneEvent('detector:x',(.25,0,0),(1,0,0),'Detector')
    a=run(m,(turn,));b=run(m,(plane,));both=run(m,(turn,plane))
    assert m.as_dict()==saved and a.study.scenario is b.study.scenario is both.study.scenario is m
    assert len(a.events)==1 and len(b.events)==2 and len(both.events)==3
    event=a.events[0];assert event.sample.time_s==pytest.approx(1.,abs=1e-8)
    assert event.sample.position_m[0]==pytest.approx(.5,abs=1e-9)
    assert event.full_stop_within_tolerance
    assert abs(event.sample.velocity_m_s[0])<=a.study.policy.velocity_tolerance_m_s
    assert event.sample.kinetic_energy_j<1e-20
    assert event.sample.total_energy_j==pytest.approx(5e-7,abs=1e-16)
    assert [e.sample.time_s for e in b.events]==pytest.approx([1-sqrt(.5),1+sqrt(.5)],abs=1e-8)
    assert both.trajectory.termination=='duration'


def test_terminal_observation_first_competing_event():
    m=model()
    queries=(VelocityReversal('turn:x',(1,0,0),'Turn'),
        PlaneEvent('near',(.25,0,0),(1,0,0),'Near stop',1,True,'ideal-stop-surface'),
        PlaneEvent('far',(.4,0,0),(1,0,0),'Far stop',1,True,'ideal-stop-surface'))
    result=run(m,queries)
    assert result.trajectory.termination=='terminal-event'
    assert [e.event_id for e in result.events]==['near']
    assert result.trajectory.samples[-1].time_s==pytest.approx(1-sqrt(.5),abs=1e-8)
    assert result.as_dict()['outcomes']['far']=='search-truncated'
    assert result.trajectory.samples[-1].position_m[0]==pytest.approx(.25,abs=1e-9)
    # Event-location uncertainty permits |F|*position_tolerance in K; this
    # is not an ODE energy-conservation tolerance.
    assert abs(result.events[0].sample.kinetic_energy_j-2.5e-7)<=1e-6*result.study.policy.position_tolerance_m


def test_two_crossings_hidden_from_one_output_interval():
    m=model(interval=2.)
    result=run(m,(PlaneEvent('hidden',(.48,0,0),(1,0,0),'Two crossings'),),max_step_s=2.)
    assert len(result.trajectory.samples)==2
    assert result.trajectory.samples[0].position_m[0]==pytest.approx(result.trajectory.samples[-1].position_m[0],abs=1e-12)
    assert [e.sample.time_s for e in result.events]==pytest.approx([.8,1.2],abs=1e-8)
    assert [e.direction for e in result.events]==[1,-1]


@pytest.mark.parametrize('direction,expected',[(1,1-sqrt(.5)),(-1,1+sqrt(.5))])
def test_crossing_direction_filter(direction,expected):
    result=run(model(),(PlaneEvent('plane',(.25,0,0),(1,0,0),'Plane',direction),))
    assert len(result.events)==1 and result.events[0].sample.time_s==pytest.approx(expected,abs=1e-8)


def test_directional_apex_is_not_full_stop_with_gravity():
    m=model(field=PrescribedField((UniformElectricField((2000,0,0)),)),
        position=(.1,0,.2),velocity=(.2,0,1.2),duration=1.,interval=.25,
        gravity=UniformGravity((0,0,-3)))
    result=run(m,(VelocityReversal('apex',(0,0,1),'Vertical reversal',-1),
        PlaneEvent('wall',(.5,0,0),(1,0,0),'Right boundary',1,True),
        PlaneEvent('floor',(0,0,0),(0,0,1),'Lower boundary',-1,True)))
    assert [e.event_id for e in result.events]==['apex','wall']
    apex=result.events[0];assert apex.sample.time_s==pytest.approx(.4,abs=1e-8)
    assert not apex.full_stop_within_tolerance
    assert apex.sample.velocity_m_s[0]==pytest.approx(1.)
    assert apex.sample.position_m[2]==pytest.approx(.44)
    assert result.events[-1].sample.time_s==pytest.approx((- .2+sqrt(.2**2+1.6))/2,abs=1e-8)
    assert max(abs(e.sample.energy_drift_j) for e in result.events)<1e-17


def test_release_on_plane_initial_rest_is_not_future_collision():
    m=model(field=PrescribedField((UniformElectricField((1000,0,0)),)),
        velocity=(0,0,0),duration=2.)
    result=run(m,(VelocityReversal('rest',(1,0,0),'Initial rest'),
        PlaneEvent('launch',(0,0,0),(1,0,0),'Launch surface',-1,True),
        PlaneEvent('arrival',(.5,0,0),(1,0,0),'Arrival surface',1,True)))
    assert dict(result.initial_conditions)=={'rest':'initial-rest','launch':'initial-on-plane'}
    assert [e.event_id for e in result.events]==['arrival']
    assert result.events[0].sample.time_s==pytest.approx(1.,abs=1e-8)


def test_grazing_is_unresolved_not_exact_hit():
    result=run(model(interval=2.),(PlaneEvent('grazing',(.5,0,0),(1,0,0),'Tangent',0,True),),max_step_s=2.)
    assert result.trajectory.termination=='event-unresolved'
    assert not result.events and result.unresolved
    assert 'grazing' in result.unresolved[0][-1]
    assert result.trajectory.samples[-1].time_s<1.


def test_nonterminal_unavailable_question_does_not_remove_resolved_others():
    result=run(model(),(PlaneEvent('contained',(0,0,0),(0,1,0),'Entire motion plane'),
        VelocityReversal('turn',(1,0,0),'Turn')))
    assert result.trajectory.termination=='duration'
    assert [e.event_id for e in result.events]==['turn']
    assert result.as_dict()['outcomes']['contained']=='unresolved'


def test_near_miss_and_no_event_are_not_global_absence_claims():
    r=run(model(),(PlaneEvent('miss',(.51,0,0),(1,0,0),'Unreached plane'),))
    assert not r.events and not r.unresolved
    assert r.as_dict()['outcomes']['miss']=='not-found-at-requested-resolution'


def test_simultaneous_detectors_keep_all_ids_without_order_claim():
    qs=(PlaneEvent('a',(.25,0,0),(1,0,0),'Plane A',1,True),
        PlaneEvent('b',(.25,0,0),(1,0,0),'Plane B',1,True))
    a=run(model(),qs);b=run(model(),tuple(reversed(qs)))
    assert {e.event_id for e in a.events}=={'a','b'}
    assert a.ordering=='coincident-or-order-unresolved-within-time-resolution'
    assert a.trajectory.samples[-1].time_s==b.trajectory.samples[-1].time_s


def test_nonuniform_volume_field_event_against_harmonic_solution_not_reserved():
    Q,R,q,mass=.3e-9,.8,-1e-9,1e-8
    field=PrescribedField((UniformSphereField(Q,R),))
    spec=MotionSpec(field,PointCharge(q,(.2,0,0)),mass,(.1,0,0),4.,.5,
        plane_normal=(0,1,0),length_scale_m=1.,relative_tolerance=1e-9)
    m=PrescribedScenario('sphere:development',spec,(('volume','Fixed volume sphere'),),
        'Independent development control; distinct from the reserved R recipe.')
    omega=sqrt(abs(q)*Q/(4*pi*EPSILON_0*mass*R**3))
    t_turn=atan2(.1, .2*omega)/omega
    t_cross=(atan2(.1,.2*omega)+pi/2)/omega
    r=run(m,(VelocityReversal('turn',(1,0,0),'Reversal',-1),
        PlaneEvent('zero',(0,0,0),(1,0,0),'Centre crossing',-1,True)))
    assert [e.sample.time_s for e in r.events]==pytest.approx([t_turn,t_cross],abs=5e-7)
    for e in r.events:
        t=e.sample.time_s
        assert e.sample.position_m[0]==pytest.approx(.2*cos(omega*t)+.1/omega*sin(omega*t),abs=2e-8)
    assert all(abs(s.position_m[0])<R for s in r.trajectory.samples)


def test_coulomb_turning_radius_from_independent_energy():
    Q=q=1e-9;mass=1e-6;r0=.2;v0=-.15
    field=PrescribedField((FixedPointField(Q),))
    spec=MotionSpec(field,PointCharge(q,(r0,0,0)),mass,(v0,0,0),1.,.1,
        length_scale_m=.3,plane_normal=(0,1,0),relative_tolerance=1e-9)
    m=PrescribedScenario('coulomb:turn',spec,(('point','Fixed source'),),'Nonlinear turning test, teacher hypothetical.')
    r=run(m,(VelocityReversal('turn',(1,0,0),'Coulomb turning',1,True),),max_step_s=.005)
    reference=(q*Q/(4*pi*EPSILON_0))/(.5*mass*v0*v0+q*Q/(4*pi*EPSILON_0*r0))
    assert r.events[0].sample.position_m[0]==pytest.approx(reference,abs=2e-8)
    assert r.trajectory.termination=='terminal-event'


def test_source_guard_is_not_reclassified_as_detector_hit():
    f=PrescribedField((FixedPointField(-1e-9),))
    spec=MotionSpec(f,PointCharge(1e-9,(.1,0,0)),1e-6,(-1,0,0),.5,.02,
        length_scale_m=.2,source_guard_m=.01,plane_normal=(0,1,0))
    m=PrescribedScenario('guard:case',spec,(('point','Attracting source'),),'Guard before plane.')
    r=run(m,(PlaneEvent('behind',(.005,0,0),(1,0,0),'Inside excluded region',-1,True),))
    assert r.trajectory.termination=='source-guard' and not r.events
    assert r.as_dict()['outcomes']['behind']=='search-truncated'
    assert r.trajectory.samples[-1].position_m[0]>.01


def test_same_model_real_revision_not_numerical_policy_edit():
    m=model();child=revise_scenario(m,scenario_id='revised',note='Faster genuine launch.',
                                   motion_changes={'initial_velocity_m_s':(1.2,0,0)})
    q=(VelocityReversal('turn',(1,0,0),'Turn',-1,True),)
    a=run(m,q);b=run(child,q)
    assert a.events[0].sample.time_s==pytest.approx(1.,abs=1e-8)
    assert b.events[0].sample.time_s==pytest.approx(1.2,abs=1e-8)
    assert child.parent_revision_id==m.revision_id
    with pytest.raises(FrozenInstanceError):a.study.scenario.spec.mass_kg=1.


@pytest.mark.parametrize('changes',[{'event_id':''},{'normal':(0,0,0)},
    {'point_m':(0,0)},{'direction':True},{'terminal':1},{'label':'bad\nlabel'},
    {'role':'finite-material-plate'}])
def test_invalid_event_requests(changes):
    args=dict(event_id='plane',point_m=(0,0,0),normal=(1,0,0),label='Plane');args.update(changes)
    with pytest.raises((ValueError,TypeError)):PlaneEvent(**args)


@pytest.mark.parametrize('kw',[{'max_step_s':0},{'time_tolerance_s':.1},
    {'position_tolerance_m':True},{'velocity_tolerance_m_s':float('nan')},
    {'max_events':False},{'max_evaluations':3}])
def test_invalid_numerical_policy(kw):
    with pytest.raises((TypeError,ValueError)):policy(**kw)


def test_unsupported_analysis_and_exhaustion_are_not_no_event():
    with pytest.raises(TypeError):EventStudy(model(),(lambda t,y:y[0],),policy())
    with pytest.raises(ValueError):EventStudy(model(),(),policy())
    with pytest.raises(ValueError):EventStudy(model(),(VelocityReversal('a',(1,0,0),'a'),)*2,policy())
    with pytest.raises(UnsupportedProblem):EventStudy(model(),(VelocityReversal('normal',(0,1,0),'Zero by symmetry'),),policy())
    with pytest.raises(MotionIntegrationError,match='budget'):
        run(model(),(PlaneEvent('far',(100,0,0),(1,0,0),'Far'),),max_step_s=.001,max_evaluations=100)
    with pytest.raises(UnsupportedProblem):
        model(field=PrescribedField((UniformRingAxisField(1e-9,.1),)),velocity=(1,0,1))


def test_event_and_energy_tampering_refused():
    r=run(model(),(VelocityReversal('turn',(1,0,0),'Turn',-1,True),))
    e=r.events[0]
    for bad in (replace(e,residual=2.),replace(e,direction=True),replace(e,terminal=False),
                replace(e,time_bracket_s=(0.,2.)),replace(e,full_stop_within_tolerance=False),
                replace(e,sample=replace(e.sample,kinetic_energy_j=1.))):
        with pytest.raises(ValueError):validate_event_run(replace(r,events=(bad,)))
    assert json.dumps(r.as_dict(),allow_nan=False)


def test_default_old_integrator_not_changed_by_event_execution():
    m=model();old=integrate_test_charge(m.spec).as_dict()
    run(m,(VelocityReversal('turn',(1,0,0),'Turn'),))
    assert integrate_test_charge(m.spec).as_dict()==old


@pytest.mark.parametrize('normal',( (1,0,0),(-1,0,0)))
def test_same_plane_reversed_normal_convention(normal):
    d=1 if normal[0]>0 else -1
    r=run(model(),(PlaneEvent('p',(.25,0,0),normal,'Plane',d,True),))
    assert r.events[0].sample.time_s==pytest.approx(1-sqrt(.5),abs=1e-8)


@pytest.mark.parametrize('interval',(.13,.5,2.))
def test_event_answer_not_saved_output_grid(interval):
    r=run(model(interval=interval),(VelocityReversal('turn',(1,0,0),'Turn',-1,True),))
    assert r.events[0].sample.time_s==pytest.approx(1.,abs=1e-8)
    assert r.events[0].time_bracket_s[1]-r.events[0].time_bracket_s[0]<=r.study.policy.time_tolerance_s*1.01


def test_initial_outgoing_or_outside_stop_surface_refused():
    q=PlaneEvent('lower',(0,0,0),(1,0,0),'Lower surface',-1,True,'ideal-stop-surface')
    for m in (model(velocity=(-1,0,0)),model(velocity=(0,0,0)),model(position=(-1,0,0))):
        with pytest.raises(UnsupportedProblem):EventStudy(m,(q,),policy())


def test_stationary_initial_state_is_not_repeated_reversal():
    m=model(field=PrescribedField((UniformElectricField((0,0,0)),)),velocity=(0,0,0))
    r=run(m,(VelocityReversal('v',(1,0,0),'At rest'),))
    assert not r.events and dict(r.initial_conditions)['v']=='initial-rest'


def test_event_count_exhaustion_reported_as_investigation_failure():
    with pytest.raises(MotionIntegrationError,match='count'):
        run(model(),(PlaneEvent('twice',(.25,0,0),(1,0,0),'Plane'),),max_events=1)


def test_numerical_policy_changes_do_not_reauthor_model():
    m=model();q=(VelocityReversal('turn',(1,0,0),'Turn',-1,True),)
    a=run(m,q);b=run(m,q,max_step_s=.013,time_tolerance_s=1e-11)
    assert a.study.scenario is b.study.scenario
    assert a.study.scenario.revision_id==b.study.scenario.revision_id
    assert abs(a.events[0].sample.time_s-b.events[0].sample.time_s)<1e-8


@pytest.mark.parametrize('case',('sheet-stop','sheet-encounter','plate-transit','oblique-gravity','source-guard'))
def test_preselected_source_and_transfer_html(case):
    from examples.motion_event_studies import make_case
    from conductor_sim.motion_event_viewer import render_event_run
    r=investigate_motion(make_case(case))
    html=render_event_run(r)
    assert 'event-data' in html and 'Physical t' in html
    assert 'no impact' in html and 'exact-ODE' in html
    assert 'authored' in html and 'fetch(' not in html
    assert len(r.events)==(2 if case=='oblique-gravity' else 0 if case=='source-guard' else 1)


def test_source_q19_givens_and_derived_stopping_distance():
    from examples.motion_event_studies import make_case,ELECTRON_Q,ELECTRON_M
    study=make_case('sheet-stop');r=investigate_motion(study)
    initial=study.scenario.spec
    assert initial.field.sources[0].field_v_m==(-2e-6/EPSILON_0,0,0)
    assert .5*ELECTRON_M*initial.initial_velocity_m_s[0]**2/ELECTRON_Q==pytest.approx(100.)
    distance=initial.particle.position_m[0]-r.events[0].sample.position_m[0]
    assert distance==pytest.approx(100*EPSILON_0/2e-6,abs=2e-12)
    assert r.events[0].full_stop_within_tolerance
    # The starting distance was disclosed, not an input copied from the answer.
    assert initial.particle.position_m[0]==.001


def test_source_q30_symbolic_relation_and_arrival_energy():
    from examples.motion_event_studies import make_case,ELECTRON_Q,ELECTRON_M
    r=investigate_motion(make_case('plate-transit'));e=r.events[0]
    ref=sqrt(2*ELECTRON_M*.01**2/(ELECTRON_Q*100))
    assert e.sample.time_s==pytest.approx(ref,abs=2e-17)
    assert e.sample.kinetic_energy_j==pytest.approx(100*ELECTRON_Q,rel=2e-8)
    assert dict(r.initial_conditions)['lower']=='initial-on-plane'


def test_display_escapes_authored_labels():
    from conductor_sim.motion_event_viewer import render_event_run
    q=VelocityReversal('x',(1,0,0),'</script><script>bad()</script>',-1,True)
    html=render_event_run(run(model(),(q,)),title='<img src=x onerror=bad()>')
    assert '<script>bad()' not in html and '<img src=x' not in html
    assert '\\u003c/script' in html


def test_event_rotation_translation_covariance():
    angle=.63;co,si=cos(angle),sin(angle)
    def rot(p):return (co*p[0]+si*p[2],p[1],-si*p[0]+co*p[2])
    offset=(.3,.2,-.1)
    def point(p):return tuple(a+b for a,b in zip(rot(p),offset))
    transformed=model(field=PrescribedField((UniformElectricField(rot((-1000,0,0))),)),
        position=offset,velocity=rot((1,0,0)))
    r=run(transformed,(PlaneEvent('p',point((.25,0,0)),rot((1,0,0)),'Transformed detector',1,True),))
    assert r.events[0].sample.time_s==pytest.approx(1-sqrt(.5),abs=1e-8)
    assert r.events[0].sample.position_m==pytest.approx(point((.25,0,0)),abs=2e-9)


def test_close_competing_events_do_not_assert_resolved_priority():
    qs=(PlaneEvent('a',(.25,0,0),(1,0,0),'A',1,True),
        PlaneEvent('b',(.25+1e-10,0,0),(1,0,0),'B',1,True))
    r=run(model(),qs)
    assert r.ordering=='coincident-or-order-unresolved-within-time-resolution'
    assert {e.event_id for e in r.events}=={'a','b'}


def test_event_too_close_to_initial_time_is_unresolved_not_silently_absent():
    m=model(field=PrescribedField((UniformElectricField((0,0,0)),)))
    r=run(m,(PlaneEvent('early',(5e-10,0,0),(1,0,0),'Early plane',1,True),),position_tolerance_m=1e-12)
    assert r.trajectory.termination=='event-unresolved'
    assert not r.events and r.unresolved
    assert r.as_dict()['outcomes']['early']=='unresolved'
