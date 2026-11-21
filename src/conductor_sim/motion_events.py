"""M24a: typed event questions on an unchanged prescribed-motion scenario.

Observe accepted RK intervals, propose topology using cubic Hermite scalars,
then locate roots with fresh RK step-doubling states, never saved-frame linear
interpolation. This is finite-resolution numerical evidence, NOT an exact-ODE
root-count proof or a contact/charge-transfer law. All source guards remain.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import copysign, fsum, hypot, isfinite, sqrt, ulp
import re

from ._motion_rk4 import rk4_step
from ._motion_validation import reconstructs_position
from .conservative_forces import CONTRIBUTION_KEYS
from .motion_1d import MotionIntegrationError, _NONRELATIVISTIC_SPEED_LIMIT_M_S
from .motion_2d import _SurfaceGuard, _ChargeGuard, _world_state
from .prescribed_fields import _dot, _sub, _mul, _unit, _vector, _number
from .prescribed_motion import (_FixedDomain, _integrate_test_charge,
                                TestChargeSample, TestChargeMotionResult)
from .scenarios import PrescribedScenario
from .solver import UnsupportedProblem


def _native(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _identity(value):
    if type(value) is not str or re.fullmatch(r'[A-Za-z][A-Za-z0-9:_-]{0,63}', value) is None:
        raise ValueError('event identity must be a bounded stable identifier')


def _common(obj):
    _identity(obj.event_id)
    if type(obj.label) is not str or not obj.label.isprintable() or not 1 <= len(obj.label) <= 120:
        raise ValueError('event label must be 1..120 printable characters')
    if type(obj.direction) is not int or obj.direction not in (-1, 0, 1):
        raise ValueError('direction must be integer -1, 0 or +1')
    if type(obj.terminal) is not bool:
        raise TypeError('terminal must be Boolean')


@dataclass(frozen=True, slots=True)
class PlaneEvent:
    """Infinite plane. Normal fixes the sign; direction filters signed crossings.

    A terminal request ends the reported interval. An ideal-stop-surface is
    an explicitly declared limit of the prescribed-field approximation, not
    a new responding conductor, collision, absorption or reflection model.
    """
    event_id: str
    point_m: tuple[float, float, float]
    normal: tuple[float, float, float]
    label: str
    direction: int = 0
    terminal: bool = False
    role: str = 'passive-detector'

    def __post_init__(self):
        _common(self)
        object.__setattr__(self, 'point_m', _vector(self.point_m, 'plane point'))
        object.__setattr__(self, 'normal', _unit(self.normal, 'plane normal'))
        if self.role not in ('passive-detector', 'ideal-stop-surface'):
            raise ValueError('unsupported surface role; only infinite prescribed-field targets')
        if self.role == 'ideal-stop-surface' and (not self.terminal or self.direction==0):
            raise ValueError('an ideal stop surface needs terminal=True and a signed outgoing direction')


@dataclass(frozen=True, slots=True)
class VelocityReversal:
    """Zero and sign reversal of v.dot(axis), not necessarily total rest."""
    event_id: str
    axis: tuple[float, float, float]
    label: str
    direction: int = 0
    terminal: bool = False

    def __post_init__(self):
        _common(self)
        object.__setattr__(self, 'axis', _unit(self.axis, 'reversal axis'))


@dataclass(frozen=True, slots=True)
class EventPolicy:
    """Numerical investigation policy, never a changed physical condition."""
    max_step_s: float
    time_tolerance_s: float
    position_tolerance_m: float
    velocity_tolerance_m_s: float
    max_evaluations: int = 100000
    max_events: int = 200

    def __post_init__(self):
        for key in ('max_step_s','time_tolerance_s','position_tolerance_m','velocity_tolerance_m_s'):
            _number(getattr(self, key), key, positive=True)
        if self.time_tolerance_s >= self.max_step_s:
            raise ValueError('time tolerance must be smaller than maximum investigation step')
        for key, lo, hi in (('max_evaluations',100,1000000),('max_events',1,1000)):
            if type(getattr(self,key)) is not int or not lo <= getattr(self,key) <= hi:
                raise ValueError(f'{key} outside bounded event policy')


@dataclass(frozen=True, slots=True)
class EventStudy:
    scenario: PrescribedScenario
    questions: tuple[PlaneEvent | VelocityReversal, ...]
    policy: EventPolicy

    def __post_init__(self):
        if type(self.scenario) is not PrescribedScenario or type(self.policy) is not EventPolicy:
            raise TypeError('require an existing PrescribedScenario and EventPolicy')
        object.__setattr__(self,'questions',_validate_questions(
            self.scenario.spec,self.questions,self.policy))

    def as_dict(self):
        return _native(dict(schema='conductor_sim.event_study.v1',
            model=self.scenario.as_dict(), model_revision_id=self.scenario.revision_id,
            questions=[dict(kind=type(q).__name__,**asdict(q)) for q in self.questions],
            numerical_policy=asdict(self.policy)))


def validate_event_requests(questions, policy):
    """Recheck authored native requests without renormalizing caller objects.

    This checks input contracts, not event-study physical admissibility.
    Physical ideal-stop and plane qualification remain in _validate_questions.
    """
    if type(policy) is not EventPolicy:
        raise TypeError('require a native EventPolicy')
    policy.__post_init__()  # Pure scalar/type checks, no mutation.
    if type(questions) not in (tuple, list) or not 1 <= len(questions) <= 16:
        raise ValueError('require 1..16 typed event questions')
    for question in questions:
        if type(question) not in (PlaneEvent, VelocityReversal):
            raise TypeError('unsupported event kind; arbitrary callbacks are not accepted')
        _common(question)
        axis = question.normal if type(question) is PlaneEvent else question.axis
        if type(axis) is not tuple:
            raise TypeError('native event direction must retain its immutable tuple')
        vector = _vector(axis, 'native event direction')
        if abs(hypot(*vector) - 1.) > 8*ulp(1.):
            raise ValueError('native event direction must be unit length')
        if type(question) is PlaneEvent:
            if type(question.point_m) is not tuple:
                raise TypeError('native event point must retain its immutable tuple')
            _vector(question.point_m, 'native event point')
            if type(question.role) is not str or question.role not in (
                    'passive-detector', 'ideal-stop-surface'):
                raise ValueError('unsupported surface role')
            if question.role == 'ideal-stop-surface' and (
                    not question.terminal or question.direction == 0):
                raise ValueError('an ideal stop surface needs a signed terminal direction')
    if len({q.event_id for q in questions}) != len(questions):
        raise ValueError('duplicate event identity')


def _validate_questions(spec, questions, policy, *, initial_force=None):
    validate_event_requests(questions, policy)
    if type(questions) not in (list,tuple) or not 1 <= len(questions) <= 16:
        raise ValueError('require 1..16 typed event questions')
    questions=tuple(questions)
    if any(type(q) not in (PlaneEvent,VelocityReversal) for q in questions):
        raise TypeError('unsupported event kind; arbitrary callbacks are not accepted')
    if len({q.event_id for q in questions}) != len(questions):
        raise ValueError('duplicate event identity')
    if policy.time_tolerance_s < 64*ulp(spec.duration_s):
        raise ValueError('event time tolerance is below resolved clock precision')
    for q in questions:
        direction = q.normal if type(q) is PlaneEvent else q.axis
        if type(q) is VelocityReversal and abs(_dot(direction,spec.plane_normal)) > 1-1e-12:
            raise UnsupportedProblem('normal velocity is identically zero in the qualified plane, not a reversal')
        if type(q) is PlaneEvent and q.role=='ideal-stop-surface':
            g=_dot(_sub(spec.particle.position_m,q.point_m),q.normal)
            if q.direction*g>0:
                raise UnsupportedProblem('initial state is beyond the declared stop surface')
            if g==0:
                v=_dot(spec.initial_velocity_m_s,q.normal)
                force=(initial_force if initial_force is not None else
                    spec.forces.observe(spec.particle.position_m,
                        charge_c=spec.particle.charge_c,mass_kg=spec.mass_kg).force_n)
                a=_dot(force,q.normal)/spec.mass_kg
                if q.direction*v>0 or (v==0 and q.direction*a>0):
                    raise UnsupportedProblem('initial release departs beyond the stop surface, not into the authored region')
    return questions


@dataclass(frozen=True, slots=True)
class LocatedEvent:
    event_id: str
    kind: str
    time_bracket_s: tuple[float, float]
    residual: float
    residual_unit: str
    direction: int
    terminal: bool
    full_stop_within_tolerance: bool
    sample: TestChargeSample


@dataclass(frozen=True, slots=True)
class EventRun:
    study: EventStudy
    trajectory: TestChargeMotionResult
    events: tuple[LocatedEvent, ...]
    initial_conditions: tuple[tuple[str,str], ...]
    unresolved: tuple[tuple[str,float,float,str], ...]
    ordering: str
    evaluations: int

    def as_dict(self):
        motion = self.trajectory.as_dict()
        # Do not mislabel new event termination as a legacy v1/v2 result.
        motion['schema']='conductor_sim.event_trajectory.v1'
        return _native(dict(schema='conductor_sim.event_run.v1',study=self.study.as_dict(),
            trajectory=motion,events=[asdict(e) for e in self.events],
            initial_conditions=self.initial_conditions,unresolved=self.unresolved,
            ordering=self.ordering,evaluations=self.evaluations,
            outcomes={q.event_id: ('located-with-unresolved-candidates' if
                any(e.event_id==q.event_id for e in self.events) and any(u[0]==q.event_id for u in self.unresolved)
                else 'located' if any(e.event_id==q.event_id for e in self.events)
                else 'unresolved' if any(u[0]==q.event_id for u in self.unresolved)
                else 'not-found-at-requested-resolution' if self.trajectory.termination=='duration'
                else 'search-truncated') for q in self.study.questions},
            qualification='Finite-resolution event search; brackets localize numerical roots, not exact ODE error. '
                'Initial zeros are separate; unlocated events are not proof of absence. '
                'Terminal planes stop observation, never imply impact/charge transfer.'))


def _sample(spec, t, state, displacement=None):
    p,v=tuple(state[:3]),tuple(state[3:])
    obs=spec.forces.observe(p,charge_c=spec.particle.charge_c,mass_kg=spec.mass_kg)
    initial=spec.forces.observe(spec.particle.position_m,charge_c=spec.particle.charge_c,mass_kg=spec.mass_kg)
    k=.5*spec.mass_kg*hypot(*v)**2
    drift=fsum((k-.5*spec.mass_kg*hypot(*spec.initial_velocity_m_s)**2,
        obs.electric_potential_energy_j-initial.electric_potential_energy_j,
        obs.gravitational_potential_energy_j-initial.gravitational_potential_energy_j))
    parts={key:getattr(obs,key) for key in CONTRIBUTION_KEYS} if spec.gravity else {}
    if displacement is None:
        displacement = _sub(p,spec.particle.position_m)
    return TestChargeSample(t,displacement,p,v,
        tuple(f/spec.mass_kg for f in obs.force_n),spec.field.electric_field_at(p),
        spec.field.potential_at(p),obs.force_n,k,obs.potential_energy_j,
        k+obs.potential_energy_j,drift,**parts)


def _value(q,y):
    return (_dot(_sub(y[:3],q.point_m),q.normal) if type(q) is PlaneEvent
            else _dot(y[3:6],q.axis))


def _derivative(q,y,rhs):
    return _dot(y[3:6],q.normal) if type(q) is PlaneEvent else _dot(rhs(y)[3:6],q.axis)


def _critical(a,b,c):
    """Critical fractions of a*s^3+b*s^2+c*s+d; stable quadratic formula."""
    A,B,C=3*a,2*b,c
    scale=max(abs(A),abs(B),abs(C),1e-300)
    A,B,C=A/scale,B/scale,C/scale
    if abs(A)<1e-15:
        roots=[] if abs(B)<1e-15 else [-C/B]
    else:
        disc=B*B-4*A*C
        if disc<0:return []
        z=-.5*(B+copysign(sqrt(disc),B))
        roots=[-B/(2*A)] if z==0 else [z/A,C/z]
    return sorted(set(x for x in roots if 0<x<1))


class _Refine(Exception):
    pass


class _Monitor:
    def __init__(self,study, *, sample_factory=None, work_error=None, switch_times=()):
        # Private adapters allow the same typed locator to retain work states.
        self.sample_factory=sample_factory;self.work_error=work_error
        self.switch_times=tuple(switch_times)
        self.study=study;self.spec=study.scenario.spec;self.policy=study.policy
        self.domain=_FixedDomain(self.spec);self.events=[];self.unresolved=[]
        self.evaluations=0;self.last={};self.ordering='resolved-separated-brackets'
        y=(*self.spec.particle.position_m,*self.spec.initial_velocity_m_s)
        self.initial=[]
        for q in study.questions:
            if abs(_value(q,y))<=self.tol(q):
                value=_value(q,y)
                tag=('initial-on-plane' if value==0 else 'initial-near-plane-within-tolerance') if type(q) is PlaneEvent else (
                    'initial-rest' if hypot(*y[3:6])==0 else
                    'initial-speed-within-rest-tolerance' if hypot(*y[3:6])<=self.policy.velocity_tolerance_m_s else
                    'initial-zero-component' if value==0 else 'initial-small-component-within-tolerance')
                self.initial.append((q.event_id,tag))

    def tol(self,q):
        return (self.policy.position_tolerance_m if type(q) is PlaneEvent
                else self.policy.velocity_tolerance_m_s)

    def __call__(self,t,y,h,half,fine,rhs,origin):
        cache={0.:y,1.:fine}
        launch = _sub(self.spec.particle.position_m, origin)

        def value(q, local):
            if type(q) is PlaneEvent:
                plane_offset = _sub(q.point_m, origin)
                return _dot(_sub(local[:3], plane_offset), q.normal)
            return _dot(local[3:6], q.axis)

        def state(x):
            if x in cache:return cache[x]
            self.evaluations+=1
            if self.evaluations>self.policy.max_evaluations:
                raise MotionIntegrationError('event investigation evaluation budget exhausted')
            dt=x*h
            try:
                coarse=rk4_step(y,dt,rhs)
                middle=rk4_step(y,dt/2,rhs)
                z=rk4_step(middle,dt/2,rhs)
            except (_SurfaceGuard,_ChargeGuard):raise _Refine()
            ps=max(1e-11*self.spec.length_scale_m,self.spec.relative_tolerance*max(
                self.spec.length_scale_m,hypot(*_sub(y[:3],launch)),hypot(*_sub(z[:3],launch))))
            vs=max(1e-10,self.spec.relative_tolerance*max(1.,hypot(*y[3:6]),hypot(*z[3:6])))
            if max(hypot(*_sub(coarse[:3],z[:3]))/(15*ps),
                   hypot(*_sub(coarse[3:6],z[3:6]))/(15*vs))>1:raise _Refine()
            if self.work_error is not None and self.work_error(coarse,z,y)>1:raise _Refine()
            start, mid, end = (_world_state(origin,s)[:3] for s in (y,middle,z))
            if (self.domain.clearance(end)<=0 or
                self.domain.charge_segment_clearance(start,mid)<=0 or
                self.domain.charge_segment_clearance(mid,end)<=0):raise _Refine()
            if hypot(*z[3:6])>=_NONRELATIVISTIC_SPEED_LIMIT_M_S:
                raise UnsupportedProblem('event evaluation exceeds nonrelativistic regime')
            cache[x]=z
            return z
        located=[];uncertain=[];event_states={}
        try:
            for q in self.study.questions:
                g0,g1=value(q,y),value(q,fine)
                d0,d1=_derivative(q,y,rhs)*h,_derivative(q,fine,rhs)*h
                a,b=2*g0-2*g1+d0+d1,-3*g0+3*g1-2*d0-d1
                if not all(isfinite(v) for v in (g0,g1,d0,d1,a,b)):
                    raise MotionIntegrationError('nonfinite event geometry or derivative')
                def polynomial(x):return ((a*x+b)*x+d0)*x+g0
                tol=self.tol(q)
                values=[g0,g1]
                # Dense polynomial proposes topology only. Check it against
                # actual independent fractional steps before trusting proposals.
                for x in (.25,.5,.75):
                    gx=value(q,state(x));values.append(gx)
                    if abs(gx-polynomial(x))>tol*.25:raise _Refine()
                if all(v==0 for v in values) and d0==d1==0:
                    # No isolated root can be asserted for a continuum of zeros.
                    if type(q) is PlaneEvent:
                        uncertain.append((q.event_id,t,t+h,'trajectory-on-or-unresolved-near-plane'))
                    continue
                nodes=[0.,*_critical(a,b,d0),1.]
                vals=[value(q,state(x)) for x in nodes]
                for x,g in zip(nodes[1:-1],vals[1:-1]):
                    if abs(g)<=tol:
                        uncertain.append((q.event_id,t,t+h,'grazing-or-near-tangent: no transverse root certified'))
                for lo,hi,gl,gh in zip(nodes,nodes[1:],vals,vals[1:]):
                    if gl!=0 and gh!=0 and (gl>0)==(gh>0):continue
                    if gl==gh==0:continue
                    direction=1 if gh>gl else -1
                    if q.direction and q.direction!=direction:continue
                    L,H=lo,hi
                    # Initial zeros and duplicate shared step-end roots are not
                    # later encounters. Initial release has a separate record.
                    if gl==0:
                        x=lo;L=H=x
                    elif gh==0:
                        x=hi;L=H=x
                    else:
                        for _ in range(90):
                            x=(L+H)/2;gx=value(q,state(x))
                            if H-L<=self.policy.time_tolerance_s/h and abs(gx)<=tol:break
                            if x==L or x==H:raise _Refine()
                            if gx==0:L=H=x;break
                            if (gl>0)!=(gx>0):H=x
                            else:L=x;gl=gx
                        else:raise _Refine()
                    time=t+x*h
                    if time==0:
                        continue
                    if time<=self.policy.time_tolerance_s:
                        uncertain.append((q.event_id,0.,min(t+h,self.policy.time_tolerance_s),
                                          'event not separated from initial time at requested resolution'))
                        continue
                    if time<=self.last.get(q.event_id,-1)+self.policy.time_tolerance_s:
                        previous=next((e for e in reversed(self.events) if e.event_id==q.event_id),None)
                        if previous is not None and previous.direction!=direction:
                            uncertain.append((q.event_id,previous.sample.time_s,time,
                                              'opposite crossings not temporally resolved'))
                        continue
                    z=state(x)
                    world = _world_state(origin, z)
                    residual=_value(q,world)
                    derivative=_derivative(q,z,rhs)
                    # A near-stationary surface root is not a certified hit.
                    if type(q) is PlaneEvent and abs(derivative)<=self.policy.velocity_tolerance_m_s:
                        uncertain.append((q.event_id,t+L*h,t+H*h,'grazing-or-unresolved-normal-speed'))
                        continue
                    if abs(residual)>tol:raise _Refine()
                    if any(abs(time-switch)<=self.policy.time_tolerance_s for switch in self.switch_times):
                        uncertain.append((q.event_id,t+L*h,t+H*h,
                            'event/switch ordering unresolved within requested time resolution'))
                        continue
                    if (self.sample_factory is not None and type(q) is VelocityReversal and
                        abs(derivative)*self.policy.max_step_s<=self.policy.velocity_tolerance_m_s):
                        uncertain.append((q.event_id,t+L*h,t+H*h,
                            'near-stationary velocity root: reversal not resolved'))
                        continue
                    row=(self.sample_factory(time,world) if self.sample_factory is not None
                         else _sample(self.spec,time,world,_sub(z[:3],launch)))
                    event_states[(q.event_id,time)]=z
                    located.append(LocatedEvent(q.event_id,
                        'plane-crossing' if type(q) is PlaneEvent else 'directional-reversal',
                        (t+L*h,t+H*h),residual,'m' if type(q) is PlaneEvent else 'm/s',
                        direction,q.terminal,hypot(*z[3:6])<=self.policy.velocity_tolerance_m_s,row))
        except _Refine:
            if h<=8*self.policy.time_tolerance_s:
                self.unresolved.append(('resolution',t,t+h,'event geometry/state could not be resolved'))
                return t,y,'event-unresolved'
            return False
        located.sort(key=lambda e:(e.time_bracket_s[0],e.event_id))
        # A tangency candidate has no physically justified exact hit time.
        # Conservatively retain only the already verified interval before it.
        if uncertain:
            for item in uncertain:
                if not any(old[0]==item[0] and old[3]==item[3] for old in self.unresolved):
                    self.unresolved.append(item)
            ids={item[0] for item in uncertain}
            located=[e for e in located if e.event_id not in ids]
            if any(q.terminal and q.event_id in ids for q in self.study.questions):
                return t,y,'event-unresolved'
        terminal=next((e for e in located if e.terminal),None)
        if terminal:
            kept=[e for e in located if e.time_bracket_s[0]<=terminal.time_bracket_s[1]]
            terminal=min((e for e in kept if e.terminal),key=lambda e:e.sample.time_s)
        else:kept=located
        if len(self.events)+len(kept)>self.policy.max_events:
            raise MotionIntegrationError('event count exceeds bounded investigation policy')
        for e in kept:
            if e.sample.time_s>self.last.get(e.event_id,-1)+self.policy.time_tolerance_s:
                self.events.append(e);self.last[e.event_id]=e.sample.time_s
        for a,b in zip(kept,kept[1:]):
            if b.time_bracket_s[0]<=a.time_bracket_s[1]+self.policy.time_tolerance_s:
                self.ordering='coincident-or-order-unresolved-within-time-resolution'
        if terminal:
            row=terminal.sample
            return row.time_s,event_states[(terminal.event_id,row.time_s)],'terminal-event'
        return None


def investigate_motion(study: EventStudy) -> EventRun:
    """Ask several typed questions without changing/rebuilding the physical model."""
    if type(study) is not EventStudy:raise TypeError('require EventStudy')
    before=study.scenario.revision_id
    monitor=_Monitor(study)
    trajectory=_integrate_test_charge(study.scenario.spec,step_observer=monitor,
                                      max_step_s=study.policy.max_step_s)
    if before!=study.scenario.revision_id:raise RuntimeError('authored model changed')
    result=EventRun(study,trajectory,tuple(monitor.events),tuple(monitor.initial),
                    tuple(monitor.unresolved),monitor.ordering,monitor.evaluations)
    validate_event_run(result)
    return result


def _validate_event_numbers(event):
    """Boolean numeric aliases must never become saved times or residuals."""
    bracket = event.time_bracket_s
    if type(bracket) not in (tuple, list) or len(bracket) != 2:
        raise ValueError('event bracket must contain two finite physical times')
    for value in bracket:
        _number(value, 'event bracket time')
    _number(event.residual, 'event residual')


def _validate_event_horizon(event, events, last_time, termination, ordering, tolerance):
    """Retain overlapping terminal alternatives, never later resolved motion.

    A cluster can have slightly different representative numerical root times.
    Only candidates whose brackets overlap the retained terminal bracket may
    extend past its representative time, and only within the declared resolution.
    Those alternative samples are not eligible for post-stop playback.
    """
    time = _number(event.sample.time_s, 'event time')
    if time <= last_time:
        return
    overlaps = any(e.terminal and e.sample.time_s == last_time and
        max(event.time_bracket_s[0], e.time_bracket_s[0]) <=
        min(event.time_bracket_s[1], e.time_bracket_s[1]) for e in events)
    if (termination != 'terminal-event' or
            ordering != 'coincident-or-order-unresolved-within-time-resolution' or
            time-last_time > tolerance or not overlaps):
        raise ValueError('event is beyond the observed interval, not an unresolved terminal alternative')


def validate_event_run(run):
    """Observation consistency, not authentication or an independent ODE proof."""
    if type(run) is not EventRun or run.trajectory.spec!=run.study.scenario.spec:
        raise ValueError('event history must retain its exact physical model')
    spec=run.study.scenario.spec;rows=run.trajectory.samples;policy=run.study.policy
    if not rows or rows[0].time_s!=0 or any(a.time_s>=b.time_s for a,b in zip(rows,rows[1:])):
        raise ValueError('event history needs ordered actual times from zero')
    if run.trajectory.termination not in ('duration','source-guard','terminal-event','event-unresolved'):
        raise ValueError('unknown event history termination')
    if run.trajectory.termination=='duration' and rows[-1].time_s!=spec.duration_s:
        raise ValueError('duration result is incomplete')
    if rows[-1].time_s>spec.duration_s or len(rows)>2001:
        raise ValueError('event history exceeds requested horizon')
    if rows[0].position_m!=spec.particle.position_m or rows[0].velocity_m_s!=spec.initial_velocity_m_s:
        raise ValueError('saved initial conditions changed')
    for row in (*rows,*(e.sample for e in run.events)):
        if type(row.time_s) not in (int,float) or not isfinite(row.time_s):
            raise ValueError('invalid saved physical time')
        for key in ('position_m','displacement_m','velocity_m_s','force_n','electric_field_v_m','acceleration_m_s2'):
            _vector(getattr(row,key),key)
        if _FixedDomain(spec).clearance(row.position_m)<=0:raise ValueError('event inside source guard')
        expected=_sample(spec,row.time_s,(*row.position_m,*row.velocity_m_s))
        if ((row.time_s == 0 and any(row.displacement_m)) or
                abs(_dot(row.displacement_m,spec.plane_normal)) > 1e-9*max(
                    spec.length_scale_m, hypot(*row.displacement_m))):
            raise ValueError('saved event displacement violates launch plane')
        if (not reconstructs_position(spec.particle.position_m, row.displacement_m, row.position_m)
                and row.displacement_m != expected.displacement_m):
            raise ValueError('saved event displacement mismatch')
        for key in ('force_n','electric_field_v_m','acceleration_m_s2'):
            if getattr(row,key)!=getattr(expected,key):raise ValueError('saved event force/field mismatch')
        for key in ('potential_v','kinetic_energy_j','potential_energy_j','total_energy_j','energy_drift_j'):
            a,b=getattr(row,key),getattr(expected,key)
            if type(a) not in (int,float) or not isfinite(a) or abs(a-b)>256*ulp(max(abs(a),abs(b),1e-300)):
                raise ValueError('saved event energy/potential mismatch')
        # Separate physical contributions remain meaningful even when totals
        # cancel. Validate every exported contribution, including declared absence.
        for key in CONTRIBUTION_KEYS:
            actual, value = getattr(row,key), getattr(expected,key)
            if value is None:
                if actual is not None:
                    raise ValueError('undeclared saved event contribution: '+key)
            elif isinstance(value,tuple):
                _vector(actual,key)
                if actual != value:
                    raise ValueError('saved event contribution mismatch: '+key)
            else:
                _number(actual,key)
                if abs(actual-value)>256*ulp(max(abs(actual),abs(value),1e-300)):
                    raise ValueError('saved event contribution mismatch: '+key)
    inventory={q.event_id:q for q in run.study.questions}
    for e in run.events:
        if e.event_id not in inventory:raise ValueError('unknown saved event identity')
        _validate_event_numbers(e)
        q=inventory[e.event_id];lo,hi=e.time_bracket_s
        if e.kind!=('plane-crossing' if type(q) is PlaneEvent else 'directional-reversal') or e.residual_unit!=('m' if type(q) is PlaneEvent else 'm/s'):
            raise ValueError('event quantity or unit was changed')
        if type(e.full_stop_within_tolerance) is not bool:
            raise ValueError('full-stop classification must be Boolean')
        _validate_event_horizon(e,run.events,rows[-1].time_s,
            run.trajectory.termination,run.ordering,policy.time_tolerance_s)
        if not (0<lo<=e.sample.time_s<=hi<=spec.duration_s and hi-lo<=policy.time_tolerance_s*1.01):
            raise ValueError('invalid root time bracket')
        if type(e.direction) is not int or e.direction not in (-1,1) or q.direction and e.direction!=q.direction:
            raise ValueError('event direction mismatch')
        g=_value(q,(*e.sample.position_m,*e.sample.velocity_m_s))
        if g!=e.residual or abs(g)>(policy.position_tolerance_m if type(q) is PlaneEvent else policy.velocity_tolerance_m_s):
            raise ValueError('event residual mismatch')
        if e.terminal!=q.terminal or type(e.terminal) is not bool:raise ValueError('event terminal flag mismatch')
        if e.full_stop_within_tolerance!=(hypot(*e.sample.velocity_m_s)<=policy.velocity_tolerance_m_s):
            raise ValueError('directional reversal mislabeled total stop')
    if run.trajectory.termination=='terminal-event':
        terms=[e for e in run.events if e.terminal]
        if not terms or min(e.sample.time_s for e in terms)!=rows[-1].time_s:
            raise ValueError('terminal event and final physical state disagree')
    if run.trajectory.termination=='event-unresolved' and not run.unresolved:
        raise ValueError('unresolved termination requires recorded reason')
    return True
