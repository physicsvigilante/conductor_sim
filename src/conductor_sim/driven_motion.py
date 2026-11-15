"""M24b: typed linear drag and scheduled uniform E over retained source models.

No oil-drop force law, responding conductor, arbitrary force callbacks, velocity
reset or conservative-energy-drift fiction. Reuse the shared adaptive vector
controller and M24a locator on smooth schedule intervals; integrate work states.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from math import fsum, hypot, inf, isfinite, ulp

from .model import Vector3
from .prescribed_fields import PrescribedField, UniformElectricField, _dot, _sub, _vector, _number
from .prescribed_motion import _FixedDomain, _in_plane
from .scenarios import PrescribedScenario
from .motion_1d import MotionIntegrationError, _NONRELATIVISTIC_SPEED_LIMIT_M_S
from .motion_2d import _integrate_vector_motion, _SurfaceGuard
from .motion_events import (EventPolicy, PlaneEvent, VelocityReversal, LocatedEvent,
                            _Monitor, _identity, _native, _validate_questions, _value)
from .solver import UnsupportedProblem


def _label(value):
    if type(value) is not str or not value.isprintable() or not 1<=len(value)<=120:
        raise ValueError('label must be 1..120 printable characters')


@dataclass(frozen=True, slots=True)
class LinearDrag:
    """F=-b*v for a stationary, prescribed medium. No hidden viscosity/radius."""
    coefficient_kg_s: float
    component_id: str = 'drag:medium'
    label: str = 'Stationary-medium linear drag'

    def __post_init__(self):
        _number(self.coefficient_kg_s,'linear drag coefficient')
        if self.coefficient_kg_s<0:raise ValueError('linear drag must be nonnegative')
        _identity(self.component_id);_label(self.label)


@dataclass(frozen=True, slots=True)
class FieldPhase:
    start_s: float
    field_v_m: Vector3

    def __post_init__(self):
        _number(self.start_s,'phase start')
        if self.start_s<0:raise ValueError('phase starts must be nonnegative')
        object.__setattr__(self,'field_v_m',_vector(self.field_v_m,'applied field'))
        if not isfinite(hypot(*self.field_v_m)):raise ValueError('field magnitude overflow')


@dataclass(frozen=True, slots=True)
class UniformFieldSchedule:
    """Right-continuous, spatially uniform piecewise-constant electric input.

    A step changes force, not position/velocity or accumulated mechanical work.
    This prescribed approximation does not solve switching electromagnetic fields.
    """
    phases: tuple[FieldPhase, ...]
    source_id: str = 'field:scheduled'
    label: str = 'Scheduled applied electric field'

    def __post_init__(self):
        if type(self.phases) not in (list,tuple) or not 1<=len(self.phases)<=32:
            raise ValueError('schedule requires 1..32 typed phases')
        object.__setattr__(self,'phases',tuple(self.phases))
        if any(type(p) is not FieldPhase for p in self.phases):raise TypeError('require FieldPhase')
        if self.phases[0].start_s!=0:raise ValueError('schedule must begin at t=0')
        if any(a.start_s>=b.start_s for a,b in zip(self.phases,self.phases[1:])):
            raise ValueError('schedule times must be strictly increasing; no automatic sorting')
        _identity(self.source_id);_label(self.label)

    def phase_at(self,time_s):
        _number(time_s,'physical schedule time')
        if time_s<0:raise ValueError('physical time must be nonnegative')
        return bisect_right(tuple(p.start_s for p in self.phases),time_s)-1


@dataclass(frozen=True, slots=True)
class DrivenModel:
    """Existing physical scenario + two explicit, immutable physical additions."""
    base: PrescribedScenario
    drag: LinearDrag
    applied: UniformFieldSchedule
    model_id: str
    provenance: str
    parent_revision_id: str | None = None

    def __post_init__(self):
        if type(self.base) is not PrescribedScenario or type(self.drag) is not LinearDrag or type(self.applied) is not UniformFieldSchedule:
            raise TypeError('require PrescribedScenario, LinearDrag and UniformFieldSchedule')
        _identity(self.model_id)
        if type(self.provenance) is not str or not self.provenance.strip():raise ValueError('explicit provenance required')
        if self.parent_revision_id is not None and (type(self.parent_revision_id) is not str or len(self.parent_revision_id)!=64):
            raise ValueError('invalid parent revision')
        used={a for a,b in self.base.source_annotations}|{'test-particle','medium','gravity'}
        if self.drag.component_id in used or self.applied.source_id in used or self.drag.component_id==self.applied.source_id:
            raise ValueError('driven source/force identities must be distinct')
        spec=self.base.spec
        rate=self.drag.coefficient_kg_s/spec.mass_kg
        if not isfinite(rate) or (self.drag.coefficient_kg_s>0 and rate==0):
            raise ValueError('drag relaxation rate not representable')
        active=[p.start_s for p in self.applied.phases if p.start_s<=spec.duration_s]
        if any(b-a<=max(128*ulp(spec.duration_s),2e-14*spec.duration_s) for a,b in zip(active,active[1:])):
            raise ValueError('switch times are not resolved at the requested physical clock')
        # Each complete phase inventory is qualified by the original source/
        # plane/axis constructor. Linear isotropic drag preserves that subspace.
        for phase in self.applied.phases:
            if hypot(*phase.field_v_m):
                replace(spec,field=PrescribedField((*spec.field.sources,
                    UniformElectricField(phase.field_v_m)),permittivity_f_m=spec.field.permittivity_f_m))

    @property
    def spec(self):return self.base.spec

    @property
    def revision_id(self):
        return sha256(json.dumps(self._record(),sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()

    def _record(self):
        return dict(schema='conductor_sim.driven_model.v1',model_id=self.model_id,
            base=self.base.as_dict(),drag=asdict(self.drag),applied=asdict(self.applied),
            provenance=self.provenance,parent_revision_id=self.parent_revision_id)

    def as_dict(self):return _native(dict(**self._record(),revision_id=self.revision_id))


def revise_driven_model(parent, *, model_id, note, drag=None, applied=None, base=None):
    """Retain the parent; change physical input parameters, not query policy."""
    if type(parent) is not DrivenModel or type(note) is not str or not note.strip():
        raise ValueError('require retained DrivenModel and an explicit revision note')
    drag=parent.drag if drag is None else drag;applied=parent.applied if applied is None else applied
    base=parent.base if base is None else base
    if type(drag) is not LinearDrag or type(applied) is not UniformFieldSchedule or type(base) is not PrescribedScenario:
        raise TypeError('invalid physical revision component')
    if (drag.component_id!=parent.drag.component_id or applied.source_id!=parent.applied.source_id or
        base.source_annotations!=parent.base.source_annotations or
        tuple(type(s) for s in base.spec.field.sources)!=tuple(type(s) for s in parent.base.spec.field.sources)):
        raise ValueError('parameter revisions preserve physical source identities and kinds')
    if (drag,applied,base)==(parent.drag,parent.applied,parent.base):raise ValueError('no physical revision requested')
    if model_id==parent.model_id:raise ValueError('child model needs a distinct model identity')
    child=DrivenModel(base,drag,applied,model_id,note,parent.revision_id)
    changes={k:{'before':a,'after':b} for k,a,b in (
        ('drag',asdict(parent.drag),asdict(drag)),('applied',asdict(parent.applied),asdict(applied)),
        ('base',parent.base.as_dict(),base.as_dict())) if a!=b}
    return child,_native(dict(parent=parent.as_dict(),child=child.as_dict(),note=note,changes=changes))


@dataclass(frozen=True, slots=True)
class DrivenStudy:
    model: DrivenModel
    questions: tuple[PlaneEvent | VelocityReversal, ...]
    policy: EventPolicy
    work_absolute_tolerance_j: float = 1e-24
    work_relative_tolerance: float = 2e-8

    def __post_init__(self):
        if type(self.model) is not DrivenModel or type(self.policy) is not EventPolicy:
            raise TypeError('require DrivenModel and EventPolicy')
        for key in ('work_absolute_tolerance_j','work_relative_tolerance'):
            _number(getattr(self,key),key,positive=True)
        if self.work_relative_tolerance>=.05:raise ValueError('work tolerance must be below 0.05')
        if type(self.questions) not in (tuple,list):raise TypeError('typed question sequence required')
        if self.policy.time_tolerance_s<128*ulp(self.model.spec.duration_s):raise ValueError('unresolved event clock')
        object.__setattr__(self,'questions',tuple(self.questions))
        if self.questions:
            force=_observe(self.model,0, self.model.spec.particle.position_m,self.model.spec.initial_velocity_m_s)['force_n']
            _validate_questions(self.model.spec,self.questions,self.policy,initial_force=force)

    @property
    def scenario(self):return self.model.base  # shared locator's retained base spec

    def as_dict(self):
        return _native(dict(schema='conductor_sim.driven_study.v1',model=self.model.as_dict(),
            model_revision_id=self.model.revision_id,questions=[dict(kind=type(q).__name__,**asdict(q)) for q in self.questions],
            numerical_policy=asdict(self.policy),work_absolute_tolerance_j=self.work_absolute_tolerance_j,
            work_relative_tolerance=self.work_relative_tolerance))


def _observe(model, phase_index, position, velocity):
    spec=model.spec;domain=_FixedDomain(spec)
    if domain.clearance(position)<=0:raise _SurfaceGuard()
    if not _in_plane(position,domain.origin,domain.normal,domain.scale):
        raise UnsupportedProblem('driven particle left qualified physical plane')
    if hypot(*velocity)>=_NONRELATIVISTIC_SPEED_LIMIT_M_S:
        raise UnsupportedProblem('driven motion must remain below 0.1c')
    fixed=spec.forces.observe(position,charge_c=spec.particle.charge_c,mass_kg=spec.mass_kg)
    applied=model.applied.phases[phase_index].field_v_m
    fe=tuple(spec.particle.charge_c*x for x in applied)
    fd=tuple(-model.drag.coefficient_kg_s*x for x in velocity)
    if any(e!=0 and f==0 for e,f in zip(applied,fe)) or (model.drag.coefficient_kg_s>0 and any(v!=0 and f==0 for v,f in zip(velocity,fd))):
        raise MotionIntegrationError('nonzero driven force underflows numerical representation')
    force=tuple(fsum(x) for x in zip(fixed.electric_force_n,fe,fixed.gravitational_force_n,fd))
    scale=max(hypot(*f) for f in (fixed.electric_force_n,fe,fixed.gravitational_force_n,fd))
    if abs(_dot(force,spec.plane_normal))>5e-9*scale:
        raise UnsupportedProblem('composed driven force violates plane qualification')
    result=dict(fixed_electric_force_n=fixed.electric_force_n,applied_electric_force_n=fe,
        gravitational_force_n=fixed.gravitational_force_n,drag_force_n=fd,force_n=force,
        acceleration_m_s2=tuple(f/spec.mass_kg for f in force),
        electric_field_v_m=tuple(fsum(x) for x in zip(spec.field.electric_field_at(position),applied)),
        fixed_potential_energy_j=fixed.electric_potential_energy_j,
        gravitational_potential_energy_j=fixed.gravitational_potential_energy_j)
    if not all(isfinite(v) for x in result.values() for v in (x if isinstance(x,tuple) else (x,))):
        raise MotionIntegrationError('nonfinite driven force observation')
    return result


@dataclass(frozen=True, slots=True)
class DrivenSample:
    time_s: float
    phase_index: int
    position_m: Vector3
    velocity_m_s: Vector3
    acceleration_m_s2: Vector3
    electric_field_v_m: Vector3
    fixed_electric_force_n: Vector3
    applied_electric_force_n: Vector3
    gravitational_force_n: Vector3
    drag_force_n: Vector3
    force_n: Vector3
    fixed_potential_energy_j: float
    gravitational_potential_energy_j: float
    kinetic_energy_j: float
    fixed_work_j: float
    applied_work_j: float
    gravitational_work_j: float
    dissipated_energy_j: float
    work_balance_error_j: float

    def as_dict(self):return _native(asdict(self))


@dataclass(frozen=True, slots=True)
class SwitchObservation:
    time_s: float
    left: DrivenSample
    right: DrivenSample


class _DrivenDynamics:
    """Private typed adapter for the unchanged shared integration-loop semantics."""
    def __init__(self,study):
        self.study=study;self.model=study.model;self.spec=self.model.spec;self.phase_index=0
        self.switch_times=tuple(p.start_s for p in self.model.applied.phases[1:] if p.start_s<=self.spec.duration_s)
        rate=self.model.drag.coefficient_kg_s/self.spec.mass_kg
        self.max_step_s=inf if rate==0 else .25/rate
        self.switches=[]
        self.k0=.5*self.spec.mass_kg*hypot(*self.spec.initial_velocity_m_s)**2

    def rhs(self,state):
        p,v=state[:3],state[3:6];obs=_observe(self.model,self.phase_index,p,v)
        derivatives=(*v,*obs['acceleration_m_s2'],_dot(obs['fixed_electric_force_n'],v),
            _dot(obs['applied_electric_force_n'],v),_dot(obs['gravitational_force_n'],v),
            self.model.drag.coefficient_kg_s*hypot(*v)**2)
        if not all(isfinite(x) for x in derivatives):raise MotionIntegrationError('nonfinite driven work derivative')
        return derivatives

    def sample(self,t,state):
        p,v=tuple(state[:3]),tuple(state[3:6]);obs=_observe(self.model,self.phase_index,p,v)
        k=.5*self.spec.mass_kg*hypot(*v)**2
        wf,wa,wg,d=state[6:10]
        if d<0 or not all(isfinite(x) for x in (k,wf,wa,wg,d)):
            raise MotionIntegrationError('invalid accumulated work/dissipation')
        return DrivenSample(t,self.phase_index,p,v,**obs,kinetic_energy_j=k,
            fixed_work_j=wf,applied_work_j=wa,gravitational_work_j=wg,dissipated_energy_j=d,
            work_balance_error_j=fsum((k,-self.k0,-wf,-wa,-wg,d)))

    def work_error(self,large,fine,old):
        return max(abs(a-b)/(15*(self.study.work_absolute_tolerance_j+self.study.work_relative_tolerance*
            max(abs(b),abs(c),abs(b-c)))) for a,b,c in zip(large[6:10],fine[6:10],old[6:10]))

    def at_boundary(self,t,state):
        phase=self.model.applied.phase_at(t)
        if phase==self.phase_index:return
        if t not in self.switch_times or phase!=self.phase_index+1:
            raise MotionIntegrationError('schedule boundary was skipped')
        left=self.sample(t,state);self.phase_index=phase
        self.switches.append(SwitchObservation(t,left,self.sample(t,state)))


@dataclass(frozen=True, slots=True)
class DrivenRun:
    study: DrivenStudy
    samples: tuple[DrivenSample,...]
    events: tuple[LocatedEvent,...]
    switches: tuple[SwitchObservation,...]
    termination: str
    guard_events: tuple
    initial_conditions: tuple
    unresolved: tuple
    ordering: str
    evaluations: int
    max_abs_work_balance_error_j: float

    def as_dict(self):
        outcomes={q.event_id:('located-with-unresolved-candidates' if any(e.event_id==q.event_id for e in self.events) and any(u[0]==q.event_id for u in self.unresolved)
            else 'located' if any(e.event_id==q.event_id for e in self.events)
            else 'unresolved' if any(u[0]==q.event_id for u in self.unresolved)
            else 'not-found-at-requested-resolution' if self.termination=='duration' else 'search-truncated') for q in self.study.questions}
        return _native(dict(schema='conductor_sim.driven_run.v1',study=self.study.as_dict(),
            samples=[s.as_dict() for s in self.samples],events=[asdict(e) for e in self.events],
            switches=[asdict(s) for s in self.switches],termination=self.termination,
            guard_events=[asdict(g) for g in self.guard_events],initial_conditions=self.initial_conditions,
            unresolved=self.unresolved,ordering=self.ordering,evaluations=self.evaluations,outcomes=outcomes,
            max_abs_work_balance_error_j=self.max_abs_work_balance_error_j,
            energy_convention='Delta K = W_fixed + W_applied + W_gravity - D; D integral b|v|^2 dt >= 0. Residual is numerical, dissipation is physical.',
            qualification='Finite-resolution piecewise-smooth prescribed-field motion; no switching electromagnetic solve, exact impact or rigorous all-root guarantee. Medium stationary; field steps have no impulses.'))


def investigate_driven_motion(study: DrivenStudy) -> DrivenRun:
    """One unchanged physical model, multiple typed questions and work accounting."""
    if type(study) is not DrivenStudy:raise TypeError('require DrivenStudy')
    digest=study.model.revision_id;dynamics=_DrivenDynamics(study)
    monitor=_Monitor(study,sample_factory=dynamics.sample,work_error=dynamics.work_error,
                     switch_times=dynamics.switch_times)
    spec=study.model.spec
    def original_observation(p):
        obs=spec.forces.observe(p,charge_c=spec.particle.charge_c,mass_kg=spec.mass_kg)
        return obs.force_n,obs.energy
    rows,guards,end,residual=_integrate_vector_motion(spec,_FixedDomain(spec),original_observation,
        surface_description='Stopped before a source guard; not a collision or scheduled field change',
        step_observer=monitor,max_step_s=study.policy.max_step_s,dynamics=dynamics)
    end='source-guard' if end in ('surface-guard','charge-guard') else end
    if digest!=study.model.revision_id:raise RuntimeError('physical model mutated')
    run=DrivenRun(study,rows,tuple(monitor.events),tuple(dynamics.switches),end,guards,
        tuple(monitor.initial),tuple(monitor.unresolved),monitor.ordering,monitor.evaluations,residual)
    validate_driven_run(run)
    return run


def validate_driven_run(run):
    """Recheck typed saved physics/ledger consistency, not an independent ODE proof."""
    if type(run) is not DrivenRun or type(run.study) is not DrivenStudy:raise TypeError('require DrivenRun')
    model=run.study.model;spec=model.spec;policy=run.study.policy;rows=run.samples
    if not rows or rows[0].time_s!=0 or any(a.time_s>=b.time_s for a,b in zip(rows,rows[1:])):
        raise ValueError('driven history requires ordered saved physical times')
    if run.termination not in ('duration','source-guard','terminal-event','event-unresolved'):
        raise ValueError('unknown driven termination')
    if rows[-1].time_s>spec.duration_s or run.termination=='duration' and rows[-1].time_s!=spec.duration_s:
        raise ValueError('driven history horizon mismatch')
    if rows[0].position_m!=spec.particle.position_m or rows[0].velocity_m_s!=spec.initial_velocity_m_s:
        raise ValueError('driven initial conditions changed')
    dyn=_DrivenDynamics(run.study)
    def verify(row,*,left=False):
        if type(row) is not DrivenSample:raise ValueError('invalid driven sample')
        _number(row.time_s,'saved time')
        if not 0<=row.time_s<=rows[-1].time_s:raise ValueError('sample outside observed interval')
        if type(row.phase_index) is not int:raise ValueError('phase index must be integer')
        expected=model.applied.phase_at(row.time_s)-(1 if left else 0)
        if row.phase_index!=expected:raise ValueError('saved phase contradicts schedule')
        _vector(row.position_m,'saved position');_vector(row.velocity_m_s,'saved velocity')
        work=(row.fixed_work_j,row.applied_work_j,row.gravitational_work_j,row.dissipated_energy_j)
        for v in work:_number(v,'saved work')
        if row.dissipated_energy_j<0:raise ValueError('negative dissipation')
        dyn.phase_index=row.phase_index;actual=dyn.sample(row.time_s,(*row.position_m,*row.velocity_m_s,*work))
        for key,val in asdict(actual).items():
            old=getattr(row,key)
            if isinstance(val,tuple):
                _vector(old,key)
                if old!=val:raise ValueError('saved driven vector mismatch: '+key)
            else:
                _number(old,key)
                if old!=val:raise ValueError('saved driven scalar mismatch: '+key)
    for row in rows:verify(row)
    if any(getattr(rows[0],k)!=0 for k in ('fixed_work_j','applied_work_j','gravitational_work_j','dissipated_energy_j')):
        raise ValueError('initial work ledger must be zero')
    if any(b.dissipated_energy_j<a.dissipated_energy_j for a,b in zip(rows,rows[1:])):
        raise ValueError('dissipation decreased')
    expected_switches=tuple(p.start_s for p in model.applied.phases[1:] if p.start_s<=rows[-1].time_s)
    if tuple(s.time_s for s in run.switches)!=expected_switches:raise ValueError('switch observations missing or reordered')
    for switch in run.switches:
        verify(switch.left,left=True);verify(switch.right)
        if switch.left.time_s!=switch.time_s or switch.right.time_s!=switch.time_s:raise ValueError('switch clock mismatch')
        for key in ('position_m','velocity_m_s','fixed_work_j','applied_work_j','gravitational_work_j','dissipated_energy_j'):
            if getattr(switch.left,key)!=getattr(switch.right,key):raise ValueError('state or work jumped at finite field step')
        if not any(row==switch.right for row in rows):raise ValueError('right switch state not retained')
    questions={q.event_id:q for q in run.study.questions}
    for e in run.events:
        verify(e.sample)
        if e.event_id not in questions:raise ValueError('unrequested driven event')
        q=questions[e.event_id];lo,hi=e.time_bracket_s
        if not 0<lo<=e.sample.time_s<=hi<=spec.duration_s or hi-lo>1.01*policy.time_tolerance_s:
            raise ValueError('invalid driven event bracket')
        if any(abs(e.sample.time_s-t)<=policy.time_tolerance_s for t in expected_switches):
            raise ValueError('event/switch ambiguity mislabelled resolved')
        if type(e.direction) is not int or e.direction not in (-1,1) or q.direction and e.direction!=q.direction:
            raise ValueError('event direction mismatch')
        if type(e.terminal) is not bool or e.terminal!=q.terminal:raise ValueError('terminal flag mismatch')
        if e.kind!=('plane-crossing' if type(q) is PlaneEvent else 'directional-reversal') or e.residual_unit!=('m' if type(q) is PlaneEvent else 'm/s'):
            raise ValueError('event kind/unit mismatch')
        value=_value(q,(*e.sample.position_m,*e.sample.velocity_m_s))
        if value!=e.residual or abs(value)>(policy.position_tolerance_m if type(q) is PlaneEvent else policy.velocity_tolerance_m_s):
            raise ValueError('driven event residual mismatch')
        if type(e.full_stop_within_tolerance) is not bool or e.full_stop_within_tolerance!=(hypot(*e.sample.velocity_m_s)<=policy.velocity_tolerance_m_s):
            raise ValueError('false full-stop classification')
    if run.termination=='terminal-event' and (not any(e.terminal for e in run.events) or min(e.sample.time_s for e in run.events if e.terminal)!=rows[-1].time_s):
        raise ValueError('terminal event/final time mismatch')
    if run.termination=='event-unresolved' and not run.unresolved:raise ValueError('missing unresolved reason')
    if run.termination=='source-guard' and not run.guard_events:raise ValueError('missing guard reason')
    if run.max_abs_work_balance_error_j!=max(abs(s.work_balance_error_j) for s in rows):
        raise ValueError('work-balance residual summary mismatch')
    return True
