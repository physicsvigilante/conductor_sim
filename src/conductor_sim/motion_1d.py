"""M16a: genuine 1D electrostatic dynamics driven by analytical force/energy.

A symmetry-preserving line is *proved from the authored geometry* before
integration. Each RHS evaluation resolves the existing electrostatic backend
and excludes self-force via force_energy; no new force law or FEMM fallback.
The time integrator is an adaptive RK4 step-doubling scheme. A surface guard
ends the solution before singular ideal point/conductor contact. It does not
claim a finite-radius particle, impact dynamics, or an exact collision time.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import ceil, fsum, hypot, isfinite
from typing import Literal

from ._resources import cleanup_resources
from ._motion_rk4 import rk4_step
from .model import ConductingPlane, ElectrostaticProblem, PointCharge, SolidSphere, SphericalShell, Vector3, _position
from .observables import force_energy
from .solver import UnsupportedProblem


def _dot(a, b):
    return fsum(x*y for x, y in zip(a, b))


def _minus(a, b):
    return tuple(x-y for x,y in zip(a,b))


def _norm(v):
    return hypot(*v)


def _add(a, b, factor):
    return tuple(a[i]+factor*b[i] for i in range(3))


_NONRELATIVISTIC_SPEED_LIMIT_M_S = .1 * 299_792_458.0


class MotionIntegrationError(RuntimeError):
    """The finite-step integration could not meet the requested accuracy."""


@dataclass(frozen=True, slots=True)
class Motion1DSpec:
    """One authored charged particle, constrained by *physical symmetry* to 1D.

    ``axis`` defines the sign of the scalar initial velocity; it is **not**
    an artificial guide or constraint force. The remaining charges are not
    currently supported in M16a. ``sample_interval_s`` is a requested output
    time spacing, not necessarily the internal solver timestep.
    """
    problem: ElectrostaticProblem
    charge_index: int
    mass_kg: float
    axis: Vector3
    initial_velocity_m_s: float
    duration_s: float
    sample_interval_s: float
    relative_tolerance: float = 2e-7
    surface_guard_m: float | None = None
    max_internal_steps: int = 100000

    def __post_init__(self):
        if not isinstance(self.problem, ElectrostaticProblem):
            raise TypeError('problem must be ElectrostaticProblem')
        if isinstance(self.charge_index, bool) or not isinstance(self.charge_index, int):
            raise TypeError('charge_index must be an integer')
        if not 0 <= self.charge_index < len(self.problem.charges):
            raise IndexError('charge index outside authored inventory')
        if len(self.problem.charges) != 1:
            raise UnsupportedProblem('M16a supports a single moving point charge only')
        if not self.problem.charges[0].charge_c:
            raise UnsupportedProblem('M16a requires a nonzero moving charge')
        for name in ('mass_kg', 'duration_s', 'sample_interval_s', 'relative_tolerance'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float,int)) or not isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be a finite positive number')
        if self.relative_tolerance >= .05:
            raise ValueError('relative_tolerance must be smaller than 0.05')
        if not isinstance(self.initial_velocity_m_s, (float,int)) or isinstance(self.initial_velocity_m_s, bool) or not isfinite(self.initial_velocity_m_s):
            raise ValueError('initial_velocity_m_s must be finite')
        if abs(self.initial_velocity_m_s) >= _NONRELATIVISTIC_SPEED_LIMIT_M_S:
            raise UnsupportedProblem('M16a is nonrelativistic; initial speed must be below 0.1c')
        if isinstance(self.max_internal_steps, bool) or not isinstance(self.max_internal_steps,int) or self.max_internal_steps < 10:
            raise ValueError('max_internal_steps must be an integer >= 10')
        if self.sample_interval_s > self.duration_s:
            raise ValueError('sample_interval_s must not exceed duration_s')
        if ceil(self.duration_s/self.sample_interval_s) > 2000:
            raise ValueError('at most 2000 requested output intervals')
        if self.surface_guard_m is not None and (isinstance(self.surface_guard_m, bool) or
                                                 not isinstance(self.surface_guard_m, (int, float)) or
                                                 not isfinite(self.surface_guard_m) or self.surface_guard_m <= 0):
            raise ValueError('surface_guard_m must be finite and positive')
        v=_position(self.axis,'motion axis')
        length=_norm(v)
        if length == 0 or not isfinite(length):
            raise ValueError('motion axis must be nonzero')
        object.__setattr__(self,'axis',tuple(x/length for x in v))
        _geometry(self)  # Physically prove invariance and a nonempty vacuum interval.

    @property
    def selected_charge(self):
        return self.problem.charges[self.charge_index]


@dataclass(frozen=True, slots=True)
class MotionSample:
    time_s: float
    displacement_m: float
    position_m: Vector3
    velocity_m_s: float
    acceleration_m_s2: float
    force_n: float
    kinetic_energy_j: float
    interaction_energy_j: float
    total_energy_j: float
    energy_drift_j: float


@dataclass(frozen=True, slots=True)
class MotionEvent:
    kind: Literal['turning-point', 'surface-guard']
    time_s: float
    displacement_m: float
    position_m: Vector3
    description: str


@dataclass(frozen=True, slots=True)
class Motion1DResult:
    spec: Motion1DSpec
    samples: tuple[MotionSample, ...]
    events: tuple[MotionEvent, ...]
    termination: Literal['duration', 'surface-guard']
    max_abs_energy_drift_j: float
    method: str = 'Adaptive RK4 step doubling; existing analytical force_energy at each RHS evaluation'

    def as_dict(self):
        """Finite, directly serializable SI trajectory data; not an HTML animation."""
        return dict(schema='conductor_sim.motion_1d.v1',
                    mode='elapsed-time classical dynamics',
                    units={'time':'s','displacement':'m','velocity':'m/s',
                           'acceleration':'m/s^2','force':'N','energy':'J'},
                    mass_kg=self.spec.mass_kg,
                    charge_c=self.spec.selected_charge.charge_c,
                    initial_position_m=self.spec.selected_charge.position_m,
                    axis=self.spec.axis,
                    termination=self.termination,
                    method=self.method,
                    guard_m=_geometry(self.spec).guard,
                    max_abs_energy_drift_j=self.max_abs_energy_drift_j,
                    samples=[asdict(x) for x in self.samples],
                    events=[asdict(x) for x in self.events])


@dataclass(frozen=True, slots=True)
class _Domain:
    lower: float
    upper: float
    scale: float
    guard: float

    def safe(self, displacement):
        return self.lower+self.guard < displacement < self.upper-self.guard

    def clearance(self, displacement):
        return min(displacement-self.lower, self.upper-displacement)-self.guard


def _geometry(spec: Motion1DSpec) -> _Domain:
    """Find the connected vacuum interval containing the initial position."""
    problem=spec.problem
    x=problem.charges[spec.charge_index].position_m
    axis=spec.axis
    conductors=problem.conductors
    if not conductors:
        raise UnsupportedProblem('M16a requires authored conductors and an analytical force backend')
    lower,upper=-float('inf'),float('inf')
    scale=0.
    if all(isinstance(c,ConductingPlane) for c in conductors):
        if not 1<=len(conductors)<=2:
            raise UnsupportedProblem('M16a supports one plane or two facing planes')
        for c in conductors:
            cos=_dot(axis,c.normal)
            if abs(abs(cos)-1)>1e-10:
                raise UnsupportedProblem('1D plane motion must follow the physical plane normal')
            height=c.signed_distance_m(x)
            if height<=0:
                raise UnsupportedProblem('moving source begins in conducting metal')
            bound=-height/cos
            if cos>0:lower=max(lower,bound)
            else:upper=min(upper,bound)
            scale=max(scale,height)
        if len(conductors)==2 and (not isfinite(lower) or not isfinite(upper)):
            raise UnsupportedProblem('two conducting planes must face one another')
    elif all(isinstance(c,SphericalShell) for c in conductors):
        # Strictly central cavity only, and every spherical conductor concentric.
        center=conductors[0].center_m
        if any(_norm(_minus(c.center_m,center))>1e-10*c.inner_radius_m for c in conductors):
            raise UnsupportedProblem('1D cavity motion requires concentric spherical shells')
        radius=min(c.inner_radius_m for c in conductors)
        delta=_minus(x,center)
        axial=_dot(delta,axis)
        if _norm(_minus(delta,tuple(axial*u for u in axis)))>1e-10*radius:
            raise UnsupportedProblem('cavity motion must lie on a line through the cavity centre')
        if _norm(delta)>=radius:
            raise UnsupportedProblem('charge must begin strictly inside innermost cavity')
        lower,upper=-radius-axial,radius-axial
        scale=radius
    elif len(conductors)==1 and isinstance(conductors[0],SolidSphere):
        c=conductors[0]
        delta=_minus(x,c.center_m)
        axial=_dot(delta,axis)
        if _norm(_minus(delta,tuple(axial*u for u in axis)))>1e-10*c.radius_m:
            raise UnsupportedProblem('exterior sphere motion must be radial')
        if _norm(delta)<=c.radius_m:
            raise UnsupportedProblem('moving charge must start outside solid sphere')
        if axial>0:lower=c.radius_m-axial
        else:upper=-c.radius_m-axial
        scale=c.radius_m
    else:
        raise UnsupportedProblem('M16a supports normal plane motion or radial single-sphere/cavity motion only')
    if upper<=lower:
        raise UnsupportedProblem('no connected vacuum interval along authored axis')
    # A finite numerical event guard also prevents evaluation on singular ideal surfaces.
    guard=(1e-6*scale if spec.surface_guard_m is None else spec.surface_guard_m)
    if not (lower+guard < 0. < upper-guard):
        raise ValueError('initial charge position must be strictly outside the surface guard')
    return _Domain(lower,upper,scale,guard)


def integrate_motion_1d(spec: Motion1DSpec, backend) -> Motion1DResult:
    """Integrate autonomous 1D classical motion, never prescribing a path.

    Turning points are detected, but motion continues. If the ideal point
    comes within the declared surface guard, stop and report an **approach**
    event, not a mathematically exact surface collision or reflected motion.
    """
    if not isinstance(spec,Motion1DSpec):
        raise TypeError('spec must be Motion1DSpec')
    domain=_geometry(spec)
    original=spec.selected_charge

    def observe(s):
        if not domain.safe(s):
            raise _SurfaceGuard()
        position=_add(original.position_m,spec.axis,s)
        charges=list(spec.problem.charges)
        charges[spec.charge_index]=PointCharge(original.charge_c,position)
        problem=replace(spec.problem,charges=tuple(charges))
        supports=getattr(backend,'supports',None)
        if callable(supports) and not supports(problem):
            raise UnsupportedProblem('selected analytical backend cannot solve a dynamic state')
        solution=backend.solve(problem)
        try:
            observations=force_energy(solution)
            if len(observations.charges)!=1 or (observations.charges[0].source.position_m!=position or
                                              observations.charges[0].source.charge_c!=original.charge_c):
                raise UnsupportedProblem('the moving authored charge identity became ambiguous')
            force=observations.charges[0].total_force_n
            along=_dot(force,spec.axis)
            transverse=_norm(_minus(force,tuple(along*u for u in spec.axis)))
            if transverse>max(1e-20,5e-9*_norm(force)):
                raise UnsupportedProblem('transverse electrostatic force breaks one-dimensional motion')
            if not isfinite(along) or not isfinite(observations.energy.total_j):
                raise MotionIntegrationError('nonfinite electrostatic force or energy')
            result=(along,observations.energy)
        except BaseException as exc:
            close=getattr(solution,'close',None)
            if callable(close): cleanup_resources((('motion solution',close),),primary=exc)
            raise
        else:
            close=getattr(solution,'close',None)
            if callable(close):close()
        return result

    def a(s):
        return observe(s)[0]/spec.mass_kg

    def rk4(s,v,h):
        # Reuse the same dimension-independent stepper as M17a, keeping
        # M16a's existing controller, surface guard and event handling.
        return rk4_step((s, v), h, lambda state: (state[1], a(state[0])))

    s,v,t=0.,float(spec.initial_velocity_m_s),0.
    _,reference_energy=observe(s)
    ref=reference_energy
    energy0=.5*spec.mass_kg*v*v
    samples=[]
    events=[]
    max_drift=0.
    def record(t,s,v):
        nonlocal max_drift
        force,energy=observe(s)
        kinetic=.5*spec.mass_kg*v*v
        drift=fsum((kinetic-energy0,
                     energy.direct_j-ref.direct_j,
                     energy.induced_j-ref.induced_j,
                     energy.background_j-ref.background_j))
        max_drift=max(max_drift,abs(drift))
        samples.append(MotionSample(t,s,_add(original.position_m,spec.axis,s),v,
                                    force/spec.mass_kg,force,kinetic,
                                    energy.total_j,kinetic+energy.total_j,drift))
    record(t,s,v)
    termination='duration'
    # Preserve output timestamps regardless of adaptive internal step sizes.
    steps_done=0
    h=spec.sample_interval_s/4
    targets=[min(spec.duration_s,i*spec.sample_interval_s)
             for i in range(1,ceil(spec.duration_s/spec.sample_interval_s)+1)]
    for target in targets:
        while t < target-2e-15*spec.duration_s:
            if steps_done>=spec.max_internal_steps:
                raise MotionIntegrationError('maximum internal substeps reached before the requested time')
            steps_done+=1
            distance=domain.clearance(s)
            closeness=max(domain.guard*.02,1e-10*domain.scale)
            if distance<=closeness:
                termination='surface-guard'
                events.append(MotionEvent('surface-guard',t,s,_add(original.position_m,spec.axis,s),
                             'Reached the numerical approach cutoff; ideal conductor contact is not resolved'))
                break
            dt=min(h,target-t)
            # Conservative prediction limits unbracketed penetration during RK stages.
            acceleration=a(s)
            speed_limit=abs(v)+abs(acceleration)*dt
            if speed_limit:
                dt=min(dt,.75*distance/speed_limit)
            min_dt=max(1e-14*spec.duration_s,1e-300)
            if dt<min_dt:
                if distance<=max(2*closeness,1e-8*domain.scale):
                    termination='surface-guard'
                    events.append(MotionEvent('surface-guard',t,s,_add(original.position_m,spec.axis,s),
                                 'Approach to singular conducting surface within integration cutoff'))
                    break
                raise MotionIntegrationError('timestep collapsed before a conductor approach event')
            try:
                large=rk4(s,v,dt)
                half=rk4(s,v,dt*.5)
                fine=rk4(half[0],half[1],dt*.5)
            except _SurfaceGuard:
                h=dt*.5
                continue
            if not domain.safe(large[0]) or not domain.safe(fine[0]):
                h=dt*.5
                continue
            if abs(fine[1]) >= _NONRELATIVISTIC_SPEED_LIMIT_M_S:
                raise UnsupportedProblem('M16a nonrelativistic speed limit reached; use a relativistic dynamics model')
            position_scale=max(1e-11*domain.scale,spec.relative_tolerance*max(domain.scale,abs(s),abs(fine[0])))
            velocity_scale=max(1e-10,spec.relative_tolerance*max(1.,abs(v),abs(fine[1])))
            error=max(abs(large[0]-fine[0])/(15*position_scale),
                      abs(large[1]-fine[1])/(15*velocity_scale))
            if error>1:
                h=max(dt*.15,dt*min(.8,.9*error**(-.2)))
                continue
            if v*fine[1]<0:
                # RK4 bisection localizes v=0 (turning), without treating it as
                # a collision or terminating the trajectory.
                lo,hi=0.,dt
                for _ in range(35):
                    mid=.5*(lo+hi)
                    vmid=rk4(s,v,mid)[1]
                    if v*vmid>0:lo=mid
                    else:hi=mid
                tau=.5*(lo+hi)
                turn_s=rk4(s,v,tau)[0]
                events.append(MotionEvent('turning-point',t+tau,turn_s,
                             _add(original.position_m,spec.axis,turn_s),
                             'Velocity reverses under the electrostatic force'))
            t+=dt;s,v=fine
            h=min(spec.sample_interval_s,max(dt*.2,dt*min(2.5,(.9/max(error,1e-10))**.2)))
        if termination=='surface-guard':
            if t>samples[-1].time_s+2e-15*spec.duration_s:record(t,s,v)
            break
        t=target
        record(t,s,v)
    return Motion1DResult(spec,tuple(samples),tuple(events),termination,max_drift)


class _SurfaceGuard(Exception):
    pass
