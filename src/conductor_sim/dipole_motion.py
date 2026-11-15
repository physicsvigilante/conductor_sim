"""Fixed-axis, ideal-hinge finite-dipole dynamics in prescribed static fields.

Reuse the M23a finite-endpoint observer and the common RK4 step/time grid.
No free rigid body, hidden inertia, source response, damping or gravity model.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import fsum, hypot, inf, isfinite, ulp

from ._motion_rk4 import rk4_step
from ._motion_time import output_times
from .dipoles import DipoleModel, observe_dipole, orient_dipole, _cross, _native
from .motion_1d import MotionIntegrationError, _NONRELATIVISTIC_SPEED_LIMIT_M_S
from .prescribed_fields import (_dot, _number, _sub, _unit,
    UniformRingAxisField, UniformDiskAxisField)
from .solver import UnsupportedProblem

_MAX_STEP_ANGLE = .05  # numerical bound, never a mechanical angular stop


@dataclass(frozen=True, slots=True, kw_only=True)
class DipoleMotionSpec:
    """I is about the ACTUAL fixed axis through model.dipole.pivot_m.

    The ideal bearing can react net force and transverse torque without work.
    Its detailed reactions are not inferred without the full mass distribution.
    theta is unwrapped, relative to the model's authored reference pose.
    """
    model: DipoleModel
    rotation_axis: tuple
    inertia_kg_m2: float
    initial_angle_rad: float
    initial_angular_velocity_rad_s: float
    duration_s: float
    sample_interval_s: float
    source_guard_m: float = 1e-5
    relative_tolerance: float = 2e-10
    max_internal_steps: int = 100000

    def __post_init__(self):
        if type(self.model) is not DipoleModel:
            raise TypeError('expected a finite DipoleModel')
        if self.model.dipole.pivot_m is None:
            raise UnsupportedProblem('dynamics requires an explicitly fixed physical pivot')
        object.__setattr__(self, 'rotation_axis', _unit(self.rotation_axis, 'fixed rotation axis'))
        for name in ('inertia_kg_m2', 'duration_s', 'sample_interval_s', 'source_guard_m'):
            object.__setattr__(self, name, _number(getattr(self, name), name, positive=True))
        for name in ('initial_angle_rad', 'initial_angular_velocity_rad_s'):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        tol = _number(self.relative_tolerance, 'relative tolerance', positive=True)
        if not 1e-12 <= tol <= 1e-4:
            raise ValueError('angular relative tolerance must be in 1e-12..1e-4')
        if type(self.max_internal_steps) is not int or not 1 <= self.max_internal_steps <= 500000:
            raise ValueError('max_internal_steps must be an integer in 1..500000')
        output_times(self.duration_s, self.sample_interval_s)
        if any(type(s.source) in (UniformRingAxisField, UniformDiskAxisField) for s in self.model.sources):
            raise UnsupportedProblem('axis-only ring/disc laws do not qualify a rotational sweep')
        if self.max_endpoint_radius_m <= 1e-12*self.model.dipole.separation_m:
            raise UnsupportedProblem('rotation axis leaves both dipole endpoints fixed')
        _resolved_angle(self.initial_angle_rad)
        _speed_check(self, self.initial_angular_velocity_rad_s)
        try:
            _state(self, self.initial_angle_rad)  # actual endpoint fields and guards
        except _SourceGuard as exc:
            raise ValueError('initial dipole endpoint is within the source/jump guard') from exc

    @property
    def max_endpoint_radius_m(self):
        d = self.model.dipole
        return max(hypot(*_cross(_sub(p, d.pivot_m), self.rotation_axis))
                   for p in d.endpoints_m)

    def as_dict(self):
        return _native(dict(schema='conductor_sim.dipole_motion_spec.v1',
            model=self.model.as_dict(), model_digest=self.model.digest,
            rotation_axis=self.rotation_axis, inertia_kg_m2=self.inertia_kg_m2,
            initial_angle_rad=self.initial_angle_rad,
            initial_angular_velocity_rad_s=self.initial_angular_velocity_rad_s,
            duration_s=self.duration_s, sample_interval_s=self.sample_interval_s,
            source_guard_m=self.source_guard_m, relative_tolerance=self.relative_tolerance,
            max_internal_steps=self.max_internal_steps,
            constraint='Ideal fixed frictionless axis through explicit pivot; no free translation',
            energy_convention='K=I*omega^2/2 + external finite-pair U; constant binding excluded'))


@dataclass(frozen=True, slots=True)
class AngularSample:
    time_s: float
    angle_rad: float
    angular_velocity_rad_s: float
    angular_acceleration_rad_s2: float
    axis_torque_n_m: float
    kinetic_energy_j: float
    interaction_energy_j: float
    total_energy_j: float
    energy_drift_j: float
    source_clearance_m: float | None


@dataclass(frozen=True, slots=True)
class DipoleMotionResult:
    spec: DipoleMotionSpec
    samples: tuple[AngularSample, ...]
    termination: str
    internal_steps: int

    def as_dict(self):
        from dataclasses import asdict
        records = []
        for row in self.samples:
            observation, _ = _state(self.spec, row.angle_rad)
            records.append(dict(**asdict(row), observation=observation))
        return _native(dict(schema='conductor_sim.dipole_motion.v1',
            spec=self.spec.as_dict(), mode='elapsed-time fixed-axis finite-dipole dynamics',
            samples=records, termination=self.termination, internal_steps=self.internal_steps,
            turning_brackets=[dict(start_s=a.time_s, end_s=b.time_s,
                interpretation='Saved angular-velocity sign change; not an exact turning event')
                for a,b in zip(self.samples,self.samples[1:])
                if a.angular_velocity_rad_s*b.angular_velocity_rad_s < 0.],
            max_abs_energy_drift_j=max(abs(r.energy_drift_j) for r in self.samples),
            method='Shared RK4 step doubling with angular error control and swept-angle source guards',
            event_note=('Requested duration completed' if self.termination == 'duration' else
                        'Stopped outside numerical source guard; no exact contact time or impact law'),
            hinge_note='Only axial torque drives the constrained DOF; full external vectors remain recorded',
            times='Saved physical times only; no interpolated or browser-integrated dynamics'))


class _SourceGuard(Exception):
    pass


def _resolved_angle(a):
    if not isfinite(a) or ulp(a) > 1e-8:
        raise MotionIntegrationError('unwrapped angle is numerically unresolved')


def _speed_check(spec, omega):
    speed = abs(omega)*spec.max_endpoint_radius_m
    if not isfinite(speed) or speed >= _NONRELATIVISTIC_SPEED_LIMIT_M_S:
        raise UnsupportedProblem('nonrelativistic endpoint speed must remain below 0.1c')


def _pose(spec, angle):
    _resolved_angle(angle)
    return orient_dipole(spec.model, angle, rotation_axis=spec.rotation_axis)


def _clearance(spec, angle):
    # Each built-in clearance is distance to its singular set or field jump,
    # hence 1-Lipschitz. The volume sphere is transparent and smooth in E.
    pose = _pose(spec, angle)
    return min((s.source.clearance_at(p) for s in spec.model.sources
                for p in pose.dipole.endpoints_m), default=inf)


def _state(spec, angle):
    c = _clearance(spec, angle)
    if c <= spec.source_guard_m:
        raise _SourceGuard('dipole endpoint is within the source/jump guard')
    o = observe_dipole(_pose(spec, angle))
    if o['torque_n_m'] is None:
        raise UnsupportedProblem('angular dynamics requires defined endpoint fields, not just finite potential')
    return o, c


def _swept_safe(spec, low, high):
    """Conservative clearance certificate for the WHOLE angular interval.

    Midpoint clearance minus radius*half-angle bounds all endpoint clearances.
    Subdivision never confuses a circle arc with a straight endpoint chord.
    Failure means unknown/unsafe, never permission to cross.
    """
    if not spec.model.sources:
        return True
    stack = [(min(low, high), max(low, high), 0)]
    visits = 0
    while stack:
        a, b, depth = stack.pop()
        visits += 1
        mid = a + (b-a)/2
        c = _clearance(spec, mid)
        if c <= spec.source_guard_m:
            return False
        if c - spec.max_endpoint_radius_m*(b-a)/2 > spec.source_guard_m:
            continue
        if depth >= 35 or visits >= 8192 or mid in (a, b):
            return False
        stack.extend(((a, mid, depth+1), (mid, b, depth+1)))
    return True


def integrate_dipole_motion(spec: DipoleMotionSpec) -> DipoleMotionResult:
    """Integrate theta, omega; never turn a static sample slider into motion."""
    if type(spec) is not DipoleMotionSpec:
        raise TypeError('expected DipoleMotionSpec')
    y = (spec.initial_angle_rad, spec.initial_angular_velocity_rad_s)
    initial, _ = _state(spec, y[0])
    k0 = .5*spec.inertia_kg_m2*y[1]**2
    records = []
    steps = 0
    termination = 'duration'
    t = 0.
    h = spec.sample_interval_s/4
    radius = spec.max_endpoint_radius_m
    near = max(.02*spec.source_guard_m, 1e-12*radius)

    def record():
        obs, c = _state(spec, y[0])
        tau = _dot(obs['torque_n_m'], spec.rotation_axis)
        k = .5*spec.inertia_kg_m2*y[1]**2
        # Differences per independent source, not a subtraction of large totals.
        du = [a['interaction_energy_j']-b['interaction_energy_j'] for a,b in
              zip(obs['source_contributions'], initial['source_contributions'])]
        drift = fsum((k-k0, *du))
        vals = (t, *y, tau/spec.inertia_kg_m2, tau, k, obs['interaction_energy_j'],
                k+obs['interaction_energy_j'], drift)
        if not all(isfinite(v) for v in vals):
            raise MotionIntegrationError('nonfinite angular history or energy')
        records.append(AngularSample(*vals, c if isfinite(c) else None))

    record()
    for target in output_times(spec.duration_s, spec.sample_interval_s):
        # Roundoff-sized residual intervals cannot support an RK step. This
        # is clock rounding below eight ulps, not physical interpolation.
        while t < target-8*ulp(target):
            if steps >= spec.max_internal_steps:
                raise MotionIntegrationError('maximum angular internal-step budget exceeded')
            steps += 1
            obs, c = _state(spec, y[0])
            margin = c-spec.source_guard_m
            if margin <= near:
                termination = 'source-guard'
                break
            acc = _dot(obs['torque_n_m'], spec.rotation_axis)/spec.inertia_kg_m2
            dt = min(h, target-t)
            rate = abs(y[1]) + abs(acc)*dt
            if rate:
                dt = min(dt, _MAX_STEP_ANGLE/rate, .4*margin/(radius*rate))
            if dt <= max(1e-14*spec.duration_s, 16*ulp(t)):
                if margin <= 2*near:
                    termination = 'source-guard'
                    break
                raise MotionIntegrationError('angular timestep collapsed away from source guard')
            seen = [y]
            def rhs(z):
                seen.append(z)
                _speed_check(spec, z[1])
                o, _ = _state(spec, z[0])
                return (z[1], _dot(o['torque_n_m'], spec.rotation_axis)/spec.inertia_kg_m2)
            try:
                large = rk4_step(y, dt, rhs)
                half = rk4_step(y, dt/2, rhs)
                fine = rk4_step(half, dt/2, rhs)
                seen.extend((large, half, fine))
                if not all(isfinite(v) for z in seen for v in z):
                    raise MotionIntegrationError('nonfinite angular RK stage')
                _speed_check(spec, fine[1])
                # Conservative stage-speed envelope covers hidden reversals as
                # well as ordinary sweeps. It is a guard for the numerical step,
                # not a rigorous global existence/error proof for the exact ODE.
                travel = 1.25*dt*max(abs(z[1]) for z in seen)
                low = min(*(z[0] for z in seen), y[0]-travel)
                high = max(*(z[0] for z in seen), y[0]+travel)
                if high-low > .2 or not _swept_safe(spec, low, high):
                    h = dt/2
                    continue
            except _SourceGuard:
                h = dt/2
                continue
            # Physical orientation is periodic: accumulating revolutions must
            # not silently loosen the angle error tolerance.
            scales = (max(1e-12, spec.relative_tolerance),
                      max(1e-12, spec.relative_tolerance*max(1.,abs(y[1]),abs(fine[1]))))
            error = max(abs(a-b)/(15*s) for a,b,s in zip(large, fine, scales))
            if not isfinite(error):
                raise MotionIntegrationError('nonfinite angular RK error')
            if error > 1.:
                h = dt*max(.1, min(.8,.9*error**(-.2)))
                continue
            y = fine
            t = target if dt == target-t else t+dt
            h = min(spec.sample_interval_s, dt*min(2.5, .9/max(error,1e-10)**.2))
        if termination != 'duration':
            if t > records[-1].time_s:
                record()
            break
        t = target
        record()
    return DipoleMotionResult(spec, tuple(records), termination, steps)


def revise_dipole_motion(parent, *, model=None, note, **changes):
    """Requalify a retained physical motion request with genuine IC/inertia edits."""
    if type(parent) is not DipoleMotionSpec or type(note) is not str or not note.strip():
        raise ValueError('require a retained motion spec and explicit revision note')
    allowed = {'initial_angle_rad', 'initial_angular_velocity_rad_s', 'inertia_kg_m2'}
    if set(changes)-allowed:
        raise ValueError('this revision edits initial angular state or inertia; author a new request for other policies')
    child = replace(parent, model=parent.model if model is None else model, **changes)
    before, after = parent.as_dict(), child.as_dict()
    edits = [dict(property=k, before=before[k], after=after[k])
             for k in sorted(allowed) if before[k] != after[k]]
    if before['model_digest'] != after['model_digest']:
        edits.append(dict(property='physical_model', before=before['model_digest'], after=after['model_digest']))
    return child, dict(schema='conductor_sim.dipole_motion_revision.v1', note=note,
                       parent=before, child=after, actual_changes=edits)


def validate_dipole_motion(result):
    """Validate typed saved observations; no arbitrary JSON trajectory import.

    Rechecks time/sample policy, initial state, rigidity/field observations,
    kinetic energy, axis torque, acceleration and energy accounting. This is
    consistency validation, not independent integration of a foreign history.
    """
    if type(result) is not DipoleMotionResult or type(result.spec) is not DipoleMotionSpec:
        raise TypeError('require a computed DipoleMotionResult')
    spec, rows = result.spec, result.samples
    if type(rows) is not tuple or not 1 <= len(rows) <= 2002 or any(type(x) is not AngularSample for x in rows):
        raise ValueError('invalid angular sample inventory')
    if result.termination not in ('duration', 'source-guard'):
        raise ValueError('invalid angular termination')
    if type(result.internal_steps) is not int or not 0 <= result.internal_steps <= spec.max_internal_steps:
        raise ValueError('invalid angular step count')
    requested = (0., *output_times(spec.duration_s,spec.sample_interval_s))
    times = tuple(r.time_s for r in rows)
    if times[0] != 0. or any(type(t) not in (int,float) or not isfinite(t) for t in times):
        raise ValueError('invalid physical clock')
    if any(b <= a for a,b in zip(times,times[1:])) or times[-1]>spec.duration_s:
        raise ValueError('nonincreasing angular clock')
    if result.termination == 'duration':
        if times != requested:
            raise ValueError('completed angular history must contain the requested physical time grid')
    elif not (times[:-1] == requested[:len(times)-1] and times[-1] < spec.duration_s):
        raise ValueError('guard history must end at its real truncated clock')
    if rows[0].angle_rad != spec.initial_angle_rad or rows[0].angular_velocity_rad_s != spec.initial_angular_velocity_rad_s:
        raise ValueError('saved angular initial state changed')
    initial,_ = _state(spec,spec.initial_angle_rad)
    k0 = .5*spec.inertia_kg_m2*spec.initial_angular_velocity_rad_s**2
    def same(a,b):
        return type(a) in (int,float) and isfinite(a) and abs(a-b)<=max(64*ulp(b),2e-10*abs(b),1e-300)
    for r in rows:
        from dataclasses import fields
        if any(type(getattr(r,f.name)) not in (int,float) or not isfinite(getattr(r,f.name))
               for f in fields(AngularSample) if f.name != 'source_clearance_m'):
            raise ValueError('angular samples require finite real scalars, not Boolean flags')
        obs,c = _state(spec,r.angle_rad)
        _speed_check(spec,r.angular_velocity_rad_s)
        k = .5*spec.inertia_kg_m2*r.angular_velocity_rad_s**2
        tau = _dot(obs['torque_n_m'],spec.rotation_axis)
        u = obs['interaction_energy_j']
        drift = fsum((k-k0, *(a['interaction_energy_j']-b['interaction_energy_j'] for a,b in
                         zip(obs['source_contributions'],initial['source_contributions']))))
        for a,b in ((r.kinetic_energy_j,k),(r.interaction_energy_j,u),
                    (r.total_energy_j,k+u),(r.energy_drift_j,drift),
                    (r.axis_torque_n_m,tau),(r.angular_acceleration_rad_s2,tau/spec.inertia_kg_m2)):
            if not same(a,b):
                raise ValueError('angular history disagrees with physical force/energy observations')
        if not (r.source_clearance_m is None if c==inf else same(r.source_clearance_m,c)):
            raise ValueError('saved angular source clearance changed')
    if result.termination=='source-guard':
        clearance=_clearance(spec,rows[-1].angle_rad)
        if clearance > spec.source_guard_m+2.01*max(.02*spec.source_guard_m,1e-12*spec.max_endpoint_radius_m):
            raise ValueError('guard termination is not near a physical source/jump')
    return True
