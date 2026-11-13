"""M17a: genuine classical vector motion in proven electrostatic symmetry planes.

The authored conductors certify an invariant plane; vector Newtonian dynamics
are integrated without projecting forces or inventing a guiding constraint.
Electrostatic forces/energies come only from existing force_energy.
This is data/physics, not a browser or finite-particle collision model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import ceil, fsum, hypot, isfinite
from typing import Literal

from ._motion_rk4 import rk4_step
from ._resources import cleanup_resources
from .model import ConductingPlane, ElectrostaticProblem, PointCharge, SolidSphere, SphericalShell, Vector3, _position
from .motion_1d import MotionIntegrationError, _NONRELATIVISTIC_SPEED_LIMIT_M_S
from .observables import force_energy
from .solver import UnsupportedProblem


def _dot(a, b):
    return fsum(x * y for x, y in zip(a, b))


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _add(a, b, scale=1.):
    return tuple(x + scale*y for x, y in zip(a, b))


def _norm(a):
    return hypot(*a)


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _unit(v):
    length = _norm(v)
    if not length or not isfinite(length):
        raise ValueError('cannot define planar direction from a zero or nonfinite vector')
    return tuple(x/length for x in v)


def _perpendicular(v):
    axis = (1., 0., 0.) if abs(v[0]) < .8*_norm(v) else (0., 1., 0.)
    return _unit(_cross(v, axis))


def _invariant_normal(radial_or_normal, velocity):
    cross = _cross(radial_or_normal, velocity)
    if _norm(cross) > 1e-12*_norm(radial_or_normal)*_norm(velocity):
        return _unit(cross)
    base = radial_or_normal if _norm(radial_or_normal) else velocity
    return _perpendicular(base if _norm(base) else (1., 0., 0.))


@dataclass(frozen=True, slots=True)
class Motion2DSpec:
    """One moving nonzero point charge, with a full vector initial velocity.

    Physical planar invariance is proved from fixed plane or concentric
    spherical geometry. The chosen plane is a report, not an imposed track.
    """
    problem: ElectrostaticProblem
    charge_index: int
    mass_kg: float
    initial_velocity_m_s: Vector3
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
        if len(self.problem.charges) != 1 or not self.problem.charges[0].charge_c:
            raise UnsupportedProblem('M17a requires exactly one nonzero moving point charge')
        for name in ('mass_kg', 'duration_s', 'sample_interval_s', 'relative_tolerance'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be a finite positive number')
        if self.relative_tolerance >= .05:
            raise ValueError('relative_tolerance must be smaller than 0.05')
        if self.sample_interval_s > self.duration_s:
            raise ValueError('sample interval cannot exceed motion duration')
        if ceil(self.duration_s/self.sample_interval_s) > 2000:
            raise ValueError('at most 2000 requested output intervals')
        if isinstance(self.max_internal_steps, bool) or not isinstance(self.max_internal_steps, int) or self.max_internal_steps < 10:
            raise ValueError('max_internal_steps must be an integer >= 10')
        if self.surface_guard_m is not None and (
                isinstance(self.surface_guard_m, bool) or not isinstance(self.surface_guard_m, (float, int))
                or not isfinite(self.surface_guard_m) or self.surface_guard_m <= 0):
            raise ValueError('surface_guard_m must be finite and positive')
        velocity = _position(self.initial_velocity_m_s, 'initial velocity')
        if _norm(velocity) >= _NONRELATIVISTIC_SPEED_LIMIT_M_S:
            raise UnsupportedProblem('M17a is nonrelativistic; initial speed must be below 0.1c')
        object.__setattr__(self, 'initial_velocity_m_s', velocity)
        _geometry(self)

    @property
    def selected_charge(self):
        return self.problem.charges[self.charge_index]


@dataclass(frozen=True, slots=True)
class Motion2DSample:
    time_s: float
    displacement_m: Vector3
    position_m: Vector3
    velocity_m_s: Vector3
    acceleration_m_s2: Vector3
    force_n: Vector3
    kinetic_energy_j: float
    interaction_energy_j: float
    total_energy_j: float
    energy_drift_j: float


@dataclass(frozen=True, slots=True)
class Motion2DEvent:
    kind: Literal['surface-guard']
    time_s: float
    position_m: Vector3
    description: str


@dataclass(frozen=True, slots=True)
class Motion2DResult:
    spec: Motion2DSpec
    samples: tuple[Motion2DSample, ...]
    events: tuple[Motion2DEvent, ...]
    termination: Literal['duration', 'surface-guard']
    plane_origin_m: Vector3
    plane_normal: Vector3
    max_abs_energy_drift_j: float
    method: str = 'Adaptive RK4 step doubling; full-vector existing analytical force_energy at each RHS evaluation'

    def as_dict(self):
        return dict(schema='conductor_sim.motion_2d.v1',
                    mode='elapsed-time planar classical vector dynamics',
                    units={'time':'s', 'position':'m', 'velocity':'m/s',
                           'acceleration':'m/s^2', 'force':'N', 'energy':'J'},
                    mass_kg=self.spec.mass_kg,
                    charge_c=self.spec.selected_charge.charge_c,
                    initial_position_m=self.spec.selected_charge.position_m,
                    initial_velocity_m_s=self.spec.initial_velocity_m_s,
                    plane_origin_m=self.plane_origin_m,
                    plane_normal=self.plane_normal,
                    termination=self.termination,
                    method=self.method,
                    guard_m=_geometry(self.spec).guard,
                    max_abs_energy_drift_j=self.max_abs_energy_drift_j,
                    samples=[asdict(s) for s in self.samples],
                    events=[asdict(e) for e in self.events])


@dataclass(frozen=True, slots=True)
class _Geometry:
    kind: Literal['planes', 'exterior-sphere', 'cavity']
    conductors: tuple
    origin: Vector3
    normal: Vector3
    scale: float
    guard: float

    def clearance(self, position):
        if self.kind == 'planes':
            return min(c.signed_distance_m(position) for c in self.conductors) - self.guard
        radius = _norm(_sub(position, self.origin))
        if self.kind == 'exterior-sphere':
            return radius - self.conductors[0].radius_m - self.guard
        return self.conductors[0].inner_radius_m - radius - self.guard

    def approach_speed(self, position, velocity, acceleration):
        """Conservative normal approach, avoiding tangential-speed overrestriction."""
        if self.kind == 'planes':
            normal = min(self.conductors, key=lambda c:c.signed_distance_m(position)).normal
            inward_speed = max(0., -_dot(velocity, normal))
            normal_acceleration = abs(_dot(acceleration, normal))
        else:
            # A central trajectory may cross r=0. The radial unit direction
            # is undefined there: bound approach by FULL speed/acceleration,
            # without a fictitious centrifugal 1/r singularity.
            inward_speed, normal_acceleration = _norm(velocity), _norm(acceleration)
        return inward_speed, normal_acceleration


def _geometry(spec: Motion2DSpec) -> _Geometry:
    conductors = spec.problem.conductors
    if not conductors:
        raise UnsupportedProblem('M17a requires stationary authored conductors')
    initial = spec.selected_charge.position_m
    velocity = spec.initial_velocity_m_s
    if all(isinstance(c, ConductingPlane) for c in conductors):
        if not 1 <= len(conductors) <= 2:
            raise UnsupportedProblem('M17a supports one plane or two facing parallel plates')
        if len(conductors) == 2 and _dot(conductors[0].normal, conductors[1].normal) > -1+1e-10:
            raise UnsupportedProblem('M17a plate pair must have opposite parallel normals')
        if any(c.signed_distance_m(initial) <= 0 for c in conductors):
            raise UnsupportedProblem('moving charge must begin on the vacuum side of each plane')
        scale = min(c.signed_distance_m(initial) for c in conductors)
        origin = initial
        normal = _invariant_normal(conductors[0].normal, velocity)
        kind = 'planes'
    elif len(conductors) == 1 and isinstance(conductors[0], SolidSphere):
        center = conductors[0].center_m
        delta = _sub(initial, center)
        if _norm(delta) <= conductors[0].radius_m:
            raise UnsupportedProblem('moving charge must begin outside solid sphere')
        origin, normal, scale = center, _invariant_normal(delta, velocity), conductors[0].radius_m
        kind = 'exterior-sphere'
    elif all(isinstance(c, SphericalShell) for c in conductors):
        shells = tuple(sorted(conductors, key=lambda c:c.inner_radius_m))
        center = shells[0].center_m
        if any(c.center_m != center for c in shells) or any(
                a.outer_radius_m >= b.inner_radius_m for a,b in zip(shells, shells[1:])):
            raise UnsupportedProblem('M17a cavity shells must be concentric and disjoint')
        r = _sub(initial, center)
        if _norm(r) >= shells[0].inner_radius_m:
            raise UnsupportedProblem('moving charge must begin in innermost vacuum cavity')
        origin, normal, scale = center, _invariant_normal(r, velocity), shells[0].inner_radius_m
        conductors, kind = shells, 'cavity'
    else:
        raise UnsupportedProblem('M17a supports only plane(s), one exterior sphere or nested spherical cavities')
    guard = 1e-6*scale if spec.surface_guard_m is None else spec.surface_guard_m
    domain = _Geometry(kind, tuple(conductors), origin, normal, scale, guard)
    if domain.clearance(initial) <= 0:
        raise ValueError('initial point is within the surface-approach guard')
    if abs(_dot(_sub(initial, origin), normal)) > 1e-9*scale or abs(_dot(velocity, normal)) > max(1e-12, 1e-9*_norm(velocity)):
        raise UnsupportedProblem('an invariant planar motion cannot be established')
    return domain


def integrate_motion_2d(spec: Motion2DSpec, backend) -> Motion2DResult:
    """Integrate full position/velocity vectors in a proved symmetry plane.

    Every RK stage first checks the true vacuum clearance. No numerical FEMM
    force fallback, trajectory projection or surface impact is supplied.
    """
    if not isinstance(spec, Motion2DSpec):
        raise TypeError('spec must be Motion2DSpec')
    geometry = _geometry(spec)
    original = spec.selected_charge
    p0 = original.position_m

    def observe(position):
        if geometry.clearance(position) <= 0:
            raise _SurfaceGuard()
        if abs(_dot(_sub(position, geometry.origin), geometry.normal)) > max(
                1e-11*geometry.scale, 5e-9*max(geometry.scale, _norm(_sub(position, geometry.origin)))):
            raise UnsupportedProblem('particle left the physically invariant plane')
        problem = replace(spec.problem, charges=(PointCharge(original.charge_c, position),))
        supports = getattr(backend, 'supports', None)
        if callable(supports) and not supports(problem):
            raise UnsupportedProblem('selected backend does not support this vector-motion state')
        solution = backend.solve(problem)
        try:
            data = force_energy(solution)
            if len(data.charges) != 1 or (data.charges[0].source.position_m != position or
                                          data.charges[0].source.charge_c != original.charge_c):
                raise UnsupportedProblem('moving point-charge identity became ambiguous')
            force = data.charges[0].total_force_n
            if not all(isfinite(x) for x in force) or not isfinite(data.energy.total_j):
                raise MotionIntegrationError('nonfinite force or interaction energy')
            if abs(_dot(force, geometry.normal)) > max(1e-20, 5e-9*_norm(force)):
                raise UnsupportedProblem('transverse force violates proven planar symmetry')
            result = force, data.energy
        except BaseException as exc:
            close = getattr(solution, 'close', None)
            if callable(close):
                cleanup_resources((('vector motion solution', close),), primary=exc)
            raise
        else:
            close = getattr(solution, 'close', None)
            if callable(close):
                close()
        return result

    def acceleration(position):
        force, _ = observe(position)
        return tuple(x/spec.mass_kg for x in force)

    def rhs(y):
        position = y[:3]
        acceleration_vector = acceleration(position)
        return (*y[3:], *acceleration_vector)

    y = (*p0, *spec.initial_velocity_m_s)
    reference_force, reference_energy = observe(p0)
    _ = reference_force
    energy0 = .5*spec.mass_kg*_norm(spec.initial_velocity_m_s)**2
    records = []
    events = []
    max_drift = 0.

    def record(t, values):
        nonlocal max_drift
        position, velocity = values[:3], values[3:]
        force, energy = observe(position)
        kinetic = .5*spec.mass_kg*_norm(velocity)**2
        drift = fsum((kinetic-energy0,
                      energy.direct_j-reference_energy.direct_j,
                      energy.induced_j-reference_energy.induced_j,
                      energy.background_j-reference_energy.background_j))
        if not all(isfinite(x) for x in (*position, *velocity, kinetic, drift)):
            raise MotionIntegrationError('motion data became nonfinite')
        max_drift = max(max_drift, abs(drift))
        records.append(Motion2DSample(t, _sub(position,p0), position, velocity,
                                      tuple(x/spec.mass_kg for x in force), force,
                                      kinetic, energy.total_j, kinetic+energy.total_j, drift))

    record(0., y)
    t = 0.
    h = spec.sample_interval_s/4
    steps_done = 0
    termination: Literal['duration', 'surface-guard'] = 'duration'
    targets = [min(spec.duration_s, i*spec.sample_interval_s)
               for i in range(1, ceil(spec.duration_s/spec.sample_interval_s)+1)]

    for target in targets:
        while t < target-2e-15*spec.duration_s:
            if steps_done >= spec.max_internal_steps:
                raise MotionIntegrationError('maximum internal RK4 substeps exceeded')
            steps_done += 1
            distance = geometry.clearance(y[:3])
            closeness = max(.02*geometry.guard, 1e-10*geometry.scale)
            if distance <= closeness:
                termination = 'surface-guard'
                events.append(Motion2DEvent('surface-guard', t, y[:3],
                              'Stopped before the ideal conductor; not an exact collision'))
                break
            dt = min(h, target-t)
            a = acceleration(y[:3])
            normal_speed, normal_accel = geometry.approach_speed(y[:3], y[3:], a)
            if normal_speed or normal_accel:
                bound = normal_speed + normal_accel*dt
                if bound:
                    dt = min(dt, .5*distance/bound)
            if dt < max(1e-14*spec.duration_s, 1e-300):
                if distance <= max(2*closeness, 1e-8*geometry.scale):
                    termination = 'surface-guard'
                    events.append(Motion2DEvent('surface-guard', t, y[:3],
                                  'Stopped before the numerical conductor approach cutoff; not an exact collision'))
                    break
                raise MotionIntegrationError('timestep collapsed before a conductor guard')
            try:
                large = rk4_step(y, dt, rhs)
                half = rk4_step(y, .5*dt, rhs)
                fine = rk4_step(half, .5*dt, rhs)
            except _SurfaceGuard:
                h = .5*dt
                continue
            if geometry.clearance(large[:3]) <= 0 or geometry.clearance(fine[:3]) <= 0:
                h = .5*dt
                continue
            if _norm(fine[3:]) >= _NONRELATIVISTIC_SPEED_LIMIT_M_S:
                raise UnsupportedProblem('M17a nonrelativistic limit reached; speed must stay below 0.1c')
            position_scale = max(1e-11*geometry.scale, spec.relative_tolerance*
                                 max(geometry.scale, _norm(_sub(y[:3],p0)), _norm(_sub(fine[:3],p0))))
            velocity_scale = max(1e-10, spec.relative_tolerance*max(1., _norm(y[3:]), _norm(fine[3:])))
            error = max(_norm(_sub(large[:3],fine[:3]))/(15*position_scale),
                        _norm(_sub(large[3:],fine[3:]))/(15*velocity_scale))
            if not isfinite(error):
                raise MotionIntegrationError('nonfinite RK4 integration error')
            if error > 1:
                h = max(.15*dt, dt*min(.8,.9*error**(-.2)))
                continue
            t += dt
            y = fine
            h = min(spec.sample_interval_s,
                    max(.2*dt, dt*min(2.5,(.9/max(error,1e-10))**.2)))
        if termination == 'surface-guard':
            if t > records[-1].time_s + 2e-15*spec.duration_s:
                record(t, y)
            break
        t = target
        record(t, y)
    return Motion2DResult(spec, tuple(records), tuple(events), termination,
                          geometry.origin, geometry.normal, max_drift)


class _SurfaceGuard(Exception):
    pass
