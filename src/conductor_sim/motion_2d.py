"""M17–M18a: classical vector motion in proven electrostatic symmetry planes.

The conductors, launch and any externally held point charges certify an invariant
plane. Newtonian dynamics is integrated without a fabricated guide force.
Electrostatic forces/energies come only from existing force_energy.
This is data/physics, not a browser or finite-particle collision model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import fsum, hypot, isfinite
from typing import Literal

from ._motion_rk4 import rk4_step
from ._motion_time import output_times
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


def _certified_normal(primary, velocity, anchors):
    """Choose the invariant plane from launch and stationary-source geometry.

    A degenerate radial/normal launch does not determine an azimuth. The first
    noncollinear stationary source may select it; all other sources must then
    lie in the *same* reflection-symmetry plane or the model is refused.
    """
    for candidate in (velocity, *anchors):
        cross = _cross(primary, candidate)
        if _norm(cross) > 1e-12*_norm(primary)*_norm(candidate):
            return _unit(cross)
    return _invariant_normal(primary, velocity)


def _invariant_normal(radial_or_normal, velocity):
    cross = _cross(radial_or_normal, velocity)
    if _norm(cross) > 1e-12*_norm(radial_or_normal)*_norm(velocity):
        return _unit(cross)
    base = radial_or_normal if _norm(radial_or_normal) else velocity
    return _perpendicular(base if _norm(base) else (1., 0., 0.))


@dataclass(frozen=True, slots=True)
class Motion2DSpec:
    """One moving nonzero point charge, with a full vector initial velocity.

    All remaining authored charges are externally held stationary. Planar
    invariance is proved from conductor and source geometry, not imposed.
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
    charge_guard_m: float | None = None

    def __post_init__(self):
        if not isinstance(self.problem, ElectrostaticProblem):
            raise TypeError('problem must be ElectrostaticProblem')
        if isinstance(self.charge_index, bool) or not isinstance(self.charge_index, int):
            raise TypeError('charge_index must be an integer')
        if not 0 <= self.charge_index < len(self.problem.charges):
            raise IndexError('charge index outside authored inventory')
        if not 1 <= len(self.problem.charges) <= 32:
            raise UnsupportedProblem('M18a supports one moving charge and at most 31 fixed charge sites')
        if any(q.charge_c == 0 for q in self.problem.charges):
            raise UnsupportedProblem('moving and fixed point charges must all be nonzero')
        positions = [q.position_m for q in self.problem.charges]
        if len(set(positions)) != len(positions):
            raise UnsupportedProblem('separate authored point charges must occupy distinct sites')
        for name in ('mass_kg', 'duration_s', 'sample_interval_s', 'relative_tolerance'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be a finite positive number')
        if self.relative_tolerance >= .05:
            raise ValueError('relative_tolerance must be smaller than 0.05')
        if self.sample_interval_s > self.duration_s:
            raise ValueError('sample interval cannot exceed motion duration')
        output_times(self.duration_s, self.sample_interval_s)
        if isinstance(self.max_internal_steps, bool) or not isinstance(self.max_internal_steps, int) or self.max_internal_steps < 10:
            raise ValueError('max_internal_steps must be an integer >= 10')
        if self.surface_guard_m is not None and (
                isinstance(self.surface_guard_m, bool) or not isinstance(self.surface_guard_m, (float, int))
                or not isfinite(self.surface_guard_m) or self.surface_guard_m <= 0):
            raise ValueError('surface_guard_m must be finite and positive')
        if self.charge_guard_m is not None and (
                isinstance(self.charge_guard_m, bool) or
                not isinstance(self.charge_guard_m, (float, int)) or
                not isfinite(self.charge_guard_m) or self.charge_guard_m <= 0):
            raise ValueError('charge_guard_m must be finite and positive')
        if self.charge_guard_m is not None and not self.stationary_charges:
            raise ValueError('charge_guard_m requires at least one externally held charge')
        velocity = _position(self.initial_velocity_m_s, 'initial velocity')
        if _norm(velocity) >= _NONRELATIVISTIC_SPEED_LIMIT_M_S:
            raise UnsupportedProblem('M17a is nonrelativistic; initial speed must be below 0.1c')
        object.__setattr__(self, 'initial_velocity_m_s', velocity)
        _geometry(self)

    @property
    def selected_charge(self):
        return self.problem.charges[self.charge_index]

    @property
    def stationary_charges(self) -> tuple[PointCharge, ...]:
        """Externally held sites in original authored order, never integrated."""
        return tuple(q for i, q in enumerate(self.problem.charges)
                     if i != self.charge_index)

    def problem_at(self, position_m: Vector3) -> ElectrostaticProblem:
        """Move only the selected authored charge; preserve every other component."""
        charges = list(self.problem.charges)
        charges[self.charge_index] = PointCharge(self.selected_charge.charge_c, position_m)
        return replace(self.problem, charges=tuple(charges))


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
    kind: Literal['surface-guard', 'charge-guard']
    time_s: float
    position_m: Vector3
    description: str


@dataclass(frozen=True, slots=True)
class Motion2DResult:
    spec: Motion2DSpec
    samples: tuple[Motion2DSample, ...]
    events: tuple[Motion2DEvent, ...]
    termination: Literal['duration', 'surface-guard', 'charge-guard']
    plane_origin_m: Vector3
    plane_normal: Vector3
    max_abs_energy_drift_j: float
    method: str = 'Adaptive RK4 step doubling; full-vector existing analytical force_energy at each RHS evaluation'

    def as_dict(self):
        data = dict(schema='conductor_sim.motion_2d.v1',
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
        if self.spec.stationary_charges:
            geometry = _geometry(self.spec)
            data.update(moving_charge_index=self.spec.charge_index,
                        stationary_charges=[dict(charge_index=i, **asdict(q))
                                            for i, q in enumerate(self.spec.problem.charges)
                                            if i != self.spec.charge_index],
                        charge_guard_m=geometry.charge_guard,
                        stationary_sources='Externally held fixed point charges; their velocities are zero. '
                                           'Only the selected authored charge is integrated.')
        return data


@dataclass(frozen=True, slots=True)
class _Geometry:
    kind: Literal['planes', 'exterior-sphere', 'cavity']
    conductors: tuple
    origin: Vector3
    normal: Vector3
    scale: float
    guard: float
    stationary: tuple[PointCharge, ...] = ()
    charge_guard: float = 0.

    def charge_clearance(self, position):
        if not self.stationary:
            return float('inf')
        return min(_norm(_sub(position, q.position_m)) for q in self.stationary)-self.charge_guard

    def charge_segment_clearance(self, start, end):
        """Conservative polyline guard; never jump over a singular fixed site.

        This is not a claim that the true curved trajectory is straight.
        On a suspected near miss the integrator subdivides the timestep.
        """
        if not self.stationary:
            return float('inf')
        direction = _sub(end, start)
        length2 = _dot(direction, direction)
        clearances = []
        for site in self.stationary:
            if length2:
                t = min(1., max(0., _dot(_sub(site.position_m,start),direction)/length2))
            else:
                t = 0.
            closest = _add(start,direction,t)
            clearances.append(_norm(_sub(closest,site.position_m))-self.charge_guard)
        return min(clearances)

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
    stationary = spec.stationary_charges
    if all(isinstance(c, ConductingPlane) for c in conductors):
        if not 1 <= len(conductors) <= 2:
            raise UnsupportedProblem('M17a supports one plane or two facing parallel plates')
        if len(conductors) == 2 and _dot(conductors[0].normal, conductors[1].normal) > -1+1e-10:
            raise UnsupportedProblem('M17a plate pair must have opposite parallel normals')
        if any(c.signed_distance_m(initial) <= 0 for c in conductors):
            raise UnsupportedProblem('moving charge must begin on the vacuum side of each plane')
        scale = min(c.signed_distance_m(initial) for c in conductors)
        origin = initial
        normal = _certified_normal(conductors[0].normal, velocity,
                                   tuple(_sub(q.position_m, initial) for q in stationary))
        kind = 'planes'
    elif len(conductors) == 1 and isinstance(conductors[0], SolidSphere):
        center = conductors[0].center_m
        delta = _sub(initial, center)
        if _norm(delta) <= conductors[0].radius_m:
            raise UnsupportedProblem('moving charge must begin outside solid sphere')
        normal = _certified_normal(delta, velocity,
                    tuple(_sub(q.position_m, center) for q in stationary))
        origin, scale = center, conductors[0].radius_m
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
        primary = r if _norm(r) else velocity
        if not _norm(primary):
            primary = next((_sub(q.position_m, center) for q in stationary
                            if _norm(_sub(q.position_m, center))), (1., 0., 0.))
        normal = _certified_normal(primary, velocity,
                    tuple(_sub(q.position_m, center) for q in stationary))
        origin, scale = center, shells[0].inner_radius_m
        conductors, kind = shells, 'cavity'
    else:
        raise UnsupportedProblem('M17a supports only plane(s), one exterior sphere or nested spherical cavities')
    guard = 1e-6*scale if spec.surface_guard_m is None else spec.surface_guard_m
    separations = [_norm(_sub(q.position_m, initial)) for q in stationary]
    charge_guard = (spec.charge_guard_m if spec.charge_guard_m is not None else
                    max(1e-8*scale, 1e-5*min(separations)) if stationary else 0.)
    domain = _Geometry(kind, tuple(conductors), origin, normal, scale, guard,
                       stationary, charge_guard)
    if domain.clearance(initial) <= 0:
        raise ValueError('initial point is within the surface-approach guard')
    if abs(_dot(_sub(initial, origin), normal)) > 1e-9*scale or abs(_dot(velocity, normal)) > max(1e-12, 1e-9*_norm(velocity)):
        raise UnsupportedProblem('an invariant planar motion cannot be established')
    if domain.charge_clearance(initial) <= 0:
        raise ValueError('moving charge starts within the stationary point-charge guard')
    for source in stationary:
        if domain.clearance(source.position_m) <= 0:
            raise UnsupportedProblem('fixed point charge must occupy the same valid vacuum region')
        if abs(_dot(_sub(source.position_m, origin), normal)) > max(
                1e-12*scale, 1e-9*_norm(_sub(source.position_m, origin))):
            raise UnsupportedProblem('fixed point charges must share the physically invariant plane')
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
        if geometry.charge_clearance(position) <= 0:
            raise _ChargeGuard()
        if abs(_dot(_sub(position, geometry.origin), geometry.normal)) > max(
                1e-11*geometry.scale, 5e-9*max(geometry.scale, _norm(_sub(position, geometry.origin)))):
            raise UnsupportedProblem('particle left the physically invariant plane')
        problem = spec.problem_at(position)
        supports = getattr(backend, 'supports', None)
        if callable(supports) and not supports(problem):
            raise UnsupportedProblem('selected backend does not support this vector-motion state')
        solution = backend.solve(problem)
        try:
            data = force_energy(solution)
            authored = set(problem.charges)
            if (len(data.charges) != len(authored) or
                    {c.source for c in data.charges} != authored):
                raise UnsupportedProblem('moving or fixed point-charge identity became ambiguous')
            moving = next(c for c in data.charges if c.source == problem.charges[spec.charge_index])
            force = moving.total_force_n
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

    samples, events, termination, drift = _integrate_vector_motion(spec, geometry, observe)
    return Motion2DResult(spec, samples, events, termination,
                          geometry.origin, geometry.normal, drift)


def _integrate_vector_motion(spec, geometry, observe, *,
                             surface_description='Stopped before the ideal conductor; not an exact collision'):
    """Shared adaptive controller for conductor and prescribed-field motion.

    The caller supplies observations and a domain; no force law lives here.
    Existing conductor stepping, error scales, sample times and guards remain
    unchanged. The prescribed-source domain supplies the same geometry contract.
    """
    p0 = spec.selected_charge.position_m

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
    termination: Literal['duration', 'surface-guard', 'charge-guard'] = 'duration'
    targets = output_times(spec.duration_s, spec.sample_interval_s)

    for target in targets:
        while t < target-2e-15*spec.duration_s:
            if steps_done >= spec.max_internal_steps:
                raise MotionIntegrationError('maximum internal RK4 substeps exceeded')
            steps_done += 1
            conductor_distance = geometry.clearance(y[:3])
            point_distance = geometry.charge_clearance(y[:3])
            near_charge = point_distance < conductor_distance
            distance = min(conductor_distance, point_distance)
            guard_kind = 'charge-guard' if near_charge else 'surface-guard'
            active_guard = geometry.charge_guard if near_charge else geometry.guard
            closeness = max(.02*active_guard, 1e-10*geometry.scale)
            if distance <= closeness:
                termination = guard_kind
                events.append(Motion2DEvent(guard_kind, t, y[:3],
                              'Stopped before the fixed point charge; not an exact collision' if near_charge else
                              surface_description))
                break
            dt = min(h, target-t)
            a = acceleration(y[:3])
            normal_speed, normal_accel = geometry.approach_speed(y[:3], y[3:], a)
            if geometry.stationary:
                # A conservative displacement bound limits leapfrogging an
                # ideal fixed point-charge singularity between RK stages.
                normal_speed = max(normal_speed, _norm(y[3:]))
                normal_accel = max(normal_accel, _norm(a))
            if normal_speed or normal_accel:
                bound = normal_speed + normal_accel*dt
                if bound:
                    dt = min(dt, .5*distance/bound)
            if dt < max(1e-14*spec.duration_s, 1e-300):
                if distance <= max(2*closeness, 1e-8*geometry.scale):
                    termination = guard_kind
                    events.append(Motion2DEvent(guard_kind, t, y[:3],
                                  'Stopped before the numerical approach cutoff; not an exact collision'))
                    break
                raise MotionIntegrationError('timestep collapsed before an approach guard')
            try:
                large = rk4_step(y, dt, rhs)
                half = rk4_step(y, .5*dt, rhs)
                fine = rk4_step(half, .5*dt, rhs)
            except (_SurfaceGuard, _ChargeGuard):
                h = .5*dt
                continue
            if (geometry.clearance(large[:3]) <= 0 or geometry.clearance(fine[:3]) <= 0 or
                    geometry.charge_clearance(large[:3]) <= 0 or
                    geometry.charge_clearance(fine[:3]) <= 0 or
                    geometry.charge_segment_clearance(y[:3],half[:3]) <= 0 or
                    geometry.charge_segment_clearance(half[:3],fine[:3]) <= 0):
                h = .5*dt
                continue
            if _norm(fine[3:]) >= _NONRELATIVISTIC_SPEED_LIMIT_M_S:
                raise UnsupportedProblem('classical motion limit reached; speed must stay below 0.1c')
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
        if termination in ('surface-guard', 'charge-guard'):
            if t > records[-1].time_s + 2e-15*spec.duration_s:
                record(t, y)
            break
        t = target
        record(t, y)
    return tuple(records), tuple(events), termination, max_drift


class _SurfaceGuard(Exception):
    pass


class _ChargeGuard(Exception):
    pass
