"""M17–M18a: classical vector motion in proven electrostatic symmetry planes.

The conductors, launch and any externally held point charges certify an invariant
plane. Newtonian dynamics is integrated without a fabricated guide force.
Electrostatic forces/energies come only from existing force_energy.
This is data/physics, not a browser or finite-particle collision model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from math import fsum, hypot, isfinite, ulp
from typing import Literal

from ._directions import unit_direction
from ._motion_rk4 import rk4_step
from ._motion_time import output_times
from ._resources import cleanup_resources
from .conservative_forces import UniformGravity
from .model import ConductingPlane, ElectrostaticProblem, PointCharge, SolidSphere, SphericalShell, Vector3, _position
from .motion_1d import MotionIntegrationError, _NONRELATIVISTIC_SPEED_LIMIT_M_S
from .observables import _exact_sum, _scaled_ratio_product, force_energy
from .solver import UnsupportedProblem


def _dot(a, b):
    return fsum(x * y for x, y in zip(a, b))


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _add(a, b, scale=1.):
    return tuple(x + scale*y for x, y in zip(a, b))


def _norm(a):
    return hypot(*a)


def _world_state(origin, state):
    """Convert RK coordinates to world position, preserving velocity and work."""
    return (*_add(origin, state[:3]), *state[3:])


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _unit(v):
    return unit_direction(v, 'cannot define planar direction from a zero or nonfinite vector')


def _perpendicular(v):
    axis = (1., 0., 0.) if abs(v[0]) < .8*_norm(v) else (0., 1., 0.)
    return _unit(_cross(v, axis))


def _exact_cross(a, b):
    """Exact rational cross product converted to scaled float64 direction, avoiding cancellation.

    Returns ``(scaled_float_cross, sin_ratio, phys_perp_exact, exact_rational_cross)``
    where ``phys_perp_exact = |a x b| / |a|`` is an exact ``Fraction`` (no underflow/overflow).
    """
    fa = tuple(Fraction(x) for x in a)
    fb = tuple(Fraction(x) for x in b)
    fc = (
        fa[1] * fb[2] - fa[2] * fb[1],
        fa[2] * fb[0] - fa[0] * fb[2],
        fa[0] * fb[1] - fa[1] * fb[0],
    )
    max_abs = max(abs(x) for x in fc)
    if max_abs == 0:
        return (0.0, 0.0, 0.0), 0.0, Fraction(), fc
    # Scale by max_abs in exact rationals so float conversion never overflows/underflows.
    scaled = tuple(float(x / max_abs) for x in fc)
    a_max = max(abs(x) for x in fa)
    b_max = max(abs(x) for x in fb)
    norm_a = hypot(*(float(x / a_max) for x in fa)) if a_max else 0.0
    norm_b = hypot(*(float(x / b_max) for x in fb)) if b_max else 0.0
    norm_s = hypot(*scaled)
    denom = a_max * b_max
    sin_ratio = (float(max_abs / denom) * norm_s / (norm_a * norm_b)) if denom and norm_a and norm_b else 0.0
    phys_perp_exact = ((max_abs * Fraction(norm_s)) / (a_max * Fraction(norm_a))) if a_max and norm_a else Fraction()
    return scaled, sin_ratio, phys_perp_exact, fc


def _exact_out_of_plane(vec, exact_normal):
    """Exact rational normal projection sine and ``Fraction`` magnitude relative to ``exact_normal``.

    Eliminates spurious out-of-plane residuals from float64 rounding, underflow, or overflow.
    Returns ``(sin_out_of_plane, phys_normal_exact)``.
    """
    fn = tuple(exact_normal)
    n_max = max(abs(x) for x in fn)
    fv = tuple(Fraction(x) for x in vec)
    v_max = max(abs(x) for x in fv)
    if n_max == 0 or v_max == 0:
        return 0.0, Fraction()
    dot_exact = sum((a * b for a, b in zip(fn, fv)), Fraction())
    if dot_exact == 0:
        return 0.0, Fraction()
    norm_n_scaled = hypot(*(float(x / n_max) for x in fn))
    norm_v_scaled = hypot(*(float(x / v_max) for x in fv))
    abs_dot = abs(dot_exact)
    sin_out = float(abs_dot / (n_max * v_max)) / (norm_n_scaled * norm_v_scaled)
    phys_comp_exact = abs_dot / (n_max * Fraction(norm_n_scaled))
    return sin_out, phys_comp_exact


def _exceeds_accel_disp(accel_exact: Fraction, duration_s: float, limit_m: float) -> bool:
    """Exact rational test for 0.5 * |accel| * duration_s^2 > limit_m (no underflow/overflow)."""
    if accel_exact == 0 or duration_s <= 0.0:
        return False
    t = Fraction(duration_s)
    return Fraction(1, 2) * abs(accel_exact) * t * t > Fraction(limit_m)


def _exceeds_vel_disp(vel_exact: Fraction, duration_s: float, limit_m: float) -> bool:
    """Exact rational test for |vel| * duration_s > limit_m (no underflow/overflow)."""
    if vel_exact == 0 or duration_s <= 0.0:
        return False
    return abs(vel_exact) * Fraction(duration_s) > Fraction(limit_m)


def _exact_sub(a, b) -> tuple[Fraction, Fraction, Fraction]:
    """Exact rational vector difference avoiding float64 rounding before cross products."""
    return (Fraction(a[0]) - Fraction(b[0]), Fraction(a[1]) - Fraction(b[1]), Fraction(a[2]) - Fraction(b[2]))


def _exact_nonzero(vec) -> bool:
    return any(x != 0 for x in vec)


def _exact_ref_axis(vec) -> tuple[float, float, float]:
    fv = tuple(Fraction(x) for x in vec)
    v_max = max(abs(x) for x in fv)
    if v_max == 0:
        return (1., 0., 0.)
    scaled = tuple(float(x / v_max) for x in fv)
    return (1., 0., 0.) if abs(scaled[0]) < .8 * hypot(*scaled) else (0., 1., 0.)


def _certified_normal(
    primary,
    velocity,
    anchors,
    *,
    gravity_anchor=None,
    has_gravity: bool = False,
    duration_s: float = 0.0,
    scale: float = 0.0,
):
    """Choose the invariant plane from launch and stationary-source geometry.

    A degenerate radial/normal launch does not determine an azimuth. The first
    noncollinear stationary source may select it; all other sources must then
    lie in the *same* reflection-symmetry plane or the model is refused.
    Returns ``(unit_normal, exact_rational_normal)``.
    """
    vel_sin_tol = 4.0 * ulp(1.0) if has_gravity else 1e-12
    anchor_sin_tol = 4.0 * ulp(1.0) if has_gravity else 1e-12
    v_scaled, v_sin, v_perp, v_fc = _exact_cross(primary, velocity)
    if _norm(v_scaled) > 0.0 and (
        v_sin > vel_sin_tol
        or (has_gravity and _exceeds_vel_disp(v_perp, duration_s, 1e-9 * scale))
    ):
        return _unit(v_scaled), v_fc
    best_anchor_scaled = None
    best_anchor_fc = None
    best_anchor_sin = -1.0
    for candidate in anchors:
        scaled_cross, sin_ratio, _, fc = _exact_cross(primary, candidate)
        if _norm(scaled_cross) > 0.0:
            if sin_ratio > anchor_sin_tol:
                return _unit(scaled_cross), fc
            if has_gravity and (best_anchor_scaled is None or sin_ratio > best_anchor_sin):
                best_anchor_scaled = scaled_cross
                best_anchor_fc = fc
                best_anchor_sin = sin_ratio
    g_scaled = (0.0, 0.0, 0.0)
    g_fc = (Fraction(), Fraction(), Fraction())
    if gravity_anchor is not None:
        g_scaled, g_sin, g_perp, g_fc = _exact_cross(primary, gravity_anchor)
        if _norm(g_scaled) > 0.0 and (
            g_sin > 4.0 * ulp(1.0)
            or _exceeds_accel_disp(g_perp, duration_s, 1e-9 * scale)
        ):
            return _unit(g_scaled), g_fc
    # When no candidate exceeded the primary threshold, still prefer any exact
    # nonzero velocity, anchor, or gravity cross product over an arbitrary fallback axis.
    if has_gravity and _norm(v_scaled) > 0.0:
        return _unit(v_scaled), v_fc
    if has_gravity and best_anchor_scaled is not None:
        return _unit(best_anchor_scaled), best_anchor_fc
    if gravity_anchor is not None and _norm(g_scaled) > 0.0:
        return _unit(g_scaled), g_fc
    base = primary if _exact_nonzero(primary) else (velocity if _exact_nonzero(velocity) else (1., 0., 0.))
    ref_axis = _exact_ref_axis(base)
    scaled_cross, _, _, fc = _exact_cross(base, ref_axis)
    return _unit(scaled_cross), fc


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
    gravity: UniformGravity | None = None

    def __post_init__(self):
        if not isinstance(self.problem, ElectrostaticProblem):
            raise TypeError('problem must be ElectrostaticProblem')
        if self.gravity is not None and type(self.gravity) is not UniformGravity:
            raise TypeError('gravity must be UniformGravity or None')
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
        if self.gravity is not None:
            self.gravity.force_at(self.selected_charge.position_m, mass_kg=self.mass_kg)
            self.gravity.potential_energy_at(self.selected_charge.position_m, mass_kg=self.mass_kg)
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
    electric_force_n: Vector3 | None = None
    gravitational_force_n: Vector3 | None = None
    electric_interaction_energy_j: float | None = None
    gravitational_potential_energy_j: float | None = None


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
        sample_rows = []
        for sample in self.samples:
            row = asdict(sample)
            if self.spec.gravity is None:
                for key in ('electric_force_n', 'gravitational_force_n',
                            'electric_interaction_energy_j', 'gravitational_potential_energy_j'):
                    row.pop(key, None)
            sample_rows.append(row)
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
                    samples=sample_rows,
                    events=[asdict(e) for e in self.events])
        if self.spec.gravity is not None:
            data['gravity'] = self.spec.gravity.as_dict()
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
    g_raw = spec.gravity.acceleration_m_s2 if spec.gravity is not None else None
    g_scaled = None
    g_scale = 0.0
    if g_raw is not None:
        g_scale = max(abs(x) for x in g_raw)
        if g_scale > 0.:
            g_scaled = tuple(x / g_scale for x in g_raw)
    gravity_anchors = (g_scaled,) if g_scaled is not None else ()

    has_gravity = spec.gravity is not None
    if all(isinstance(c, ConductingPlane) for c in conductors):
        if not 1 <= len(conductors) <= 2:
            raise UnsupportedProblem('M17a supports one plane or two facing parallel plates')
        if len(conductors) == 2 and _dot(conductors[0].normal, conductors[1].normal) > -1+1e-10:
            raise UnsupportedProblem('M17a plate pair must have opposite parallel normals')
        if any(c.signed_distance_m(initial) <= 0 for c in conductors):
            raise UnsupportedProblem('moving charge must begin on the vacuum side of each plane')
        scale = min(c.signed_distance_m(initial) for c in conductors)
        origin = initial
        normal, exact_normal = _certified_normal(
            conductors[0].normal, velocity,
            tuple(_exact_sub(q.position_m, initial) for q in stationary),
            gravity_anchor=g_raw if g_scale > 0. else None,
            has_gravity=has_gravity,
            duration_s=spec.duration_s,
            scale=scale,
        )
        kind = 'planes'
    elif len(conductors) == 1 and isinstance(conductors[0], SolidSphere):
        center = conductors[0].center_m
        delta = _sub(initial, center)
        if _norm(delta) <= conductors[0].radius_m:
            raise UnsupportedProblem('moving charge must begin outside solid sphere')
        origin, scale = center, conductors[0].radius_m
        normal, exact_normal = _certified_normal(
            _exact_sub(initial, center), velocity,
            tuple(_exact_sub(q.position_m, center) for q in stationary),
            gravity_anchor=g_raw if g_scale > 0. else None,
            has_gravity=has_gravity,
            duration_s=spec.duration_s,
            scale=scale,
        )
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
        r_exact = _exact_sub(initial, center)
        primary = r_exact if _exact_nonzero(r_exact) else velocity
        if not _exact_nonzero(primary):
            primary = next(
                (a for a in (_exact_sub(q.position_m, center) for q in stationary) if _exact_nonzero(a)),
                None,
            )
        if primary is None or not _exact_nonzero(primary):
            primary = next((g for g in gravity_anchors if _norm(g)), (1., 0., 0.))
        origin, scale = center, shells[0].inner_radius_m
        normal, exact_normal = _certified_normal(
            primary, velocity,
            tuple(_exact_sub(q.position_m, center) for q in stationary),
            gravity_anchor=g_raw if g_scale > 0. else None,
            has_gravity=has_gravity,
            duration_s=spec.duration_s,
            scale=scale,
        )
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
    _, init_out_exact = _exact_out_of_plane(_exact_sub(initial, origin), exact_normal)
    v_sin_out, v_out_exact = _exact_out_of_plane(velocity, exact_normal)
    if (init_out_exact > Fraction(1, 10**9) * Fraction(scale)
            or (v_out_exact > Fraction(1, 10**12) and v_sin_out > 1e-9)):
        raise UnsupportedProblem('an invariant planar motion cannot be established')
    if spec.gravity is not None and _norm(velocity) > 0.:
        if v_sin_out > 64.0 * ulp(1.0) or _exceeds_vel_disp(v_out_exact, spec.duration_s, 1e-9 * scale):
            raise UnsupportedProblem('an invariant planar motion cannot be established')
    saved_n = tuple(Fraction(x) for x in normal)
    if g_raw is not None and g_scale > 0.:
        _, a_out_exact = _exact_out_of_plane(g_raw, exact_normal)
        saved_g_sin, _ = _exact_out_of_plane(g_raw, saved_n)
        if (saved_g_sin > 64.0 * ulp(1.0)
                or _exceeds_accel_disp(a_out_exact, spec.duration_s, 1e-9 * scale)):
            raise UnsupportedProblem('uniform gravity must lie in the physically invariant plane')
    if domain.charge_clearance(initial) <= 0:
        raise ValueError('moving charge starts within the stationary point-charge guard')
    for source in stationary:
        if domain.clearance(source.position_m) <= 0:
            raise UnsupportedProblem('fixed point charge must occupy the same valid vacuum region')
        src_delta = _exact_sub(source.position_m, origin)
        if spec.gravity is not None:
            rel_delta = _exact_sub(source.position_m, initial)
            saved_src_sin, _ = _exact_out_of_plane(src_delta, saved_n)
            saved_rel_sin, _ = _exact_out_of_plane(rel_delta, saved_n)
            if saved_src_sin > 64.0 * ulp(1.0) or saved_rel_sin > 64.0 * ulp(1.0):
                raise UnsupportedProblem('fixed point charges must share the physically invariant plane')
        else:
            src_sin_out, src_out_exact = _exact_out_of_plane(src_delta, exact_normal)
            if src_out_exact > Fraction(1, 10**12) * Fraction(scale) and src_sin_out > 1e-9:
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
            if spec.gravity is None:
                if abs(_dot(force, geometry.normal)) > max(1e-20, 5e-9*_norm(force)):
                    raise UnsupportedProblem('transverse force violates proven planar symmetry')
            elif abs(_dot(force, geometry.normal)) > 5e-9 * _norm(force):
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

    samples, events, termination, drift = _integrate_vector_motion(spec, geometry, observe, gravity=spec.gravity)
    if spec.gravity is None:
        return Motion2DResult(spec, samples, events, termination, geometry.origin, geometry.normal, drift)
    return Motion2DResult(
        spec, samples, events, termination, geometry.origin, geometry.normal, drift,
        'Adaptive RK4 step doubling; full-vector existing analytical force_energy composed with UniformGravity at each RHS evaluation',
    )


def _rk4_delta(state, dt, rhs):
    """Pure RK4 increment vector without adding to ``state`` (preserves low bits for compensated sums)."""
    k1 = rhs(state)
    k2 = rhs(tuple(a + 0.5 * dt * b for a, b in zip(state, k1)))
    k3 = rhs(tuple(a + 0.5 * dt * b for a, b in zip(state, k2)))
    k4 = rhs(tuple(a + dt * b for a, b in zip(state, k3)))
    return tuple(dt * (a + 2.0 * b + 2.0 * c + d) / 6.0 for a, b, c, d in zip(k1, k2, k3, k4))


def _neumaier_add(acc, comp, delta):
    """Neumaier compensated vector addition preserving ~106-bit state precision across many steps."""
    new_acc = []
    new_comp = []
    for a, c, d in zip(acc, comp, delta):
        s = a + d
        c = c + ((a - s) + d if abs(a) >= abs(d) else (d - s) + a)
        new_acc.append(s)
        new_comp.append(c)
    return tuple(new_acc), tuple(new_comp)


def _integrate_vector_motion(spec, geometry, observe, *,
                             surface_description='Stopped before the ideal conductor; not an exact collision',
                             step_observer=None, max_step_s=None, dynamics=None, gravity=None):
    """Shared adaptive controller for conductor and prescribed-field motion.

    The caller supplies observations and a domain; no force law lives here.
    Existing conductor stepping, error scales, sample times and guards remain
    unchanged. The prescribed-source domain supplies the same geometry contract.
    """
    p0 = spec.selected_charge.position_m
    if dynamics is not None:
        # Private, typed M24b adapter; no unrestricted public force callbacks.
        from .driven_motion import _DrivenDynamics
        if type(dynamics) is not _DrivenDynamics or dynamics.spec is not spec:
            raise TypeError('require the bound built-in driven dynamics adapter')
    grav_force = gravity.force_at(p0, mass_kg=spec.mass_kg) if gravity is not None else None

    def acceleration(position):
        force, _ = observe(position)
        if grav_force is not None:
            force = tuple(fsum((a, b)) for a, b in zip(force, grav_force))
            if not all(isfinite(x) for x in force):
                raise MotionIntegrationError('nonfinite composed force')
        return tuple(x/spec.mass_kg for x in force)

    origin = (0., 0., 0.)
    launch = p0  # launch coordinates in the current RK frame
    plane_tolerance = 5e-9 if isinstance(geometry, _Geometry) else 1e-11

    def rhs(y):
        displacement = _sub(y[:3], launch)
        if abs(_dot(displacement, geometry.normal)) > plane_tolerance * max(
                geometry.scale, _norm(displacement)):
            raise UnsupportedProblem('particle left the physically invariant plane')
        world = _world_state(origin, y)
        if dynamics is not None:
            return dynamics.rhs(world)
        acceleration_vector = acceleration(world[:3])
        return (*y[3:], *acceleration_vector)

    y = (*p0, *spec.initial_velocity_m_s)
    if dynamics is not None:
        y = (*y, 0., 0., 0., 0.)  # fixed/applied/gravity work, dissipation
    use_comp = gravity is not None and dynamics is None
    y_acc = y
    y_comp = (0.0,) * len(y)
    reference_force, reference_energy = observe(p0)
    _ = reference_force
    local_gravity = (
        UniformGravity(gravity.acceleration_m_s2, reference_position_m=p0)
        if gravity is not None else None
    )
    origin_gravity = (
        UniformGravity(gravity.acceleration_m_s2, reference_position_m=(0., 0., 0.))
        if gravity is not None else None
    )
    ref_u_grav = gravity.potential_energy_at(p0, mass_kg=spec.mass_kg) if gravity is not None else 0.0
    v0_norm = _norm(spec.initial_velocity_m_s)
    energy0 = _scaled_ratio_product((0.5, spec.mass_kg, v0_norm, v0_norm))
    total_j0 = _exact_sum((energy0, reference_energy.total_j, ref_u_grav)) if gravity is not None else (energy0 + reference_energy.total_j)
    peak_grav_exchange = 0.0
    peak_disp = 0.0
    tilted_plane = sum(x != 0.0 for x in geometry.normal) >= 2
    v_n_prefix = 0.0
    records = []
    events = []
    max_drift = 0.

    def record(t, values):
        nonlocal max_drift, peak_grav_exchange, peak_disp, v_n_prefix
        displacement = _sub(values[:3], launch)
        values = _world_state(origin, values)
        if dynamics is not None:
            sample = dynamics.sample(t, values)
            max_drift = max(max_drift, abs(sample.work_balance_error_j))
            records.append(sample)
            return
        position, velocity = values[:3], values[3:]
        elec_force, energy = observe(position)
        v_norm = _norm(velocity)
        kinetic = _scaled_ratio_product((0.5, spec.mass_kg, v_norm, v_norm))
        if gravity is None:
            force = elec_force
            interaction_j = energy.total_j
            total_j = kinetic + interaction_j
            drift = fsum((kinetic-energy0,
                          energy.direct_j-reference_energy.direct_j,
                          energy.induced_j-reference_energy.induced_j,
                          energy.background_j-reference_energy.background_j))
            if not all(isfinite(x) for x in (*position, *velocity, kinetic, drift)):
                raise MotionIntegrationError('motion data became nonfinite')
            sample = Motion2DSample(t, displacement, position, velocity,
                                    tuple(x/spec.mass_kg for x in force), force,
                                    kinetic, interaction_j, total_j, drift)
        else:
            assert grav_force is not None and local_gravity is not None and origin_gravity is not None
            from ._motion_validation import _gravity_planar_velocity_ok
            if tilted_plane:
                v_n_prefix = max(v_n_prefix, max(abs(velocity[i] * geometry.normal[i]) for i in range(3)))
            if not _gravity_planar_velocity_ok(velocity, geometry.normal, tilted_plane, v_n_prefix):
                raise UnsupportedProblem('velocity left the physically invariant plane')
            force = tuple(_exact_sum((a, b)) for a, b in zip(elec_force, grav_force))
            u_grav = gravity.potential_energy_at(position, mass_kg=spec.mass_kg)
            interaction_j = _exact_sum((energy.total_j, u_grav))
            total_j = _exact_sum((kinetic, energy.total_j, u_grav))
            # Check world-position displacement and gravitational delta strictly
            # against trajectory displacement / gravitational exchange (never
            # masking precision loss with constant launch kinetic energy).
            disp_norm = _norm(displacement)
            peak_disp = max(peak_disp, disp_norm)
            world_disp = _sub(position, p0)
            if _norm(_sub(world_disp, displacement)) > max(1e-6 * peak_disp, 256.0 * ulp(peak_disp)):
                raise MotionIntegrationError('world position roundoff lost displacement resolution')
            delta_u_grav = origin_gravity.potential_energy_at(displacement, mass_kg=spec.mass_kg)
            world_du_grav = local_gravity.potential_energy_at(position, mass_kg=spec.mass_kg)
            saved_du_grav = _exact_sum((u_grav, -ref_u_grav))
            peak_grav_exchange = max(peak_grav_exchange, abs(delta_u_grav), abs(world_du_grav))
            du_tol = max(1e-6 * peak_grav_exchange, 256.0 * ulp(peak_grav_exchange))
            if abs(world_du_grav - delta_u_grav) > du_tol:
                raise MotionIntegrationError('world position roundoff lost gravitational potential energy resolution')
            if abs(saved_du_grav - delta_u_grav) > du_tol:
                raise MotionIntegrationError('gravitational potential energy gauge lacks binary64 precision for trajectory energy changes')
            drift = _exact_sum((kinetic-energy0,
                          energy.direct_j-reference_energy.direct_j,
                          energy.induced_j-reference_energy.induced_j,
                          energy.background_j-reference_energy.background_j,
                          world_du_grav))
            v0_vec = spec.initial_velocity_m_s
            dv_vec = _sub(velocity, v0_vec)
            vsum_vec = tuple(velocity[i] + v0_vec[i] for i in range(3))
            exact_dk = _exact_sum(_scaled_ratio_product((0.5, spec.mass_kg, dv_vec[i], vsum_vec[i])) for i in range(3))
            saved_dk = _exact_sum((kinetic, -energy0))
            dk_scale = max(peak_grav_exchange, abs(exact_dk), abs(saved_dk), abs(_exact_sum((energy.total_j, -reference_energy.total_j))))
            if abs(saved_dk - exact_dk) > max(1e-6 * dk_scale, 256.0 * ulp(dk_scale)):
                raise MotionIntegrationError('kinetic energy lacks binary64 precision to resolve trajectory energy exchange')
            total_scale = max(peak_grav_exchange, abs(exact_dk), abs(saved_dk), abs(_exact_sum((energy.total_j, -reference_energy.total_j))))
            if abs(_exact_sum((total_j, -total_j0)) - drift) > max(1e-6 * total_scale, 256.0 * ulp(total_scale)):
                raise MotionIntegrationError('saved total energy lacks binary64 precision to match energy drift')
            if not all(isfinite(x) for x in (*position, *velocity, *force, kinetic, interaction_j, total_j, drift, u_grav)):
                raise MotionIntegrationError('motion data became nonfinite')
            sample = Motion2DSample(t, displacement, position, velocity,
                                    tuple(x/spec.mass_kg for x in force), force,
                                    kinetic, interaction_j, total_j, drift,
                                    elec_force, grav_force, energy.total_j, u_grav)
        max_drift = max(max_drift, abs(drift))
        records.append(sample)

    record(0., y)
    t = 0.
    t_acc = 0.
    t_comp = 0.
    h = spec.sample_interval_s/4
    steps_done = 0
    termination: Literal['duration', 'surface-guard', 'charge-guard'] = 'duration'
    targets = output_times(spec.duration_s, spec.sample_interval_s)
    if dynamics is not None:
        targets = tuple(sorted(set((*targets, *dynamics.switch_times))))

    for target in targets:
        while t < target-2e-15*spec.duration_s:
            if steps_done >= spec.max_internal_steps:
                raise MotionIntegrationError('maximum internal RK4 substeps exceeded')
            steps_done += 1
            position = _add(origin, y[:3])
            conductor_distance = geometry.clearance(position)
            point_distance = geometry.charge_clearance(position)
            near_charge = point_distance < conductor_distance
            distance = min(conductor_distance, point_distance)
            guard_kind = 'charge-guard' if near_charge else 'surface-guard'
            active_guard = geometry.charge_guard if near_charge else geometry.guard
            closeness = max(.02*active_guard, 1e-10*geometry.scale)
            if distance <= closeness:
                termination = guard_kind
                events.append(Motion2DEvent(guard_kind, t, position,
                              'Stopped before the fixed point charge; not an exact collision' if near_charge else
                              surface_description))
                break
            dt = min(h, target-t)
            if max_step_s is not None:
                dt = min(dt, max_step_s)  # Opt-in investigation resolution, not output spacing.
            if dynamics is not None:
                dt = min(dt, dynamics.max_step_s)
            a = rhs(y)[3:6]
            normal_speed, normal_accel = geometry.approach_speed(position, y[3:6], a)
            if geometry.stationary:
                # A conservative displacement bound limits leapfrogging an
                # ideal fixed point-charge singularity between RK stages.
                normal_speed = max(normal_speed, _norm(y[3:6]))
                normal_accel = max(normal_accel, _norm(a))
            if normal_speed or normal_accel:
                bound = normal_speed + normal_accel*dt
                if bound:
                    dt = min(dt, .5*distance/bound)
            if dt < max(1e-14*spec.duration_s, 1e-300):
                if distance <= max(2*closeness, 1e-8*geometry.scale):
                    termination = guard_kind
                    events.append(Motion2DEvent(guard_kind, t, position,
                                  'Stopped before the numerical approach cutoff; not an exact collision'))
                    break
                raise MotionIntegrationError('timestep collapsed before an approach guard')
            # Keep established world-coordinate stepping while it resolves each
            # increment. Rebase an axis before rounding can exceed the requested
            # relative accuracy of a half-step, including release from rest.
            next_origin = list(origin)
            coordinates = list(y[:3])
            half_dt = dt / 2
            for i in range(3):
                motion = half_dt * (abs(y[i+3]) + half_dt * abs(a[i]) / 2)
                spacing = ulp(position[i])
                # Rounding a moving coordinate can also create an apparent
                # transverse displacement in a certified oblique plane.
                unresolved = (spacing > spec.relative_tolerance * motion or
                              spacing * abs(geometry.normal[i]) > plane_tolerance * motion)
                if origin[i] == 0. and spacing > ulp(position[i]-p0[i]) and unresolved:
                    next_origin[i] = p0[i]
                    coordinates[i] = position[i] - p0[i]
            if tuple(next_origin) != origin:
                if use_comp:
                    rebase_delta = tuple(
                        -(next_origin[i] - origin[i]) if i < 3 else 0.0
                        for i in range(len(y))
                    )
                    y_acc, y_comp = _neumaier_add(y_acc, y_comp, rebase_delta)
                    y = tuple(a + c for a, c in zip(y_acc, y_comp))
                else:
                    y = (*coordinates, *y[3:])
                origin = tuple(next_origin)
                launch = _sub(p0, origin)
            try:
                if use_comp:
                    d_large = _rk4_delta(y, dt, rhs)
                    large_acc, large_comp = _neumaier_add(y_acc, y_comp, d_large)
                    large = tuple(a + c for a, c in zip(large_acc, large_comp))
                    d_half1 = _rk4_delta(y, .5*dt, rhs)
                    half_acc, half_comp = _neumaier_add(y_acc, y_comp, d_half1)
                    half = tuple(a + c for a, c in zip(half_acc, half_comp))
                    d_half2 = _rk4_delta(half, .5*dt, rhs)
                    fine_acc, fine_comp = _neumaier_add(half_acc, half_comp, d_half2)
                    fine = tuple(a + c for a, c in zip(fine_acc, fine_comp))
                else:
                    large = rk4_step(y, dt, rhs)
                    half = rk4_step(y, .5*dt, rhs)
                    fine = rk4_step(half, .5*dt, rhs)
            except (_SurfaceGuard, _ChargeGuard):
                h = .5*dt
                continue
            coarse_point, midpoint, endpoint = (_add(origin, s[:3]) for s in (large, half, fine))
            if (geometry.clearance(coarse_point) <= 0 or geometry.clearance(endpoint) <= 0 or
                    geometry.charge_clearance(coarse_point) <= 0 or
                    geometry.charge_clearance(endpoint) <= 0 or
                    geometry.charge_segment_clearance(position,midpoint) <= 0 or
                    geometry.charge_segment_clearance(midpoint,endpoint) <= 0):
                h = .5*dt
                continue
            if _norm(fine[3:6]) >= _NONRELATIVISTIC_SPEED_LIMIT_M_S:
                raise UnsupportedProblem('classical motion limit reached; speed must stay below 0.1c')
            position_scale = max(1e-11*geometry.scale, spec.relative_tolerance*
                                 max(geometry.scale, _norm(_sub(y[:3], launch)),
                                     _norm(_sub(fine[:3], launch))))
            velocity_scale = max(1e-10, spec.relative_tolerance*max(1., _norm(y[3:6]), _norm(fine[3:6])))
            error = max(_norm(_sub(large[:3],fine[:3]))/(15*position_scale),
                        _norm(_sub(large[3:6],fine[3:6]))/(15*velocity_scale))
            if dynamics is not None:
                error = max(error, dynamics.work_error(large, fine, y))
            if not isfinite(error):
                raise MotionIntegrationError('nonfinite RK4 integration error')
            if error > 1:
                h = max(.15*dt, dt*min(.8,.9*error**(-.2)))
                continue
            if step_observer is not None:
                # Only a numerically accepted step is offered. A monitor may
                # ask for subdivision or end the reported observation interval;
                # it never supplies a force law or alters default trajectories.
                decision = step_observer(t, y, dt, half, fine, rhs, origin)
                if decision is False:
                    h = .5*dt
                    continue
                if decision is not None:
                    t, y, termination = decision
                    y_acc, y_comp = y, (0.0,) * len(y)
                    t_acc, t_comp = t, 0.0
                    break
            if use_comp:
                t_sum = t_acc + dt
                t_comp += (t_acc - t_sum) + dt if abs(t_acc) >= abs(dt) else (dt - t_sum) + t_acc
                t_acc = t_sum
                t = t_acc + t_comp
                y_acc, y_comp = fine_acc, fine_comp
                y = fine
            else:
                t += dt
                y = fine
            h = min(spec.sample_interval_s,
                    max(.2*dt, dt*min(2.5,(.9/max(error,1e-10))**.2)))
        if termination != 'duration':
            if t > records[-1].time_s + 2e-15*spec.duration_s:
                record(t, y)
            break
        t = target
        t_acc, t_comp = target, 0.0
        if dynamics is not None:
            dynamics.at_boundary(t, _world_state(origin, y))
        record(t, y)
    return tuple(records), tuple(events), termination, max_drift


class _SurfaceGuard(Exception):
    pass


class _ChargeGuard(Exception):
    pass
