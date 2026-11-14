"""M19a test-charge motion: F=qE, U=qV in immutable prescribed fields.

Reuse the M17/M18 adaptive RK4 controller. Do not include the moving charge
in the field inventory, solve conductor induction, or project away forces.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import hypot, inf, isfinite

from ._motion_time import output_times
from .model import PointCharge, Vector3
from .motion_1d import MotionIntegrationError, _NONRELATIVISTIC_SPEED_LIMIT_M_S
from .motion_2d import _integrate_vector_motion, _perpendicular, _SurfaceGuard
from .observables import EnergyBreakdown
from .prescribed_fields import (
    PrescribedField, UniformElectricField, FixedPointField, UniformSphereField,
    UniformShellField, UniformRingAxisField, UniformDiskAxisField, UniformRodField,
    _dot, _sub, _mul, _number, _vector, _unit, _on_line,
)
from .solver import UnsupportedProblem


def _parallel(a, b):
    length = hypot(*a)
    return length == 0 or hypot(*_sub(a, _mul(b, _dot(a, b)))) <= 1e-11*length


def _in_plane(p, origin, normal, scale):
    delta = _sub(p, origin)
    return abs(_dot(delta, normal)) <= 1e-11*max(scale, hypot(*delta))


def _certify(spec):
    origin, normal, scale = spec.particle.position_m, spec.plane_normal, spec.length_scale_m
    if spec.axis is not None and not _parallel(spec.initial_velocity_m_s, spec.axis):
        raise UnsupportedProblem('axis-only field requires an axial launch velocity')
    if abs(_dot(spec.initial_velocity_m_s, normal)) > 1e-11*hypot(*spec.initial_velocity_m_s):
        raise UnsupportedProblem('initial velocity must lie in the physical motion plane')
    for source in spec.field.sources:
        if isinstance(source, UniformElectricField):
            valid = (_parallel(source.field_v_m, spec.axis) if spec.axis is not None else
                     abs(_dot(source.field_v_m, normal)) <= 1e-11*hypot(*source.field_v_m))
        elif isinstance(source, UniformRodField):
            direction = _sub(source.end_m, source.start_m)
            centre = tuple((x+y)/2 for x,y in zip(source.start_m,source.end_m))
            if spec.axis is not None:
                valid = (_on_line(centre, origin, spec.axis, scale) and
                         (_parallel(direction, spec.axis) or
                          abs(_dot(direction, spec.axis)) <= 1e-11*hypot(*direction)))
            else:
                valid = ((_in_plane(source.start_m,origin,normal,scale) and
                          _in_plane(source.end_m,origin,normal,scale)) or
                         (_in_plane(centre,origin,normal,scale) and _parallel(direction,normal)))
        else:
            centre = source.position_m if isinstance(source, FixedPointField) else source.center_m
            valid = (_on_line(centre, origin, spec.axis, scale) if spec.axis is not None else
                     _in_plane(centre, origin, normal, scale))
            if isinstance(source, (UniformRingAxisField, UniformDiskAxisField)):
                valid = valid and _parallel(source.axis, spec.axis)
        if not valid:
            raise UnsupportedProblem('fixed sources do not preserve the required physical axis/plane')


@dataclass(frozen=True, slots=True)
class TestChargeMotionSpec:
    field: PrescribedField
    particle: PointCharge
    mass_kg: float
    initial_velocity_m_s: Vector3
    duration_s: float
    sample_interval_s: float
    plane_normal: Vector3 | None = None
    relative_tolerance: float = 2e-7
    length_scale_m: float = 1.
    source_guard_m: float = 1e-6
    max_internal_steps: int = 100000

    def __post_init__(self):
        if not isinstance(self.field, PrescribedField) or not isinstance(self.particle, PointCharge):
            raise TypeError('expected PrescribedField and a separate test-particle PointCharge')
        _number(self.particle.charge_c, 'test charge', nonzero=True)
        for name in ('mass_kg','duration_s','sample_interval_s','relative_tolerance',
                     'length_scale_m','source_guard_m'):
            _number(getattr(self,name), name, positive=True)
        if self.relative_tolerance >= .05:
            raise ValueError('relative_tolerance must be below 0.05')
        if type(self.max_internal_steps) is not int or self.max_internal_steps < 10:
            raise ValueError('max_internal_steps must be an integer >= 10')
        output_times(self.duration_s,self.sample_interval_s)
        velocity = _vector(self.initial_velocity_m_s, 'initial velocity')
        if hypot(*velocity) >= _NONRELATIVISTIC_SPEED_LIMIT_M_S:
            raise UnsupportedProblem('classical test-charge speed must stay below 0.1c')
        object.__setattr__(self,'initial_velocity_m_s',velocity)
        normal = self.plane_normal
        if normal is None:
            normal = _perpendicular(self.axis) if self.axis is not None else (0.,1.,0.)
        normal = _unit(normal,'motion-plane normal')
        object.__setattr__(self,'plane_normal',normal)
        if self.axis is not None and abs(_dot(self.axis,normal)) > 1e-11:
            raise UnsupportedProblem('axis-only source axis must belong to the motion plane')
        _certify(self)
        if _FixedDomain(self).clearance(self.particle.position_m) <= 0:
            raise ValueError('initial position lies within a source singularity/discontinuity guard')
        # Establish the actual observation domain before integration starts.
        self.field.electric_field_at(self.particle.position_m)
        self.field.potential_at(self.particle.position_m)

    @property
    def axis(self):
        return next((s.axis for s in self.field.sources if isinstance(
            s,(UniformRingAxisField,UniformDiskAxisField))),None)

    @property
    def selected_charge(self):
        return self.particle


@dataclass(frozen=True, slots=True)
class TestChargeSample:
    time_s: float
    displacement_m: Vector3
    position_m: Vector3
    velocity_m_s: Vector3
    acceleration_m_s2: Vector3
    electric_field_v_m: Vector3
    potential_v: float
    force_n: Vector3
    kinetic_energy_j: float
    potential_energy_j: float
    total_energy_j: float
    energy_drift_j: float


@dataclass(frozen=True, slots=True)
class TestChargeEvent:
    kind: str
    time_s: float
    position_m: Vector3
    description: str


@dataclass(frozen=True, slots=True)
class TestChargeMotionResult:
    spec: TestChargeMotionSpec
    samples: tuple[TestChargeSample, ...]
    events: tuple[TestChargeEvent, ...]
    termination: str
    max_abs_energy_drift_j: float

    def as_dict(self):
        return dict(schema='conductor_sim.test_charge_motion.v1',
                    mode='test charge in prescribed static fields; no back-reaction',
                    energy_convention='U=q*V; source self/mutual energies excluded; no induced-energy half factor',
                    units=dict(time='s',position='m',velocity='m/s',acceleration='m/s^2',
                               electric_field='V/m',potential='V',force='N',energy='J'),
                    field=self.spec.field.as_dict(),particle=asdict(self.spec.particle),
                    mass_kg=self.spec.mass_kg,initial_velocity_m_s=self.spec.initial_velocity_m_s,
                    plane_normal=self.spec.plane_normal,axis=self.spec.axis,
                    duration_s=self.spec.duration_s,sample_interval_s=self.spec.sample_interval_s,
                    source_guard_m=self.spec.source_guard_m,length_scale_m=self.spec.length_scale_m,
                    method='Shared M17/M18 adaptive RK4 controller; F=qE from fixed sources',
                    termination=self.termination,max_abs_energy_drift_j=self.max_abs_energy_drift_j,
                    samples=[asdict(s) for s in self.samples],events=[asdict(e) for e in self.events])


def _point_segment_distance(point, start, end):
    d = _sub(end,start)
    dd = _dot(d,d)
    t = min(1.,max(0.,_dot(_sub(point,start),d)/dd)) if dd else 0.
    return hypot(*_sub(_sub(point,start),_mul(d,t)))


def _segment_distance(a,b,c,d):
    values = [_point_segment_distance(a,c,d),_point_segment_distance(b,c,d),
              _point_segment_distance(c,a,b),_point_segment_distance(d,a,b)]
    u,v,w = _sub(b,a),_sub(d,c),_sub(a,c)
    aa,bb,cc,dd,ee = _dot(u,u),_dot(u,v),_dot(v,v),_dot(u,w),_dot(v,w)
    determinant = aa*cc-bb*bb
    if determinant > 1e-14*aa*cc:
        s,t = (bb*ee-cc*dd)/determinant,(aa*ee-bb*dd)/determinant
        if 0 <= s <= 1 and 0 <= t <= 1:
            values.append(hypot(*(w[i]+s*u[i]-t*v[i] for i in range(3))))
    return min(values)


class _FixedDomain:
    """Geometry contract used by the existing controller, not a new integrator."""
    stationary = ()
    charge_guard = 0.

    def __init__(self,spec):
        self.spec = spec
        self.origin,self.normal = spec.particle.position_m,spec.plane_normal
        self.scale,self.guard = spec.length_scale_m,spec.source_guard_m

    def clearance(self,p):
        return min((s.clearance_at(p) for s in self.spec.field.sources),default=inf)-self.guard

    def charge_clearance(self,p):
        return inf

    def approach_speed(self,p,v,a):
        return hypot(*v),hypot(*a)

    def charge_segment_clearance(self,start,end):
        # Reuse the controller's conservative chord/subdivision hook for ALL
        # source singularities/jumps, so no step leaps across an excluded site.
        distances = []
        for s in self.spec.field.sources:
            if isinstance(s,FixedPointField):
                distances.append(_point_segment_distance(s.position_m,start,end))
            elif isinstance(s,UniformRodField):
                distances.append(_segment_distance(start,end,s.start_m,s.end_m))
            elif isinstance(s,UniformShellField):
                lo = _point_segment_distance(s.center_m,start,end)
                hi = max(hypot(*_sub(start,s.center_m)),hypot(*_sub(end,s.center_m)))
                distances.append(0. if lo <= s.radius_m <= hi else min(abs(lo-s.radius_m),abs(hi-s.radius_m)))
            elif isinstance(s,UniformDiskAxisField):
                z0,z1 = s._coordinate(start),s._coordinate(end)
                distances.append(0. if z0*z1 <= 0 else min(abs(z0),abs(z1)))
        return min(distances,default=inf)-self.guard


def integrate_test_charge(spec: TestChargeMotionSpec) -> TestChargeMotionResult:
    if not isinstance(spec,TestChargeMotionSpec):
        raise TypeError('spec must be TestChargeMotionSpec')
    domain = _FixedDomain(spec)
    charge = spec.particle.charge_c

    def observe(p):
        if domain.clearance(p) <= 0:
            raise _SurfaceGuard()
        if not _in_plane(p,domain.origin,domain.normal,domain.scale):
            raise UnsupportedProblem('particle left the certified physical plane')
        field = spec.field.electric_field_at(p)
        potential = spec.field.potential_at(p)
        force,energy = _mul(field,charge),charge*potential
        if not all(isfinite(x) for x in (*force,energy,*(x/spec.mass_kg for x in force))):
            raise MotionIntegrationError('test-charge force/energy/acceleration became nonfinite')
        if abs(_dot(force,domain.normal)) > 5e-9*hypot(*force):
            raise UnsupportedProblem('transverse force violates the certified motion plane')
        return force,EnergyBreakdown(0.,0.,energy,'Prescribed source: full qV, no induced half factor')

    rows,events,termination,drift = _integrate_vector_motion(
        spec,domain,observe,surface_description=
        'Stopped before a fixed-source singularity or field jump; not an exact collision')
    samples = tuple(TestChargeSample(
        s.time_s,s.displacement_m,s.position_m,s.velocity_m_s,s.acceleration_m_s2,
        spec.field.electric_field_at(s.position_m),spec.field.potential_at(s.position_m),
        s.force_n,s.kinetic_energy_j,s.interaction_energy_j,s.total_energy_j,s.energy_drift_j)
        for s in rows)
    return TestChargeMotionResult(spec,samples,tuple(TestChargeEvent(
        'source-guard',e.time_s,e.position_m,e.description) for e in events),
        'source-guard' if termination!='duration' else 'duration',drift)
