"""Fixed, nonresponding electrostatic sources (SI), separate from conductors.

Ring/disc formulas are deliberately axis-only. They must never be used to
paint off-axis field maps. Charged material is not solved as an equipotential.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import copysign, fsum, hypot, inf, isfinite, log1p, pi, ulp
from typing import Protocol

from .model import EPSILON_0, Vector3
from .solver import SingularField, UnsupportedProblem

ZERO = (0., 0., 0.)


def _number(value, name, *, positive=False, nonzero=False):
    if (type(value) not in (int, float) or not isfinite(value) or
            (positive and value <= 0) or (nonzero and value == 0)):
        raise ValueError(f'{name} must be finite' +
                         (' and positive' if positive else ' and nonzero' if nonzero else ''))
    return float(value)


def _vector(value, name):
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f'{name} must contain three finite numbers')
    return tuple(_number(x, name) for x in value)


def _sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def _dot(a, b):
    return fsum(x*y for x, y in zip(a, b))


def _mul(a, scale):
    return tuple(scale*x for x in a)


def _unit(value, name):
    v = _vector(value, name)
    length = hypot(*v)
    if not isfinite(length) or length == 0:
        raise ValueError(f'{name} must have finite nonzero length')
    return tuple(x/length for x in v)


def _on_line(point, origin, axis, scale):
    d = _sub(point, origin)
    s = _dot(d, axis)
    return hypot(*_sub(d, _mul(axis, s))) <= 1e-11*max(scale, hypot(*d))


class ElectricFieldProvider(Protocol):
    """The shared observation contract; no probe charge enters these methods."""
    def potential_at(self, point_m: Vector3) -> float: ...
    def electric_field_at(self, point_m: Vector3) -> Vector3: ...


class _FixedSource:
    def _sample(self, point_m, permittivity_f_m, need_field):
        p = _vector(point_m, 'sample position')
        eps = _number(permittivity_f_m, 'permittivity', positive=True)
        potential, field = self._values(p, 1/(4*pi*eps), need_field)
        if not all(isfinite(x) for x in (potential, *field)):
            raise ValueError('fixed-source observation overflowed or became nonfinite')
        return potential, field

    def potential_at(self, point_m, *, permittivity_f_m=EPSILON_0):
        return self._sample(point_m, permittivity_f_m, False)[0]

    def electric_field_at(self, point_m, *, permittivity_f_m=EPSILON_0):
        return self._sample(point_m, permittivity_f_m, True)[1]

    def clearance_at(self, point_m):
        """Distance to a singularity/field jump, not an object collision radius."""
        return inf

    def as_dict(self):
        return dict(kind=type(self).__name__, **asdict(self))


@dataclass(frozen=True, slots=True)
class UniformElectricField(_FixedSource):
    field_v_m: Vector3
    reference_point_m: Vector3 = ZERO
    reference_potential_v: float = 0.

    def __post_init__(self):
        for name in ('field_v_m', 'reference_point_m'):
            object.__setattr__(self, name, _vector(getattr(self, name), name))
        _number(self.reference_potential_v, 'reference potential')

    def _values(self, p, k, need_field):
        return (self.reference_potential_v-_dot(self.field_v_m, _sub(p, self.reference_point_m)),
                self.field_v_m)


@dataclass(frozen=True, slots=True)
class FixedPointField(_FixedSource):
    charge_c: float
    position_m: Vector3 = ZERO

    def __post_init__(self):
        _number(self.charge_c, 'source charge', nonzero=True)
        object.__setattr__(self, 'position_m', _vector(self.position_m, 'source position'))

    def _values(self, p, k, need_field):
        d = _sub(p, self.position_m)
        r = hypot(*d)
        if r == 0:
            raise SingularField('fixed point-source position is singular')
        return k*self.charge_c/r, _mul(d, (k*self.charge_c/r)/r/r)

    def clearance_at(self, point_m):
        return hypot(*_sub(point_m, self.position_m))


@dataclass(frozen=True, slots=True)
class _RadialDistribution(_FixedSource):
    charge_c: float
    radius_m: float
    center_m: Vector3 = ZERO

    def __post_init__(self):
        _number(self.charge_c, 'source charge', nonzero=True)
        _number(self.radius_m, 'source radius', positive=True)
        object.__setattr__(self, 'center_m', _vector(self.center_m, 'source centre'))


@dataclass(frozen=True, slots=True)
class UniformSphereField(_RadialDistribution):
    """Uniform fixed volume charge, NOT a conducting solid sphere.

    Material is mechanically transparent in this ideal field model. The field
    is continuous at the radius, so a test particle may cross it.
    """
    def _values(self, p, k, need_field):
        d = _sub(p, self.center_m)
        r, a = hypot(*d), self.radius_m
        if r <= a:
            return k*self.charge_c/(2*a)*(3-(r/a)**2), _mul(d, k*self.charge_c/a/a/a)
        return k*self.charge_c/r, _mul(d, (k*self.charge_c/r)/r/r)


@dataclass(frozen=True, slots=True)
class UniformShellField(_RadialDistribution):
    """Uniform fixed surface charge; motion stops before the field jump."""
    def _values(self, p, k, need_field):
        d = _sub(p, self.center_m)
        r = hypot(*d)
        if abs(r-self.radius_m) <= 8*ulp(max(r,self.radius_m)) and need_field:
            raise SingularField('thin-shell surface field is discontinuous')
        field = ZERO if r <= self.radius_m else _mul(d, (k*self.charge_c/r)/r/r)
        return k*self.charge_c/max(r, self.radius_m), field

    def clearance_at(self, point_m):
        return abs(hypot(*_sub(point_m, self.center_m))-self.radius_m)


@dataclass(frozen=True, slots=True)
class _AxialDistribution(_RadialDistribution):
    axis: Vector3 = (0., 0., 1.)

    def __post_init__(self):
        _RadialDistribution.__post_init__(self)
        object.__setattr__(self, 'axis', _unit(self.axis, 'source axis'))

    def _coordinate(self, p):
        if not _on_line(p, self.center_m, self.axis, self.radius_m):
            raise UnsupportedProblem('ring/disc source supports only its symmetry axis; off-axis fields are not implemented')
        return _dot(_sub(p, self.center_m), self.axis)


@dataclass(frozen=True, slots=True)
class UniformRingAxisField(_AxialDistribution):
    """Thin ring: V=kQ/sqrt(a^2+z^2), E_z=-dV/dz, ONLY on its axis."""
    def _values(self, p, k, need_field):
        z = self._coordinate(p)
        h = hypot(self.radius_m, z)
        return k*self.charge_c/h, _mul(self.axis, k*self.charge_c*(z/h)/h/h)


@dataclass(frozen=True, slots=True)
class UniformDiskAxisField(_AxialDistribution):
    """Uniform fixed disc surface charge, ONLY on its axis, not a metal disc."""
    def _values(self, p, k, need_field):
        z = self._coordinate(p)
        h = hypot(self.radius_m, z)
        potential = 2*k*self.charge_c/(h+abs(z))
        if z == 0:
            if need_field:
                raise SingularField('thin-disc surface field is discontinuous')
            return potential, ZERO
        return potential, _mul(self.axis, copysign(1., z)*2*k*self.charge_c/h/(h+abs(z)))

    def clearance_at(self, point_m):
        return abs(self._coordinate(point_m))


@dataclass(frozen=True, slots=True)
class UniformRodField(_FixedSource):
    """Finite uniformly charged line segment; exact field off the segment."""
    charge_c: float
    start_m: Vector3
    end_m: Vector3

    def __post_init__(self):
        _number(self.charge_c, 'source charge', nonzero=True)
        for name in ('start_m', 'end_m'):
            object.__setattr__(self, name, _vector(getattr(self, name), name))
        _number(hypot(*_sub(self.end_m, self.start_m)), 'rod length', positive=True)

    def _coordinates(self, p):
        direction = _sub(self.end_m, self.start_m)
        length = hypot(*direction)
        n = tuple(x/length for x in direction)
        delta = _sub(p, self.start_m)
        s = _dot(delta, n)
        transverse = _sub(delta, _mul(n, s))
        rho = hypot(*transverse)
        if rho <= 16*ulp(max(length,hypot(*delta))):
            transverse,rho = ZERO,0.
        return length, n, s, transverse, rho

    def _values(self, p, k, need_field):
        length, n, s, transverse, rho = self._coordinates(p)
        u = s-length
        if rho == 0 and 0 <= s <= length:
            raise SingularField('uniform line-charge segment is singular')
        d0, d1 = hypot(rho, s), hypot(rho, u)
        # Rationalized sum avoids cancellation close to the rod; log1p is
        # accurate far away. Axial E uses the rationalized reciprocal difference.
        gap = (rho*(rho/(d0+abs(s))) + rho*(rho/(d1+abs(u))) +
               2*max(-s, u, 0.))
        potential = k*self.charge_c/length*log1p(2*length/gap)
        axial = k*self.charge_c*(s+u)/d0/d1/(d0+d1)
        if rho == 0:
            radial_factor = 0.
        elif s*u > 0:
            radial_factor = k*self.charge_c*(s+u)/d0/d1/(s*d1+u*d0)
        else:
            radial_factor = k*self.charge_c/length*(s/d0-u/d1)/rho/rho
        return potential, tuple(axial*n[i]+radial_factor*transverse[i] for i in range(3))

    def clearance_at(self, point_m):
        length, _, s, _, rho = self._coordinates(point_m)
        return hypot(rho, max(-s, s-length, 0.))


SOURCE_TYPES = (UniformElectricField, FixedPointField, UniformSphereField,
                UniformShellField, UniformRingAxisField, UniformDiskAxisField, UniformRodField)


@dataclass(frozen=True, slots=True)
class PrescribedField:
    """Linear superposition of fixed sources in one homogeneous permittivity.

    This is not a conductor-equilibrium solve. Sources never see the probe.
    Only certified built-in source geometries are accepted for now.
    """
    sources: tuple
    permittivity_f_m: float = EPSILON_0

    def __post_init__(self):
        sources = tuple(self.sources)
        if len(sources) > 64 or any(type(s) not in SOURCE_TYPES for s in sources):
            raise UnsupportedProblem('prescribed field requires at most 64 supported fixed sources, not conductor objects')
        object.__setattr__(self, 'sources', sources)
        _number(self.permittivity_f_m, 'permittivity', positive=True)

    def potential_at(self, point_m):
        point = _vector(point_m, 'sample position')
        value = fsum(s.potential_at(point, permittivity_f_m=self.permittivity_f_m) for s in self.sources)
        if not isfinite(value):
            raise ValueError('superposed potential is not finite')
        return value

    def electric_field_at(self, point_m):
        point = _vector(point_m, 'sample position')
        fields = [s.electric_field_at(point, permittivity_f_m=self.permittivity_f_m) for s in self.sources]
        value = tuple(fsum(f[i] for f in fields) for i in range(3))
        if not all(isfinite(x) for x in value):
            raise ValueError('superposed electric field is not finite')
        return value

    def as_dict(self):
        return dict(schema='conductor_sim.prescribed_field.v1',
                    model='fixed nonresponding sources; no conductor induction',
                    permittivity_f_m=self.permittivity_f_m,
                    sources=[s.as_dict() for s in self.sources])
