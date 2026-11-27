"""Exact Kelvin-image reference for an EXTERNAL point charge and a grounded solid sphere.

This is not the interior-cavity image method: for a source distance d > a,
the image is -q*a/d at a*a/d INSIDE the physical sphere. The conductor is
solid, not a hollow SphericalShell, and supports no interior charge sources.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import hypot, isclose, isfinite, pi, sin, sqrt
from typing import Literal

from .model import ElectrostaticProblem, Grounded, PointCharge, SolidSphere, Vector3, _position
from .solver import ConductorResult, SingularField, UnsupportedProblem


def _sphere_strictly_outside(point: Vector3, center: Vector3, radius: float) -> bool:
    """Exact |point - center|^2 > radius^2 check, immune to hypot rounding near surface."""
    delta = tuple(point[i] - center[i] for i in range(3))
    if all(isfinite(x) for x in delta):
        d = hypot(*delta)
        if isfinite(d):
            if d > radius * (1.0 + 1e-12):
                return True
            if d < radius * (1.0 - 1e-12):
                return False
    f_delta = tuple(Fraction(point[i]) - Fraction(center[i]) for i in range(3))
    return sum(x * x for x in f_delta) > Fraction(radius) * Fraction(radius)


def _diff(a: Vector3, b: Vector3) -> Vector3:
    return tuple(a[i] - b[i] for i in range(3))


def _norm(v: Vector3) -> float:
    # hypot is free of intermediate overflow/underflow at any admitted scale.
    return hypot(*v)


def _chord(u: Vector3, n: Vector3) -> float:
    """|u-n| for unit vectors: 2*sin(theta/2), accurate even as theta -> 0."""
    return hypot(*(u[i]-n[i] for i in range(3)))


def _field(k: float, q: float, v: Vector3, r: float) -> Vector3:
    """k*q*v/r**3 for |v| = r, formed as (k*(q/r))/r times the unit vector.

    Forming k*(q/r) first means neither k*q nor q/r**2 can overflow or
    underflow while the field itself is a normal float.
    """
    scale = (k*(q/r))/r
    return tuple(scale*(x/r) for x in v)


def _pair_force(k: float, q: float, q_other: float, v: Vector3, r: float) -> Vector3:
    """Coulomb force k*q*q_other*v/r**3 as (k*(q/r))*(q_other/r)*(v/r)."""
    scale = (k*(q/r))*(q_other/r)
    return tuple(scale*(x/r) for x in v)


@dataclass(frozen=True, slots=True)
class ExternalGroundedSphereSolution:
    sphere: SolidSphere
    source: PointCharge
    permittivity_f_per_m: float
    conductor_results: tuple[ConductorResult, ...]

    @property
    def symmetry_axis(self) -> Vector3:
        dvec = _diff(self.source.position_m, self.sphere.center_m)
        d = _norm(dvec)
        return tuple(x/d for x in dvec)

    @property
    def source_distance_m(self) -> float:
        return _norm(_diff(self.source.position_m, self.sphere.center_m))

    @property
    def image_charge_c(self) -> float:
        return -self.source.charge_c*(self.sphere.radius_m/self.source_distance_m)

    @property
    def image_position_m(self) -> Vector3:
        a, d, n = self.sphere.radius_m, self.source_distance_m, self.symmetry_axis
        return tuple(self.sphere.center_m[i] + a*(a/d)*n[i] for i in range(3))

    @property
    def _k(self) -> float:
        return 1/(4*pi*self.permittivity_f_per_m)

    def potential_at(self, point_m: Vector3) -> float:
        point = _position(point_m, 'sample point')
        r = _norm(_diff(point, self.sphere.center_m))
        if r <= self.sphere.radius_m:
            return 0.0
        source_distance = _norm(_diff(point, self.source.position_m))
        if source_distance == 0:
            raise SingularField('point-charge potential is singular at the external source')
        return self._k*(self.source.charge_c/source_distance+
                        self.image_charge_c/_norm(_diff(point,self.image_position_m)))

    def electric_field_at(self, point_m: Vector3) -> Vector3:
        point = _position(point_m, 'sample point')
        r = _norm(_diff(point, self.sphere.center_m))
        if isclose(r, self.sphere.radius_m, rel_tol=1e-12):
            raise SingularField('field has distinct one-sided limits at sphere surface')
        if r < self.sphere.radius_m:
            return (0.0,0.0,0.0)
        direct = _diff(point, self.source.position_m)
        dist = _norm(direct)
        if dist == 0:
            raise SingularField('electric field is singular at the external charge')
        reflected = _diff(point, self.image_position_m)
        r_img = _norm(reflected)
        source_term = _field(self._k, self.source.charge_c, direct, dist)
        image_term = _field(self._k, self.image_charge_c, reflected, r_img)
        return tuple(source_term[i]+image_term[i] for i in range(3))

    def surface_charge_density(self, surface: Literal['outer'], theta_rad: float) -> float:
        from math import isfinite
        if surface != 'outer':
            raise ValueError('solid sphere has only an outer conducting surface')
        if not isfinite(theta_rad) or not 0 <= theta_rad <= pi:
            raise ValueError('angle must lie in [0, pi]')
        return self._density_from_chord(2*sin(theta_rad/2))

    def _density_from_chord(self, chord: float) -> float:
        """Grounded Kelvin density at the surface point with |u-n| = chord.

        |a*u - d*n|**2 = (d-a)**2 + a*d*chord**2 avoids cancellation near
        contact; the ratios avoid forming length**3 at extreme scales.
        """
        a,d,q = self.sphere.radius_m, self.source_distance_m, self.source.charge_c
        gap = d-a
        length = hypot(gap, sqrt(d)*sqrt(a)*chord)
        return -(q/(4*pi*a))*(gap/length)*((d+a)/length)/length

    def surface_charge_density_at(self, surface: Literal['outer'], point_m: Vector3) -> float:
        point = _position(point_m,'surface point')
        delta = _diff(point,self.sphere.center_m)
        radius = _norm(delta)
        if not isclose(radius,self.sphere.radius_m,rel_tol=1e-10,abs_tol=1e-13):
            raise ValueError('point must lie on sphere surface')
        if surface != 'outer':
            raise ValueError('solid sphere has only an outer conducting surface')
        # Evaluate at the intended surface point: the radial projection of
        # the admitted query, through the chord to the source direction.
        unit = tuple(x/radius for x in delta)
        return self._density_from_chord(_chord(unit, self.symmetry_axis))

    @property
    def induced_force_on_charge_n(self) -> Vector3:
        """Exact physical force on source due to the grounded conductor, excluding self-field."""
        v = _diff(self.source.position_m, self.image_position_m)
        r = _norm(v)
        d = self.source_distance_m
        a = self.sphere.radius_m
        if r == 0.0 or d - a < 0.01 * a:
            from .observables import _scaled_ratio_product
            delta = _diff(self.source.position_m, self.sphere.center_m)
            f_delta = tuple(Fraction(self.source.position_m[i]) - Fraction(self.sphere.center_m[i]) for i in range(3))
            f_d2 = sum(x * x for x in f_delta)
            f_a2 = Fraction(a) * Fraction(a)
            u_denom = float((f_d2 - f_a2) / f_d2)
            mag = _scaled_ratio_product((-self._k, self.source.charge_c, self.source.charge_c, a / d), (d, d, u_denom, u_denom))
            return tuple(mag * (delta[i] / d) for i in range(3))
        return _pair_force(self._k, self.source.charge_c, self.image_charge_c, v, r)

    @property
    def force_on_charge_n(self) -> Vector3:
        """Total physical force on the charge due to the conducting sphere."""
        return self.induced_force_on_charge_n

    @property
    def interaction_energy_j(self) -> float:
        """Induced-charge interaction energy, with its physical factor 1/2."""
        r = _norm(_diff(self.source.position_m, self.image_position_m))
        d = self.source_distance_m
        a = self.sphere.radius_m
        if r == 0.0 or d - a < 0.01 * a:
            from .observables import _scaled_ratio_product
            f_delta = tuple(Fraction(self.source.position_m[i]) - Fraction(self.sphere.center_m[i]) for i in range(3))
            f_d2 = sum(x * x for x in f_delta)
            f_a2 = Fraction(a) * Fraction(a)
            u_denom = float((f_d2 - f_a2) / f_d2)
            return _scaled_ratio_product((-0.5, self._k, self.source.charge_c, self.source.charge_c, a / d), (d, u_denom))
        return (self._k * (self.source.charge_c / r)) * self.image_charge_c / 2


class ExternalGroundedSphereBackend:
    """Exact solution for one nonzero point charge outside one grounded solid sphere."""

    def supports(self, problem: ElectrostaticProblem) -> bool:
        if len(problem.conductors) != 1 or len(problem.charges) != 1:
            return False
        sphere, source = problem.conductors[0], problem.charges[0]
        return (isinstance(sphere, SolidSphere) and isinstance(sphere.constraint, Grounded)
                and source.charge_c != 0
                and _sphere_strictly_outside(source.position_m, sphere.center_m, sphere.radius_m))

    def solve(self, problem: ElectrostaticProblem) -> ExternalGroundedSphereSolution:
        if not self.supports(problem):
            raise UnsupportedProblem(
                'external grounded-sphere solution needs exactly one nonzero point '
                'charge strictly outside one grounded solid sphere; isolated, '
                'interior, touching and multi-source problems are not supported')
        sphere,source=problem.conductors[0],problem.charges[0]
        d=_norm(_diff(source.position_m,sphere.center_m))
        induced=-source.charge_c*(sphere.radius_m/d)
        observed=ConductorResult(sphere.name,0.0,induced,0.0,induced)
        return ExternalGroundedSphereSolution(sphere,source,problem.permittivity_f_per_m,(observed,))
