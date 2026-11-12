"""Exact Kelvin-image reference for an EXTERNAL point charge and a grounded solid sphere.

This is not the interior-cavity image method: for a source distance d > a,
the image is -q*a/d at a*a/d INSIDE the physical sphere. The conductor is
solid, not a hollow SphericalShell, and supports no interior charge sources.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, isclose, pi, sqrt
from typing import Literal

from .model import ElectrostaticProblem, Grounded, PointCharge, SolidSphere, Vector3, _position
from .solver import ConductorResult, SingularField, UnsupportedProblem


def _diff(a: Vector3, b: Vector3) -> Vector3:
    return tuple(a[i] - b[i] for i in range(3))


def _norm(v: Vector3) -> float:
    return sqrt(sum(x*x for x in v))


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
        return -self.source.charge_c*self.sphere.radius_m/self.source_distance_m

    @property
    def image_position_m(self) -> Vector3:
        a, d, n = self.sphere.radius_m, self.source_distance_m, self.symmetry_axis
        return tuple(self.sphere.center_m[i] + a*a/d*n[i] for i in range(3))

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
        return tuple(self._k*(self.source.charge_c*direct[i]/dist**3+
                              self.image_charge_c*reflected[i]/r_img**3) for i in range(3))

    def surface_charge_density(self, surface: Literal['outer'], theta_rad: float) -> float:
        from math import isfinite
        if surface != 'outer':
            raise ValueError('solid sphere has only an outer conducting surface')
        if not isfinite(theta_rad) or not 0 <= theta_rad <= pi:
            raise ValueError('angle must lie in [0, pi]')
        a,d,q = self.sphere.radius_m, self.source_distance_m, self.source.charge_c
        length = sqrt(d*d+a*a-2*d*a*cos(theta_rad))
        return -q*(d*d-a*a)/(4*pi*a*length**3)

    def surface_charge_density_at(self, surface: Literal['outer'], point_m: Vector3) -> float:
        point = _position(point_m,'surface point')
        delta = _diff(point,self.sphere.center_m)
        radius = _norm(delta)
        if not isclose(radius,self.sphere.radius_m,rel_tol=1e-10,abs_tol=1e-13):
            raise ValueError('point must lie on sphere surface')
        dot = sum(delta[i]*self.symmetry_axis[i] for i in range(3))/radius
        from math import acos
        return self.surface_charge_density(surface,acos(max(-1.,min(1.,dot))))

    @property
    def induced_force_on_charge_n(self) -> Vector3:
        """Exact physical force on source due to the grounded conductor, excluding self-field."""
        v = _diff(self.source.position_m,self.image_position_m)
        r = _norm(v)
        multiplier = self._k*self.source.charge_c*self.image_charge_c/r**3
        return tuple(multiplier*x for x in v)

    @property
    def interaction_energy_j(self) -> float:
        """Induced-charge interaction energy, with its physical factor 1/2."""
        return self._k*self.source.charge_c*self.image_charge_c/(2*_norm(
            _diff(self.source.position_m,self.image_position_m)))


class ExternalGroundedSphereBackend:
    """Exact solution for one nonzero point charge outside one grounded solid sphere."""

    def supports(self, problem: ElectrostaticProblem) -> bool:
        if len(problem.conductors) != 1 or len(problem.charges) != 1:
            return False
        sphere, source = problem.conductors[0],problem.charges[0]
        return (isinstance(sphere,SolidSphere) and isinstance(sphere.constraint,Grounded)
                and source.charge_c != 0
                and _norm(_diff(source.position_m,sphere.center_m)) > sphere.radius_m)

    def solve(self, problem: ElectrostaticProblem) -> ExternalGroundedSphereSolution:
        if not self.supports(problem):
            raise UnsupportedProblem(
                'external grounded-sphere solution needs exactly one nonzero point '
                'charge strictly outside one grounded solid sphere; isolated, '
                'interior, touching and multi-source problems are not supported')
        sphere,source=problem.conductors[0],problem.charges[0]
        d=_norm(_diff(source.position_m,sphere.center_m))
        induced=-source.charge_c*sphere.radius_m/d
        observed=ConductorResult(sphere.name,0.0,induced,0.0,induced)
        return ExternalGroundedSphereSolution(sphere,source,problem.permittivity_f_per_m,(observed,))
