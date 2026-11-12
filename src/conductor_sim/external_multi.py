"""Exact Kelvin-image superposition for multiple charges outside one solid sphere.

No FEMM calls. Supports grounded and fixed-net-charge metal and arbitrarily
oriented charge positions. Image charges are mathematical devices INSIDE metal,
not physical point-charge components supplied by the author.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import acos, cos, fsum, hypot, isclose, isfinite, pi, sin
from typing import Literal

from .model import (ElectrostaticProblem, FixedNetCharge, Grounded, PointCharge,
                    SolidSphere, Vector3, _position)
from .solver import ConductorResult, SingularField, UnsupportedProblem


def difference(a: Vector3, b: Vector3) -> Vector3:
    return tuple(a[i]-b[i] for i in range(3))


def norm(a: Vector3) -> float:
    return hypot(*a)


def dot(a: Vector3, b: Vector3) -> float:
    return fsum(a[i]*b[i] for i in range(3))


def active_external_sites(problem: ElectrostaticProblem) -> tuple[PointCharge, ...]:
    """Combine exactly coincident authored sites and remove cancelled sites.

    Physical validity of each originally authored charge is checked separately,
    including sites that cancel; malformed geometry cannot be laundered by zero.
    """
    amounts: dict[Vector3, list[float]] = {}
    for item in problem.charges:
        amounts.setdefault(item.position_m, []).append(item.charge_c)
    try:
        sites = tuple(PointCharge(fsum(parts), position)
                      for position, parts in sorted(amounts.items()))
    except (ValueError, OverflowError) as exc:
        raise UnsupportedProblem('nonfinite combined external point charges') from exc
    return tuple(site for site in sites if site.charge_c != 0)


def external_sphere_supported(problem: ElectrostaticProblem, *, min_components: int = 2) -> bool:
    if len(problem.conductors) != 1 or len(problem.charges) < min_components:
        return False
    sphere = problem.conductors[0]
    if not (isinstance(sphere, SolidSphere)
            and isinstance(sphere.constraint, (Grounded, FixedNetCharge))):
        return False
    if any(not isfinite(norm(difference(q.position_m, sphere.center_m)))
           or norm(difference(q.position_m, sphere.center_m)) <= sphere.radius_m
           for q in problem.charges):
        return False
    try:
        sites = active_external_sites(problem)
        images_total = fsum(-p.charge_c*sphere.radius_m /
                            norm(difference(p.position_m, sphere.center_m))
                            for p in sites)
        Q = (0.0 if isinstance(sphere.constraint, Grounded)
             else sphere.constraint.charge_c)
        central = 0.0 if isinstance(sphere.constraint, Grounded) else Q-images_total
        potential = central/(4*pi*problem.permittivity_f_per_m*sphere.radius_m)
        return (0 < len(sites) <= 32 and isfinite(images_total)
                and isfinite(central) and isfinite(potential))
    except (ValueError, OverflowError, ZeroDivisionError, UnsupportedProblem):
        return False


@dataclass(frozen=True, slots=True)
class ExternalMultiSphereSolution:
    """Exact image-charge result; charge indexing refers to active grouped sites."""
    sphere: SolidSphere
    sites: tuple[PointCharge, ...]
    permittivity_f_per_m: float
    conductor_results: tuple[ConductorResult, ...]

    @property
    def _k(self) -> float:
        return 1/(4*pi*self.permittivity_f_per_m)

    @property
    def image_sources(self) -> tuple[PointCharge, ...]:
        a = self.sphere.radius_m
        center = self.sphere.center_m
        images = []
        for site in self.sites:
            delta = difference(site.position_m, center)
            d = norm(delta)
            images.append(PointCharge(-site.charge_c*a/d,
                                      tuple(center[i] + (a/d)**2*delta[i] for i in range(3))))
        return tuple(images)

    @property
    def central_image_charge_c(self) -> float:
        if isinstance(self.sphere.constraint, Grounded):
            return 0.0
        return self.sphere.constraint.charge_c - fsum(q.charge_c for q in self.image_sources)

    @property
    def conductor_potential_v(self) -> float:
        return self.conductor_results[0].potential_v

    def potential_at(self, point_m: Vector3) -> float:
        pt = _position(point_m, 'sample point')
        r = norm(difference(pt, self.sphere.center_m))
        if r <= self.sphere.radius_m:
            return self.conductor_potential_v
        contributions = []
        for q in (*self.sites, *self.image_sources):
            distance = norm(difference(pt, q.position_m))
            if distance == 0:
                raise SingularField('potential is singular at an external point charge')
            contributions.append(q.charge_c/distance)
        central = self.central_image_charge_c
        if central:
            contributions.append(central/r)
        return self._k*fsum(contributions)

    def electric_field_at(self, point_m: Vector3) -> Vector3:
        pt = _position(point_m, 'sample point')
        delta = difference(pt, self.sphere.center_m)
        r = norm(delta)
        if isclose(r, self.sphere.radius_m, rel_tol=1e-12):
            raise SingularField('electric field jumps at the conducting surface')
        if r < self.sphere.radius_m:
            return (0.0, 0.0, 0.0)
        images = self.image_sources
        terms = [[], [], []]
        for q in (*self.sites, *images):
            displacement = difference(pt, q.position_m)
            distance = norm(displacement)
            if distance == 0:
                raise SingularField('electric field is singular at an external point charge')
            for i in range(3):
                terms[i].append(q.charge_c*displacement[i]/distance**3)
        central = self.central_image_charge_c
        if central:
            for i in range(3):
                terms[i].append(central*delta[i]/r**3)
        return tuple(self._k*fsum(term) for term in terms)

    def surface_charge_density_at(self, surface: Literal['outer'], point_m: Vector3) -> float:
        if surface != 'outer':
            raise ValueError('solid sphere has only an outer conducting surface')
        pt = _position(point_m, 'surface point')
        delta = difference(pt, self.sphere.center_m)
        a = self.sphere.radius_m
        if not isclose(norm(delta), a, rel_tol=1e-10, abs_tol=1e-13):
            raise ValueError('sample must lie on the conducting sphere')
        # Each grounded Kelvin pair contributes independently; the central
        # compensation contributes spatially uniform sigma.
        values = []
        for site in self.sites:
            d = norm(difference(site.position_m, self.sphere.center_m))
            distance = norm(difference(pt, site.position_m))
            values.append(-site.charge_c*(d*d-a*a)/(4*pi*a*distance**3))
        values.append(self.central_image_charge_c/(4*pi*a*a))
        return fsum(values)

    def surface_charge_density(self, surface: Literal['outer'], theta_rad: float) -> float:
        """Polar profile only when all sites lie on a common central axis.

        In a general 3D arrangement there is NO unique one-dimensional angular
        profile. Use surface_charge_density_at(face,world_point) instead.
        """
        if surface != 'outer':
            raise ValueError('solid sphere has only an outer face')
        if not isfinite(theta_rad) or not 0 <= theta_rad <= pi:
            raise ValueError('theta must be in [0,pi]')
        far = max(self.sites, key=lambda p: (norm(difference(p.position_m,self.sphere.center_m)),
                                             p.position_m))
        delta = difference(far.position_m,self.sphere.center_m)
        d = norm(delta)
        axis = tuple(x/d for x in delta)
        for site in self.sites:
            displacement = difference(site.position_m,self.sphere.center_m)
            z = dot(displacement,axis)
            if norm(tuple(displacement[i]-z*axis[i] for i in range(3))) > 1e-11*self.sphere.radius_m:
                raise UnsupportedProblem('no unique polar density profile for non-collinear sources; use surface_charge_density_at')
        ref = (1.,0.,0.) if abs(axis[0]) < .9 else (0.,1.,0.)
        projection = dot(ref,axis)
        transverse = tuple(ref[i]-projection*axis[i] for i in range(3))
        length = norm(transverse)
        perpendicular = tuple(x/length for x in transverse)
        a = self.sphere.radius_m
        point = tuple(self.sphere.center_m[i] + a*(cos(theta_rad)*axis[i]+sin(theta_rad)*perpendicular[i]) for i in range(3))
        return self.surface_charge_density_at(surface,point)

    def conductor_force_on_charge_n(self, index: int) -> Vector3:
        """Force on an active external site from the METAL only, no direct pair force."""
        site = self.sites[index]
        pt = site.position_m
        contributions = [[],[],[]]
        for image in self.image_sources:
            delta = difference(pt,image.position_m)
            r = norm(delta)
            for k in range(3):
                contributions[k].append(image.charge_c*delta[k]/r**3)
        central = self.central_image_charge_c
        if central:
            delta = difference(pt,self.sphere.center_m)
            r = norm(delta)
            for k in range(3):
                contributions[k].append(central*delta[k]/r**3)
        return tuple(self._k*site.charge_c*fsum(parts) for parts in contributions)

    def force_on_charge_n(self, index: int) -> Vector3:
        """Total force on active site, including other REAL charges, excluding its own field."""
        site = self.sites[index]
        conductor = self.conductor_force_on_charge_n(index)
        terms = [[conductor[k]] for k in range(3)]
        for j,other in enumerate(self.sites):
            if j == index:
                continue
            delta = difference(site.position_m,other.position_m)
            r = norm(delta)
            for k in range(3):
                terms[k].append(self._k*site.charge_c*other.charge_c*delta[k]/r**3)
        return tuple(fsum(part) for part in terms)

    @property
    def interaction_energy_j(self) -> float:
        """Variable electrostatic energy excluding point/sphere constant self energies.

        Grounded-image term has factor 1/2. For fixed-Q metal, the quadratic
        central-monopole correction is k*((Q+S)^2-Q^2)/(2*a), S=sum(q*a/d).
        Its position derivative gives the correct total force on each charge.
        """
        k = self._k
        pairs = []
        for i,source in enumerate(self.sites):
            for other in self.sites[i+1:]:
                pairs.append(k*source.charge_c*other.charge_c/
                             norm(difference(source.position_m,other.position_m)))
        images = self.image_sources
        grounded_image = .5*k*fsum(
            source.charge_c*image.charge_c/
            norm(difference(source.position_m,image.position_m))
            for source in self.sites for image in images
        )
        central_energy = 0.0
        if isinstance(self.sphere.constraint, FixedNetCharge):
            Q = self.sphere.constraint.charge_c
            S = -fsum(image.charge_c for image in images)
            # Expanded difference of squares avoids cancelling two close large terms.
            central_energy = k*(Q*S+S*S/2)/self.sphere.radius_m
        return fsum((*pairs,grounded_image,central_energy))


class ExternalMultiSphereBackend:
    """Exact solution for >=2 authored charges external to one grounded/isolated sphere."""
    def supports(self, problem: ElectrostaticProblem) -> bool:
        return external_sphere_supported(problem, min_components=2)

    def solve(self, problem: ElectrostaticProblem) -> ExternalMultiSphereSolution:
        if not self.supports(problem):
            raise UnsupportedProblem('multiple external charges require one grounded or fixed-Q solid sphere, all sites strictly outside, and finite source strengths')
        sphere = problem.conductors[0]
        sites = active_external_sites(problem)
        images_total = fsum(-site.charge_c*sphere.radius_m/
                            norm(difference(site.position_m,sphere.center_m)) for site in sites)
        Q = images_total if isinstance(sphere.constraint,Grounded) else sphere.constraint.charge_c
        central = 0.0 if isinstance(sphere.constraint,Grounded) else Q-images_total
        potential = central/(4*pi*problem.permittivity_f_per_m*sphere.radius_m)
        conductor = ConductorResult(sphere.name,potential,Q,0.0,Q)
        return ExternalMultiSphereSolution(sphere,sites,problem.permittivity_f_per_m,(conductor,))
