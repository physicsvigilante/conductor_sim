"""Exact exterior point charge next to a fixed-net-charge solid conducting sphere.

The grounded Kelvin image pair is supplemented by a central image monopole.
The latter enforces the *actual prescribed conductor net charge* (not a
prescribed voltage). This independent reference never calls FEMM.
"""
from __future__ import annotations

from math import isfinite, pi

from .external_sphere import (
    ExternalGroundedSphereBackend, ExternalGroundedSphereSolution, _diff, _norm,
)
from .model import ElectrostaticProblem, FixedNetCharge, SolidSphere, Vector3, _position
from .solver import ConductorResult, UnsupportedProblem


class ExternalIsolatedSphereSolution(ExternalGroundedSphereSolution):
    """Exact Kelvin-image field for a fixed-net-charge (isolated) solid sphere.

    For radius a, source q at distance d>a, and prescribed sphere charge Q:
      q_image = -q*a/d at r_image = a**2/d,
      q_center = Q-q_image = Q+q*a/d at the sphere centre.

    The grounded Kelvin pair vanishes on r=a, while the central monopole
    sets the equipotential to k*q_center/a. Their image charges sum to Q.
    The centre monopole represents both the prescribed net charge and the
    charge redistributed to maintain a constant potential; it is NOT an
    additional authored physical point charge.
    """

    @property
    def central_image_charge_c(self) -> float:
        return self.conductor_results[0].net_charge_c - self.image_charge_c

    @property
    def conductor_potential_v(self) -> float:
        return self._k*self.central_image_charge_c/self.sphere.radius_m

    def potential_at(self, point_m: Vector3) -> float:
        point = _position(point_m, 'sample point')
        r = _norm(_diff(point, self.sphere.center_m))
        base = super().potential_at(point)
        return base + self._k*self.central_image_charge_c / (
            self.sphere.radius_m if r <= self.sphere.radius_m else r
        )

    def electric_field_at(self, point_m: Vector3) -> Vector3:
        point = _position(point_m, 'sample point')
        base = super().electric_field_at(point)
        delta = _diff(point, self.sphere.center_m)
        r = _norm(delta)
        if r < self.sphere.radius_m:
            return base
        multiplier = self._k*self.central_image_charge_c/(r**3)
        return tuple(base[i] + multiplier*delta[i] for i in range(3))

    def surface_charge_density(self, surface: str, theta_rad: float) -> float:
        # super() validates face and angle before adding the constant term.
        grounded_density = super().surface_charge_density(surface, theta_rad)
        return grounded_density + self.central_image_charge_c/(4*pi*self.sphere.radius_m**2)

    @property
    def induced_force_on_charge_n(self) -> Vector3:
        """Total sphere-on-point force, including the original net charge Q.

        The M7 property name is kept for API compatibility. With Q!=0 this
        is not merely an induced/polarization force: it also includes the
        Coulomb contribution from the sphere's prescribed total charge.
        """
        grounded_force = super().induced_force_on_charge_n
        delta = _diff(self.source.position_m, self.sphere.center_m)
        d = self.source_distance_m
        multiplier = self._k*self.source.charge_c*self.central_image_charge_c/d**3
        return tuple(grounded_force[i]+multiplier*delta[i] for i in range(3))

    @property
    def interaction_energy_j(self) -> float:
        """Source--sphere interaction energy relative to infinite separation.

        The constant self-energy Q^2/(2*C_sphere) of the isolated sphere
        is excluded: it does not depend on source position and has no
        effect on the force. Polarization terms include the physical 1/2.
        """
        q = self.source.charge_c
        Q = self.conductor_results[0].net_charge_c
        a, d = self.sphere.radius_m, self.source_distance_m
        return (super().interaction_energy_j
                + self._k*q*Q/d
                + self._k*q*q*a/(2*d*d))


class ExternalIsolatedSphereBackend:
    """Analytical backend for one nonzero external charge, one isolated sphere."""

    def supports(self, problem: ElectrostaticProblem) -> bool:
        if len(problem.conductors) != 1 or len(problem.charges) != 1:
            return False
        sphere, source = problem.conductors[0], problem.charges[0]
        if not (isinstance(sphere, SolidSphere)
                and isinstance(sphere.constraint, FixedNetCharge)
                and source.charge_c != 0):
            return False
        distance = _norm(_diff(source.position_m, sphere.center_m))
        return distance > sphere.radius_m and isfinite(
            sphere.constraint.charge_c+source.charge_c
        )

    def solve(self, problem: ElectrostaticProblem) -> ExternalIsolatedSphereSolution:
        if not self.supports(problem):
            raise UnsupportedProblem(
                'isolated exterior-sphere solution requires exactly one nonzero '
                'point charge strictly outside one FixedNetCharge SolidSphere; '
                'grounded, touching, interior, multiple-source and general '
                'conductor geometries are not supported by this backend'
            )
        sphere, source = problem.conductors[0], problem.charges[0]
        Q = sphere.constraint.charge_c
        q_image = -source.charge_c*sphere.radius_m/_norm(
            _diff(source.position_m, sphere.center_m)
        )
        potential = (Q-q_image)/(4*pi*problem.permittivity_f_per_m*sphere.radius_m)
        result = ConductorResult(sphere.name, potential, Q, 0.0, Q)
        return ExternalIsolatedSphereSolution(
            sphere, source, problem.permittivity_f_per_m, (result,)
        )


class ExternalSphereBackend:
    """Common author-facing analytical dispatch for grounded OR isolated spheres.

    Delegates only to exact independently derived reference backends. Never
    substitutes an analytical answer for a failed numerical FEMM run.
    """

    def supports(self, problem: ElectrostaticProblem) -> bool:
        return (ExternalGroundedSphereBackend().supports(problem)
                or ExternalIsolatedSphereBackend().supports(problem))

    def solve(self, problem: ElectrostaticProblem):
        if ExternalGroundedSphereBackend().supports(problem):
            return ExternalGroundedSphereBackend().solve(problem)
        if ExternalIsolatedSphereBackend().supports(problem):
            return ExternalIsolatedSphereBackend().solve(problem)
        raise UnsupportedProblem('external sphere: expected one grounded or '
                                 'fixed-net-charge SolidSphere with one external charge')
