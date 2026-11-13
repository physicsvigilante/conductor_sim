"""Exact image-charge electrostatics above an ideal infinite grounded plane.

The positive normal defines the physical vacuum half-space. The grounded
metal occupies the other half-space. This reference is fully 3D: unlike an
axisymmetric FEMM solve, point sources need not share a common axis.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import fsum, hypot, isclose, isfinite, pi

from .model import ConductingPlane, ElectrostaticProblem, Grounded, PointCharge, Vector3, _position
from .solver import ConductorResult, SingularField, UnsupportedProblem


def _minus(a: Vector3, b: Vector3) -> Vector3:
    return tuple(a[i] - b[i] for i in range(3))


def _dot(a: Vector3, b: Vector3) -> float:
    return fsum(a[i] * b[i] for i in range(3))


def _mirror(point: Vector3, plane: ConductingPlane) -> Vector3:
    h = plane.signed_distance_m(point)
    return tuple(point[i] - 2*h*plane.normal[i] for i in range(3))


def _sites(problem: ElectrostaticProblem) -> tuple[PointCharge, ...]:
    grouped: dict[Vector3, list[float]] = {}
    for source in problem.charges:
        grouped.setdefault(source.position_m, []).append(source.charge_c)
    return tuple(PointCharge(q, pos) for pos, qs in sorted(grouped.items())
                 if (q := fsum(qs)) != 0.0)


@dataclass(frozen=True, slots=True)
class GroundedPlaneSolution:
    plane: ConductingPlane
    sources: tuple[PointCharge, ...]
    permittivity_f_per_m: float
    conductor_results: tuple[ConductorResult, ...]

    @property
    def symmetry_axis(self) -> Vector3:
        # This is a natural visualization meridian, not a claim that an
        # arbitrary multi-source configuration has axial symmetry.
        return self.plane.normal

    @property
    def _k(self) -> float:
        return 1 / (4*pi*self.permittivity_f_per_m)

    def potential_at(self, point_m: Vector3) -> float:
        point = _position(point_m, "sample point")
        if self.plane.signed_distance_m(point) <= 0:
            return 0.0
        terms = []
        for q in self.sources:
            distance = hypot(*_minus(point, q.position_m))
            if distance == 0:
                raise SingularField("potential is singular at an ideal point charge")
            image_distance = hypot(*_minus(point, _mirror(q.position_m, self.plane)))
            terms.append(q.charge_c * (1/distance - 1/image_distance))
        return self._k * fsum(terms)

    def electric_field_at(self, point_m: Vector3) -> Vector3:
        point = _position(point_m, "sample point")
        height = self.plane.signed_distance_m(point)
        if isclose(height, 0.0, abs_tol=1e-14):
            raise SingularField("electric field has distinct one-sided limits at plane")
        if height < 0:
            return (0.0, 0.0, 0.0)
        terms = [[], [], []]
        for q in self.sources:
            direct = _minus(point, q.position_m)
            dr = hypot(*direct)
            if dr == 0:
                raise SingularField("electric field is singular at an ideal point charge")
            image = _minus(point, _mirror(q.position_m, self.plane))
            di = hypot(*image)
            for i in range(3):
                terms[i].append(q.charge_c * (direct[i]/dr**3 - image[i]/di**3))
        return tuple(self._k * fsum(component) for component in terms)

    def surface_charge_density_at(self, surface: str, point_m: Vector3) -> float:
        if surface != 'outer':
            raise ValueError("the ideal plane has one exposed conducting surface")
        point = _position(point_m, "surface point")
        if not isclose(self.plane.signed_distance_m(point), 0.0, abs_tol=1e-12):
            raise ValueError("surface-density point must lie on conducting plane")
        return -fsum(q.charge_c * self.plane.signed_distance_m(q.position_m) /
                     (2*pi*hypot(*_minus(point, q.position_m))**3)
                     for q in self.sources)

    def surface_charge_density(self, surface: str, radial_distance_m: float) -> float:
        """One-source radial sigma about the point's projection on the plane.

        General multi-charge profiles use ``surface_charge_density_at`` at
        a fully specified 3D position, without assuming axial symmetry.
        """
        if surface != 'outer':
            raise ValueError("the plane has one exposed conducting surface")
        if len(self.sources) != 1:
            raise UnsupportedProblem("radial sigma needs one source; use density_at")
        if not isfinite(radial_distance_m) or radial_distance_m < 0:
            raise ValueError("radial_distance_m must be finite and nonnegative")
        q = self.sources[0]
        h = self.plane.signed_distance_m(q.position_m)
        # Radial coordinate is measured from the charge's projection onto
        # the infinite plane, independent of the author's arbitrary point_m.
        return -q.charge_c*h/(2*pi*(h*h + radial_distance_m**2)**1.5)

    def force_on_charge_n(self, index: int, *, include_direct: bool = True) -> Vector3:
        """Net force on one *grouped* active point charge, excluding self-field."""
        source = self.sources[index]
        items = [[], [], []]
        for j, other in enumerate(self.sources):
            if include_direct and j != index:
                v = _minus(source.position_m, other.position_m)
                d = hypot(*v)
                for i in range(3):
                    items[i].append(source.charge_c*other.charge_c*v[i]/d**3)
            reflected = _minus(source.position_m, _mirror(other.position_m, self.plane))
            d = hypot(*reflected)
            for i in range(3):
                items[i].append(-source.charge_c*other.charge_c*reflected[i]/d**3)
        return tuple(self._k*fsum(x) for x in items)

    @property
    def induced_force_on_charge_n(self) -> Vector3:
        if len(self.sources) != 1:
            raise UnsupportedProblem("use force_on_charge_n(index) for multiple sources")
        return self.force_on_charge_n(0, include_direct=False)

    @property
    def interaction_energy_j(self) -> float:
        """External real-real Coulomb energy plus half the image interaction."""
        direct = [self.sources[i].charge_c*self.sources[j].charge_c /
                  hypot(*_minus(self.sources[i].position_m, self.sources[j].position_m))
                  for i in range(len(self.sources)) for j in range(i+1, len(self.sources))]
        image = [(-q.charge_c*other.charge_c / hypot(*_minus(
            q.position_m, _mirror(other.position_m, self.plane))))
                 for q in self.sources for other in self.sources]
        return self._k*(fsum(direct) + 0.5*fsum(image))


class GroundedPlaneBackend:
    """Image-charge solver for one infinite plane and one or more above-plane charges."""
    def supports(self, problem: ElectrostaticProblem) -> bool:
        if len(problem.conductors) != 1 or not isinstance(problem.conductors[0], ConductingPlane):
            return False
        plane = problem.conductors[0]
        return (isinstance(plane.constraint, Grounded) and bool(_sites(problem)) and len(_sites(problem)) <= 32 and
                all(plane.signed_distance_m(q.position_m) > 0 for q in problem.charges))

    def solve(self, problem: ElectrostaticProblem) -> GroundedPlaneSolution:
        if not self.supports(problem):
            raise UnsupportedProblem("grounded-plane images require exactly one plane and "
                                     "1–32 nonzero sites entirely in its positive-normal half-space")
        plane = problem.conductors[0]
        sources = _sites(problem)
        total = -fsum(q.charge_c for q in sources)
        results = (ConductorResult(plane.name, 0.0, total, 0.0, total),)
        return GroundedPlaneSolution(plane, sources, problem.permittivity_f_per_m, results)
