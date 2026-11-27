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


def _mirror_displacement(target: Vector3, source: Vector3, plane: ConductingPlane) -> Vector3:
    """Translation-invariant displacement ``target - mirror(source, plane)``.

    Algebraically ``(target - source) + 2*h(source)*n``, avoiding world-coordinate
    cancellation when ``|source| >> h(source)`` (including ``target == source``).
    """
    h = plane.signed_distance_m(source)
    return tuple((target[i] - source[i]) + 2.0 * h * plane.normal[i] for i in range(3))


def _sites(problem: ElectrostaticProblem) -> tuple[PointCharge, ...]:
    grouped: dict[Vector3, list[float]] = {}
    for source in problem.charges:
        grouped.setdefault(source.position_m, []).append(source.charge_c)
    return tuple(PointCharge(q, pos) for pos, qs in sorted(grouped.items())
                 if (q := fsum(qs)) != 0.0)


def _checked_site_index(index: int, count: int) -> int:
    """Grouped-site index policy of ``ForceEnergyResult.force_on_charge_n``.

    Negative, boolean and non-integer indices are rejected, so a legacy
    per-site force can never select a site and then fail to exclude it.
    """
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError('source index must be an integer')
    if not 0 <= index < count:
        raise IndexError('source index is outside the grouped active sites')
    return index


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
        height = self.plane.signed_distance_m(point)
        if height <= 0:
            return 0.0
        terms = []
        for q in self.sources:
            distance = hypot(*_minus(point, q.position_m))
            if distance == 0:
                raise SingularField("potential is singular at an ideal point charge")
            image_distance = hypot(*_mirror_displacement(point, q.position_m, self.plane))
            # Rationalize the difference: ri^2-r^2 = 4*h*h_source.
            # Direct reciprocal subtraction loses weak far/near-plane values.
            source_height = self.plane.signed_distance_m(q.position_m)
            relative_separation = (2*(height/image_distance)*source_height /
                                   (.5*distance+.5*image_distance))
            terms.append(q.charge_c * relative_separation / distance)
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
            image = _mirror_displacement(point, q.position_m, self.plane)
            di = hypot(*image)
            source_height = self.plane.signed_distance_m(q.position_m)
            # (di-dr)/di, obtained without subtracting nearly equal distances.
            separation = 2*(height/di)*source_height/(.5*dr+.5*di)
            ratio = dr/di
            inverse_cube_difference = separation*(1+ratio+ratio*ratio)
            for i in range(3):
                terms[i].append(q.charge_c * (
                    (direct[i]/dr)*inverse_cube_difference/dr/dr
                    - (2*source_height/di)*self.plane.normal[i]/di/di))
        return tuple(self._k * fsum(component) for component in terms)

    def surface_charge_density_at(self, surface: str, point_m: Vector3) -> float:
        if surface != 'outer':
            raise ValueError("the ideal plane has one exposed conducting surface")
        point = _position(point_m, "surface point")
        if not isclose(self.plane.signed_distance_m(point), 0.0, abs_tol=1e-12):
            raise ValueError("surface-density point must lie on conducting plane")
        from fractions import Fraction
        from sys import float_info
        from .model import _plane_exact_numerator, _plane_raw_normal
        from .observables import _scaled_ratio_product
        min_normal = float_info.min * 1e4
        terms = []
        for q in self.sources:
            delta = _minus(point, q.position_m)
            r = hypot(*delta) if all(isfinite(d) for d in delta) else float("inf")
            hq = self.plane.signed_distance_m(q.position_m)
            if isfinite(r) and r >= min_normal and isfinite(hq) and abs(hq) >= min_normal:
                terms.append(_scaled_ratio_product((q.charge_c, hq), (2.0 * pi, r, r, r)))
                continue
            delta_f = tuple(Fraction(point[i]) - Fraction(q.position_m[i]) for i in range(3))
            d_max = max(abs(x) for x in delta_f)
            if d_max == 0:
                raise ZeroDivisionError("surface-density point coincides with point source")
            r_scaled = hypot(*(float(x / d_max) for x in delta_f))
            if isfinite(hq) and abs(hq) >= min_normal:
                exact_hq = Fraction(hq)
            else:
                rn = _plane_raw_normal(self.plane.normal)
                r_max = max(abs(Fraction(x)) for x in rn)
                r_norm = hypot(*(float(Fraction(x) / r_max) for x in rn))
                exact_hq = _plane_exact_numerator(q.position_m, self.plane.point_m, self.plane.normal, rn) / (r_max * Fraction(r_norm))
            exact_term = (
                Fraction(q.charge_c) * exact_hq
                / (Fraction(2.0 * pi) * (d_max ** 3) * Fraction(r_scaled) ** 3)
            )
            try:
                terms.append(float(exact_term))
            except OverflowError:
                terms.append(float("inf") if exact_term > 0 else float("-inf"))
        return -fsum(terms)

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
        from fractions import Fraction
        from sys import float_info
        from .model import _plane_exact_numerator, _plane_raw_normal
        from .observables import _scaled_ratio_product
        min_normal = float_info.min * 1e4
        r_tot = hypot(h, radial_distance_m) if isfinite(h) else float("inf")
        if isfinite(h) and abs(h) >= min_normal and isfinite(r_tot) and r_tot >= min_normal:
            return -_scaled_ratio_product((q.charge_c, h), (2.0 * pi, r_tot, r_tot, r_tot))
        if isfinite(h) and abs(h) >= min_normal:
            exact_hq = Fraction(h)
        else:
            rn = _plane_raw_normal(self.plane.normal)
            r_max = max(abs(Fraction(x)) for x in rn)
            r_norm = hypot(*(float(Fraction(x) / r_max) for x in rn))
            exact_hq = _plane_exact_numerator(q.position_m, self.plane.point_m, self.plane.normal, rn) / (r_max * Fraction(r_norm))
        exact_r = Fraction(radial_distance_m)
        d_max = max(abs(exact_hq), exact_r)
        if d_max == 0:
            raise ZeroDivisionError("surface-density point coincides with point source")
        r_scaled = hypot(float(exact_hq / d_max), float(exact_r / d_max))
        exact_term = (
            Fraction(q.charge_c) * exact_hq
            / (Fraction(2.0 * pi) * (d_max ** 3) * Fraction(r_scaled) ** 3)
        )
        try:
            return -float(exact_term)
        except OverflowError:
            return float("-inf") if exact_term > 0 else float("inf")

    def force_on_charge_n(self, index: int, *, include_direct: bool = True) -> Vector3:
        """Net force on one *grouped* active point charge, excluding self-field."""
        source = self.sources[_checked_site_index(index, len(self.sources))]
        items = [[], [], []]
        for j, other in enumerate(self.sources):
            if include_direct and j != index:
                v = _minus(source.position_m, other.position_m)
                d = hypot(*v)
                for i in range(3):
                    items[i].append(source.charge_c*other.charge_c*v[i]/d**3)
            reflected = _mirror_displacement(source.position_m, other.position_m, self.plane)
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
        image = [(-q.charge_c*other.charge_c / hypot(*_mirror_displacement(
            q.position_m, other.position_m, self.plane)))
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
