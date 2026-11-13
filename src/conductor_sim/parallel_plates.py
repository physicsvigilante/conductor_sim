"""Analytical reference for two infinite parallel conducting half-spaces.

Both surface normals must face *into the same open slab*, so the authored
vacuum is the intersection of their positive-normal half-spaces. A linear
Dirichlet background supports arbitrary prescribed plate potentials. Point
sources use a finite *reciprocal, plate-symmetric* image-charge series, not a claim of exact
numerical convergence at an arbitrary sampling location.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import fsum, hypot, isfinite, pi

from .grounded_plane import _sites, _dot, _minus
from .model import (
    ConductingPlane, ElectrostaticProblem, FixedPotential, Grounded,
    PointCharge, Vector3, _position,
)
from .solver import ConductorResult, SingularField, UnsupportedProblem


def _voltage(plate: ConductingPlane) -> float:
    return 0.0 if isinstance(plate.constraint, Grounded) else plate.constraint.potential_v


def _layout(problem: ElectrostaticProblem):
    if (len(problem.conductors) != 2 or
            any(not isinstance(c, ConductingPlane) for c in problem.conductors)):
        raise UnsupportedProblem("parallel plates require exactly two ConductingPlane objects")
    a, b = problem.conductors
    # Opposing normals are necessary: two equal-facing surfaces do not bound
    # an open slab, and a tilted pair is not an infinite parallel capacitor.
    if hypot(*(a.normal[i]+b.normal[i] for i in range(3))) > 1e-12:
        raise UnsupportedProblem("plate normals must be parallel, opposite and face the gap")
    gap = a.signed_distance_m(b.point_m)
    if not isfinite(gap) or gap <= 0 or b.signed_distance_m(a.point_m) <= 0:
        raise UnsupportedProblem("plate normals must face an open positive-width gap")
    # Reject even zero or exactly cancelling charges authored inside metal.
    for charge in problem.charges:
        if not (a.signed_distance_m(charge.position_m) > 0 and
                b.signed_distance_m(charge.position_m) > 0):
            raise UnsupportedProblem("all point charges must lie strictly between the plates")
    sources = _sites(problem)
    if len(sources) > 32:
        raise UnsupportedProblem("parallel-plate image reference supports at most 32 active sites")
    return a, b, gap, sources


@dataclass(frozen=True, slots=True)
class ParallelPlateSolution:
    """Finite-order 3D image-reference solution; no FEMM data is substituted."""

    lower: ConductingPlane
    upper: ConductingPlane
    gap_m: float
    sources: tuple[PointCharge, ...]
    permittivity_f_per_m: float
    image_order: int
    conductor_results: tuple[ConductorResult, ...]

    @property
    def symmetry_axis(self) -> Vector3:
        """A natural meridian direction, not axial symmetry for general sources."""
        return self.lower.normal

    @property
    def _k(self) -> float:
        return 1.0 / (4.0*pi*self.permittivity_f_per_m)

    def _height(self, point: Vector3) -> float:
        return self.lower.signed_distance_m(point)

    def _background_v(self, height: float) -> float:
        v0 = _voltage(self.lower)
        return v0 + (_voltage(self.upper)-v0)*(height/self.gap_m)

    def _transverse(self, point: Vector3, source: PointCharge):
        delta = _minus(point, source.position_m)
        axial = _dot(delta, self.lower.normal)
        tangent = tuple(delta[i]-axial*self.lower.normal[i] for i in range(3))
        return tangent, hypot(*tangent)

    def _image_terms(self, height: float, source_height: float):
        """Reciprocal finite kernel, averaged over the two plate orientations.

        Positive images use n=-N..N. Negative images use the average of
        -N..N and 1-N..1+N. Averaging preserves exchange reciprocity AND
        plate-swap symmetry at finite order, removing the old midpoint bias.
        It does not make finite-order Dirichlet fields exact everywhere.
        """
        d = self.gap_m
        for n in range(-self.image_order, self.image_order+1):
            yield n, (height-source_height-2*n*d), (height+source_height-2*n*d), (height+source_height-2*(n+1)*d)

    def _source_potential(self, point: Vector3, source: PointCharge, *, exclude_direct=False):
        _, radial = self._transverse(point, source)
        terms = []
        for n, z1, z2, z3 in self._image_terms(self._height(point), self._height(source.position_m)):
            r2, r3 = hypot(radial, z2), hypot(radial, z3)
            if exclude_direct and n == 0:
                terms.append(-.5/r2-.5/r3)
                continue
            r1 = hypot(radial, z1)
            if r1 == 0:
                raise SingularField("potential is singular at an ideal point charge")
            # Rationalized reciprocal differences reduce distant-image cancellation.
            terms.append(.5*((z2-z1)/r2)*((z2+z1)/(r1+r2))/r1
                         + .5*((z3-z1)/r3)*((z3+z1)/(r1+r3))/r1)
        return self._k*source.charge_c*fsum(terms)

    def _source_field(self, point: Vector3, source: PointCharge, *, exclude_direct=False):
        tangent, radial = self._transverse(point, source)
        axial_terms, radial_terms = [], []
        for n, z1, z2, z3 in self._image_terms(self._height(point), self._height(source.position_m)):
            if exclude_direct and n == 0:
                inv1 = 0.
            else:
                r1 = hypot(radial, z1)
                if r1 == 0:
                    raise SingularField("electric field is singular at an ideal point charge")
                inv1 = 1/r1**3
            inv2, inv3 = .5/hypot(radial, z2)**3, .5/hypot(radial, z3)**3
            axial_terms.append(z1*inv1-z2*inv2-z3*inv3)
            radial_terms.append(inv1-inv2-inv3)
        axial, transverse = fsum(axial_terms), fsum(radial_terms)
        return tuple(self._k*source.charge_c*(axial*self.lower.normal[i]+transverse*tangent[i])
                     for i in range(3))

    def potential_at(self, point_m: Vector3) -> float:
        point = _position(point_m, "sample point")
        h = self._height(point)
        if h <= 0:
            return _voltage(self.lower)
        if h >= self.gap_m:
            return _voltage(self.upper)
        return fsum((self._background_v(h), *(self._source_potential(point, q) for q in self.sources)))

    def _field_inside(self, point: Vector3) -> Vector3:
        """Vacuum-side E, also evaluable on the plates for one-sided sigma."""
        background = -(_voltage(self.upper)-_voltage(self.lower))/self.gap_m
        fields = [self._source_field(point, q) for q in self.sources]
        return tuple(fsum((background*self.lower.normal[i], *(f[i] for f in fields)))
                     for i in range(3))

    def force_energy(self):
        """Mechanical force/energy with only the direct point self-field omitted."""
        from .observables import _plate_observables
        return _plate_observables(self)

    def force_on_charge_n(self, index: int, *, include_direct: bool = True) -> Vector3:
        """Force on a grouped site; conductor contribution includes the bias field."""
        result = self.force_energy()
        total = result.force_on_charge_n(index)  # shared strict index validation
        return total if include_direct else result.charges[index].conductor_force_n

    @property
    def interaction_energy_j(self) -> float:
        """Mechanical potential, NOT total field energy of an infinite capacitor."""
        return self.force_energy().energy.total_j

    def electric_field_at(self, point_m: Vector3) -> Vector3:
        point = _position(point_m, "sample point")
        h = self._height(point)
        if abs(h) <= 1e-14 or abs(h-self.gap_m) <= 1e-14:
            raise SingularField("electric field has distinct one-sided limits at a plate")
        if h < 0 or h > self.gap_m:
            return (0.0,0.0,0.0)
        return self._field_inside(point)

    def surface_charge_density_at(self, surface: str, point_m: Vector3,
                                  *, conductor_name: str | None = None) -> float:
        """Local sigma on the vacuum-facing side of the named infinite plate."""
        if surface != "outer":
            raise ValueError("each ideal plate exposes only one vacuum-facing surface")
        if conductor_name not in (self.lower.name,self.upper.name):
            raise ValueError("conductor_name must name one of the two plates")
        plate = self.lower if conductor_name == self.lower.name else self.upper
        point = _position(point_m, "surface point")
        if abs(plate.signed_distance_m(point)) > 1e-11:
            raise ValueError("surface-density point must lie on the requested plate")
        e = self._field_inside(point)
        return self.permittivity_f_per_m*_dot(e,plate.normal)


class ParallelPlateBackend:
    """Two facing infinite plates; analytical field uses truncated images.

    image_order=N averages the two plate-oriented truncations at n=-N..N.
    The reciprocal kernel preserves plate-swap symmetry at every order. Increasing N improves convergence but costs O(sources*N)
    work per field sample. The source-free uniform field is exact.
    """

    def __init__(self, *, image_order: int = 48):
        if isinstance(image_order,bool) or not isinstance(image_order,int) or not 1 <= image_order <= 512:
            raise ValueError("image_order must be an integer in [1,512]")
        self.image_order = image_order

    def supports(self, problem: ElectrostaticProblem) -> bool:
        try:
            _layout(problem)
        except UnsupportedProblem:
            return False
        return True

    def solve(self, problem: ElectrostaticProblem) -> ParallelPlateSolution:
        a,b,d,sources = _layout(problem)
        v0,v1 = _voltage(a),_voltage(b)
        if v0 == v1:
            # Finite source-induced *total* charge on either infinite face,
            # via the exact weighting potentials z/d and 1-z/d.
            qa = -fsum(q.charge_c*(1-a.signed_distance_m(q.position_m)/d)
                       for q in sources)
            qb = -fsum(q.charge_c*(a.signed_distance_m(q.position_m)/d)
                       for q in sources)
            results = (
                ConductorResult(a.name,v0,qa,0.0,qa),
                ConductorResult(b.name,v1,qb,0.0,qb),
            )
        else:
            # Different Dirichlet values create an infinite uniform capacitor
            # surface density; its integral is unbounded on infinite plates.
            results = (
                ConductorResult(a.name,v0,None,0.0,None),
                ConductorResult(b.name,v1,None,0.0,None),
            )
        return ParallelPlateSolution(a,b,d,sources,problem.permittivity_f_per_m,
                                     self.image_order,results)
