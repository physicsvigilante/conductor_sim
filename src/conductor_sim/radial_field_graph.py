"""Prescribed spherically symmetric field from a continuous radial field graph."""
from bisect import bisect_right
from dataclasses import dataclass, field
from fractions import Fraction

from ._coulomb_sum import reciprocal_distance_sum
from .affine_fields import _point, _triple
from .axial_field_graph import AxialFieldGraph
from .model import Vector3
from .solver import UnsupportedProblem


def _constant_plus_inverse_radius(constant, weight, squared):
    terms = {Fraction(1): constant}
    if weight:
        terms[squared] = terms.get(squared, 0) + weight
    return reciprocal_distance_sum(terms)


@dataclass(frozen=True, slots=True)
class RadialFieldGraph:
    """Interpret graph position as radius and its field as the signed outward component."""

    profile: AxialFieldGraph
    center_m: Vector3 = (0., 0., 0.)
    _squared_knots: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if type(self.profile) is not AxialFieldGraph:
            raise TypeError("expected a piecewise-linear field graph")
        if self.profile.knots[0][0] < 0:
            raise UnsupportedProblem("radial graph requires nonnegative radii")
        object.__setattr__(self, "center_m", _triple(self.center_m))
        object.__setattr__(self, "_squared_knots", tuple(r*r for r, _ in self.profile.knots))

    def superpose(self, other):
        """Combine two radial fields over the common interval about their shared centre."""
        if type(other) is not RadialFieldGraph:
            raise TypeError("expected another RadialFieldGraph")
        if self.center_m != other.center_m:
            raise UnsupportedProblem("radial superposition requires a common centre")
        return RadialFieldGraph(self.profile.superpose(other.profile), self.center_m)

    def _segment(self, point_m):
        delta = tuple(x-c for x,c in zip(_point(point_m), _point(self.center_m)))
        squared = sum(x*x for x in delta)
        if not self._squared_knots[0] <= squared <= self._squared_knots[-1]:
            raise UnsupportedProblem("probe outside the authored radial interval")
        i = min(bisect_right(self._squared_knots, squared)-1, len(self._squared_knots)-2)
        (a, ea), (b, eb) = self.profile.knots[i:i+2]
        slope = (eb-ea)/(b-a)
        return delta, squared, a, slope, ea-slope*a

    def potential_at(self, point_m):
        _, squared, a, slope, intercept = self._segment(point_m)
        va = self.profile.reference_potential_v - self.profile._area(a)
        # Integrate E_r = slope*r + intercept from a to r, preserving irrational radii.
        constant = va - slope*(squared-a*a)/2 + intercept*a
        return _constant_plus_inverse_radius(constant, -intercept*squared, squared)

    def electric_field_at(self, point_m):
        delta, squared, _, slope, intercept = self._segment(point_m)
        if not squared:
            if intercept:
                raise UnsupportedProblem("nonzero limiting radial field has no direction at the centre")
            return (0., 0., 0.)
        return tuple(_constant_plus_inverse_radius(slope*x, intercept*x, squared) for x in delta)
