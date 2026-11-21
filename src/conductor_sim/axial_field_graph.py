"""Continuous piecewise-linear E_x(x) profiles on bounded, common Cartesian axes."""
from bisect import bisect_right
from dataclasses import dataclass, field
from fractions import Fraction

from .affine_fields import AffineFieldAssembly, _linear_integral, _observed, _scalar
from .solver import UnsupportedProblem


def _exact(value):
    return value if isinstance(value, Fraction) else Fraction(_scalar(value))


@dataclass(frozen=True, slots=True)
class AxialFieldGraph:
    """Knots are (x in m, E_x in V/m). The first knot carries the potential reference."""

    knots: tuple
    reference_potential_v: float = 0.
    _positions: tuple = field(init=False, repr=False, compare=False)
    _areas: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        knots = tuple((_exact(x), _exact(e)) for x, e in self.knots)
        if len(knots) < 2:
            raise ValueError("field graph needs at least two knots")
        areas = [Fraction(0)]
        for (a, ea), (b, eb) in zip(knots, knots[1:]):
            if b <= a:
                raise ValueError("field graph positions must strictly increase")
            areas.append(areas[-1] + (b-a)*(ea+eb)/2)
        object.__setattr__(self, "knots", knots)
        object.__setattr__(self, "reference_potential_v", _exact(self.reference_potential_v))
        object.__setattr__(self, "_positions", tuple(x for x, _ in knots))
        object.__setattr__(self, "_areas", tuple(areas))

    @classmethod
    def from_affine(cls, assembly, start_m, end_m):
        """Restrict an existing affine assembly to (x, 0, 0) on the stated interval."""
        if type(assembly) is not AffineFieldAssembly:
            raise TypeError("expected an AffineFieldAssembly")
        a, b = _exact(start_m), _exact(end_m)
        origin = (a, Fraction(0), Fraction(0))
        end = (b, Fraction(0), Fraction(0))
        reference = assembly._potential(origin)
        return cls(((a, assembly._field(origin)[0]), (b, assembly._field(end)[0])), reference)

    def _segment(self, position_m):
        x = _exact(position_m)
        if not self._positions[0] <= x <= self._positions[-1]:
            raise UnsupportedProblem("position outside the authored field graph")
        i = min(bisect_right(self._positions, x)-1, len(self.knots)-2)
        a, b = self._positions[i:i+2]
        return i, (x-a)/(b-a)

    def _field(self, x):
        i, s = self._segment(x)
        ea, eb = self.knots[i][1], self.knots[i+1][1]
        return ea + s*(eb-ea)

    def _area(self, x):
        i, s = self._segment(x)
        (a, ea), (b, eb) = self.knots[i:i+2]
        return self._areas[i] + (b-a)*_linear_integral(ea, eb, s)

    def axial_field_v_m(self, position_m):
        return _observed(self._field(position_m))

    def potential_v(self, position_m):
        return _observed(self.reference_potential_v - self._area(position_m))

    def field_work_j(self, charge_c, start_m, end_m):
        """Signed work by the field on a fixed charge between two graph positions."""
        return _observed(_exact(charge_c) * (self._area(end_m) - self._area(start_m)))

    def superpose(self, other):
        """Sum fields and independently referenced potentials on their common interval."""
        if type(other) is not AxialFieldGraph:
            raise TypeError("expected another AxialFieldGraph on the same x axis")
        lo = max(self._positions[0], other._positions[0])
        hi = min(self._positions[-1], other._positions[-1])
        if lo >= hi:
            raise UnsupportedProblem("graphs need an overlapping interval of positive length")
        grid = sorted({lo, hi} | {x for graph in (self, other) for x in graph._positions if lo < x < hi})
        reference = sum(graph.reference_potential_v - graph._area(lo) for graph in (self, other))
        return AxialFieldGraph(tuple((x, self._field(x) + other._field(x)) for x in grid), reference)
