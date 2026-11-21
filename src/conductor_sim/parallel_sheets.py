"""Prescribed uniform infinite sheets normal to x in a homogeneous dielectric."""
from dataclasses import dataclass
from fractions import Fraction

from .affine_fields import _observed, _point
from .axial_field_graph import AxialFieldGraph, _exact
from .model import EPSILON_0
from .solver import UnsupportedProblem


@dataclass(frozen=True, slots=True)
class ParallelSheetField:
    """Sheets are (x in m, signed density in C/m²). Zero bias denotes isolated sheets."""

    sheets: tuple
    permittivity_f_m: float = EPSILON_0
    reference_x_m: float = 0.
    reference_potential_v: float = 0.
    bias_v_m: float = 0.

    def __post_init__(self):
        for name in ("permittivity_f_m", "reference_x_m", "reference_potential_v", "bias_v_m"):
            object.__setattr__(self, name, _exact(getattr(self, name)))
        if self.permittivity_f_m <= 0:
            raise ValueError("permittivity must be positive")
        density = {}
        for position, sigma in self.sheets:
            x = _exact(position)
            density[x] = density.get(x, 0) + _exact(sigma)
        object.__setattr__(self, "sheets", tuple(sorted((x, s) for x, s in density.items() if s)))

    @classmethod
    def from_potential_segments(cls, knots, *, permittivity_f_m=EPSILON_0):
        """Infer sheets at slope breaks of a continuous piecewise-linear V(x).

        The two end segments continue linearly beyond the supplied interval in
        the returned model. Unseen external sources are not identifiable.
        """
        points = tuple((_exact(x), _exact(v)) for x, v in knots)
        if len(points) < 2:
            raise ValueError("potential profile needs at least two knots")
        epsilon = _exact(permittivity_f_m)
        fields = []
        for (a, va), (b, vb) in zip(points, points[1:]):
            if b <= a:
                raise ValueError("potential knot positions must strictly increase")
            fields.append((va-vb)/(b-a))
        sheets = tuple((x, epsilon*(right-left))
                       for (x, _), left, right in zip(points[1:-1], fields, fields[1:]))
        x0, potential = points[0]
        return cls(sheets, permittivity_f_m=epsilon, reference_x_m=x0,
                   reference_potential_v=potential, bias_v_m=(fields[0]+fields[-1])/2)

    def _field(self, x):
        signed = Fraction(0)
        for position, sigma in self.sheets:
            if x == position:
                raise UnsupportedProblem("electric field jumps at a charged sheet")
            signed += sigma if x > position else -sigma
        return self.bias_v_m + signed/(2*self.permittivity_f_m)

    def _integral(self, start, end):
        area = sum(sigma*(abs(end-x)-abs(start-x)) for x, sigma in self.sheets)
        return self.bias_v_m*(end-start) + area/(2*self.permittivity_f_m)

    def electric_field_at(self, point_m):
        return (_observed(self._field(_point(point_m)[0])), 0., 0.)

    def potential_at(self, point_m):
        x = _point(point_m)[0]
        return _observed(self.reference_potential_v - self._integral(self.reference_x_m, x))

    def field_work_j(self, charge_c, start_m, end_m):
        """Signed work by the prescribed field, including paths across charged planes."""
        return _observed(_exact(charge_c)*self._integral(_point(start_m)[0], _point(end_m)[0]))

    def gap_graph(self, start_m, end_m):
        """Expose a continuous axial profile on a closed interval within one region."""
        a, b = _exact(start_m), _exact(end_m)
        if a >= b:
            raise ValueError("gap interval requires increasing endpoints")
        if any(a <= x <= b for x, _ in self.sheets):
            raise UnsupportedProblem("continuous graph interval touches or crosses a charged sheet")
        value = self._field(a)
        reference = self.reference_potential_v - self._integral(self.reference_x_m, a)
        return AxialFieldGraph(((a, value), (b, value)), reference)
