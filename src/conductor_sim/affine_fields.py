"""Static potential, electric field and field work for conservative affine fields (SI)."""
from dataclasses import dataclass
from fractions import Fraction

from .model import Vector3
from .prescribed_fields import ZERO, UniformElectricField
from .solver import UnsupportedProblem


def _scalar(value):
    if type(value) not in (int, float):
        raise ValueError("expected a finite real SI value")
    try:
        Fraction(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError("expected a finite real SI value") from exc
    return value


def _triple(value):
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError("expected three SI components")
    return tuple(_scalar(x) for x in value)


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _linear_integral(first, last, fraction):
    """Integral of a linear scalar over [0, fraction] on a unit interval."""
    return fraction * (first + fraction * (last - first) / 2)


def _observed(value):
    try:
        rounded = float(value)
    except OverflowError as exc:
        raise ValueError("observation exceeds binary64 range") from exc
    if value and rounded == 0:
        raise ValueError("nonzero observation underflows binary64")
    return rounded


@dataclass(frozen=True, slots=True)
class ConservativeAffineField:
    """E(r) = field_v_m + gradient_v_m2 @ (r - reference_point_m)."""

    gradient_v_m2: tuple[Vector3, Vector3, Vector3]
    field_v_m: Vector3 = ZERO
    reference_point_m: Vector3 = ZERO
    reference_potential_v: float = 0.

    def __post_init__(self):
        rows = self.gradient_v_m2
        if not isinstance(rows, (list, tuple)) or len(rows) != 3:
            raise ValueError("expected a three-by-three gradient")
        rows = tuple(_triple(row) for row in rows)
        if any(rows[i][j] != rows[j][i] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise UnsupportedProblem("electrostatic affine field requires a symmetric gradient")
        object.__setattr__(self, "gradient_v_m2", rows)
        for name in ("field_v_m", "reference_point_m"):
            object.__setattr__(self, name, _triple(getattr(self, name)))
        _scalar(self.reference_potential_v)

    def _field(self, point):
        offset = tuple(x - Fraction(r) for x, r in zip(point, self.reference_point_m))
        return tuple(Fraction(e) + _dot(tuple(map(Fraction, row)), offset)
                     for e, row in zip(self.field_v_m, self.gradient_v_m2))

    def _potential(self, point):
        offset = tuple(x - Fraction(r) for x, r in zip(point, self.reference_point_m))
        twice_mean = tuple(Fraction(e) + f for e, f in zip(self.field_v_m, self._field(point)))
        return Fraction(self.reference_potential_v) - _dot(twice_mean, offset) / 2

    def electric_field_at(self, point_m):
        return tuple(map(_observed, self._field(_point(point_m))))

    def potential_at(self, point_m):
        return _observed(self._potential(_point(point_m)))


def _point(point):
    return tuple(map(Fraction, _triple(point)))


@dataclass(frozen=True, slots=True)
class AffineFieldAssembly:
    """Sum independently referenced affine and uniform potentials and fields."""

    sources: tuple[ConservativeAffineField | UniformElectricField, ...]

    def __post_init__(self):
        sources = []
        for source in self.sources:
            if type(source) is UniformElectricField:
                source = ConservativeAffineField((ZERO, ZERO, ZERO), source.field_v_m,
                                                 source.reference_point_m, source.reference_potential_v)
            if type(source) is not ConservativeAffineField:
                raise UnsupportedProblem("affine assembly requires prescribed affine or uniform fields")
            sources.append(source)
        if not sources:
            raise ValueError("affine assembly requires at least one source")
        object.__setattr__(self, "sources", tuple(sources))

    def _field(self, point):
        return tuple(map(sum, zip(*(s._field(point) for s in self.sources))))

    def electric_field_at(self, point_m):
        return tuple(map(_observed, self._field(_point(point_m))))

    def _potential(self, point):
        return sum(s._potential(point) for s in self.sources)

    def potential_at(self, point_m):
        return _observed(self._potential(_point(point_m)))

    def field_work_j(self, charge_c, start_m, end_m):
        """Signed work by the field, using the exact straight-segment mean field."""
        start, end = _point(start_m), _point(end_m)
        delta = tuple(b - a for a, b in zip(start, end))
        midpoint = tuple((a + b) / 2 for a, b in zip(start, end))
        return _observed(Fraction(_scalar(charge_c)) * _dot(self._field(midpoint), delta))
