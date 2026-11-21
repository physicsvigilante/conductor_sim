"""Quasistatic rearrangement work for isolated point charges in a uniform medium."""
from dataclasses import dataclass, replace
from fractions import Fraction
from math import isqrt, pi

from .model import ElectrostaticProblem, Vector3
from .solver import UnsupportedProblem


@dataclass(frozen=True, slots=True)
class PointChargeRearrangement:
    initial: ElectrostaticProblem
    final: ElectrostaticProblem
    external_work_j: float

    @property
    def field_work_j(self) -> float:
        return -self.external_work_j


def _rational_sqrt(value):
    numerator = isqrt(value.numerator)
    denominator = isqrt(value.denominator)
    if numerator**2 == value.numerator and denominator**2 == value.denominator:
        return Fraction(numerator, denominator)
    return None


def _inverse_distance_bounds(squared_distance, precision):
    """Rational bounds for 1/sqrt(squared_distance) with relative binary precision."""
    exact = _rational_sqrt(squared_distance)
    if exact is not None:
        return 1 / exact, 1 / exact
    a, b = squared_distance.denominator, squared_distance.numerator
    shift = precision + (b.bit_length() - a.bit_length()) // 2
    unit = Fraction(2) ** -shift
    scaled_square = Fraction(a, b) / unit**2
    floor_root = isqrt(scaled_square.numerator // scaled_square.denominator)
    low = floor_root * unit
    if floor_root**2 == scaled_square:
        return low, low
    return low, low + unit


def _combine_commensurate_terms(terms):
    """Group reciprocal distances whose ratio is rational, without factoring coordinates."""
    groups = {}
    for distance, weight in terms:
        for reference in groups:
            ratio = _rational_sqrt(reference / distance)
            if ratio is not None:
                groups[reference] += weight * ratio
                break
        else:
            groups[distance] = weight
    return [(distance, weight) for distance, weight in groups.items() if weight]


def _work(coefficients, permittivity):
    # Aggregate equal pair distances first. This proves cancellation for
    # symmetric configurations even when the common distance is irrational.
    terms = [(distance, weight) for distance, weight in coefficients.items() if weight]
    if not terms:
        return 0.0
    denominator = Fraction(4 * pi) * Fraction(permittivity)
    for bits in (64, 128, 256, 512, 1024):
        if bits == 128:
            # The first bounds usually suffice. A second pass can prove exact
            # cancellation such as 1/d - 2/(2d), even for irrational d.
            terms = _combine_commensurate_terms(terms)
            if not terms:
                return 0.0
        lower = upper = Fraction()
        for distance, weight in terms:
            lo, hi = _inverse_distance_bounds(distance, bits)
            if weight < 0:
                lo, hi = hi, lo
            lower += weight * lo
            upper += weight * hi
        try:
            low, high = float(lower / denominator), float(upper / denominator)
        except OverflowError:
            continue
        if low == high and (low or lower == upper == 0):
            return low
    raise UnsupportedProblem("signed point-charge work cannot be resolved within numerical precision")


def rearrange_point_charges(
    initial: ElectrostaticProblem, final_positions_m: tuple[Vector3, ...],
) -> PointChargeRearrangement:
    """Return external work at unchanged kinetic energy, excluding point self energies.

    Positions follow the initial charge order. Both endpoints are static and use
    the same uniform permittivity. The final problem can seed another rearrangement.
    """
    if not isinstance(initial, ElectrostaticProblem) or initial.conductors:
        raise UnsupportedProblem("requires free point charges without responding conductors")
    positions = tuple(final_positions_m)
    if len(positions) != len(initial.charges):
        raise ValueError("provide a final position for each initial charge")
    if isinstance(initial.permittivity_f_per_m, bool) or any(
        isinstance(site.charge_c, bool) for site in initial.charges
    ):
        raise ValueError("charge and permittivity must be numerical SI values")
    final = replace(initial, charges=tuple(replace(site, position_m=position)
                                          for site, position in zip(initial.charges, positions)))
    coefficients = {}
    for sign, endpoint in ((-1, initial), (1, final)):
        sites = [(Fraction(site.charge_c), tuple(map(Fraction, site.position_m)))
                 for site in endpoint.charges if site.charge_c]
        for i, (charge, point) in enumerate(sites):
            for other_charge, other_point in sites[i+1:]:
                distance = sum((x-y)**2 for x, y in zip(point, other_point))
                if not distance:
                    raise UnsupportedProblem("coincident nonzero point charges have singular pair energy")
                coefficients[distance] = coefficients.get(distance, 0) + sign*charge*other_charge
    work = _work(coefficients, initial.permittivity_f_per_m)
    return PointChargeRearrangement(initial, final, work)
