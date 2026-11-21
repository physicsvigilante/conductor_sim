"""Bounded reciprocal-distance summation for static electrostatic observations."""
from fractions import Fraction
from math import isqrt, pi

from .solver import UnsupportedProblem


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


def reciprocal_distance_sum_at_most(terms, maximum):
    """Certify a geometric row of positive weights against an exact limit."""
    for bits in (64, 128, 256, 512, 1024):
        lower = upper = Fraction(0)
        for squared_distance, weight in terms:
            lo, hi = _inverse_distance_bounds(squared_distance, bits)
            lower += weight * lo
            upper += weight * hi
        if upper <= maximum:
            return True
        if lower > maximum:
            return False
    raise UnsupportedProblem("far-separation admission unresolved at the threshold")


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


def coulomb_sum(coefficients, permittivity, *, offset=0):
    """Round the Coulomb sum and an optional exact additive observable together."""
    denominator = Fraction(4 * pi) * Fraction(permittivity)
    if offset:
        # A constant is a unit reciprocal-distance term in the scaled numerator.
        coefficients = coefficients.copy()
        coefficients[1] = coefficients.get(1, 0) + denominator*offset
    return reciprocal_distance_sum(coefficients, denominator)


def reciprocal_distance_sum(coefficients, denominator=1):
    """Round a signed sum of weights over distance, divided by a positive factor."""
    # Aggregate equal pair distances first. This proves cancellation for
    # symmetric configurations even when the common distance is irrational.
    terms = [(distance, weight) for distance, weight in coefficients.items() if weight]
    if not terms:
        return 0.0
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
    raise UnsupportedProblem("signed Coulomb sum cannot be resolved within numerical precision")
