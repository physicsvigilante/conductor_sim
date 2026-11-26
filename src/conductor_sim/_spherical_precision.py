"""Round sums of rational coefficients and square roots at the observable boundary."""

from fractions import Fraction
from math import isfinite, isqrt

from .solver import UnsupportedProblem


def _rational_root(value: Fraction) -> Fraction | None:
    n, d = isqrt(value.numerator), isqrt(value.denominator)
    if n * n == value.numerator and d * d == value.denominator:
        return Fraction(n, d)
    return None


def _root_interval(value: Fraction, bits: int) -> tuple[Fraction, Fraction]:
    n, d = value.numerator, value.denominator
    shift = bits - (n.bit_length() - d.bit_length()) // 2
    # Bound sqrt(n/d) on a binary grid with approximately `bits` significant bits.
    scaled_n, scaled_d = (n << (2 * shift), d) if shift >= 0 else (n, d << (-2 * shift))
    step = Fraction(1, 1 << shift) if shift >= 0 else Fraction(1 << (-shift))
    lower = isqrt(scaled_n // scaled_d)
    return lower * step, (lower + 1) * step


class _RootSum:
    """Accumulate c*sqrt(s) terms exactly until their final conversion to binary64."""

    def __init__(self, constant: Fraction = Fraction()) -> None:
        self._terms: dict[Fraction, Fraction] = {}
        self.add(constant)

    def add(self, coefficient: Fraction, radicand: Fraction = Fraction(1)) -> None:
        if not coefficient:
            return
        root = _rational_root(radicand)
        if root is not None:
            coefficient *= root
            radicand = Fraction(1)
        combined = self._terms.get(radicand, Fraction()) + coefficient
        if combined:
            self._terms[radicand] = combined
        else:
            self._terms.pop(radicand, None)

    def add_scaled(self, other: "_RootSum", scale: Fraction,
                   radicand: Fraction = Fraction(1)) -> None:
        for s, c in other._terms.items():
            self.add(scale * c, radicand * s)

    def _collect_equivalent_roots(self) -> "_RootSum":
        # Different squared distances can produce the same irrational root up to
        # a rational factor. Resolve these exact cancellations before refining.
        result = _RootSum()
        for s, c in self._terms.items():
            for base in result._terms:
                factor = _rational_root(s / base)
                if factor is not None:
                    result.add(c * factor, base)
                    break
            else:
                result.add(c, s)
        return result

    def _bounds(self, bits: int) -> tuple[Fraction, Fraction]:
        lower = upper = self._terms.get(Fraction(1), Fraction())
        for s, c in self._terms.items():
            if s == 1:
                continue
            a, b = _root_interval(s, bits)
            if c < 0:
                a, b = b, a
            lower += c * a
            upper += c * b
        return lower, upper

    def rounded(self, label: str, *, square_over: Fraction | None = None) -> float:
        terms = self
        for bits in (96, 192, 384, 768, 1536, 3072, 6144, 12288):
            if bits == 192:
                terms = self._collect_equivalent_roots()
            if square_over is not None and len(terms._terms) == 1:
                (s, c), = terms._terms.items()
                a = b = s * c**2 / square_over
            else:
                a, b = terms._bounds(bits)
                if square_over is not None:
                    low = Fraction() if a <= 0 <= b else min(a * a, b * b)
                    a, b = low / square_over, max(a * a, b * b) / square_over
            if a == b == 0:
                return 0.0
            if a <= 0 <= b:
                continue
            lo, hi = _float_or_infinity(a), _float_or_infinity(b)
            if lo == hi:
                if not isfinite(lo) or lo == 0:
                    raise UnsupportedProblem(f"{label} outside the binary64 range")
                return lo
        raise UnsupportedProblem(f"{label} cannot be resolved to binary64 precision")


def _float_or_infinity(value: Fraction) -> float:
    try:
        return float(value)
    except OverflowError:
        return float("inf") if value > 0 else float("-inf")
