"""Shared bounded direction normalization; ordinary binary64 arithmetic is unchanged.

Callers retain their own type, shape and physical-domain checks. This helper
only fixes loss of relative accuracy when the norm itself is subnormal.
"""
from __future__ import annotations

from math import hypot, isfinite
from sys import float_info


def unit_direction(values, error_message="direction must be finite and nonzero"):
    """Normalize a nonzero finite direction without amplifying a subnormal norm.

    A zero, nonfinite, or overflowing norm still refuses. Do not use this to
    replace M40a's original raw query rays: those have different precision
    semantics and are deliberately normalized inside their physical kernel.
    """
    length = hypot(*values)
    if not isfinite(length) or length == 0:
        raise ValueError(error_message)
    if length < float_info.min:
        scale = max(abs(x) for x in values)
        values = tuple(x / scale for x in values)
        length = hypot(*values)
    return tuple(x / length for x in values)
