"""Bounded output-time grids shared by scalar and vector motion.

Floating-point division of an exact decimal interval count may round upward
(e.g. .07/.01). Record the requested endpoint once, not twice.
"""
from math import ceil, isfinite, ulp


def output_times(duration: float, interval: float) -> tuple[float, ...]:
    """Return increasing positive sample times ending exactly at duration."""
    if not (isfinite(duration) and isfinite(interval) and 0 < interval <= duration):
        raise ValueError('output interval must be positive and at most the duration')
    ratio = duration / interval
    if not isfinite(ratio):
        raise ValueError('at most 2000 requested output intervals')
    nearest = round(ratio)
    count = nearest if abs(ratio-nearest) <= 4*ulp(ratio) else ceil(ratio)
    if not 1 <= count <= 2000:
        raise ValueError('at most 2000 requested output intervals')
    return tuple(i*interval for i in range(1,count) if i*interval < duration) + (duration,)
