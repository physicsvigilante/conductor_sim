"""Shared, dimension-independent RK4 step used by electrostatic motion solvers.

This is a numerical stepper, not a second electrostatic force provider. The
accepted M16a timestep controller and its sample/event semantics stay intact.
"""
from __future__ import annotations


def rk4_step(state, h, rhs):
    """Advance one tuple state using an explicitly supplied derivative."""
    k1 = rhs(state)
    def shifted(k, scale):
        return tuple(value + scale * slope for value, slope in zip(state, k))
    k2 = rhs(shifted(k1, h / 2))
    k3 = rhs(shifted(k2, h / 2))
    k4 = rhs(shifted(k3, h))
    return tuple(value + h * (a + 2*b + 2*c + d)/6
                 for value, a, b, c, d in zip(state, k1, k2, k3, k4))
