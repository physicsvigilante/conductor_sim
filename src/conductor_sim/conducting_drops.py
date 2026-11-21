"""Compose independent, static conducting-drop equilibria at constant density."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import fsum, isclose, isfinite, pi

from .mixed_radial import MixedRadialBackend, MixedRadialSolution
from .model import EPSILON_0, ElectrostaticProblem, FixedNetCharge, SolidSphere


@dataclass(frozen=True, slots=True)
class ConductingDropEquilibria:
    initial: tuple[MixedRadialSolution, ...]
    final: MixedRadialSolution
    constituents: tuple[SolidSphere, ...]


def _solve_drop(drop, permittivity):
    problem = ElectrostaticProblem(conductors=(drop,), permittivity_f_per_m=permittivity)
    state = MixedRadialBackend().solve(problem)
    # Check the radial backend's intermediate rounding before exposing its result.
    # A small q/r can underflow even when q/(4*pi*epsilon*r) is representable.
    exact = Fraction(drop.constraint.charge_c) / (
        Fraction(4*pi) * Fraction(permittivity) * Fraction(drop.radius_m))
    voltage = state.conductor_results[0].potential_v
    if (not isfinite(voltage) or (exact and not voltage)
            or not isclose(voltage, float(exact), rel_tol=1e-13, abs_tol=0)):
        raise ValueError("drop potential is outside the supported numerical range")
    return state


def _combined_state(constituents, permittivity):
    if len(constituents) < 2:
        raise ValueError("stage requires at least two constituent drops")
    for drop in constituents:
        _validate_drop(drop)
    scale = max(drop.radius_m for drop in constituents)
    volume_ratio = fsum((drop.radius_m / scale) ** 3 for drop in constituents)
    radius = scale * volume_ratio ** (1 / 3)
    charge = float(sum((Fraction(drop.constraint.charge_c) for drop in constituents), Fraction()))
    return _solve_drop(SolidSphere(radius, FixedNetCharge(charge), name="combined drop"),
                       permittivity)


def _validate_drop(drop):
    if not isinstance(drop, SolidSphere) or not isinstance(drop.constraint, FixedNetCharge):
        raise ValueError("each drop must be an isolated solid sphere with fixed net charge")
    if type(drop.radius_m) is bool or type(drop.constraint.charge_c) is bool:
        raise ValueError("expected numerical radius and charge")


def combine_isolated_drops(
    drops: tuple[SolidSphere | ConductingDropEquilibria, ...],
    *, permittivity_f_per_m: float = EPSILON_0,
) -> ConductingDropEquilibria:
    """Conserve volume and charge between isolated spherical equilibria.

    Each input has its own space. Pass a previous result as a stage to retain
    constituent charges and volumes before rounding its displayed final sphere.
    """
    drops = tuple(drops)
    if len(drops) < 2:
        raise ValueError("combining drops requires at least two spheres")
    if type(permittivity_f_per_m) is bool:
        raise ValueError("expected a numerical permittivity")
    initial = []
    constituents = []
    for drop in drops:
        if isinstance(drop, ConductingDropEquilibria):
            if drop.final.permittivity_f_per_m != permittivity_f_per_m:
                raise ValueError("all stages must use the same permittivity")
            if drop.final != _combined_state(drop.constituents, permittivity_f_per_m):
                raise ValueError("stage equilibrium disagrees with its constituent drops")
            initial.append(drop.final)
            constituents.extend(drop.constituents)
        else:
            _validate_drop(drop)
            initial.append(_solve_drop(drop, permittivity_f_per_m))
            constituents.append(drop)
    final = _combined_state(constituents, permittivity_f_per_m)
    return ConductingDropEquilibria(tuple(initial), final, tuple(constituents))
