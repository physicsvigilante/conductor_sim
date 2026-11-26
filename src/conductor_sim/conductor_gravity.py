"""Static equilibrium and small-oscillation stability for 1D responding-conductor + gravity motion."""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import fsum, hypot, isfinite, sqrt, ulp
from typing import Literal

from ._resources import cleanup_resources
from .model import PointCharge, Vector3
from .motion_1d import Motion1DSpec, _add, _dot, _geometry, _minus, _norm
from .observables import force_energy
from .solver import UnsupportedProblem

__all__ = ["ConductingGravityEquilibrium", "find_1d_equilibria"]


@dataclass(frozen=True, slots=True)
class ConductingGravityEquilibrium:
    """One 1D static force-balance root along a certified responding-conductor symmetry axis."""

    position_m: Vector3
    displacement_m: float
    electric_force_n: float
    gravitational_force_n: float
    net_force_n: float
    stiffness_n_per_m: float
    stability: Literal["stable", "unstable", "indeterminate"]
    small_oscillation_angular_frequency_rad_s: float | None
    electric_interaction_energy_j: float
    gravitational_potential_energy_j: float
    total_potential_energy_j: float


def _refine_bracket(eval_net, a: float, fa: float, b: float, fb: float, scale: float) -> float:
    """Robust Illinois regula-falsi / bisection hybrid preserving exact bracketing."""
    lo, hi = a, b
    flo, fhi = fa, fb
    for iteration in range(96):
        width = hi - lo
        if width <= 4.0 * ulp(max(abs(lo), abs(hi), scale, 1.0)):
            break
        if iteration % 3 != 2 and fhi != flo:
            mid = lo - flo * width / (fhi - flo)
            if not (lo + 0.05 * width < mid < hi - 0.05 * width):
                mid = 0.5 * (lo + hi)
        else:
            mid = 0.5 * (lo + hi)
        fmid = eval_net(mid)
        if fmid == 0.0:
            return mid
        if (flo < 0.0 < fmid) or (fmid < 0.0 < flo):
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return 0.5 * (lo + hi)


def find_1d_equilibria(
    spec: Motion1DSpec,
    backend,
    *,
    displacement_bounds_m: tuple[float, float] | None = None,
    scan_samples: int = 129,
) -> tuple[ConductingGravityEquilibrium, ...]:
    """Locate 1D force equilibria and local linear stability along a certified Motion1DSpec axis.

    Uses the exact same responding-conductor backend, force_energy provider, symmetry
    qualification, and optional UniformGravity as integrate_motion_1d. For semi-infinite
    vacuum intervals (e.g., single plane or exterior solid sphere), supply explicit finite
    displacement_bounds_m=(s_min, s_max) strictly inside the surface-guard domain.
    """
    if not isinstance(spec, Motion1DSpec):
        raise TypeError("spec must be Motion1DSpec")
    if isinstance(scan_samples, bool) or not isinstance(scan_samples, int) or not 9 <= scan_samples <= 4097:
        raise ValueError("scan_samples must be an integer between 9 and 4097")

    domain = _geometry(spec)
    safe_lo = domain.lower + domain.guard
    safe_hi = domain.upper - domain.guard
    if displacement_bounds_m is None:
        if not (isfinite(safe_lo) and isfinite(safe_hi)):
            raise UnsupportedProblem(
                "unbounded 1D domain requires explicit finite displacement_bounds_m=(s_min, s_max)"
            )
        margin = max(0.05 * domain.guard, 1e-9 * domain.scale)
        s_lo = safe_lo + margin
        s_hi = safe_hi - margin
    else:
        if (
            not isinstance(displacement_bounds_m, (tuple, list))
            or len(displacement_bounds_m) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) for v in displacement_bounds_m)
        ):
            raise ValueError("displacement_bounds_m must be a pair of finite numbers")
        s_lo, s_hi = float(displacement_bounds_m[0]), float(displacement_bounds_m[1])
        if s_lo <= safe_lo or s_hi >= safe_hi:
            raise UnsupportedProblem("displacement_bounds_m must stay strictly inside the safe vacuum domain")
    if s_hi <= s_lo:
        raise ValueError("displacement_bounds_m must satisfy s_min < s_max inside the safe domain")

    original = spec.selected_charge
    gravity = spec.gravity
    grav_along = (
        _dot(gravity.force_at(original.position_m, mass_kg=spec.mass_kg), spec.axis)
        if gravity is not None else 0.0
    )

    def observe(s: float):
        if not domain.safe(s):
            raise UnsupportedProblem("equilibrium probe crossed the conductor surface guard")
        position = _add(original.position_m, spec.axis, s)
        charges = list(spec.problem.charges)
        charges[spec.charge_index] = PointCharge(original.charge_c, position)
        problem = replace(spec.problem, charges=tuple(charges))
        supports = getattr(backend, "supports", None)
        if callable(supports) and not supports(problem):
            raise UnsupportedProblem("selected analytical backend does not support this equilibrium state")
        solution = backend.solve(problem)
        try:
            obs = force_energy(solution)
            if (
                len(obs.charges) != 1
                or obs.charges[0].source.position_m != position
                or obs.charges[0].source.charge_c != original.charge_c
            ):
                raise UnsupportedProblem("authored charge identity became ambiguous")
            elec_vec = obs.charges[0].total_force_n
            elec_along = _dot(elec_vec, spec.axis)
            transverse = _norm(_minus(elec_vec, tuple(elec_along * u for u in spec.axis)))
            if transverse > max(1e-20, 5e-9 * _norm(elec_vec)):
                raise UnsupportedProblem("transverse electrostatic force breaks 1D equilibrium")
            net = fsum((elec_along, grav_along)) if gravity is not None else elec_along
            u_elec = obs.energy.total_j
            u_grav = gravity.potential_energy_at(position, mass_kg=spec.mass_kg) if gravity is not None else 0.0
            if not all(isfinite(v) for v in (elec_along, net, u_elec, u_grav, fsum((u_elec, u_grav)))):
                raise UnsupportedProblem("nonfinite equilibrium force or energy")
            return position, elec_along, net, u_elec, u_grav
        except BaseException as exc:
            close = getattr(solution, "close", None)
            if callable(close):
                cleanup_resources((("equilibrium solution", close),), primary=exc)
            raise
        else:
            close = getattr(solution, "close", None)
            if callable(close):
                close()

    def net_force(s: float) -> float:
        return observe(s)[2]

    grid = [s_lo + (s_hi - s_lo) * i / (scan_samples - 1) for i in range(scan_samples)]
    values = [net_force(s) for s in grid]
    roots: list[float] = []
    dedup_tol = 1e-10 * max(1.0, domain.scale, s_hi - s_lo)

    for idx, (s, val) in enumerate(zip(grid, values)):
        if val == 0.0:
            if not roots or abs(s - roots[-1]) > dedup_tol:
                roots.append(s)
        if idx + 1 < len(grid):
            s_next, v_next = grid[idx + 1], values[idx + 1]
            if (val < 0.0 < v_next) or (v_next < 0.0 < val):
                root = _refine_bracket(net_force, s, val, s_next, v_next, domain.scale)
                if not roots or abs(root - roots[-1]) > dedup_tol:
                    roots.append(root)

    equilibria: list[ConductingGravityEquilibrium] = []
    for s_eq in roots:
        pos, elec_along, net, u_elec, u_grav = observe(s_eq)
        clearance = domain.clearance(s_eq)
        step = min(1e-6 * max(domain.scale, abs(s_eq), 1.0), 0.25 * clearance)
        if step <= 0.0 or not (domain.safe(s_eq - step) and domain.safe(s_eq + step)):
            stiffness = 0.0
            stability: Literal["stable", "unstable", "indeterminate"] = "indeterminate"
            omega = None
        else:
            f_plus = net_force(s_eq + step)
            f_minus = net_force(s_eq - step)
            # k_eff = -dF/ds = d^2U/ds^2; positive k_eff => restoring force => stable equilibrium.
            stiffness = -(f_plus - f_minus) / (2.0 * step)
            force_scale = max(abs(elec_along), abs(grav_along), abs(f_plus), abs(f_minus), 1e-30)
            stiffness_floor = 1e-10 * force_scale / max(domain.scale, 1.0)
            if not isfinite(stiffness) or abs(stiffness) <= stiffness_floor:
                stability = "indeterminate"
                omega = None
            elif stiffness > 0.0:
                stability = "stable"
                omega = sqrt(stiffness / spec.mass_kg)
                if not isfinite(omega):
                    raise UnsupportedProblem("small-oscillation angular frequency outside finite binary64 range")
            else:
                stability = "unstable"
                omega = None
        equilibria.append(ConductingGravityEquilibrium(
            position_m=pos,
            displacement_m=s_eq,
            electric_force_n=elec_along,
            gravitational_force_n=grav_along,
            net_force_n=net,
            stiffness_n_per_m=stiffness,
            stability=stability,
            small_oscillation_angular_frequency_rad_s=omega,
            electric_interaction_energy_j=u_elec,
            gravitational_potential_energy_j=u_grav,
            total_potential_energy_j=fsum((u_elec, u_grav)),
        ))
    return tuple(equilibria)
