"""Static equilibrium and small-oscillation stability for 1D responding-conductor + gravity motion."""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import frexp, fsum, isfinite, ldexp, sqrt, ulp
from typing import Literal

from ._resources import cleanup_resources
from .model import PointCharge, Vector3
from .motion_1d import Motion1DSpec, _add, _dot, _geometry, _minus, _norm
from .observables import force_energy
from .solver import UnsupportedProblem

__all__ = ["ConductingGravityEquilibrium", "find_1d_equilibria"]

_INV_PHI = (sqrt(5.0) - 1.0) / 2.0


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


def _scaled_sqrt_ratio(numerator: float, denominator: float) -> float:
    """Compute sqrt(numerator / denominator) without intermediate overflow or underflow."""
    mn, en = frexp(numerator)
    md, ed = frexp(denominator)
    diff = en - ed
    if diff & 1:
        mn *= 2.0
        diff -= 1
    try:
        omega = ldexp(sqrt(mn / md), diff // 2)
    except OverflowError as exc:
        raise UnsupportedProblem("small-oscillation angular frequency outside finite binary64 range") from exc
    if not isfinite(omega) or omega == 0.0:
        raise UnsupportedProblem("small-oscillation angular frequency outside finite binary64 range")
    return omega


def _refine_bracket(eval_net, a: float, fa: float, b: float, fb: float, scale: float) -> float:
    """Robust regula-falsi / bisection hybrid preserving exact bracketing down to ULP scale."""
    lo, hi = a, b
    flo, fhi = fa, fb
    for iteration in range(96):
        width = hi - lo
        if width <= 4.0 * ulp(max(abs(lo), abs(hi), scale)):
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


def _locate_extremum(eval_net, a: float, b: float, *, maximize: bool, scale: float) -> tuple[float, float]:
    """Golden-section search for a local maximum or minimum of F(s) on [a, b]."""
    lo, hi = a, b
    c = hi - _INV_PHI * (hi - lo)
    d = lo + _INV_PHI * (hi - lo)
    fc = eval_net(c)
    fd = eval_net(d)
    for _ in range(96):
        if hi - lo <= 16.0 * ulp(max(abs(lo), abs(hi), scale)):
            break
        better_c = (fc >= fd) if maximize else (fc <= fd)
        if better_c:
            hi, d, fd = d, c, fc
            c = hi - _INV_PHI * (hi - lo)
            fc = eval_net(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + _INV_PHI * (hi - lo)
            fd = eval_net(d)
    s_best = c if ((fc >= fd) if maximize else (fc <= fd)) else d
    return s_best, (fc if s_best == c else fd)


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
        if not isinstance(displacement_bounds_m, (tuple, list)) or len(displacement_bounds_m) != 2:
            raise ValueError("displacement_bounds_m must be a pair of finite numbers")
        bounds_f: list[float] = []
        for v in displacement_bounds_m:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError("displacement_bounds_m must be a pair of finite numbers")
            try:
                fv = float(v)
            except OverflowError as exc:
                raise ValueError("displacement_bounds_m must be a pair of finite numbers") from exc
            if not isfinite(fv):
                raise ValueError("displacement_bounds_m must be a pair of finite numbers")
            bounds_f.append(fv)
        s_lo, s_hi = bounds_f[0], bounds_f[1]
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
            result = (position, elec_along, net, u_elec, u_grav)
        except BaseException as exc:
            close = getattr(solution, "close", None)
            if callable(close):
                cleanup_resources((("equilibrium solution", close),), primary=exc)
            raise
        else:
            close = getattr(solution, "close", None)
            if callable(close):
                close()
        return result

    def net_force(s: float) -> float:
        return observe(s)[2]

    grid = [s_lo + (s_hi - s_lo) * i / (scan_samples - 1) for i in range(scan_samples)]
    values = [net_force(s) for s in grid]
    span_scale = max(domain.scale, s_hi - s_lo)
    dedup_tol = 1e-10 * span_scale
    candidate_roots: list[float] = []

    def add_root(r: float) -> None:
        if all(abs(r - existing) > dedup_tol for existing in candidate_roots):
            candidate_roots.append(r)

    def inspect_extremum(left_s: float, left_v: float, right_s: float, right_v: float, *, maximize: bool) -> None:
        s_ext, f_ext = _locate_extremum(net_force, left_s, right_s, maximize=maximize, scale=span_scale)
        crossed = (f_ext > 0.0) if maximize else (f_ext < 0.0)
        is_strict_interior = (
            (f_ext > left_v and f_ext > right_v) if maximize
            else (f_ext < left_v and f_ext < right_v)
        )
        if not crossed and not is_strict_interior and f_ext != 0.0:
            return
        _, elec_e, _, _, _ = observe(s_ext)
        # Use only the local force cancellation scale at s_ext; distant
        # singular-boundary grid samples must never inflate the zero band.
        f_scale = max(abs(elec_e), abs(grav_along))
        zero_band = 64.0 * ulp(f_scale) if f_scale > 0.0 else 0.0
        if f_ext == 0.0 or abs(f_ext) <= zero_band:
            # Treat extremum values inside the floating-point cancellation band
            # (whether +/- a few ULPs) as a single tangent equilibrium rather
            # than splitting roundoff noise into two spurious close roots.
            if crossed or is_strict_interior or f_ext == 0.0:
                add_root(s_ext)
        elif crossed:
            add_root(_refine_bracket(net_force, left_s, left_v, s_ext, f_ext, span_scale))
            add_root(_refine_bracket(net_force, s_ext, f_ext, right_s, right_v, span_scale))

    for idx, (s, val) in enumerate(zip(grid, values)):
        if val == 0.0:
            add_root(s)
        if idx + 1 < len(grid):
            s_next, v_next = grid[idx + 1], values[idx + 1]
            if (val < 0.0 < v_next) or (v_next < 0.0 < val):
                add_root(_refine_bracket(net_force, s, val, s_next, v_next, span_scale))
        # Inspect interior AND boundary local extrema approaching zero from one
        # side to catch close root pairs inside any cell (including boundary
        # cells) as well as tangent roots (F=0, F'=0).
        if idx == 0:
            v_right = values[1]
            if val < 0.0 and v_right <= val:
                inspect_extremum(grid[0], val, grid[1], v_right, maximize=True)
            elif val > 0.0 and v_right >= val:
                inspect_extremum(grid[0], val, grid[1], v_right, maximize=False)
        elif idx == len(grid) - 1:
            v_left = values[-2]
            if val < 0.0 and v_left <= val:
                inspect_extremum(grid[-2], v_left, grid[-1], val, maximize=True)
            elif val > 0.0 and v_left >= val:
                inspect_extremum(grid[-2], v_left, grid[-1], val, maximize=False)
        else:
            v_prev, v_next = values[idx - 1], values[idx + 1]
            if val < 0.0 and v_prev <= val and v_next <= val:
                inspect_extremum(grid[idx - 1], v_prev, grid[idx + 1], v_next, maximize=True)
            elif val > 0.0 and v_prev >= val and v_next >= val:
                inspect_extremum(grid[idx - 1], v_prev, grid[idx + 1], v_next, maximize=False)

    candidate_roots.sort()
    equilibria: list[ConductingGravityEquilibrium] = []
    for root_idx, s_eq in enumerate(candidate_roots):
        pos, elec_along, net, u_elec, u_grav = observe(s_eq)
        clearance = domain.clearance(s_eq)
        dist_lo = s_eq - domain.lower
        dist_hi = domain.upper - s_eq
        finite_dists = [d for d in (dist_lo, dist_hi) if isfinite(d)]
        local_scale = min(finite_dists) if finite_dists else max(domain.scale, abs(s_eq), s_hi - s_lo)
        step = min(1e-5 * local_scale, 0.25 * clearance)
        neighbor_dists = []
        if root_idx > 0:
            neighbor_dists.append(s_eq - candidate_roots[root_idx - 1])
        if root_idx + 1 < len(candidate_roots):
            neighbor_dists.append(candidate_roots[root_idx + 1] - s_eq)
        if neighbor_dists:
            step = min(step, 0.25 * min(neighbor_dists))
        min_step = 128.0 * ulp(max(abs(s_eq), local_scale))
        if step <= min_step or not (domain.safe(s_eq - step) and domain.safe(s_eq + step)):
            stiffness = 0.0
            stability: Literal["stable", "unstable", "indeterminate"] = "indeterminate"
            omega = None
        else:
            f_plus = net_force(s_eq + step)
            f_minus = net_force(s_eq - step)
            # k_eff = -dF/ds = d^2U/ds^2; positive k_eff => restoring force => stable equilibrium.
            stiffness = -(f_plus - f_minus) / (2.0 * step)
            force_scale = max(abs(elec_along), abs(grav_along), abs(f_plus), abs(f_minus))
            stiffness_floor = 1e-8 * force_scale / local_scale if local_scale > 0.0 else 0.0
            if not isfinite(stiffness) or abs(stiffness) <= stiffness_floor:
                stability = "indeterminate"
                omega = None
            elif f_minus > 0.0 and f_plus < 0.0 and stiffness > 0.0:
                stability = "stable"
                omega = _scaled_sqrt_ratio(stiffness, spec.mass_kg)
            elif f_minus < 0.0 and f_plus > 0.0 and stiffness < 0.0:
                stability = "unstable"
                omega = None
            else:
                # Same-sign or zero perturbations (e.g. quadratic tangent F'=0)
                # do not certify linear stability.
                stability = "indeterminate"
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
