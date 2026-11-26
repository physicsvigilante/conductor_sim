"""Static equilibrium and small-oscillation stability for 1D responding-conductor + gravity motion."""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import frexp, fsum, isfinite, ldexp, sqrt, ulp
from sys import float_info
from typing import Literal

from ._resources import cleanup_resources
from .model import PointCharge, Vector3
from .motion_1d import Motion1DSpec, _add, _dot, _geometry, _minus, _norm
from .observables import force_energy
from .solver import UnsupportedProblem

__all__ = ["ConductingGravityEquilibrium", "find_1d_equilibria"]

_INV_PHI = (sqrt(5.0) - 1.0) / 2.0
_MIN_NORMAL = float_info.min


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
    converged = False
    for iteration in range(2048):
        width = hi - lo
        if width <= 4.0 * ulp(max(abs(lo), abs(hi), scale)):
            converged = True
            break
        if iteration % 2 == 0 and fhi != flo:
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
    if not converged and (hi - lo) > 16.0 * ulp(max(abs(lo), abs(hi), scale)):
        raise UnsupportedProblem("1D equilibrium bracket failed to converge to binary64 precision")
    mid = 0.5 * (lo + hi)
    fmid = eval_net(mid)
    return min(((abs(flo), lo), (abs(fmid), mid), (abs(fhi), hi)), key=lambda item: item[0])[1]


def _locate_extremum(
    eval_net,
    a: float,
    b: float,
    *,
    maximize: bool,
    scale: float,
    prefer_left: bool = True,
) -> tuple[float, float]:
    """Golden-section search for a local maximum or minimum of F(s) on [a, b]."""
    lo, hi = a, b
    flo = eval_net(lo)
    fhi = eval_net(hi)
    c = hi - _INV_PHI * (hi - lo)
    d = lo + _INV_PHI * (hi - lo)
    fc = eval_net(c)
    fd = eval_net(d)
    for _ in range(3000):
        if hi - lo <= 16.0 * ulp(max(abs(lo), abs(hi), scale)):
            break
        if fc != fd:
            better_c = (fc > fd) if maximize else (fc < fd)
        elif flo != fc and fhi == fd:
            # Flat tail on the right (e.g. wide scan toward +infinity): shrink toward left.
            better_c = True
        elif fhi != fd and flo == fc:
            # Flat tail on the left (e.g. wide scan toward -infinity): shrink toward right.
            better_c = False
        else:
            # All four probes tied (e.g. conductor-side endpoint coincides with
            # asymptotic constant force): shrink toward the physical conductor side.
            better_c = prefer_left
        if better_c:
            hi, fhi, d, fd = d, fd, c, fc
            c = hi - _INV_PHI * (hi - lo)
            fc = eval_net(c)
        else:
            lo, flo, c, fc = c, fc, d, fd
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
    grav_vec = (
        gravity.force_at(original.position_m, mass_kg=spec.mass_kg)
        if gravity is not None else (0.0, 0.0, 0.0)
    )
    grav_along = _dot(grav_vec, spec.axis) if gravity is not None else 0.0

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
            net_vec = (
                tuple(fsum((e, g)) for e, g in zip(elec_vec, grav_vec))
                if gravity is not None else elec_vec
            )
            transverse_net = _norm(_minus(net_vec, tuple(net * u for u in spec.axis)))
            f_vec_scale = max(_norm(elec_vec), _norm(grav_vec))
            if (transverse_net > max(5e-9 * _norm(net_vec), 256.0 * ulp(f_vec_scale)) or
                    0.5 * (transverse_net / spec.mass_kg) * (spec.duration_s * spec.duration_s) > 1e-9 * domain.scale):
                raise UnsupportedProblem("transverse net force breaks 1D equilibrium")
            u_elec = obs.energy.total_j
            u_grav = gravity.potential_energy_at(position, mass_kg=spec.mass_kg) if gravity is not None else 0.0
            if not all(isfinite(v) for v in (elec_along, net, u_elec, u_grav, fsum((u_elec, u_grav)))):
                raise UnsupportedProblem("nonfinite equilibrium force or energy")
            if (0.0 < abs(elec_along) < _MIN_NORMAL) or (0.0 < abs(grav_along) < _MIN_NORMAL):
                raise UnsupportedProblem("equilibrium force underflows into subnormal binary64 precision")
            result = (position, elec_along, net, u_elec, u_grav, _norm(net_vec))
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

    def zero_band(f_scale: float) -> float:
        if f_scale <= 0.0:
            return 0.0
        # Cap at 1e-12 * f_scale so subnormal forces (where ulp(f_scale) is a
        # large fraction of f_scale) never erase genuine sign-changing brackets.
        return min(64.0 * ulp(f_scale), 1e-12 * f_scale)

    grid = [
        s_lo if i == 0 else (
            s_hi if i == scan_samples - 1
            else ((scan_samples - 1 - i) / (scan_samples - 1)) * s_lo + (i / (scan_samples - 1)) * s_hi
        )
        for i in range(scan_samples)
    ]
    values: list[float] = []
    scales: list[float] = []
    for s in grid:
        _, elec_s, net_s, _, _, _ = observe(s)
        values.append(net_s)
        scales.append(max(abs(elec_s), abs(grav_along)))
    # Treat grid values inside the local floating-point cancellation band as
    # numerically zero so +/-few-ULP noise at a grid-aligned tangent cannot
    # manufacture false opposite-sign brackets.
    eff_values = [0.0 if (v == 0.0 or abs(v) <= zero_band(sc)) else v for v, sc in zip(values, scales)]
    # Keep convergence and deduplication tied to the local physical domain scale
    # (never the full scan width s_hi - s_lo) so wide scan bounds such as 1e6 m
    # cannot merge distinct nearby roots.
    local_root_scale = domain.scale
    candidate_roots: list[float] = []

    def add_root(r: float) -> None:
        if all(
            abs(r - existing) > 128.0 * ulp(max(abs(r), abs(existing), local_root_scale))
            for existing in candidate_roots
        ):
            candidate_roots.append(r)

    def inspect_extremum(
        left_s: float,
        left_v: float,
        right_s: float,
        right_v: float,
        *,
        maximize: bool,
        allow_tangent: bool = True,
    ) -> None:
        if isfinite(domain.lower) and not isfinite(domain.upper):
            prefer_left = True
        elif isfinite(domain.upper) and not isfinite(domain.lower):
            prefer_left = False
        else:
            prefer_left = (left_s - domain.lower) <= (domain.upper - right_s)
        s_ext, f_ext = _locate_extremum(
            net_force, left_s, right_s, maximize=maximize, scale=local_root_scale, prefer_left=prefer_left,
        )
        crossed = (f_ext > 0.0) if maximize else (f_ext < 0.0)
        is_strict_interior = (
            (f_ext > left_v and f_ext > right_v) if maximize
            else (f_ext < left_v and f_ext < right_v)
        )
        if not crossed and not is_strict_interior and f_ext != 0.0:
            return
        _, elec_e, _, _, _, _ = observe(s_ext)
        # Use only the local force cancellation scale at s_ext; distant
        # singular-boundary grid samples must never inflate the zero band.
        band = zero_band(max(abs(elec_e), abs(grav_along)))
        if f_ext == 0.0 or abs(f_ext) <= band:
            # Treat extremum values inside the floating-point cancellation band
            # (whether +/- a few ULPs) as a single tangent equilibrium rather
            # than splitting roundoff noise into two spurious close roots.
            if allow_tangent and (crossed or is_strict_interior or f_ext == 0.0):
                add_root(s_ext)
        elif crossed:
            if (left_v < 0.0 < f_ext) or (f_ext < 0.0 < left_v):
                add_root(_refine_bracket(net_force, left_s, left_v, s_ext, f_ext, local_root_scale))
            if (f_ext < 0.0 < right_v) or (right_v < 0.0 < f_ext):
                add_root(_refine_bracket(net_force, s_ext, f_ext, right_s, right_v, local_root_scale))

    n_grid = len(grid)
    # 1. Exact/near-zero contiguous runs on the grid (handles exact boundary
    #    roots, simple crossings near a grid point, and grid-aligned tangents).
    run_idx = 0
    while run_idx < n_grid:
        if eff_values[run_idx] != 0.0:
            run_idx += 1
            continue
        run_end = run_idx
        while run_end + 1 < n_grid and eff_values[run_end + 1] == 0.0:
            run_end += 1
        if run_idx == 0 and run_end == n_grid - 1:
            raise UnsupportedProblem(
                "net force vanishes or underflows across the entire equilibrium scan interval"
            )
        if run_end > run_idx and any(
            values[k] == 0.0 and values[k + 1] == 0.0 and net_force(0.5 * (grid[k] + grid[k + 1])) == 0.0
            for k in range(run_idx, run_end)
        ):
            raise UnsupportedProblem(
                "net force vanishes or underflows across an equilibrium scan plateau"
            )
        has_exact_zero = any(values[k] == 0.0 for k in range(run_idx, run_end + 1))
        if run_idx > 0 and run_end < n_grid - 1:
            lv, rv = eff_values[run_idx - 1], eff_values[run_end + 1]
            if (lv < 0.0 < rv) or (rv < 0.0 < lv):
                add_root(_refine_bracket(
                    net_force, grid[run_idx - 1], values[run_idx - 1],
                    grid[run_end + 1], values[run_end + 1], local_root_scale,
                ))
            else:
                inspect_extremum(
                    grid[run_idx - 1], values[run_idx - 1],
                    grid[run_end + 1], values[run_end + 1],
                    maximize=(lv < 0.0),
                    allow_tangent=True,
                )
        else:
            for k in range(run_idx, run_end + 1):
                if values[k] == 0.0:
                    add_root(grid[k])
            if run_idx == 0 and run_end + 1 < n_grid:
                rv = values[run_end + 1]
                if has_exact_zero:
                    # An exact boundary zero may still share its first cell with
                    # a second interior root if F genuinely crosses beyond zero_band.
                    inspect_extremum(
                        grid[run_end], values[run_end],
                        grid[run_end + 1], rv,
                        maximize=(eff_values[run_end + 1] < 0.0),
                        allow_tangent=False,
                    )
                elif (values[run_end] < 0.0 < rv) or (rv < 0.0 < values[run_end]):
                    add_root(_refine_bracket(net_force, grid[run_end], values[run_end], grid[run_end + 1], rv, local_root_scale))
                else:
                    inspect_extremum(grid[0], values[0], grid[run_end + 1], rv, maximize=(rv < 0.0))
            elif run_end == n_grid - 1 and run_idx > 0:
                lv = values[run_idx - 1]
                if has_exact_zero:
                    inspect_extremum(
                        grid[run_idx - 1], lv,
                        grid[run_idx], values[run_idx],
                        maximize=(eff_values[run_idx - 1] < 0.0),
                        allow_tangent=False,
                    )
                elif (lv < 0.0 < values[run_idx]) or (values[run_idx] < 0.0 < lv):
                    add_root(_refine_bracket(net_force, grid[run_idx - 1], lv, grid[run_idx], values[run_idx], local_root_scale))
                else:
                    inspect_extremum(grid[run_idx - 1], lv, grid[-1], values[-1], maximize=(lv < 0.0))
        run_idx = run_end + 1

    # 2. Opposite-sign cells strictly outside the cancellation band.
    for idx in range(n_grid - 1):
        ev, ev_next = eff_values[idx], eff_values[idx + 1]
        if (ev < 0.0 < ev_next) or (ev_next < 0.0 < ev):
            add_root(_refine_bracket(net_force, grid[idx], values[idx], grid[idx + 1], values[idx + 1], local_root_scale))

    # 3. Local discrete extrema (interior and boundary) approaching zero from
    #    one side, to catch close root pairs inside a single cell or off-grid
    #    tangent roots (F=0, F'=0).
    for idx in range(n_grid):
        ev = eff_values[idx]
        if ev == 0.0:
            continue
        if idx == 0:
            ev_right = eff_values[1]
            if ev < 0.0 and ev_right < 0.0 and values[1] <= values[0]:
                inspect_extremum(grid[0], values[0], grid[1], values[1], maximize=True)
            elif ev > 0.0 and ev_right > 0.0 and values[1] >= values[0]:
                inspect_extremum(grid[0], values[0], grid[1], values[1], maximize=False)
        elif idx == n_grid - 1:
            ev_left = eff_values[-2]
            if ev < 0.0 and ev_left < 0.0 and values[-2] <= values[-1]:
                inspect_extremum(grid[-2], values[-2], grid[-1], values[-1], maximize=True)
            elif ev > 0.0 and ev_left > 0.0 and values[-2] >= values[-1]:
                inspect_extremum(grid[-2], values[-2], grid[-1], values[-1], maximize=False)
        else:
            ev_prev, ev_next = eff_values[idx - 1], eff_values[idx + 1]
            if ev < 0.0 and ev_prev < 0.0 and ev_next < 0.0 and values[idx - 1] <= values[idx] and values[idx + 1] <= values[idx]:
                inspect_extremum(grid[idx - 1], values[idx - 1], grid[idx + 1], values[idx + 1], maximize=True)
            elif ev > 0.0 and ev_prev > 0.0 and ev_next > 0.0 and values[idx - 1] >= values[idx] and values[idx + 1] >= values[idx]:
                inspect_extremum(grid[idx - 1], values[idx - 1], grid[idx + 1], values[idx + 1], maximize=False)

    candidate_roots.sort()
    equilibria: list[ConductingGravityEquilibrium] = []
    for root_idx, s_eq in enumerate(candidate_roots):
        pos, elec_along, net, u_elec, u_grav, net_mag = observe(s_eq)
        eq_f_scale = max(abs(elec_along), abs(grav_along))
        if max(abs(net), net_mag) > max(1e-6 * eq_f_scale, 256.0 * ulp(eq_f_scale)):
            raise UnsupportedProblem(
                "displacement coordinates cannot resolve equilibrium force balance to required precision"
            )
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
