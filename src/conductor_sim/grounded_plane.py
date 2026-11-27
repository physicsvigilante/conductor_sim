"""Exact image-charge electrostatics above an ideal infinite grounded plane.

The positive normal defines the physical vacuum half-space. The grounded
metal occupies the other half-space. This reference is fully 3D: unlike an
axisymmetric FEMM solve, point sources need not share a common axis.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import frexp, fsum, hypot, isclose, isfinite, ldexp, pi

from .model import ConductingPlane, ElectrostaticProblem, Grounded, PointCharge, Vector3, _position
from .solver import ConductorResult, SingularField, UnsupportedProblem


def _minus(a: Vector3, b: Vector3) -> Vector3:
    return tuple(a[i] - b[i] for i in range(3))


def _dot(a: Vector3, b: Vector3) -> float:
    return fsum(a[i] * b[i] for i in range(3))


def _mirror(point: Vector3, plane: ConductingPlane) -> Vector3:
    h = plane.signed_distance_m(point)
    return tuple(point[i] - 2*h*plane.normal[i] for i in range(3))


def _mirror_displacement(target: Vector3, source: Vector3, plane: ConductingPlane) -> Vector3:
    """Translation-invariant displacement ``target - mirror(source, plane)``.

    Algebraically ``(target - source) + 2*h(source)*n``, avoiding world-coordinate
    cancellation when ``|source| >> h(source)`` (including ``target == source``).
    """
    h = plane.signed_distance_m(source)
    return tuple(
        (target[i] - source[i]) + (2.0 * h * plane.normal[i] if plane.normal[i] != 0.0 else 0.0)
        for i in range(3)
    )


def _plane_unit_normal_parts(
    plane: ConductingPlane,
) -> tuple[tuple[float, int], tuple[float, int], tuple[float, int]]:
    """Per-component ``(mantissa, exp)`` of the exact unit normal, preserving underflowing authored axes."""
    from .model import _plane_raw_normal
    from .observables import _scaled_ratio_parts, _vector_norm_parts
    rn = _plane_raw_normal(plane.normal)
    rn_comps = tuple(frexp(x) if x != 0.0 else (0.0, 0) for x in rn)
    rn_m, rn_e = _vector_norm_parts(rn_comps)
    out = []
    for i in range(3):
        if rn[i] == 0.0:
            out.append((0.0, 0))
        elif abs(plane.normal[i]) >= 1e-290:
            out.append(frexp(plane.normal[i]))
        else:
            rm_i, re_i = rn_comps[i]
            out.append(_scaled_ratio_parts((rm_i,), (rn_m,), re_i - rn_e))
    return tuple(out)


def _plane_height_parts(point: Vector3, plane: ConductingPlane) -> tuple[float, int]:
    """Return ``(mantissa, exp)`` for ``plane.signed_distance_m(point)`` without overflow/underflow."""
    from .observables import _fraction_to_parts, _scaled_ratio_parts, _vector_norm_parts
    h = plane.signed_distance_m(point)
    if isfinite(h) and abs(h) >= 1e-290:
        return frexp(h)
    from .model import _plane_exact_numerator, _plane_raw_normal
    rn = _plane_raw_normal(plane.normal)
    num_h = _plane_exact_numerator(point, plane.point_m, plane.normal, rn)
    if num_h == 0:
        return 0.0, 0
    hm, he = _fraction_to_parts(num_h)
    rn_m, rn_e = _vector_norm_parts(tuple(frexp(x) if x != 0.0 else (0.0, 0) for x in rn))
    return _scaled_ratio_parts((hm,), (rn_m,), he - rn_e)


def _mirror_displacement_parts(
    target: Vector3,
    source: Vector3,
    plane: ConductingPlane,
) -> tuple[tuple[tuple[float, int], tuple[float, int], tuple[float, int]], float, int]:
    """Return ``(comp_parts, dist_mantissa, dist_exp)`` for ``target - mirror(source, plane)``.

    Each component keeps its own binary exponent so tiny transverse offsets or
    tiny authored normal components never underflow before charge factors
    restore their scale, while avoiding world-coordinate translation loss,
    ``2*h`` overflow, and ``0*inf = nan``.
    """
    from .observables import (
        _component_diff_parts,
        _fraction_to_parts,
        _scaled_ratio_parts,
        _vector_norm_parts,
    )
    h = plane.signed_distance_m(source)
    if isfinite(h) and abs(h) >= 1e-290:
        hm, he = frexp(h)
        unit_n = _plane_unit_normal_parts(plane)
        comps = []
        exact_ctx = None
        for i in range(3):
            dm_i, de_i = _component_diff_parts(target[i], source[i])
            nm_i, ne_i = unit_n[i]
            if nm_i == 0.0:
                comps.append((dm_i, de_i))
                continue
            hm_i, he_i = _scaled_ratio_parts((2.0, hm, nm_i), (), he + ne_i)
            if dm_i == 0.0:
                comps.append((hm_i, he_i))
                continue
            if hm_i == 0.0:
                comps.append((dm_i, de_i))
                continue
            max_e = max(de_i, he_i)
            min_e = min(de_i, he_i)
            if max_e - min_e > 60:
                comps.append((dm_i, de_i) if de_i > he_i else (hm_i, he_i))
                continue
            s = ldexp(dm_i, de_i - max_e) + ldexp(hm_i, he_i - max_e)
            if abs(s) >= 0.01:
                sm, se = frexp(s)
                comps.append((sm, max_e + se))
                continue
            if exact_ctx is None:
                from fractions import Fraction
                from .model import _plane_exact_numerator, _plane_raw_normal
                rn = _plane_raw_normal(plane.normal)
                fn = tuple(Fraction(x) for x in rn)
                n_sq = sum(x * x for x in fn)
                num_h = _plane_exact_numerator(source, plane.point_m, plane.normal, rn)
                exact_ctx = (Fraction, fn, n_sq, num_h)
            Fraction_cls, fn, n_sq, num_h = exact_ctx
            v_fi = (Fraction_cls(target[i]) - Fraction_cls(source[i])) + (2 * num_h * fn[i]) / n_sq
            comps.append(_fraction_to_parts(v_fi))
        comp_tuple = tuple(comps)
        dm, de = _vector_norm_parts(comp_tuple)
        return comp_tuple, dm, de
    from fractions import Fraction
    from .model import _plane_exact_numerator, _plane_raw_normal
    rn = _plane_raw_normal(plane.normal)
    fn = tuple(Fraction(x) for x in rn)
    n_sq = sum(x * x for x in fn)
    num_h = _plane_exact_numerator(source, plane.point_m, plane.normal, rn)
    comp_tuple = tuple(
        _fraction_to_parts(
            (Fraction(target[i]) - Fraction(source[i])) + (2 * num_h * fn[i]) / n_sq
        )
        for i in range(3)
    )
    dm, de = _vector_norm_parts(comp_tuple)
    return comp_tuple, dm, de


def _sites(problem: ElectrostaticProblem) -> tuple[PointCharge, ...]:
    grouped: dict[Vector3, list[float]] = {}
    for source in problem.charges:
        grouped.setdefault(source.position_m, []).append(source.charge_c)
    return tuple(PointCharge(q, pos) for pos, qs in sorted(grouped.items())
                 if (q := fsum(qs)) != 0.0)


def _checked_site_index(index: int, count: int) -> int:
    """Grouped-site index policy of ``ForceEnergyResult.force_on_charge_n``.

    Negative, boolean and non-integer indices are rejected, so a legacy
    per-site force can never select a site and then fail to exclude it.
    """
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError('source index must be an integer')
    if not 0 <= index < count:
        raise IndexError('source index is outside the grouped active sites')
    return index


@dataclass(frozen=True, slots=True)
class GroundedPlaneSolution:
    plane: ConductingPlane
    sources: tuple[PointCharge, ...]
    permittivity_f_per_m: float
    conductor_results: tuple[ConductorResult, ...]

    @property
    def symmetry_axis(self) -> Vector3:
        # This is a natural visualization meridian, not a claim that an
        # arbitrary multi-source configuration has axial symmetry.
        return self.plane.normal

    @property
    def _k(self) -> float:
        return 1 / (4*pi*self.permittivity_f_per_m)

    @staticmethod
    def _needs_exact_sum(parts: list[tuple[float, int]], total: float) -> bool:
        nonzero = [(m, e) for m, e in parts if m != 0.0]
        if len(nonzero) < 2:
            return False
        if not isfinite(total) or total == 0.0:
            return True
        max_e = max(e for _, e in nonzero)
        return frexp(abs(total))[1] < max_e - 10

    def _plane_spread_digits(self, extra_points: tuple[Vector3, ...] = ()) -> int:
        exps = [frexp(abs(self.permittivity_f_per_m))[1]]
        from .model import _plane_raw_normal
        for v in (*_plane_raw_normal(self.plane.normal), *self.plane.point_m):
            if v != 0.0:
                exps.append(frexp(abs(v))[1])
        for q in self.sources:
            if q.charge_c != 0.0:
                exps.append(frexp(abs(q.charge_c))[1])
            for v in q.position_m:
                if v != 0.0:
                    exps.append(frexp(abs(v))[1])
        for pt in extra_points:
            for v in pt:
                if v != 0.0:
                    exps.append(frexp(abs(v))[1])
        if len(exps) < 2:
            return 0
        return ((max(exps) - min(exps)) * 302) // 1000 + 20

    def _exact_field_or_potential(self, point: Vector3, *, want_field: bool):
        """Trap-isolated adaptive-precision rationalized plane evaluation for cancellation/extremes."""
        from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
        from fractions import Fraction
        from .model import _plane_exact_numerator, _plane_raw_normal
        from .observables import _PI_120

        def _fdec(val: Fraction) -> Decimal:
            return Decimal(val.numerator) / Decimal(val.denominator)

        rn = _plane_raw_normal(self.plane.normal)
        fn = tuple(Fraction(x) for x in rn)
        n_sq = sum(x * x for x in fn)
        num_hp = _plane_exact_numerator(point, self.plane.point_m, self.plane.normal, rn)
        p_f = tuple(Fraction(x) for x in point)
        src_data = []
        for q in self.sources:
            num_hq = _plane_exact_numerator(q.position_m, self.plane.point_m, self.plane.normal, rn)
            q_f = tuple(Fraction(x) for x in q.position_m)
            v_dir = tuple(p_f[i] - q_f[i] for i in range(3))
            d_dir2 = sum(x * x for x in v_dir)
            d_img2 = d_dir2 + (4 * num_hp * num_hq) / n_sq
            src_data.append((q.charge_c, num_hq, v_dir, d_dir2, d_img2))

        max_digits = 4000
        spread = self._plane_spread_digits((point,))
        zero_check_digits = min(max_digits, max(240, 2 * spread + 120))
        digits = min(max_digits, max(80, spread + 60))
        while True:
            ctx = Context(prec=digits, rounding=ROUND_HALF_EVEN, Emax=999999999, Emin=-999999999, traps=[], flags=[])
            with localcontext(ctx):
                dk = Decimal(1) / (Decimal(4) * Decimal(_PI_120) * Decimal.from_float(self.permittivity_f_per_m))
                n_norm = _fdec(n_sq).sqrt()
                unit_n = tuple(_fdec(fn[i]) / n_norm for i in range(3))
                hp = _fdec(num_hp) / n_norm
                if not want_field:
                    total = Decimal(0)
                    max_term = Decimal(0)
                    for qc, num_hq, _, d_dir2, d_img2 in src_data:
                        dq = Decimal.from_float(qc)
                        hq = _fdec(num_hq) / n_norm
                        dd = _fdec(d_dir2).sqrt()
                        di = _fdec(d_img2).sqrt()
                        term = dk * dq * Decimal(4) * hp * hq / (dd * di * (dd + di))
                        total += term
                        if abs(term) > max_term:
                            max_term = abs(term)
                    if total == 0:
                        if max_term == 0 or digits >= zero_check_digits:
                            return 0.0
                        digits = min(max_digits, max(zero_check_digits, 2 * digits))
                        continue
                    lost = max(0, max_term.adjusted() - total.adjusted()) if max_term > 0 else 0
                    if digits - lost >= 40 or digits >= max_digits:
                        return float(total)
                    digits = min(max_digits, max(2 * digits, lost + 80))
                    continue

                totals = [Decimal(0), Decimal(0), Decimal(0)]
                max_terms = [Decimal(0), Decimal(0), Decimal(0)]
                for qc, num_hq, v_dir, d_dir2, d_img2 in src_data:
                    dq = Decimal.from_float(qc)
                    hq = _fdec(num_hq) / n_norm
                    dd = _fdec(d_dir2).sqrt()
                    di = _fdec(d_img2).sqrt()
                    rho = dd / di
                    ratio_fac = (Decimal(1) + rho + rho * rho) / (Decimal(1) + rho)
                    diff_inv3 = Decimal(4) * hp * hq * ratio_fac / (dd * dd * dd * di * di)
                    inv_di3 = Decimal(2) * hq / (di * di * di)
                    for i in range(3):
                        t1 = dk * dq * _fdec(v_dir[i]) * diff_inv3
                        t2 = -dk * dq * inv_di3 * unit_n[i]
                        totals[i] += t1 + t2
                        peak = max(abs(t1), abs(t2))
                        if peak > max_terms[i]:
                            max_terms[i] = peak
                need_more = False
                next_digits = digits
                for i in range(3):
                    if totals[i] == 0:
                        if max_terms[i] > 0 and digits < zero_check_digits:
                            need_more = True
                            next_digits = max(next_digits, max(zero_check_digits, 2 * digits))
                    elif max_terms[i] > 0:
                        lost = max(0, max_terms[i].adjusted() - totals[i].adjusted())
                        if digits - lost < 40 and digits < max_digits:
                            need_more = True
                            next_digits = max(next_digits, max(2 * digits, lost + 80))
                if not need_more or digits >= max_digits:
                    return tuple(float(x) for x in totals)
                digits = min(max_digits, next_digits)

    def _exact_force_on_charge_n(self, index: int, *, include_direct: bool) -> Vector3:
        """Trap-isolated adaptive-precision rationalized plane force evaluation."""
        from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
        from fractions import Fraction
        from .model import _plane_exact_numerator, _plane_raw_normal
        from .observables import _PI_120

        def _fdec(val: Fraction) -> Decimal:
            return Decimal(val.numerator) / Decimal(val.denominator)

        source = self.sources[index]
        rn = _plane_raw_normal(self.plane.normal)
        fn = tuple(Fraction(x) for x in rn)
        n_sq = sum(x * x for x in fn)
        num_hi = _plane_exact_numerator(source.position_m, self.plane.point_m, self.plane.normal, rn)
        pi_f = tuple(Fraction(x) for x in source.position_m)
        pair_data = []
        for j, other in enumerate(self.sources):
            num_hj = _plane_exact_numerator(other.position_m, self.plane.point_m, self.plane.normal, rn)
            pj_f = tuple(Fraction(x) for x in other.position_m)
            v_dir = tuple(pi_f[c] - pj_f[c] for c in range(3))
            d_dir2 = sum(x * x for x in v_dir)
            v_img = tuple(v_dir[c] + (2 * num_hj * fn[c]) / n_sq for c in range(3))
            d_img2 = d_dir2 + (4 * num_hi * num_hj) / n_sq
            pair_data.append((j, other.charge_c, num_hj, v_dir, d_dir2, v_img, d_img2))

        max_digits = 4000
        spread = self._plane_spread_digits()
        zero_check_digits = min(max_digits, max(240, 2 * spread + 120))
        digits = min(max_digits, max(80, spread + 60))
        while True:
            ctx = Context(prec=digits, rounding=ROUND_HALF_EVEN, Emax=999999999, Emin=-999999999, traps=[], flags=[])
            with localcontext(ctx):
                dk = Decimal(1) / (Decimal(4) * Decimal(_PI_120) * Decimal.from_float(self.permittivity_f_per_m))
                dqi = Decimal.from_float(source.charge_c)
                n_norm = _fdec(n_sq).sqrt()
                unit_n = tuple(_fdec(fn[c]) / n_norm for c in range(3))
                hi = _fdec(num_hi) / n_norm
                totals = [Decimal(0), Decimal(0), Decimal(0)]
                max_terms = [Decimal(0), Decimal(0), Decimal(0)]
                for j, qj_f, num_hj, v_dir, d_dir2, v_img, d_img2 in pair_data:
                    dqj = Decimal.from_float(qj_f)
                    di = _fdec(d_img2).sqrt()
                    di3 = di * di * di
                    if include_direct and j != index:
                        dd = _fdec(d_dir2).sqrt()
                        hj = _fdec(num_hj) / n_norm
                        rho = dd / di
                        ratio_fac = (Decimal(1) + rho + rho * rho) / (Decimal(1) + rho)
                        diff_inv3 = Decimal(4) * hi * hj * ratio_fac / (dd * dd * dd * di * di)
                        inv_di3 = Decimal(2) * hj / di3
                        for c in range(3):
                            t1 = dk * dqi * dqj * _fdec(v_dir[c]) * diff_inv3
                            t2 = -dk * dqi * dqj * inv_di3 * unit_n[c]
                            totals[c] += t1 + t2
                            peak = max(abs(t1), abs(t2))
                            if peak > max_terms[c]:
                                max_terms[c] = peak
                    else:
                        for c in range(3):
                            t = -dk * dqi * dqj * _fdec(v_img[c]) / di3
                            totals[c] += t
                            if abs(t) > max_terms[c]:
                                max_terms[c] = abs(t)
                need_more = False
                next_digits = digits
                for c in range(3):
                    if totals[c] == 0:
                        if max_terms[c] > 0 and digits < zero_check_digits:
                            need_more = True
                            next_digits = max(next_digits, max(zero_check_digits, 2 * digits))
                    elif max_terms[c] > 0:
                        lost = max(0, max_terms[c].adjusted() - totals[c].adjusted())
                        if digits - lost < 40 and digits < max_digits:
                            need_more = True
                            next_digits = max(next_digits, max(2 * digits, lost + 80))
                if not need_more or digits >= max_digits:
                    return tuple(float(x) for x in totals)
                digits = min(max_digits, next_digits)

    def _exact_interaction_energy_j(self) -> float:
        """Trap-isolated adaptive-precision rationalized plane interaction energy."""
        from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
        from fractions import Fraction
        from .model import _plane_exact_numerator, _plane_raw_normal
        from .observables import _PI_120

        def _fdec(val: Fraction) -> Decimal:
            return Decimal(val.numerator) / Decimal(val.denominator)

        rn = _plane_raw_normal(self.plane.normal)
        fn = tuple(Fraction(x) for x in rn)
        n_sq = sum(x * x for x in fn)
        num_h = [_plane_exact_numerator(q.position_m, self.plane.point_m, self.plane.normal, rn) for q in self.sources]
        pos_f = [tuple(Fraction(x) for x in q.position_m) for q in self.sources]
        pair_data = []
        for i in range(len(self.sources)):
            for j in range(i + 1, len(self.sources)):
                v_dir = tuple(pos_f[i][c] - pos_f[j][c] for c in range(3))
                d_dir2 = sum(x * x for x in v_dir)
                d_img2 = d_dir2 + (4 * num_h[i] * num_h[j]) / n_sq
                pair_data.append((i, j, d_dir2, d_img2))

        max_digits = 4000
        spread = self._plane_spread_digits()
        zero_check_digits = min(max_digits, max(240, 2 * spread + 120))
        digits = min(max_digits, max(80, spread + 60))
        while True:
            ctx = Context(prec=digits, rounding=ROUND_HALF_EVEN, Emax=999999999, Emin=-999999999, traps=[], flags=[])
            with localcontext(ctx):
                dk = Decimal(1) / (Decimal(4) * Decimal(_PI_120) * Decimal.from_float(self.permittivity_f_per_m))
                n_norm = _fdec(n_sq).sqrt()
                hs = [_fdec(x) / n_norm for x in num_h]
                dqs = [Decimal.from_float(q.charge_c) for q in self.sources]
                total = Decimal(0)
                max_term = Decimal(0)
                for i in range(len(self.sources)):
                    t = -Decimal("0.25") * dk * dqs[i] * dqs[i] / hs[i]
                    total += t
                    if abs(t) > max_term:
                        max_term = abs(t)
                for i, j, d_dir2, d_img2 in pair_data:
                    dd = _fdec(d_dir2).sqrt()
                    di = _fdec(d_img2).sqrt()
                    t = dk * dqs[i] * dqs[j] * Decimal(4) * hs[i] * hs[j] / (dd * di * (dd + di))
                    total += t
                    if abs(t) > max_term:
                        max_term = abs(t)
                if total == 0:
                    if max_term == 0 or digits >= zero_check_digits:
                        return 0.0
                    digits = min(max_digits, max(zero_check_digits, 2 * digits))
                    continue
                lost = max(0, max_term.adjusted() - total.adjusted()) if max_term > 0 else 0
                if digits - lost >= 40 or digits >= max_digits:
                    return float(total)
                digits = min(max_digits, max(2 * digits, lost + 80))

    def potential_at(self, point_m: Vector3) -> float:
        from .observables import (
            _displacement_parts,
            _scaled_ratio_parts,
            _scaled_ratio_product,
            _sum_scaled_ratios,
        )
        point = _position(point_m, "sample point")
        h_p_m, h_p_e = _plane_height_parts(point, self.plane)
        if h_p_m <= 0.0:
            return 0.0
        four_pi = 4.0 * pi
        eps = self.permittivity_f_per_m
        parts = []
        for q in self.sources:
            _, d_m, d_e = _displacement_parts(point, q.position_m)
            if d_m == 0.0:
                raise SingularField("potential is singular at an ideal point charge")
            _, dm, de = _mirror_displacement_parts(point, q.position_m, self.plane)
            h_q_m, h_q_e = _plane_height_parts(q.position_m, self.plane)
            rho = min(1.0, _scaled_ratio_product((d_m,), (dm,), d_e - de))
            parts.append(
                _scaled_ratio_parts(
                    (4.0, q.charge_c, h_p_m, h_q_m),
                    (four_pi, eps, d_m, dm, dm, 1.0 + rho),
                    h_p_e + h_q_e - d_e - 2 * de,
                )
            )
        total = _sum_scaled_ratios(parts)
        if self._needs_exact_sum(parts, total):
            return self._exact_field_or_potential(point, want_field=False)
        return total

    def electric_field_at(self, point_m: Vector3) -> Vector3:
        from .observables import (
            _displacement_parts,
            _scaled_ratio_parts,
            _scaled_ratio_product,
            _sum_scaled_ratios,
        )
        point = _position(point_m, "sample point")
        height = self.plane.signed_distance_m(point)
        if isclose(height, 0.0, abs_tol=1e-14):
            raise SingularField("electric field has distinct one-sided limits at plane")
        if height < 0:
            return (0.0, 0.0, 0.0)
        h_p_m, h_p_e = _plane_height_parts(point, self.plane)
        unit_n = _plane_unit_normal_parts(self.plane)
        four_pi = 4.0 * pi
        eps = self.permittivity_f_per_m
        terms = [[], [], []]
        for q in self.sources:
            comps_d, d_m, d_e = _displacement_parts(point, q.position_m)
            if d_m == 0.0:
                raise SingularField("electric field is singular at an ideal point charge")
            _, dm, de = _mirror_displacement_parts(point, q.position_m, self.plane)
            h_q_m, h_q_e = _plane_height_parts(q.position_m, self.plane)
            rho = min(1.0, _scaled_ratio_product((d_m,), (dm,), d_e - de))
            ratio_fac = (1.0 + rho + rho * rho) / (1.0 + rho)
            for i in range(3):
                mc, ec = comps_d[i]
                if mc != 0.0:
                    terms[i].append(
                        _scaled_ratio_parts(
                            (4.0, q.charge_c, h_p_m, h_q_m, ratio_fac, mc),
                            (four_pi, eps, d_m, d_m, d_m, dm, dm),
                            h_p_e + h_q_e + ec - 3 * d_e - 2 * de,
                        )
                    )
                nm, ne = unit_n[i]
                if nm != 0.0:
                    terms[i].append(
                        _scaled_ratio_parts(
                            (-2.0, q.charge_c, h_q_m, nm),
                            (four_pi, eps, dm, dm, dm),
                            h_q_e + ne - 3 * de,
                        )
                    )
        out = tuple(_sum_scaled_ratios(component) for component in terms)
        if any(self._needs_exact_sum(terms[i], out[i]) for i in range(3)):
            return self._exact_field_or_potential(point, want_field=True)
        return out

    def surface_charge_density_at(self, surface: str, point_m: Vector3) -> float:
        if surface != 'outer':
            raise ValueError("the ideal plane has one exposed conducting surface")
        point = _position(point_m, "surface point")
        if not isclose(self.plane.signed_distance_m(point), 0.0, abs_tol=1e-12):
            raise ValueError("surface-density point must lie on conducting plane")
        from fractions import Fraction
        from sys import float_info
        from .model import _plane_exact_numerator, _plane_raw_normal
        from .observables import _scaled_ratio_product
        min_normal = float_info.min * 1e4
        terms = []
        for q in self.sources:
            delta = _minus(point, q.position_m)
            r = hypot(*delta) if all(isfinite(d) for d in delta) else float("inf")
            hq = self.plane.signed_distance_m(q.position_m)
            if isfinite(r) and r >= min_normal and isfinite(hq) and abs(hq) >= min_normal:
                terms.append(_scaled_ratio_product((q.charge_c, hq), (2.0 * pi, r, r, r)))
                continue
            delta_f = tuple(Fraction(point[i]) - Fraction(q.position_m[i]) for i in range(3))
            d_max = max(abs(x) for x in delta_f)
            if d_max == 0:
                raise ZeroDivisionError("surface-density point coincides with point source")
            r_scaled = hypot(*(float(x / d_max) for x in delta_f))
            if isfinite(hq) and abs(hq) >= min_normal:
                exact_hq = Fraction(hq)
            else:
                rn = _plane_raw_normal(self.plane.normal)
                r_max = max(abs(Fraction(x)) for x in rn)
                r_norm = hypot(*(float(Fraction(x) / r_max) for x in rn))
                exact_hq = _plane_exact_numerator(q.position_m, self.plane.point_m, self.plane.normal, rn) / (r_max * Fraction(r_norm))
            exact_term = (
                Fraction(q.charge_c) * exact_hq
                / (Fraction(2.0 * pi) * (d_max ** 3) * Fraction(r_scaled) ** 3)
            )
            try:
                terms.append(float(exact_term))
            except OverflowError:
                terms.append(float("inf") if exact_term > 0 else float("-inf"))
        return -fsum(terms)

    def surface_charge_density(self, surface: str, radial_distance_m: float) -> float:
        """One-source radial sigma about the point's projection on the plane.

        General multi-charge profiles use ``surface_charge_density_at`` at
        a fully specified 3D position, without assuming axial symmetry.
        """
        if surface != 'outer':
            raise ValueError("the plane has one exposed conducting surface")
        if len(self.sources) != 1:
            raise UnsupportedProblem("radial sigma needs one source; use density_at")
        if not isfinite(radial_distance_m) or radial_distance_m < 0:
            raise ValueError("radial_distance_m must be finite and nonnegative")
        q = self.sources[0]
        h = self.plane.signed_distance_m(q.position_m)
        from fractions import Fraction
        from sys import float_info
        from .model import _plane_exact_numerator, _plane_raw_normal
        from .observables import _scaled_ratio_product
        min_normal = float_info.min * 1e4
        r_tot = hypot(h, radial_distance_m) if isfinite(h) else float("inf")
        if isfinite(h) and abs(h) >= min_normal and isfinite(r_tot) and r_tot >= min_normal:
            return -_scaled_ratio_product((q.charge_c, h), (2.0 * pi, r_tot, r_tot, r_tot))
        if isfinite(h) and abs(h) >= min_normal:
            exact_hq = Fraction(h)
        else:
            rn = _plane_raw_normal(self.plane.normal)
            r_max = max(abs(Fraction(x)) for x in rn)
            r_norm = hypot(*(float(Fraction(x) / r_max) for x in rn))
            exact_hq = _plane_exact_numerator(q.position_m, self.plane.point_m, self.plane.normal, rn) / (r_max * Fraction(r_norm))
        exact_r = Fraction(radial_distance_m)
        d_max = max(abs(exact_hq), exact_r)
        if d_max == 0:
            raise ZeroDivisionError("surface-density point coincides with point source")
        r_scaled = hypot(float(exact_hq / d_max), float(exact_r / d_max))
        exact_term = (
            Fraction(q.charge_c) * exact_hq
            / (Fraction(2.0 * pi) * (d_max ** 3) * Fraction(r_scaled) ** 3)
        )
        try:
            return -float(exact_term)
        except OverflowError:
            return float("-inf") if exact_term > 0 else float("inf")

    def force_on_charge_n(self, index: int, *, include_direct: bool = True) -> Vector3:
        """Net force on one *grouped* active point charge, excluding self-field."""
        from .observables import (
            _displacement_parts,
            _scaled_ratio_parts,
            _scaled_ratio_product,
            _sum_scaled_ratios,
        )
        source = self.sources[_checked_site_index(index, len(self.sources))]
        items = [[], [], []]
        four_pi = 4.0 * pi
        eps = self.permittivity_f_per_m
        unit_n = _plane_unit_normal_parts(self.plane) if include_direct else None
        h_i_m, h_i_e = _plane_height_parts(source.position_m, self.plane) if include_direct else (0.0, 0)
        for j, other in enumerate(self.sources):
            if include_direct and j != index:
                # Exact cancellation-free direct + image combination using r_img^2 - r_dir^2 = 4*h_i*h_j:
                #   r_dir/r_dir^3 - r_img/r_img^3 = r_dir * [4*h_i*h_j*(1+rho+rho^2)/(r_dir^3*r_img^2*(1+rho))] - 2*h_j*n/r_img^3
                comps_d, d_m, d_e = _displacement_parts(source.position_m, other.position_m)
                _, dm, de = _mirror_displacement_parts(source.position_m, other.position_m, self.plane)
                h_j_m, h_j_e = _plane_height_parts(other.position_m, self.plane)
                rho = min(1.0, _scaled_ratio_product((d_m,), (dm,), d_e - de))
                ratio_fac = (1.0 + rho + rho * rho) / (1.0 + rho)
                for i in range(3):
                    mc, ec = comps_d[i]
                    if mc != 0.0:
                        items[i].append(
                            _scaled_ratio_parts(
                                (4.0, source.charge_c, other.charge_c, h_i_m, h_j_m, ratio_fac, mc),
                                (four_pi, eps, d_m, d_m, d_m, dm, dm),
                                h_i_e + h_j_e + ec - 3 * d_e - 2 * de,
                            )
                        )
                    nm, ne = unit_n[i]
                    if nm != 0.0:
                        items[i].append(
                            _scaled_ratio_parts(
                                (-2.0, source.charge_c, other.charge_c, h_j_m, nm),
                                (four_pi, eps, dm, dm, dm),
                                h_j_e + ne - 3 * de,
                            )
                        )
                continue
            comps_r, dm, de = _mirror_displacement_parts(source.position_m, other.position_m, self.plane)
            for i in range(3):
                mc, ec = comps_r[i]
                if mc != 0.0:
                    items[i].append(
                        _scaled_ratio_parts(
                            (-source.charge_c, other.charge_c, mc),
                            (four_pi, eps, dm, dm, dm),
                            ec - 3 * de,
                        )
                    )
        out = tuple(_sum_scaled_ratios(x) for x in items)
        if any(self._needs_exact_sum(items[i], out[i]) for i in range(3)):
            return self._exact_force_on_charge_n(index, include_direct=include_direct)
        return out

    @property
    def induced_force_on_charge_n(self) -> Vector3:
        if len(self.sources) != 1:
            raise UnsupportedProblem("use force_on_charge_n(index) for multiple sources")
        return self.force_on_charge_n(0, include_direct=False)

    @property
    def interaction_energy_j(self) -> float:
        """External real-real Coulomb energy plus half the image interaction."""
        from .observables import (
            _displacement_parts,
            _scaled_ratio_parts,
            _scaled_ratio_product,
            _sum_scaled_ratios,
        )
        four_pi = 4.0 * pi
        eps = self.permittivity_f_per_m
        heights = [_plane_height_parts(q.position_m, self.plane) for q in self.sources]
        parts = []
        for i, q in enumerate(self.sources):
            h_i_m, h_i_e = heights[i]
            # Self-image energy: -0.5 * k * q^2 / (2 * h_i) = -0.25 * k * q^2 / h_i
            parts.append(
                _scaled_ratio_parts((-0.25, q.charge_c, q.charge_c), (four_pi, eps, h_i_m), -h_i_e)
            )
            for j in range(i + 1, len(self.sources)):
                other = self.sources[j]
                h_j_m, h_j_e = heights[j]
                _, d_m, d_e = _displacement_parts(q.position_m, other.position_m)
                _, dm, de = _mirror_displacement_parts(q.position_m, other.position_m, self.plane)
                rho = min(1.0, _scaled_ratio_product((d_m,), (dm,), d_e - de))
                # Combined direct + cross-image pair: k*q_i*q_j*(1/r_dir - 1/r_img) = k*q_i*q_j*4*h_i*h_j/(r_dir*r_img^2*(1+rho))
                parts.append(
                    _scaled_ratio_parts(
                        (4.0, q.charge_c, other.charge_c, h_i_m, h_j_m),
                        (four_pi, eps, d_m, dm, dm, 1.0 + rho),
                        h_i_e + h_j_e - d_e - 2 * de,
                    )
                )
        total = _sum_scaled_ratios(parts)
        if self._needs_exact_sum(parts, total):
            return self._exact_interaction_energy_j()
        return total


class GroundedPlaneBackend:
    """Image-charge solver for one infinite plane and one or more above-plane charges."""
    def supports(self, problem: ElectrostaticProblem) -> bool:
        if len(problem.conductors) != 1 or not isinstance(problem.conductors[0], ConductingPlane):
            return False
        plane = problem.conductors[0]
        return (isinstance(plane.constraint, Grounded) and bool(_sites(problem)) and len(_sites(problem)) <= 32 and
                all(plane.signed_distance_m(q.position_m) > 0 for q in problem.charges))

    def solve(self, problem: ElectrostaticProblem) -> GroundedPlaneSolution:
        if not self.supports(problem):
            raise UnsupportedProblem("grounded-plane images require exactly one plane and "
                                     "1–32 nonzero sites entirely in its positive-normal half-space")
        plane = problem.conductors[0]
        sources = _sites(problem)
        total = -fsum(q.charge_c for q in sources)
        results = (ConductorResult(plane.name, 0.0, total, 0.0, total),)
        return GroundedPlaneSolution(plane, sources, problem.permittivity_f_per_m, results)
