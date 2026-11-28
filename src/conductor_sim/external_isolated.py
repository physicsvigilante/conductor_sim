"""Exact exterior point charge next to a fixed-net-charge solid conducting sphere.

The grounded Kelvin image pair is supplemented by a central image monopole.
The latter enforces the *actual prescribed conductor net charge* (not a
prescribed voltage). This independent reference never calls FEMM.
"""
from __future__ import annotations

from math import isclose, isfinite, pi

from .external_sphere import (
    ExternalGroundedSphereBackend, ExternalGroundedSphereSolution, _diff, _field, _norm,
    _pair_force, _sphere_strictly_outside,
)
from .model import ElectrostaticProblem, FixedNetCharge, SolidSphere, Vector3, _position
from .solver import ConductorResult, SingularField, UnsupportedProblem
from .external_multi import ExternalMultiSphereBackend


def _isolated_exact_force_and_energy(
    sphere: SolidSphere,
    source_pos: Vector3,
    q: float,
    Q: float,
    permittivity_f_per_m: float,
) -> tuple[Vector3, float]:
    """Exact cancellation-free rationalized force and energy for one isolated sphere + point charge."""
    from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
    from fractions import Fraction
    from .observables import _PI_120

    def _fdec(val: Fraction) -> Decimal:
        return Decimal(val.numerator) / Decimal(val.denominator)

    fa = Fraction(sphere.radius_m)
    fa2 = fa * fa
    fa3 = fa2 * fa
    f_delta = tuple(Fraction(source_pos[i]) - Fraction(sphere.center_m[i]) for i in range(3))
    fd2 = sum(x * x for x in f_delta)
    fd2_minus_a2 = fd2 - fa2
    fq = Fraction(q)
    fQ = Fraction(Q)

    ctx = Context(prec=80, rounding=ROUND_HALF_EVEN, Emax=999999999, Emin=-999999999, traps=[], flags=[])
    with localcontext(ctx):
        dk = Decimal(1) / (Decimal(4) * Decimal(_PI_120) * Decimal.from_float(permittivity_f_per_m))
        d = _fdec(fd2).sqrt()
        d2_dec = _fdec(fd2)
        d4_dec = d2_dec * d2_dec
        gap_dec = _fdec(fd2_minus_a2)
        abs_q = _fdec(abs(fq))
        abs_Q = _fdec(abs(fQ))
        if fq * fQ > 0:
            # Rationalized force numerator: Q^2*d^2*P^2 - q^2*R^2 (exact Fraction, zero cancellation)
            p_f = fd2_minus_a2 * fd2_minus_a2
            r_f = fa3 * (2 * fd2 - fa2)
            n_f = (fQ * fQ) * fd2 * (p_f * p_f) - (fq * fq) * (r_f * r_f)
            if n_f == 0:
                force = (0.0, 0.0, 0.0)
            else:
                denom_f = d4_dec * _fdec(p_f) * (abs_Q * d * _fdec(p_f) + abs_q * _fdec(r_f))
                scale_f = dk * abs_q * _fdec(n_f) / denom_f
                force = tuple(float(scale_f * _fdec(f_delta[i])) for i in range(3))
            # Rationalized energy numerator: Q^2*d^2*(2*(d^2-a^2))^2 - q^2*a^6 (exact Fraction)
            p_u = 2 * fd2_minus_a2
            n_u = (fQ * fQ) * fd2 * (p_u * p_u) - (fq * fq) * (fa3 * fa3)
            if n_u == 0:
                energy = 0.0
            else:
                denom_u = Decimal(2) * d2_dec * gap_dec * (abs_Q * d * _fdec(p_u) + abs_q * _fdec(fa3))
                energy = float(dk * abs_q * _fdec(n_u) / denom_u)
        else:
            dq = _fdec(fq)
            dQ = _fdec(fQ)
            p_f_dec = gap_dec * gap_dec
            r_f_dec = _fdec(fa3 * (2 * fd2 - fa2))
            scale_f = dk * dq * (dQ * d * p_f_dec - dq * r_f_dec) / (d4_dec * p_f_dec)
            force = tuple(float(scale_f * _fdec(f_delta[i])) for i in range(3))
            energy = float(
                dk * dq * (Decimal(2) * dQ * d * gap_dec - dq * _fdec(fa3))
                / (Decimal(2) * d2_dec * gap_dec)
            )
        return force, energy


def _isolated_conductor_potential_v(
    sphere: SolidSphere,
    source_pos: Vector3,
    q: float,
    Q: float,
    permittivity_f_per_m: float,
) -> float:
    """Overflow/underflow- and cancellation-safe potential (Q/a + q/d) / (4*pi*eps)."""
    from .observables import _scaled_ratio_parts, _sum_scaled_ratios
    d = _norm(_diff(source_pos, sphere.center_m))
    four_pi = 4.0 * pi
    p_Q = _scaled_ratio_parts((Q,), (four_pi, permittivity_f_per_m, sphere.radius_m))
    p_q = _scaled_ratio_parts((q,), (four_pi, permittivity_f_per_m, d))
    val = _sum_scaled_ratios([p_Q, p_q])
    if p_Q[0] != 0.0 and p_q[0] != 0.0:
        from math import frexp
        max_e = max(p_Q[1], p_q[1])
        if not isfinite(val) or val == 0.0 or frexp(abs(val))[1] < max_e - 10:
            from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
            from fractions import Fraction
            from .observables import _PI_120
            fa = Fraction(sphere.radius_m)
            f_delta = tuple(Fraction(source_pos[i]) - Fraction(sphere.center_m[i]) for i in range(3))
            fd2 = sum(x * x for x in f_delta)
            fq = Fraction(q)
            fQ = Fraction(Q)
            ctx = Context(prec=80, rounding=ROUND_HALF_EVEN, Emax=999999999, Emin=-999999999, traps=[], flags=[])
            with localcontext(ctx):
                dk = Decimal(1) / (Decimal(4) * Decimal(_PI_120) * Decimal.from_float(permittivity_f_per_m))
                d_dec = (Decimal(fd2.numerator) / Decimal(fd2.denominator)).sqrt()
                a_dec = Decimal.from_float(sphere.radius_m)
                if fQ * fq < 0:
                    # Q*d + q*a = sgn(Q) * (Q^2*d^2 - q^2*a^2) / (|Q|*d + |q|*a)
                    num = fQ * fQ * fd2 - fq * fq * fa * fa
                    if num == 0:
                        return 0.0
                    sgn = Decimal(1) if fQ > 0 else Decimal(-1)
                    num_d = Decimal(num.numerator) / Decimal(num.denominator)
                    den_d = (Decimal(abs(fQ).numerator) / Decimal(abs(fQ).denominator)) * d_dec + (
                        Decimal(abs(fq).numerator) / Decimal(abs(fq).denominator)
                    ) * a_dec
                    return float(dk * sgn * num_d / (a_dec * d_dec * den_d))
                return float(dk * (Decimal.from_float(Q) / a_dec + Decimal.from_float(q) / d_dec))
    return val


class ExternalIsolatedSphereSolution(ExternalGroundedSphereSolution):
    """Exact Kelvin-image field for a fixed-net-charge (isolated) solid sphere.

    For radius a, source q at distance d>a, and prescribed sphere charge Q:
      q_image = -q*a/d at r_image = a**2/d,
      q_center = Q-q_image = Q+q*a/d at the sphere centre.

    The grounded Kelvin pair vanishes on r=a, while the central monopole
    sets the equipotential to k*q_center/a. Their image charges sum to Q.
    The centre monopole represents both the prescribed net charge and the
    charge redistributed to maintain a constant potential; it is NOT an
    additional authored physical point charge.
    """

    @property
    def central_image_charge_c(self) -> float:
        return self.conductor_results[0].net_charge_c - self.image_charge_c

    @property
    def conductor_potential_v(self) -> float:
        return _isolated_conductor_potential_v(
            self.sphere,
            self.source.position_m,
            self.source.charge_c,
            self.conductor_results[0].net_charge_c,
            self.permittivity_f_per_m,
        )

    def _isolated_exact_eval(self, point: Vector3, *, want_field: bool):
        """Trap-isolated adaptive-precision grounded Kelvin + central monopole evaluation."""
        from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
        from fractions import Fraction
        from .observables import _PI_120

        def _fdec(val: Fraction) -> Decimal:
            return Decimal(val.numerator) / Decimal(val.denominator)

        a = Fraction(self.sphere.radius_m)
        a2 = a * a
        c = tuple(Fraction(x) for x in self.sphere.center_m)
        p = tuple(Fraction(x) for x in point)
        s = tuple(Fraction(x) for x in self.source.position_m)
        r_src = tuple(s[i] - c[i] for i in range(3))
        r_p = tuple(p[i] - c[i] for i in range(3))
        d2 = sum(x * x for x in r_src)
        rp2 = sum(x * x for x in r_p)
        v_dir = tuple(p[i] - s[i] for i in range(3))
        dist2 = sum(x * x for x in v_dir)
        v_ref = tuple(r_p[i] - (a2 / d2) * r_src[i] for i in range(3))
        r_img2 = sum(x * x for x in v_ref)
        Q_val = self.conductor_results[0].net_charge_c
        max_digits = 4000
        digits = 120
        while True:
            ctx = Context(prec=digits, rounding=ROUND_HALF_EVEN, Emax=999999999, Emin=-999999999, traps=[], flags=[])
            with localcontext(ctx):
                dk = Decimal(1) / (Decimal(4) * Decimal(_PI_120) * Decimal.from_float(self.permittivity_f_per_m))
                dq = Decimal.from_float(self.source.charge_c)
                dQ = Decimal.from_float(Q_val)
                da = Decimal.from_float(self.sphere.radius_m)
                d_src = _fdec(d2).sqrt()
                r_p_norm = _fdec(rp2).sqrt()
                dist = _fdec(dist2).sqrt()
                r_img = _fdec(r_img2).sqrt()
                dq_img = -dq * da / d_src
                dq_center = dQ - dq_img
                if not want_field:
                    v_gnd = dk * dq * _fdec((d2 - a2) * (rp2 - a2)) / (d_src * dist * r_img * (d_src * r_img + da * dist))
                    v_cen = dk * dq_center / r_p_norm
                    total = v_gnd + v_cen
                    peak = max(abs(v_gnd), abs(v_cen))
                    if total == 0:
                        if peak == 0 or digits >= 960:
                            return 0.0
                        digits = min(max_digits, 2 * digits)
                        continue
                    lost = max(0, peak.adjusted() - total.adjusted()) if peak > 0 else 0
                    if digits - lost >= 40 or digits >= max_digits:
                        return float(total)
                    digits = min(max_digits, max(2 * digits, lost + 80))
                    continue

                dist3 = dist * dist * dist
                r_img3 = r_img * r_img * r_img
                rp3 = r_p_norm * r_p_norm * r_p_norm
                totals = []
                need_more = False
                next_digits = digits
                for i in range(3):
                    t_dir = dk * dq * _fdec(v_dir[i]) / dist3
                    t_img = dk * dq_img * _fdec(v_ref[i]) / r_img3
                    t_cen = dk * dq_center * _fdec(r_p[i]) / rp3
                    tot = t_dir + t_img + t_cen
                    totals.append(tot)
                    peak = max(abs(t_dir), abs(t_img), abs(t_cen))
                    if tot == 0:
                        if peak > 0 and digits < 960:
                            need_more = True
                            next_digits = max(next_digits, 2 * digits)
                    elif peak > 0:
                        lost = max(0, peak.adjusted() - tot.adjusted())
                        if digits - lost < 40 and digits < max_digits:
                            need_more = True
                            next_digits = max(next_digits, max(2 * digits, lost + 80))
                if not need_more or digits >= max_digits:
                    return tuple(float(x) for x in totals)
                digits = min(max_digits, next_digits)

    def potential_at(self, point_m: Vector3) -> float:
        from .observables import _scaled_ratio_parts, _sum_scaled_ratios
        point = _position(point_m, 'sample point')
        r = _norm(_diff(point, self.sphere.center_m))
        a = self.sphere.radius_m
        if r <= a or isclose(r, a, rel_tol=1e-12):
            from fractions import Fraction
            fa = Fraction(a)
            rp2_exact = sum((Fraction(point[i]) - Fraction(self.sphere.center_m[i])) ** 2 for i in range(3))
            if rp2_exact <= fa * fa:
                return self.conductor_potential_v
            if r <= a:
                if point == self.source.position_m:
                    raise SingularField('point-charge potential is singular at the external source')
                return self._isolated_exact_eval(point, want_field=False)
        base = super().potential_at(point)
        if not isfinite(r):
            return self._isolated_exact_eval(point, want_field=False)
        four_pi = 4.0 * pi
        eps = self.permittivity_f_per_m
        Q = self.conductor_results[0].net_charge_c
        c_parts = [
            _scaled_ratio_parts((Q,), (four_pi, eps, r)),
            _scaled_ratio_parts((self.source.charge_c, self.sphere.radius_m), (four_pi, eps, self.source_distance_m, r)),
        ]
        central = _sum_scaled_ratios(c_parts)
        total = base + central
        if (
            not (isfinite(base) and isfinite(central) and isfinite(total))
            or (base != 0.0 and central != 0.0 and abs(total) < 0.01 * max(abs(base), abs(central)))
            or (c_parts[0][0] != 0.0 and c_parts[1][0] != 0.0 and (central == 0.0 or abs(central) < 0.01 * max(abs(_sum_scaled_ratios([c_parts[0]])), abs(_sum_scaled_ratios([c_parts[1]])))))
        ):
            return self._isolated_exact_eval(point, want_field=False)
        return total

    def electric_field_at(self, point_m: Vector3) -> Vector3:
        from .observables import _scaled_ratio_parts, _sum_scaled_ratios
        point = _position(point_m, 'sample point')
        base = super().electric_field_at(point)
        delta = _diff(point, self.sphere.center_m)
        r = _norm(delta)
        if r < self.sphere.radius_m:
            return base
        if not isfinite(r):
            return self._isolated_exact_eval(point, want_field=True)
        four_pi = 4.0 * pi
        eps = self.permittivity_f_per_m
        Q = self.conductor_results[0].net_charge_c
        q = self.source.charge_c
        a = self.sphere.radius_m
        d_src = self.source_distance_m
        out = []
        for i in range(3):
            c_Q = _scaled_ratio_parts((Q, delta[i]), (four_pi, eps, r, r, r))
            c_ind = _scaled_ratio_parts((q, a, delta[i]), (four_pi, eps, d_src, r, r, r))
            central_i = _sum_scaled_ratios([c_Q, c_ind])
            tot_i = base[i] + central_i
            if (
                not (isfinite(base[i]) and isfinite(central_i) and isfinite(tot_i))
                or (base[i] != 0.0 and central_i != 0.0 and abs(tot_i) < 0.01 * max(abs(base[i]), abs(central_i)))
                or (c_Q[0] != 0.0 and c_ind[0] != 0.0 and (central_i == 0.0 or abs(central_i) < 0.01 * max(abs(_sum_scaled_ratios([c_Q])), abs(_sum_scaled_ratios([c_ind])))))
            ):
                return self._isolated_exact_eval(point, want_field=True)
            out.append(tot_i)
        return tuple(out)

    def _isolated_density_from_chord_sq(self, chord_sq_fn) -> float:
        """Cancellation-free isolated-sphere surface charge density given exact chord^2(ctx)."""
        from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
        from fractions import Fraction
        from .observables import _PI_120

        def _fdec(val: Fraction) -> Decimal:
            return Decimal(val.numerator) / Decimal(val.denominator)

        fa = Fraction(self.sphere.radius_m)
        f_delta = tuple(Fraction(self.source.position_m[i]) - Fraction(self.sphere.center_m[i]) for i in range(3))
        fd2 = sum(x * x for x in f_delta)
        fd2_minus_a2 = fd2 - fa * fa
        Q_val = self.conductor_results[0].net_charge_c
        max_digits = 5000
        digits = 120
        while True:
            ctx = Context(prec=digits, rounding=ROUND_HALF_EVEN, Emax=999999999, Emin=-999999999, traps=[], flags=[])
            with localcontext(ctx):
                four_pi = Decimal(4) * Decimal(_PI_120)
                da = Decimal.from_float(self.sphere.radius_m)
                dq = Decimal.from_float(self.source.charge_c)
                dQ = Decimal.from_float(Q_val)
                d = _fdec(fd2).sqrt()
                gap_sq = _fdec(fd2_minus_a2)
                L0 = gap_sq / (d + da)
                c2 = chord_sq_fn(_fdec, d)
                t = da * d * c2
                L02 = L0 * L0
                L2 = L02 + t
                L = L2.sqrt()
                L3 = L2 * L
                # Exact cancellation-free numerator of L^6 - d^2*(d^2 - a^2)^2:
                #   -a*L0^2*(3d - a)*(2d^2 - a*d + a^2) + t*(3*L0^4 + 3*L0^2*t + t^2)
                neg_part = -da * L02 * (Decimal(3) * d - da) * (Decimal(2) * d * d - da * d + da * da)
                pos_part = t * (Decimal(3) * L02 * L02 + Decimal(3) * L02 * t + t * t)
                num = neg_part + pos_part
                denom = four_pi * da * d * L3 * (L3 + d * gap_sq)
                sigma_ind = dq * num / denom
                sigma_Q = dQ / (four_pi * da * da)
                tot = sigma_ind + sigma_Q
                # Check precision loss in both (neg_part + pos_part) and (sigma_ind + sigma_Q)
                lost = 0
                if neg_part != 0 and pos_part != 0:
                    p_n = max(abs(neg_part), abs(pos_part))
                    lost = max(lost, (p_n.adjusted() - num.adjusted()) if num != 0 else digits)
                if sigma_ind != 0 and sigma_Q != 0:
                    p_s = max(abs(sigma_ind), abs(sigma_Q))
                    lost = max(lost, (p_s.adjusted() - tot.adjusted()) if tot != 0 else digits)
                if (tot != 0 and digits - lost >= 40) or digits >= max_digits or (tot == 0 and digits >= 480):
                    return float(tot)
                digits = min(max_digits, max(2 * digits, lost + 80))

    def _isolated_density_from_chord(self, chord: float) -> float:
        from decimal import Decimal
        dc = Decimal.from_float(chord)
        return self._isolated_density_from_chord_sq(lambda _fdec, _d: dc * dc)

    def _isolated_density_at_point(self, point: Vector3) -> float:
        """Cancellation-free surface density at an authored 3D surface point (exact Cartesian chord^2)."""
        from decimal import Decimal
        from fractions import Fraction
        c = tuple(Fraction(x) for x in self.sphere.center_m)
        f_rp = tuple(Fraction(point[i]) - c[i] for i in range(3))
        f_rs = tuple(Fraction(self.source.position_m[i]) - c[i] for i in range(3))
        f_rp2 = sum(x * x for x in f_rp)
        f_d2 = sum(x * x for x in f_rs)
        f_dot = sum(f_rp[i] * f_rs[i] for i in range(3))
        f_cross2 = f_rp2 * f_d2 - f_dot * f_dot

        def _chord_sq(_fdec, d_dec):
            if f_dot == 0:
                return Decimal(2)
            rp_norm = _fdec(f_rp2).sqrt()
            prod = rp_norm * d_dec
            if f_dot > 0:
                # 2*(1 - cos) = 2*sin^2/(1 + cos) = 2*|rp x rs|^2 / (|rp|*d*(|rp|*d + rp.rs))
                return Decimal(2) * _fdec(f_cross2) / (prod * (prod + _fdec(f_dot)))
            return Decimal(2) + Decimal(2) * _fdec(-f_dot) / prod

        return self._isolated_density_from_chord_sq(_chord_sq)

    def surface_charge_density(self, surface: str, theta_rad: float) -> float:
        # super() validates face and angle before adding the constant term.
        from math import frexp, sin
        from .observables import _scaled_ratio_parts, _sum_scaled_ratios
        grounded_density = super().surface_charge_density(surface, theta_rad)
        a = self.sphere.radius_m
        Q = self.conductor_results[0].net_charge_c
        central = _sum_scaled_ratios([
            _scaled_ratio_parts((Q,), (4.0 * pi, a, a)),
            _scaled_ratio_parts((self.source.charge_c,), (4.0 * pi, self.source_distance_m, a)),
        ])
        total = grounded_density + central
        if (
            not (isfinite(grounded_density) and isfinite(central) and isfinite(total))
            or (grounded_density != 0.0 and central != 0.0 and (
                total == 0.0 or frexp(abs(total))[1] < max(frexp(abs(grounded_density))[1], frexp(abs(central))[1]) - 10
            ))
        ):
            return self._isolated_density_from_chord(2.0 * sin(theta_rad / 2.0))
        return total

    def surface_charge_density_at(self, surface: str, point_m: Vector3) -> float:
        # The grounded part is evaluated at the intended surface point; the
        # central image adds a uniform density.
        from math import frexp
        from .observables import _scaled_ratio_parts, _sum_scaled_ratios
        grounded_density = super().surface_charge_density_at(surface, point_m)
        a = self.sphere.radius_m
        Q = self.conductor_results[0].net_charge_c
        central = _sum_scaled_ratios([
            _scaled_ratio_parts((Q,), (4.0 * pi, a, a)),
            _scaled_ratio_parts((self.source.charge_c,), (4.0 * pi, self.source_distance_m, a)),
        ])
        total = grounded_density + central
        if (
            not (isfinite(grounded_density) and isfinite(central) and isfinite(total))
            or (grounded_density != 0.0 and central != 0.0 and (
                total == 0.0 or frexp(abs(total))[1] < max(frexp(abs(grounded_density))[1], frexp(abs(central))[1]) - 10
            ))
        ):
            return self._isolated_density_at_point(_position(point_m, 'surface point'))
        return total

    @property
    def induced_force_on_charge_n(self) -> Vector3:
        """Total sphere-on-point force, including the original net charge Q.

        The M7 property name is kept for API compatibility. With Q!=0 this
        is not merely an induced/polarization force: it also includes the
        Coulomb contribution from the sphere's prescribed total charge.
        """
        from fractions import Fraction
        from math import frexp
        from .observables import _scaled_ratio_parts, _scaled_ratio_product, _sum_scaled_ratios
        q = self.source.charge_c
        Q = self.conductor_results[0].net_charge_c
        a = self.sphere.radius_m
        delta = _diff(self.source.position_m, self.sphere.center_m)
        d = self.source_distance_m
        if d - a < 0.01 * a:
            f_delta = tuple(Fraction(self.source.position_m[i]) - Fraction(self.sphere.center_m[i]) for i in range(3))
            f_d2 = sum(x * x for x in f_delta)
            f_a2 = Fraction(a) * Fraction(a)
            u_denom = float((f_d2 - f_a2) / f_d2)
        else:
            u_denom = ((d - a) / d) * (1.0 + a / d)
        ratio_sq = _scaled_ratio_product((a, a), (d, d))
        four_pi = 4.0 * pi
        eps = self.permittivity_f_per_m
        out = []
        for i in range(3):
            polar = _scaled_ratio_parts(
                (-q, q, a, a, a, 2.0 - ratio_sq, delta[i]),
                (four_pi, eps, d, d, d, d, d, d, u_denom, u_denom),
            )
            mono = _scaled_ratio_parts((q, Q, delta[i]), (four_pi, eps, d, d, d))
            tot = _sum_scaled_ratios([polar, mono])
            if polar[0] != 0.0 and mono[0] != 0.0:
                max_e = max(polar[1], mono[1])
                if not isfinite(tot) or tot == 0.0 or frexp(abs(tot))[1] < max_e - 10:
                    exact_f, _ = _isolated_exact_force_and_energy(
                        self.sphere, self.source.position_m, q, Q, eps
                    )
                    return exact_f
            out.append(tot)
        return tuple(out)

    @property
    def interaction_energy_j(self) -> float:
        """Source--sphere interaction energy relative to infinite separation.

        The constant self-energy Q^2/(2*C_sphere) of the isolated sphere
        is excluded: it does not depend on source position and has no
        effect on the force. Polarization terms include the physical 1/2.
        """
        from fractions import Fraction
        from math import frexp
        from .observables import _scaled_ratio_parts, _sum_scaled_ratios
        q = self.source.charge_c
        Q = self.conductor_results[0].net_charge_c
        a, d = self.sphere.radius_m, self.source_distance_m
        if d - a < 0.01 * a:
            f_delta = tuple(Fraction(self.source.position_m[i]) - Fraction(self.sphere.center_m[i]) for i in range(3))
            f_d2 = sum(x * x for x in f_delta)
            f_a2 = Fraction(a) * Fraction(a)
            u_denom = float((f_d2 - f_a2) / f_d2)
        else:
            u_denom = ((d - a) / d) * (1.0 + a / d)
        four_pi = 4.0 * pi
        eps = self.permittivity_f_per_m
        polar = _scaled_ratio_parts((-0.5, q, q, a, a, a), (four_pi, eps, d, d, d, d, u_denom))
        mono = _scaled_ratio_parts((q, Q), (four_pi, eps, d))
        tot = _sum_scaled_ratios([polar, mono])
        if polar[0] != 0.0 and mono[0] != 0.0:
            max_e = max(polar[1], mono[1])
            if not isfinite(tot) or tot == 0.0 or frexp(abs(tot))[1] < max_e - 10:
                _, exact_u = _isolated_exact_force_and_energy(
                    self.sphere, self.source.position_m, q, Q, eps
                )
                return exact_u
        return tot


class ExternalIsolatedSphereBackend:
    """Analytical backend for one nonzero external charge, one isolated sphere."""

    def supports(self, problem: ElectrostaticProblem) -> bool:
        if len(problem.conductors) != 1 or len(problem.charges) != 1:
            return False
        sphere, source = problem.conductors[0], problem.charges[0]
        if not (isinstance(sphere, SolidSphere)
                and isinstance(sphere.constraint, FixedNetCharge)
                and source.charge_c != 0):
            return False
        return (
            _sphere_strictly_outside(source.position_m, sphere.center_m, sphere.radius_m)
            and isfinite(sphere.constraint.charge_c + source.charge_c)
        )

    def solve(self, problem: ElectrostaticProblem) -> ExternalIsolatedSphereSolution:
        if not self.supports(problem):
            raise UnsupportedProblem(
                'isolated exterior-sphere solution requires exactly one nonzero '
                'point charge strictly outside one FixedNetCharge SolidSphere; '
                'grounded, touching, interior, multiple-source and general '
                'conductor geometries are not supported by this backend'
            )
        sphere, source = problem.conductors[0], problem.charges[0]
        Q = sphere.constraint.charge_c
        potential = _isolated_conductor_potential_v(
            sphere, source.position_m, source.charge_c, Q, problem.permittivity_f_per_m
        )
        result = ConductorResult(sphere.name, potential, Q, 0.0, Q)
        return ExternalIsolatedSphereSolution(
            sphere, source, problem.permittivity_f_per_m, (result,)
        )


class ExternalSphereBackend:
    """Common author-facing analytical dispatch for grounded OR isolated spheres.

    Delegates only to exact independently derived reference backends. Never
    substitutes an analytical answer for a failed numerical FEMM run.
    """

    def supports(self, problem: ElectrostaticProblem) -> bool:
        return (ExternalGroundedSphereBackend().supports(problem)
                or ExternalIsolatedSphereBackend().supports(problem)
                or ExternalMultiSphereBackend().supports(problem))

    def solve(self, problem: ElectrostaticProblem):
        if ExternalGroundedSphereBackend().supports(problem):
            return ExternalGroundedSphereBackend().solve(problem)
        if ExternalIsolatedSphereBackend().supports(problem):
            return ExternalIsolatedSphereBackend().solve(problem)
        if ExternalMultiSphereBackend().supports(problem):
            return ExternalMultiSphereBackend().solve(problem)
        raise UnsupportedProblem('external sphere: expected one grounded or fixed-net-charge '
                                 'SolidSphere with one or more external charges')
