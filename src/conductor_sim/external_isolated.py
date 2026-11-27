"""Exact exterior point charge next to a fixed-net-charge solid conducting sphere.

The grounded Kelvin image pair is supplemented by a central image monopole.
The latter enforces the *actual prescribed conductor net charge* (not a
prescribed voltage). This independent reference never calls FEMM.
"""
from __future__ import annotations

from math import isfinite, pi

from .external_sphere import (
    ExternalGroundedSphereBackend, ExternalGroundedSphereSolution, _diff, _field, _norm,
    _pair_force, _sphere_strictly_outside,
)
from .model import ElectrostaticProblem, FixedNetCharge, SolidSphere, Vector3, _position
from .solver import ConductorResult, UnsupportedProblem
from .external_multi import ExternalMultiSphereBackend


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
        from .observables import _scaled_ratio_product
        return _scaled_ratio_product(
            (self.central_image_charge_c,),
            (4.0 * pi, self.permittivity_f_per_m, self.sphere.radius_m),
        )

    def _isolated_exact_eval(self, point: Vector3, *, want_field: bool):
        """Trap-isolated high-precision grounded Kelvin + central monopole evaluation."""
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
                        if peak == 0 or digits >= 480:
                            return 0.0
                        digits *= 2
                        continue
                    lost = max(0, peak.adjusted() - total.adjusted()) if peak > 0 else 0
                    if digits - lost >= 40 or digits >= 960:
                        return float(total)
                    digits = min(960, max(2 * digits, lost + 60))
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
                        if peak > 0 and digits < 480:
                            need_more = True
                            next_digits = max(next_digits, 2 * digits)
                    elif peak > 0:
                        lost = max(0, peak.adjusted() - tot.adjusted())
                        if digits - lost < 40 and digits < 960:
                            need_more = True
                            next_digits = max(next_digits, max(2 * digits, lost + 60))
                if not need_more or digits >= 960:
                    return tuple(float(x) for x in totals)
                digits = min(960, next_digits)

    def potential_at(self, point_m: Vector3) -> float:
        from .observables import _scaled_ratio_product
        point = _position(point_m, 'sample point')
        r = _norm(_diff(point, self.sphere.center_m))
        if r <= self.sphere.radius_m:
            return self.conductor_potential_v
        base = super().potential_at(point)
        if not isfinite(r):
            return self._isolated_exact_eval(point, want_field=False)
        central = _scaled_ratio_product(
            (self.central_image_charge_c,),
            (4.0 * pi, self.permittivity_f_per_m, r),
        )
        total = base + central
        if (
            not (isfinite(base) and isfinite(central) and isfinite(total))
            or (base != 0.0 and central != 0.0 and abs(total) < 0.01 * max(abs(base), abs(central)))
        ):
            return self._isolated_exact_eval(point, want_field=False)
        return total

    def electric_field_at(self, point_m: Vector3) -> Vector3:
        from .observables import _scaled_ratio_product
        point = _position(point_m, 'sample point')
        base = super().electric_field_at(point)
        delta = _diff(point, self.sphere.center_m)
        r = _norm(delta)
        if r < self.sphere.radius_m:
            return base
        if not isfinite(r):
            return self._isolated_exact_eval(point, want_field=True)
        central = tuple(
            _scaled_ratio_product(
                (self.central_image_charge_c, delta[i]),
                (4.0 * pi, self.permittivity_f_per_m, r, r, r),
            )
            for i in range(3)
        )
        out = tuple(base[i] + central[i] for i in range(3))
        if any(
            not (isfinite(base[i]) and isfinite(central[i]) and isfinite(out[i]))
            or (base[i] != 0.0 and central[i] != 0.0 and abs(out[i]) < 0.01 * max(abs(base[i]), abs(central[i])))
            for i in range(3)
        ):
            return self._isolated_exact_eval(point, want_field=True)
        return out

    def surface_charge_density(self, surface: str, theta_rad: float) -> float:
        # super() validates face and angle before adding the constant term.
        from .observables import _scaled_ratio_product
        grounded_density = super().surface_charge_density(surface, theta_rad)
        a = self.sphere.radius_m
        return grounded_density + _scaled_ratio_product((self.central_image_charge_c,), (4.0 * pi, a, a))

    def surface_charge_density_at(self, surface: str, point_m: Vector3) -> float:
        # The grounded part is evaluated at the intended surface point; the
        # central image adds a uniform density.
        from .observables import _scaled_ratio_product
        grounded_density = super().surface_charge_density_at(surface, point_m)
        a = self.sphere.radius_m
        return grounded_density + _scaled_ratio_product((self.central_image_charge_c,), (4.0 * pi, a, a))

    @property
    def induced_force_on_charge_n(self) -> Vector3:
        """Total sphere-on-point force, including the original net charge Q.

        The M7 property name is kept for API compatibility. With Q!=0 this
        is not merely an induced/polarization force: it also includes the
        Coulomb contribution from the sphere's prescribed total charge.
        """
        from fractions import Fraction
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
            out.append(_sum_scaled_ratios([polar, mono]))
        return tuple(out)

    @property
    def interaction_energy_j(self) -> float:
        """Source--sphere interaction energy relative to infinite separation.

        The constant self-energy Q^2/(2*C_sphere) of the isolated sphere
        is excluded: it does not depend on source position and has no
        effect on the force. Polarization terms include the physical 1/2.
        """
        from fractions import Fraction
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
        return _sum_scaled_ratios([polar, mono])


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
        from .observables import _scaled_ratio_product
        sphere, source = problem.conductors[0], problem.charges[0]
        Q = sphere.constraint.charge_c
        d = _norm(_diff(source.position_m, sphere.center_m))
        q_image = _scaled_ratio_product((-source.charge_c, sphere.radius_m), (d,))
        potential = _scaled_ratio_product(
            (Q - q_image,),
            (4.0 * pi, problem.permittivity_f_per_m, sphere.radius_m),
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
