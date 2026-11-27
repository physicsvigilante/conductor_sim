"""Exact Kelvin-image reference for an EXTERNAL point charge and a grounded solid sphere.

This is not the interior-cavity image method: for a source distance d > a,
the image is -q*a/d at a*a/d INSIDE the physical sphere. The conductor is
solid, not a hollow SphericalShell, and supports no interior charge sources.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import hypot, isclose, isfinite, pi, sin, sqrt
from typing import Literal

from .model import ElectrostaticProblem, Grounded, PointCharge, SolidSphere, Vector3, _position
from .solver import ConductorResult, SingularField, UnsupportedProblem


def _sphere_strictly_outside(point: Vector3, center: Vector3, radius: float) -> bool:
    """Require binary64 radial separation d > radius AND exact |point - center|^2 > radius^2.

    Refuses both truly interior points whose hypot rounds up to > radius and
    binary64-unresolved near-surface points where hypot rounds down to <= radius
    (so every admitted source has positive radial clearance across all sphere APIs).
    """
    delta = tuple(point[i] - center[i] for i in range(3))
    if not all(isfinite(x) for x in delta):
        return False
    d = hypot(*delta)
    if not isfinite(d) or d <= radius:
        return False
    if d > radius * (1.0 + 1e-12):
        return True
    f_delta = tuple(Fraction(point[i]) - Fraction(center[i]) for i in range(3))
    return sum(x * x for x in f_delta) > Fraction(radius) * Fraction(radius)


def _diff(a: Vector3, b: Vector3) -> Vector3:
    return tuple(a[i] - b[i] for i in range(3))


def _norm(v: Vector3) -> float:
    # hypot is free of intermediate overflow/underflow at any admitted scale.
    return hypot(*v)


def _chord(u: Vector3, n: Vector3) -> float:
    """|u-n| for unit vectors: 2*sin(theta/2), accurate even as theta -> 0."""
    return hypot(*(u[i]-n[i] for i in range(3)))


def _field(k: float, q: float, v: Vector3, r: float) -> Vector3:
    """k*q*v/r**3 for |v| = r without intermediate overflow or underflow."""
    from .observables import _scaled_ratio_product
    return tuple(_scaled_ratio_product((k, q, x), (r, r, r)) for x in v)


def _pair_force(k: float, q: float, q_other: float, v: Vector3, r: float) -> Vector3:
    """Coulomb force k*q*q_other*v/r**3 without intermediate overflow or underflow."""
    from .observables import _scaled_ratio_product
    return tuple(_scaled_ratio_product((k, q, q_other, x), (r, r, r)) for x in v)


@dataclass(frozen=True, slots=True)
class ExternalGroundedSphereSolution:
    sphere: SolidSphere
    source: PointCharge
    permittivity_f_per_m: float
    conductor_results: tuple[ConductorResult, ...]

    @property
    def symmetry_axis(self) -> Vector3:
        dvec = _diff(self.source.position_m, self.sphere.center_m)
        d = _norm(dvec)
        return tuple(x/d for x in dvec)

    @property
    def source_distance_m(self) -> float:
        return _norm(_diff(self.source.position_m, self.sphere.center_m))

    @property
    def image_charge_c(self) -> float:
        from .observables import _scaled_ratio_product
        return _scaled_ratio_product((-self.source.charge_c, self.sphere.radius_m), (self.source_distance_m,))

    @property
    def image_position_m(self) -> Vector3:
        from .observables import _scaled_ratio_product
        a, d, n = self.sphere.radius_m, self.source_distance_m, self.symmetry_axis
        return tuple(self.sphere.center_m[i] + _scaled_ratio_product((a, a, n[i]), (d,)) for i in range(3))

    @property
    def _k(self) -> float:
        return 1/(4*pi*self.permittivity_f_per_m)

    def _kelvin_decimal_eval(self, point: Vector3, *, want_field: bool):
        """High-precision trap-isolated Kelvin real+image sum when cancellation or overflow occurs."""
        from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
        from math import frexp
        from .observables import _PI_120
        exps = [
            frexp(abs(v))[1]
            for v in (
                self.sphere.radius_m,
                *self.sphere.center_m,
                *point,
                *self.source.position_m,
            )
            if v != 0.0
        ]
        spread_dec = ((max(exps) - min(exps)) // 3 + 40) if len(exps) >= 2 else 0
        prec = max(120, min(800, 80 + spread_dec))
        ctx = Context(prec=prec, rounding=ROUND_HALF_EVEN, Emax=999999999, Emin=-999999999, traps=[], flags=[])
        with localcontext(ctx):
            da = Decimal.from_float(self.sphere.radius_m)
            dc = tuple(Decimal.from_float(x) for x in self.sphere.center_m)
            dp = tuple(Decimal.from_float(x) for x in point)
            ds = tuple(Decimal.from_float(x) for x in self.source.position_m)
            dq = Decimal.from_float(self.source.charge_c)
            dk = Decimal(1) / (Decimal(4) * Decimal(_PI_120) * Decimal.from_float(self.permittivity_f_per_m))
            r_src = tuple(ds[i] - dc[i] for i in range(3))
            r_p = tuple(dp[i] - dc[i] for i in range(3))
            d_src = sum(x * x for x in r_src).sqrt()
            dq_img = -dq * da / d_src
            scale = da * da / (d_src * d_src)
            v_dir = tuple(dp[i] - ds[i] for i in range(3))
            dist = sum(x * x for x in v_dir).sqrt()
            v_ref = tuple(r_p[i] - scale * r_src[i] for i in range(3))
            r_img = sum(x * x for x in v_ref).sqrt()
            if not want_field:
                return float(dk * (dq / dist + dq_img / r_img))
            dist3 = dist * dist * dist
            r_img3 = r_img * r_img * r_img
            return tuple(float(dk * (dq * v_dir[i] / dist3 + dq_img * v_ref[i] / r_img3)) for i in range(3))

    def potential_at(self, point_m: Vector3) -> float:
        from math import ldexp
        from .observables import _scaled_ratio_parts, _sum_scaled_ratios
        point = _position(point_m, 'sample point')
        r = _norm(_diff(point, self.sphere.center_m))
        if r <= self.sphere.radius_m:
            return 0.0
        source_distance = _norm(_diff(point, self.source.position_m))
        if source_distance == 0:
            raise SingularField('point-charge potential is singular at the external source')
        r_img = _norm(_diff(point, self.image_position_m))
        if not (isfinite(source_distance) and isfinite(r_img) and r_img > 0.0):
            return self._kelvin_decimal_eval(point, want_field=False)
        p_src = _scaled_ratio_parts((self._k, self.source.charge_c), (source_distance,))
        p_img = _scaled_ratio_parts(
            (-self._k, self.source.charge_c, self.sphere.radius_m),
            (self.source_distance_m, r_img),
        )
        max_e = max(p_src[1], p_img[1])
        s_sum = ldexp(p_src[0], p_src[1] - max_e) + ldexp(p_img[0], p_img[1] - max_e)
        if abs(s_sum) < 0.01:
            return self._kelvin_decimal_eval(point, want_field=False)
        return _sum_scaled_ratios([p_src, p_img])

    def electric_field_at(self, point_m: Vector3) -> Vector3:
        from math import ldexp
        from .observables import _scaled_ratio_parts, _sum_scaled_ratios
        point = _position(point_m, 'sample point')
        r = _norm(_diff(point, self.sphere.center_m))
        if isclose(r, self.sphere.radius_m, rel_tol=1e-12):
            raise SingularField('field has distinct one-sided limits at sphere surface')
        if r < self.sphere.radius_m:
            return (0.0,0.0,0.0)
        direct = _diff(point, self.source.position_m)
        dist = _norm(direct)
        if dist == 0:
            raise SingularField('electric field is singular at the external charge')
        reflected = _diff(point, self.image_position_m)
        r_img = _norm(reflected)
        if not (isfinite(dist) and isfinite(r_img) and r_img > 0.0):
            return self._kelvin_decimal_eval(point, want_field=True)
        d_src = self.source_distance_m
        a = self.sphere.radius_m
        q = self.source.charge_c
        out = []
        for i in range(3):
            p_src = _scaled_ratio_parts((self._k, q, direct[i]), (dist, dist, dist))
            p_img = _scaled_ratio_parts((-self._k, q, a, reflected[i]), (d_src, r_img, r_img, r_img))
            if p_src[0] != 0.0 and p_img[0] != 0.0:
                max_e = max(p_src[1], p_img[1])
                s_sum = ldexp(p_src[0], p_src[1] - max_e) + ldexp(p_img[0], p_img[1] - max_e)
                if abs(s_sum) < 0.01:
                    return self._kelvin_decimal_eval(point, want_field=True)
            out.append(_sum_scaled_ratios([p_src, p_img]))
        return tuple(out)

    def surface_charge_density(self, surface: Literal['outer'], theta_rad: float) -> float:
        from math import isfinite
        if surface != 'outer':
            raise ValueError('solid sphere has only an outer conducting surface')
        if not isfinite(theta_rad) or not 0 <= theta_rad <= pi:
            raise ValueError('angle must lie in [0, pi]')
        return self._density_from_chord(2*sin(theta_rad/2))

    def _density_from_chord(self, chord: float) -> float:
        """Grounded Kelvin density at the surface point with |u-n| = chord.

        |a*u - d*n|**2 = (d-a)**2 + a*d*chord**2 avoids cancellation near
        contact; exact Fraction (d^2 - a^2) prevents hypot rounding error in
        the near-contact gap, and scaled ratios avoid length**3 overflow/underflow.
        """
        from .observables import _scaled_ratio_product
        a, d, q = self.sphere.radius_m, self.source_distance_m, self.source.charge_c
        if d - a < 0.01 * a:
            f_delta = tuple(Fraction(self.source.position_m[i]) - Fraction(self.sphere.center_m[i]) for i in range(3))
            f_d2 = sum(x * x for x in f_delta)
            f_a2 = Fraction(a) * Fraction(a)
            u_denom = float((f_d2 - f_a2) / f_d2)
            one_plus_ratio = 1.0 + a / d
            gap = _scaled_ratio_product((u_denom, d), (one_plus_ratio,))
        else:
            one_plus_ratio = 1.0 + a / d
            gap = d - a
        chord_term = sqrt(d) * sqrt(a) * chord if chord > 0.0 else 0.0
        if not isfinite(chord_term):
            chord_term = _scaled_ratio_product((sqrt(d), sqrt(a), chord))
        length = hypot(gap, chord_term)
        return _scaled_ratio_product((-q, gap, d, one_plus_ratio), (4.0 * pi, a, length, length, length))

    def surface_charge_density_at(self, surface: Literal['outer'], point_m: Vector3) -> float:
        point = _position(point_m,'surface point')
        delta = _diff(point,self.sphere.center_m)
        radius = _norm(delta)
        if not isclose(radius,self.sphere.radius_m,rel_tol=1e-10,abs_tol=1e-13):
            raise ValueError('point must lie on sphere surface')
        if surface != 'outer':
            raise ValueError('solid sphere has only an outer conducting surface')
        # Evaluate at the intended surface point: the radial projection of
        # the admitted query, through the chord to the source direction.
        unit = tuple(x/radius for x in delta)
        return self._density_from_chord(_chord(unit, self.symmetry_axis))

    @property
    def induced_force_on_charge_n(self) -> Vector3:
        """Exact physical force on source due to the grounded conductor, excluding self-field."""
        from .observables import _scaled_ratio_product
        d = self.source_distance_m
        a = self.sphere.radius_m
        delta = _diff(self.source.position_m, self.sphere.center_m)
        if d - a < 0.01 * a:
            f_delta = tuple(Fraction(self.source.position_m[i]) - Fraction(self.sphere.center_m[i]) for i in range(3))
            f_d2 = sum(x * x for x in f_delta)
            f_a2 = Fraction(a) * Fraction(a)
            u_denom = float((f_d2 - f_a2) / f_d2)
        else:
            u_denom = ((d - a) / d) * (1.0 + a / d)
        mag = _scaled_ratio_product(
            (-self._k, self.source.charge_c, self.source.charge_c, a),
            (d, d, d, u_denom, u_denom),
        )
        return tuple(mag * (delta[i] / d) for i in range(3))

    @property
    def force_on_charge_n(self) -> Vector3:
        """Total physical force on the charge due to the conducting sphere."""
        return self.induced_force_on_charge_n

    @property
    def interaction_energy_j(self) -> float:
        """Induced-charge interaction energy, with its physical factor 1/2."""
        from .observables import _scaled_ratio_product
        d = self.source_distance_m
        a = self.sphere.radius_m
        if d - a < 0.01 * a:
            f_delta = tuple(Fraction(self.source.position_m[i]) - Fraction(self.sphere.center_m[i]) for i in range(3))
            f_d2 = sum(x * x for x in f_delta)
            f_a2 = Fraction(a) * Fraction(a)
            u_denom = float((f_d2 - f_a2) / f_d2)
        else:
            u_denom = ((d - a) / d) * (1.0 + a / d)
        return _scaled_ratio_product(
            (-0.5, self._k, self.source.charge_c, self.source.charge_c, a),
            (d, d, u_denom),
        )


class ExternalGroundedSphereBackend:
    """Exact solution for one nonzero point charge outside one grounded solid sphere."""

    def supports(self, problem: ElectrostaticProblem) -> bool:
        if len(problem.conductors) != 1 or len(problem.charges) != 1:
            return False
        sphere, source = problem.conductors[0], problem.charges[0]
        return (isinstance(sphere, SolidSphere) and isinstance(sphere.constraint, Grounded)
                and source.charge_c != 0
                and _sphere_strictly_outside(source.position_m, sphere.center_m, sphere.radius_m))

    def solve(self, problem: ElectrostaticProblem) -> ExternalGroundedSphereSolution:
        if not self.supports(problem):
            raise UnsupportedProblem(
                'external grounded-sphere solution needs exactly one nonzero point '
                'charge strictly outside one grounded solid sphere; isolated, '
                'interior, touching and multi-source problems are not supported')
        from .observables import _scaled_ratio_product
        sphere, source = problem.conductors[0], problem.charges[0]
        d = _norm(_diff(source.position_m, sphere.center_m))
        induced = _scaled_ratio_product((-source.charge_c, sphere.radius_m), (d,))
        observed = ConductorResult(sphere.name, 0.0, induced, 0.0, induced)
        return ExternalGroundedSphereSolution(sphere, source, problem.permittivity_f_per_m, (observed,))
