"""Exact Kelvin-image superposition for multiple charges outside one solid sphere.

No FEMM calls. Supports grounded and fixed-net-charge metal and arbitrarily
oriented charge positions. Image charges are mathematical devices INSIDE metal,
not physical point-charge components supplied by the author.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import acos, cos, fsum, hypot, isclose, isfinite, pi, sin, sqrt
from typing import Literal

from .external_sphere import _sphere_strictly_outside
from .model import (ElectrostaticProblem, FixedNetCharge, Grounded, PointCharge,
                    SolidSphere, Vector3, _position)
from .solver import ConductorResult, SingularField, UnsupportedProblem


def difference(a: Vector3, b: Vector3) -> Vector3:
    return tuple(a[i]-b[i] for i in range(3))


def norm(a: Vector3) -> float:
    return hypot(*a)


def dot(a: Vector3, b: Vector3) -> float:
    return fsum(a[i]*b[i] for i in range(3))


def _field(k: float, q: float, v: Vector3, r: float) -> Vector3:
    """k*q*v/r**3 for |v| = r, formed as (k*(q/r))/r times the unit vector.

    Forming k*(q/r) first means neither k*q nor q/r**2 can overflow or
    underflow while the field itself is a normal float.
    """
    scale = (k*(q/r))/r
    return tuple(scale*(x/r) for x in v)


def _pair_force(k: float, q: float, q_other: float, v: Vector3, r: float) -> Vector3:
    """Coulomb force k*q*q_other*v/r**3 as (k*(q/r))*(q_other/r)*(v/r)."""
    scale = (k*(q/r))*(q_other/r)
    return tuple(scale*(x/r) for x in v)


def active_external_sites(problem: ElectrostaticProblem) -> tuple[PointCharge, ...]:
    """Combine exactly coincident authored sites and remove cancelled sites.

    Physical validity of each originally authored charge is checked separately,
    including sites that cancel; malformed geometry cannot be laundered by zero.
    """
    amounts: dict[Vector3, list[float]] = {}
    for item in problem.charges:
        amounts.setdefault(item.position_m, []).append(item.charge_c)
    try:
        sites = tuple(PointCharge(fsum(parts), position)
                      for position, parts in sorted(amounts.items()))
    except (ValueError, OverflowError) as exc:
        raise UnsupportedProblem('nonfinite combined external point charges') from exc
    return tuple(site for site in sites if site.charge_c != 0)


def _multi_conductor_potential_v(
    sphere: SolidSphere,
    sites: tuple[PointCharge, ...],
    permittivity_f_per_m: float,
) -> float:
    """Overflow/underflow- and cancellation-safe conductor potential (Q/a + sum(q_j/d_j))/(4*pi*eps)."""
    if isinstance(sphere.constraint, Grounded):
        return 0.0
    from math import frexp
    from .observables import _scaled_ratio_parts, _sum_scaled_ratios
    four_pi = 4.0 * pi
    Q = sphere.constraint.charge_c
    parts = [_scaled_ratio_parts((Q,), (four_pi, permittivity_f_per_m, sphere.radius_m))]
    for site in sites:
        d = norm(difference(site.position_m, sphere.center_m))
        parts.append(_scaled_ratio_parts((site.charge_c,), (four_pi, permittivity_f_per_m, d)))
    val = _sum_scaled_ratios(parts)
    nonzero = [(m, e) for m, e in parts if m != 0.0]
    if len(nonzero) >= 2 and (not isfinite(val) or val == 0.0 or frexp(abs(val))[1] < max(e for _, e in nonzero) - 10):
        from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
        from fractions import Fraction
        from .observables import _PI_120, _sphere_geom_spread_digits

        def _fdec(v: Fraction) -> Decimal:
            return Decimal(v.numerator) / Decimal(v.denominator)

        c = tuple(Fraction(x) for x in sphere.center_m)
        d2s = [sum((Fraction(site.position_m[i]) - c[i]) ** 2 for i in range(3)) for site in sites]
        max_digits = 5000
        geom_spread = _sphere_geom_spread_digits(sphere, sites)
        zero_req_base = min(max_digits, max(480, 4 * geom_spread + 120))
        digits = min(max_digits, max(120, 2 * geom_spread + 80))
        while True:
            ctx = Context(prec=digits, rounding=ROUND_HALF_EVEN, Emax=999999999, Emin=-999999999, traps=[], flags=[])
            with localcontext(ctx):
                dk = Decimal(1) / (Decimal(4) * Decimal(_PI_120) * Decimal.from_float(permittivity_f_per_m))
                terms = [dk * Decimal.from_float(Q) / Decimal.from_float(sphere.radius_m)]
                for site, d2 in zip(sites, d2s):
                    terms.append(dk * Decimal.from_float(site.charge_c) / _fdec(d2).sqrt())
                nz = [abs(t) for t in terms if t != 0]
                if not nz:
                    return 0.0
                max_t, min_t = max(nz), min(nz)
                span_need = (max_t.adjusted() - min_t.adjusted() + 80) if len(nz) >= 2 else 0
                if span_need > digits and digits < max_digits:
                    digits = min(max_digits, max(2 * digits, span_need))
                    continue
                tot = sum(terms, Decimal(0))
                if tot == 0:
                    req = max(zero_req_base, span_need)
                    if digits >= req or digits >= max_digits:
                        return 0.0
                    digits = min(max_digits, max(req, 2 * digits))
                    continue
                lost = max(0, max_t.adjusted() - tot.adjusted())
                if digits - lost >= 40 or digits >= max_digits:
                    return float(tot)
                digits = min(max_digits, max(2 * digits, lost + 80))
    return val


def external_sphere_supported(problem: ElectrostaticProblem, *, min_components: int = 2) -> bool:
    if len(problem.conductors) != 1 or len(problem.charges) < min_components:
        return False
    sphere = problem.conductors[0]
    if not (isinstance(sphere, SolidSphere)
            and isinstance(sphere.constraint, (Grounded, FixedNetCharge))):
        return False
    if any(not isfinite(norm(difference(q.position_m, sphere.center_m)))
           or not _sphere_strictly_outside(q.position_m, sphere.center_m, sphere.radius_m)
           for q in problem.charges):
        return False
    try:
        from .observables import _scaled_ratio_product
        sites = active_external_sites(problem)
        images_total = fsum(_scaled_ratio_product((-p.charge_c, sphere.radius_m),
                                                  (norm(difference(p.position_m, sphere.center_m)),))
                            for p in sites)
        Q = (0.0 if isinstance(sphere.constraint, Grounded)
             else sphere.constraint.charge_c)
        central = 0.0 if isinstance(sphere.constraint, Grounded) else Q-images_total
        potential = _multi_conductor_potential_v(sphere, sites, problem.permittivity_f_per_m)
        return (0 < len(sites) <= 32 and isfinite(images_total)
                and isfinite(central) and isfinite(potential))
    except (ValueError, OverflowError, ZeroDivisionError, UnsupportedProblem):
        return False


@dataclass(frozen=True, slots=True)
class ExternalMultiSphereSolution:
    """Exact image-charge result; charge indexing refers to active grouped sites."""
    sphere: SolidSphere
    sites: tuple[PointCharge, ...]
    permittivity_f_per_m: float
    conductor_results: tuple[ConductorResult, ...]

    @property
    def _k(self) -> float:
        return 1/(4*pi*self.permittivity_f_per_m)

    @property
    def image_sources(self) -> tuple[PointCharge, ...]:
        from .observables import _scaled_ratio_product
        a = self.sphere.radius_m
        center = self.sphere.center_m
        images = []
        for site in self.sites:
            delta = difference(site.position_m, center)
            d = norm(delta)
            images.append(PointCharge(
                _scaled_ratio_product((-site.charge_c, a), (d,)),
                tuple(center[i] + _scaled_ratio_product((a, a, delta[i]), (d, d)) for i in range(3)),
            ))
        return tuple(images)

    @property
    def central_image_charge_c(self) -> float:
        if isinstance(self.sphere.constraint, Grounded):
            return 0.0
        return self.sphere.constraint.charge_c - fsum(q.charge_c for q in self.image_sources)

    @property
    def conductor_potential_v(self) -> float:
        return self.conductor_results[0].potential_v

    def _multi_exact_eval(self, pt: Vector3, *, want_field: bool):
        """Trap-isolated adaptive-precision Kelvin superposition for multi-source exterior sphere."""
        from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
        from fractions import Fraction
        from .observables import _PI_120

        def _fdec(val: Fraction) -> Decimal:
            return Decimal(val.numerator) / Decimal(val.denominator)

        a = Fraction(self.sphere.radius_m)
        a2 = a * a
        c = tuple(Fraction(x) for x in self.sphere.center_m)
        p = tuple(Fraction(x) for x in pt)
        r_p = tuple(p[i] - c[i] for i in range(3))
        rp2 = sum(x * x for x in r_p)
        fixed = isinstance(self.sphere.constraint, FixedNetCharge)
        Q_val = self.sphere.constraint.charge_c if fixed else 0.0
        site_data = []
        for site in self.sites:
            s = tuple(Fraction(x) for x in site.position_m)
            r_src = tuple(s[i] - c[i] for i in range(3))
            d2 = sum(x * x for x in r_src)
            v_dir = tuple(p[i] - s[i] for i in range(3))
            dist2 = sum(x * x for x in v_dir)
            v_ref = tuple(r_p[i] - (a2 / d2) * r_src[i] for i in range(3))
            r_img2 = sum(x * x for x in v_ref)
            site_data.append((site.charge_c, d2, v_dir, dist2, v_ref, r_img2))

        max_digits = 5000
        digits = 120
        while True:
            ctx = Context(prec=digits, rounding=ROUND_HALF_EVEN, Emax=999999999, Emin=-999999999, traps=[], flags=[])
            with localcontext(ctx):
                dk = Decimal(1) / (Decimal(4) * Decimal(_PI_120) * Decimal.from_float(self.permittivity_f_per_m))
                da = Decimal.from_float(self.sphere.radius_m)
                r_p_norm = _fdec(rp2).sqrt()
                q_imgs = []
                d_srcs = []
                dists = []
                r_imgs = []
                for qc, d2, _, dist2, _, r_img2 in site_data:
                    d_src = _fdec(d2).sqrt()
                    d_srcs.append(d_src)
                    dists.append(_fdec(dist2).sqrt())
                    r_imgs.append(_fdec(r_img2).sqrt())
                    q_imgs.append(-Decimal.from_float(qc) * da / d_src)
                q_center = (Decimal.from_float(Q_val) - sum(q_imgs, Decimal(0))) if fixed else Decimal(0)
                if not want_field:
                    terms = []
                    for idx, (qc, d2, _, _, _, _) in enumerate(site_data):
                        dq = Decimal.from_float(qc)
                        terms.append(
                            dk * dq * _fdec((d2 - a2) * (rp2 - a2)) / (
                                d_srcs[idx] * dists[idx] * r_imgs[idx] * (d_srcs[idx] * r_imgs[idx] + da * dists[idx])
                            )
                        )
                    if fixed:
                        if Q_val != 0.0:
                            terms.append(dk * Decimal.from_float(Q_val) / r_p_norm)
                        for qi in q_imgs:
                            terms.append(-dk * qi / r_p_norm)
                    nz = [abs(t) for t in terms if t != 0]
                    if not nz:
                        return 0.0
                    max_t, min_t = max(nz), min(nz)
                    span_need = (max_t.adjusted() - min_t.adjusted() + 80) if len(nz) >= 2 else 0
                    if span_need > digits and digits < max_digits:
                        digits = min(max_digits, max(2 * digits, span_need))
                        continue
                    total = sum(terms, Decimal(0))
                    if total == 0:
                        if digits >= max(480, span_need) or digits >= max_digits:
                            return 0.0
                        digits = min(max_digits, max(480, span_need, 2 * digits))
                        continue
                    lost = max(0, max_t.adjusted() - total.adjusted())
                    if digits - lost >= 40 or digits >= max_digits:
                        return float(total)
                    digits = min(max_digits, max(2 * digits, lost + 80))
                    continue

                rp3 = r_p_norm * r_p_norm * r_p_norm
                term_lists = ([], [], [])
                for idx, (qc, _, v_dir, _, v_ref, _) in enumerate(site_data):
                    dq = Decimal.from_float(qc)
                    dist3 = dists[idx] * dists[idx] * dists[idx]
                    r_img3 = r_imgs[idx] * r_imgs[idx] * r_imgs[idx]
                    for i in range(3):
                        if v_dir[i] != 0:
                            term_lists[i].append(dk * dq * _fdec(v_dir[i]) / dist3)
                        if v_ref[i] != 0:
                            term_lists[i].append(dk * q_imgs[idx] * _fdec(v_ref[i]) / r_img3)
                        if fixed and r_p[i] != 0:
                            term_lists[i].append(-dk * q_imgs[idx] * _fdec(r_p[i]) / rp3)
                if fixed and Q_val != 0.0:
                    dQ = Decimal.from_float(Q_val)
                    for i in range(3):
                        if r_p[i] != 0:
                            term_lists[i].append(dk * dQ * _fdec(r_p[i]) / rp3)
                totals = [Decimal(0), Decimal(0), Decimal(0)]
                need_more = False
                next_digits = digits
                for i in range(3):
                    nz = [abs(t) for t in term_lists[i] if t != 0]
                    if not nz:
                        continue
                    max_t, min_t = max(nz), min(nz)
                    span_need = (max_t.adjusted() - min_t.adjusted() + 80) if len(nz) >= 2 else 0
                    if span_need > digits and digits < max_digits:
                        need_more = True
                        next_digits = max(next_digits, max(2 * digits, span_need))
                        continue
                    tot = sum(term_lists[i], Decimal(0))
                    totals[i] = tot
                    if tot == 0:
                        req = max(480, span_need)
                        if digits < req and digits < max_digits:
                            need_more = True
                            next_digits = max(next_digits, max(req, 2 * digits))
                    else:
                        lost = max(0, max_t.adjusted() - tot.adjusted())
                        if digits - lost < 40 and digits < max_digits:
                            need_more = True
                            next_digits = max(next_digits, max(2 * digits, lost + 80))
                if not need_more or digits >= max_digits:
                    return tuple(float(x) for x in totals)
                digits = min(max_digits, next_digits)

    def potential_at(self, point_m: Vector3) -> float:
        from math import frexp
        from .observables import _scaled_ratio_parts, _scaled_ratio_product, _sum_scaled_ratios
        pt = _position(point_m, 'sample point')
        r_p = difference(pt, self.sphere.center_m)
        r = norm(r_p)
        if r <= self.sphere.radius_m:
            return self.conductor_potential_v
        a = self.sphere.radius_m
        four_pi = 4.0 * pi
        eps = self.permittivity_f_per_m
        fixed = isinstance(self.sphere.constraint, FixedNetCharge)
        parts = []
        need_exact = not isfinite(r)
        for site in self.sites:
            dist = norm(difference(pt, site.position_m))
            if dist == 0:
                raise SingularField('potential is singular at an external point charge')
            r_src = difference(site.position_m, self.sphere.center_m)
            d_src = norm(r_src)
            img_off = tuple(_scaled_ratio_product((a, a, r_src[i]), (d_src, d_src)) for i in range(3))
            reflected = tuple(r_p[i] - img_off[i] for i in range(3))
            r_img = norm(reflected)
            if (
                not (isfinite(dist) and isfinite(r_img) and r_img > 0.0)
                or any(r_src[i] != 0.0 and abs(img_off[i]) < 1e-290 for i in range(3))
                or any(
                    r_p[i] != 0.0 and img_off[i] != 0.0 and abs(reflected[i]) < 0.01 * max(abs(r_p[i]), abs(img_off[i]))
                    for i in range(3)
                )
            ):
                need_exact = True
                continue
            parts.append(_scaled_ratio_parts((site.charge_c,), (four_pi, eps, dist)))
            parts.append(_scaled_ratio_parts((-site.charge_c, a), (four_pi, eps, d_src, r_img)))
            if fixed:
                parts.append(_scaled_ratio_parts((site.charge_c, a), (four_pi, eps, d_src, r)))
        if need_exact:
            return self._multi_exact_eval(pt, want_field=False)
        if fixed:
            parts.append(_scaled_ratio_parts((self.sphere.constraint.charge_c,), (four_pi, eps, r)))
        total = _sum_scaled_ratios(parts)
        nonzero = [(m, e) for m, e in parts if m != 0.0]
        if len(nonzero) >= 2 and (not isfinite(total) or total == 0.0 or frexp(abs(total))[1] < max(e for _, e in nonzero) - 10):
            return self._multi_exact_eval(pt, want_field=False)
        return total

    def electric_field_at(self, point_m: Vector3) -> Vector3:
        from math import frexp
        from .observables import _scaled_ratio_parts, _scaled_ratio_product, _sum_scaled_ratios
        pt = _position(point_m, 'sample point')
        r_p = difference(pt, self.sphere.center_m)
        r = norm(r_p)
        if isclose(r, self.sphere.radius_m, rel_tol=1e-12):
            raise SingularField('electric field jumps at the conducting surface')
        if r < self.sphere.radius_m:
            return (0.0, 0.0, 0.0)
        a = self.sphere.radius_m
        four_pi = 4.0 * pi
        eps = self.permittivity_f_per_m
        fixed = isinstance(self.sphere.constraint, FixedNetCharge)
        terms = [[], [], []]
        need_exact = not isfinite(r)
        for site in self.sites:
            direct = difference(pt, site.position_m)
            dist = norm(direct)
            if dist == 0:
                raise SingularField('electric field is singular at an external point charge')
            r_src = difference(site.position_m, self.sphere.center_m)
            d_src = norm(r_src)
            img_off = tuple(_scaled_ratio_product((a, a, r_src[i]), (d_src, d_src)) for i in range(3))
            reflected = tuple(r_p[i] - img_off[i] for i in range(3))
            r_img = norm(reflected)
            if (
                not (isfinite(dist) and isfinite(r_img) and r_img > 0.0)
                or any(r_src[i] != 0.0 and abs(img_off[i]) < 1e-290 for i in range(3))
                or any(
                    r_p[i] != 0.0 and img_off[i] != 0.0 and abs(reflected[i]) < 0.01 * max(abs(r_p[i]), abs(img_off[i]))
                    for i in range(3)
                )
            ):
                need_exact = True
                continue
            for i in range(3):
                terms[i].append(_scaled_ratio_parts((site.charge_c, direct[i]), (four_pi, eps, dist, dist, dist)))
                terms[i].append(_scaled_ratio_parts((-site.charge_c, a, reflected[i]), (four_pi, eps, d_src, r_img, r_img, r_img)))
                if fixed:
                    terms[i].append(_scaled_ratio_parts((site.charge_c, a, r_p[i]), (four_pi, eps, d_src, r, r, r)))
        if need_exact:
            return self._multi_exact_eval(pt, want_field=True)
        if fixed:
            Q = self.sphere.constraint.charge_c
            for i in range(3):
                terms[i].append(_scaled_ratio_parts((Q, r_p[i]), (four_pi, eps, r, r, r)))
        out = tuple(_sum_scaled_ratios(term) for term in terms)
        for i in range(3):
            nonzero = [(m, e) for m, e in terms[i] if m != 0.0]
            if len(nonzero) >= 2 and (not isfinite(out[i]) or out[i] == 0.0 or frexp(abs(out[i]))[1] < max(e for _, e in nonzero) - 10):
                return self._multi_exact_eval(pt, want_field=True)
        return out

    def surface_charge_density_at(self, surface: Literal['outer'], point_m: Vector3) -> float:
        if surface != 'outer':
            raise ValueError('solid sphere has only an outer conducting surface')
        pt = _position(point_m, 'surface point')
        delta = difference(pt, self.sphere.center_m)
        a = self.sphere.radius_m
        radius = norm(delta)
        if not isclose(radius, a, rel_tol=1e-10, abs_tol=1e-13):
            raise ValueError('sample must lie on the conducting sphere')
        # Evaluate at the intended surface point (radial projection of the
        # admitted query). Each grounded Kelvin pair contributes
        # -q(d-a)(d+a)/(4 pi a L**3) with L = |a u - d n| written as
        # hypot(d-a, sqrt(a d)|u-n|): no cancellation near contact and no
        # L**3 overflow. The central compensation is spatially uniform.
        unit = tuple(x/radius for x in delta)
        values = []
        for site in self.sites:
            offset = difference(site.position_m, self.sphere.center_m)
            d = norm(offset)
            chord = norm(tuple(unit[i]-offset[i]/d for i in range(3)))
            gap = d-a
            length = hypot(gap, sqrt(a)*sqrt(d)*chord)
            values.append(-(site.charge_c/(4*pi*a))*(gap/length)*((d+a)/length)/length)
        values.append(self.central_image_charge_c/(4*pi*a)/a)
        return fsum(values)

    def surface_charge_density(self, surface: Literal['outer'], theta_rad: float) -> float:
        """Polar profile only when all sites lie on a common central axis.

        In a general 3D arrangement there is NO unique one-dimensional angular
        profile. Use surface_charge_density_at(face,world_point) instead.
        """
        if surface != 'outer':
            raise ValueError('solid sphere has only an outer face')
        if not isfinite(theta_rad) or not 0 <= theta_rad <= pi:
            raise ValueError('theta must be in [0,pi]')
        far = max(self.sites, key=lambda p: (norm(difference(p.position_m,self.sphere.center_m)),
                                             p.position_m))
        delta = difference(far.position_m,self.sphere.center_m)
        d = norm(delta)
        axis = tuple(x/d for x in delta)
        for site in self.sites:
            displacement = difference(site.position_m,self.sphere.center_m)
            z = dot(displacement,axis)
            if norm(tuple(displacement[i]-z*axis[i] for i in range(3))) > 1e-11*self.sphere.radius_m:
                raise UnsupportedProblem('no unique polar density profile for non-collinear sources; use surface_charge_density_at')
        ref = (1.,0.,0.) if abs(axis[0]) < .9 else (0.,1.,0.)
        projection = dot(ref,axis)
        transverse = tuple(ref[i]-projection*axis[i] for i in range(3))
        length = norm(transverse)
        perpendicular = tuple(x/length for x in transverse)
        a = self.sphere.radius_m
        point = tuple(self.sphere.center_m[i] + a*(cos(theta_rad)*axis[i]+sin(theta_rad)*perpendicular[i]) for i in range(3))
        return self.surface_charge_density_at(surface,point)

    def conductor_force_on_charge_n(self, index: int) -> Vector3:
        """Force on an active external site from the METAL only, no direct pair force."""
        from .grounded_plane import _checked_site_index
        from .observables import _sphere_force_only
        idx = _checked_site_index(index, len(self.sites))
        return _sphere_force_only(self, idx, include_direct=False)

    def force_on_charge_n(self, index: int) -> Vector3:
        """Total force on active site, including other REAL charges, excluding its own field."""
        from .grounded_plane import _checked_site_index
        from .observables import _sphere_force_only
        idx = _checked_site_index(index, len(self.sites))
        return _sphere_force_only(self, idx, include_direct=True)

    @property
    def interaction_energy_j(self) -> float:
        """Variable electrostatic energy excluding point/sphere constant self energies.

        Grounded-image term has factor 1/2. For fixed-Q metal, the quadratic
        central-monopole correction is k*((Q+S)^2-Q^2)/(2*a), S=sum(q*a/d).
        Its position derivative gives the correct total force on each charge.
        """
        from .observables import _sphere_energy_only
        return _sphere_energy_only(self)


class ExternalMultiSphereBackend:
    """Exact solution for >=2 authored charges external to one grounded/isolated sphere."""
    def supports(self, problem: ElectrostaticProblem) -> bool:
        return external_sphere_supported(problem, min_components=2)

    def solve(self, problem: ElectrostaticProblem) -> ExternalMultiSphereSolution:
        if not self.supports(problem):
            raise UnsupportedProblem('multiple external charges require one grounded or fixed-Q solid sphere, all sites strictly outside, and finite source strengths')
        from .observables import _scaled_ratio_product
        sphere = problem.conductors[0]
        sites = active_external_sites(problem)
        images_total = fsum(
            _scaled_ratio_product((-site.charge_c, sphere.radius_m),
                                  (norm(difference(site.position_m, sphere.center_m)),))
            for site in sites
        )
        Q = images_total if isinstance(sphere.constraint,Grounded) else sphere.constraint.charge_c
        potential = _multi_conductor_potential_v(sphere, sites, problem.permittivity_f_per_m)
        conductor = ConductorResult(sphere.name,potential,Q,0.0,Q)
        return ExternalMultiSphereSolution(sphere,sites,problem.permittivity_f_per_m,(conductor,))
