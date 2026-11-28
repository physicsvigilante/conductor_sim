"""Optional, self-field-free force/energy observations of a static solution.

No numerical backend is selected or solved here. Unsupported solutions
fail explicitly. Fixed-voltage energies are mechanical interaction potentials
(with voltage-source work accounted for), not total field/capacitor energies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
from math import copysign, frexp, fsum, hypot, isfinite, ldexp, pi
from typing import Protocol

from .model import FixedNetCharge, PointCharge, Vector3, _position
from .solver import UnsupportedProblem

_ZERO: Vector3 = (0., 0., 0.)
_PI_120 = (
    "3.141592653589793238462643383279502884197169399375105820974944"
    "592307816406286208998628034825342117067982148086513282306647"
)


def _sum_vectors(*vectors: Vector3) -> Vector3:
    return tuple(fsum(v[i] for v in vectors) for i in range(3))


@dataclass(frozen=True, slots=True)
class ChargeForce:
    """Force on one grouped active site, not on one coincident authored component.

    induced: charge-induced polarization, including the site's own images.
    background: conductor field with the external point charges removed.
    direct: all other real sites, excluding the singular direct self-field.
    """
    source: PointCharge
    direct_force_n: Vector3
    induced_force_n: Vector3
    background_force_n: Vector3 = _ZERO
    _total_override_n: Vector3 | None = field(default=None, repr=False, compare=False)
    _total_override_for: tuple[Vector3, Vector3, Vector3] | None = field(default=None, repr=False, compare=False)
    _conductor_override_n: Vector3 | None = field(default=None, repr=False, compare=False)
    _conductor_override_for: tuple[Vector3, Vector3] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source, PointCharge):
            raise TypeError('force source must be a PointCharge')
        for name in ('direct_force_n', 'induced_force_n', 'background_force_n'):
            object.__setattr__(self, name, _position(getattr(self, name), name))
        key = (self.direct_force_n, self.induced_force_n, self.background_force_n)
        if self._total_override_n is not None:
            if self._total_override_for is None:
                object.__setattr__(self, '_total_override_n', _position(self._total_override_n, 'total_force_n'))
                object.__setattr__(self, '_total_override_for', key)
            elif self._total_override_for != key:
                object.__setattr__(self, '_total_override_n', None)
                object.__setattr__(self, '_total_override_for', None)
            else:
                object.__setattr__(self, '_total_override_n', _position(self._total_override_n, 'total_force_n'))
        ckey = (self.induced_force_n, self.background_force_n)
        if self._conductor_override_n is not None:
            if self._conductor_override_for is None:
                object.__setattr__(
                    self, '_conductor_override_n', _position(self._conductor_override_n, 'conductor_force_n')
                )
                object.__setattr__(self, '_conductor_override_for', ckey)
            elif self._conductor_override_for != ckey:
                object.__setattr__(self, '_conductor_override_n', None)
                object.__setattr__(self, '_conductor_override_for', None)
            else:
                object.__setattr__(
                    self, '_conductor_override_n', _position(self._conductor_override_n, 'conductor_force_n')
                )

    @property
    def conductor_force_n(self) -> Vector3:
        if self._conductor_override_n is not None:
            return self._conductor_override_n
        return _sum_vectors(self.induced_force_n, self.background_force_n)

    @property
    def total_force_n(self) -> Vector3:
        if self._total_override_n is not None:
            return self._total_override_n
        return _sum_vectors(self.direct_force_n, self.induced_force_n, self.background_force_n)


@dataclass(frozen=True, slots=True)
class EnergyBreakdown:
    """Position-dependent mechanical energy, with divergent/constant self terms omitted.

    The factor 1/2 belongs to induced_j, NOT to background_j. A common voltage
    shift changes background_j by sum(q)*shift but leaves forces and fixed-
    charge displacement work unchanged. Read reference before comparing states.
    """
    direct_j: float
    induced_j: float
    background_j: float
    reference: str
    _total_override_j: float | None = field(default=None, repr=False, compare=False)
    _total_override_for: tuple[float, float, float] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not all(isfinite(x) for x in (self.direct_j, self.induced_j, self.background_j)):
            raise ValueError('energy contributions must be finite')
        key = (self.direct_j, self.induced_j, self.background_j)
        if self._total_override_j is not None:
            if self._total_override_for is None:
                if not isfinite(self._total_override_j):
                    raise ValueError('total energy must be finite')
                object.__setattr__(self, '_total_override_for', key)
            elif self._total_override_for != key:
                object.__setattr__(self, '_total_override_j', None)
                object.__setattr__(self, '_total_override_for', None)
            elif not isfinite(self._total_override_j):
                raise ValueError('total energy must be finite')
        if not isinstance(self.reference, str) or not self.reference:
            raise ValueError('energy reference must be explicit')

    @property
    def total_j(self) -> float:
        if self._total_override_j is not None:
            return self._total_override_j
        return fsum((self.direct_j, self.induced_j, self.background_j))


@dataclass(frozen=True, slots=True)
class ForceEnergyResult:
    """Immutable observations from one solution; force indices follow charges.

    Built-in providers use lexicographically sorted, grouped active source
    positions. Canceled/zero sites are omitted. There is no persistent particle
    identity across separate solves: use source.position_m and charge_c.
    """
    charges: tuple[ChargeForce, ...]
    energy: EnergyBreakdown
    method: str
    image_order: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'charges', tuple(self.charges))
        if any(not isinstance(c, ChargeForce) for c in self.charges):
            raise TypeError('charges must contain ChargeForce observations')
        if not isinstance(self.energy, EnergyBreakdown):
            raise TypeError('energy must be an EnergyBreakdown')
        if not isinstance(self.method, str) or not self.method:
            raise ValueError('observation method must be explicit')

    def force_on_charge_n(self, index: int) -> Vector3:
        """Total force; negative, boolean and non-integer indices are rejected."""
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError('source index must be an integer')
        if not 0 <= index < len(self.charges):
            raise IndexError('source index is outside the grouped active sites')
        return self.charges[index].total_force_n


class ForceEnergyProvider(Protocol):
    """Optional solution capability; ordinary FieldSolution remains unchanged."""
    def force_energy(self) -> ForceEnergyResult: ...


def force_energy(solution: object) -> ForceEnergyResult:
    """Read the optional provider; never substitute analytical output for FEMM.

    Legacy analytical solutions use explicit compatibility adapters.
    A custom solution can implement force_energy() returning ForceEnergyResult.
    Provider exceptions propagate. This does not acquire or close its session.
    """
    provider = getattr(solution, 'force_energy', None)
    if callable(provider):
        result = provider()
    else:
        # Compatibility adapters retain existing single-source properties and
        # multi-source methods rather than renaming/changing those public APIs.
        from .grounded_plane import GroundedPlaneSolution
        from .external_sphere import ExternalGroundedSphereSolution
        from .external_multi import ExternalMultiSphereSolution
        if isinstance(solution, GroundedPlaneSolution):
            result = _plane_observables(solution)
        elif isinstance(solution, (ExternalGroundedSphereSolution, ExternalMultiSphereSolution)):
            result = _sphere_observables(solution)
        else:
            # The same exact Kelvin cavity kernel applies to a single shell
            # and the innermost cavity of a concentric nested-shell system.
            # Neither backend is a numerical FEMM force provider.
            from .multi_image import MultiImageShellSolution
            from .nested_shells import NestedShellSolution
            from .analytical import ConcentricShellSolution
            from .image_charge import ImageChargeShellSolution
            if isinstance(solution, (MultiImageShellSolution, NestedShellSolution)):
                result = _cavity_observables(solution)
            elif isinstance(solution, (ConcentricShellSolution, ImageChargeShellSolution)):
                # Older exact single-source and centered-source solutions have
                # the same Kelvin boundary-value problem. Adapt their *saved*
                # source/charge records without solving a second problem.
                site = (PointCharge(solution.charge_c,solution.shell.center_m)
                        if isinstance(solution,ConcentricShellSolution) else solution.source)
                grouped = (site,) if site.charge_c else ()
                reference = MultiImageShellSolution(solution.shell,grouped,
                                    solution.permittivity_f_per_m,solution.conductor_results)
                result = _cavity_observables(reference)
            else:
                raise UnsupportedProblem(
                    f'{type(solution).__name__} does not provide self-field-free force/energy; '
                    'analytical planes, exterior spheres, parallel plates, and spherical '
                    'cavities are supported; numerical FEMM force is not')
    if not isinstance(result, ForceEnergyResult):
        raise TypeError('force_energy() must return ForceEnergyResult')
    return result


def _exact_sum(values) -> float:
    """Overflow-safe sum of finite floats: fast fsum with Fraction fallback on intermediate overflow."""
    items = tuple(values)
    try:
        return fsum(items)
    except OverflowError:
        exact = sum(Fraction(x) for x in items)
        try:
            return float(exact)
        except OverflowError:
            return float('inf') if exact > 0 else float('-inf')


def _scaled_ratio_parts(
    nums: tuple[float, ...],
    denoms: tuple[float, ...] = (),
    extra_exp: int = 0,
) -> tuple[float, int]:
    """Return (mantissa, exponent) for prod(nums) / prod(denoms) * 2**extra_exp."""
    mantissa = 1.0
    exponent = extra_exp
    for x in nums:
        if x == 0.0:
            return 0.0, 0
        m, e = frexp(x)
        mantissa *= m
        exponent += e
        mantissa, de = frexp(mantissa)
        exponent += de
    for x in denoms:
        m, e = frexp(x)
        mantissa /= m
        exponent -= e
        mantissa, de = frexp(mantissa)
        exponent += de
    return mantissa, exponent


def _scaled_ratio_product(nums: tuple[float, ...], denoms: tuple[float, ...] = (), extra_exp: int = 0) -> float:
    """Evaluate prod(nums) / prod(denoms) * 2**extra_exp without intermediate overflow or underflow."""
    mantissa, exponent = _scaled_ratio_parts(nums, denoms, extra_exp)
    if mantissa == 0.0:
        return 0.0
    try:
        return ldexp(mantissa, exponent)
    except OverflowError:
        return copysign(float('inf'), mantissa)


def _component_diff_parts(ai: float, bi: float) -> tuple[float, int]:
    """Return exact ``(mantissa, exp)`` for ``ai - bi`` without overflow."""
    d = ai - bi
    if isfinite(d):
        return frexp(d) if d != 0.0 else (0.0, 0)
    # Overflow implies opposite signs with magnitude ~1e308; halving is exact.
    m, e = frexp(0.5 * ai - 0.5 * bi)
    return (m, e + 1) if m != 0.0 else (0.0, 0)


def _fraction_to_parts(val: Fraction) -> tuple[float, int]:
    """Convert an exact nonzero Fraction to normalized ``(mantissa, exp)`` without overflow/underflow."""
    if val == 0:
        return 0.0, 0
    neg = val < 0
    mag = -val if neg else val
    shift = mag.numerator.bit_length() - mag.denominator.bit_length()
    shifted = mag * Fraction(1, 1 << shift) if shift >= 0 else mag * (1 << (-shift))
    m, e = frexp(float(shifted))
    return (-m if neg else m), shift + e


def _vector_norm_parts(comps: tuple[tuple[float, int], ...]) -> tuple[float, int]:
    """Return ``(dist_mantissa, dist_exp)`` for the Euclidean norm of per-component ``(m, e)`` parts."""
    nonzero = [(m, e) for m, e in comps if m != 0.0]
    if not nonzero:
        return 0.0, 0
    max_e = max(e for _, e in nonzero)
    d_s = hypot(*(ldexp(m, e - max_e) for m, e in nonzero))
    dm, de = frexp(d_s)
    return dm, max_e + de


def _displacement_parts(
    a: Vector3, b: Vector3
) -> tuple[tuple[tuple[float, int], tuple[float, int], tuple[float, int]], float, int]:
    """Return ``(comp_parts, dist_mantissa, dist_exp)`` for ``a - b``.

    Preserves each component's independent binary exponent so tiny transverse
    offsets (e.g. ``1e-308``) alongside large longitudinal distances never
    underflow to zero before charge factors restore their scale.
    """
    comps = tuple(_component_diff_parts(a[i], b[i]) for i in range(3))
    dm, de = _vector_norm_parts(comps)
    return comps, dm, de


def _sum_scaled_ratios(parts: list[tuple[float, int]]) -> float:
    """Sum (mantissa, exponent) terms without intermediate overflow, underflow, or double rounding."""
    nonzero = [(m, e) for m, e in parts if m != 0.0]
    if not nonzero:
        return 0.0
    max_e = max(e for _, e in nonzero)
    min_e = min(e for _, e in nonzero)
    if max_e - min_e <= 950:
        total = fsum(ldexp(m, e - max_e) for m, e in nonzero)
        if total != 0.0:
            m_tot, e_tot = frexp(total)
            final_e = max_e + e_tot
            if final_e > -1020:
                try:
                    return ldexp(m_tot, final_e)
                except OverflowError:
                    return copysign(float('inf'), m_tot)
    # Wide exponent spread (>950 bits), cancellation, or subnormal result:
    # sum exactly in Fraction and perform one IEEE-754 round-to-nearest-even step.
    exact = sum(Fraction(m) * (1 << (e - min_e)) for m, e in nonzero)
    if exact == 0:
        return 0.0
    neg = exact < 0
    mag = -exact if neg else exact
    true_exp = min_e + mag.numerator.bit_length() - mag.denominator.bit_length()
    if true_exp > 1030:
        return copysign(float('inf'), -1.0 if neg else 1.0)
    if true_exp < -1080:
        return 0.0
    exact_val = exact * (1 << min_e) if min_e >= 0 else exact * Fraction(1, 1 << (-min_e))
    try:
        return float(exact_val)
    except OverflowError:
        return copysign(float('inf'), -1.0 if neg else 1.0)


def _direct(sources: tuple[PointCharge, ...], permittivity_f_per_m: float):
    """Pair energy counted once; equal/opposite forces, never direct self terms."""
    four_pi = 4.0 * pi
    forces = [[[], [], []] for _ in sources]
    energies = []
    for i, a in enumerate(sources):
        for j in range(i+1, len(sources)):
            b = sources[j]
            comps, dm, de = _displacement_parts(a.position_m, b.position_m)
            if dm == 0.0:
                raise ValueError('coincident force sites must be grouped before observation')
            energies.append(
                _scaled_ratio_parts((a.charge_c, b.charge_c), (four_pi, permittivity_f_per_m, dm), -de)
            )
            for c in range(3):
                mc, ec = comps[c]
                if mc == 0.0:
                    continue
                m_f, e_f = _scaled_ratio_parts(
                    (a.charge_c, b.charge_c, mc),
                    (four_pi, permittivity_f_per_m, dm, dm, dm),
                    ec - 3 * de,
                )
                forces[i][c].append((m_f, e_f))
                forces[j][c].append((-m_f, e_f))
    return tuple(tuple(_sum_scaled_ratios(c) for c in f) for f in forces), _sum_scaled_ratios(energies)


def _plane_observables(solution) -> ForceEnergyResult:
    from .grounded_plane import _mirror_displacement_parts
    sources, eps = solution.sources, solution.permittivity_f_per_m
    four_pi = 4.0 * pi
    direct, pair_energy = _direct(sources, eps)
    charges = tuple(
        ChargeForce(
            site,
            direct[i],
            solution.force_on_charge_n(i, include_direct=False),
            _total_override_n=solution.force_on_charge_n(i, include_direct=True),
        )
        for i, site in enumerate(sources)
    )
    induced_parts = []
    for a in sources:
        for b in sources:
            _, dm, de = _mirror_displacement_parts(a.position_m, b.position_m, solution.plane)
            induced_parts.append(
                _scaled_ratio_parts((-0.5, a.charge_c, b.charge_c), (four_pi, eps, dm), -de)
            )
    induced = _sum_scaled_ratios(induced_parts)
    energy = EnergyBreakdown(
        pair_energy,
        induced,
        0.,
        'Grounded plane: mechanical interaction energy; point self energies omitted; '
        'zero at separated charges infinitely far from the conductor.',
        _total_override_j=solution.interaction_energy_j,
    )
    return ForceEnergyResult(charges, energy, 'analytical grounded-plane images')


def _sphere_observables(solution) -> ForceEnergyResult:
    # Reuse the Kelvin source construction for both legacy single and multi APIs.
    from .external_multi import ExternalMultiSphereSolution, difference, norm
    if isinstance(solution, ExternalMultiSphereSolution):
        reference = solution
    else:
        reference = ExternalMultiSphereSolution(solution.sphere, (solution.source,),
                        solution.permittivity_f_per_m, solution.conductor_results)
    sources, images = reference.sites, reference.image_sources
    eps = reference.permittivity_f_per_m
    four_pi = 4.0 * pi
    direct, pair_energy = _direct(sources, eps)
    fixed = isinstance(reference.sphere.constraint, FixedNetCharge)
    Q = reference.sphere.constraint.charge_c if fixed else 0.
    S = -fsum(image.charge_c for image in images) if fixed else 0.
    a = reference.sphere.radius_m
    center = reference.sphere.center_m
    charges = []
    if len(sources) == 1:
        site = sources[0]
        delta_c = difference(site.position_m, center)
        dc = norm(delta_c)
        r0_hat = tuple(x / dc for x in delta_c)
        ratio = a / dc
        if dc - a < 0.01 * a:
            # Near contact, binary64 hypot rounds sqrt(r^2) before subtracting a,
            # losing precision in d^2 - a^2 for off-axis positions; evaluate
            # (d^2 - a^2) / d^2 exactly in Fraction from the authored coordinates.
            f_delta = tuple(Fraction(site.position_m[c]) - Fraction(center[c]) for c in range(3))
            f_d2 = sum(x * x for x in f_delta)
            f_a2 = Fraction(a) * Fraction(a)
            u_denom = float((f_d2 - f_a2) / f_d2)
        else:
            u_denom = ((dc - a) / dc) * (1.0 + ratio)
        if fixed:
            # Exact cancellation-, overflow-, and underflow-free isolated-sphere polarization:
            #   F_ind = -k * q^2 * a^3 * (2 - (a/d)^2) / (d^5 * (1 - (a/d)^2)^2) * r_hat
            #   U_ind = -0.5 * k * q^2 * a^3 / (d^4 * (1 - (a/d)^2))
            ratio_sq = _scaled_ratio_product((a, a), (dc, dc))
            induced_vec = tuple(
                _scaled_ratio_product(
                    (-site.charge_c, site.charge_c, a, a, a, 2.0 - ratio_sq, delta_c[c]),
                    (four_pi, eps, dc, dc, dc, dc, dc, dc, u_denom, u_denom),
                )
                for c in range(3)
            )
            induced_energy = _scaled_ratio_product(
                (-0.5, site.charge_c, site.charge_c, a, a, a),
                (four_pi, eps, dc, dc, dc, dc, u_denom),
            )
            bg_energy = _scaled_ratio_product((Q, site.charge_c), (four_pi, eps, dc))
        else:
            # Exact near-contact- and far-field-safe grounded-sphere polarization:
            #   F_ind = -k * q^2 * a / (d^3 * (1 - (a/d)^2)^2) * r_hat
            #   U_ind = -0.5 * k * q^2 * a / (d^2 * (1 - (a/d)^2))
            induced_vec = tuple(
                _scaled_ratio_product(
                    (-site.charge_c, site.charge_c, a, delta_c[c]),
                    (four_pi, eps, dc, dc, dc, dc, u_denom, u_denom),
                )
                for c in range(3)
            )
            induced_energy = _scaled_ratio_product(
                (-0.5, site.charge_c, site.charge_c, a),
                (four_pi, eps, dc, dc, u_denom),
            )
            bg_energy = 0.0
        background = tuple(
            _scaled_ratio_product((site.charge_c, Q, delta_c[c]), (four_pi, eps, dc, dc, dc))
            for c in range(3)
        )
        if not all(isfinite(x) for x in (*induced_vec, *background, induced_energy, bg_energy)):
            raise UnsupportedProblem(
                'decomposed exterior-sphere force or energy contribution overflows binary64'
            )
        total_force_override = None
        conductor_force_override = None
        total_energy_override = None
        same_sign = Q != 0.0 and ((site.charge_c > 0.0) == (Q > 0.0))
        if fixed and same_sign:
            force_cancels = any(
                induced_vec[c] != 0.0
                and background[c] != 0.0
                and abs(fsum((induced_vec[c], background[c]))) < 0.01 * max(abs(induced_vec[c]), abs(background[c]))
                for c in range(3)
            )
            energy_cancels = (
                induced_energy != 0.0
                and bg_energy != 0.0
                and abs(fsum((induced_energy, bg_energy))) < 0.01 * max(abs(induced_energy), abs(bg_energy))
            )
            if force_cancels or energy_cancels:
                from .external_isolated import _isolated_exact_force_and_energy
                exact_f, exact_u = _isolated_exact_force_and_energy(
                    reference.sphere, site.position_m, site.charge_c, Q, eps
                )
                if force_cancels:
                    total_force_override = exact_f
                    conductor_force_override = exact_f
                if energy_cancels:
                    total_energy_override = exact_u
        charges.append(
            ChargeForce(
                site,
                direct[0],
                induced_vec,
                background,
                _total_override_n=total_force_override,
                _conductor_override_n=conductor_force_override,
            )
        )
    else:
        total_energy_override = None
        # Multi-source exterior sphere (fixed-Q or grounded): evaluate Kelvin
        # images and any central compensator in a fresh trap-isolated Decimal
        # Context (explicit rounding=ROUND_HALF_EVEN, traps=[], flags=[], so
        # neither caller context nor DefaultContext traps can interfere), with
        # precision adapting to authored coordinate/charge spreads and any
        # runtime cancellation in force, induced energy, or compensator S.
        def _clean_ctx(digits: int) -> Context:
            return Context(
                prec=digits,
                rounding=ROUND_HALF_EVEN,
                Emax=999999999,
                Emin=-999999999,
                traps=[],
                flags=[],
            )

        da = Decimal.from_float(a)
        dc_vec = tuple(Decimal.from_float(x) for x in center)
        dQ = Decimal.from_float(Q)
        dqs = [Decimal.from_float(s.charge_c) for s in sources]
        with localcontext(_clean_ctx(60)):
            dk = Decimal(1) / (Decimal(4) * Decimal(_PI_120) * Decimal.from_float(reference.permittivity_f_per_m))
        with localcontext(_clean_ctx(100)):
            geom_exps = [da.adjusted()]
            for m in range(3):
                if dc_vec[m] != 0:
                    geom_exps.append(dc_vec[m].adjusted())
            for s in sources:
                for m in range(3):
                    dp = Decimal.from_float(s.position_m[m])
                    if dp != 0:
                        geom_exps.append(dp.adjusted())
                    diff_m = dp - dc_vec[m]
                    if diff_m != 0:
                        geom_exps.append(diff_m.adjusted())
        charge_exps = [dq.adjusted() for dq in dqs if dq != 0]
        geom_spread = (max(geom_exps) - min(geom_exps)) if geom_exps else 0
        charge_spread = (max(charge_exps) - min(charge_exps)) if charge_exps else 0
        prec = max(90, min(2400, 90 + 3 * geom_spread + charge_spread))
        induced_vecs: list[Vector3] = []
        backgrounds: list[Vector3] = []
        induced_energy = 0.0
        bg_energy = 0.0
        for _ in range(5):
            with localcontext(_clean_ctx(prec)):
                drs = [
                    tuple(Decimal.from_float(s.position_m[m]) - dc_vec[m] for m in range(3))
                    for s in sources
                ]
                dds = [sum(x * x for x in r).sqrt() for r in drs]
                dq_imgs = [-dqs[j] * da / dds[j] for j in range(len(sources))]
                dS = sum(-qi for qi in dq_imgs) if fixed else Decimal(0)
                dr_imgs = [
                    tuple((da * da / (dds[j] * dds[j])) * drs[j][m] for m in range(3))
                    for j in range(len(sources))
                ]
                max_lost = 0
                if fixed and dS != 0:
                    s_sc = max(abs(qi) for qi in dq_imgs)
                    if s_sc > 0:
                        lost = s_sc.adjusted() - dS.adjusted()
                        if lost > max_lost:
                            max_lost = lost
                induced_vecs = []
                backgrounds = []
                induced_nz = []
                bg_nz = []
                for i in range(len(sources)):
                    ri, di, qi = drs[i], dds[i], dqs[i]
                    di3 = di * di * di
                    f_ind = [Decimal(0), Decimal(0), Decimal(0)]
                    f_sc = [Decimal(0), Decimal(0), Decimal(0)]
                    for j in range(len(sources)):
                        w = tuple(ri[m] - dr_imgs[j][m] for m in range(3))
                        wn = sum(x * x for x in w).sqrt()
                        wn3 = wn * wn * wn
                        for m in range(3):
                            term = qi * dq_imgs[j] * w[m] / wn3
                            f_ind[m] += term
                            at = abs(term)
                            if at > f_sc[m]:
                                f_sc[m] = at
                    for m in range(3):
                        if fixed:
                            term = qi * dS * ri[m] / di3
                            f_ind[m] += term
                            at = abs(term)
                            if at > f_sc[m]:
                                f_sc[m] = at
                        if f_ind[m] != 0 and f_sc[m] > 0:
                            lost = f_sc[m].adjusted() - f_ind[m].adjusted()
                            if lost > max_lost:
                                max_lost = lost
                    induced_vecs.append(tuple(float(dk * x) for x in f_ind))
                    backgrounds.append(tuple(float(dk * qi * dQ * ri[m] / di3) for m in range(3)))
                    induced_nz.append(tuple(x != 0 for x in f_ind))
                    bg_nz.append(tuple(fixed and qi != 0 and dQ != 0 and ri[m] != 0 for m in range(3)))
                kelvin_u = Decimal(0)
                u_sc = Decimal(0)
                for i in range(len(sources)):
                    for j in range(len(sources)):
                        w = tuple(drs[i][m] - dr_imgs[j][m] for m in range(3))
                        wn = sum(x * x for x in w).sqrt()
                        term = Decimal("0.5") * dqs[i] * dq_imgs[j] / wn
                        kelvin_u += term
                        at = abs(term)
                        if at > u_sc:
                            u_sc = at
                comp_u = Decimal("0.5") * dS * dS / da if fixed else Decimal(0)
                if abs(comp_u) > u_sc:
                    u_sc = abs(comp_u)
                u_raw = kelvin_u + comp_u
                if u_raw != 0 and u_sc > 0:
                    lost = u_sc.adjusted() - u_raw.adjusted()
                    if lost > max_lost:
                        max_lost = lost
                induced_energy = float(dk * u_raw)
                bg_energy = float(dk * dQ * dS / da) if fixed else 0.0
                u_ind_nz = u_raw != 0
                u_bg_nz = fixed and dQ != 0 and dS != 0
                if prec - max_lost >= 35 or prec >= 3200:
                    break
                prec = min(3200, max(2 * prec, max_lost + 60))

        def _fsum_needs_exact(
            terms: tuple[float, ...],
            exact_nonzero: tuple[bool, ...],
        ) -> bool:
            if not all(isfinite(x) for x in terms):
                return True
            if any(nz and abs(x) < 1e-300 for x, nz in zip(terms, exact_nonzero)):
                return True
            if sum(x != 0.0 for x in terms) < 2:
                return False
            try:
                total_val = fsum(terms)
            except OverflowError:
                return True
            return abs(total_val) < 0.01 * max(abs(x) for x in terms)

        for i, site in enumerate(sources):
            cond_ov = None
            tot_ov = None
            direct_nz = tuple(
                any(
                    site.charge_c != 0.0
                    and sources[j].charge_c != 0.0
                    and site.position_m[c] != sources[j].position_m[c]
                    for j in range(len(sources))
                    if j != i
                )
                for c in range(3)
            )
            cond_cancel = any(
                _fsum_needs_exact(
                    (induced_vecs[i][c], backgrounds[i][c]),
                    (induced_nz[i][c], bg_nz[i][c]),
                )
                for c in range(3)
            )
            tot_cancel = any(
                _fsum_needs_exact(
                    (direct[i][c], induced_vecs[i][c], backgrounds[i][c]),
                    (direct_nz[c], induced_nz[i][c], bg_nz[i][c]),
                )
                for c in range(3)
            )
            if cond_cancel:
                cond_ov = _sphere_force_only(reference, i, include_direct=False)
            if tot_cancel:
                tot_ov = _sphere_force_only(reference, i, include_direct=True)
            charges.append(
                ChargeForce(
                    site,
                    direct[i],
                    induced_vecs[i],
                    backgrounds[i],
                    _total_override_n=tot_ov,
                    _conductor_override_n=cond_ov,
                )
            )
        pair_u_nz = any(
            sources[i].charge_c != 0.0 and sources[j].charge_c != 0.0
            for i in range(len(sources))
            for j in range(i + 1, len(sources))
        )
        if _fsum_needs_exact(
            (pair_energy, induced_energy, bg_energy),
            (pair_u_nz, u_ind_nz, u_bg_nz),
        ):
            total_energy_override = _sphere_energy_only(reference)
    energy = EnergyBreakdown(pair_energy, induced_energy, bg_energy,
        ('Fixed-Q exterior sphere: mechanical interaction energy; constant sphere Q^2/(2C) '
         'and point self energies omitted; zero at infinite separation.' if fixed else
         'Grounded exterior sphere: mechanical interaction energy; point self energies '
         'omitted; zero at infinite separation.'),
        _total_override_j=total_energy_override)
    return ForceEnergyResult(tuple(charges), energy, 'analytical exterior-sphere Kelvin images')


def _sphere_geom_spread_digits(sphere, sources: tuple[PointCharge, ...]) -> int:
    exps = [frexp(abs(sphere.radius_m))[1]]
    for x in sphere.center_m:
        if x != 0.0:
            exps.append(frexp(abs(x))[1])
    for s in sources:
        for c in range(3):
            if s.position_m[c] != 0.0:
                exps.append(frexp(abs(s.position_m[c]))[1])
            d = s.position_m[c] - sphere.center_m[c]
            if isfinite(d) and d != 0.0:
                exps.append(frexp(abs(d))[1])
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            for c in range(3):
                d = sources[i].position_m[c] - sources[j].position_m[c]
                if isfinite(d) and d != 0.0:
                    exps.append(frexp(abs(d))[1])
    if len(exps) < 2:
        return 0
    return ((max(exps) - min(exps)) * 302) // 1000 + 20


def _sphere_force_only(reference, index: int, *, include_direct: bool) -> Vector3:
    """Compute one site's conductor or total force without requiring unrelated energy/sites to be finite."""
    sources = reference.sites
    eps = reference.permittivity_f_per_m
    fixed = isinstance(reference.sphere.constraint, FixedNetCharge)
    Q = reference.sphere.constraint.charge_c if fixed else 0.0
    if len(sources) == 1:
        site = sources[0]
        if fixed:
            from .external_isolated import ExternalIsolatedSphereSolution
            from .solver import ConductorResult
            sol = ExternalIsolatedSphereSolution(
                reference.sphere, site, eps,
                (ConductorResult(reference.sphere.name, reference.conductor_potential_v, Q, 0.0, Q),),
            )
            return sol.induced_force_on_charge_n
        from .external_sphere import ExternalGroundedSphereSolution
        sol = ExternalGroundedSphereSolution(
            reference.sphere, site, eps, reference.conductor_results
        )
        return sol.induced_force_on_charge_n

    def _fdec(val: Fraction) -> Decimal:
        return Decimal(val.numerator) / Decimal(val.denominator)

    fa = Fraction(reference.sphere.radius_m)
    fa2 = fa * fa
    fc = tuple(Fraction(x) for x in reference.sphere.center_m)
    f_pos = [tuple(Fraction(x) for x in s.position_m) for s in sources]
    f_drs = [tuple(f_pos[k][m] - fc[m] for m in range(3)) for k in range(len(sources))]
    f_d2 = [sum(x * x for x in r) for r in f_drs]
    f_img_off = [tuple((fa2 / f_d2[j]) * f_drs[j][m] for m in range(3)) for j in range(len(sources))]
    f_ws = [
        tuple(f_drs[index][m] - f_img_off[j][m] for m in range(3))
        for j in range(len(sources))
    ]
    f_wn2 = [sum(x * x for x in w) for w in f_ws]
    f_vds = [
        tuple(f_pos[index][m] - f_pos[j][m] for m in range(3)) if (include_direct and j != index) else None
        for j in range(len(sources))
    ]
    f_dn2 = [sum(x * x for x in vd) if vd is not None else None for vd in f_vds]

    da = Decimal.from_float(reference.sphere.radius_m)
    dQ = Decimal.from_float(Q)
    dqs = [Decimal.from_float(s.charge_c) for s in sources]
    max_digits = 5000
    geom_spread = _sphere_geom_spread_digits(reference.sphere, sources)
    zero_req_base = min(max_digits, max(480, 4 * geom_spread + 120))
    digits = min(max_digits, max(120, 2 * geom_spread + 80))
    while True:
        ctx = Context(prec=digits, rounding=ROUND_HALF_EVEN, Emax=999999999, Emin=-999999999, traps=[], flags=[])
        with localcontext(ctx):
            dk = Decimal(1) / (Decimal(4) * Decimal(_PI_120) * Decimal.from_float(eps))
            dds = [_fdec(d2).sqrt() for d2 in f_d2]
            dq_imgs = [-dqs[j] * da / dds[j] for j in range(len(sources))]
            di = dds[index]
            di3 = di * di * di
            qi = dqs[index]
            term_lists = ([], [], [])
            for j in range(len(sources)):
                wn = _fdec(f_wn2[j]).sqrt()
                wn3 = wn * wn * wn
                for m in range(3):
                    if f_ws[j][m] != 0:
                        term_lists[m].append(dk * qi * dq_imgs[j] * _fdec(f_ws[j][m]) / wn3)
                if include_direct and j != index:
                    vd = f_vds[j]
                    dn = _fdec(f_dn2[j]).sqrt()
                    dn3 = dn * dn * dn
                    for m in range(3):
                        if vd[m] != 0:
                            term_lists[m].append(dk * qi * dqs[j] * _fdec(vd[m]) / dn3)
            if fixed:
                for m in range(3):
                    if f_drs[index][m] != 0:
                        rm_dec = _fdec(f_drs[index][m])
                        if Q != 0.0:
                            term_lists[m].append(dk * qi * dQ * rm_dec / di3)
                        for qi_img in dq_imgs:
                            term_lists[m].append(-dk * qi * qi_img * rm_dec / di3)
            need_more = False
            next_digits = digits
            totals = [Decimal(0), Decimal(0), Decimal(0)]
            for m in range(3):
                nz = [abs(t) for t in term_lists[m] if t != 0]
                if not nz:
                    continue
                max_t, min_t = max(nz), min(nz)
                span_need = (max_t.adjusted() - min_t.adjusted() + 80) if len(nz) >= 2 else 0
                if span_need > digits and digits < max_digits:
                    need_more = True
                    next_digits = max(next_digits, max(2 * digits, span_need))
                    continue
                tot = sum(term_lists[m], Decimal(0))
                totals[m] = tot
                if tot == 0:
                    req = max(zero_req_base, span_need)
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


def _sphere_energy_only(reference) -> float:
    """Compute total exterior-sphere interaction energy without requiring site forces to be finite."""
    sources = reference.sites
    eps = reference.permittivity_f_per_m
    fixed = isinstance(reference.sphere.constraint, FixedNetCharge)
    Q = reference.sphere.constraint.charge_c if fixed else 0.0
    if len(sources) == 1:
        site = sources[0]
        if fixed:
            from .external_isolated import ExternalIsolatedSphereSolution
            from .solver import ConductorResult
            sol = ExternalIsolatedSphereSolution(
                reference.sphere, site, eps,
                (ConductorResult(reference.sphere.name, reference.conductor_potential_v, Q, 0.0, Q),),
            )
            return sol.interaction_energy_j
        from .external_sphere import ExternalGroundedSphereSolution
        sol = ExternalGroundedSphereSolution(
            reference.sphere, site, eps, reference.conductor_results
        )
        return sol.interaction_energy_j

    def _fdec(val: Fraction) -> Decimal:
        return Decimal(val.numerator) / Decimal(val.denominator)

    fa = Fraction(reference.sphere.radius_m)
    fa2 = fa * fa
    fc = tuple(Fraction(x) for x in reference.sphere.center_m)
    f_pos = [tuple(Fraction(x) for x in s.position_m) for s in sources]
    f_drs = [tuple(f_pos[k][m] - fc[m] for m in range(3)) for k in range(len(sources))]
    f_d2 = [sum(x * x for x in r) for r in f_drs]
    f_img_off = [tuple((fa2 / f_d2[j]) * f_drs[j][m] for m in range(3)) for j in range(len(sources))]
    f_dn2_pairs = [
        (i, j, sum((f_pos[i][m] - f_pos[j][m]) ** 2 for m in range(3)))
        for i in range(len(sources))
        for j in range(i + 1, len(sources))
    ]
    f_wn2_pairs = [
        (i, j, sum((f_drs[i][m] - f_img_off[j][m]) ** 2 for m in range(3)))
        for i in range(len(sources))
        for j in range(len(sources))
    ]

    da = Decimal.from_float(reference.sphere.radius_m)
    dQ = Decimal.from_float(Q)
    dqs = [Decimal.from_float(s.charge_c) for s in sources]
    max_digits = 5000
    geom_spread = _sphere_geom_spread_digits(reference.sphere, sources)
    zero_req_base = min(max_digits, max(480, 4 * geom_spread + 120))
    digits = min(max_digits, max(120, 2 * geom_spread + 80))
    while True:
        ctx = Context(prec=digits, rounding=ROUND_HALF_EVEN, Emax=999999999, Emin=-999999999, traps=[], flags=[])
        with localcontext(ctx):
            dk = Decimal(1) / (Decimal(4) * Decimal(_PI_120) * Decimal.from_float(eps))
            dds = [_fdec(d2).sqrt() for d2 in f_d2]
            dq_imgs = [-dqs[j] * da / dds[j] for j in range(len(sources))]
            dS = sum((-qi for qi in dq_imgs), Decimal(0)) if fixed else Decimal(0)
            terms = []
            for i, j, dn2 in f_dn2_pairs:
                terms.append(dk * dqs[i] * dqs[j] / _fdec(dn2).sqrt())
            for i, j, wn2 in f_wn2_pairs:
                terms.append(Decimal("0.5") * dk * dqs[i] * dq_imgs[j] / _fdec(wn2).sqrt())
            if fixed:
                if dS != 0:
                    terms.append(Decimal("0.5") * dk * dS * dS / da)
                if Q != 0.0 and dS != 0:
                    terms.append(dk * dQ * dS / da)
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


def _plate_observables(solution) -> ForceEnergyResult:
    sources = solution.sources
    direct, pair_energy = _direct(sources, solution.permittivity_f_per_m)
    from .parallel_plates import _voltage
    e0 = -(_voltage(solution.upper)-_voltage(solution.lower))/solution.gap_m
    charges, induced_energy, background_energy = [], [], []
    for i, site in enumerate(sources):
        fields = [[], [], []]
        potentials = []
        for other in sources:
            # Omit the real n=0 term for every pair here: it is in _direct.
            # Every image, including this site's induced images, is retained.
            potentials.append(solution._source_potential(site.position_m, other, exclude_direct=True))
            field = solution._source_field(site.position_m, other, exclude_direct=True)
            for c in range(3):
                fields[c].append(field[c])
        induced = tuple(site.charge_c*fsum(f) for f in fields)
        background = tuple(site.charge_c*e0*n for n in solution.lower.normal)
        charges.append(ChargeForce(site, direct[i], induced, background))
        induced_energy.append(.5*site.charge_c*fsum(potentials))
        background_energy.append(site.charge_c*solution._background_v(solution._height(site.position_m)))
    energy = EnergyBreakdown(pair_energy, fsum(induced_energy), fsum(background_energy),
        'Fixed-voltage plates: mechanical interaction potential with voltage-source work '
        'accounted for; point self energies and the empty-capacitor energy omitted. '
        'Finite even when infinite biased-plate total field energy is unbounded. '
        'Displacement work requires unchanged charges, geometry, voltages and voltage gauge.')
    return ForceEnergyResult(tuple(charges), energy,
        'analytical plate-symmetric reciprocal image series (finite order)', solution.image_order)


def _cavity_voltage_parts(shells, total_q: float, k: float) -> tuple[float, float]:
    """Induced and source-free uniform cavity voltages, without subtraction.

    In concentric geometry, the nearest grounded shell fixes the outer end
    of this radial voltage sum. Until that shell, each gap's flux is total_q
    plus the enclosed prescribed shell charges. With no ground, include the
    exterior term relative to infinity. This is the exact Gauss-law response,
    computed separately so a large fixed Q cannot hide a small source charge.
    """
    from .model import Grounded
    first_ground = next((i for i, shell in enumerate(shells)
                         if isinstance(shell.constraint, Grounded)), len(shells))
    weights = [(1/shells[i].outer_radius_m - 1/shells[i+1].inner_radius_m)
               for i in range(min(first_ground, len(shells)-1))]
    if first_ground == len(shells):
        weights.append(1/shells[-1].outer_radius_m)
    background = []
    net = []
    for i, weight in enumerate(weights):
        net.append(shells[i].constraint.charge_c)
        background.append(fsum(net)*weight)
    return k*total_q*fsum(weights), k*fsum(background)



def _cavity_observables(solution) -> ForceEnergyResult:
    """Exact Kelvin cavity response, independent of concentric outer shells.

    Inner-surface polarization contributes a reciprocal image Green kernel.
    The innermost shell's spatially uniform voltage contains (i) a source-
    independent background and (ii) a source-induced constant term. The
    latter belongs to the *half-weighted* induced energy. Both constants
    have zero force and cancel in fixed-inventory displacement work.

    The image of a centered charge is at infinity; the finite limiting
    potential -q/a is retained by ``image_term`` without a fictitious point.
    """
    from .kelvin import image_term
    from .multi_image import MultiImageShellSolution
    from .nested_shells import NestedShellSolution
    if isinstance(solution, MultiImageShellSolution):
        sources = solution.charges  # grouped active sites, no phantom zeros
        shell = solution.shell
        radial_shells = (shell,)
        method = 'exact spherical cavity Kelvin images (single shell)'
    elif isinstance(solution, NestedShellSolution):
        sources = solution.inner_reference.charges
        shell = solution.radial_shells[0]
        radial_shells = solution.radial_shells
        method = 'exact Kelvin cavity images with concentric-shell radial coupling'
    else:
        raise TypeError('cavity observation requires an analytical shell solution')

    k = solution._k
    direct, direct_energy = _direct(sources, solution.permittivity_f_per_m)
    charges = []
    image_energies = []
    for i, site in enumerate(sources):
        f_terms = [[], [], []]
        potential_terms = []
        for other in sources:
            potential, field = image_term(site.position_m, other.position_m,
                                          shell.center_m, shell.inner_radius_m)
            potential_terms.append(other.charge_c*potential)
            for axis in range(3):
                f_terms[axis].append(other.charge_c*field[axis])
        induced = tuple(k*site.charge_c*fsum(axis) for axis in f_terms)
        image_energies.append(.5*k*site.charge_c*fsum(potential_terms))
        charges.append(ChargeForce(site, direct[i], induced, _ZERO))

    total_q = fsum(site.charge_c for site in sources)
    uniform_induced, background_v = _cavity_voltage_parts(radial_shells, total_q, k)
    induced_energy = fsum((*image_energies, .5*total_q*uniform_induced))
    energy = EnergyBreakdown(
        direct_energy, induced_energy, total_q*background_v,
        'Spherical cavity: mechanical point-charge interaction energy. '
        'Direct point self terms and conductor-only constants omitted. '
        'The source-dependent uniform shell voltage enters induced energy '
        'with the factor 1/2; the source-free background enters in full. '
        'Both are position-independent at fixed charge inventory, geometry, '
        'electrical constraints and potential gauge. Quasistatic displacement '
        'work is an energy difference only under those fixed conditions.')
    return ForceEnergyResult(tuple(charges), energy, method)
