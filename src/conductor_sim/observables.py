"""Optional, self-field-free force/energy observations of a static solution.

No numerical backend is selected or solved here. Unsupported solutions
fail explicitly. Fixed-voltage energies are mechanical interaction potentials
(with voltage-source work accounted for), not total field/capacitor energies.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
from math import copysign, frexp, fsum, hypot, isfinite, ldexp
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

    def __post_init__(self) -> None:
        if not isinstance(self.source, PointCharge):
            raise TypeError('force source must be a PointCharge')
        for name in ('direct_force_n', 'induced_force_n', 'background_force_n'):
            object.__setattr__(self, name, _position(getattr(self, name), name))

    @property
    def conductor_force_n(self) -> Vector3:
        return _sum_vectors(self.induced_force_n, self.background_force_n)

    @property
    def total_force_n(self) -> Vector3:
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

    def __post_init__(self) -> None:
        if not all(isfinite(x) for x in (self.direct_j, self.induced_j, self.background_j)):
            raise ValueError('energy contributions must be finite')
        if not isinstance(self.reference, str) or not self.reference:
            raise ValueError('energy reference must be explicit')

    @property
    def total_j(self) -> float:
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


def _sum_scaled_ratios(parts: list[tuple[float, int]]) -> float:
    """Sum (mantissa, exponent) terms without intermediate overflow or underflow."""
    nonzero = [(m, e) for m, e in parts if m != 0.0]
    if not nonzero:
        return 0.0
    max_e = max(e for _, e in nonzero)
    total = fsum(ldexp(m, e - max_e) for m, e in nonzero)
    if total == 0.0:
        return 0.0
    m_tot, e_tot = frexp(total)
    try:
        return ldexp(m_tot, max_e + e_tot)
    except OverflowError:
        return copysign(float('inf'), m_tot)


def _direct(sources: tuple[PointCharge, ...], k: float):
    """Pair energy counted once; equal/opposite forces, never direct self terms."""
    forces = [[[], [], []] for _ in sources]
    energies = []
    for i, a in enumerate(sources):
        for j in range(i+1, len(sources)):
            b = sources[j]
            delta = tuple(a.position_m[c]-b.position_m[c] for c in range(3))
            distance = hypot(*delta)
            if distance == 0:
                raise ValueError('coincident force sites must be grouped before observation')
            energies.append(_scaled_ratio_parts((k, a.charge_c, b.charge_c), (distance,)))
            for c in range(3):
                m_f, e_f = _scaled_ratio_parts((k, a.charge_c, b.charge_c, delta[c]), (distance, distance, distance))
                forces[i][c].append((m_f, e_f))
                forces[j][c].append((-m_f, e_f))
    return tuple(tuple(_sum_scaled_ratios(c) for c in f) for f in forces), _sum_scaled_ratios(energies)


def _plane_observables(solution) -> ForceEnergyResult:
    from .grounded_plane import _mirror_displacement_scaled
    sources, k = solution.sources, solution._k
    direct, pair_energy = _direct(sources, k)
    charges = tuple(ChargeForce(site, direct[i], solution.force_on_charge_n(i, include_direct=False))
                    for i, site in enumerate(sources))
    induced_parts = []
    for a in sources:
        for b in sources:
            _, dm, de = _mirror_displacement_scaled(a.position_m, b.position_m, solution.plane)
            induced_parts.append(_scaled_ratio_parts((-0.5, k, a.charge_c, b.charge_c), (dm,), -de))
    induced = _sum_scaled_ratios(induced_parts)
    energy = EnergyBreakdown(pair_energy, induced, 0.,
        'Grounded plane: mechanical interaction energy; point self energies omitted; '
        'zero at separated charges infinitely far from the conductor.')
    return ForceEnergyResult(charges, energy, 'analytical grounded-plane images')


def _sphere_observables(solution) -> ForceEnergyResult:
    # Reuse the Kelvin source construction for both legacy single and multi APIs.
    from .external_multi import ExternalMultiSphereSolution, difference, norm
    if isinstance(solution, ExternalMultiSphereSolution):
        reference = solution
    else:
        reference = ExternalMultiSphereSolution(solution.sphere, (solution.source,),
                        solution.permittivity_f_per_m, solution.conductor_results)
    sources, images, k = reference.sites, reference.image_sources, reference._k
    direct, pair_energy = _direct(sources, k)
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
            self_mag = _scaled_ratio_product(
                (-k, site.charge_c, site.charge_c, a, a, a, 2.0 - ratio_sq),
                (dc, dc, dc, dc, dc, u_denom, u_denom),
            )
            induced_energy = _scaled_ratio_product(
                (-0.5, k, site.charge_c, site.charge_c, a, a, a),
                (dc, dc, dc, dc, u_denom),
            )
            bg_energy = _scaled_ratio_product((k, Q, site.charge_c), (dc,))
        else:
            # Exact near-contact- and far-field-safe grounded-sphere polarization:
            #   F_ind = -k * q^2 * a / (d^3 * (1 - (a/d)^2)^2) * r_hat
            #   U_ind = -0.5 * k * q^2 * a / (d^2 * (1 - (a/d)^2))
            self_mag = _scaled_ratio_product(
                (-k, site.charge_c, site.charge_c, a),
                (dc, dc, dc, u_denom, u_denom),
            )
            induced_energy = _scaled_ratio_product(
                (-0.5, k, site.charge_c, site.charge_c, a),
                (dc, dc, u_denom),
            )
            bg_energy = 0.0
        induced_vec = tuple(self_mag * r0_hat[c] for c in range(3))
        background = tuple(_scaled_ratio_product((k, site.charge_c, Q, r0_hat[c]), (dc, dc)) for c in range(3))
        charges.append(ChargeForce(site, direct[0], induced_vec, background))
    else:
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
                if prec - max_lost >= 35 or prec >= 3200:
                    break
                prec = min(3200, max(2 * prec, max_lost + 60))
        for i, site in enumerate(sources):
            charges.append(ChargeForce(site, direct[i], induced_vecs[i], backgrounds[i]))
    energy = EnergyBreakdown(pair_energy, induced_energy, bg_energy,
        ('Fixed-Q exterior sphere: mechanical interaction energy; constant sphere Q^2/(2C) '
         'and point self energies omitted; zero at infinite separation.' if fixed else
         'Grounded exterior sphere: mechanical interaction energy; point self energies '
         'omitted; zero at infinite separation.'))
    return ForceEnergyResult(tuple(charges), energy, 'analytical exterior-sphere Kelvin images')


def _plate_observables(solution) -> ForceEnergyResult:
    sources, k = solution.sources, solution._k
    direct, pair_energy = _direct(sources, k)
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
    direct, direct_energy = _direct(sources, k)
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
