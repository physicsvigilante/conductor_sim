"""Optional, self-field-free force/energy observations of a static solution.

No numerical backend is selected or solved here. Unsupported solutions
fail explicitly. Fixed-voltage energies are mechanical interaction potentials
(with voltage-source work accounted for), not total field/capacitor energies.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import fsum, hypot, isfinite
from typing import Protocol

from .model import FixedNetCharge, PointCharge, Vector3, _position
from .solver import UnsupportedProblem

_ZERO: Vector3 = (0., 0., 0.)


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
            pair = k*a.charge_c*b.charge_c/distance
            energies.append(pair)
            for c in range(3):
                value = pair*(delta[c]/distance)/distance
                forces[i][c].append(value)
                forces[j][c].append(-value)
    return tuple(tuple(fsum(c) for c in f) for f in forces), fsum(energies)


def _plane_observables(solution) -> ForceEnergyResult:
    from .grounded_plane import _mirror, _minus
    sources, k = solution.sources, solution._k
    direct, pair_energy = _direct(sources, k)
    charges = tuple(ChargeForce(site, direct[i], solution.force_on_charge_n(i, include_direct=False))
                    for i, site in enumerate(sources))
    induced = -.5*k*fsum(a.charge_c*b.charge_c/hypot(*_minus(
        a.position_m, _mirror(b.position_m, solution.plane))) for a in sources for b in sources)
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
    charges = []
    for i, site in enumerate(sources):
        terms = [[], [], []]
        for image in images:
            delta = difference(site.position_m, image.position_m)
            d = norm(delta)
            for c in range(3):
                terms[c].append(k*site.charge_c*image.charge_c*(delta[c]/d)/d/d)
        delta = difference(site.position_m, reference.sphere.center_m)
        d = norm(delta)
        # Split Q from the induced central compensator S, not by subtracting
        # two large, nearly equal total forces after the fact.
        for c in range(3):
            terms[c].append(k*site.charge_c*S*(delta[c]/d)/d/d)
        background = tuple(k*site.charge_c*Q*(x/d)/d/d for x in delta)
        charges.append(ChargeForce(site, direct[i], tuple(fsum(t) for t in terms), background))
    kelvin_energy = .5*k*fsum(site.charge_c*image.charge_c/norm(difference(
        site.position_m, image.position_m)) for site in sources for image in images)
    energy = EnergyBreakdown(pair_energy, fsum((kelvin_energy, .5*k*S*S/a)), k*Q*S/a,
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
