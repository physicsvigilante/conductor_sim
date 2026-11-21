"""Quasistatic rearrangement work for isolated point charges in a uniform medium."""
from dataclasses import dataclass, replace
from fractions import Fraction
from ._coulomb_sum import coulomb_sum
from .affine_fields import AffineFieldAssembly

from .model import ElectrostaticProblem, PointCharge, Vector3
from .solver import UnsupportedProblem


@dataclass(frozen=True, slots=True)
class PointChargeRearrangement:
    initial: ElectrostaticProblem
    final: ElectrostaticProblem
    external_work_j: float

    @property
    def field_work_j(self) -> float:
        return -self.external_work_j


def _sites(problem):
    if not isinstance(problem, ElectrostaticProblem) or problem.conductors:
        raise UnsupportedProblem("requires free point charges without responding conductors")
    if isinstance(problem.permittivity_f_per_m, bool) or any(
        isinstance(site.charge_c, bool) for site in problem.charges
    ):
        raise ValueError("charge and permittivity must be numerical SI values")
    sites = [(Fraction(site.charge_c), tuple(map(Fraction, site.position_m)))
             for site in problem.charges if site.charge_c]
    if len({position for _, position in sites}) != len(sites):
        raise UnsupportedProblem("coincident nonzero point charges have singular pair energy")
    return sites


def _interactions(coefficients, site, others, sign):
    charge, point = site
    for other_charge, other_point in others:
        squared = sum((x-y)**2 for x, y in zip(point, other_point))
        coefficients[squared] = coefficients.get(squared, 0) + sign*charge*other_charge


def _pairs(coefficients, sites, sign):
    for i, site in enumerate(sites):
        _interactions(coefficients, site, sites[i+1:], sign)


def interaction_energy_j(problem: ElectrostaticProblem) -> float:
    """Pair energy relative to all free charges infinitely separated; exclude self energies."""
    coefficients = {}
    _pairs(coefficients, _sites(problem), 1)
    return coulomb_sum(coefficients, problem.permittivity_f_per_m)


def configuration_energy_j(problem: ElectrostaticProblem, background: AffineFieldAssembly) -> float:
    """Mutual pair energy plus coupling to an independently prescribed background."""
    if type(background) is not AffineFieldAssembly:
        raise UnsupportedProblem("requires a prescribed affine field assembly")
    sites = _sites(problem)
    coefficients = {}
    _pairs(coefficients, sites, 1)
    external = sum(charge * background._potential(position) for charge, position in sites)
    return coulomb_sum(coefficients, problem.permittivity_f_per_m, offset=external)


@dataclass(frozen=True, slots=True)
class PointChargeInsertion:
    final: ElectrostaticProblem
    external_work_j: float


def insert_point_charge(initial: ElectrostaticProblem, charge: PointCharge) -> PointChargeInsertion:
    """Bring one charge from infinity at unchanged kinetic energy, holding existing sites fixed."""
    if not isinstance(charge, PointCharge):
        raise TypeError("expected a PointCharge")
    _sites(initial)
    final = replace(initial, charges=(*initial.charges, charge))
    sites = _sites(final)
    coefficients = {}
    if charge.charge_c:
        _interactions(coefficients, sites[-1], sites[:-1], 1)
    return PointChargeInsertion(final, coulomb_sum(coefficients, final.permittivity_f_per_m))


def rearrange_point_charges(
    initial: ElectrostaticProblem, final_positions_m: tuple[Vector3, ...],
) -> PointChargeRearrangement:
    """Return external work at unchanged kinetic energy, excluding point self energies.

    Positions follow the initial charge order. Both endpoints are static and use
    the same uniform permittivity. The final problem can seed another rearrangement.
    """
    initial_sites = _sites(initial)
    positions = tuple(final_positions_m)
    if len(positions) != len(initial.charges):
        raise ValueError("provide a final position for each initial charge")
    final = replace(initial, charges=tuple(replace(site, position_m=position)
                                          for site, position in zip(initial.charges, positions)))
    coefficients = {}
    _pairs(coefficients, initial_sites, -1)
    _pairs(coefficients, _sites(final), 1)
    work = coulomb_sum(coefficients, initial.permittivity_f_per_m)
    return PointChargeRearrangement(initial, final, work)
