"""Quasistatic rearrangement work for isolated point charges in a uniform medium."""
from dataclasses import dataclass, replace
from fractions import Fraction
from ._coulomb_sum import coulomb_sum

from .model import ElectrostaticProblem, Vector3
from .solver import UnsupportedProblem


@dataclass(frozen=True, slots=True)
class PointChargeRearrangement:
    initial: ElectrostaticProblem
    final: ElectrostaticProblem
    external_work_j: float

    @property
    def field_work_j(self) -> float:
        return -self.external_work_j


def rearrange_point_charges(
    initial: ElectrostaticProblem, final_positions_m: tuple[Vector3, ...],
) -> PointChargeRearrangement:
    """Return external work at unchanged kinetic energy, excluding point self energies.

    Positions follow the initial charge order. Both endpoints are static and use
    the same uniform permittivity. The final problem can seed another rearrangement.
    """
    if not isinstance(initial, ElectrostaticProblem) or initial.conductors:
        raise UnsupportedProblem("requires free point charges without responding conductors")
    positions = tuple(final_positions_m)
    if len(positions) != len(initial.charges):
        raise ValueError("provide a final position for each initial charge")
    if isinstance(initial.permittivity_f_per_m, bool) or any(
        isinstance(site.charge_c, bool) for site in initial.charges
    ):
        raise ValueError("charge and permittivity must be numerical SI values")
    final = replace(initial, charges=tuple(replace(site, position_m=position)
                                          for site, position in zip(initial.charges, positions)))
    coefficients = {}
    for sign, endpoint in ((-1, initial), (1, final)):
        sites = [(Fraction(site.charge_c), tuple(map(Fraction, site.position_m)))
                 for site in endpoint.charges if site.charge_c]
        for i, (charge, point) in enumerate(sites):
            for other_charge, other_point in sites[i+1:]:
                distance = sum((x-y)**2 for x, y in zip(point, other_point))
                if not distance:
                    raise UnsupportedProblem("coincident nonzero point charges have singular pair energy")
                coefficients[distance] = coefficients.get(distance, 0) + sign*charge*other_charge
    work = coulomb_sum(coefficients, initial.permittivity_f_per_m)
    return PointChargeRearrangement(initial, final, work)
