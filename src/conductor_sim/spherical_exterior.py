"""Exterior-only observations for one spherical outer boundary enclosing sealed cavities."""
from dataclasses import dataclass, field
from fractions import Fraction

from ._coulomb_sum import coulomb_sum
from .affine_fields import _point, _scalar, _triple
from .cavity_gauss import GaussCavityAssembly, GaussCavityLedger, _outer_charge, solve_cavity_gauss
from .model import Vector3
from .solver import UnsupportedProblem


@dataclass(frozen=True, slots=True)
class SphericalCavityExterior:
    """Declare a spherical outer surface in an otherwise source-free infinite dielectric."""

    assembly: GaussCavityAssembly
    radius_m: float
    center_m: Vector3 = (0., 0., 0.)
    ledger: GaussCavityLedger = field(init=False, repr=False)
    _charge: Fraction = field(init=False, repr=False)

    def __post_init__(self):
        if _scalar(self.radius_m) <= 0:
            raise ValueError("exterior boundary radius must be positive")
        object.__setattr__(self, "center_m", _triple(self.center_m))
        ledger = solve_cavity_gauss(self.assembly)
        object.__setattr__(self, "ledger", ledger)
        object.__setattr__(self, "_charge", _outer_charge(ledger.assembly))

    def _displacement(self, point_m):
        d = tuple(p-c for p,c in zip(_point(point_m), _point(self.center_m)))
        squared = sum(x*x for x in d)
        if squared <= Fraction(self.radius_m)**2:
            raise UnsupportedProblem("probe must lie strictly outside the declared spherical boundary")
        return d, squared

    def potential_at(self, point_m):
        """Potential in volts, with zero at infinity."""
        _, squared = self._displacement(point_m)
        return coulomb_sum({squared: self._charge}, self.assembly.permittivity_f_per_m)

    def electric_field_at(self, point_m):
        """Cartesian electric field in V/m, positive radially outward for positive charge."""
        d, squared = self._displacement(point_m)
        return tuple(coulomb_sum({squared: self._charge*x/squared}, self.assembly.permittivity_f_per_m)
                     for x in d)
