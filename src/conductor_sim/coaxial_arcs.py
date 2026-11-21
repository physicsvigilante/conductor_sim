"""Scalar observations on the common z axis of fixed planar circular charge arcs."""
from dataclasses import dataclass
from fractions import Fraction

from ._coulomb_sum import coulomb_sum
from .model import EPSILON_0
from .prescribed_fields import _number


@dataclass(frozen=True, slots=True)
class CircularArcCharge:
    """Total signed charge on a specified arc. Angular density may be nonuniform."""
    charge_c: float
    radius_m: float
    start_deg: float
    sweep_deg: float
    center_z_m: float = 0.

    def __post_init__(self):
        _number(self.charge_c, 'arc charge')
        _number(self.radius_m, 'arc radius', positive=True)
        _number(self.start_deg, 'starting angle')
        _number(self.sweep_deg, 'counterclockwise sweep', positive=True)
        _number(self.center_z_m, 'axial centre')
        if self.sweep_deg > 360:
            raise ValueError('arc sweep exceeds one revolution')


@dataclass(frozen=True, slots=True)
class CoaxialArcAssembly:
    """All centres lie on the common z axis; the arcs lie in parallel xy planes."""
    arcs: tuple[CircularArcCharge, ...]
    permittivity_f_per_m: float = EPSILON_0

    def __post_init__(self):
        arcs = tuple(self.arcs)
        if any(not isinstance(arc, CircularArcCharge) for arc in arcs):
            raise TypeError('expected fixed circular charge arcs')
        _number(self.permittivity_f_per_m, 'permittivity', positive=True)
        object.__setattr__(self, 'arcs', arcs)

    def _terms(self, z_m, *, axial_field=False):
        _number(z_m, 'axial position')
        z = Fraction(z_m)
        for arc in self.arcs:
            dz = z - Fraction(arc.center_z_m)
            distance_squared = Fraction(arc.radius_m)**2 + dz**2
            weight = Fraction(arc.charge_c)
            if axial_field:
                weight *= dz / distance_squared
            yield distance_squared, weight

    def _observe(self, terms):
        coefficients = {}
        for distance_squared, weight in terms:
            coefficients[distance_squared] = coefficients.get(distance_squared, 0) + weight
        return coulomb_sum(coefficients, self.permittivity_f_per_m)

    def potential_v(self, z_m):
        """Electric potential relative to infinity at (0, 0, z_m)."""
        return self._observe(self._terms(z_m))

    def axial_field_v_m(self, z_m):
        """Signed z component only. Transverse components depend on angular density."""
        return self._observe(self._terms(z_m, axial_field=True))

    def field_work_j(self, probe_charge_c, start_z_m, end_z_m):
        """Field work between two axial endpoints with the arc charges held fixed."""
        _number(probe_charge_c, 'probe charge')
        q = Fraction(probe_charge_c)
        terms = ((distance, sign*q*weight) for z, sign in ((start_z_m, 1), (end_z_m, -1))
                 for distance, weight in self._terms(z))
        return self._observe(terms)
