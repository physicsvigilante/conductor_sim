"""Electrostatic energy barrier along a prescribed, straight frictionless guide."""
from dataclasses import dataclass
from fractions import Fraction
from math import pi

from .affine_fields import AffineFieldAssembly, _dot, _linear_integral, _observed, _point, _scalar
from .model import EPSILON_0
from .prescribed_fields import PrescribedField, UniformElectricField, UniformSphereField
from .solver import UnsupportedProblem


@dataclass(frozen=True, slots=True)
class SegmentEnergyBarrier:
    """Potential energy rise from the start and earliest fraction attaining its maximum."""

    rise_j: float
    peak_fraction: Fraction


def _sphere_endpoint_fields(sphere, endpoints, permittivity):
    radius = Fraction(sphere.radius_m)
    center = _point(sphere.center_m)
    offsets = [tuple(x - c for x, c in zip(point, center)) for point in endpoints]
    if any(_dot(d, d) > radius**2 for d in offsets):
        raise UnsupportedProblem("the entire guide must lie within each uniform volume sphere")
    # A ball is convex. Gauss's law gives this affine field on the complete segment.
    slope = Fraction(sphere.charge_c) / (4 * Fraction(pi) * permittivity * radius**3)
    return [tuple(slope * x for x in d) for d in offsets]


def straight_path_energy_profile(sources, charge_c, start_m, end_m, *, permittivity_f_m=None):
    """Qualify the sources and construct q[V(s) - V(start)] on the closed segment.

    Accepts a qualified PrescribedField, an AffineFieldAssembly, or an iterable of
    uniform fields, affine assemblies and containing uniform volume spheres. The ideal guide supplies a transverse reaction doing zero work.
    """
    start, end = _point(start_m), _point(end_m)
    delta = tuple(b - a for a, b in zip(start, end))
    if start == end:
        raise ValueError("guide endpoints must be distinct")
    charge = Fraction(_scalar(charge_c))
    if type(sources) is PrescribedField:
        if permittivity_f_m is not None and _scalar(permittivity_f_m) != sources.permittivity_f_m:
            raise ValueError("permittivity conflicts with the authored field")
        permittivity_f_m = sources.permittivity_f_m
        sources = sources.sources
    elif type(sources) is AffineFieldAssembly:
        sources = (sources,)
    eps = Fraction(_scalar(EPSILON_0 if permittivity_f_m is None else permittivity_f_m))
    if eps <= 0:
        raise ValueError("permittivity must be positive")

    initial_slope = Fraction(0)
    final_slope = Fraction(0)
    for source in sources:
        if type(source) is UniformSphereField:
            first, last = _sphere_endpoint_fields(source, (start, end), eps)
        elif type(source) is UniformElectricField:
            first = last = _point(source.field_v_m)
        elif type(source) is AffineFieldAssembly:
            first, last = source._field(start), source._field(end)
        else:
            raise UnsupportedProblem("guide requires uniform fields, affine assemblies or a containing uniform volume sphere")
        initial_slope -= charge * _dot(first, delta)
        final_slope -= charge * _dot(last, delta)

    return StraightPathEnergyProfile(initial_slope, final_slope)


@dataclass(frozen=True, slots=True)
class StraightPathEnergyProfile:
    """Energy derivatives (J per unit segment fraction) at the two ends."""

    initial_slope: Fraction
    final_slope: Fraction

    def __post_init__(self):
        for name in ("initial_slope", "final_slope"):
            value = getattr(self, name)
            if not isinstance(value, Fraction):
                object.__setattr__(self, name, Fraction(_scalar(value)))

    def energy_change_j(self, fraction):
        """Signed change from the starting potential energy at 0 <= fraction <= 1."""
        s = fraction if isinstance(fraction, Fraction) else Fraction(_scalar(fraction))
        if not 0 <= s <= 1:
            raise ValueError("segment fraction must be between zero and one")
        return _observed(_linear_integral(self.initial_slope, self.final_slope, s))

    def barrier(self):
        if self.initial_slope > 0 > self.final_slope:
            peak = self.initial_slope / (self.initial_slope - self.final_slope)
        elif self.initial_slope + self.final_slope > 0:
            peak = Fraction(1)
        else:
            peak = Fraction(0)
        return SegmentEnergyBarrier(self.energy_change_j(peak), peak)


def straight_path_energy_barrier(sources, charge_c, start_m, end_m, *, permittivity_f_m=None):
    """Return the maximum energy rise and its earliest location along the guide."""
    return straight_path_energy_profile(
        sources, charge_c, start_m, end_m, permittivity_f_m=permittivity_f_m
    ).barrier()
