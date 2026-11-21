"""Prepare isolated concentric conductors from their specified equilibrium potentials."""
from collections.abc import Mapping
from dataclasses import replace
from fractions import Fraction
from math import isfinite, pi

from .mixed_radial import MixedRadialBackend
from .model import ElectrostaticProblem, FixedNetCharge, SolidSphere
from .nested_shells import NestedShellBackend
from .solver import UnsupportedProblem


def _voltage(value):
    if type(value) not in (float, int) or not isfinite(value):
        raise ValueError("target potentials must be finite SI numbers")
    return Fraction(value)


def prepare_radial_voltages(problem: ElectrostaticProblem,
                            potentials_v: Mapping[str, float]) -> ElectrostaticProblem:
    """Set all conductor potentials, then disconnect the supplies at fixed geometry.

    Returned net charges are isolated. Potentials use zero at infinity. The
    original net charges/grounding are superseded by the preparation step.
    """
    if not isinstance(problem, ElectrostaticProblem):
        raise TypeError("expected an ElectrostaticProblem")
    backend = (MixedRadialBackend() if any(isinstance(s, SolidSphere) for s in problem.conductors)
               else NestedShellBackend())
    if not backend.supports(problem):
        raise UnsupportedProblem("potential preparation requires a supported concentric radial problem")
    if not isinstance(potentials_v, Mapping) or set(potentials_v) != {s.name for s in problem.conductors}:
        raise ValueError("supply one target potential for every named conductor")
    shapes = sorted(problem.conductors, key=lambda s: s.outer_radius_m)
    targets = [_voltage(potentials_v[s.name]) for s in shapes]
    factor = Fraction(4*pi) * Fraction(problem.permittivity_f_per_m)
    cavity_charge = sum((Fraction(p.charge_c) for p in problem.charges), Fraction())
    incoming = cavity_charge
    fluxes = []
    prepared = {}
    for i, shape in enumerate(shapes):
        a = Fraction(shape.outer_radius_m)
        if i+1 < len(shapes):
            b = Fraction(shapes[i+1].inner_radius_m)
            outgoing = factor * (targets[i] - targets[i+1]) / (1/a - 1/b)
        else:
            outgoing = factor * a * targets[i]
        net = outgoing - incoming
        try:
            charge = float(net)
        except OverflowError as exc:
            raise UnsupportedProblem("prepared charge exceeds the numerical range") from exc
        if net and charge == 0:
            raise UnsupportedProblem("prepared charge lies below the numerical range")
        prepared[shape.name] = replace(shape, constraint=FixedNetCharge(charge))
        fluxes.append(outgoing)
        incoming = outgoing

    result = replace(problem, conductors=tuple(prepared[s.name] for s in problem.conductors))
    solved = backend.solve(result)
    observed = [solved.conductor_by_name(s.name) for s in shapes]
    tolerance = Fraction(1, 10**11)
    # Check the physical charge inventory itself. The native backend's rounded
    # cavity subtotal can lose a small source before adding the conductor charge.
    enclosed = cavity_charge
    realized_fluxes = []
    for i, shape in enumerate(shapes):
        enclosed += Fraction(prepared[shape.name].constraint.charge_c)
        if abs(enclosed - fluxes[i]) > tolerance * abs(fluxes[i]):
            raise UnsupportedProblem("rounded isolated charges cannot reproduce the required radial flux")
        realized_fluxes.append(enclosed)
    exterior = Fraction(shapes[-1].outer_radius_m)
    actual_v = realized_fluxes[-1] / (factor * exterior)
    zero_voltage_tolerance = Fraction(1, 10**12)
    for i in reversed(range(len(shapes))):
        if i+1 < len(shapes):
            inverse_gap = 1/Fraction(shapes[i].outer_radius_m) - 1/Fraction(shapes[i+1].inner_radius_m)
            actual_v += realized_fluxes[i] * inverse_gap / factor
        limit = tolerance * abs(targets[i]) if targets[i] else zero_voltage_tolerance
        if abs(actual_v - targets[i]) > limit:
            raise UnsupportedProblem("rounded isolated charges cannot reproduce a requested conductor potential")
    for i, row in enumerate(observed):
        potential = Fraction(row.potential_v)
        limit = tolerance * abs(targets[i]) if targets[i] else zero_voltage_tolerance
        if abs(potential - targets[i]) > limit:
            raise UnsupportedProblem("isolated charges cannot reproduce a requested conductor potential")
        if abs(Fraction(row.outer_surface_charge_c) - fluxes[i]) > tolerance * abs(fluxes[i]):
            raise UnsupportedProblem("isolated charges cannot reproduce the required radial flux")
        if i+1 < len(shapes):
            requested = targets[i] - targets[i+1]
            actual = potential - Fraction(observed[i+1].potential_v)
            if abs(actual - requested) > tolerance * abs(requested):
                raise UnsupportedProblem("isolated charges cannot resolve a requested gap voltage")
    return result
