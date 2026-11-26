"""Piecewise-homogeneous linear isotropic dielectric layers for radial and planar stacks."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import ulp
from typing import Literal

from .solver import UnsupportedProblem
from .spherical_conductor import _finite_si

__all__ = ["DielectricLayer", "DielectricInterfaceCharge", "DielectricLayerField"]

_ZERO = Fraction()


@dataclass(frozen=True, slots=True)
class DielectricLayer:
    """One homogeneous linear isotropic dielectric layer (relative permittivity kappa >= 1)."""

    thickness_m: float
    relative_permittivity: float

    def __post_init__(self) -> None:
        _finite_si(self.thickness_m, "dielectric layer thickness", positive=True)
        _finite_si(self.relative_permittivity, "relative permittivity", positive=True)
        if self.relative_permittivity < 1.0:
            raise UnsupportedProblem("linear isotropic dielectric requires relative_permittivity >= 1")
        object.__setattr__(self, "thickness_m", float(self.thickness_m))
        object.__setattr__(self, "relative_permittivity", float(self.relative_permittivity))


@dataclass(frozen=True, slots=True)
class DielectricLayerField:
    """Uniform displacement, field, and polarization in one planar dielectric sub-layer."""

    gap_index: int
    layer_index: int
    x_start_m: float
    x_end_m: float
    relative_permittivity: float
    displacement_c_per_m2: float
    electric_field_v_per_m: float
    polarization_c_per_m2: float


@dataclass(frozen=True, slots=True)
class DielectricInterfaceCharge:
    """Bound surface charge at a conductor/dielectric or dielectric/dielectric boundary.

    For planar stacks, coordinate_m is x in metres and total_bound_charge_c = sigma * area.
    For spherical stacks, coordinate_m is radius r in metres and total_bound_charge_c = sigma * 4*pi*r^2.
    """

    region: Literal["gap", "exterior"]
    gap_index: int | None
    coordinate_m: float
    kind: Literal["conductor_left", "conductor_right", "conductor_inner", "conductor_outer", "dielectric_interface"]
    left_kappa: float | None
    right_kappa: float | None
    sigma_bound_c_per_m2: float
    total_bound_charge_c: float


def _validated_gap_layers(
    gap_layers: tuple[tuple[DielectricLayer, ...], ...],
    spans: tuple[tuple[Fraction, Fraction], ...],
    label: str,
) -> tuple[tuple[tuple[Fraction, Fraction, Fraction], ...], ...]:
    """Validate ordered layer tuples against gap spans, returning (lo, hi, kappa) in Fraction."""
    if len(gap_layers) != len(spans):
        raise UnsupportedProblem(f"{label} expects one dielectric layer tuple per gap ({len(spans)})")
    validated = []
    for gap_index, (raw_layers, (start, end)) in enumerate(zip(gap_layers, spans)):
        if not isinstance(raw_layers, (tuple, list)) or not raw_layers:
            raise UnsupportedProblem(f"{label} gap {gap_index} requires one or more DielectricLayer entries")
        layers = tuple(raw_layers)
        if any(type(layer) is not DielectricLayer for layer in layers):
            raise UnsupportedProblem(f"{label} gap {gap_index} entries must be DielectricLayer objects")
        expected_width = end - start
        widths_bin = tuple(Fraction(layer.thickness_m) for layer in layers)
        total_bin = sum(widths_bin, _ZERO)
        if total_bin == expected_width:
            widths = widths_bin
        else:
            # Accept authored decimal spellings summing to the gap width (e.g. 0.001 + 0.002 == 0.003)
            # or binary64 input rounding of the exact span.
            widths_dec = tuple(Fraction(str(layer.thickness_m)) for layer in layers)
            span_dec = Fraction(str(float(end))) - Fraction(str(float(start)))
            if sum(widths_dec, _ZERO) == span_dec and abs(expected_width - span_dec) <= Fraction(4 * ulp(max(abs(float(start)), abs(float(end)), float(expected_width)))):
                # Reconcile final boundary exactly to `end` (at most a few ULPs) so geometry stays watertight.
                widths = widths_dec[:-1] + (expected_width - sum(widths_dec[:-1], _ZERO),)
            else:
                max_scale = max(float(expected_width), *(layer.thickness_m for layer in layers))
                if abs(total_bin - expected_width) <= Fraction((len(layers) + 2) * ulp(max_scale)):
                    widths = widths_bin[:-1] + (expected_width - sum(widths_bin[:-1], _ZERO),)
                else:
                    raise UnsupportedProblem(
                        f"{label} gap {gap_index} layer thicknesses must sum exactly to the gap width"
                    )
        if any(w <= 0 for w in widths):
            raise UnsupportedProblem(f"{label} gap {gap_index} has non-positive reconciled layer thickness")
        bounds = []
        cursor = start
        cursor_dec = Fraction(str(float(start)))
        for idx, (width, layer) in enumerate(zip(widths, layers)):
            cursor_dec += Fraction(str(layer.thickness_m))
            if idx == len(layers) - 1:
                nxt = end
            else:
                nxt_raw = cursor + width
                nxt_dec = Fraction(float(cursor_dec))
                if cursor < nxt_dec < end and abs(nxt_dec - nxt_raw) <= Fraction(4 * ulp(max(abs(float(cursor)), abs(float(nxt_raw))))):
                    nxt = nxt_dec
                else:
                    nxt = nxt_raw
            bounds.append((cursor, nxt, Fraction(layer.relative_permittivity)))
            cursor = nxt
        validated.append(tuple(bounds))
    return tuple(validated)


def _validated_exterior_layers(
    exterior_layers: tuple[DielectricLayer, ...],
    start: Fraction,
) -> tuple[tuple[Fraction, Fraction, Fraction], ...]:
    """Validate concentric finite dielectric coatings outside the outermost spherical conductor."""
    if not exterior_layers:
        return ()
    if any(type(layer) is not DielectricLayer for layer in exterior_layers):
        raise UnsupportedProblem("exterior_layers entries must be DielectricLayer objects")
    bounds = []
    cursor = start
    cursor_dec = Fraction(str(float(start)))
    for layer in exterior_layers:
        cursor_dec += Fraction(str(layer.thickness_m))
        nxt_raw = cursor + Fraction(layer.thickness_m)
        nxt_dec = Fraction(float(cursor_dec))
        if nxt_dec > cursor and abs(nxt_dec - nxt_raw) <= Fraction(4 * ulp(max(abs(float(cursor)), abs(float(nxt_raw))))):
            nxt = nxt_dec
        else:
            nxt = nxt_raw
        bounds.append((cursor, nxt, Fraction(layer.relative_permittivity)))
        cursor = nxt
    return tuple(bounds)
