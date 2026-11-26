"""Piecewise-homogeneous linear isotropic dielectric layers for radial and planar stacks."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isfinite, ulp
from typing import Literal

from .solver import UnsupportedProblem
from .spherical_conductor import _finite_si, _representable

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


def _width_ulp(expected_width: Fraction, layers: tuple[DielectricLayer, ...]) -> Fraction:
    """One ULP at the scale of the gap width, safe even when expected_width > DBL_MAX."""
    try:
        scale = max(float(expected_width), *(layer.thickness_m for layer in layers))
        return Fraction(ulp(scale))
    except OverflowError:
        # start and end are finite floats, so (end - start) / 2 <= DBL_MAX always fits in binary64.
        half_scale = max(float(expected_width / 2), *(layer.thickness_m for layer in layers))
        return 2 * Fraction(ulp(half_scale))


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
            # or binary64 input rounding of the exact span, without ever inflating a thin layer.
            w_ulp = _width_ulp(expected_width, layers)
            end_ulp = Fraction(ulp(max(abs(float(start)), abs(float(end)))))
            widths_dec = tuple(Fraction(str(layer.thickness_m)) for layer in layers)
            span_dec = Fraction(str(float(end))) - Fraction(str(float(start)))
            if (
                sum(widths_dec, _ZERO) == span_dec
                and abs(expected_width - span_dec) <= 4 * max(w_ulp, end_ulp)
                and abs(expected_width - span_dec) * 10**12 <= min(expected_width, widths_dec[-1])
            ):
                widths = widths_dec[:-1] + (expected_width - sum(widths_dec[:-1], _ZERO),)
            elif abs(total_bin - expected_width) <= (len(layers) + 2) * w_ulp:
                widths = widths_bin[:-1] + (expected_width - sum(widths_bin[:-1], _ZERO),)
            else:
                raise UnsupportedProblem(
                    f"{label} gap {gap_index} layer thicknesses must sum exactly to the gap width"
                )
        if any(w <= 0 for w in widths):
            raise UnsupportedProblem(f"{label} gap {gap_index} has non-positive reconciled layer thickness")
        # Build strictly increasing unsnapped boundaries first so snapping can never drift past `end`.
        raw_points = [start]
        cursor = start
        for idx, width in enumerate(widths):
            cursor = end if idx == len(widths) - 1 else cursor + width
            raw_points.append(cursor)

        points = [start]
        cursor_dec = Fraction(str(float(start)))
        for idx, layer in enumerate(layers):
            cursor_dec += Fraction(str(layer.thickness_m))
            if idx == len(layers) - 1:
                nxt = end
            else:
                nxt_raw = raw_points[idx + 1]
                upper = raw_points[idx + 2]
                nxt = nxt_raw
                try:
                    dec_f = float(cursor_dec)
                except OverflowError:
                    dec_f = float("inf")
                if isfinite(dec_f) and Fraction(str(dec_f)) == cursor_dec:
                    nxt_dec = Fraction(dec_f)
                    min_adj = min(layer.thickness_m, layers[idx + 1].thickness_m)
                    tol = Fraction(4 * ulp(min_adj))
                    if points[-1] < nxt_dec < upper and abs(nxt_dec - nxt_raw) <= tol:
                        nxt = nxt_dec
            points.append(nxt)

        bounds = tuple(
            (points[idx], points[idx + 1], Fraction(layer.relative_permittivity))
            for idx, layer in enumerate(layers)
        )
        if bounds[0][0] != start or bounds[-1][1] != end or any(lo >= hi for lo, hi, _ in bounds):
            raise UnsupportedProblem(f"{label} gap {gap_index} has invalid layer boundary ordering")
        validated.append(bounds)
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
        _representable(nxt_raw, "exterior dielectric radius")
        nxt = nxt_raw
        try:
            dec_f = float(cursor_dec)
        except OverflowError:
            dec_f = float("inf")
        if isfinite(dec_f) and Fraction(str(dec_f)) == cursor_dec:
            nxt_dec = Fraction(dec_f)
            tol = Fraction(4 * ulp(layer.thickness_m))
            if nxt_dec > cursor and abs(nxt_dec - nxt_raw) <= tol:
                nxt = nxt_dec
        if nxt <= cursor:
            raise UnsupportedProblem("exterior dielectric layer thickness is not positive")
        bounds.append((cursor, nxt, Fraction(layer.relative_permittivity)))
        cursor = nxt
    return tuple(bounds)
