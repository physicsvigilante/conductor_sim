"""Fringe-free planar conductor stacks with zero exterior displacement, SI."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import KW_ONLY, dataclass, field
from fractions import Fraction

from .dielectric_layers import (
    DielectricInterfaceCharge,
    DielectricLayer,
    DielectricLayerField,
    _validated_gap_layers,
)
from .model import EPSILON_0
from .solver import SingularField, UnsupportedProblem
from .spherical_conductor import _finite_si, _identity, _representable
from .stack_coupling import (
    ConductorGroup, PlanarVoltageReference, _Coupling, _Gap, _State, _solve_coupling,
)

__all__ = ["PlanarStackProblem", "PlanarPlateResult", "PlanarStackSolution", "solve_planar_stack"]

_ZERO = Fraction()


@dataclass(frozen=True, slots=True)
class PlanarStackProblem:
    """Ordered ideal plates normal to the x axis, with common reference area.

    Positions set the gap thicknesses. Exterior D must be zero on both sides.
    Prescribe a terminal potential or set a voltage_reference for a neutral
    charge-constrained stack. The reference is a gauge and exchanges no charge.
    Optionally supply gap_layers (one ordered tuple of DielectricLayer per gap,
    relative to vacuum EPSILON_0) for piecewise-homogeneous layered dielectrics.
    """

    plate_ids: tuple[str, ...]
    plate_positions_m: tuple[float, ...]
    area_m2: float
    groups: tuple[ConductorGroup, ...]
    _: KW_ONLY
    d_left_c_per_m2: float
    d_right_c_per_m2: float
    permittivity_f_per_m: float = EPSILON_0
    voltage_reference: PlanarVoltageReference | None = None
    gap_layers: tuple[tuple[DielectricLayer, ...], ...] | None = None

    def __post_init__(self) -> None:
        _finite_si(self.area_m2, "reference area", positive=True)
        _finite_si(self.permittivity_f_per_m, "permittivity", positive=True)
        for boundary in (self.d_left_c_per_m2, self.d_right_c_per_m2):
            _finite_si(boundary, "exterior displacement")
            if boundary != 0:
                raise UnsupportedProblem("planar stack requires zero exterior displacement on both sides")
        if isinstance(self.plate_ids, str):
            raise ValueError("plate IDs must be a sequence of IDs")
        ids, positions = tuple(self.plate_ids), tuple(self.plate_positions_m)
        if len(ids) < 2 or len(ids) != len(positions):
            raise ValueError("expected at least two plates, each with an x position")
        for name in ids:
            _identity(name, "plate ID")
        if len(set(ids)) != len(ids):
            raise ValueError("plate IDs must be unique")
        for position in positions:
            _finite_si(position, "plate position")
        if any(left >= right for left, right in zip(positions, positions[1:])):
            raise ValueError("plate positions must be strictly increasing")
        if self.voltage_reference is not None and type(self.voltage_reference) is not PlanarVoltageReference:
            raise UnsupportedProblem("expected a PlanarVoltageReference")
        if self.gap_layers is not None:
            if self.permittivity_f_per_m != EPSILON_0:
                raise UnsupportedProblem(
                    "planar gap_layers are defined relative to vacuum EPSILON_0; "
                    "omit non-vacuum permittivity_f_per_m when specifying gap_layers"
                )
            if not isinstance(self.gap_layers, (tuple, list)):
                raise UnsupportedProblem("gap_layers must be a sequence of per-gap DielectricLayer tuples")
            normalized = tuple(
                tuple(row) if isinstance(row, (tuple, list)) else row for row in self.gap_layers
            )
            spans = tuple((Fraction(a), Fraction(b)) for a, b in zip(positions, positions[1:]))
            _validated_gap_layers(normalized, spans, "planar stack")
            object.__setattr__(self, "gap_layers", normalized)
        object.__setattr__(self, "plate_ids", ids)
        object.__setattr__(self, "plate_positions_m", positions)
        object.__setattr__(self, "groups", tuple(self.groups))


@dataclass(frozen=True, slots=True)
class PlanarPlateResult:
    plate_id: str
    potential_v: float
    net_charge_c: float
    left_face_charge_c: float
    right_face_charge_c: float
    left_sigma_c_per_m2: float
    right_sigma_c_per_m2: float


@dataclass(frozen=True, slots=True)
class PlanarStackSolution:
    """Plate, gap, layer-field, and dielectric-interface observables ordered along +x."""

    problem: PlanarStackProblem
    plate_results: tuple[PlanarPlateResult, ...]
    gap_displacements_c_per_m2: tuple[float, ...]
    gap_fields_v_per_m: tuple[float | None, ...]
    gap_layer_fields: tuple[DielectricLayerField, ...]
    dielectric_interfaces: tuple[DielectricInterfaceCharge, ...]
    _state: _State = field(repr=False)
    _gap_sublayers: tuple[tuple[tuple[Fraction, Fraction, Fraction, Fraction], ...], ...] = field(repr=False)

    @property
    def capacitance_matrix_f(self) -> tuple[tuple[float, ...], ...]:
        """Maxwell matrix in plate order, with a uniform-voltage nullspace."""
        return self._state.coupling.capacitance_matrix_f

    @property
    def field_energy_j(self) -> float:
        return _representable(self._state.energy, "field energy")

    def potential_at(self, x_m: float) -> float:
        """Voltage at x, continuous across every plate and dielectric interface."""
        _finite_si(x_m, "sample x")
        positions = self.problem.plate_positions_m
        i = bisect_left(positions, x_m)
        if i == 0:
            return self.plate_results[0].potential_v
        if i == len(positions):
            return self.plate_results[-1].potential_v
        if x_m == positions[i]:
            return self.plate_results[i].potential_v
        x = Fraction(x_m)
        voltage = self._state.voltages[i - 1]
        for lo, hi, _kappa, field_v_m in self._gap_sublayers[i - 1]:
            if x <= hi:
                return _representable(voltage - (x - lo) * field_v_m, "planar potential")
            voltage -= (hi - lo) * field_v_m
        return self.plate_results[i].potential_v

    def electric_field_at(self, x_m: float) -> float:
        """Signed x field in V/m; plates and kappa-discontinuous interfaces raise SingularField."""
        _finite_si(x_m, "sample x")
        positions = self.problem.plate_positions_m
        i = bisect_left(positions, x_m)
        if i < len(positions) and x_m == positions[i]:
            raise SingularField("electric field requires a side at a plate")
        if i == 0 or i == len(positions):
            return 0.0
        x = Fraction(x_m)
        layers = self._gap_sublayers[i - 1]
        for idx, (lo, hi, kappa, field_v_m) in enumerate(layers):
            if x < hi:
                return _representable(field_v_m, "planar electric field")
            if x == hi and idx + 1 < len(layers):
                next_kappa = layers[idx + 1][2]
                if kappa != next_kappa:
                    raise SingularField("electric field has a jump discontinuity at a dielectric interface")
                return _representable(field_v_m, "planar electric field")
        return _representable(layers[-1][3], "planar electric field")


def _solve_planar_stack(
    problem: PlanarStackProblem, initial_charges: tuple[Fraction, ...] | None = None,
) -> PlanarStackSolution:
    positions = tuple(map(Fraction, problem.plate_positions_m))
    area = Fraction(problem.area_m2)
    eps0 = Fraction(EPSILON_0)
    spans = tuple((left, right) for left, right in zip(positions, positions[1:]))

    if problem.gap_layers is None:
        epsilon = Fraction(problem.permittivity_f_per_m)
        kappa_uniform = epsilon / eps0
        validated_gaps = tuple(((lo, hi, kappa_uniform),) for lo, hi in spans)
        gap_elastances = tuple((hi - lo) / (epsilon * area) for lo, hi in spans)
    else:
        validated_gaps = _validated_gap_layers(problem.gap_layers, spans, "planar stack")
        gap_elastances = tuple(
            sum(((hi - lo) / (eps0 * kappa * area) for lo, hi, kappa in layers), _ZERO)
            for layers in validated_gaps
        )

    coupling = _Coupling(tuple(_Gap(elastance) for elastance in gap_elastances))
    state = _solve_coupling(coupling, problem.plate_ids, problem.groups,
                            reference=problem.voltage_reference, initial_charges=initial_charges)
    results = tuple(PlanarPlateResult(name, _representable(v, "plate potential"),
                                      _representable(q, "plate charge"),
                                      _representable(a, "left-face charge"),
                                      _representable(b, "right-face charge"),
                                      _representable(a / area, "left surface density"),
                                      _representable(b / area, "right surface density"))
                    for name, v, q, a, b in zip(problem.plate_ids, state.voltages, state.charges,
                                                state.left_faces, state.right_faces))
    displacements = tuple(q / area for q in state.right_faces[:-1])

    gap_fields_out: list[float | None] = []
    layer_fields: list[DielectricLayerField] = []
    interfaces: list[DielectricInterfaceCharge] = []
    gap_sublayers: list[tuple[tuple[Fraction, Fraction, Fraction, Fraction], ...]] = []

    for g, (d_gap, layers) in enumerate(zip(displacements, validated_gaps)):
        kappas = {kappa for _, _, kappa in layers}
        sub = []
        for idx, (lo, hi, kappa) in enumerate(layers):
            eps_layer = (Fraction(problem.permittivity_f_per_m) if problem.gap_layers is None else eps0 * kappa)
            e_layer = d_gap / eps_layer
            sub.append((lo, hi, kappa, e_layer))
            if problem.gap_layers is not None:
                p_layer = (1 - eps0 / eps_layer) * d_gap
                layer_fields.append(DielectricLayerField(
                    g, idx,
                    _representable(lo, "layer start"),
                    _representable(hi, "layer end"),
                    _representable(kappa, "relative permittivity"),
                    _representable(d_gap, "gap displacement"),
                    _representable(e_layer, "layer electric field"),
                    _representable(p_layer, "layer polarization"),
                ))
        gap_sublayers.append(tuple(sub))
        if len(kappas) == 1:
            gap_fields_out.append(_representable(sub[0][3], "gap field"))
        else:
            gap_fields_out.append(None)

        if problem.gap_layers is not None:
            k0 = layers[0][2]
            sigma_left = -(1 - 1 / k0) * d_gap
            interfaces.append(DielectricInterfaceCharge(
                "gap", g, float(layers[0][0]), "conductor_left", None, float(k0),
                _representable(sigma_left, "bound surface density"),
                _representable(sigma_left * area, "bound face charge"),
            ))
            for idx in range(len(layers) - 1):
                k1, k2 = layers[idx][2], layers[idx + 1][2]
                sigma_if = (1 / k2 - 1 / k1) * d_gap
                interfaces.append(DielectricInterfaceCharge(
                    "gap", g, float(layers[idx][1]), "dielectric_interface", float(k1), float(k2),
                    _representable(sigma_if, "bound surface density"),
                    _representable(sigma_if * area, "bound interface charge"),
                ))
            kl = layers[-1][2]
            sigma_right = (1 - 1 / kl) * d_gap
            interfaces.append(DielectricInterfaceCharge(
                "gap", g, float(layers[-1][1]), "conductor_right", float(kl), None,
                _representable(sigma_right, "bound surface density"),
                _representable(sigma_right * area, "bound face charge"),
            ))

    return PlanarStackSolution(
        problem, results,
        tuple(_representable(d, "gap displacement") for d in displacements),
        tuple(gap_fields_out),
        tuple(layer_fields),
        tuple(interfaces),
        state,
        tuple(gap_sublayers),
    )


def solve_planar_stack(problem: PlanarStackProblem) -> PlanarStackSolution:
    """Solve a homogeneous or piecewise-layered planar stack in the 1D idealization."""
    if type(problem) is not PlanarStackProblem:
        raise UnsupportedProblem("expected a PlanarStackProblem")
    return _solve_planar_stack(problem)
