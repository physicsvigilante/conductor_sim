"""Fringe-free planar conductor stacks with zero exterior displacement, SI."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import KW_ONLY, dataclass, field
from fractions import Fraction

from .model import EPSILON_0
from .solver import SingularField, UnsupportedProblem
from .spherical_conductor import _finite_si, _identity, _representable
from .stack_coupling import (
    ConductorGroup, PlanarVoltageReference, _Coupling, _Gap, _State, _solve_coupling,
)

__all__ = ["PlanarStackProblem", "PlanarPlateResult", "PlanarStackSolution", "solve_planar_stack"]


@dataclass(frozen=True, slots=True)
class PlanarStackProblem:
    """Ordered ideal plates normal to the x axis, with common reference area.

    Positions set the gap thicknesses. Exterior D must be zero on both sides.
    Prescribe a terminal potential or set a voltage_reference for a neutral
    charge-constrained stack. The reference is a gauge and exchanges no charge.
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
    """Plate and gap observables ordered in the positive x direction."""

    problem: PlanarStackProblem
    plate_results: tuple[PlanarPlateResult, ...]
    gap_displacements_c_per_m2: tuple[float, ...]
    gap_fields_v_per_m: tuple[float, ...]
    _state: _State = field(repr=False)
    _gap_fields: tuple[Fraction, ...] = field(repr=False)

    @property
    def capacitance_matrix_f(self) -> tuple[tuple[float, ...], ...]:
        """Maxwell matrix in plate order, with a uniform-voltage nullspace."""
        return self._state.coupling.capacitance_matrix_f

    @property
    def field_energy_j(self) -> float:
        return _representable(self._state.energy, "field energy")

    def potential_at(self, x_m: float) -> float:
        """Voltage at x, including each plate and the constant-potential exteriors."""
        _finite_si(x_m, "sample x")
        positions = self.problem.plate_positions_m
        i = bisect_left(positions, x_m)
        if i == 0:
            return self.plate_results[0].potential_v
        if i == len(positions):
            return self.plate_results[-1].potential_v
        distance = Fraction(x_m) - Fraction(positions[i - 1])
        return _representable(self._state.voltages[i - 1] - distance * self._gap_fields[i - 1],
                              "planar potential")

    def electric_field_at(self, x_m: float) -> float:
        """Signed x field in volts per meter; a plate requires a one-sided limit."""
        _finite_si(x_m, "sample x")
        positions = self.problem.plate_positions_m
        i = bisect_left(positions, x_m)
        if i < len(positions) and x_m == positions[i]:
            raise SingularField("electric field requires a side at a plate")
        return self.gap_fields_v_per_m[i - 1] if 0 < i < len(positions) else 0.0


def _solve_planar_stack(
    problem: PlanarStackProblem, initial_charges: tuple[Fraction, ...] | None = None,
) -> PlanarStackSolution:
    positions = tuple(map(Fraction, problem.plate_positions_m))
    area = Fraction(problem.area_m2)
    epsilon = Fraction(problem.permittivity_f_per_m)
    coupling = _Coupling(tuple(_Gap((right - left) / (epsilon * area))
                               for left, right in zip(positions, positions[1:])))
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
    displacement = tuple(q / area for q in state.right_faces[:-1])
    fields = tuple(d / epsilon for d in displacement)
    return PlanarStackSolution(problem, results,
                               tuple(_representable(d, "gap displacement") for d in displacement),
                               tuple(_representable(e, "gap field") for e in fields), state, fields)


def solve_planar_stack(problem: PlanarStackProblem) -> PlanarStackSolution:
    """Solve a homogeneous planar stack in the one-dimensional idealization."""
    if type(problem) is not PlanarStackProblem:
        raise UnsupportedProblem("expected a PlanarStackProblem")
    return _solve_planar_stack(problem)
