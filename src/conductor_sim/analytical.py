"""Exact spherical-symmetry reference, not a general 3D or FEMM solver."""
from __future__ import annotations
from dataclasses import dataclass
from math import isclose, pi, sqrt
from .model import ElectrostaticProblem, FixedNetCharge, SphericalShell, Vector3
from .model import _position
from .solver import ConductorResult, SingularField, UnsupportedProblem

def _distance_vector(a: Vector3, b: Vector3) -> tuple[Vector3, float]:
    delta = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    return delta, sqrt(sum(v*v for v in delta))

@dataclass(frozen=True, slots=True)
class ConcentricShellSolution:
    shell: SphericalShell
    charge_c: float
    permittivity_f_per_m: float
    conductor_results: tuple[ConductorResult, ...]
    @property
    def _k(self) -> float:
        return 1 / (4*pi*self.permittivity_f_per_m)
    @property
    def _outer_charge_c(self) -> float:
        return self.conductor_results[0].outer_surface_charge_c
    @property
    def _conductor_potential_v(self) -> float:
        return self.conductor_results[0].potential_v
    def potential_at(self, point_m: Vector3) -> float:
        point_m = _position(point_m, "sample point")
        _, r = _distance_vector(point_m, self.shell.center_m)
        if r == 0 and self.charge_c != 0:
            raise SingularField("potential is singular at point charge")
        if r < self.shell.inner_radius_m:
            if self.charge_c == 0:
                return self._conductor_potential_v
            return self._conductor_potential_v + self._k * self.charge_c * (1/r - 1/self.shell.inner_radius_m)
        if r <= self.shell.outer_radius_m:
            return self._conductor_potential_v
        return self._k*self._outer_charge_c/r
    def electric_field_at(self, point_m: Vector3) -> Vector3:
        point_m = _position(point_m, "sample point")
        delta, r = _distance_vector(point_m, self.shell.center_m)
        if r == 0:
            if self.charge_c:
                raise SingularField("electric field is singular at point charge")
            return (0.0, 0.0, 0.0)
        if isclose(r, self.shell.inner_radius_m, rel_tol=1e-12) or isclose(r, self.shell.outer_radius_m, rel_tol=1e-12):
            raise SingularField("electric field has a jump at conductor surface")
        if r < self.shell.inner_radius_m:
            q = self.charge_c
        elif r < self.shell.outer_radius_m:
            q = 0.0
        else:
            q = self._outer_charge_c
        factor = self._k*q/(r*r*r)
        return tuple(factor*v for v in delta)

class ConcentricShellBackend:
    """Only one shell with every charge exactly at its center."""
    def supports(self, problem: ElectrostaticProblem) -> bool:
        if len(problem.conductors) != 1:
            return False
        shell = problem.conductors[0]
        return all(c.position_m == shell.center_m for c in problem.charges)
    def solve(self, problem: ElectrostaticProblem) -> ConcentricShellSolution:
        if not self.supports(problem):
            raise UnsupportedProblem("requires one spherical shell with charges at its center")
        shell = problem.conductors[0]
        q = sum(c.charge_c for c in problem.charges)
        q_net = shell.constraint.charge_c if isinstance(shell.constraint, FixedNetCharge) else -q
        q_outer = q + q_net
        v = q_outer/(4*pi*problem.permittivity_f_per_m*shell.outer_radius_m)
        data = ConductorResult(shell.name, v, q_net, -q, q_outer)
        return ConcentricShellSolution(shell, q, problem.permittivity_f_per_m, (data,))
