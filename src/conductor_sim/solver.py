"""Solver and solution contracts, independent of numerical libraries."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
from .model import ElectrostaticProblem, Vector3

class UnsupportedProblem(ValueError):
    """The chosen backend cannot solve the requested arrangement."""

class SingularField(ValueError):
    """Field sampling is undefined at a singular source or jump."""

@dataclass(frozen=True, slots=True)
class ConductorResult:
    name: str
    potential_v: float
    # None means the ideal infinite-conductor total is undefined/unbounded.
    # Never substitute a finite number or NaN for that physical limit.
    net_charge_c: float | None
    inner_surface_charge_c: float | None
    outer_surface_charge_c: float | None

class FieldSolution(Protocol):
    @property
    def conductor_results(self) -> tuple[ConductorResult, ...]: ...
    def potential_at(self, point_m: Vector3) -> float: ...
    def electric_field_at(self, point_m: Vector3) -> Vector3: ...

class ElectrostaticBackend(Protocol):
    def supports(self, problem: ElectrostaticProblem) -> bool: ...
    def solve(self, problem: ElectrostaticProblem) -> FieldSolution: ...
