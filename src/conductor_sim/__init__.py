"""Compositional electrostatics models and analytical reference backend."""
from .analytical import ConcentricShellBackend, ConcentricShellSolution
from .model import EPSILON_0, ElectrostaticProblem, FixedNetCharge, Grounded, PointCharge, SphericalShell
from .solver import ConductorResult, ElectrostaticBackend, FieldSolution, SingularField, UnsupportedProblem

__all__ = [
    "EPSILON_0", "ElectrostaticProblem", "FixedNetCharge", "Grounded",
    "PointCharge", "SphericalShell", "ConcentricShellBackend",
    "ConcentricShellSolution", "ConductorResult", "ElectrostaticBackend",
    "FieldSolution", "SingularField", "UnsupportedProblem",
]
