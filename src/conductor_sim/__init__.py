"""Compose electrostatics problems without binding them to a particular solver."""

from .analytical import ConcentricShellBackend, ConcentricShellSolution
from .image_charge import ImageChargeShellBackend, ImageChargeShellSolution
from .multi_image import MultiImageShellBackend, MultiImageShellSolution
from .nested_shells import NestedShellBackend, NestedShellSolution
from .femm5 import Femm5Backend, Femm5ExecutionError, Femm5Settings, Femm5Unavailable
from .external_sphere import ExternalGroundedSphereBackend, ExternalGroundedSphereSolution
from .external_isolated import (ExternalIsolatedSphereBackend, ExternalIsolatedSphereSolution,
                                ExternalSphereBackend)
from .external_multi import ExternalMultiSphereBackend, ExternalMultiSphereSolution
from .femm5_external import Femm5ExternalSphereBackend, Femm5ExternalSphereSolution
from .model import (
    EPSILON_0,
    ElectrostaticProblem,
    FixedNetCharge,
    Grounded,
    PointCharge,
    SphericalShell,
    SolidSphere,
)
from .solver import ConductorResult, ElectrostaticBackend, FieldSolution, SingularField, UnsupportedProblem

__all__ = [
    "EPSILON_0",
    "Femm5Backend",
    "Femm5ExternalSphereBackend",
    "Femm5ExternalSphereSolution",
    "ExternalGroundedSphereBackend",
    "ExternalGroundedSphereSolution",
    "ExternalIsolatedSphereBackend",
    "ExternalIsolatedSphereSolution",
    "ExternalMultiSphereBackend",
    "ExternalMultiSphereSolution",
    "ExternalSphereBackend",
    "Femm5Settings",
    "Femm5Unavailable",
    "Femm5ExecutionError",
    "ElectrostaticProblem",
    "FixedNetCharge",
    "Grounded",
    "PointCharge",
    "SphericalShell",
    "SolidSphere",
    "ConcentricShellBackend",
    "ConcentricShellSolution",
    "ImageChargeShellBackend",
    "ImageChargeShellSolution",
    "MultiImageShellBackend",
    "MultiImageShellSolution",
    "NestedShellBackend",
    "NestedShellSolution",
    "ConductorResult",
    "ElectrostaticBackend",
    "FieldSolution",
    "SingularField",
    "UnsupportedProblem",
]
