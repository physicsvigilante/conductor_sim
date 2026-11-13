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
from .grounded_plane import GroundedPlaneBackend, GroundedPlaneSolution
from .parallel_plates import ParallelPlateBackend, ParallelPlateSolution
from .femm5_parallel import Femm5ParallelPlateBackend, Femm5ParallelPlateSolution
from .femm5_plane import Femm5PlaneBackend, Femm5PlaneSolution
from .model import (
    EPSILON_0,
    ElectrostaticProblem,
    ConductingPlane,
    FixedNetCharge,
    FixedPotential,
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
    "FixedPotential",
    "ParallelPlateBackend",
    "ParallelPlateSolution",
    "Femm5ParallelPlateBackend",
    "Femm5ParallelPlateSolution",
    "Grounded",
    "ConductingPlane",
    "GroundedPlaneBackend",
    "GroundedPlaneSolution",
    "Femm5PlaneBackend",
    "Femm5PlaneSolution",
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

from .observables import (ChargeForce, EnergyBreakdown, ForceEnergyResult,
                          ForceEnergyProvider, force_energy)
__all__ += ["ChargeForce", "EnergyBreakdown", "ForceEnergyResult", "ForceEnergyProvider", "force_energy"]

from .position_sweep import (PositionSweepSpec, sample_position_sweep, sweep_slice)
__all__ += ["PositionSweepSpec", "sample_position_sweep", "sweep_slice"]

from .connections import (ElectricalLink, ConnectionPlan, ConductorTransfer,
                          ConnectionResult, ConnectedShellBackend)
__all__ += ['ElectricalLink', 'ConnectionPlan', 'ConductorTransfer',
            'ConnectionResult', 'ConnectedShellBackend']

from .mixed_radial import MixedRadialBackend, MixedRadialSolution
__all__ += ["MixedRadialBackend", "MixedRadialSolution"]

from .motion_1d import (Motion1DSpec, MotionSample, MotionEvent, Motion1DResult,
                        MotionIntegrationError, integrate_motion_1d)
__all__ += ["Motion1DSpec", "MotionSample", "MotionEvent", "Motion1DResult",
            "MotionIntegrationError", "integrate_motion_1d"]

from .motion_viewer import sample_motion_1d, motion_slice
__all__ += ['sample_motion_1d', 'motion_slice']
