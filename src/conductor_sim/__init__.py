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

from .motion_2d import (Motion2DSpec, Motion2DSample, Motion2DEvent,
                        Motion2DResult, integrate_motion_2d)
__all__ += ["Motion2DSpec", "Motion2DSample", "Motion2DEvent",
            "Motion2DResult", "integrate_motion_2d"]

from .motion_viewer_2d import sample_motion_2d, motion_slice_2d
__all__ += ['sample_motion_2d', 'motion_slice_2d']

from .prescribed_fields import (ElectricFieldProvider, PrescribedField,
    UniformElectricField, FixedPointField, UniformRingAxisField, UniformRingField,
    UniformDiskAxisField, UniformRodField, UniformSphereField, UniformShellField)
from .prescribed_motion import (TestChargeMotionSpec, TestChargeSample,
    TestChargeEvent, TestChargeMotionResult, integrate_test_charge)
__all__ += ['ElectricFieldProvider', 'PrescribedField', 'UniformElectricField',
    'FixedPointField', 'UniformRingAxisField', 'UniformRingField', 'UniformDiskAxisField',
    'UniformRodField', 'UniformSphereField', 'UniformShellField',
    'TestChargeMotionSpec', 'TestChargeSample', 'TestChargeEvent',
    'TestChargeMotionResult', 'integrate_test_charge']

from .prescribed_viewer import sample_test_charge_viewer, test_charge_slice
__all__ += ['sample_test_charge_viewer','test_charge_slice']

from .scenario_comparison import (prepare_scenario_comparison,
    render_scenario_comparison, write_scenario_comparison)
__all__ += ['prepare_scenario_comparison','render_scenario_comparison',
            'write_scenario_comparison']

from .conservative_forces import UniformGravity, PrescribedElectricForce, ConservativeForceSystem
__all__ += ['UniformGravity', 'PrescribedElectricForce', 'ConservativeForceSystem']

from .trajectory_sweep import (SweepAxis, TrajectorySweepPlan, SweepEntry,
                               TrajectorySweepRun, run_trajectory_sweep)
from .trajectory_sweep_viewer import (render_trajectory_sweep,
                                      write_trajectory_sweep_html)
__all__ += ["SweepAxis", "TrajectorySweepPlan", "SweepEntry",
            "TrajectorySweepRun", "run_trajectory_sweep",
            "render_trajectory_sweep", "write_trajectory_sweep_html"]

from .event_sweep import (EventSweepPlan, EventSweepEntry,
                          EventSweepRun, run_event_sweep)
from .event_sweep_viewer import render_event_sweep, write_event_sweep_html
__all__ += ["EventSweepPlan", "EventSweepEntry", "EventSweepRun",
            "run_event_sweep", "render_event_sweep", "write_event_sweep_html"]

from .structural_family import (StructuralCase, StructuralFamilyPlan,
                                StructuralFamilyEntry, StructuralFamilyRun,
                                run_structural_family)
from .structural_family_viewer import (render_structural_family,
                                       write_structural_family_html)
__all__ += ["StructuralCase", "StructuralFamilyPlan", "StructuralFamilyEntry",
            "StructuralFamilyRun", "run_structural_family",
            "render_structural_family", "write_structural_family_html"]

from .structural_event_family import (
    StructuralEventFamilyPlan, StructuralEventEntry,
    StructuralEventFamilyRun, run_structural_event_family,
)
from .structural_event_viewer import (
    render_structural_event_family, write_structural_event_family_html,
)
__all__ += ["StructuralEventFamilyPlan", "StructuralEventEntry",
            "StructuralEventFamilyRun", "run_structural_event_family",
            "render_structural_event_family", "write_structural_event_family_html"]

from .structural_parameter_family import (
    BranchSweep, StructuralParameterPlan, BranchSweepEntry,
    StructuralParameterRun, run_structural_parameter_family,
)
__all__ += ["BranchSweep", "StructuralParameterPlan", "BranchSweepEntry",
            "StructuralParameterRun", "run_structural_parameter_family"]

from .nested_event_family import (
    NestedEventPlan, NestedVariationEvent, NestedBranchEvent,
    NestedEventRun, run_nested_event_family,
)
__all__ += ["NestedEventPlan", "NestedVariationEvent", "NestedBranchEvent",
            "NestedEventRun", "run_nested_event_family"]

from .coupled_physical_family import (
    CoupledPropertyEdit, CoupledHypothesis, CoupledPhysicalPlan,
    CoupledPhysicalEntry, CoupledPhysicalRun, run_coupled_physical_family,
)
__all__ += ["CoupledPropertyEdit", "CoupledHypothesis", "CoupledPhysicalPlan",
            "CoupledPhysicalEntry", "CoupledPhysicalRun", "run_coupled_physical_family"]

from .coupled_event_family import (
    CoupledEventPlan, CoupledParentEvent, CoupledChildEvent,
    CoupledEventRun, run_coupled_event_family,
)
__all__ += ["CoupledEventPlan", "CoupledParentEvent", "CoupledChildEvent",
            "CoupledEventRun", "run_coupled_event_family"]

from .unified_observation_graph import (
    StructuralWorldNode, PhysicalWorldNode, WorldComparison, UnifiedObservationPlan,
    UnifiedObservationEntry, UnifiedObservationRun,
    run_unified_observation_graph,
)
__all__ += ["StructuralWorldNode", "PhysicalWorldNode", "WorldComparison", "UnifiedObservationPlan",
            "UnifiedObservationEntry", "UnifiedObservationRun",
            "run_unified_observation_graph"]

from .source_world_ancestry import SourceWorldAncestry
__all__ += ["SourceWorldAncestry"]

from .graph_event_inquiry import (
    GraphEventInquiryPlan, GraphEventInquiryEntry, GraphEventInquiryRun,
    run_graph_event_inquiry,
)
__all__ += [
    "GraphEventInquiryPlan", "GraphEventInquiryEntry", "GraphEventInquiryRun",
    "run_graph_event_inquiry",
]

from .graph_world_fork import (
    GraphWorldForkPlan, GraphWorldForkEntry, GraphWorldForkRun,
    run_graph_world_fork,
)
__all__ += [
    "GraphWorldForkPlan", "GraphWorldForkEntry", "GraphWorldForkRun",
    "run_graph_world_fork",
]


from .grounded_sphere_wire import (
    GroundedSphereWireObservation, GroundedSphereWireTrace,
    observe_grounded_sphere_wire, observe_grounded_sphere_motion,
)
__all__ += [
    "GroundedSphereWireObservation", "GroundedSphereWireTrace",
    "observe_grounded_sphere_wire", "observe_grounded_sphere_motion",
]


from .prescribed_ground_wire import (
    PrescribedSphereRadialSpec, PrescribedSphereSample, PrescribedSphereTrace,
    sample_prescribed_sphere_radial, author_prescribed_grounded_sphere,
)
__all__ += [
    "author_prescribed_grounded_sphere", "PrescribedSphereRadialSpec", "PrescribedSphereSample", "PrescribedSphereTrace",
    "sample_prescribed_sphere_radial",
]


from .conducting_drops import ConductingDropEquilibria, combine_isolated_drops
__all__ += ["ConductingDropEquilibria", "combine_isolated_drops"]


from .point_charge_work import PointChargeRearrangement, rearrange_point_charges
__all__ += ["PointChargeRearrangement", "rearrange_point_charges"]

from .radial_voltage import prepare_radial_voltages
__all__ += ["prepare_radial_voltages"]

from .coaxial_arcs import CircularArcCharge, CoaxialArcAssembly
__all__ += ["CircularArcCharge", "CoaxialArcAssembly"]
