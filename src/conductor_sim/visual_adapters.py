"""Optional visual metadata adapters; the sampling/tracing engine knows no solver."""
from math import acos, fsum, hypot, pi

from .model import ConductingPlane
from .grounded_plane import GroundedPlaneSolution
from .parallel_plates import ParallelPlateSolution
from .analytical import ConcentricShellSolution
from .image_charge import ImageChargeShellSolution
from .multi_image import MultiImageShellSolution
from .nested_shells import NestedShellSolution
from .femm5 import Femm5Solution
from .femm5_external import Femm5ExternalSphereSolution
from .femm5_parallel import Femm5ParallelPlateSolution
from .external_sphere import ExternalGroundedSphereSolution
from .external_isolated import ExternalIsolatedSphereSolution
from .external_multi import ExternalMultiSphereSolution
from .visualization import SamplingContext, SourceMask, CylindricalDomain, dot, sub


def context_for_solution(problem, solution):
    """Describe masks, valid domain and density API without inventing field values.

    Extensible solvers can instantiate SamplingContext directly. The caller
    retains ownership of a supplied solution; only sample_sweep owns/closes it.
    """
    density = None
    method = 'unavailable'
    radius = None
    cylinder = None
    conductor = problem.conductors[0] if problem.conductors else None
    center = (conductor.point_m if isinstance(conductor,ConductingPlane) else
              conductor.center_m if conductor is not None else (0.,0.,0.))
    notes = ['Field lines are geometry traces, not charge trajectories; line count is not flux.',
             'Profiles follow a great circle on spheres or a signed line on planes; neither is automatically a surface integral.',
             'The colour scale is shared across frames; display clipping does not alter exported values.']
    masks = []
    conductor_method = 'Supplied solver conductor properties; provenance not specified.'
    from .femm5_plane import Femm5PlaneSolution
    if isinstance(solution, (Femm5Solution,Femm5ExternalSphereSolution,
                             Femm5PlaneSolution,Femm5ParallelPlateSolution)):
        if isinstance(solution,Femm5ParallelPlateSolution):
            conductor_method = ('FEMM measured electrode Q/V on two finite disks, '
                                'with an insulated outer cylindrical side; not infinite-plate totals.')
            center = solution._origin_m
            cylinder = CylindricalDomain(solution._origin_m,solution.symmetry_axis,
                                         solution.outer_radius_m,0.,solution.gap_m)
        elif isinstance(solution,Femm5PlaneSolution):
            conductor_method = ('FEMM fixed-voltage finite plane boundary; total charge '
                                'measured on its truncated conductor segment, not exact infinite-plane Q.')
            # FEMM's finite upper-hemisphere is centered on the common
            # charge footpoint, not the arbitrary authored plane anchor.
            center = solution._origin_m
        else:
            conductor_method = ('FEMM net charge and voltage; solid sphere outer charge is measured.'
                                if isinstance(solution,Femm5ExternalSphereSolution) else
                                'FEMM net charge and voltage; inner/outer totals inferred using Gauss’s law.')
            center = (solution.sphere.center_m if isinstance(solution,Femm5ExternalSphereSolution)
                      else solution.shell.center_m)
        radius = None if isinstance(solution,Femm5ParallelPlateSolution) else solution.outer_radius_m
        masks = [SourceMask(s.position_m,s.radius_m,'numerical-core') for s in solution.numerical_sources]
        if isinstance(solution,(Femm5PlaneSolution,Femm5ParallelPlateSolution)):
            density = lambda name,face,point: solution.surface_charge_density_at(face,point,
                                                                                  conductor_name=name)
        else:
            def density(name, face, point):
                normal = sub(point,center)
                cosine = dot(normal,solution.symmetry_axis)/hypot(*normal)
                return solution.surface_charge_density(face,acos(max(-1.,min(1.,cosine))),conductor_name=name)
        method = 'FEMM one-sided near-surface E estimate (mesh/offset dependent)'
        notes += ['FEMM finite charged source spheres approximate ideal points; their interiors are excluded.']
        if isinstance(solution,Femm5ParallelPlateSolution):
            notes.append('FEMM plates are finite disks with an artificial zero-radial-flux cylindrical side.')
            notes.append('Measured finite electrode charges include the capacitor background and source induction.')
            notes.append('Local FEMM E/sigma are mesh sensitive, not substituted analytical reference fields.')
        elif isinstance(solution,Femm5PlaneSolution):
            notes.append('Numerical plane is a finite truncated boundary with an approximate outer Robin condition.')
            notes.append('Plane charge is measured on the finite modeled disk; the ideal infinite plane has additional charge beyond the computational boundary.')
        elif isinstance(solution,Femm5ExternalSphereSolution):
            notes.append('Solid sphere total/outer charge is FEMM-measured; local density uses near-surface E.')
        else:
            notes.append('Shell inner/outer surface-charge totals are Gauss-law inferences, not local-density integrals.')
    else:
        sites = {}
        for source in problem.charges:
            sites.setdefault(source.position_m,[]).append(source.charge_c)
        a = min((s.inner_radius_m or s.outer_radius_m
                 for s in problem.conductors if not isinstance(s,ConductingPlane)),
                default=min((abs(conductor.signed_distance_m(p)) for p,_ in sites.items())
                            if isinstance(conductor,ConductingPlane) and sites else [1.]))
        for point,charges in sites.items():
            if fsum(charges) != 0:
                masks.append(SourceMask(point,a*.015,'display-exclusion'))
        known = isinstance(solution,(NestedShellSolution,MultiImageShellSolution,
                                    ImageChargeShellSolution,ConcentricShellSolution,
                                    ExternalGroundedSphereSolution, ExternalMultiSphereSolution,
                                    GroundedPlaneSolution,ParallelPlateSolution))
        if known:
            conductor_method = 'Analytical reference conductor properties.'
        notes += [('Analytical reference samples, not FEMM output.' if known else
                   'Custom solver samples; configure SamplingContext for its numerical domain and exclusions.'),
                  'Small display-exclusion discs avoid ideal point singularities; they are not physical source radii.']
        if isinstance(solution,ParallelPlateSolution):
            density = lambda name,face,point: solution.surface_charge_density_at(
                face,point,conductor_name=name)
            method = f'parallel-plate symmetric image series (order {solution.image_order})'
            conductor_method = ('Analytical infinite plates: charge totals are excess induced totals '
                                'only if plate potentials match; otherwise undefined (n/a).')
            notes.append('Image-charge fields are truncated approximations; test convergence by increasing image_order.')
            notes.append('Infinite plates at different potentials have unbounded total charge, not a finite plate Q.')
        elif isinstance(solution,GroundedPlaneSolution):
            density = lambda name,face,point: solution.surface_charge_density_at(face,point)
            method = 'exact grounded infinite-plane mirror-charge density'
        elif isinstance(solution,ExternalMultiSphereSolution):
            density = lambda name,face,point: solution.surface_charge_density_at(face,point)
            method = 'exact Kelvin multi-image external surface density'
        elif isinstance(solution,ExternalGroundedSphereSolution):
            density = lambda name,face,point: solution.surface_charge_density_at(face,point)
            method = ('exact Kelvin images plus central compensator surface density'
                      if isinstance(solution, ExternalIsolatedSphereSolution) else
                      'exact Kelvin image-charge surface density')
        elif isinstance(solution,NestedShellSolution):
            density = solution.surface_charge_density_at
            method = 'exact analytical surface density'
        elif isinstance(solution,MultiImageShellSolution):
            density = lambda name,face,point: solution.surface_charge_density_at(face,point)
            method = 'exact analytical surface density'
        elif isinstance(solution,ImageChargeShellSolution):
            axis = sub(solution.source.position_m,solution.shell.center_m)
            d = hypot(*axis)
            def density(name,face,point):
                delta = sub(point,solution.shell.center_m)
                cosine = dot(delta,axis)/(hypot(*delta)*d) if d else 1.
                return solution.surface_charge_density(face,acos(max(-1.,min(1.,cosine))))
            method = 'exact analytical surface density'
        elif isinstance(solution,ConcentricShellSolution):
            def density(name,face,point):
                c = solution.conductor_results[0]
                q = c.inner_surface_charge_c if face == 'inner' else c.outer_surface_charge_c
                r = solution.shell.inner_radius_m if face == 'inner' else solution.shell.outer_radius_m
                return q/(4*pi*r*r)
            method = 'uniform analytical density from conductor surface total'
    return SamplingContext(problem,solution,type(solution).__name__,tuple(masks),center,radius,density,method,tuple(notes),
                           conductor_results_method=conductor_method,
                           domain_cylinder=cylinder)
