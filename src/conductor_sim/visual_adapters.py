"""Optional visual metadata adapters; the sampling/tracing engine knows no solver."""
from math import acos, fsum, hypot, pi

from .analytical import ConcentricShellSolution
from .image_charge import ImageChargeShellSolution
from .multi_image import MultiImageShellSolution
from .nested_shells import NestedShellSolution
from .femm5 import Femm5Solution
from .femm5_external import Femm5ExternalSphereSolution
from .external_sphere import ExternalGroundedSphereSolution
from .visualization import SamplingContext, SourceMask, dot, sub


def context_for_solution(problem, solution):
    """Describe masks, valid domain and density API without inventing field values.

    Extensible solvers can instantiate SamplingContext directly. The caller
    retains ownership of a supplied solution; only sample_sweep owns/closes it.
    """
    density = None
    method = 'unavailable'
    radius = None
    center = problem.conductors[0].center_m if problem.conductors else (0.,0.,0.)
    notes = ['Field lines are geometry traces, not charge trajectories; line count is not flux.',
             'Profiles follow a great circle, not an automatically integrated spherical charge.',
             'The colour scale is shared across frames; display clipping does not alter exported values.']
    masks = []
    conductor_method = 'Supplied solver conductor properties; provenance not specified.'
    if isinstance(solution, (Femm5Solution,Femm5ExternalSphereSolution)):
        conductor_method = ('FEMM net charge and voltage; solid sphere outer charge is measured.'
                            if isinstance(solution,Femm5ExternalSphereSolution) else
                            'FEMM net charge and voltage; inner/outer totals inferred using Gauss’s law.')
        radius = solution.outer_radius_m
        center = (solution.sphere.center_m if isinstance(solution,Femm5ExternalSphereSolution)
                  else solution.shell.center_m)
        masks = [SourceMask(s.position_m,s.radius_m,'numerical-core') for s in solution.numerical_sources]
        def density(name, face, point):
            normal = sub(point,center)
            cosine = dot(normal,solution.symmetry_axis)/hypot(*normal)
            return solution.surface_charge_density(face,acos(max(-1.,min(1.,cosine))),conductor_name=name)
        method = 'FEMM one-sided near-surface E estimate (mesh/offset dependent)'
        notes += ['FEMM finite charged source spheres approximate ideal points; their interiors are excluded.']
        if isinstance(solution, Femm5ExternalSphereSolution):
            notes.append('Solid sphere total/outer charge is FEMM-measured; local density uses near-surface E.')
        else:
            notes.append('Shell inner/outer surface-charge totals are Gauss-law inferences, not local-density integrals.')
    else:
        sites = {}
        for source in problem.charges:
            sites.setdefault(source.position_m,[]).append(source.charge_c)
        a = min((s.inner_radius_m or s.outer_radius_m
                 for s in problem.conductors),default=1.)
        for point,charges in sites.items():
            if fsum(charges) != 0:
                masks.append(SourceMask(point,a*.015,'display-exclusion'))
        known = isinstance(solution,(NestedShellSolution,MultiImageShellSolution,
                                    ImageChargeShellSolution,ConcentricShellSolution,
                                    ExternalGroundedSphereSolution))
        if known:
            conductor_method = 'Analytical reference conductor properties.'
        notes += [('Analytical reference samples, not FEMM output.' if known else
                   'Custom solver samples; configure SamplingContext for its numerical domain and exclusions.'),
                  'Small display-exclusion discs avoid ideal point singularities; they are not physical source radii.']
        if isinstance(solution,ExternalGroundedSphereSolution):
            density = lambda name,face,point: solution.surface_charge_density_at(face,point)
            method = 'exact Kelvin image-charge surface density'
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
                           conductor_results_method=conductor_method)
