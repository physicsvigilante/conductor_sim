"""Optional visual metadata adapters; the sampling/tracing engine knows no solver."""
from fractions import Fraction
from math import acos, fsum, hypot, pi

from .model import ConductingPlane, _plane_raw_normal
from .geometry import _scaled_norm_fraction
from .observables import _scaled_ratio_product
from .grounded_plane import GroundedPlaneSolution
from .parallel_plates import ParallelPlateSolution
from .analytical import ConcentricShellSolution
from .image_charge import ImageChargeShellSolution
from .multi_image import MultiImageShellSolution
from .nested_shells import NestedShellSolution
from .mixed_radial import MixedRadialSolution
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
    plane_profile_at = None
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
        sites = {point:fsum(charges) for point,charges in sites.items() if fsum(charges)!=0}
        a = min((s.inner_radius_m or s.outer_radius_m
                 for s in problem.conductors if not isinstance(s,ConductingPlane)),
                default=min((abs(c.signed_distance_m(p)) for c in problem.conductors
                             if isinstance(c,ConductingPlane) for p in sites),default=1.))
        for point in sites:
            clearances = []
            for c in problem.conductors:
                if isinstance(c,ConductingPlane):
                    clearances.append(abs(c.signed_distance_m(point)))
                else:
                    distance = hypot(*sub(point,c.center_m))
                    clearances.extend(abs(distance-r) for r in
                                      (c.inner_radius_m,c.outer_radius_m) if r>0)
            neighbours = [hypot(*sub(point,other)) for other in sites if other!=point]
            size = min(a*.015, min(clearances,default=float('inf'))/4,
                       min(neighbours,default=float('inf'))/4)
            masks.append(SourceMask(point,size,'display-exclusion'))
        known = isinstance(solution,(NestedShellSolution,MixedRadialSolution,MultiImageShellSolution,
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
            def plane_profile_at(name,face,origin_m,u_vec,v_vec,x0,y0,tangent,d):
                if face != 'outer':
                    raise ValueError("the ideal plane has one exposed conducting surface")
                raw_n = _plane_raw_normal(solution.plane.normal)
                raw_u = _plane_raw_normal(u_vec)
                raw_v = _plane_raw_normal(v_vec)
                fn = tuple(Fraction(x) for x in raw_n)
                n_sq = sum(x * x for x in fn)
                fu_raw = tuple(Fraction(x) for x in raw_u)
                fv_raw = tuple(Fraction(x) for x in raw_v)
                u_norm = _scaled_norm_fraction(fu_raw)
                v_norm = _scaled_norm_fraction(fv_raw)
                fu = tuple(x / u_norm for x in fu_raw)
                fv = tuple(x / v_norm for x in fv_raw)
                raw_nx = sum(fn[i] * fu_raw[i] for i in range(3))
                raw_ny = sum(fn[i] * fv_raw[i] for i in range(3))
                fnx = Fraction(0) if raw_nx == 0 else raw_nx / u_norm
                fny = Fraction(0) if raw_ny == 0 else raw_ny / v_norm
                fdenom = fnx * fnx + fny * fny
                if fdenom > 0:
                    fd0 = sum(
                        (Fraction(origin_m[i]) - Fraction(solution.plane.point_m[i])) * fn[i]
                        for i in range(3)
                    )
                    fx0 = -fd0 * fnx / fdenom
                    fy0 = -fd0 * fny / fdenom
                else:
                    fx0, fy0 = Fraction(x0), Fraction(y0)
                fx = fx0 + Fraction(d) * Fraction(tangent[0])
                fy = fy0 + Fraction(d) * Fraction(tangent[1])
                pos_offset = tuple(fx * fu[i] + fy * fv[i] for i in range(3))
                h_off = sum(
                    ((Fraction(origin_m[i]) - Fraction(solution.plane.point_m[i])) + pos_offset[i]) * fn[i]
                    for i in range(3)
                ) / n_sq
                terms = []
                for q in solution.sources:
                    hq = solution.plane.signed_distance_m(q.position_m)
                    delta_f = tuple(
                        (Fraction(origin_m[i]) - Fraction(q.position_m[i])) + pos_offset[i] - h_off * fn[i]
                        for i in range(3)
                    )
                    d_max = max(abs(x) for x in delta_f)
                    if d_max == 0:
                        raise ZeroDivisionError("surface-density point coincides with point source")
                    r_scaled = hypot(*(float(x / d_max) for x in delta_f))
                    try:
                        d_max_f = float(d_max)
                    except OverflowError:
                        d_max_f = float("inf")
                    if d_max_f > 0.0 and d_max_f < float("inf"):
                        r_val = d_max_f * r_scaled
                        if r_val < float("inf"):
                            terms.append(_scaled_ratio_product((q.charge_c, hq), (2.0 * pi, r_val, r_val, r_val)))
                            continue
                    exact_term = (
                        Fraction(q.charge_c) * Fraction(hq)
                        / (Fraction(2.0 * pi) * (d_max ** 3) * Fraction(r_scaled) ** 3)
                    )
                    try:
                        terms.append(float(exact_term))
                    except OverflowError:
                        terms.append(float("inf") if exact_term > 0 else float("-inf"))
                return -fsum(terms)
            method = 'exact grounded infinite-plane mirror-charge density'
        elif isinstance(solution,ExternalMultiSphereSolution):
            density = lambda name,face,point: solution.surface_charge_density_at(face,point)
            method = 'exact Kelvin multi-image external surface density'
        elif isinstance(solution,ExternalGroundedSphereSolution):
            density = lambda name,face,point: solution.surface_charge_density_at(face,point)
            method = ('exact Kelvin images plus central compensator surface density'
                      if isinstance(solution, ExternalIsolatedSphereSolution) else
                      'exact Kelvin image-charge surface density')
        elif isinstance(solution,(NestedShellSolution,MixedRadialSolution)):
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
                           domain_cylinder=cylinder,
                           plane_profile_at=plane_profile_at)
