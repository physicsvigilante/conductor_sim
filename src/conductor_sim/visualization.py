"""Solver-independent plane sampling, contours, bounded streamlines and sweeps.

This module does not solve electrostatics. It samples a supplied FieldSolution.
Unexpected solver errors propagate; only explicitly excluded spatial regions
are masked. Field lines stop at physical faces, source masks and domain bounds.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import acos, cos, fsum, hypot, isfinite, pi, sin, sqrt
from pathlib import Path
from typing import Callable, Iterable
import json
from itertools import islice
import os
import tempfile

from .model import ConductingPlane, ElectrostaticProblem, Vector3, _position
from .geometry import SphereBoundary, CylinderSideBoundary, contains_metal, physical_boundaries, slice_geometry
from .solver import FieldSolution, OutsideDomain, SingularField
from ._resources import cleanup_resources

Vec2 = tuple[float, float]
Density = Callable[[str, str, Vector3], float]


def dot(a, b):
    return fsum(x*y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def unit(v):
    n = hypot(*v)
    if not isfinite(n) or n == 0:
        raise ValueError("direction must be finite and nonzero")
    return tuple(x/n for x in v)


@dataclass(frozen=True)
class SliceSpec:
    """An orthonormal 3D plane, rectangular coordinates and square grid count."""
    origin_m: Vector3 = (0., 0., 0.)
    u: Vector3 = (1., 0., 0.)
    v: Vector3 = (0., 0., 1.)
    x_range_m: tuple[float, float] = (-.3, .3)
    y_range_m: tuple[float, float] = (-.3, .3)
    resolution: int = 61

    def __post_init__(self):
        object.__setattr__(self, 'origin_m', _position(self.origin_m, 'plane origin'))
        u, v = unit(_position(self.u, 'u')), unit(_position(self.v, 'v'))
        if abs(dot(u, v)) > 1e-10:
            raise ValueError('plane axes must be orthogonal')
        object.__setattr__(self, 'u', u)
        object.__setattr__(self, 'v', v)
        for name in ('x_range_m', 'y_range_m'):
            bounds = getattr(self, name)
            if len(bounds) != 2 or not all(isfinite(x) for x in bounds) or bounds[0] >= bounds[1]:
                raise ValueError('slice bounds must be finite and strictly increasing')
            object.__setattr__(self, name, tuple(float(x) for x in bounds))
        if isinstance(self.resolution, bool) or not isinstance(self.resolution, int) or not 9 <= self.resolution <= 201:
            raise ValueError('resolution must be an integer in [9, 201]')

    @property
    def normal(self):
        a, b = self.u, self.v
        return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

    def world(self, xy: Vec2) -> Vector3:
        return tuple(self.origin_m[i] + xy[0]*self.u[i] + xy[1]*self.v[i] for i in range(3))

    def project(self, point: Vector3):
        delta = sub(point, self.origin_m)
        return dot(delta, self.u), dot(delta, self.v), dot(delta, self.normal)

    def contains(self, xy):
        return self.x_range_m[0] <= xy[0] <= self.x_range_m[1] and self.y_range_m[0] <= xy[1] <= self.y_range_m[1]

    @classmethod
    def meridian(cls, problem, axis=(0., 0., 1.), *, extent_m=None, resolution=61):
        if not problem.conductors:
            raise ValueError('meridian helper requires a conductor')
        v = unit(_position(axis, 'meridian axis'))
        ref = (1., 0., 0.) if abs(v[0]) < .9 else (0., 1., 0.)
        u = unit(tuple(ref[i]-dot(ref, v)*v[i] for i in range(3)))
        conductor = problem.conductors[0]
        if isinstance(conductor, ConductingPlane):
            # A meridian of an infinite plane always contains its normal. A
            # tangential slice would not show the charge/surface separation.
            v = conductor.normal
            ref = (1.,0.,0.) if abs(v[0]) < .9 else (0.,1.,0.)
            u = unit(tuple(ref[i]-dot(ref,v)*v[i] for i in range(3)))
            center = conductor.point_m
            sites = _sites(problem)
            if sites:
                # A plane anchor is arbitrary; a meridian should contain the
                # active charge's normal line, not miss a translated source.
                point = sites[0][0]
                height = conductor.signed_distance_m(point)
                center = tuple(point[i]-height*v[i] for i in range(3))
                transverse = [tuple(p[i]-center[i]-dot(sub(p,center),v)*v[i]
                                    for i in range(3)) for p,_ in sites]
                tangent = max(transverse,key=lambda t:hypot(*t))
                if hypot(*tangent) > 1e-12*max(1.,hypot(*sub(point,center))):
                    u = unit(tangent)
        else:
            center = conductor.center_m
        reach = max([hypot(*sub(s.center_m,center))+s.outer_radius_m
                     for s in problem.conductors if not isinstance(s,ConductingPlane)] +
                    [hypot(*sub(p,center)) for p,_ in _sites(problem)] +
                    [abs(c.signed_distance_m(center)) for c in problem.conductors
                     if isinstance(c,ConductingPlane)] + [0.01])
        extent = extent_m if extent_m is not None else 1.55*reach
        if not isfinite(extent) or extent <= 0:
            raise ValueError('meridian extent must be finite and positive')
        return cls(center,u,v,(-extent,extent),(-extent,extent),resolution)


@dataclass(frozen=True)
class SourceMask:
    center_m: Vector3
    radius_m: float
    kind: str  # numerical-core or display-exclusion; never a physical shell

    def __post_init__(self):
        object.__setattr__(self, 'center_m', _position(self.center_m, 'source mask'))
        if not isfinite(self.radius_m) or self.radius_m <= 0:
            raise ValueError('source mask radius must be finite and positive')


@dataclass(frozen=True)
class CylindricalDomain:
    """Finite axisymmetric FEMM domain, not a physical conductor."""
    origin_m: Vector3
    axis: Vector3
    radius_m: float
    min_height_m: float
    max_height_m: float

    def __post_init__(self):
        object.__setattr__(self,'origin_m',_position(self.origin_m,'cylinder origin'))
        object.__setattr__(self,'axis',unit(_position(self.axis,'cylinder axis')))
        if (not isfinite(self.radius_m) or self.radius_m<=0 or
                not isfinite(self.min_height_m) or not isfinite(self.max_height_m)
                or self.min_height_m>=self.max_height_m):
            raise ValueError('invalid numerical cylinder domain')

    def radial_distance_m(self,point):
        delta=sub(point,self.origin_m)
        axial=dot(delta,self.axis)
        return hypot(*(delta[i]-axial*self.axis[i] for i in range(3)))


@dataclass
class SamplingContext:
    """Visual adapter contract. A caller can supply any solver implementing V/E."""
    problem: ElectrostaticProblem
    solution: FieldSolution
    backend: str
    masks: tuple[SourceMask, ...] = ()
    domain_center_m: Vector3 = (0., 0., 0.)
    domain_radius_m: float | None = None
    density_at: Density | None = None
    density_method: str = 'unavailable'
    notes: tuple[str, ...] = ()
    conductor_results_method: str = 'Supplied solver conductor properties; provenance not specified.'
    domain_cylinder: CylindricalDomain | None = None

    def region(self, point):
        if self.domain_radius_m is not None and hypot(*sub(point, self.domain_center_m)) >= self.domain_radius_m:
            return 'outside-domain'
        if self.domain_cylinder is not None and (
                self.domain_cylinder.radial_distance_m(point) >= self.domain_cylinder.radius_m):
            return 'outside-domain'
        if any(hypot(*sub(point, m.center_m)) <= m.radius_m for m in self.masks):
            return 'source-mask'
        if any(contains_metal(conductor,point) for conductor in self.problem.conductors):
            return 'metal'
        return 'vacuum'

    def barriers(self):
        for conductor in self.problem.conductors:
            yield from physical_boundaries(conductor)
        for mask in self.masks:
            yield SphereBoundary(mask.center_m,mask.radius_m,'source-mask')
        if self.domain_radius_m is not None:
            yield SphereBoundary(self.domain_center_m,self.domain_radius_m,'outside-domain')
        if self.domain_cylinder is not None:
            c=self.domain_cylinder
            yield CylinderSideBoundary(c.origin_m,c.axis,c.radius_m,
                                       c.min_height_m,c.max_height_m)

    def first_hit(self, a: Vector3, b: Vector3):
        """Find first exact physical boundary crossing on a traced 3D segment."""
        best = None
        for barrier in self.barriers():
            t = barrier.first_hit(a,b)
            if t is not None and (best is None or t < best[0]):
                best = (t,barrier.kind)
        return best


def _finite(value, what):
    result = float(value)
    if not isfinite(result):
        raise ValueError(f'nonfinite {what} returned by solver')
    return result


def trace_field_line(context: SamplingContext, plane: SliceSpec, seed: Vec2, *,
                     direction=1, step_m=.001, max_steps=700, field_floor=1e-8):
    """Midpoint integration of unit E with geometric stopping and hard step bounds.

    Non-planar E terminates instead of pretending a projected curve is a true
    field line. Returned vertices are always ordered in the direction of E.
    """
    if direction not in (-1, 1) or not isfinite(step_m) or step_m <= 0:
        raise ValueError('direction must be +/-1 and step_m positive')
    if not isinstance(max_steps, int) or not 1 <= max_steps <= 5000:
        raise ValueError('max_steps must be in [1,5000]')
    if not isfinite(field_floor) or field_floor < 0:
        raise ValueError('field_floor must be finite and nonnegative')
    if len(seed) != 2 or not all(isfinite(x) for x in seed):
        raise ValueError('seed must contain two finite coordinates')
    points = [tuple(seed)]
    reason = 'step-limit'

    def slope(xy):
        if not plane.contains(xy):
            return None, 'slice-boundary'
        world = plane.world(xy)
        region = context.region(world)
        if region != 'vacuum':
            return None, region
        try:
            vector = _position(context.solution.electric_field_at(world), 'solver field')
        except SingularField:
            return None, 'surface'
        size = hypot(*vector)
        if size <= field_floor:
            return None, 'zero-field'
        if abs(dot(vector, plane.normal)) > 1e-7*size:
            return None, 'out-of-plane'
        return (direction*dot(vector, plane.u)/size, direction*dot(vector, plane.v)/size), None

    for _ in range(max_steps):
        current = points[-1]
        first, reason1 = slope(current)
        if first is None:
            reason = reason1
            break
        trial = tuple(current[i] + step_m*first[i] for i in range(2))
        hit = context.first_hit(plane.world(current), plane.world(trial))
        if hit:
            points.append(tuple(current[i] + hit[0]*(1-1e-7)*(trial[i]-current[i]) for i in range(2)))
            reason = hit[1]
            break
        midpoint = tuple((current[i]+trial[i])/2 for i in range(2))
        second, reason2 = slope(midpoint)
        if second is None:
            reason = reason2
            break
        nxt = tuple(current[i]+step_m*second[i] for i in range(2))
        hit = context.first_hit(plane.world(current), plane.world(nxt))
        if hit:
            points.append(tuple(current[i]+hit[0]*(1-1e-7)*(nxt[i]-current[i]) for i in range(2)))
            reason = hit[1]
            break
        if not plane.contains(nxt):
            reason = 'slice-boundary'
            break
        points.append(nxt)
    if direction == -1:
        points.reverse()
    return {'points_m': points, 'termination': reason, 'direction': 'electric-field'}


def contour_segments(grid, levels, context: SamplingContext, plane: SliceSpec):
    """Piecewise-linear contours on triangulated cells; masks are not interpolated."""
    xs, ys, values, regions = grid['x_m'], grid['y_m'], grid['potential_v'], grid['region']
    out = []
    for level in levels:
        segments = []
        for j in range(len(ys)-1):
            for i in range(len(xs)-1):
                ids = ((j,i), (j,i+1), (j+1,i+1), (j+1,i))
                for corners in ((0,1,2),(0,2,3)):
                    ids3 = [ids[k] for k in corners]
                    if any(regions[y][x] != 'vacuum' or values[y][x] is None for y,x in ids3):
                        continue
                    verts = [(xs[x],ys[y],values[y][x]) for y,x in ids3]
                    intersections = []
                    for a,b in ((0,1),(1,2),(2,0)):
                        p,q = verts[a],verts[b]
                        if (p[2] <= level < q[2]) or (q[2] <= level < p[2]):
                            t = (level-p[2])/(q[2]-p[2])
                            intersections.append((p[0]+t*(q[0]-p[0]),p[1]+t*(q[1]-p[1])))
                    if len(intersections) == 2 and hypot(*sub(*intersections)) > 1e-15:
                        if context.first_hit(*(plane.world(p) for p in intersections)) is None:
                            segments.append(intersections)
        out.append({'level_v': level, 'segments_m': segments})
    return out


def _sites(problem):
    grouped = {}
    for charge in problem.charges:
        grouped.setdefault(charge.position_m, []).append(charge.charge_c)
    return [(p, fsum(qs)) for p,qs in sorted(grouped.items()) if fsum(qs) != 0]


def _default_seeds(context, plane):
    seeds = []
    for position, q in _sites(context.problem):
        x,y,h = plane.project(position)
        if abs(h) > 1e-10:
            continue
        radius = max((m.radius_m for m in context.masks if m.center_m == position), default=.001)
        for i in range(10):
            angle = 2*pi*(i+.25)/10
            seeds.append(((x+radius*1.4*cos(angle),y+radius*1.4*sin(angle)), 1 if q > 0 else -1))
    shells = sorted((s for s in context.problem.conductors
                     if not isinstance(s,ConductingPlane)),key=lambda s:s.inner_radius_m)
    for i,shell in enumerate(shells):
        x,y,h = plane.project(shell.center_m)
        if abs(h) > 1e-10:
            continue
        r = (shell.outer_radius_m+shells[i+1].inner_radius_m)/2 if i+1 < len(shells) else shell.outer_radius_m*1.25
        for k in range(8):
            t = 2*pi*(k+.25)/8
            for direction in (-1,1):
                seeds.append(((x+r*cos(t),y+r*sin(t)),direction))
    if not seeds and len(context.problem.conductors) == 2 and all(
            isinstance(c,ConductingPlane) for c in context.problem.conductors):
        geometries = [slice_geometry(c,plane) for c in context.problem.conductors]
        if all(g['visible'] for g in geometries):
            first,second = geometries
            midpoint = tuple((first['point_m'][i]+second['point_m'][i])/2 for i in range(2))
            nx,ny = first['normal_xy']
            tangent = (-ny/hypot(nx,ny),nx/hypot(nx,ny))
            span = min(plane.x_range_m[1]-plane.x_range_m[0],
                       plane.y_range_m[1]-plane.y_range_m[0])*.3
            for fraction in (-1.,-.5,0.,.5,1.):
                xy = tuple(midpoint[i]+fraction*span*tangent[i] for i in range(2))
                if plane.contains(xy) and context.region(plane.world(xy)) == 'vacuum':
                    seeds.extend((xy,direction) for direction in (-1,1))
    return seeds[:96]


def _profile_value(context, name, face, point):
    """Missing density data is null, never zero or an analytical fallback."""
    if context.region(point) in ('outside-domain','source-mask'):
        return None
    try:
        return _finite(context.density_at(name,face,point),'surface density')
    except OutsideDomain:
        # The one-sided numerical density probe can leave the domain even
        # when the surface point itself is just inside its radial edge.
        return None


def sample_frame(context: SamplingContext, plane: SliceSpec, *, label='Simulation',
                 seeds=None, line_steps=700, profile_samples=91):
    """Materialize one serializable frame. No live solver is retained in output."""
    if not isinstance(label, str):
        raise ValueError('frame label must be text')
    if not isinstance(profile_samples, int) or not 9 <= profile_samples <= 361:
        raise ValueError('profile_samples must be in [9,361]')
    n = plane.resolution
    xs = [plane.x_range_m[0]+i*(plane.x_range_m[1]-plane.x_range_m[0])/(n-1) for i in range(n)]
    ys = [plane.y_range_m[0]+i*(plane.y_range_m[1]-plane.y_range_m[0])/(n-1) for i in range(n)]
    grid = {'x_m':xs,'y_m':ys,'potential_v':[],'electric_field_v_per_m':[],'region':[]}
    for y in ys:
        vv,ee,rr = [],[],[]
        for x in xs:
            point = plane.world((x,y))
            region = context.region(point)
            if region in ('outside-domain','source-mask'):
                vv.append(None); ee.append(None)
            else:
                vv.append(_finite(context.solution.potential_at(point), 'potential'))
                try:
                    vector = _position(context.solution.electric_field_at(point), 'solver field')
                    ee.append(vector)
                except SingularField:
                    ee.append(None)
                    region = 'surface'
            rr.append(region)
        grid['potential_v'].append(vv);grid['electric_field_v_per_m'].append(ee);grid['region'].append(rr)
    observed = sorted(abs(v) for row in grid['potential_v'] for v in row if v is not None)
    clip = observed[int(.95*(len(observed)-1))] if observed else 1.
    clip = clip if clip > 1e-12 else 1.
    low = max(-clip, min((v for row in grid['potential_v'] for v in row if v is not None), default=0))
    high = min(clip, max((v for row in grid['potential_v'] for v in row if v is not None), default=0))
    levels = [low+(min(high,clip)-low)*i/11 for i in range(1,11)] if high-low > 1e-10 else []
    lines = []
    terminations = {}
    step = min(xs[1]-xs[0],ys[1]-ys[0])*.2
    seed_list = list(islice(iter(_default_seeds(context,plane) if seeds is None else seeds),97))
    if len(seed_list) > 96:
        raise ValueError('at most 96 seed traces per frame')
    for xy,direction in seed_list:
        line = trace_field_line(context,plane,xy,direction=direction,step_m=step,max_steps=line_steps)
        terminations[line['termination']] = terminations.get(line['termination'],0)+1
        if len(line['points_m']) > 1:
            lines.append(line)
    profiles = []
    if context.density_at is not None:
        for conductor in context.problem.conductors:
            if isinstance(conductor, ConductingPlane):
                geom = slice_geometry(conductor,plane)
                if not geom['visible']:
                    continue
                x0,y0 = geom['point_m']; nx,ny = geom['normal_xy']
                norm = hypot(nx,ny)
                tangent = (-ny/norm,nx/norm)
                # A signed in-plane distance profile, not a radial or 3D
                # surface integral; can show asymmetric multi-source cases.
                span = hypot(plane.x_range_m[1]-plane.x_range_m[0],
                             plane.y_range_m[1]-plane.y_range_m[0])/2
                distances = [-span+2*span*i/(profile_samples-1)
                             for i in range(profile_samples)]
                values=[]
                for d in distances:
                    point = plane.world((x0+d*tangent[0],y0+d*tangent[1]))
                    values.append(_profile_value(context,conductor.name,'outer',point))
                profiles.append({'conductor':conductor.name,'surface':'outer',
                                 'coordinate_m':distances,'sigma_c_per_m2':values,
                                 'method':context.density_method})
                continue
            for face in (('outer',) if conductor.inner_radius_m == 0 else ('inner','outer')):
                radius = conductor.inner_radius_m if face == 'inner' else conductor.outer_radius_m
                angles = [pi*i/(profile_samples-1) for i in range(profile_samples)]
                values = []
                for t in angles:
                    point = tuple(conductor.center_m[i]+radius*(cos(t)*plane.v[i]+sin(t)*plane.u[i]) for i in range(3))
                    values.append(_profile_value(context,conductor.name,face,point))
                profiles.append({'conductor':conductor.name,'surface':face,'theta_rad':angles,
                                 'sigma_c_per_m2':values,'method':context.density_method})
    geometry = [slice_geometry(c,plane) for c in context.problem.conductors]
    charges = [{'charge_c':q,'position_m':list(p),'slice_position_m':plane.project(p)} for p,q in _sites(context.problem)]
    return {'label':label,'backend':context.backend,'plane':asdict(plane),'grid':grid,
            'contours':contour_segments(grid,levels,context,plane),'field_lines':lines,
            'line_termination_counts':terminations,
            'profiles':profiles,'geometry':geometry,'charges':charges,
            'conductor_results':[asdict(c) for c in context.solution.conductor_results],
            'source_masks':[asdict(m) for m in context.masks],
            'domain_radius_m':context.domain_radius_m,
            'domain_cylinder':(asdict(context.domain_cylinder) if context.domain_cylinder else None),
            'display_clip_v':clip,
            'notes':list(context.notes),'density_method':context.density_method,
            'conductor_results_method':context.conductor_results_method}


def sample_sweep(cases: Iterable[tuple[str,ElectrostaticProblem]], backend, *, plane=None, resolution=61,
                 line_steps=700, seeds=None):
    """Precompute at most 24 states; close owned numerical sessions, even on error.

    A caller-supplied plane is reused. Otherwise one fixed plane and extent
    are chosen from the first case. Browser playback never launches a solver.
    """
    from .visual_adapters import context_for_solution
    cases = list(islice(iter(cases),25))
    if not 1 <= len(cases) <= 24:
        raise ValueError('a sweep needs 1 to 24 cases')
    # A generator of authored seeds describes the same sweep in every state.
    # Consume it once, boundedly, before acquiring any solver resources.
    if seeds is not None:
        seeds = tuple(islice(iter(seeds), 97))
        if len(seeds) > 96:
            raise ValueError('at most 96 seed traces per frame')
    frames = []
    for label,problem in cases:
        solution = backend.solve(problem)
        try:
            context = context_for_solution(problem,solution)
            if plane is None:
                axis = getattr(solution,'symmetry_axis',None)
                if axis is None:
                    conductor = problem.conductors[0]
                    center = (conductor.point_m if isinstance(conductor,ConductingPlane)
                              else conductor.center_m)
                    positions = [sub(p,center) for p,q in _sites(problem)]
                    axis = max(positions,key=lambda p:hypot(*p),default=(0.,0.,1.))
                    axis = axis if hypot(*axis) else (0.,0.,1.)
                plane = SliceSpec.meridian(problem,axis,resolution=resolution)
            frames.append(sample_frame(context,plane,label=label,line_steps=line_steps,seeds=seeds))
        except BaseException as error:
            close = getattr(solution,'close',None)
            if callable(close):
                cleanup_resources((("sampled solution",close),),primary=error)
            raise
        else:
            close = getattr(solution,'close',None)
            if callable(close):
                close()
    bundle = {'schema':'conductor_sim.visualization.v1','units':{'length':'m','potential':'V','field':'V/m','density':'C/m^2'},
              'playback':'precomputed states; not browser-time numerical solving',
              'color_limit_v':max(f['display_clip_v'] for f in frames),'frames':frames}
    json.dumps(bundle,allow_nan=False)  # fail before any file is written
    return bundle


def write_visualization(bundle, path: str | Path, *, title='Conductor simulation') -> Path:
    """Write a self-contained offline HTML file, atomically, with escaped data."""
    from .viewer import render_html
    path = Path(path)
    text = render_html(bundle,title=title)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp = tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            stream.write(text)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path
