"""Conservative interpolation stencils for stored potential data, not a solver.

Ordinary vacuum cells use bilinear interpolation. Cells cut by a conducting
plane use clipped triangles anchored to the stored conductor voltage. Missing
samples, source masks and spherical metal interfaces are never bridged. This
coarse display mesh uses stored samples; optional local source refinements
query the same solver for additional vacuum-side potential values.
"""
from math import ceil, hypot, sqrt


def _project(point, plane):
    d = [point[i]-plane['origin_m'][i] for i in range(3)]
    u, v = plane['u'], plane['v']
    n = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
    return tuple(sum(d[i]*a[i] for i in range(3)) for a in (u, v, n))


def _distance_bounds(poly, center):
    """Minimum and maximum distance from a point to a convex polygon."""
    x, y = center
    far = max(hypot(p[0]-x, p[1]-y) for p in poly)
    near = far
    signs = []
    for a, b in zip(poly, poly[1:]+poly[:1]):
        dx, dy = b[0]-a[0], b[1]-a[1]
        length2 = dx*dx+dy*dy
        t = max(0., min(1., ((x-a[0])*dx+(y-a[1])*dy)/length2)) if length2 else 0.
        near = min(near, hypot(x-a[0]-t*dx, y-a[1]-t*dy))
        signs.append(dx*(y-a[1])-dy*(x-a[0]))
    if all(s >= 0 for s in signs) or all(s <= 0 for s in signs):
        near = 0.
    return near, far


def potential_interpolation(frame):
    """Return per-cell safe bilinear flags and exceptional boundary triangles.

    A null node is not replaced with a neighbour or a zero. Conservative gaps
    may remain near an unresolved mask/curved interface, at most grid-scale.
    This also accepts older v1 frames without changing their stored values.
    """
    grid = frame['grid']
    xs, ys = grid['x_m'], grid['y_m']
    shapes = frame['geometry']
    planes = [s for s in shapes if s['kind'] == 'plane' and s['visible']]
    spheres = [s for s in shapes if s['kind'] == 'sphere' and s['visible']]
    filled = any(s.get('fills_metal', False) for s in shapes)
    volts = {s['name']: s['potential_v'] for s in frame['conductor_results']}
    masks = []
    for mask in frame['source_masks']:
        x, y, h = _project(mask['center_m'], frame['plane'])
        if abs(h) <= mask['radius_m']:
            masks.append(((x, y), sqrt(max(0., mask['radius_m']**2-h*h))))

    def signed(p, shape):
        return sum((p[i]-shape['point_m'][i])*shape['normal_xy'][i] for i in range(2))

    def clear(poly):
        if filled or any(p[2] is None for p in poly):
            return False
        # The polygon is convex; checking all vertices suffices for a plane.
        if any(signed(p, shape) < -1e-14 for p in poly for shape in planes):
            return False
        for center, radius in masks:
            if _distance_bounds(poly, center)[0] <= radius:
                return False
        for shape in spheres:
            near, far = _distance_bounds(poly, shape['center_m'])
            inner, outer = shape['inner_radius_m'], shape['outer_radius_m']
            if not ((inner > 0 and far <= inner) or near >= outer):
                return False
        return True

    def clip(poly, shape):
        out = []
        for a, b in zip(poly, poly[1:]+poly[:1]):
            da, db = signed(a, shape), signed(b, shape)
            if da >= 0:
                out.append(a)
            if (da < 0 <= db) or (db < 0 <= da):
                t = da/(da-db)
                out.append([a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1]), volts.get(shape['name'])])
        return out

    flags, boundary = [], []
    for j in range(len(ys)-1):
        row = []
        for i in range(len(xs)-1):
            ids = ((j, i), (j, i+1), (j+1, i+1), (j+1, i))
            poly = [[xs[x], ys[y], grid['potential_v'][y][x]
                     if grid['region'][y][x] in ('vacuum', 'surface') else None] for y, x in ids]
            full = clear(poly)
            row.append(full)
            if full:
                continue
            triangles = []
            for indices in ((0, 1, 2), (0, 2, 3)):
                part = [poly[k] for k in indices]
                for shape in planes:
                    part = clip(part, shape)
                    if len(part) < 3:
                        break
                if len(part) < 3 or not clear(part):
                    continue
                # Domain boundaries are convex; new plane intersections must
                # not be fabricated from a missing numerical-domain sample.
                if any(grid['region'][y][x] in ('outside-domain', 'source-mask')
                       or grid['potential_v'][y][x] is None for y, x in [ids[k] for k in indices]):
                    continue
                for k in range(1, len(part)-1):
                    a, b, c = part[0], part[k], part[k+1]
                    area = (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
                    if area != 0:
                        triangles.append([a, b, c])
            if triangles:
                boundary.append({'cell': [i, j], 'triangles': triangles})
        flags.append(row)
    return {'method': 'masked-bilinear-with-plane-boundaries', 'cells': flags, 'boundary': boundary}


def potential_refinements(context, plane, grid, masks, *, target_subdivisions=5, max_samples=8000):
    """Sample small, bounded high-resolution patches beside excluded source cores.

    This queries the *same* solver for additional vacuum-side potentials.
    A nearby coarse grid cell may be blank because it touches a small mask;
    it need not remain blank when its subcells can be sampled safely. The
    actual mask is never sampled or bridged. No new physical model is used.
    """
    from .visualization import _finite
    xs, ys = grid['x_m'], grid['y_m']
    dx, dy = xs[1]-xs[0], ys[1]-ys[0]
    base_step = min(dx, dy)
    patches = []
    for mask in masks:
        x, y, h = plane.project(mask.center_m)
        if abs(h) >= mask.radius_m:
            continue
        radius = sqrt(max(0., mask.radius_m**2-h*h))
        halfwidth = radius + 1.5*max(dx,dy)
        xlo, xhi = max(xs[0], x-halfwidth), min(xs[-1], x+halfwidth)
        ylo, yhi = max(ys[0], y-halfwidth), min(ys[-1], y+halfwidth)
        if xlo >= xhi or ylo >= yhi:
            continue
        patches.append((xlo,xhi,ylo,yhi))
    if not patches:
        return []
    counts = [(max(3,ceil((b-a)/base_step*target_subdivisions)+1),
               max(3,ceil((d-c)/base_step*target_subdivisions)+1))
              for a,b,c,d in patches]
    estimated = sum(nx*ny for nx,ny in counts)
    if estimated > max_samples:
        # Bound numerical queries for heavily populated authored scenes.
        shrink = sqrt(max_samples/estimated)
        counts = [(max(3, int((nx-1)*shrink)+1),
                   max(3, int((ny-1)*shrink)+1)) for nx,ny in counts]
    result = []
    for (xlo,xhi,ylo,yhi),(nx,ny) in zip(patches,counts):
        xx = [xlo+(xhi-xlo)*i/(nx-1) for i in range(nx)]
        yy = [ylo+(yhi-ylo)*j/(ny-1) for j in range(ny)]
        fine = {'x_m':xx,'y_m':yy,'potential_v':[],'region':[]}
        for y in yy:
            row, regions = [], []
            for x in xx:
                point = plane.world((x,y))
                kind = context.region(point)
                regions.append(kind)
                if kind in ('source-mask','outside-domain'):
                    row.append(None)
                else:
                    row.append(_finite(context.solution.potential_at(point), 'refined potential'))
            fine['region'].append(regions)
            fine['potential_v'].append(row)
        # The caller computes conservative subcell stencils using the same
        # original conductor and source-mask geometry as the coarse grid.
        result.append({'grid':fine})
    return result


def potential_colour_scale(frames):
    """One sign-preserving, unclipped asinh colour range for the whole sweep."""
    maximum = 0.
    for frame in frames:
        grid = frame['grid']
        for row, regions in zip(grid['potential_v'], grid['region']):
            for value, region in zip(row, regions):
                if value is not None and region in ('vacuum', 'surface'):
                    maximum = max(maximum, abs(value))
        for patch in frame.get('potential_refinements', ()):
            fine = patch['grid']
            for row, regions in zip(fine['potential_v'], fine['region']):
                for value, region in zip(row, regions):
                    if value is not None and region in ('vacuum', 'surface'):
                        maximum = max(maximum, abs(value))
        for conductor in frame['conductor_results']:
            maximum = max(maximum, abs(conductor['potential_v']))
    maximum = maximum or 1.
    soft = max(max(f.get('display_clip_v', 1.) for f in frames)/3, maximum/1e6)
    return {'kind': 'asinh', 'max_abs_v': maximum, 'softening_v': min(maximum, soft)}
