"""Bounded, progressive seed families for offline field-line pools.

Seeds are interleaved across each active source and conductor face so a prefix
covers the scene, rather than exhausting one charge before showing the next.
The dyadic sequence fills gaps reproducibly. Line counts are not flux weights.
"""
from math import cos, hypot, pi, sin
from itertools import zip_longest


def _progressive(count=32):
    for i in range(1, count+1):
        n, fraction, weight = i, 0., .5
        while n:
            fraction += (n & 1)*weight
            n >>= 1
            weight *= .5
        yield fraction


def _segment_in_view(point, tangent, plane):
    lo, hi = -float('inf'), float('inf')
    for p, t, bounds in zip(point, tangent, (plane.x_range_m, plane.y_range_m)):
        if abs(t) < 1e-15:
            if not bounds[0] <= p <= bounds[1]:
                return None
        else:
            a, b = sorted(((bounds[0]-p)/t, (bounds[1]-p)/t))
            lo, hi = max(lo, a), min(hi, b)
    return (lo, hi) if lo < hi else None


def seed_pool(context, plane, limit=96):
    from .geometry import slice_geometry
    from .model import ConductingPlane, _position
    from .solver import OutsideDomain, SingularField
    from .visualization import _sites, dot

    span = min(plane.x_range_m[1]-plane.x_range_m[0], plane.y_range_m[1]-plane.y_range_m[0])
    families = []
    sites = _sites(context.problem)

    def valid(xy):
        return plane.contains(xy) and context.region(plane.world(xy)) == 'vacuum'

    def away_from_face(xy, normal):
        if not valid(xy):
            return None
        try:
            vector = _position(context.solution.electric_field_at(plane.world(xy)), 'surface seed field')
        except (SingularField, OutsideDomain):
            return None
        magnitude = hypot(*vector)
        if magnitude <= 1e-8 or abs(dot(vector, plane.normal)) > 1e-7*magnitude:
            return None
        component = dot(vector, plane.u)*normal[0]+dot(vector, plane.v)*normal[1]
        return 1 if component >= 0 else -1

    # One circle per charge avoids redundant radial copies of the same line.
    for index, (position, charge) in enumerate(sites):
        x, y, h = plane.project(position)
        if abs(h) > 1e-10:
            continue
        radius = max((m.radius_m for m in context.masks if m.center_m == position), default=span*.002)*1.4
        group = []
        for fraction in _progressive():
            angle = 2*pi*(fraction+.03125)
            xy = (x+radius*cos(angle), y+radius*sin(angle))
            if valid(xy):
                group.append((xy, 1 if charge > 0 else -1, f'charge:{index}'))
        families.append(group)

    # Source-free slab: one face already spans the complete uniform field.
    empty_slab = not sites and len(context.problem.conductors) == 2 and all(
        isinstance(c, ConductingPlane) for c in context.problem.conductors)
    for ci, conductor in enumerate(context.problem.conductors):
        if empty_slab and ci == 1:
            continue
        geom = slice_geometry(conductor, plane)
        if not geom['visible']:
            continue
        if isinstance(conductor, ConductingPlane):
            nx, ny = geom['normal_xy']
            norm = hypot(nx, ny)
            normal = (nx/norm, ny/norm)
            tangent = (-normal[1], normal[0])
            point = geom['point_m']
            widths = [abs(other.signed_distance_m(conductor.point_m)) for other in context.problem.conductors
                      if isinstance(other, ConductingPlane) and other is not conductor]
            offset = min([span*.002]+[w*.01 for w in widths if w > 0])
            point = (point[0]+offset*normal[0], point[1]+offset*normal[1])
            interval = _segment_in_view(point, tangent, plane)
            group = []
            if interval:
                lo, hi = interval
                for fraction in _progressive():
                    t = lo+(hi-lo)*(.03+.94*fraction)
                    xy = (point[0]+t*tangent[0], point[1]+t*tangent[1])
                    direction = away_from_face(xy, normal)
                    if direction is not None:
                        group.append((xy, direction, f'face:{conductor.name}:outer'))
            families.append(group)
        else:
            cx, cy = geom['center_m']
            for face in ('inner', 'outer'):
                radius = geom[f'{face}_radius_m']
                if radius <= 0:
                    continue
                sign = -1 if face == 'inner' else 1
                offset = min(span*.002, radius*.01)
                group = []
                for fraction in _progressive():
                    angle = 2*pi*(fraction+.03125)
                    normal = (sign*cos(angle), sign*sin(angle))
                    xy = (cx+(radius+sign*offset)*cos(angle), cy+(radius+sign*offset)*sin(angle))
                    direction = away_from_face(xy, normal)
                    if direction is not None:
                        group.append((xy, direction, f'face:{conductor.name}:{face}'))
                families.append(group)
    out = []
    for row in zip_longest(*families):
        for item in row:
            if item is not None:
                out.append(item)
                if len(out) == limit:
                    return out
    return out


def _path_samples(points, count=13):
    lengths = [0.]
    for a, b in zip(points, points[1:]):
        lengths.append(lengths[-1]+hypot(b[0]-a[0], b[1]-a[1]))
    if not lengths[-1]:
        return [points[0]]*count
    out, k = [], 0
    for j in range(count):
        distance = lengths[-1]*j/(count-1)
        while k < len(points)-2 and lengths[k+1] < distance:
            k += 1
        width = lengths[k+1]-lengths[k]
        t = (distance-lengths[k])/width if width else 0.
        out.append(tuple(points[k][i]+t*(points[k+1][i]-points[k][i]) for i in range(2)))
    return out


def distinct_pool(lines, plane):
    """Suppress near-identical complete curves, retaining deterministic order."""
    tolerance = .004*min(plane.x_range_m[1]-plane.x_range_m[0], plane.y_range_m[1]-plane.y_range_m[0])
    kept, samples = [], []
    for line in lines:
        curve = _path_samples(line['points_m'])
        if any(all(hypot(a[0]-b[0], a[1]-b[1]) < tolerance for a, b in zip(curve, previous))
               for previous in samples):
            continue
        kept.append(line)
        samples.append(curve)
    return kept
