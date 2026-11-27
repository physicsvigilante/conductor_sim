"""Reusable solver-independent conductor geometry and 3D field-trace barriers.

Geometry is responsible for inside/outside tests, exact segment crossings,
and cross-section metadata. It never computes an electrostatic solution.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import copysign, fsum, hypot, sqrt, isfinite
from sys import float_info

from .model import (
    ConductingPlane,
    SphericalShell,
    SolidSphere,
    Vector3,
    _plane_exact_numerator,
    _plane_raw_normal,
)

_MIN_NORMAL = float_info.min


def _sub(a, b):
    return tuple(x-y for x, y in zip(a,b))


def _dot(a, b):
    return sum(x*y for x, y in zip(a,b))


def _segment_roots(a, b, c):
    """Stable quadratic roots on (0,1]; never discard a positive early hit."""
    if a == 0:
        return ()
    discriminant = b*b-4*a*c
    if discriminant < 0:
        return ()
    q = -.5*(b+copysign(sqrt(discriminant),b))
    roots = (q/a,c/q) if q else (-b/(2*a),)
    return tuple(t for t in roots if 0 < t <= 1)


@dataclass(frozen=True)
class SphereBoundary:
    center_m: Vector3
    radius_m: float
    kind: str = "surface"

    def first_hit(self, a: Vector3, b: Vector3):
        d = _sub(b,a)
        l2 = _dot(d,d)
        if l2 == 0:
            return None
        f = _sub(a,self.center_m)
        distance = hypot(*f)
        c = (distance-self.radius_m)*(distance+self.radius_m)
        hits = _segment_roots(l2,2*_dot(f,d),c)
        return min(hits) if hits else None


def _fast_plane_distance(
    point: Vector3,
    point_m: Vector3,
    normal: Vector3,
    rn: Vector3,
) -> float | None:
    """Return fast binary64 signed distance only when strictly cancellation/underflow/overflow-free."""
    max_coord = max(
        (max(abs(point[i]), abs(point_m[i])) for i in range(3) if rn[i] != 0.0),
        default=0.0,
    )
    if max_coord == 0.0:
        return 0.0
    if not (
        _MIN_NORMAL * 1e4 <= max_coord <= 1e150
        and all((rn[i] == 0.0) or (abs(normal[i]) >= _MIN_NORMAL * 1e4) for i in range(3))
    ):
        return None
    terms = tuple((point[i] - point_m[i]) * normal[i] for i in range(3))
    if not all(isfinite(t) for t in terms):
        return None
    raw = fsum(terms)
    bound = max(
        (max(abs(point[i]), abs(point_m[i])) * abs(normal[i]) for i in range(3)),
        default=0.0,
    )
    if isfinite(raw) and abs(raw) >= 0.01 * bound and abs(raw) >= _MIN_NORMAL * 1e4:
        return raw
    return None


@dataclass(frozen=True)
class PlaneBoundary:
    point_m: Vector3
    normal: Vector3
    kind: str = "surface"
    raw_normal: Vector3 | None = None

    def first_hit(self, a: Vector3, b: Vector3):
        rn = _plane_raw_normal(self.normal, self.raw_normal)
        h0 = _fast_plane_distance(a, self.point_m, self.normal, rn)
        h1 = _fast_plane_distance(b, self.point_m, self.normal, rn)
        if h0 is not None and h1 is not None:
            if h0 == 0.0 or (h0 > 0.0 and h1 > 0.0) or (h0 < 0.0 and h1 < 0.0):
                return None
            if h1 == 0.0:
                return 1.0
            scale = max(abs(h0), abs(h1))
            sh0, sh1 = h0 / scale, h1 / scale
            t = sh0 / (sh0 - sh1)
            if 0.0 < t <= 1.0:
                return t
        # Exact unnormalized rational dot products: ||rn|| cancels identically
        # in the intersection ratio H0 / (H0 - H1), eliminating overflow and
        # underflow-to-zero failures at extreme scales.
        H0 = _plane_exact_numerator(a, self.point_m, self.normal, rn)
        H1 = _plane_exact_numerator(b, self.point_m, self.normal, rn)
        if H0 == 0 or (H0 > 0 and H1 > 0) or (H0 < 0 and H1 < 0):
            return None
        if H1 == 0:
            return 1.0
        t = float(H0 / (H0 - H1))
        return t if 0.0 < t <= 1.0 else None


@dataclass(frozen=True)
class CylinderSideBoundary:
    """Finite numerical cylinder side: radial boundary, not a conductor.

    Used to stop field lines before leaving the FEMM parallel-plate mesh.
    The physical plate barriers independently stop traces at z=0 and z=d.
    """
    origin_m: Vector3
    axis: Vector3
    radius_m: float
    min_height_m: float
    max_height_m: float
    kind: str = "outside-domain"

    def first_hit(self,a:Vector3,b:Vector3):
        delta=_sub(b,a)
        offset=_sub(a,self.origin_m)
        h0=_dot(offset,self.axis)
        dh=_dot(delta,self.axis)
        radial=tuple(offset[i]-h0*self.axis[i] for i in range(3))
        dr=tuple(delta[i]-dh*self.axis[i] for i in range(3))
        aa=_dot(dr,dr)
        if aa==0:
            return None
        bb=2*_dot(radial,dr)
        distance=hypot(*radial)
        cc=(distance-self.radius_m)*(distance+self.radius_m)
        hits=[]
        for t in _segment_roots(aa,bb,cc):
            if self.min_height_m<=h0+t*dh<=self.max_height_m:
                hits.append(t)
        return min(hits) if hits else None


def contains_metal(shape, point: Vector3) -> bool:
    if isinstance(shape, ConductingPlane):
        rn = _plane_raw_normal(shape.normal)
        fast = _fast_plane_distance(point, shape.point_m, shape.normal, rn)
        if fast is not None:
            return fast <= 0.0
        return _plane_exact_numerator(point, shape.point_m, shape.normal, rn) <= 0
    if isinstance(shape,(SphericalShell,SolidSphere)):
        r = hypot(*_sub(point,shape.center_m))
        return shape.inner_radius_m <= r <= shape.outer_radius_m
    raise TypeError(f"unsupported physical conductor shape: {type(shape).__name__}")


def physical_boundaries(shape):
    if isinstance(shape,ConductingPlane):
        return (PlaneBoundary(shape.point_m, shape.normal, raw_normal=shape._raw_normal),)
    if isinstance(shape,(SphericalShell,SolidSphere)):
        return tuple(SphereBoundary(shape.center_m,r) for r in
                     (shape.inner_radius_m,shape.outer_radius_m) if r>0)
    raise TypeError(f"unsupported physical conductor shape: {type(shape).__name__}")


def slice_geometry(shape, plane):
    """Serialisable cross-section (sphere rings or a planar half-space)."""
    if isinstance(shape,ConductingPlane):
        rn = _plane_raw_normal(shape.normal)
        fn = tuple(Fraction(x) for x in rn)
        fu = tuple(Fraction(x) for x in plane.u)
        fv = tuple(Fraction(x) for x in plane.v)
        fnx = sum(fn[i] * fu[i] for i in range(3))
        fny = sum(fn[i] * fv[i] for i in range(3))
        fd0 = sum(
            (Fraction(plane.origin_m[i]) - Fraction(shape.point_m[i])) * fn[i]
            for i in range(3)
        )
        fdenom = fnx * fnx + fny * fny
        if fdenom == 0:
            return {"kind":"plane","name":shape.name,"visible":False,
                    "fills_metal":fd0<=0,
                    "point_m":[0.,0.],"normal_xy":[0.,0.]}
        m = max(abs(fnx), abs(fny))
        sx, sy = float(fnx / m), float(fny / m)
        snorm = hypot(sx, sy)
        nx_u, ny_u = sx / snorm, sy / snorm
        fx0 = -fd0 * fnx / fdenom
        fy0 = -fd0 * fny / fdenom
        try:
            x0, y0 = float(fx0), float(fy0)
        except OverflowError:
            x0, y0 = float("inf"), float("inf")
        if not (isfinite(x0) and isfinite(y0)):
            return {"kind":"plane","name":shape.name,"visible":False,
                    "fills_metal":fd0<=0,
                    "point_m":[0.,0.],"normal_xy":[0.,0.]}
        return {"kind":"plane","name":shape.name,"visible":True,
                "point_m":[x0,y0],"normal_xy":[nx_u,ny_u]}
    if isinstance(shape,(SphericalShell,SolidSphere)):
        x,y,h = plane.project(shape.center_m)
        return {"kind":"sphere","name":shape.name,"center_m":[x,y],
                "distance_to_plane_m":h,
                "inner_radius_m":sqrt(max(0,shape.inner_radius_m**2-h*h)),
                "outer_radius_m":sqrt(max(0,shape.outer_radius_m**2-h*h)),
                "visible":abs(h)<shape.outer_radius_m}
    raise TypeError(f"unsupported physical conductor shape: {type(shape).__name__}")
