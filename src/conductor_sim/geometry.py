"""Reusable solver-independent conductor geometry and 3D field-trace barriers.

Geometry is responsible for inside/outside tests, exact segment crossings,
and cross-section metadata. It never computes an electrostatic solution.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import hypot, sqrt, isfinite

from .model import ConductingPlane, SphericalShell, SolidSphere, Vector3


def _sub(a, b):
    return tuple(x-y for x, y in zip(a,b))


def _dot(a, b):
    return sum(x*y for x, y in zip(a,b))


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
        proj = _dot(f,d)
        discr = proj*proj-l2*(_dot(f,f)-self.radius_m*self.radius_m)
        if discr < 0:
            return None
        hits = [t for t in ((-proj-sqrt(discr))/l2,(-proj+sqrt(discr))/l2)
                if 1e-10 < t <= 1]
        return min(hits) if hits else None


@dataclass(frozen=True)
class PlaneBoundary:
    point_m: Vector3
    normal: Vector3
    kind: str = "surface"

    def first_hit(self,a: Vector3,b: Vector3):
        h0 = _dot(_sub(a,self.point_m),self.normal)
        h1 = _dot(_sub(b,self.point_m),self.normal)
        if h0 == h1 or h0*h1 > 0:
            return None
        t = h0/(h0-h1)
        return t if 1e-10 < t <= 1 else None


def contains_metal(shape, point: Vector3) -> bool:
    if isinstance(shape, ConductingPlane):
        return _dot(_sub(point,shape.point_m),shape.normal) <= 0
    if isinstance(shape,(SphericalShell,SolidSphere)):
        r = hypot(*_sub(point,shape.center_m))
        return shape.inner_radius_m <= r <= shape.outer_radius_m
    raise TypeError(f"unsupported physical conductor shape: {type(shape).__name__}")


def physical_boundaries(shape):
    if isinstance(shape,ConductingPlane):
        return (PlaneBoundary(shape.point_m,shape.normal),)
    if isinstance(shape,(SphericalShell,SolidSphere)):
        return tuple(SphereBoundary(shape.center_m,r) for r in
                     (shape.inner_radius_m,shape.outer_radius_m) if r>0)
    raise TypeError(f"unsupported physical conductor shape: {type(shape).__name__}")


def slice_geometry(shape, plane):
    """Serialisable cross-section (sphere rings or a planar half-space)."""
    if isinstance(shape,ConductingPlane):
        nx = _dot(shape.normal,plane.u)
        ny = _dot(shape.normal,plane.v)
        d0 = _dot(_sub(plane.origin_m,shape.point_m),shape.normal)
        denom = nx*nx+ny*ny
        if denom < 1e-20:
            return {"kind":"plane","name":shape.name,"visible":False,
                    "point_m":[0.,0.],"normal_xy":[0.,0.]}
        xy = [-d0*nx/denom,-d0*ny/denom]
        return {"kind":"plane","name":shape.name,"visible":True,
                "point_m":xy,"normal_xy":[nx,ny]}
    if isinstance(shape,(SphericalShell,SolidSphere)):
        x,y,h = plane.project(shape.center_m)
        return {"kind":"sphere","name":shape.name,"center_m":[x,y],
                "distance_to_plane_m":h,
                "inner_radius_m":sqrt(max(0,shape.inner_radius_m**2-h*h)),
                "outer_radius_m":sqrt(max(0,shape.outer_radius_m**2-h*h)),
                "visible":abs(h)<shape.outer_radius_m}
    raise TypeError(f"unsupported physical conductor shape: {type(shape).__name__}")
