"""Actual FEMM 5 axisymmetric *finite-domain* reference for one grounded plane.

A true infinite plane cannot be meshed. We mesh its upper vacuum half-space
in a large, finite hemispherical domain with a dipolar Robin outer boundary.
The grounded planar segment itself carries one FEMM conductor property.
FEMM's numerical charge source is a finite charged conductor, not a 2D point
that would unintentionally revolve into a charged ring.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot, isclose, isfinite, sqrt, pi
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

from ._resources import cleanup_resources
from .femm5 import (Femm5ExecutionError, Femm5Session, Femm5Settings,
                    Femm5Unavailable, NumericalSource, _as_finite_values,
                    _dot, _half_circle, _label_air, _mark_arcs, _norm,
                    _perpendicular)
from .grounded_plane import GroundedPlaneBackend
from .model import EPSILON_0, ConductingPlane, ElectrostaticProblem, Vector3, _position
from .solver import ConductorResult, SingularField, UnsupportedProblem

_SOURCE_NAME = "__conductor_sim_plane_source"


def _diff(a, b):
    return tuple(a[i]-b[i] for i in range(3))


def _compile(client: Any, problem: ElectrostaticProblem, settings: Femm5Settings):
    plane, charge = problem.conductors[0], problem.charges[0]
    n = plane.normal
    h = plane.signed_distance_m(charge.position_m)
    # Translate axis to the charge's orthogonal projection on the plane.
    # No authored coordinate alignment is needed for one point source.
    foot = tuple(charge.position_m[i] - h*n[i] for i in range(3))
    rc = min(h*settings.source_radius_fraction, h/5)
    R = h*settings.outer_radius_factor
    if not all(isfinite(v) and v>0 for v in (h,rc,R)) or R <= h+2*rc:
        raise UnsupportedProblem("cannot construct a finite FEMM plane domain")
    client.newdocument(1)
    client.ei_probdef("meters", "axi", settings.solver_precision)
    epsilon = problem.permittivity_f_per_m
    client.ei_addmaterial("Vacuum",epsilon/EPSILON_0,epsilon/EPSILON_0,0)
    client.ei_addconductorprop(plane.name,0,0,1)   # FEMM's grounded plane segment
    client.ei_addconductorprop(_SOURCE_NAME,0,charge.charge_c,0)
    # The grounded-plane image solution is asymptotically dipolar: V~R^-2.
    # This Robin coefficient targets that decay, not the monopole sphere case.
    client.ei_addboundprop("open_dipole",0,0,2*epsilon/R,0,1)
    # Quadrant of the r>=0, z>=0 axisymmetric cross section: one grounded
    # horizontal segment, one symmetry axis, and one curved exterior arc.
    radial = (0.0,h,3*h,R)
    axial = (0.0,h-rc,h+rc,R)
    for r in radial:
        client.ei_addnode(r,0)
    for z in axial:
        client.ei_addnode(0,z)
    for lo,hi in zip(radial,radial[1:]):
        client.ei_addsegment(lo,0,hi,0)
        client.ei_selectsegment((lo+hi)/2,0)
        # Refine the near-charge plane region, but permit coarse far-field mesh.
        length = min(max(hi-lo, h/16), h/6 if hi<=3*h else R/24)
        client.ei_setsegmentprop("<None>",length*settings.mesh_scale,0,0,0,plane.name)
        client.ei_clearselected()
    for lo,hi in zip(axial,axial[1:]):
        client.ei_addsegment(0,lo,0,hi)
    client.ei_addarc(R,0,0,R,90,settings.arc_degrees)
    client.ei_selectarcsegment(R/sqrt(2),R/sqrt(2))
    client.ei_setarcsegmentprop(settings.arc_degrees,"open_dipole",0,0,"<None>")
    client.ei_clearselected()
    for r,z in ((0,h-rc),(rc,h),(0,h+rc)):
        client.ei_addnode(r,z)
    _half_circle(client,rc,settings.arc_degrees,h)
    _mark_arcs(client,rc,settings.arc_degrees,"<None>",_SOURCE_NAME,h)
    _label_air(client,rc/2,rc/6*settings.mesh_scale,z=h)
    _label_air(client,h/2,h/10*settings.mesh_scale,z=h/3)
    _label_air(client,R*.45,h/2*settings.mesh_scale,z=R*.25)
    return R,rc,n,foot,h


@dataclass(slots=True)
class Femm5PlaneSolution:
    plane: ConductingPlane
    source_position_m: Vector3
    _origin_m: Vector3
    outer_radius_m: float
    conductor_results: tuple[ConductorResult, ...]
    _session: Any = field(repr=False)
    _client: Any = field(repr=False)
    _workspace: TemporaryDirectory[str] = field(repr=False)
    numerical_sources: tuple[NumericalSource, ...] = ()
    source_radius_m: float | None = None
    numerical_source_charge_c: float | None = None
    symmetry_axis: Vector3 = (0.,0.,1.)
    permittivity_f_per_m: float = EPSILON_0
    _closed: bool = field(default=False, init=False, repr=False)

    def __enter__(self):
        if self._closed:
            raise RuntimeError("FEMM plane solution has been closed")
        return self

    def __exit__(self, exc_type, exc, tb):
        cleanup_resources((("FEMM plane solution", self.close),), primary=exc)

    def close(self):
        if not self._closed:
            self._closed=True
            cleanup_resources((("FEMM plane session",self._session.close),
                               ("FEMM plane workspace",self._workspace.cleanup)))

    def __del__(self):
        if not getattr(self,"_closed",True):
            try: self.close()
            except Exception: pass

    def _coord(self,point_m):
        if self._closed:
            raise RuntimeError("FEMM plane solution has been closed")
        point = _position(point_m,"FEMM plane sample point")
        delta = _diff(point,self._origin_m)
        z = _dot(delta,self.symmetry_axis)
        perp = tuple(delta[i]-z*self.symmetry_axis[i] for i in range(3))
        r = hypot(*perp)
        if hypot(r,z) >= self.outer_radius_m:
            raise UnsupportedProblem("sample lies outside finite FEMM plane domain")
        for core in self.numerical_sources:
            if hypot(r,z-core.axial_offset_m)<=core.radius_m:
                raise UnsupportedProblem("sample lies inside/on the finite source core")
        return r,z,perp

    def potential_at(self,point_m):
        r,z,_ = self._coord(point_m)
        if z<=0:
            return 0.0  # prescribed grounded boundary and conductor half-space
        return _as_finite_values(self._client.eo_getpointvalues(r,z),8,
                                 "FEMM plane potential")[0]

    def electric_field_at(self,point_m):
        r,z,perp = self._coord(point_m)
        if isclose(z,0.,abs_tol=1e-14):
            raise SingularField("field has a discontinuity on conducting plane")
        for core in self.numerical_sources:
            if isclose(hypot(r,z-core.axial_offset_m),core.radius_m,rel_tol=1e-12):
                raise SingularField("field has a discontinuity on numerical core")
        if z<0:
            return (0.,0.,0.)
        values=_as_finite_values(self._client.eo_getpointvalues(r,z),8,
                                 "FEMM plane electric field")
        er,ez=values[3],values[4]
        return tuple(ez*self.symmetry_axis[i]+(er*perp[i]/r if r else 0.0)
                     for i in range(3))

    def surface_charge_density_at(self,surface,point_m,*,conductor_name=None,
                                  offset_fraction=.02):
        if surface!="outer":
            raise ValueError("grounded plane exposes only an outer surface")
        if conductor_name not in (None,self.plane.name):
            raise ValueError("unknown plane conductor")
        point=_position(point_m,"plane surface point")
        if not isclose(self.plane.signed_distance_m(point),0.,abs_tol=1e-12):
            raise ValueError("density sample must lie on conducting plane")
        if not isfinite(offset_fraction) or not 0<offset_fraction<=.1:
            raise ValueError("offset_fraction must be in (0,.1]")
        core=self.numerical_sources[0]
        h=core.axial_offset_m
        gap=min(h*offset_fraction,(h-core.radius_m)/5)
        sample=tuple(point[i]+gap*self.symmetry_axis[i] for i in range(3))
        return self.permittivity_f_per_m*_dot(self.electric_field_at(sample),
                                               self.symmetry_axis)

    def surface_charge_density(self,surface,radial_distance_m,*,offset_fraction=.02):
        if not isfinite(radial_distance_m) or radial_distance_m<0:
            raise ValueError("radial distance must be finite and nonnegative")
        tangent=_perpendicular(self.symmetry_axis)
        point=tuple(self._origin_m[i]+radial_distance_m*tangent[i] for i in range(3))
        return self.surface_charge_density_at(surface,point,offset_fraction=offset_fraction)


class Femm5PlaneBackend:
    """A real axisymmetric FEMM solve for one charge above a grounded plane."""
    def __init__(self,*,settings: Femm5Settings|None=None,
                 binding_path: str|Path|None=None,
                 session_factory: Callable[[],Any]|None=None):
        self.settings = settings or Femm5Settings()
        self._session_factory = session_factory or (lambda: Femm5Session(binding_path))

    def supports(self,problem: ElectrostaticProblem) -> bool:
        if not GroundedPlaneBackend().supports(problem):
            return False
        plane=problem.conductors[0]
        active = [q for q in problem.charges if q.charge_c != 0]
        return (len(problem.charges)==1 and len(active)==1 and
                plane.name != _SOURCE_NAME and
                not plane.name.startswith("__conductor_sim_") and
                plane.name != "<None>")

    def solve(self,problem: ElectrostaticProblem) -> Femm5PlaneSolution:
        if not self.supports(problem):
            raise UnsupportedProblem("FEMM plane requires one grounded infinite-plane "
                                     "component and one nonzero source on its vacuum side; "
                                     "multiple or other 3D source arrangements need analytical/3D support")
        session=self._session_factory()
        workspace=TemporaryDirectory(prefix="conductor_sim_femm_plane_")
        try:
            client=session.open()
            R,rc,axis,foot,h = _compile(client,problem,self.settings)
            client.ei_saveas((Path(workspace.name)/"grounded_plane.fee").as_posix())
            client.ei_analyze(1)
            client.ei_loadsolution()
            plane=problem.conductors[0]
            _,charge_measured=_as_finite_values(client.eo_getconductorproperties(_SOURCE_NAME),2,
                                                  "plane numerical source charge")
            plane_v,plane_charge=_as_finite_values(client.eo_getconductorproperties(plane.name),2,
                                                   "truncated grounded plane charge")
            source=problem.charges[0]
            numerical=(NumericalSource(_SOURCE_NAME,source.position_m,h,rc,charge_measured),)
            results=(ConductorResult(plane.name,plane_v,plane_charge,0.,plane_charge),)
            return Femm5PlaneSolution(plane,source.position_m,foot,R,results,
                                      session,client,workspace,numerical,
                                      rc,charge_measured,axis,problem.permittivity_f_per_m)
        except BaseException as exc:
            error=exc
            if isinstance(exc,Exception) and not isinstance(exc,(
                    Femm5Unavailable,Femm5ExecutionError,UnsupportedProblem)):
                error=Femm5ExecutionError(f"FEMM 5 grounded plane solve failed: {exc}")
            cleanup_resources((("FEMM plane session",session.close),
                               ("FEMM plane workspace",workspace.cleanup)),primary=error)
            if error is exc:
                raise
            raise error from exc
