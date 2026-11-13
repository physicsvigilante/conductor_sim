"""Actual FEMM 5 axisymmetric *finite-domain* solver for one grounded plane.

A true infinite plane cannot be meshed. We mesh its upper vacuum half-space
in a large, finite hemispherical domain with a dipolar Robin outer boundary.
The grounded planar segment itself carries one FEMM conductor property.
Each grouped active charge on the same plane-normal symmetry axis becomes an
independent finite FEMM charged sphere, never a 2D ring. Non-axisymmetric
source arrangements are explicitly refused before opening any FEMM session.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot, isclose, isfinite, sqrt, pi
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

from ._resources import cleanup_resources
from .femm5 import (Femm5ExecutionError, Femm5Session, Femm5Settings,
                    Femm5Unavailable, NumericalSource, SourceCore, _as_finite_values,
                    _dot, _half_circle, _label_air, _mark_arcs, _norm,
                    _perpendicular)
from .grounded_plane import GroundedPlaneBackend, _sites
from .model import EPSILON_0, ConductingPlane, ElectrostaticProblem, Vector3, _position
from .solver import ConductorResult, OutsideDomain, SingularField, UnsupportedProblem

_SOURCE_NAME = "__conductor_sim_plane_source"


def _diff(a, b):
    return tuple(a[i]-b[i] for i in range(3))


# A finite axisymmetric plane is symmetric under rotations about the *normal*.
# Merely collinear point positions along a tilted line do not preserve the
# conducting plane. Small roundoff in an authored world transform is tolerated.
_MAX_ACTIVE_SITES = 16


def _layout(problem: ElectrostaticProblem, settings: Femm5Settings):
    """Validate numerical capability and compile a bounded, source-only layout.

    This runs before any FEMM session is started, so unsupported physical
    arrangements never leave a partial numerical solve behind.
    """
    if not GroundedPlaneBackend().supports(problem):
        raise UnsupportedProblem('FEMM grounded plane requires one plane and charges strictly on its vacuum side')
    plane = problem.conductors[0]
    if plane.name == '<None>' or plane.name.startswith('__conductor_sim_'):
        raise UnsupportedProblem('FEMM plane conductor name is reserved')
    sites = _sites(problem)  # group coincident sites, preserving small signed residuals
    if not 1 <= len(sites) <= _MAX_ACTIVE_SITES:
        raise UnsupportedProblem('FEMM plane permits 1–16 distinct nonzero charge sites')
    axis = plane.normal
    foot = None
    arranged = []
    for site in sites:
        h = plane.signed_distance_m(site.position_m)
        candidate = tuple(site.position_m[i]-h*axis[i] for i in range(3))
        if not isfinite(h) or h <= 0 or not all(isfinite(x) for x in candidate):
            raise UnsupportedProblem('source geometry is not finite and strictly above the plane')
        if foot is None:
            foot = candidate
        # Only coordinate roundoff may be ignored; finite transverse offsets
        # must be solved by the independent 3D analytical reference instead.
        if hypot(*(candidate[i]-foot[i] for i in range(3))) > max(1e-13, h*1e-11):
            raise UnsupportedProblem('non-axisymmetric plane sources must share one normal-line footpoint')
        arranged.append((h,site))
    arranged.sort(key=lambda x: x[0])
    heights = [h for h,_ in arranged]
    R = max(heights)*settings.outer_radius_factor
    if not isfinite(R) or R <= max(heights):
        raise UnsupportedProblem('cannot construct a finite FEMM plane domain')
    cores = []
    for i,(h,site) in enumerate(arranged):
        neighbor = min((abs(h-other_h) for j,(other_h,_) in enumerate(arranged)
                        if i != j),default=float('inf'))
        rc = min(h*settings.source_radius_fraction,h/5,neighbor/4,(R-h)/5)
        if not isfinite(rc) or rc <= 0 or h+rc >= R:
            raise UnsupportedProblem('FEMM source sites lack clearance for separate finite cores')
        name = _SOURCE_NAME if i == 0 else f'{_SOURCE_NAME}_{i}'
        cores.append(SourceCore(name,site.position_m,h,rc,site.charge_c))
    return R,foot,axis,tuple(cores)


def _compile(client: Any, problem: ElectrostaticProblem, settings: Femm5Settings, layout):
    plane = problem.conductors[0]
    R, foot, axis, cores = layout
    h_min = cores[0].axial_offset_m
    client.newdocument(1)
    client.ei_probdef('meters','axi',settings.solver_precision)
    epsilon = problem.permittivity_f_per_m
    client.ei_addmaterial('Vacuum',epsilon/EPSILON_0,epsilon/EPSILON_0,0)
    client.ei_addconductorprop(plane.name,0,0,1)   # fixed 0 V, measured finite-disk charge
    for source in cores:
        client.ei_addconductorprop(source.name,0,source.prescribed_charge_c,0)
    # Each point and its grounded-plane image form a dipole at large radius.
    # The superposed dipolar Robin condition is approximate for finite R.
    client.ei_addboundprop('open_dipole',0,0,2*epsilon/R,0,1)
    # r>=0,z>=0 quadrant. The planar edge is ONE physical conductor.
    radial = sorted({0.,h_min,min(R*.4,3*h_min),R})
    axial = sorted({0., R, *[core.axial_offset_m + sign*core.radius_m
                             for core in cores for sign in (-1,1)]})
    for r in radial:
        client.ei_addnode(r,0)
    for z in axial:
        client.ei_addnode(0,z)
    for lo,hi in zip(radial,radial[1:]):
        client.ei_addsegment(lo,0,hi,0)
        client.ei_selectsegment((lo+hi)/2,0)
        length = min(max(hi-lo,h_min/16),h_min/6 if hi <= 3*h_min else R/24)
        client.ei_setsegmentprop('<None>',length*settings.mesh_scale,0,0,0,plane.name)
        client.ei_clearselected()
    for lo,hi in zip(axial,axial[1:]):
        client.ei_addsegment(0,lo,0,hi)
    client.ei_addarc(R,0,0,R,90,settings.arc_degrees)
    client.ei_selectarcsegment(R/sqrt(2),R/sqrt(2))
    client.ei_setarcsegmentprop(settings.arc_degrees,'open_dipole',0,0,'<None>')
    client.ei_clearselected()
    for source in cores:
        h,rc = source.axial_offset_m,source.radius_m
        for r,z in ((0,h-rc),(rc,h),(0,h+rc)):
            client.ei_addnode(r,z)
        _half_circle(client,rc,settings.arc_degrees,h)
        _mark_arcs(client,rc,settings.arc_degrees,'<None>',source.name,h)
        _label_air(client,rc/2,rc/6*settings.mesh_scale,z=h)
    _label_air(client,h_min/2,h_min/10*settings.mesh_scale,z=h_min/3)
    _label_air(client,R*.45,h_min/2*settings.mesh_scale,z=R*.25)
    return R,foot,axis,cores


@dataclass(slots=True)
class Femm5PlaneSolution:
    plane: ConductingPlane
    source_position_m: Vector3 | None
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
            raise OutsideDomain("sample lies outside finite FEMM plane domain")
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
        # Use the nearest source's clearance to avoid the finite charged cores.
        h=min(core.axial_offset_m for core in self.numerical_sources)
        clearance=min(core.axial_offset_m-core.radius_m for core in self.numerical_sources)
        gap=min(h*offset_fraction,clearance/5)
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
    """Real FEMM: up to 16 active point-charge sites on one plane normal."""
    def __init__(self,*,settings: Femm5Settings|None=None,
                 binding_path: str|Path|None=None,
                 session_factory: Callable[[],Any]|None=None):
        self.settings = settings or Femm5Settings()
        self._session_factory = session_factory or (lambda: Femm5Session(binding_path))

    def supports(self,problem: ElectrostaticProblem) -> bool:
        try:
            _layout(problem,self.settings)
            return True
        except (UnsupportedProblem,ValueError,OverflowError,ZeroDivisionError):
            return False

    def solve(self,problem: ElectrostaticProblem) -> Femm5PlaneSolution:
        layout = _layout(problem,self.settings)  # reject unsupported BEFORE session
        session=self._session_factory()
        workspace=TemporaryDirectory(prefix='conductor_sim_femm_plane_')
        try:
            client=session.open()
            R,foot,axis,cores = _compile(client,problem,self.settings,layout)
            client.ei_saveas((Path(workspace.name)/'grounded_plane.fee').as_posix())
            client.ei_analyze(1)
            client.ei_loadsolution()
            plane=problem.conductors[0]
            plane_v,plane_charge=_as_finite_values(client.eo_getconductorproperties(plane.name),2,
                                                   'truncated grounded plane charge')
            measured=[]
            for core in cores:
                _,charge_measured=_as_finite_values(client.eo_getconductorproperties(core.name),2,
                                                    'measured FEMM source '+core.name)
                measured.append(NumericalSource(core.name,core.position_m,core.axial_offset_m,
                                                core.radius_m,charge_measured))
            numerical=tuple(measured)
            results=(ConductorResult(plane.name,plane_v,plane_charge,0.,plane_charge),)
            # These legacy singular-source attributes are intentionally absent
            # on a multi-core result, rather than choosing an arbitrary source.
            source_position=cores[0].position_m if len(cores)==1 else None
            source_radius=cores[0].radius_m if len(cores)==1 else None
            measured_source=measured[0].measured_charge_c if len(cores)==1 else None
            return Femm5PlaneSolution(plane,source_position,foot,R,results,
                                      session,client,workspace,numerical,
                                      source_radius,measured_source,axis,problem.permittivity_f_per_m)
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
