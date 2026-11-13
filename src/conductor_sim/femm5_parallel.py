"""Real FEMM 5 axisymmetric two-electrode slab, finite radial domain.

Two facing ideal ConductingPlane components are compiled as *finite disks*
at z=0 and z=d in a cylindrical FEMM meridian (0 <= r <= R). The side
at r=R has the natural homogeneous Neumann condition (zero radial flux).
This artificial boundary keeps the source-free uniform capacitor exact and
permits independent finite-disk charge checks. It is not an infinite-plate
FEMM model or a general non-axisymmetric solver.

All real charges become independently charged small FEMM spheres on one
normal line. Any off-axis active source is rejected before a FEMM session
is started. Numerical outputs are measured by FEMM, not substituted by
the separate ParallelPlateBackend image reference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import fsum, hypot, isclose, isfinite, pi
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

from ._resources import cleanup_resources
from .femm5 import (
    Femm5ExecutionError, Femm5Session, Femm5Settings, Femm5Unavailable,
    NumericalSource, SourceCore, _as_finite_values, _dot, _half_circle,
    _label_air, _mark_arcs, _perpendicular,
)
from .model import EPSILON_0, ConductingPlane, ElectrostaticProblem, Vector3, _position
from .parallel_plates import _layout as _analytic_layout, _voltage
from .solver import ConductorResult, OutsideDomain, SingularField, UnsupportedProblem

_SOURCE_PREFIX = "__conductor_sim_plate_pair_source"
_MAX_ACTIVE_SITES = 16


def _diff(a, b):
    return tuple(a[i]-b[i] for i in range(3))


def _layout(problem: ElectrostaticProblem, settings: Femm5Settings):
    """Pre-session validation: real axisymmetry, finite geometry, distinct cores."""
    lower, upper, gap, sites = _analytic_layout(problem)
    for p in (lower,upper):
        if p.name == "<None>" or p.name.startswith("__conductor_sim_"):
            raise UnsupportedProblem("FEMM plate name is reserved")
    if len(sites) > _MAX_ACTIVE_SITES:
        raise UnsupportedProblem("FEMM parallel plates permit at most 16 active axial sites")
    axis = lower.normal
    foot = None
    ordered = []
    for site in sites:
        h = lower.signed_distance_m(site.position_m)
        candidate = tuple(site.position_m[i]-h*axis[i] for i in range(3))
        if not all(isfinite(x) for x in candidate):
            raise UnsupportedProblem("FEMM source footpoint is nonfinite")
        if foot is None:
            foot = candidate
        # Distinct transverse offsets cause a 3D non-axisymmetric problem:
        # arbitrary collinearity alone is not sufficient.
        if hypot(*(candidate[i]-foot[i] for i in range(3))) > max(1e-13,gap*1e-11):
            raise UnsupportedProblem("FEMM parallel-plate sources require one common normal-line footpoint")
        ordered.append((h,site))
    if foot is None:
        foot = lower.point_m  # source-free uniform slab: symmetry origin arbitrary
    ordered.sort(key=lambda item:item[0])
    R = gap*settings.outer_radius_factor
    if not isfinite(R) or R <= 2*gap:
        raise UnsupportedProblem("FEMM parallel-plate radial domain cannot be constructed")
    cores=[]
    for i,(h,site) in enumerate(ordered):
        clear=min(h,gap-h)
        next_distance=min((abs(h-other_h) for j,(other_h,_) in enumerate(ordered) if i!=j),
                          default=float("inf"))
        rc=min(clear*settings.source_radius_fraction, gap/20, next_distance/4)
        if not isfinite(rc) or rc <= 0 or h-rc <= 0 or h+rc >= gap:
            raise UnsupportedProblem("FEMM parallel-plate source core lacks slab clearance")
        name=f"{_SOURCE_PREFIX}_{i}"
        cores.append(SourceCore(name,site.position_m,h,rc,site.charge_c))
    return R, foot, axis, gap, tuple(cores)


def _compile(client:Any, problem:ElectrostaticProblem, settings:Femm5Settings, layout):
    """Rectangle r in [0,R], z in [0,d] with two physical electrode disks."""
    lower,upper=problem.conductors
    R,foot,axis,d,cores=layout
    client.newdocument(1)
    client.ei_probdef("meters","axi",settings.solver_precision)
    eps=problem.permittivity_f_per_m
    client.ei_addmaterial("Vacuum",eps/EPSILON_0,eps/EPSILON_0,0)
    for plate in (lower,upper):
        client.ei_addconductorprop(plate.name,_voltage(plate),0,1)
    for source in cores:
        client.ei_addconductorprop(source.name,0,source.prescribed_charge_c,0)

    # The actual FEMM geometry has a finite radius R and insulating side.
    # No ei_setsegmentprop call is made for r=R; FEMM's natural boundary
    # condition is n dot D = 0. Do NOT turn the side into another conductor.
    radial=sorted({0.,d/8,d/3,d,R/2,R})
    axial=sorted({0.,d/2,d, *[c.axial_offset_m+sign*c.radius_m
                              for c in cores for sign in (-1,1)]})
    for r in radial:
        client.ei_addnode(r,0)
        client.ei_addnode(r,d)
    for z in axial:
        client.ei_addnode(0,z)
    for lo,hi in zip(radial,radial[1:]):
        for plate,z in ((lower,0.),(upper,d)):
            client.ei_addsegment(lo,z,hi,z)
            client.ei_selectsegment((lo+hi)/2,z)
            length=min((hi-lo)/2,d/12 if hi <= d else R/24)
            client.ei_setsegmentprop("<None>",length*settings.mesh_scale,0,0,0,plate.name)
            client.ei_clearselected()
    for lo,hi in zip(axial,axial[1:]):
        client.ei_addsegment(0,lo,0,hi)
    client.ei_addsegment(R,0,R,d) # natural insulated cylindrical side

    for source in cores:
        h,rc=source.axial_offset_m,source.radius_m
        client.ei_addnode(0,h-rc)
        client.ei_addnode(rc,h)
        client.ei_addnode(0,h+rc)
        _half_circle(client,rc,settings.arc_degrees,h)
        _mark_arcs(client,rc,settings.arc_degrees,"<None>",source.name,h)
        _label_air(client,rc/2,max(rc/6*settings.mesh_scale,rc/40),z=h)
    _label_air(client,R/3,min(d/12*settings.mesh_scale,R/20),z=d/2)
    return R,foot,axis,d,cores


@dataclass(slots=True)
class Femm5ParallelPlateSolution:
    """Live finite-disk measurement, with independently labeled FEMM source cores."""
    lower: ConductingPlane
    upper: ConductingPlane
    _origin_m: Vector3
    outer_radius_m: float
    gap_m: float
    conductor_results: tuple[ConductorResult, ...]
    _session: Any = field(repr=False)
    _client: Any = field(repr=False)
    _workspace: TemporaryDirectory[str] = field(repr=False)
    numerical_sources: tuple[NumericalSource, ...] = ()
    symmetry_axis: Vector3 = (0.,0.,1.)
    permittivity_f_per_m: float = EPSILON_0
    _closed: bool = field(default=False,init=False,repr=False)

    def __enter__(self):
        if self._closed:
            raise RuntimeError("FEMM parallel-plate solution is closed")
        return self

    def __exit__(self,exc_type,exc,tb):
        cleanup_resources((("FEMM parallel-plate solution",self.close),),primary=exc)

    def close(self):
        if not self._closed:
            self._closed=True
            cleanup_resources((("FEMM parallel-plate session",self._session.close),
                               ("FEMM parallel-plate workspace",self._workspace.cleanup)))

    def __del__(self):
        if not getattr(self,"_closed",True):
            try:self.close()
            except Exception:pass

    def _coord(self,point_m):
        if self._closed:
            raise RuntimeError("FEMM parallel-plate solution is closed")
        point=_position(point_m,"FEMM parallel-plate sample")
        delta=_diff(point,self._origin_m)
        z=_dot(delta,self.symmetry_axis)
        perp=tuple(delta[i]-z*self.symmetry_axis[i] for i in range(3))
        r=hypot(*perp)
        if r>=self.outer_radius_m:
            raise OutsideDomain("sample lies outside finite FEMM cylindrical radius")
        if 0 < z < self.gap_m:
            for core in self.numerical_sources:
                if hypot(r,z-core.axial_offset_m) <= core.radius_m:
                    raise UnsupportedProblem("sample lies inside/on a numerical charged core")
        return r,z,perp

    def potential_at(self,point_m):
        r,z,_=self._coord(point_m)
        if z<=0:
            return _voltage(self.lower)
        if z>=self.gap_m:
            return _voltage(self.upper)
        return _as_finite_values(self._client.eo_getpointvalues(r,z),8,
                                 "FEMM parallel-plate potential")[0]

    def electric_field_at(self,point_m):
        r,z,perp=self._coord(point_m)
        if isclose(z,0.,abs_tol=1e-14) or isclose(z,self.gap_m,abs_tol=1e-14):
            raise SingularField("electric field has distinct one-sided limits at plates")
        if z<0 or z>self.gap_m:
            return (0.,0.,0.)
        values=_as_finite_values(self._client.eo_getpointvalues(r,z),8,
                                 "FEMM parallel-plate electric field")
        er,ez=values[3],values[4]
        return tuple(ez*self.symmetry_axis[i]+(er*perp[i]/r if r else 0.0)
                     for i in range(3))

    def surface_charge_density_at(self,surface,point_m,*,conductor_name=None,
                                  offset_fraction=.02):
        if surface!="outer":
            raise ValueError("parallel plates have one vacuum-facing surface each")
        if conductor_name not in (self.lower.name,self.upper.name):
            raise ValueError("conductor_name must identify one parallel plate")
        plate=self.lower if conductor_name==self.lower.name else self.upper
        point=_position(point_m,"FEMM plate surface point")
        if abs(plate.signed_distance_m(point))>1e-11:
            raise ValueError("density point must lie on requested plane")
        if not isfinite(offset_fraction) or not 0<offset_fraction<=.1:
            raise ValueError("offset_fraction must lie in (0,.1]")
        nearest=min((min(c.axial_offset_m,self.gap_m-c.axial_offset_m)-c.radius_m
                     for c in self.numerical_sources),default=self.gap_m)
        offset=min(self.gap_m*offset_fraction,nearest/5)
        sample=tuple(point[i]+offset*plate.normal[i] for i in range(3))
        return self.permittivity_f_per_m*_dot(self.electric_field_at(sample),plate.normal)

    def surface_charge_density(self,surface,radial_distance_m,*,conductor_name,
                               offset_fraction=.02):
        if not isfinite(radial_distance_m) or radial_distance_m<0:
            raise ValueError("radial distance must be finite and nonnegative")
        if conductor_name not in (self.lower.name,self.upper.name):
            raise ValueError("unknown plate")
        tangent=_perpendicular(self.symmetry_axis)
        plate=self.lower if conductor_name==self.lower.name else self.upper
        point=tuple(self._origin_m[i]+(self.gap_m if plate is self.upper else 0)*
                    self.symmetry_axis[i]+radial_distance_m*tangent[i] for i in range(3))
        return self.surface_charge_density_at(surface,point,conductor_name=conductor_name,
                                              offset_fraction=offset_fraction)


class Femm5ParallelPlateBackend:
    """FEMM rectangle with two fixed-potential disks and an insulated side."""

    def __init__(self,*,settings:Femm5Settings|None=None,
                 binding_path:str|Path|None=None,
                 session_factory:Callable[[],Any]|None=None):
        self.settings=settings or Femm5Settings()
        self._session_factory=session_factory or (lambda:Femm5Session(binding_path))

    def supports(self,problem:ElectrostaticProblem)->bool:
        try:
            _layout(problem,self.settings)
            return True
        except (UnsupportedProblem,ValueError,OverflowError,ZeroDivisionError):
            return False

    def solve(self,problem:ElectrostaticProblem)->Femm5ParallelPlateSolution:
        layout=_layout(problem,self.settings) # strictly before FEMM launch
        session=self._session_factory()
        workspace=TemporaryDirectory(prefix="conductor_sim_femm_parallel_")
        try:
            client=session.open()
            R,foot,axis,d,cores=_compile(client,problem,self.settings,layout)
            client.ei_saveas((Path(workspace.name)/"parallel_plates.fee").as_posix())
            client.ei_analyze(1)
            client.ei_loadsolution()
            measured=[]
            for plate in problem.conductors:
                voltage,charge=_as_finite_values(client.eo_getconductorproperties(plate.name),
                                                 2,"FEMM plate "+plate.name)
                measured.append(ConductorResult(plate.name,voltage,charge,0.,charge))
            numerical=[]
            for c in cores:
                _,charge=_as_finite_values(client.eo_getconductorproperties(c.name),
                                           2,"measured FEMM source "+c.name)
                numerical.append(NumericalSource(c.name,c.position_m,c.axial_offset_m,
                                                  c.radius_m,charge))
            return Femm5ParallelPlateSolution(problem.conductors[0],problem.conductors[1],
                                               foot,R,d,tuple(measured),session,client,
                                               workspace,tuple(numerical),axis,
                                               problem.permittivity_f_per_m)
        except BaseException as exc:
            error=exc
            if isinstance(exc,Exception) and not isinstance(exc,(
                    Femm5Unavailable,Femm5ExecutionError,UnsupportedProblem)):
                error=Femm5ExecutionError(f"FEMM 5 parallel-plate solve failed: {exc}")
            cleanup_resources((("FEMM parallel-plate session",session.close),
                               ("FEMM parallel-plate workspace",workspace.cleanup)),
                              primary=error)
            if error is exc:
                raise
            raise error from exc
