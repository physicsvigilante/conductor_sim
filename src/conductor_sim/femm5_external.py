"""Real FEMM 5 exterior-source solver for grounded or fixed-charge solid spheres.

This backend never delegates numeric queries to the exact image solver. It
constructs one real solid sphere and one finite charged source core outside
it, then returns values measured by FEMM's own electrostatic postprocessor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, fsum, hypot, isclose, isfinite, pi, sin, sqrt
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Literal

from .femm5 import (
    Femm5ExecutionError, Femm5Session, Femm5Settings, Femm5Unavailable,
    NumericalSource, _as_finite_values, _dot, _half_circle, _label_air,
    _mark_arcs, _norm, _perpendicular,
)
from .model import EPSILON_0, ElectrostaticProblem, Grounded, FixedNetCharge, SolidSphere, Vector3, _position
from .solver import ConductorResult, SingularField, UnsupportedProblem
from .external_isolated import ExternalSphereBackend
from .external_multi import active_external_sites, difference as _multi_difference, norm as _multi_norm

_SOURCE_NAME = '__conductor_sim_external_source'


def _difference(a: Vector3, b: Vector3) -> Vector3:
    return tuple(a[i]-b[i] for i in range(3))


@dataclass(frozen=True, slots=True)
class _ExternalSourceCore:
    """Prescribed finite source geometry BEFORE FEMM, never a measured result."""
    name: str
    position_m: Vector3
    axial_offset_m: float
    radius_m: float
    prescribed_charge_c: float


def _sites_with_axis(problem: ElectrostaticProblem):
    """Return deterministic common axis and axial coordinates of grouped sites.

    Must only be called after supports() validates the arrangement.
    """
    sphere=problem.conductors[0]
    sites=active_external_sites(problem)
    far=max(sites,key=lambda p:(_multi_norm(_multi_difference(p.position_m,sphere.center_m)),
                                p.position_m))
    vector=_multi_difference(far.position_m,sphere.center_m)
    length=_multi_norm(vector)
    axis=tuple(x/length for x in vector)
    axial=sorted(((_dot(_multi_difference(p.position_m,sphere.center_m),axis),p)
                  for p in sites),key=lambda pair:(pair[0],pair[1].position_m))
    return axis,tuple(axial)


def _compile(client: Any, problem: ElectrostaticProblem, settings: Femm5Settings):
    sphere=problem.conductors[0]
    b=sphere.radius_m
    axis,sites=_sites_with_axis(problem)
    cores=[]
    for i,(z,source) in enumerate(sites):
        clearance=abs(z)-b
        neighbor=min((abs(z-other_z) for j,(other_z,_) in enumerate(sites) if i!=j),
                     default=float('inf'))
        radius=min(b*settings.source_radius_fraction,clearance/6,neighbor/4)
        if not isfinite(radius) or radius<=0:
            raise UnsupportedProblem('external numerical cores require finite, nonoverlapping clearance')
        name=_SOURCE_NAME if len(sites)==1 else f'{_SOURCE_NAME}_{i+1}'
        cores.append(_ExternalSourceCore(name,source.position_m,z,radius,source.charge_c))
    R=max(b*settings.outer_radius_factor,
          2.5*max(abs(c.axial_offset_m)+c.radius_m for c in cores))
    if not isfinite(R):
        raise UnsupportedProblem('nonfinite FEMM outer domain')
    delta=settings.arc_degrees
    client.newdocument(1)
    client.ei_probdef('meters','axi',settings.solver_precision)
    rel_eps=problem.permittivity_f_per_m/EPSILON_0
    client.ei_addmaterial('Vacuum',rel_eps,rel_eps,0)
    if isinstance(sphere.constraint, Grounded):
        client.ei_addconductorprop(sphere.name,0,0,1)
    else:
        assert isinstance(sphere.constraint, FixedNetCharge)
        client.ei_addconductorprop(sphere.name,0,sphere.constraint.charge_c,0)
    for core in cores:
        client.ei_addconductorprop(core.name,0,core.prescribed_charge_c,0)
    # Robin boundary approximates open space, including multipoles. Domain
    # enlargement is explicitly compared against exact images in live tests.
    client.ei_addboundprop('open_robin',0,0,problem.permittivity_f_per_m/R,0,1)
    for rad,zcenter in ((b,0.0),*((c.radius_m,c.axial_offset_m) for c in cores),(R,0.0)):
        for rho,z in ((0,zcenter-rad),(rad,zcenter),(0,zcenter+rad)):
            client.ei_addnode(rho,z)
        _half_circle(client,rad,delta,zcenter)
    heights=sorted({-R,-b,b,R,*(
        z for c in cores for z in (c.axial_offset_m-c.radius_m,
                                    c.axial_offset_m+c.radius_m))})
    for low,high in zip(heights,heights[1:]):
        client.ei_addsegment(0,low,0,high)
    _mark_arcs(client,b,delta,'<None>',sphere.name)
    for core in cores:
        _mark_arcs(client,core.radius_m,delta,'<None>',core.name,core.axial_offset_m)
    _mark_arcs(client,R,delta,'open_robin','<None>')
    _label_air(client,b/2,b/8*settings.mesh_scale)
    for core in cores:
        _label_air(client,core.radius_m/2,core.radius_m/6*settings.mesh_scale,
                   z=core.axial_offset_m)
    _label_air(client,(b+R)/2,b/4*settings.mesh_scale)
    return R,tuple(cores),axis


@dataclass(slots=True)
class Femm5ExternalSphereSolution:
    sphere: SolidSphere
    outer_radius_m: float
    conductor_results: tuple[ConductorResult, ...]
    _session: Any = field(repr=False)
    _client: Any = field(repr=False)
    _workspace: TemporaryDirectory[str] = field(repr=False)
    numerical_sources: tuple[NumericalSource, ...] = ()
    source_radius_m: float | None = None
    numerical_source_charge_c: float | None = None
    symmetry_axis: Vector3 = (0.0,0.0,1.0)
    permittivity_f_per_m: float = EPSILON_0
    _closed: bool = field(default=False,init=False,repr=False)

    def __enter__(self):
        if self._closed:
            raise RuntimeError('FEMM solution has been closed')
        return self

    def __exit__(self,*args):
        self.close()

    def close(self):
        if not self._closed:
            self._closed=True
            try:
                self._session.close()
            finally:
                self._workspace.cleanup()

    def __del__(self):
        if not getattr(self,'_closed',True):
            try:self.close()
            except Exception:pass

    def _sample_coordinates(self, point_m):
        if self._closed:
            raise RuntimeError('FEMM solution has been closed')
        point=_position(point_m,'FEMM sample point')
        delta=_difference(point,self.sphere.center_m)
        z=_dot(delta,self.symmetry_axis)
        rho=tuple(delta[i]-z*self.symmetry_axis[i] for i in range(3))
        r=_norm(rho)
        if _norm(delta)>=self.outer_radius_m:
            raise UnsupportedProblem('point outside finite FEMM computational domain')
        if any(hypot(r,z-s.axial_offset_m)<=s.radius_m for s in self.numerical_sources):
            raise UnsupportedProblem('point lies inside/on regularized numerical source core')
        return r,z,rho

    def potential_at(self,point_m):
        r,z,_=self._sample_coordinates(point_m)
        return _as_finite_values(self._client.eo_getpointvalues(r,z),8,'point values')[0]

    def electric_field_at(self,point_m):
        point=_position(point_m,'FEMM sample point')
        delta=_difference(point,self.sphere.center_m)
        if isclose(_norm(delta),self.sphere.radius_m,rel_tol=1e-12):
            raise SingularField('electric field has a jump at sphere surface')
        if any(isclose(_norm(_difference(point,s.position_m)),s.radius_m,rel_tol=1e-12)
               for s in self.numerical_sources):
            raise SingularField('field jumps on regularized source-core surface')
        r,z,rho=self._sample_coordinates(point)
        value=_as_finite_values(self._client.eo_getpointvalues(r,z),8,'point values')
        er,ez=value[3],value[4]
        return tuple(ez*self.symmetry_axis[i]+(er*rho[i]/r if r else 0.0)
                     for i in range(3))

    def surface_charge_density(
        self, surface: Literal['outer'], theta_rad: float,
        *, offset_fraction: float=.02, conductor_name: str | None=None,
    ) -> float:
        if surface!='outer':
            raise ValueError('solid sphere has no inner conducting face')
        if conductor_name not in (None,self.sphere.name):
            raise ValueError('unknown physical conductor')
        if not isfinite(theta_rad) or not 0<=theta_rad<=pi:
            raise ValueError('theta must lie in [0,pi]')
        if not isfinite(offset_fraction) or not 0<offset_fraction<=.1:
            raise ValueError('offset_fraction must lie in (0,0.1]')
        b=self.sphere.radius_m
        clearance=min(abs(c.axial_offset_m)-b-c.radius_m
                      for c in self.numerical_sources)
        gap=min(b*offset_fraction,clearance/5)
        radius=b+gap
        tangent=_perpendicular(self.symmetry_axis)
        normal=tuple(cos(theta_rad)*self.symmetry_axis[i]+sin(theta_rad)*tangent[i] for i in range(3))
        point=tuple(self.sphere.center_m[i]+radius*normal[i] for i in range(3))
        field=self.electric_field_at(point)
        return self.permittivity_f_per_m*_dot(field,normal)


class Femm5ExternalSphereBackend:
    """Numerical backend for axial sources outside one grounded or isolated sphere."""

    def __init__(self, *, settings: Femm5Settings | None=None,
                 binding_path: str | Path | None=None,
                 session_factory: Callable[[],Any] | None=None):
        self.settings=settings or Femm5Settings()
        self._session_factory=session_factory or (lambda:Femm5Session(binding_path))

    def supports(self,problem:ElectrostaticProblem) -> bool:
        if not ExternalSphereBackend().supports(problem):
            return False
        sphere=problem.conductors[0]
        if sphere.name.startswith(_SOURCE_NAME):
            return False
        sites=active_external_sites(problem)
        if not 1 <= len(sites) <= 16:
            return False
        axis,_=_sites_with_axis(problem)
        for site in sites:
            delta=_difference(site.position_m,sphere.center_m)
            z=_dot(delta,axis)
            transverse=tuple(delta[i]-z*axis[i] for i in range(3))
            if _norm(transverse)>1e-11*sphere.radius_m:
                return False
        return True

    def solve(self,problem:ElectrostaticProblem) -> Femm5ExternalSphereSolution:
        if not self.supports(problem):
            raise UnsupportedProblem(
                'FEMM 5 exterior sphere requires one Grounded or FixedNetCharge SolidSphere '
                'and up to 16 grouped external charge sites on a common central axis; '
                'non-collinear arrangements require an analytical or 3D numerical backend')
        session=self._session_factory()
        workspace=TemporaryDirectory(prefix='conductor_sim_femm_external_')
        try:
            client=session.open()
            R,cores,axis=_compile(client,problem,self.settings)
            client.ei_saveas((Path(workspace.name)/'external_sphere.fee').as_posix())
            client.ei_analyze(1)
            client.ei_loadsolution()
            sphere=problem.conductors[0]
            v,net=_as_finite_values(client.eo_getconductorproperties(sphere.name),2,
                                    'solid sphere conductor properties')
            measured=[]
            for core in cores:
                _,charge=_as_finite_values(client.eo_getconductorproperties(core.name),2,
                                            f'external source {core.name} charge')
                measured.append(NumericalSource(core.name,core.position_m,
                                                core.axial_offset_m,core.radius_m,charge))
            total=fsum(s.measured_charge_c for s in measured)
            results=(ConductorResult(sphere.name,v,net,0.0,net),)
            source_radius=measured[0].radius_m if len(measured)==1 else None
            return Femm5ExternalSphereSolution(
                sphere,R,results,session,client,workspace,tuple(measured),
                source_radius,total,axis,problem.permittivity_f_per_m)
        except Exception as exc:
            try:
                session.close()
            finally:
                workspace.cleanup()
            if isinstance(exc,(Femm5Unavailable,Femm5ExecutionError,UnsupportedProblem)):
                raise
            raise Femm5ExecutionError(f'FEMM 5 exterior-sphere solve failed: {exc}') from exc
