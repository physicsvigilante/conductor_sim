"""Real FEMM 5 exterior-source solver for grounded or fixed-charge solid spheres.

This backend never delegates numeric queries to the exact image solver. It
constructs one real solid sphere and one finite charged source core outside
it, then returns values measured by FEMM's own electrostatic postprocessor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, hypot, isclose, isfinite, pi, sin, sqrt
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

_SOURCE_NAME = '__conductor_sim_external_source'


def _difference(a: Vector3, b: Vector3) -> Vector3:
    return tuple(a[i]-b[i] for i in range(3))


def _compile(client: Any, problem: ElectrostaticProblem, settings: Femm5Settings):
    sphere, source = problem.conductors[0], problem.charges[0]
    b=sphere.radius_m
    displacement=_difference(source.position_m,sphere.center_m)
    d=_norm(displacement)
    axis=tuple(x/d for x in displacement)
    clearance=d-b
    # Source remains a distinct FINITE charged conductor, not FEMM pointprop
    # (which could otherwise turn into an azimuthal ring).
    source_radius=min(b*settings.source_radius_fraction, clearance/6)
    if not isfinite(source_radius) or source_radius<=0:
        raise UnsupportedProblem('external charge has insufficient geometric clearance')
    R=max(b*settings.outer_radius_factor, (d+source_radius)*2.5)
    if not isfinite(R):
        raise UnsupportedProblem('nonfinite FEMM domain')
    delta=settings.arc_degrees
    client.newdocument(1)
    client.ei_probdef('meters','axi',settings.solver_precision)
    rel_eps=problem.permittivity_f_per_m/EPSILON_0
    client.ei_addmaterial('Vacuum',rel_eps,rel_eps,0)
    if isinstance(sphere.constraint, Grounded):
        client.ei_addconductorprop(sphere.name,0,0,1)  # fixed 0 V, net Q unknown
    else:
        assert isinstance(sphere.constraint, FixedNetCharge)
        client.ei_addconductorprop(sphere.name,0,sphere.constraint.charge_c,0)  # fixed total Q, V unknown
    client.ei_addconductorprop(_SOURCE_NAME,0,source.charge_c,0)  # fixed source net Q
    # Monopolar open-boundary approximation; vary R to assess multipolar error.
    client.ei_addboundprop('open_robin',0,0,problem.permittivity_f_per_m/R,0,1)
    for rad, zcenter in ((b,0.0),(source_radius,d),(R,0.0)):
        for rho,z in ((0,zcenter-rad),(rad,zcenter),(0,zcenter+rad)):
            client.ei_addnode(rho,z)
        _half_circle(client,rad,delta,zcenter)
    heights=sorted((-R,-b,b,d-source_radius,d+source_radius,R))
    for low,high in zip(heights,heights[1:]):
        client.ei_addsegment(0,low,0,high)
    _mark_arcs(client,b,delta,'<None>',sphere.name)
    _mark_arcs(client,source_radius,delta,'<None>',_SOURCE_NAME,d)
    _mark_arcs(client,R,delta,'open_robin','<None>')
    _label_air(client,b/2,b/8*settings.mesh_scale)
    _label_air(client,source_radius/2,source_radius/6*settings.mesh_scale,z=d)
    # External vacuum label along the equator, safely away from axial source.
    _label_air(client,(b+R)/2,b/4*settings.mesh_scale)
    return R, source_radius, axis, d


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
        distance=self.numerical_sources[0].axial_offset_m
        clearance=distance-b-self.numerical_sources[0].radius_m
        gap=min(b*offset_fraction,clearance/5)
        radius=b+gap
        tangent=_perpendicular(self.symmetry_axis)
        normal=tuple(cos(theta_rad)*self.symmetry_axis[i]+sin(theta_rad)*tangent[i] for i in range(3))
        point=tuple(self.sphere.center_m[i]+radius*normal[i] for i in range(3))
        field=self.electric_field_at(point)
        return self.permittivity_f_per_m*_dot(field,normal)


class Femm5ExternalSphereBackend:
    """Numerical backend for one charge outside a grounded OR isolated solid sphere."""

    def __init__(self, *, settings: Femm5Settings | None=None,
                 binding_path: str | Path | None=None,
                 session_factory: Callable[[],Any] | None=None):
        self.settings=settings or Femm5Settings()
        self._session_factory=session_factory or (lambda:Femm5Session(binding_path))

    def supports(self,problem:ElectrostaticProblem) -> bool:
        return (ExternalSphereBackend().supports(problem) and
                problem.conductors[0].name != _SOURCE_NAME)

    def solve(self,problem:ElectrostaticProblem) -> Femm5ExternalSphereSolution:
        if not self.supports(problem):
            raise UnsupportedProblem(
                'FEMM 5 exterior sphere requires exactly one external nonzero point '
                'charge and one Grounded or FixedNetCharge SolidSphere; other geometries unsupported')
        session=self._session_factory()
        workspace=TemporaryDirectory(prefix='conductor_sim_femm_external_')
        try:
            client=session.open()
            R,source_radius,axis,d=_compile(client,problem,self.settings)
            client.ei_saveas((Path(workspace.name)/'external_sphere.fee').as_posix())
            client.ei_analyze(1)
            client.ei_loadsolution()
            sphere,charge=problem.conductors[0],problem.charges[0]
            v,net=_as_finite_values(client.eo_getconductorproperties(sphere.name),2,
                                    'solid sphere conductor properties')
            _,measured=_as_finite_values(client.eo_getconductorproperties(_SOURCE_NAME),2,
                                          'external source conductor properties')
            results=(ConductorResult(sphere.name,v,net,0.0,net),)
            source=(NumericalSource(_SOURCE_NAME,charge.position_m,d,source_radius,measured),)
            return Femm5ExternalSphereSolution(
                sphere,R,results,session,client,workspace,source,
                source_radius,measured,axis,problem.permittivity_f_per_m)
        except Exception as exc:
            try:
                session.close()
            finally:
                workspace.cleanup()
            if isinstance(exc,(Femm5Unavailable,Femm5ExecutionError,UnsupportedProblem)):
                raise
            raise Femm5ExecutionError(f'FEMM 5 exterior-sphere solve failed: {exc}') from exc
