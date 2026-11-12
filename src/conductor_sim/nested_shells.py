"""Exact radial-conductor coupling for nonintersecting concentric spherical shells.

Point charges can occupy arbitrary positions in the innermost cavity. Their
nonuniform induced charge on the innermost inner face is obtained by Kelvin
images; every outer region is spherically symmetric because conducting shells
shield higher multipoles. Independently solves grounded/fixed-Q constraints.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isclose, pi, sqrt, isfinite, fsum
from typing import Literal

from .model import ElectrostaticProblem, FixedNetCharge, Grounded, SphericalShell, Vector3
from .solver import ConductorResult, SingularField, UnsupportedProblem
from .multi_image import MultiImageShellBackend, MultiImageShellSolution


def _dist(a: Vector3, b: Vector3) -> float:
    return sqrt(sum((x-y)**2 for x,y in zip(a,b)))


def _solve_linear(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Small dense Gaussian elimination with row pivoting; no NumPy dependency."""
    n = len(rhs)
    if not n:
        return []
    work = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    magnitude = max(abs(x) for row in matrix for x in row)
    if magnitude == 0:
        raise UnsupportedProblem("singular grounding constraints")
    for col in range(n):
        pivot = max(range(col,n), key=lambda i: abs(work[i][col]))
        if abs(work[pivot][col]) < 1e-13*magnitude:
            raise UnsupportedProblem("ill-conditioned nested-shell ground constraints")
        work[col],work[pivot] = work[pivot],work[col]
        for i in range(col+1,n):
            factor=work[i][col]/work[col][col]
            for k in range(col+1,n+1):
                work[i][k] -= factor*work[col][k]
            work[i][col] = 0.0
    out=[0.]*n
    for i in reversed(range(n)):
        out[i]=(work[i][-1]-sum(work[i][j]*out[j] for j in range(i+1,n)))/work[i][i]
    return out


def _sorted_shells(problem: ElectrostaticProblem) -> tuple[SphericalShell,...]:
    return tuple(sorted(problem.conductors,key=lambda s:(s.inner_radius_m,s.outer_radius_m)))


@dataclass(frozen=True, slots=True)
class NestedShellSolution:
    """Exact potential, field and charges for concentric nested shells."""
    radial_shells: tuple[SphericalShell,...]
    conductor_results: tuple[ConductorResult,...]  # original authored order
    charge_inside_c: float
    permittivity_f_per_m: float
    inner_reference: MultiImageShellSolution

    @property
    def _k(self) -> float:
        return 1/(4*pi*self.permittivity_f_per_m)

    def conductor_by_name(self, name: str) -> ConductorResult:
        for result in self.conductor_results:
            if result.name==name:
                return result
        raise KeyError(name)

    def _fluxes(self) -> tuple[float,...]:
        return tuple(self.conductor_by_name(s.name).outer_surface_charge_c for s in self.radial_shells)

    def potential_at(self, point_m: Vector3) -> float:
        delta=tuple(point_m[i]-self.radial_shells[0].center_m[i] for i in range(3))
        r=sqrt(sum(x*x for x in delta))
        shells=self.radial_shells
        if r < shells[0].inner_radius_m:
            return (self.conductor_by_name(shells[0].name).potential_v+
                    self.inner_reference.potential_at(point_m))
        for i,s in enumerate(shells):
            if r <= s.outer_radius_m:
                return self.conductor_by_name(s.name).potential_v
            if i+1<len(shells) and r < shells[i+1].inner_radius_m:
                next_shell=shells[i+1]
                return (self.conductor_by_name(next_shell.name).potential_v+
                        self._k*self._fluxes()[i]*(1/r-1/next_shell.inner_radius_m))
        return self._k*self._fluxes()[-1]/r

    def electric_field_at(self, point_m: Vector3) -> Vector3:
        delta=tuple(point_m[i]-self.radial_shells[0].center_m[i] for i in range(3))
        r=sqrt(sum(x*x for x in delta))
        shells=self.radial_shells
        if any(isclose(r,s.inner_radius_m,rel_tol=1e-12) or
               isclose(r,s.outer_radius_m,rel_tol=1e-12) for s in shells):
            raise SingularField("field jumps at a conductor surface")
        if r < shells[0].inner_radius_m:
            return self.inner_reference.electric_field_at(point_m)
        for i,s in enumerate(shells):
            if r < s.outer_radius_m:
                return (0.,0.,0.)
            if i+1<len(shells) and r < shells[i+1].inner_radius_m:
                scale=self._k*self._fluxes()[i]/r**3
                return tuple(scale*x for x in delta)
        scale=self._k*self._fluxes()[-1]/r**3
        return tuple(scale*x for x in delta)

    def surface_charge_density_at(self, conductor_name: str,
                                   surface: Literal['inner','outer'],point_m:Vector3)->float:
        s=next((item for item in self.radial_shells if item.name==conductor_name),None)
        if s is None:
            raise KeyError(conductor_name)
        if surface not in ('inner','outer'):
            raise ValueError("surface must be inner or outer")
        radius=s.inner_radius_m if surface=='inner' else s.outer_radius_m
        if len(point_m)!=3 or not all(isfinite(x) for x in point_m) or not isclose(
                _dist(point_m,s.center_m),radius,rel_tol=1e-10,abs_tol=1e-13):
            raise ValueError("point must lie on the requested conductor surface")
        if surface=='inner' and s is self.radial_shells[0]:
            return self.inner_reference.surface_charge_density_at('inner',point_m)
        c=self.conductor_by_name(conductor_name)
        q=c.inner_surface_charge_c if surface=='inner' else c.outer_surface_charge_c
        return q/(4*pi*radius*radius)


class NestedShellBackend:
    """Independent exact benchmark for one or more strictly nested shells."""

    def supports(self,problem:ElectrostaticProblem)->bool:
        shells=_sorted_shells(problem)
        if not shells or any(not isinstance(s,SphericalShell) for s in shells):
            return False
        center=shells[0].center_m
        if any(s.center_m != center for s in shells):
            return False
        if any(a.outer_radius_m >= b.inner_radius_m for a,b in zip(shells,shells[1:])):
            return False
        return all(_dist(source.position_m,center)<shells[0].inner_radius_m
                   for source in problem.charges)

    def solve(self,problem:ElectrostaticProblem)->NestedShellSolution:
        if not self.supports(problem):
            raise UnsupportedProblem("nested-shell reference requires concentric, nonintersecting shells and charges in the innermost cavity")
        shells=_sorted_shells(problem)
        n=len(shells)
        q0=fsum(x.charge_c for x in problem.charges)
        net=[s.constraint.charge_c if isinstance(s.constraint,FixedNetCharge) else 0. for s in shells]
        grounded=[i for i,s in enumerate(shells) if isinstance(s.constraint,Grounded)]
        gaps=[1/shells[i].outer_radius_m - 1/shells[i+1].inner_radius_m
              for i in range(n-1)]
        def coefficient(i:int,j:int)->float:
            # Derivative (V_i / Coulomb constant) wrt shell net charge Q_j.
            return 1/shells[-1].outer_radius_m + sum(gaps[k] for k in range(i,n-1) if j<=k)
        def offset(i:int)->float:
            return q0*(1/shells[-1].outer_radius_m + sum(gaps[k] for k in range(i,n-1)))
        if grounded:
            matrix=[[coefficient(i,j) for j in grounded] for i in grounded]
            rhs=[-offset(i)-sum(coefficient(i,j)*net[j] for j in range(n) if j not in grounded)
                 for i in grounded]
            values=_solve_linear(matrix,rhs)
            for j,value in zip(grounded,values):
                net[j]=value
        flux=[]
        enclosed=q0
        results={}
        for i,s in enumerate(shells):
            inner=-enclosed
            enclosed+=net[i]
            flux.append(enclosed)
            outer=enclosed
            results[s.name]=(inner,outer)
        voltages=[]
        for i,s in enumerate(shells):
            reduced=flux[-1]/shells[-1].outer_radius_m + sum(
                flux[j]*gaps[j] for j in range(i,n-1))
            voltages.append(reduced/(4*pi*problem.permittivity_f_per_m))
        # Analytical cavity reference is grounded at the first conductor;
        # add the independently solved inner-conductor voltage afterwards.
        reference=MultiImageShellBackend().solve(ElectrostaticProblem(
            charges=problem.charges,
            conductors=(SphericalShell(shells[0].inner_radius_m,shells[0].outer_radius_m,
                                        Grounded(),center_m=shells[0].center_m),),
            permittivity_f_per_m=problem.permittivity_f_per_m))
        by_name={s.name:ConductorResult(s.name,voltages[i],net[i],*results[s.name])
                 for i,s in enumerate(shells)}
        return NestedShellSolution(shells,tuple(by_name[s.name] for s in problem.conductors),
                                   q0,problem.permittivity_f_per_m,reference)
