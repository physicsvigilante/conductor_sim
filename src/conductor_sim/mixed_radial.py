"""Exact, source-free, radially symmetric solid-sphere and shell equilibria.

One solid conducting sphere is surrounded by one or more separated concentric
SphericalShell conductors. No point charges, eccentricity, dielectrics or
wire geometry are silently approximated by this solver. M14b adds an explicit
analytical domain rather than misusing a phantom cavity in the solid.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import fsum, hypot, isclose, pi
from typing import Literal

from .model import (ElectrostaticProblem, FixedNetCharge, Grounded,
                    SolidSphere, SphericalShell, Vector3, _position)
from .nested_shells import _solve_linear
from .solver import ConductorResult, SingularField, UnsupportedProblem


RadialConductor = SolidSphere | SphericalShell


def _layout(problem: ElectrostaticProblem) -> tuple[RadialConductor, ...]:
    """Check the *entire* authored geometry before doing any linear algebra."""
    shapes = tuple(sorted(problem.conductors,
                          key=lambda c: (getattr(c, 'inner_radius_m', -1.),
                                         getattr(c, 'outer_radius_m', -1.))))
    if (not 2 <= len(shapes) <= 8
            or not isinstance(shapes[0], SolidSphere)
            or any(not isinstance(c, SphericalShell) for c in shapes[1:])
            or problem.charges):
        raise UnsupportedProblem(
            'mixed radial solver requires one innermost solid sphere, '
            '1–7 surrounding concentric shells, and no point sources')
    center = shapes[0].center_m
    if any(c.center_m != center for c in shapes):
        raise UnsupportedProblem('mixed radial conductors must share one center')
    if any(a.outer_radius_m >= b.inner_radius_m for a, b in zip(shapes, shapes[1:])):
        raise UnsupportedProblem('mixed radial conductors must be strictly separated')
    if any(not isinstance(c.constraint, (FixedNetCharge, Grounded)) for c in shapes):
        raise UnsupportedProblem('only isolated net charges or grounding are supported')
    return shapes


@dataclass(frozen=True, slots=True)
class MixedRadialSolution:
    """Spherical potential, E, conductor Q and face sigma, without fake faces."""

    radial_shells: tuple[RadialConductor, ...]
    conductor_results: tuple[ConductorResult, ...]  # original authored order
    permittivity_f_per_m: float
    charge_inside_c: float = 0.

    @property
    def _k(self) -> float:
        return 1/(4*pi*self.permittivity_f_per_m)

    def conductor_by_name(self, name: str) -> ConductorResult:
        for result in self.conductor_results:
            if result.name == name:
                return result
        raise KeyError(name)

    def _fluxes(self) -> tuple[float, ...]:
        return tuple(self.conductor_by_name(c.name).outer_surface_charge_c
                     for c in self.radial_shells)

    def potential_at(self, point_m: Vector3) -> float:
        point = _position(point_m, 'potential sample')
        center = self.radial_shells[0].center_m
        r = hypot(*(point[i]-center[i] for i in range(3)))
        shapes, flux = self.radial_shells, self._fluxes()
        for i, conductor in enumerate(shapes):
            if r <= conductor.outer_radius_m:
                return self.conductor_by_name(conductor.name).potential_v
            if i+1 < len(shapes) and r < shapes[i+1].inner_radius_m:
                next_shell = shapes[i+1]
                return (self.conductor_by_name(next_shell.name).potential_v
                        + self._k*flux[i]*(1/r - 1/next_shell.inner_radius_m))
        return self._k*flux[-1]/r

    def electric_field_at(self, point_m: Vector3) -> Vector3:
        point = _position(point_m, 'electric-field sample')
        center = self.radial_shells[0].center_m
        delta = tuple(point[i]-center[i] for i in range(3))
        r = hypot(*delta)
        shapes, flux = self.radial_shells, self._fluxes()
        for conductor in shapes:
            boundaries = ([conductor.outer_radius_m]
                          if isinstance(conductor, SolidSphere)
                          else [conductor.inner_radius_m, conductor.outer_radius_m])
            if any(isclose(r, radius, rel_tol=1e-12, abs_tol=1e-14)
                   for radius in boundaries):
                raise SingularField('electric field has one-sided limits at a metal surface')
        if r < shapes[0].outer_radius_m:
            return (0.,0.,0.)
        for i, conductor in enumerate(shapes):
            if r < conductor.outer_radius_m:
                return (0.,0.,0.)
            if i+1 < len(shapes) and r < shapes[i+1].inner_radius_m:
                return tuple(self._k*flux[i]*x/r**3 for x in delta)
        return tuple(self._k*flux[-1]*x/r**3 for x in delta)

    def surface_charge_density_at(self, conductor_name: str,
                                  surface: Literal['inner','outer'],
                                  point_m: Vector3) -> float:
        conductor = next((c for c in self.radial_shells if c.name == conductor_name), None)
        if conductor is None:
            raise KeyError(conductor_name)
        if surface not in ('inner','outer'):
            raise ValueError('surface must be inner or outer')
        if surface == 'inner' and isinstance(conductor, SolidSphere):
            raise ValueError('a solid conducting sphere has no inner surface')
        point = _position(point_m, 'surface point')
        radius = (conductor.inner_radius_m if surface == 'inner'
                  else conductor.outer_radius_m)
        r = hypot(*(point[i]-conductor.center_m[i] for i in range(3)))
        if not isclose(r, radius, rel_tol=1e-10, abs_tol=1e-13):
            raise ValueError('point must lie on the named conductor surface')
        result = self.conductor_by_name(conductor_name)
        q = (result.inner_surface_charge_c if surface == 'inner'
             else result.outer_surface_charge_c)
        return q/(4*pi*radius**2)


class MixedRadialBackend:
    """One solid sphere inside 1–7 concentric shells; static analytical only."""

    def supports(self, problem: ElectrostaticProblem) -> bool:
        try:
            _layout(problem)
        except (TypeError, AttributeError, ValueError, UnsupportedProblem):
            return False
        return True

    def solve(self, problem: ElectrostaticProblem) -> MixedRadialSolution:
        shapes = _layout(problem)
        # Reuse the same exact radial capacitance-response coefficients as
        # electrical connection plans, and the independently tested solver.
        from .connections import _coefficients
        A, offsets = _coefficients(shapes, 0.)
        n = len(shapes)
        net = [c.constraint.charge_c if isinstance(c.constraint, FixedNetCharge)
               else 0. for c in shapes]
        unknown = [i for i,c in enumerate(shapes) if isinstance(c.constraint, Grounded)]
        if unknown:
            matrix = [[A[i][j] for j in unknown] for i in unknown]
            rhs = [-offsets[i]-fsum(A[i][j]*net[j] for j in range(n) if j not in unknown)
                   for i in unknown]
            for j,value in zip(unknown, _solve_linear(matrix, rhs)):
                net[j] = value
        flux = [fsum(net[:i+1]) for i in range(n)]
        inv_outer = 1/shapes[-1].outer_radius_m
        gaps = [1/shapes[i].outer_radius_m - 1/shapes[i+1].inner_radius_m
                for i in range(n-1)]
        results = {}
        for i,c in enumerate(shapes):
            voltage = (flux[-1]*inv_outer+fsum(flux[j]*gaps[j] for j in range(i,n-1))) / (
                4*pi*problem.permittivity_f_per_m)
            inner_charge = -flux[i-1] if i else 0.
            results[c.name] = ConductorResult(c.name, voltage, net[i],
                                              inner_charge, flux[i])
        return MixedRadialSolution(shapes,
                                   tuple(results[c.name] for c in problem.conductors),
                                   problem.permittivity_f_per_m)
