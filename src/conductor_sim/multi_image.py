"""Independent exact superposition of Kelvin images inside one spherical shell.

Unlike FEMM's axisymmetric backend, this mathematical reference supports
arbitrarily positioned point charges within one spherical cavity. It is not
numerical FEMM output and it does not describe several physical shells.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isclose, isfinite, pi, sqrt
from typing import Literal

from .model import ElectrostaticProblem, FixedNetCharge, PointCharge, SphericalShell, Vector3
from .kelvin import image_term
from .model import _position
from .solver import ConductorResult, SingularField, UnsupportedProblem


def _delta(a: Vector3, b: Vector3) -> Vector3:
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def _norm(a: Vector3) -> float:
    return sqrt(sum(x*x for x in a))


@dataclass(frozen=True, slots=True)
class MultiImageShellSolution:
    """Exact cavity fields and surface sigma for any finite set of internal charges."""

    shell: SphericalShell
    charges: tuple[PointCharge, ...]
    permittivity_f_per_m: float
    conductor_results: tuple[ConductorResult, ...]

    @property
    def _k(self) -> float:
        return 1/(4*pi*self.permittivity_f_per_m)

    def potential_at(self, point_m: Vector3) -> float:
        point_m = _position(point_m, "sample point")
        center_vec = _delta(point_m, self.shell.center_m)
        r = _norm(center_vec)
        a, b = self.shell.inner_radius_m, self.shell.outer_radius_m
        if r < a:
            v = self.conductor_results[0].potential_v
            for source in self.charges:
                dist = _norm(_delta(point_m, source.position_m))
                if dist == 0:
                    raise SingularField("potential is singular at nonzero point source")
                image_v, _ = image_term(point_m, source.position_m, self.shell.center_m, a)
                v += self._k*source.charge_c*(1/dist + image_v)
            return v
        if r <= b:
            return self.conductor_results[0].potential_v
        return self._k*self.conductor_results[0].outer_surface_charge_c/r

    def electric_field_at(self, point_m: Vector3) -> Vector3:
        point_m = _position(point_m, "sample point")
        center_vec = _delta(point_m, self.shell.center_m)
        r = _norm(center_vec)
        a, b = self.shell.inner_radius_m, self.shell.outer_radius_m
        if isclose(r, a, rel_tol=1e-12) or isclose(r, b, rel_tol=1e-12):
            raise SingularField("field has distinct one-sided conductor surface limits")
        if a < r < b:
            return (0., 0., 0.)
        if r > b:
            scale = self._k * self.conductor_results[0].outer_surface_charge_c / r**3
            return tuple(scale*v for v in center_vec)
        field = [0., 0., 0.]
        for source in self.charges:
            dv = _delta(point_m, source.position_m)
            d = _norm(dv)
            if d == 0:
                raise SingularField("field is singular at nonzero point source")
            _, image_e = image_term(point_m, source.position_m, self.shell.center_m, a)
            scale = self._k*source.charge_c
            for i in range(3):
                field[i] += scale*(dv[i]/d**3 + image_e[i])
        return tuple(field)

    def surface_charge_density_at(self, surface: Literal["inner", "outer"], point_m: Vector3) -> float:
        """Exact local sigma at a point on the requested spherical surface."""
        if surface not in ("inner", "outer"):
            raise ValueError("surface must be 'inner' or 'outer'")
        expected_radius = self.shell.inner_radius_m if surface == "inner" else self.shell.outer_radius_m
        if len(point_m) != 3 or any(not isfinite(x) for x in point_m):
            raise ValueError("surface point must have three finite coordinates")
        radius = _norm(_delta(point_m, self.shell.center_m))
        if not isclose(radius, expected_radius, rel_tol=1e-10, abs_tol=1e-13):
            raise ValueError("point is not on the specified spherical surface")
        if surface == "outer":
            return self.conductor_results[0].outer_surface_charge_c/(4*pi*expected_radius**2)
        a = self.shell.inner_radius_m
        c = self.shell.center_m
        sigma = 0.
        for source in self.charges:
            displacement = _delta(source.position_m, c)
            d = _norm(displacement)
            separation = _norm(_delta(point_m, source.position_m))
            sigma -= source.charge_c*(a*a-d*d)/(4*pi*a*separation**3)
        return sigma


class MultiImageShellBackend:
    """Exact 3D multi-source reference within a single spherical shell."""

    def supports(self, problem: ElectrostaticProblem) -> bool:
        if len(problem.conductors) != 1:
            return False
        shell = problem.conductors[0]
        return isinstance(shell, SphericalShell) and all(
            _norm(_delta(q.position_m, shell.center_m)) < shell.inner_radius_m
            for q in problem.charges
        )

    def solve(self, problem: ElectrostaticProblem) -> MultiImageShellSolution:
        if not self.supports(problem):
            raise UnsupportedProblem("image-charge superposition requires sources inside one spherical shell")
        shell = problem.conductors[0]
        # Coincident authored charges are physically one source. Cancellation
        # to zero must not produce a fictitious singularity in the reference.
        by_position: dict[Vector3, list[float]] = {}
        for charge in problem.charges:
            by_position.setdefault(charge.position_m, []).append(charge.charge_c)
        sources = tuple(PointCharge(q, pos) for pos, charges in sorted(by_position.items())
                        if (q := fsum(charges)) != 0.0)
        q = fsum(source.charge_c for source in sources)
        q_shell = shell.constraint.charge_c if isinstance(shell.constraint, FixedNetCharge) else -q
        q_outer = q+q_shell
        potential = q_outer/(4*pi*problem.permittivity_f_per_m*shell.outer_radius_m)
        result = ConductorResult(shell.name, potential, q_shell, -q, q_outer)
        return MultiImageShellSolution(shell, sources, problem.permittivity_f_per_m, (result,))
