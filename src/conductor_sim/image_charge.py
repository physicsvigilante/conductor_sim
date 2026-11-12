"""Exact image-charge reference for a charge inside a spherical conductor.

The reference is independent of FEMM: it uses the Kelvin image construction
in the cavity and Gauss's law outside a connected spherical conductor.
This geometry is valid for homogeneous, linear permittivity throughout.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite, pi, sqrt
from typing import Literal

from .model import ElectrostaticProblem, FixedNetCharge, PointCharge, SphericalShell, Vector3
from .solver import ConductorResult, SingularField, UnsupportedProblem


def _subtract(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _distance(v: Vector3) -> float:
    return sqrt(sum(x * x for x in v))


def _scaled(v: Vector3, scale: float) -> Vector3:
    return tuple(x * scale for x in v)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ImageChargeShellSolution:
    """Exact potential, field and induced sigma for one point source in a sphere.

    Theta is measured from the center-to-source direction; for centered
    charges it is arbitrary. The inner surface outward *metal* normal points
    into the cavity, so positive enclosed point charge induces negative sigma.
    """

    shell: SphericalShell
    source: PointCharge
    permittivity_f_per_m: float
    conductor_results: tuple[ConductorResult, ...]

    @property
    def _k(self) -> float:
        return 1.0 / (4 * pi * self.permittivity_f_per_m)

    @property
    def displacement_m(self) -> float:
        return _distance(_subtract(self.source.position_m, self.shell.center_m))

    def _image(self) -> tuple[float, Vector3] | None:
        d = self.displacement_m
        if d == 0:
            return None
        a = self.shell.inner_radius_m
        delta = _subtract(self.source.position_m, self.shell.center_m)
        return (-self.source.charge_c * a / d,
                tuple(self.shell.center_m[i] + (a*a / (d*d)) * delta[i] for i in range(3)))

    def potential_at(self, point_m: Vector3) -> float:
        v_shell = self.conductor_results[0].potential_v
        from_center = _subtract(point_m, self.shell.center_m)
        r = _distance(from_center)
        a, b = self.shell.inner_radius_m, self.shell.outer_radius_m
        if r < a:
            from_source = _subtract(point_m, self.source.position_m)
            distance = _distance(from_source)
            q = self.source.charge_c
            if distance == 0 and q != 0:
                raise SingularField("potential is singular at the point source")
            if q == 0:
                return v_shell
            if self.displacement_m == 0:
                return v_shell + self._k * q * (1/distance - 1/a)
            image_charge, image_position = self._image()  # type: ignore[misc]
            return v_shell + self._k * (
                q / distance + image_charge / _distance(_subtract(point_m, image_position))
            )
        if r <= b:
            return v_shell
        return self._k * self.conductor_results[0].outer_surface_charge_c / r

    def electric_field_at(self, point_m: Vector3) -> Vector3:
        from_center = _subtract(point_m, self.shell.center_m)
        r = _distance(from_center)
        a, b = self.shell.inner_radius_m, self.shell.outer_radius_m
        if isclose(r, a, rel_tol=1e-12) or isclose(r, b, rel_tol=1e-12):
            raise SingularField("field has distinct one-sided conductor surface limits")
        if a < r < b:
            return (0.0, 0.0, 0.0)
        if r > b:
            factor = self._k * self.conductor_results[0].outer_surface_charge_c / (r**3)
            return _scaled(from_center, factor)
        q = self.source.charge_c
        from_source = _subtract(point_m, self.source.position_m)
        distance = _distance(from_source)
        if distance == 0 and q != 0:
            raise SingularField("field is singular at point source")
        if q == 0:
            return (0.0, 0.0, 0.0)
        charge_field = _scaled(from_source, self._k * q / distance**3)
        image = self._image()
        if image is None:
            return charge_field
        iq, ip = image
        delta = _subtract(point_m, ip)
        scale = self._k * iq / _distance(delta)**3
        return tuple(charge_field[i] + delta[i] * scale for i in range(3))  # type: ignore[return-value]

    def surface_charge_density(self, surface: Literal["inner", "outer"], theta_rad: float) -> float:
        if surface not in ("inner", "outer"):
            raise ValueError("surface must be inner or outer")
        if not isfinite(theta_rad) or not 0 <= theta_rad <= pi:
            raise ValueError("theta must lie in [0, pi]")
        if surface == "outer":
            return self.conductor_results[0].outer_surface_charge_c / (4*pi*self.shell.outer_radius_m**2)
        a = self.shell.inner_radius_m
        d = self.displacement_m
        if d == 0:
            return -self.source.charge_c / (4*pi*a*a)
        from math import cos
        return -self.source.charge_c * (a*a - d*d) / (
            4*pi*a*(a*a + d*d - 2*a*d*cos(theta_rad))**1.5
        )


class ImageChargeShellBackend:
    """Analytical benchmark for one source in one spherical shell.

    This is *not* a general multi-charge solver. Coincident authored charges
    may be combined in a later milestone; mixed locations are rejected here.
    """

    def supports(self, problem: ElectrostaticProblem) -> bool:
        if len(problem.conductors) != 1 or len(problem.charges) != 1:
            return False
        shell = problem.conductors[0]
        return (
            isinstance(shell, SphericalShell)
            and _distance(_subtract(problem.charges[0].position_m, shell.center_m)) < shell.inner_radius_m
        )

    def solve(self, problem: ElectrostaticProblem) -> ImageChargeShellSolution:
        if not self.supports(problem):
            raise UnsupportedProblem("image-charge reference requires one charge inside one spherical shell")
        shell = problem.conductors[0]
        q = problem.charges[0].charge_c
        q_shell = shell.constraint.charge_c if isinstance(shell.constraint, FixedNetCharge) else -q
        q_outer = q + q_shell
        potential = q_outer/(4*pi*problem.permittivity_f_per_m*shell.outer_radius_m)
        c = ConductorResult(shell.name, potential, q_shell, -q, q_outer)
        return ImageChargeShellSolution(shell, problem.charges[0], problem.permittivity_f_per_m, (c,))
