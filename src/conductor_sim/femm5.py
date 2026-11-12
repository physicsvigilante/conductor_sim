"""FEMM 5 electrostatics adapter: one sphere and one localized axisymmetric source.

This is a deliberately limited, axisymmetric numerical vertical slice.  It produces a real
FEMM 5 axisymmetric solve, *not* a disguised analytical solution.  FEMM is an
optional external application; no FEMM import is performed at package import.

Ownership: Femm5Solution keeps its FEMM session alive for field queries. Use
``with backend.solve(problem) as solution: ...`` or explicitly call ``close()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import util
from math import cos, hypot, isclose, isfinite, pi, sin, sqrt
from os import environ
from pathlib import Path
from socket import AF_INET, SOCK_STREAM, socket
from tempfile import TemporaryDirectory
from typing import Any, Callable, Literal

from .model import EPSILON_0, ElectrostaticProblem, FixedNetCharge, Grounded, SphericalShell, Vector3
from .solver import ConductorResult, SingularField, UnsupportedProblem


_SOURCE_CORE = "__conductor_sim_source_core"  # purely numerical, NOT a user conductor


def _dot(a: Vector3, b: Vector3) -> float:
    return sum(x*y for x, y in zip(a,b))


def _norm(v: Vector3) -> float:
    return sqrt(_dot(v,v))


def _difference(a: Vector3, b: Vector3) -> Vector3:
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def _source_frame(problem: ElectrostaticProblem) -> tuple[Vector3, float]:
    """Rotate local +z through the center-to-charge displacement.

    Every source must occupy the same position; if they do, their total
    charge can be represented by a single small off-center sphere. This is a
    symmetry detection/compiler decision, not a restriction on world axes.
    """
    shell = problem.conductors[0]
    if not problem.charges:
        return (0.0, 0.0, 1.0), 0.0
    delta = _difference(problem.charges[0].position_m, shell.center_m)
    distance = _norm(delta)
    if distance == 0:
        return (0.0, 0.0, 1.0), 0.0
    return (tuple(v/distance for v in delta), distance)


def _perpendicular(axis: Vector3) -> Vector3:
    """Deterministic unit vector in the plane perpendicular to the axis."""
    reference = (1.0, 0.0, 0.0) if abs(axis[0]) < 0.9 else (0.0, 1.0, 0.0)
    projection = _dot(reference, axis)
    out = tuple(reference[i] - projection*axis[i] for i in range(3))
    length = _norm(out)
    return tuple(v/length for v in out)


class Femm5Unavailable(RuntimeError):
    """FEMM 5's TCP Python binding or executable could not be started."""


class Femm5ExecutionError(RuntimeError):
    """FEMM 5 failed during geometry construction, solution, or data extraction."""


def _load_binding(path: str | Path | None) -> Any:
    """Load the user-selected FEMM 5 binding under a *private* module name.

    Explicit paths avoid accidentally importing FEMM 4.2's incompatible
    ``femm`` module from the global Python environment.
    """
    raw = str(path) if path is not None else environ.get("FEMM5_BINDING", "")
    if not raw:
        raise Femm5Unavailable(
            "Set FEMM5_BINDING to the FEMM 5 pyfemm/femm.py file, or pass "
            "binding_path=...; pip install pyfemm may load FEMM 4.2 instead"
        )
    file = Path(raw).expanduser().resolve()
    if not file.is_file() or file.name != "femm.py":
        raise Femm5Unavailable(f"FEMM5_BINDING must point to an existing femm.py file: {file}")
    spec = util.spec_from_file_location("_conductor_sim_femm5_binding", file)
    if spec is None or spec.loader is None:
        raise Femm5Unavailable(f"Cannot import FEMM 5 binding from {file}")
    binding = util.module_from_spec(spec)
    try:
        spec.loader.exec_module(binding)
    except Exception as exc:
        raise Femm5Unavailable(f"Could not load FEMM 5 binding at {file}: {exc}") from exc
    if not callable(getattr(binding, "FemmSession", None)):
        raise Femm5Unavailable(
            f"{file} does not expose FemmSession(port=..., headless=True); "
            "verify that it is FEMM 5 pyfemm-qt and not FEMM 4.2"
        )
    return binding


def _unused_tcp_port() -> int:
    """Get a likely unused port; there is a small unavoidable bind race."""
    with socket(AF_INET, SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class Femm5Session:
    """Own one dedicated headless FEMM 5 process/session on a nondefault port."""

    def __init__(self, binding_path: str | Path | None = None, port: int | None = None):
        if port is not None and not 1024 <= port <= 65535:
            raise ValueError("port must be an unprivileged TCP port between 1024 and 65535")
        self.binding_path = binding_path
        self.port = port
        self._client: Any = None

    def open(self) -> Any:
        if self._client is not None:
            raise RuntimeError("FEMM session already open")
        binding = _load_binding(self.binding_path)
        port = self.port if self.port is not None else _unused_tcp_port()
        try:
            self._client = binding.FemmSession(port=port, headless=True)
        except Exception as exc:
            raise Femm5Unavailable(
                f"Could not launch/connect to FEMM 5 headless on port {port}; "
                "check FEMM_DIR and FEMM 5 installation"
            ) from exc
        return self._client

    def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            client.close()


@dataclass(frozen=True, slots=True)
class Femm5Settings:
    """Mesh/domain parameters independent of a particular shell's dimensions."""

    outer_radius_factor: float = 8.0
    arc_degrees: float = 5.0
    mesh_scale: float = 1.0
    solver_precision: float = 1e-9
    source_radius_fraction: float = 0.10  # radius / cavity radius, centered-source regularization

    def __post_init__(self) -> None:
        if not (isfinite(self.outer_radius_factor) and self.outer_radius_factor >= 3):
            raise ValueError("outer_radius_factor must be finite and >= 3")
        if not (isfinite(self.arc_degrees) and 0 < self.arc_degrees <= 15):
            raise ValueError("arc_degrees must lie in (0, 15]")
        if not (isfinite(self.mesh_scale) and 0 < self.mesh_scale <= 5):
            raise ValueError("mesh_scale must lie in (0, 5]")
        if not (isfinite(self.solver_precision) and 0 < self.solver_precision < 1):
            raise ValueError("solver_precision must lie in (0, 1)")
        if not (isfinite(self.source_radius_fraction) and .02 <= self.source_radius_fraction <= .2):
            raise ValueError("source_radius_fraction must lie in [0.02, 0.2]")


def _label_air(client: Any, x: float, size: float, z: float = 0.0) -> None:
    client.ei_addblocklabel(x, z)
    client.ei_selectlabel(x, z)
    client.ei_setblockprop("Vacuum", 0, size, 0)
    client.ei_clearselected()


def _half_circle(client: Any, radius: float, degrees: float, z_center: float = 0.0) -> None:
    """Two quarter-arcs representing a 3D sphere on the r >= 0 half-plane."""
    client.ei_addarc(0, z_center-radius, radius, z_center, 90, degrees)
    client.ei_addarc(radius, z_center, 0, z_center+radius, 90, degrees)


def _mark_arcs(client: Any, radius: float, degrees: float, boundary: str,
               conductor: str, z_center: float = 0.0) -> None:
    midpoint = radius / sqrt(2)
    for z in (z_center-midpoint, z_center+midpoint):
        client.ei_selectarcsegment(midpoint, z)
        client.ei_setarcsegmentprop(degrees, boundary, 0, 0, conductor)
        client.ei_clearselected()


def _compile_shell(
    client: Any, problem: ElectrostaticProblem, settings: Femm5Settings
) -> tuple[float, float | None, Vector3, float]:
    """Emit one axisymmetric shell, with optional *finite spherical source core*.

    A point charge is represented by a small charged *sphere* centered on
    the symmetry axis. For off-center positions the true point-charge limit
    requires source-core radius convergence: an off-center finite conductor
    polarizes in the shell's field, unlike an ideal point source. The FEMM
    source is NOT a point placed at nonzero cylindrical radius (a ring).
    Field and potential samples inside the artificial source are rejected.

    Physical shell inner and outer surfaces use one connected conductor
    property. The source core is a separate, artificial FEMM conductor and
    must never appear as a physical shell in the returned problem result.
    """
    shell = problem.conductors[0]
    a, b = shell.inner_radius_m, shell.outer_radius_m
    R = b * settings.outer_radius_factor
    d = settings.arc_degrees
    q = sum(charge.charge_c for charge in problem.charges)
    symmetry_axis, source_offset = _source_frame(problem)
    # Reserve clearance near the inner shell, and ensure even the closest
    # part of the numerical source is separated from the conductor wall.
    source_radius = (min(a * settings.source_radius_fraction, (a-source_offset)/4)
                     if q != 0.0 else None)

    client.newdocument(1)  # FEMM electrostatics
    client.ei_probdef("meters", "axi", settings.solver_precision)
    relative_eps = problem.permittivity_f_per_m / EPSILON_0
    client.ei_addmaterial("Vacuum", relative_eps, relative_eps, 0)
    if isinstance(shell.constraint, FixedNetCharge):
        client.ei_addconductorprop(shell.name, 0, shell.constraint.charge_c, 0)  # total Q fixed
    else:
        assert isinstance(shell.constraint, Grounded)
        client.ei_addconductorprop(shell.name, 0, 0, 1)  # fixed V=0; total Q is unknown
    if source_radius is not None:
        client.ei_addconductorprop(_SOURCE_CORE, 0, q, 0)  # total source Q fixed

    # Outer Robin condition for the net monopole; is valid when grounded
    # because outside a grounded spherical shell the solution is V=0.
    client.ei_addboundprop("open_robin", 0, 0, problem.permittivity_f_per_m / R, 0, 1)

    # All three physical/outer boundaries share the shell-centered axis;
    # the numerical source's arcs are translated by its axial displacement.
    radii = (a, b, R)
    for radius in radii:
        for x, z in ((0, -radius), (radius, 0), (0, radius)):
            client.ei_addnode(x, z)
        _half_circle(client, radius, d)
    axis_heights = {z for radius in radii for z in (-radius, radius)}
    if source_radius is not None:
        for x, z in ((0, source_offset-source_radius),
                     (source_radius, source_offset), (0, source_offset+source_radius)):
            client.ei_addnode(x, z)
        _half_circle(client, source_radius, d, source_offset)
        axis_heights.update((source_offset-source_radius, source_offset+source_radius))
    axis_heights = sorted(axis_heights)
    for z0, z1 in zip(axis_heights, axis_heights[1:]):
        client.ei_addsegment(0, z0, 0, z1)

    for radius in (a, b):
        _mark_arcs(client, radius, d, "<None>", shell.name)
    if source_radius is not None:
        _mark_arcs(client, source_radius, d, "<None>", _SOURCE_CORE, source_offset)
    _mark_arcs(client, R, d, "open_robin", "<None>")

    if source_radius is not None:
        _label_air(client, source_radius / 2, source_radius / 6 * settings.mesh_scale,
                   source_offset)
        cavity_label_r = (source_offset + source_radius + a) / 2
        _label_air(client, cavity_label_r,
                   min((a - source_offset - source_radius) / 8, a / 8) * settings.mesh_scale)
    else:
        _label_air(client, a / 2, (a / 8) * settings.mesh_scale)
    _label_air(client, (a + b) / 2, min((b - a) / 2, b / 12) * settings.mesh_scale)
    _label_air(client, (b + R) / 2, (b / 4) * settings.mesh_scale)
    return R, source_radius, symmetry_axis, source_offset


def _as_finite_values(data: Any, size: int, description: str) -> tuple[float, ...]:
    if not isinstance(data, (tuple, list)) or len(data) < size:
        raise Femm5ExecutionError(f"FEMM returned malformed {description}: {data!r}")
    try:
        values = tuple(float(x) for x in data[:size])
    except (TypeError, ValueError) as exc:
        raise Femm5ExecutionError(f"FEMM returned nonnumeric {description}: {data!r}") from exc
    if not all(isfinite(value) for value in values):
        raise Femm5ExecutionError(f"FEMM returned nonfinite {description}: {data!r}")
    return values


@dataclass(slots=True)
class Femm5Solution:
    """Live numerical solution; call close() to release the FEMM process.

    Potential/field are queried from FEMM, never substituted by the reference
    solver.  Shell-specific surface charge split is a Gauss-law inference;
    only total conductor charge/potential comes directly from FEMM.
    """

    shell: SphericalShell
    outer_radius_m: float
    conductor_results: tuple[ConductorResult, ...]
    _session: Any = field(repr=False)
    _client: Any = field(repr=False)
    _workspace: TemporaryDirectory[str] = field(repr=False)
    source_radius_m: float | None = None
    numerical_source_charge_c: float | None = None
    symmetry_axis: Vector3 = (0.0, 0.0, 1.0)
    source_displacement_m: float = 0.0
    permittivity_f_per_m: float = EPSILON_0
    _closed: bool = field(default=False, init=False, repr=False)

    def __enter__(self) -> Femm5Solution:
        if self._closed:
            raise RuntimeError("FEMM solution has been closed")
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._session.close()
            finally:
                self._workspace.cleanup()

    def __del__(self) -> None:
        # A fallback only; callers should use a context manager for reliable cleanup.
        if not getattr(self, "_closed", True):
            try:
                self.close()
            except Exception:
                pass

    def _coordinate_data(self, point_m: Vector3) -> tuple[float, float, Vector3, Vector3]:
        if self._closed:
            raise RuntimeError("FEMM solution has been closed")
        if len(point_m) != 3 or any(not isfinite(v) for v in point_m):
            raise ValueError("sample point must be three finite coordinates")
        delta = _difference(point_m, self.shell.center_m)
        z = _dot(delta, self.symmetry_axis)
        rho = tuple(delta[i]-z*self.symmetry_axis[i] for i in range(3))
        r = _norm(rho)
        if _norm(delta) >= self.outer_radius_m:
            raise UnsupportedProblem("sample point lies outside FEMM's finite computational domain")
        if self.source_radius_m is not None and hypot(r, z-self.source_displacement_m) <= self.source_radius_m:
            raise UnsupportedProblem("sample lies in/on the regularized source core; "
                                     "only the point-charge exterior is represented")
        return r, z, delta, rho

    def _sample(self, point_m: Vector3) -> tuple[float, ...]:
        r, z, _, _ = self._coordinate_data(point_m)
        values = self._client.eo_getpointvalues(r, z)
        return _as_finite_values(values, 8, "electrostatic point values")

    def potential_at(self, point_m: Vector3) -> float:
        return self._sample(point_m)[0]

    def electric_field_at(self, point_m: Vector3) -> Vector3:
        if len(point_m) != 3 or any(not isfinite(v) for v in point_m):
            raise ValueError("sample point must be three finite coordinates")
        delta_pre = _difference(point_m, self.shell.center_m)
        distance_pre = _norm(delta_pre)
        # A one-sided jump is a more precise failure than core exclusion.
        if self.source_radius_m is not None:
            dz = _dot(delta_pre, self.symmetry_axis) - self.source_displacement_m
            transverse = tuple(delta_pre[i] - _dot(delta_pre, self.symmetry_axis)*self.symmetry_axis[i]
                               for i in range(3))
            if isclose(hypot(_norm(transverse), dz), self.source_radius_m, rel_tol=1e-12):
                raise SingularField("field has a jump at source-core surface")
        r, z, delta, rho = self._coordinate_data(point_m)
        distance = _norm(delta)
        if any(isclose(distance, radius, rel_tol=1e-12) for radius in
               (self.shell.inner_radius_m, self.shell.outer_radius_m)):
            raise SingularField("field has distinct one-sided conductor surface limits")
        if self.source_radius_m is not None and isclose(
                hypot(r, z-self.source_displacement_m), self.source_radius_m, rel_tol=1e-12):
            raise SingularField("field is discontinuous at source-core surface")
        values = _as_finite_values(self._client.eo_getpointvalues(r, z), 8,
                                   "electrostatic point values")
        er, ez = values[3], values[4]
        return tuple(ez*self.symmetry_axis[i] + (er*rho[i]/r if r else 0.0)
                     for i in range(3))

    def surface_charge_density(
        self, surface: Literal["inner", "outer"], theta_rad: float,
        *, offset_fraction: float = 0.02,
    ) -> float:
        """Estimate sigma from FEMM one-sided E close to the shell surface.

        The metal outward normal points *toward cavity* on the inner face.
        theta is relative to the positive center-to-charge symmetry axis.
        This is a near-boundary field estimate, not a direct FEMM surface-
        element density integral; inspect offset/mesh sensitivity before use.
        """
        if surface not in ("inner", "outer"):
            raise ValueError("surface must be 'inner' or 'outer'")
        if not isfinite(theta_rad) or not 0 <= theta_rad <= pi:
            raise ValueError("theta_rad must be in [0, pi]")
        if not isfinite(offset_fraction) or not 0 < offset_fraction <= 0.10:
            raise ValueError("offset_fraction must be in (0, 0.10]")
        a, b = self.shell.inner_radius_m, self.shell.outer_radius_m
        if surface == "inner":
            gap = min(a * offset_fraction,
                      (a-self.source_displacement_m-(self.source_radius_m or 0)) / 5)
            radius = a-gap
            normal_sign = -1
        else:
            radius = b*(1+offset_fraction)
            normal_sign = +1
        radial = _perpendicular(self.symmetry_axis)
        normal = tuple(cos(theta_rad)*self.symmetry_axis[i] +
                       sin(theta_rad)*radial[i] for i in range(3))
        point = tuple(self.shell.center_m[i] + radius*normal[i] for i in range(3))
        field = self.electric_field_at(point)
        return normal_sign*self.permittivity_f_per_m * _dot(field, normal)


class Femm5Backend:
    """FEMM 5 backend for one sphere and a single point-source location.

    Grounded or fixed-net-charge shell. Any number of point charges at *the
    same location* are merged into one charged source core. An arbitrary
    source direction is rotated into the local FEMM axis (and rotated back
    on field sampling). Multiple different source locations are not yet
    supported, even if collinear; they need separate source geometries.
    """

    def __init__(
        self,
        *,
        settings: Femm5Settings | None = None,
        binding_path: str | Path | None = None,
        session_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.settings = settings or Femm5Settings()
        self._session_factory = session_factory or (lambda: Femm5Session(binding_path))

    def supports(self, problem: ElectrostaticProblem) -> bool:
        if len(problem.conductors) != 1:
            return False
        shell = problem.conductors[0]
        return (
            isinstance(shell, SphericalShell)
            and isinstance(shell.constraint, (Grounded, FixedNetCharge))
            and shell.name != _SOURCE_CORE
            and len({charge.position_m for charge in problem.charges}) <= 1
            and all(_norm(_difference(charge.position_m, shell.center_m)) < shell.inner_radius_m
                    for charge in problem.charges)
            and isfinite(sum(charge.charge_c for charge in problem.charges))
        )

    def solve(self, problem: ElectrostaticProblem) -> Femm5Solution:
        if not self.supports(problem):
            raise UnsupportedProblem(
                "FEMM 5 supports one spherical shell with Grounded or FixedNetCharge "
                "and one source position strictly inside the cavity; "
                "multiple independent source positions are not yet supported"
            )
        session = self._session_factory()
        workspace = TemporaryDirectory(prefix="conductor_sim_femm5_")
        try:
            client = session.open()
            radius, source_radius, axis, displacement = _compile_shell(
                client, problem, self.settings
            )
            client.ei_saveas((Path(workspace.name) / "charged_shell.fee").as_posix())
            client.ei_analyze(1)
            client.ei_loadsolution()
            shell = problem.conductors[0]
            voltage, shell_charge = _as_finite_values(
                client.eo_getconductorproperties(shell.name), 2, "shell conductor results"
            )
            source_charge: float | None = None
            if source_radius is not None:
                _, source_charge = _as_finite_values(
                    client.eo_getconductorproperties(_SOURCE_CORE), 2,
                    "regularized source charge results"
                )
            # Surface charge partition follows from Gauss's law, using FEMM's
            # measured Q_source and Q_shell. The local surface sigma(r,z) is
            # not independently sampled by this API.
            inner = -(source_charge if source_charge is not None else 0.0)
            conductor = ConductorResult(
                shell.name, voltage, shell_charge, inner, shell_charge - inner
            )
            return Femm5Solution(
                shell=shell, outer_radius_m=radius,
                conductor_results=(conductor,), _session=session, _client=client,
                _workspace=workspace, source_radius_m=source_radius,
                numerical_source_charge_c=source_charge,
                symmetry_axis=axis, source_displacement_m=displacement,
                permittivity_f_per_m=problem.permittivity_f_per_m,
            )
        except Exception as exc:
            try:
                session.close()
            finally:
                workspace.cleanup()
            if isinstance(exc, (Femm5Unavailable, UnsupportedProblem, Femm5ExecutionError)):
                raise
            raise Femm5ExecutionError(f"FEMM 5 spherical-shell solve failed: {exc}") from exc
