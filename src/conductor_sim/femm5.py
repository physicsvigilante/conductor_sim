"""FEMM 5 electrostatics adapter: one spherical shell and centered charges.

This is a deliberately limited, axisymmetric numerical vertical slice.  It produces a real
FEMM 5 axisymmetric solve, *not* a disguised analytical solution.  FEMM is an
optional external application; no FEMM import is performed at package import.

Ownership: Femm5Solution keeps its FEMM session alive for field queries. Use
``with backend.solve(problem) as solution: ...`` or explicitly call ``close()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import util
from math import hypot, isclose, isfinite, sqrt
from os import environ
from pathlib import Path
from socket import AF_INET, SOCK_STREAM, socket
from tempfile import TemporaryDirectory
from typing import Any, Callable

from .model import EPSILON_0, ElectrostaticProblem, FixedNetCharge, Grounded, SphericalShell, Vector3
from .solver import ConductorResult, SingularField, UnsupportedProblem


_SOURCE_CORE = "__conductor_sim_source_core"  # purely numerical, NOT a user conductor


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


def _label_air(client: Any, x: float, size: float) -> None:
    client.ei_addblocklabel(x, 0)
    client.ei_selectlabel(x, 0)
    client.ei_setblockprop("Vacuum", 0, size, 0)
    client.ei_clearselected()


def _half_circle(client: Any, radius: float, degrees: float) -> None:
    """Two quarter-arcs spanning the right half-plane, r >= 0."""
    client.ei_addarc(0, -radius, radius, 0, 90, degrees)
    client.ei_addarc(radius, 0, 0, radius, 90, degrees)


def _mark_arcs(client: Any, radius: float, degrees: float, boundary: str, conductor: str) -> None:
    midpoint = radius / sqrt(2)
    for z in (-midpoint, midpoint):
        client.ei_selectarcsegment(midpoint, z)
        client.ei_setarcsegmentprop(degrees, boundary, 0, 0, conductor)
        client.ei_clearselected()


def _compile_shell(
    client: Any, problem: ElectrostaticProblem, settings: Femm5Settings
) -> tuple[float, float | None]:
    """Emit one axisymmetric shell, with optional *finite spherical source core*.

    A centered ideal point charge is represented for the FEMM solve by a tiny
    concentric conductor carrying the source's *total fixed charge*. This is
    NOT a literal off-axis 2D FEMM point property, which would revolve into
    a ring. For spherical symmetry the field and potential at radii greater
    than the source core match the ideal centered point charge; sampling
    inside the numerical core is rejected.

    Physical shell inner and outer surfaces use one connected conductor
    property. The source core is a separate, artificial FEMM conductor and
    must never appear as a physical shell in the returned problem result.
    """
    shell = problem.conductors[0]
    a, b = shell.inner_radius_m, shell.outer_radius_m
    R = b * settings.outer_radius_factor
    d = settings.arc_degrees
    q = sum(charge.charge_c for charge in problem.charges)
    source_radius = a * settings.source_radius_fraction if q != 0.0 else None

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

    radii = ((source_radius,) if source_radius is not None else ()) + (a, b, R)
    for radius in radii:
        for x, z in ((0, -radius), (radius, 0), (0, radius)):
            client.ei_addnode(x, z)
        _half_circle(client, radius, d)

    axis_heights = sorted({z for radius in radii for z in (-radius, radius)})
    for z0, z1 in zip(axis_heights, axis_heights[1:]):
        client.ei_addsegment(0, z0, 0, z1)

    for radius in (a, b):
        _mark_arcs(client, radius, d, "<None>", shell.name)
    if source_radius is not None:
        _mark_arcs(client, source_radius, d, "<None>", _SOURCE_CORE)
    _mark_arcs(client, R, d, "open_robin", "<None>")

    if source_radius is not None:
        _label_air(client, source_radius / 2, source_radius / 6 * settings.mesh_scale)
        _label_air(client, (source_radius + a) / 2,
                   min((a - source_radius) / 8, a / 8) * settings.mesh_scale)
    else:
        _label_air(client, a / 2, (a / 8) * settings.mesh_scale)
    _label_air(client, (a + b) / 2, min((b - a) / 2, b / 12) * settings.mesh_scale)
    _label_air(client, (b + R) / 2, (b / 4) * settings.mesh_scale)
    return R, source_radius


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

    def _sample(self, point_m: Vector3) -> tuple[float, ...]:
        if self._closed:
            raise RuntimeError("FEMM solution has been closed")
        if len(point_m) != 3 or any(not isfinite(v) for v in point_m):
            raise ValueError("sample point must be three finite coordinates")
        cx, cy, cz = self.shell.center_m
        dx, dy, dz = point_m[0] - cx, point_m[1] - cy, point_m[2] - cz
        r = hypot(dx, dy)
        distance = hypot(r, dz)
        if distance >= self.outer_radius_m:
            raise UnsupportedProblem("sample point lies outside FEMM's finite computational domain")
        if self.source_radius_m is not None and distance <= self.source_radius_m:
            raise UnsupportedProblem("sample point lies in the regularized source core; "
                                     "only the ideal-charge exterior is represented")
        values = self._client.eo_getpointvalues(r, dz)
        return _as_finite_values(values, 8, "electrostatic point values")

    def potential_at(self, point_m: Vector3) -> float:
        return self._sample(point_m)[0]

    def electric_field_at(self, point_m: Vector3) -> Vector3:
        cx, cy, cz = self.shell.center_m
        dx, dy, dz = point_m[0] - cx, point_m[1] - cy, point_m[2] - cz
        r = hypot(dx, dy)
        distance = hypot(r, dz)
        surfaces = (self.shell.inner_radius_m, self.shell.outer_radius_m)
        if self.source_radius_m is not None:
            surfaces += (self.source_radius_m,)
        if any(isclose(distance, radius, rel_tol=1e-12) for radius in surfaces):
            raise SingularField("electric field has a one-sided jump at the conductor surface")
        _, _, _, er, ez, _, _, _ = self._sample(point_m)
        if r == 0:
            return (0.0, 0.0, ez)
        return (er * dx / r, er * dy / r, ez)


class Femm5Backend:
    """FEMM 5 backend for one spherical shell and centered point charges.

    Grounded or fixed-shell-charge constraints are supported. Any nonzero
    centered point charges are represented by *one* concentric numerical
    source core. General 3D/off-center charges are rejected, not approximated.
    Real FEMM numerical accuracy is checked separately from fake-client tests.
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
            and all(charge.position_m == shell.center_m for charge in problem.charges)
            and isfinite(sum(charge.charge_c for charge in problem.charges))
        )

    def solve(self, problem: ElectrostaticProblem) -> Femm5Solution:
        if not self.supports(problem):
            raise UnsupportedProblem(
                "FEMM 5 supports one spherical shell with Grounded or FixedNetCharge "
                "and only centered point charges (no arbitrary 3D/off-center charges)"
            )
        session = self._session_factory()
        workspace = TemporaryDirectory(prefix="conductor_sim_femm5_")
        try:
            client = session.open()
            radius, source_radius = _compile_shell(client, problem, self.settings)
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
            )
        except Exception as exc:
            try:
                session.close()
            finally:
                workspace.cleanup()
            if isinstance(exc, (Femm5Unavailable, UnsupportedProblem, Femm5ExecutionError)):
                raise
            raise Femm5ExecutionError(f"FEMM 5 spherical-shell solve failed: {exc}") from exc
