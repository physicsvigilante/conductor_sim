"""FEMM 5 electrostatics adapter: concentric shells and axial charge sites.

This is a deliberately limited, axisymmetric numerical vertical slice.  It produces a real
FEMM 5 axisymmetric solve, *not* a disguised analytical solution.  FEMM is an
optional external application; no FEMM import is performed at package import.

Ownership: Femm5Solution keeps its FEMM session alive for field queries. Use
``with backend.solve(problem) as solution: ...`` or explicitly call ``close()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import util
from math import cos, fsum, hypot, isclose, isfinite, pi, sin, sqrt
from os import environ
from pathlib import Path
from socket import AF_INET, SOCK_STREAM, socket
from tempfile import TemporaryDirectory
from typing import Any, Callable, Literal

from .model import EPSILON_0, ElectrostaticProblem, FixedNetCharge, Grounded, SphericalShell, Vector3
from .solver import ConductorResult, SingularField, UnsupportedProblem
from .model import _position
from ._resources import cleanup_resources


_SOURCE_CORE = "__conductor_sim_source_core"  # purely numerical, NOT a user conductor


def _dot(a: Vector3, b: Vector3) -> float:
    return sum(x*y for x, y in zip(a,b))


def _norm(v: Vector3) -> float:
    return sqrt(_dot(v,v))


def _difference(a: Vector3, b: Vector3) -> Vector3:
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def _active_sites(problem: ElectrostaticProblem) -> tuple[tuple[Vector3, float], ...]:
    """Combine charges only when their *physical positions* coincide exactly.

    Drop zero-net sites: cancelling charges have no physical field and need
    no artificial FEMM conductor. Sorting makes the resulting geometry
    deterministic under permutation of authored charge components.
    """
    grouped: dict[Vector3, list[float]] = {}
    for charge in problem.charges:
        grouped.setdefault(charge.position_m, []).append(charge.charge_c)
    result = ((position, fsum(charges)) for position, charges in grouped.items())
    return tuple(sorted((position, q) for position, q in result if q != 0.0))


def _source_frame(problem: ElectrostaticProblem) -> tuple[Vector3, float]:
    """Choose a deterministic common symmetry axis through the shell center.

    The farthest active site (with a tuple tie break) sets its *positive*
    direction. Collinearity is checked separately in supports(); this
    function never licenses an off-axis arrangement.
    """
    sites = _active_sites(problem)
    if not sites:
        return (0.0, 0.0, 1.0), 0.0
    shell = problem.conductors[0]
    offsets = [_difference(pos, shell.center_m) for pos, _ in sites]
    farthest = max(offsets, key=lambda v: (_norm(v), v))
    distance = _norm(farthest)
    if distance == 0:
        return (0.0, 0.0, 1.0), 0.0
    return tuple(x/distance for x in farthest), distance



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
    source_radius_fraction: float = 0.10  # radius / cavity radius, numerical source regularization

    def __post_init__(self) -> None:
        if not (isfinite(self.outer_radius_factor) and self.outer_radius_factor >= 3):
            raise ValueError("outer_radius_factor must be finite and >= 3")
        if not (isfinite(self.arc_degrees) and 0 < self.arc_degrees <= 10):
            raise ValueError("arc_degrees must lie in (0, 10] (FEMM clamps larger values)")
        if not (isfinite(self.mesh_scale) and 0 < self.mesh_scale <= 5):
            raise ValueError("mesh_scale must lie in (0, 5]")
        if not (isfinite(self.solver_precision) and 1e-16 <= self.solver_precision <= 1e-8):
            raise ValueError("solver_precision must lie in [1e-16, 1e-8] (FEMM supported range)")
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


@dataclass(frozen=True, slots=True)
class SourceCore:
    """Numerical geometry; these are NOT physical authored conductors."""
    name: str
    position_m: Vector3
    axial_offset_m: float
    radius_m: float
    prescribed_charge_c: float


@dataclass(frozen=True, slots=True)
class NumericalSource:
    """Postprocessed FEMM charge on an artificial finite source conductor."""
    name: str
    position_m: Vector3
    axial_offset_m: float
    radius_m: float
    measured_charge_c: float


def _radial_shells(problem: ElectrostaticProblem) -> tuple[SphericalShell, ...]:
    """Canonical radial order; authored conductor order is preserved in results."""
    return tuple(sorted(problem.conductors, key=lambda s: (s.inner_radius_m,s.outer_radius_m)))


def _compile_shell(
    client: Any, problem: ElectrostaticProblem, settings: Femm5Settings
) -> tuple[float, tuple[SourceCore, ...], Vector3, tuple[SphericalShell, ...]]:
    """Compile nested concentric shells plus zero or more axial source cores.

    The same compiler handles one shell and any number of nonintersecting
    concentric shells. Each *physical* shell uses ONE connected FEMM
    conductor property for both faces, with its own grounding/charge rule.
    Each active charge site becomes an independent, small fixed-Q source
    sphere entirely inside the innermost cavity. These artificial conductors
    are never returned as physical problem components.
    """
    shells = _radial_shells(problem)
    inner = shells[0]
    a = inner.inner_radius_m
    R = shells[-1].outer_radius_m * settings.outer_radius_factor
    degrees = settings.arc_degrees
    axis, _ = _source_frame(problem)

    sites = sorted((
        (_dot(_difference(pos, inner.center_m), axis), pos, charge)
        for pos, charge in _active_sites(problem)
    ), key=lambda item: (item[0], item[1]))
    cores: list[SourceCore] = []
    for i, (z, pos, charge) in enumerate(sites):
        neighbor = min((abs(z-z_other) for j,(z_other,_,_) in enumerate(sites)
                        if j!=i),default=float("inf"))
        clearance = a-abs(z)
        radius = min(a*settings.source_radius_fraction,clearance/4,neighbor/4)
        if radius <= 0 or not isfinite(radius):
            raise UnsupportedProblem("cannot construct nonoverlapping FEMM source cores")
        name = _SOURCE_CORE if len(sites)==1 else f"{_SOURCE_CORE}_{i+1}"
        cores.append(SourceCore(name,pos,z,radius,charge))

    client.newdocument(1)
    client.ei_probdef("meters","axi",settings.solver_precision)
    relative_eps = problem.permittivity_f_per_m / EPSILON_0
    client.ei_addmaterial("Vacuum",relative_eps,relative_eps,0)
    for shell in shells:
        if isinstance(shell.constraint, FixedNetCharge):
            client.ei_addconductorprop(shell.name,0,shell.constraint.charge_c,0)
        else:
            assert isinstance(shell.constraint, Grounded)
            client.ei_addconductorprop(shell.name,0,0,1)
    for core in cores:
        client.ei_addconductorprop(core.name,0,core.prescribed_charge_c,0)

    # Exterior of the outermost spherical conductor has only a monopole.
    # The Robin boundary does not prescribe conductor potentials.
    client.ei_addboundprop("open_robin",0,0,problem.permittivity_f_per_m/R,0,1)

    physical_radii = tuple(radius for shell in shells
                           for radius in (shell.inner_radius_m,shell.outer_radius_m))
    for radius in (*physical_radii, R):
        for x,z in ((0,-radius),(radius,0),(0,radius)):
            client.ei_addnode(x,z)
        _half_circle(client,radius,degrees)
    heights = {z for radius in (*physical_radii,R) for z in (-radius,radius)}
    for core in cores:
        zc=core.axial_offset_m
        rad=core.radius_m
        for x,z in ((0,zc-rad),(rad,zc),(0,zc+rad)):
            client.ei_addnode(x,z)
        _half_circle(client,rad,degrees,zc)
        heights.update((zc-rad,zc+rad))
    ordered=sorted(heights)
    for z0,z1 in zip(ordered,ordered[1:]):
        client.ei_addsegment(0,z0,0,z1)

    for shell in shells:
        for radius in (shell.inner_radius_m,shell.outer_radius_m):
            _mark_arcs(client,radius,degrees,"<None>",shell.name)
    for core in cores:
        _mark_arcs(client,core.radius_m,degrees,"<None>",core.name,core.axial_offset_m)
    _mark_arcs(client,R,degrees,"open_robin","<None>")

    for core in cores:
        _label_air(client,core.radius_m/2,
                   core.radius_m/6*settings.mesh_scale,core.axial_offset_m)
    # r=a/2,z=0 lies in the cavity and outside every regularized source core.
    _label_air(client,a/2,a/8*settings.mesh_scale)
    for i,shell in enumerate(shells):
        a_i,b_i=shell.inner_radius_m,shell.outer_radius_m
        _label_air(client,(a_i+b_i)/2,
                   min((b_i-a_i)/2,b_i/12)*settings.mesh_scale)
        if i+1<len(shells):
            next_a=shells[i+1].inner_radius_m
            _label_air(client,(b_i+next_a)/2,
                       min((next_a-b_i)/4,b_i/8)*settings.mesh_scale)
    last_b=shells[-1].outer_radius_m
    _label_air(client,(last_b+R)/2,last_b/4*settings.mesh_scale)
    return R,tuple(cores),axis,shells



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
    numerical_sources: tuple[NumericalSource, ...] = ()
    radial_shells: tuple[SphericalShell, ...] = ()  # includes innermost 'shell'
    symmetry_axis: Vector3 = (0.0, 0.0, 1.0)
    source_displacement_m: float = 0.0
    permittivity_f_per_m: float = EPSILON_0
    _closed: bool = field(default=False, init=False, repr=False)

    def conductor_by_name(self, name: str) -> ConductorResult:
        for conductor in self.conductor_results:
            if conductor.name == name:
                return conductor
        raise KeyError(name)

    def __enter__(self) -> Femm5Solution:
        if self._closed:
            raise RuntimeError("FEMM solution has been closed")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        cleanup_resources((("FEMM solution", self.close),), primary=exc)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            cleanup_resources((("FEMM session", self._session.close),
                               ("FEMM workspace", self._workspace.cleanup)))

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
        point_m = _position(point_m, "FEMM sample point")
        delta = _difference(point_m, self.shell.center_m)
        z = _dot(delta, self.symmetry_axis)
        rho = tuple(delta[i]-z*self.symmetry_axis[i] for i in range(3))
        r = _norm(rho)
        if _norm(delta) >= self.outer_radius_m:
            raise UnsupportedProblem("sample point lies outside FEMM's finite computational domain")
        if any(hypot(r, z-core.axial_offset_m) <= core.radius_m
               for core in self.numerical_sources):
            raise UnsupportedProblem("sample lies in/on a regularized source core; "
                                     "only the point-charge exterior is represented")
        return r, z, delta, rho

    def _sample(self, point_m: Vector3) -> tuple[float, ...]:
        r, z, _, _ = self._coordinate_data(point_m)
        values = self._client.eo_getpointvalues(r, z)
        return _as_finite_values(values, 8, "electrostatic point values")

    def potential_at(self, point_m: Vector3) -> float:
        return self._sample(point_m)[0]

    def electric_field_at(self, point_m: Vector3) -> Vector3:
        point_m = _position(point_m, "FEMM sample point")
        delta_pre = _difference(point_m, self.shell.center_m)
        # A one-sided jump is a more precise failure than core exclusion.
        z_pre = _dot(delta_pre, self.symmetry_axis)
        rho_pre = tuple(delta_pre[i] - z_pre*self.symmetry_axis[i] for i in range(3))
        for core in self.numerical_sources:
            if isclose(hypot(_norm(rho_pre), z_pre-core.axial_offset_m),
                       core.radius_m, rel_tol=1e-12):
                raise SingularField("field has a jump at source-core surface")
        r, z, delta, rho = self._coordinate_data(point_m)
        distance = _norm(delta)
        if any(isclose(distance, radius, rel_tol=1e-12)
               for shell in (self.radial_shells or (self.shell,))
               for radius in (shell.inner_radius_m,shell.outer_radius_m)):
            raise SingularField("field has distinct one-sided conductor surface limits")
        values = _as_finite_values(self._client.eo_getpointvalues(r, z), 8,
                                   "electrostatic point values")
        er, ez = values[3], values[4]
        return tuple(ez*self.symmetry_axis[i] + (er*rho[i]/r if r else 0.0)
                     for i in range(3))

    def surface_charge_density(
        self, surface: Literal["inner","outer"], theta_rad: float,
        *, offset_fraction: float=0.02, conductor_name: str | None=None,
    ) -> float:
        """Estimate local sigma from one-sided FEMM E, never from images.

        For more than one shell `conductor_name` is required. Theta refers
        to the shared, automatically chosen source symmetry axis; for
        other shells sigma should be uniform to numerical accuracy.
        The sample is constrained to the adjacent vacuum gap, not a metal.
        """
        if surface not in ("inner","outer"):
            raise ValueError("surface must be inner or outer")
        if not isfinite(theta_rad) or not 0 <= theta_rad <= pi:
            raise ValueError("theta_rad must be in [0,pi]")
        if not isfinite(offset_fraction) or not 0<offset_fraction<=.1:
            raise ValueError("offset_fraction must lie in (0,.1]")
        shells=self.radial_shells or (self.shell,)
        if conductor_name is None:
            if len(shells) != 1:
                raise UnsupportedProblem("conductor_name required for multiple shells")
            index=0
        else:
            index=next((i for i,s in enumerate(shells) if s.name==conductor_name),-1)
            if index<0:
                raise ValueError(f"unknown physical conductor {conductor_name!r}")
        shell=shells[index]
        a,b=shell.inner_radius_m,shell.outer_radius_m
        if surface=="inner":
            if index==0:
                clearance=min((a-abs(c.axial_offset_m)-c.radius_m
                               for c in self.numerical_sources),default=a)
            else:
                clearance=a-shells[index-1].outer_radius_m
            gap=min(a*offset_fraction,clearance/5)
            radius=a-gap
            sign=-1
        else:
            if index+1<len(shells):
                clearance=shells[index+1].inner_radius_m-b
            else:
                clearance=self.outer_radius_m-b
            gap=min(b*offset_fraction,clearance/5)
            radius=b+gap
            sign=+1
        radial=_perpendicular(self.symmetry_axis)
        normal=tuple(cos(theta_rad)*self.symmetry_axis[i]+
                     sin(theta_rad)*radial[i] for i in range(3))
        point=tuple(self.shell.center_m[i]+radius*normal[i] for i in range(3))
        electric=self.electric_field_at(point)
        return sign*self.permittivity_f_per_m*_dot(electric,normal)


class Femm5Backend:
    """FEMM 5 backend for nested concentric shells and axial charge sites.

    Each physical shell has its own grounded or fixed-Q constraint. One
    independent charged core represents each active point-charge site.
    Non-concentric/intersecting shells, charges outside the innermost
    cavity and non-collinear source sites are explicitly rejected.
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
        if not 1 <= len(problem.conductors) <= 8:
            return False
        shells=_radial_shells(problem)
        center=shells[0].center_m
        if any(not isinstance(shell,SphericalShell)
               or not isinstance(shell.constraint,(Grounded,FixedNetCharge))
               or shell.name.startswith(_SOURCE_CORE)
               or shell.name == "<None>"
               or shell.center_m != center for shell in shells):
            return False
        if any(left.outer_radius_m >= right.inner_radius_m
               for left,right in zip(shells,shells[1:])):
            return False
        a=shells[0].inner_radius_m
        if any(_norm(_difference(q.position_m,center)) >= a for q in problem.charges):
            return False
        try:
            sites=_active_sites(problem)
            axis,_=_source_frame(problem)
        except (OverflowError,ValueError):
            return False
        if len(sites)>16:
            return False
        tolerance=1e-11*a
        for position,_ in sites:
            delta=_difference(position,center)
            z=_dot(delta,axis)
            transverse=tuple(delta[i]-z*axis[i] for i in range(3))
            if _norm(transverse)>tolerance:
                return False
        return True

    def solve(self, problem: ElectrostaticProblem) -> Femm5Solution:
        if not self.supports(problem):
            raise UnsupportedProblem(
                "FEMM 5 M4b supports up to 8 strictly nested concentric shells "
                "and up to 16 charge sites on one symmetry axis inside their "
                "innermost cavity; eccentric, overlapping, or non-collinear "
                "configurations require a different solver"
            )
        session = self._session_factory()
        workspace = TemporaryDirectory(prefix="conductor_sim_femm5_")
        try:
            client = session.open()
            radius, cores, axis, radial_shells = _compile_shell(client, problem, self.settings)
            client.ei_saveas((Path(workspace.name) / "charged_shell.fee").as_posix())
            client.ei_analyze(1)
            client.ei_loadsolution()
            # Observe each numerical charge and each *physical* conductor.
            # No analytical potential is substituted for FEMM output.
            measured: list[NumericalSource] = []
            for core in cores:
                _, charge = _as_finite_values(
                    client.eo_getconductorproperties(core.name),2,
                    f"numerical source {core.name} charge results")
                measured.append(NumericalSource(core.name,core.position_m,
                                                core.axial_offset_m,core.radius_m,charge))
            measured_total=fsum(item.measured_charge_c for item in measured)
            observed_shells: dict[str,tuple[float,float]]={}
            for shell in radial_shells:
                observed_shells[shell.name]=_as_finite_values(
                    client.eo_getconductorproperties(shell.name),2,
                    f"physical shell {shell.name} results")
            # On each metal inner surface: Q_inner = -Q_enclosed just inside.
            # Outer face charge is Q_shell - Q_inner. This is a Gauss-law
            # inference based on independently FEMM-measured net charges.
            enclosed=measured_total
            result_by_name: dict[str,ConductorResult]={}
            enclosed_terms = [measured_total]
            for shell in radial_shells:
                potential,net=observed_shells[shell.name]
                inner=-enclosed
                enclosed_terms.append(net)
                outer=fsum(enclosed_terms)
                result_by_name[shell.name]=ConductorResult(
                    shell.name,potential,net,inner,outer)
                enclosed=outer
            # Return results in *authored* conductor order, not sorted order.
            results=tuple(result_by_name[s.name] for s in problem.conductors)
            one=measured[0] if len(measured)==1 else None
            return Femm5Solution(
                shell=radial_shells[0], radial_shells=radial_shells,
                outer_radius_m=radius,conductor_results=results,
                _session=session,_client=client,_workspace=workspace,
                source_radius_m=one.radius_m if one is not None else None,
                numerical_source_charge_c=measured_total if measured else None,
                numerical_sources=tuple(measured),
                symmetry_axis=axis,
                source_displacement_m=(
                    abs(one.axial_offset_m) if one is not None else
                    max((abs(item.axial_offset_m) for item in measured),default=0.0)
                ),
                permittivity_f_per_m=problem.permittivity_f_per_m,
            )
        except BaseException as exc:
            error = exc
            if isinstance(exc, Exception) and not isinstance(
                    exc, (Femm5Unavailable, UnsupportedProblem, Femm5ExecutionError)):
                error = Femm5ExecutionError(f"FEMM 5 spherical-shell solve failed: {exc}")
            cleanup_resources((("FEMM session", session.close),
                               ("FEMM workspace", workspace.cleanup)), primary=error)
            if error is exc:
                raise
            raise error from exc
