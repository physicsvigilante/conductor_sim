"""M38a: externally prescribed 3D straight-line path past one grounded sphere.

Every time sample is a NEW public M37a Kelvin observation. The path is
kinematic (an external actuator), not a free Newtonian charged-particle orbit.
The qualified path is checked continuously at its analytic closest approach:
a browser or a finite list of samples cannot hide a metal/guard crossing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from math import fsum, hypot, isfinite, ulp

from .grounded_sphere_wire import observe_grounded_sphere_wire
from .model import ElectrostaticProblem, Grounded, PointCharge, SolidSphere, Vector3
from .motion_1d import _NONRELATIVISTIC_SPEED_LIMIT_M_S
from .solver import UnsupportedProblem


SCHEMA = "conductor_sim.prescribed_ground_line.v1"
PROTOCOL = "external prescribed CONSTANT 3D Cartesian velocity; static Kelvin observations; NOT Newtonian dynamics"
SOURCE_ROLE = "teacher_hypothetical_off_axis_transfer_not_original_workbook_p8_Q21"
UNITS = {
    "time": "s", "distance": "m", "position": "m",
    "velocity": "m/s", "radial_velocity": "m/s",
    "charge": "C", "current": "A", "force": "N",
    "energy": "J", "permittivity": "F/m",
}


def _real(value, label, *, positive=False):
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError(f"{label} must be a finite non-Boolean SI real")
    if positive and value <= 0:
        raise ValueError(f"{label} must be strictly positive")
    return float(value)


def _vector(value, label):
    if type(value) not in (tuple, list) or len(value) != 3:
        raise ValueError(f"{label} must be a real SI three-vector")
    v = tuple(_real(x, label) for x in value)
    if not isfinite(hypot(*v)):
        raise ValueError(f"{label} has unrepresentable magnitude")
    return v


def _position(x0, velocity, time):
    return tuple(x0[j] + velocity[j] * time for j in range(3))


def _closest(x0, centre, velocity, duration):
    """Constrained analytic closest approach on the WHOLE line segment."""
    r0 = tuple(x0[j] - centre[j] for j in range(3))
    speed = hypot(*velocity)
    if not isfinite(hypot(*r0)) or not isfinite(speed) or speed == 0:
        raise ValueError("source offset or path speed is not representable")
    try:
        along = fsum(r0[j] * (velocity[j] / speed) for j in range(3))
        unconstrained = -along / speed
    except OverflowError as exc:
        raise ValueError("closest-approach geometry overflows") from exc
    if not isfinite(unconstrained):
        raise ValueError("closest-approach time is not numerically resolved")
    time = max(0.0, min(duration, unconstrained))
    point = _position(x0, velocity, time)
    if not all(isfinite(x) for x in point):
        raise ValueError("closest-approach position overflows")
    distance = hypot(*(point[j] - centre[j] for j in range(3)))
    if not isfinite(distance):
        raise ValueError("closest-approach distance overflows")
    return time, distance


@dataclass(frozen=True, slots=True)
class PrescribedSphereLineSpec:
    """One supported static Kelvin model + authentic constant 3D velocity."""

    problem: ElectrostaticProblem
    velocity_m_s: Vector3
    duration_s: float
    sample_count: int = 41
    surface_guard_m: float | None = None

    def __post_init__(self):
        if type(self.problem) is not ElectrostaticProblem:
            raise TypeError("line protocol requires an immutable ElectrostaticProblem")
        if len(self.problem.charges) != 1 or len(self.problem.conductors) != 1:
            raise UnsupportedProblem("line protocol requires one charge and one solid conductor")
        sphere, source = self.problem.conductors[0], self.problem.charges[0]
        if type(source) is not PointCharge or type(sphere) is not SolidSphere or type(sphere.constraint) is not Grounded:
            raise UnsupportedProblem("line protocol requires one external point charge and one grounded solid sphere")
        q = _real(source.charge_c, "moving charge")
        a = _real(sphere.radius_m, "solid sphere radius", positive=True)
        _real(self.problem.permittivity_f_per_m, "permittivity", positive=True)
        if q == 0:
            raise UnsupportedProblem("grounding-current source charge cannot be zero")
        initial = _vector(source.position_m, "initial source position")
        centre = _vector(sphere.center_m, "sphere centre")
        velocity = _vector(self.velocity_m_s, "Cartesian source velocity")
        speed = hypot(*velocity)
        if speed == 0:
            raise ValueError("nonzero physical velocity required; this is not a static sweep")
        if speed >= _NONRELATIVISTIC_SPEED_LIMIT_M_S:
            raise UnsupportedProblem("prescribed velocity must remain below 0.1c")
        duration = _real(self.duration_s, "prescribed duration", positive=True)
        if type(self.sample_count) is not int or not 3 <= self.sample_count <= 201:
            raise ValueError("sample_count must be an integer between 3 and 201")
        guard = a * 1e-6 if self.surface_guard_m is None else _real(
            self.surface_guard_m, "surface approach guard", positive=True
        )
        if not isfinite(guard) or not isfinite(a + guard) or guard <= 0:
            raise ValueError("surface guard is not a positive finite physical clearance")
        end = _position(initial, velocity, duration)
        if not all(isfinite(x) for x in end) or end == initial:
            raise UnsupportedProblem("prescribed physical displacement overflows or is not resolved")
        # No valid local metal separation when world-coordinate ulps dwarf
        # the asserted physical surface guard (even if sampled endpoints look safe).
        resolution = max(ulp(abs(z)) for z in (*initial, *centre, *end))
        if 64 * resolution >= min(a, guard):
            raise UnsupportedProblem("world-space geometry cannot resolve guarded solid sphere")
        step = _position(initial, velocity, duration / (self.sample_count - 1))
        if step == initial:
            raise UnsupportedProblem("saved physical line step is smaller than coordinate resolution")
        closest_t, closest_d = _closest(initial, centre, velocity, duration)
        # Conservative floating-point margin: never accept a grazing path
        # when the clearance cannot be numerically distinguished from guard.
        tolerance = 128 * ulp(max(closest_d, a + guard))
        if not closest_d > a + guard + tolerance:
            raise UnsupportedProblem("continuous path intersects or cannot resolve grounded-sphere guard")
        object.__setattr__(self, "velocity_m_s", velocity)
        object.__setattr__(self, "duration_s", duration)
        object.__setattr__(self, "surface_guard_m", guard)

    @property
    def closest_approach(self):
        source = self.problem.charges[0]
        sphere = self.problem.conductors[0]
        return _closest(source.position_m, sphere.center_m, self.velocity_m_s, self.duration_s)


@dataclass(frozen=True, slots=True)
class PrescribedSphereLineSample:
    index: int
    time_s: float
    radial_distance_m: float
    radial_velocity_m_s: float
    source_position_m: Vector3
    source_velocity_m_s: Vector3
    induced_sphere_net_charge_c: float
    wire_current_sphere_to_ground_a: float
    wire_current_ground_to_sphere_a: float
    induced_force_on_source_n: Vector3
    induced_mechanical_energy_j: float


@dataclass(frozen=True, slots=True)
class PrescribedSphereLineTrace:
    spec: PrescribedSphereLineSpec
    samples: tuple[PrescribedSphereLineSample, ...]

    def as_dict(self):
        validate_prescribed_sphere_line(self)
        sphere, source = self.spec.problem.conductors[0], self.spec.problem.charges[0]
        closest_t, closest_d = self.spec.closest_approach
        return {
            "schema": SCHEMA,
            "protocol": PROTOCOL,
            "source_role": SOURCE_ROLE,
            "original_workbook_case": "p8 Multiple Correct Q21 is RADIAL; this path is a teacher extension",
            "units": dict(UNITS),
            "sphere_name": sphere.name,
            "sphere_constraint": "Grounded",
            "sphere_is_solid": True,
            "sphere_center_m": list(sphere.center_m),
            "sphere_radius_m": sphere.radius_m,
            "permittivity_f_per_m": self.spec.problem.permittivity_f_per_m,
            "source_charge_c": source.charge_c,
            "initial_source_position_m": list(source.position_m),
            "constant_cartesian_velocity_m_s": list(self.spec.velocity_m_s),
            "duration_s": self.spec.duration_s,
            "surface_guard_m": self.spec.surface_guard_m,
            "sample_count": self.spec.sample_count,
            "closest_approach_time_s": closest_t,
            "closest_approach_distance_m": closest_d,
            "samples": [asdict(row) for row in self.samples],
        }


def sample_prescribed_sphere_line(spec: PrescribedSphereLineSpec) -> PrescribedSphereLineTrace:
    """Real authored positions -> fresh solved Kelvin force, charge and current."""
    if type(spec) is not PrescribedSphereLineSpec:
        raise TypeError("expected PrescribedSphereLineSpec")
    # Revalidate at use time: a frozen dataclass is not an immutability proof
    # against object.__setattr__ or mutable nested legacy input.
    checked = replace(spec)
    sphere, source = checked.problem.conductors[0], checked.problem.charges[0]
    x0 = source.position_m
    v = checked.velocity_m_s
    rows = []
    previous = None
    for i in range(checked.sample_count):
        t = checked.duration_s * i / (checked.sample_count - 1)
        point = x0 if i == 0 else _position(x0, v, t)
        if not all(isfinite(x) for x in point) or (previous is not None and point == previous):
            raise UnsupportedProblem("actual saved physical positions are not numerically resolved")
        shifted = replace(checked.problem, charges=(PointCharge(source.charge_c, point),))
        observed = observe_grounded_sphere_wire(shifted, v)
        if observed.source_position_m != point or observed.sphere_name != sphere.name:
            raise ValueError("native Kelvin state lost the authored physical identity")
        if observed.distance_from_center_m <= sphere.radius_m + checked.surface_guard_m:
            raise UnsupportedProblem("saved point enters guarded metal")
        rows.append(PrescribedSphereLineSample(
            i, t, observed.distance_from_center_m, observed.radial_velocity_m_s,
            point, v, observed.induced_sphere_net_charge_c,
            observed.wire_current_sphere_to_ground_a,
            observed.wire_current_ground_to_sphere_a,
            observed.induced_force_on_source_n, observed.induced_mechanical_energy_j,
        ))
        previous = point
    return PrescribedSphereLineTrace(spec, tuple(rows))


def validate_prescribed_sphere_line(trace: PrescribedSphereLineTrace) -> None:
    """Strict native re-execution, including Boolean-vs-zero tamper refusal."""
    if type(trace) is not PrescribedSphereLineTrace or type(trace.spec) is not PrescribedSphereLineSpec:
        raise TypeError("expected typed native prescribed-line trace")
    if type(trace.samples) is not tuple or any(type(row) is not PrescribedSphereLineSample for row in trace.samples):
        raise ValueError("malformed native line observation rows")
    authentic = sample_prescribed_sphere_line(trace.spec)
    if len(authentic.samples) != len(trace.samples):
        raise ValueError("native line sample count differs from recomputed physical history")
    def stable(rows):
        return json.dumps([asdict(row) for row in rows], sort_keys=True,
                          separators=(",", ":"), allow_nan=False)
    if stable(authentic.samples) != stable(trace.samples):
        raise ValueError("line sample, time, geometry, current or force differs from native re-execution")
