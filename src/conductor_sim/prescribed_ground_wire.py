"""M37b: a teacher-prescribed CONSTANT radial speed past one grounded sphere.

This is an externally maintained kinematic protocol, not free M16a motion.
Every stored observation is a NEW native Kelvin equilibrium from M37a.
A time slider never solves electric fields or Newton's law in the browser.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import fsum, hypot, isfinite

from .grounded_sphere_wire import observe_grounded_sphere_wire
from .model import ElectrostaticProblem, Grounded, PointCharge, SolidSphere, Vector3
from .motion_1d import _NONRELATIVISTIC_SPEED_LIMIT_M_S
from .solver import UnsupportedProblem

PROTOCOL = (
    "EXTERNALLY MAINTAINED constant radial speed; real prescribed elapsed-time "
    "snapshots of static Kelvin equilibria; NOT free Newtonian motion"
)
SCHEMA = "conductor_sim.prescribed_ground_wire.v1"
UNITS = {
    "time": "s", "distance": "m", "position": "m",
    "speed": "m/s", "charge": "C", "current": "A",
    "force": "N", "energy": "J", "permittivity": "F/m",
}


def _real(value, label, *, positive=False):
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError(f"{label} must be a finite non-Boolean real SI number")
    if positive and value <= 0:
        raise ValueError(f"{label} must be strictly positive")
    return float(value)


@dataclass(frozen=True, slots=True)
class PrescribedSphereRadialSpec:
    """One fixed sphere, one moving external charge, signed constant radial speed.

    Positive signed radial speed is OUTWARD; negative is INWARD regardless
    of the world-axis orientation. The source is moved externally; its
    induced electrostatic force is observed, NOT used to integrate a path.
    """
    problem: ElectrostaticProblem
    radial_speed_m_s: float
    duration_s: float
    sample_count: int = 31
    surface_guard_m: float | None = None

    def __post_init__(self):
        if type(self.problem) is not ElectrostaticProblem:
            raise TypeError("requires an immutable ElectrostaticProblem")
        if len(self.problem.charges) != 1 or len(self.problem.conductors) != 1:
            raise UnsupportedProblem("prescribed ground wire requires exactly one source and one conductor")
        sphere, charge = self.problem.conductors[0], self.problem.charges[0]
        if type(sphere) is not SolidSphere or type(sphere.constraint) is not Grounded:
            raise UnsupportedProblem("requires a physically SOLID grounded sphere, not a shell or isolated metal")
        q = _real(charge.charge_c, "source charge")
        a = _real(sphere.radius_m, "sphere radius", positive=True)
        _real(self.problem.permittivity_f_per_m, "permittivity", positive=True)
        if q == 0:
            raise UnsupportedProblem("moving point source must have nonzero charge")
        v = _real(self.radial_speed_m_s, "authored radial speed")
        if v == 0:
            raise ValueError("a static source is not a prescribed moving-charge protocol")
        if abs(v) >= _NONRELATIVISTIC_SPEED_LIMIT_M_S:
            raise UnsupportedProblem("nonrelativistic prescribed speed must be below 0.1c")
        duration = _real(self.duration_s, "prescribed duration", positive=True)
        if type(self.sample_count) is not int or not 3 <= self.sample_count <= 201:
            raise ValueError("sample_count must be an integer between 3 and 201")
        guard = a*1e-6 if self.surface_guard_m is None else _real(
            self.surface_guard_m, "surface approach guard", positive=True)
        if not isfinite(guard) or guard <= 0:
            raise ValueError("surface approach guard must be finite and strictly positive")
        d0 = hypot(*(charge.position_m[j]-sphere.center_m[j] for j in range(3)))
        if not isfinite(d0) or not d0 > a + guard:
            raise UnsupportedProblem("initial source must remain strictly outside guarded solid sphere")
        end = d0 + v*duration
        if not isfinite(end) or not end > a + guard:
            raise UnsupportedProblem("prescribed motion must not reach or cross conductor/surface guard")
        if not isfinite(abs(v)*duration) or end == d0:
            raise ValueError("prescribed displacement underflows or overflows")
        object.__setattr__(self, "radial_speed_m_s", v)
        object.__setattr__(self, "duration_s", duration)
        object.__setattr__(self, "surface_guard_m", guard)

    @property
    def distance_initial_m(self) -> float:
        c = self.problem.charges[0].position_m
        center = self.problem.conductors[0].center_m
        return hypot(*(c[i]-center[i] for i in range(3)))

    @property
    def radial_axis(self) -> Vector3:
        point = self.problem.charges[0].position_m
        center = self.problem.conductors[0].center_m
        d = self.distance_initial_m
        return tuple((point[i]-center[i])/d for i in range(3))


@dataclass(frozen=True, slots=True)
class PrescribedSphereSample:
    index: int
    time_s: float
    radial_distance_m: float
    source_position_m: Vector3
    source_velocity_m_s: Vector3
    induced_sphere_net_charge_c: float
    wire_current_sphere_to_ground_a: float
    wire_current_ground_to_sphere_a: float
    induced_force_on_source_n: Vector3
    induced_mechanical_energy_j: float


@dataclass(frozen=True, slots=True)
class PrescribedSphereTrace:
    spec: PrescribedSphereRadialSpec
    samples: tuple[PrescribedSphereSample, ...]

    def as_dict(self):
        problem = self.spec.problem
        sphere, source = problem.conductors[0], problem.charges[0]
        return {
            "schema": SCHEMA,
            "protocol": PROTOCOL,
            "actual_time_origin": "teacher-prescribed constant-speed kinematics; NOT M16a integration",
            "units": dict(UNITS),
            "source_name": sphere.name,
            "sphere_constraint": "Grounded",
            "sphere_is_solid": True,
            "sphere_center_m": list(sphere.center_m),
            "sphere_radius_m": sphere.radius_m,
            "permittivity_f_per_m": problem.permittivity_f_per_m,
            "source_charge_c": source.charge_c,
            "initial_source_position_m": list(source.position_m),
            "radial_axis": list(self.spec.radial_axis),
            "signed_constant_radial_speed_m_s": self.spec.radial_speed_m_s,
            "duration_s": self.spec.duration_s,
            "sample_count": self.spec.sample_count,
            "surface_guard_m": self.spec.surface_guard_m,
            "samples": [asdict(s) for s in self.samples],
        }


def sample_prescribed_sphere_radial(spec: PrescribedSphereRadialSpec) -> PrescribedSphereTrace:
    """Recompute full Kelvin response at each actual *prescribed* timestamp."""
    if type(spec) is not PrescribedSphereRadialSpec:
        raise TypeError("requires a qualified PrescribedSphereRadialSpec")
    problem = spec.problem
    sphere, source = problem.conductors[0], problem.charges[0]
    axis = spec.radial_axis
    d0 = spec.distance_initial_m
    velocity = tuple(spec.radial_speed_m_s*axis[i] for i in range(3))
    samples = []
    for i in range(spec.sample_count):
        time_s = spec.duration_s*i/(spec.sample_count-1)
        distance = d0 + spec.radial_speed_m_s*time_s
        if not isfinite(distance) or distance <= sphere.radius_m+spec.surface_guard_m:
            raise UnsupportedProblem("source approaches the ideal metal/surface guard")
        point = (source.position_m if i == 0 else
                 tuple(sphere.center_m[k]+distance*axis[k] for k in range(3)))
        if not all(isfinite(x) for x in point):
            raise ValueError("world-space physical source position overflowed")
        moved = replace(problem, charges=(PointCharge(source.charge_c,point),))
        obs = observe_grounded_sphere_wire(moved,velocity)
        # A large world translation cannot silently change the intended
        # physical separation below double-precision resolution.
        scale = max(abs(distance),abs(sphere.radius_m),abs(spec.radial_speed_m_s*time_s))
        if abs(obs.distance_from_center_m-distance)>3e-11*scale:
            raise UnsupportedProblem("physical radial separation lost numerical accuracy")
        samples.append(PrescribedSphereSample(
            i,time_s,obs.distance_from_center_m,point,velocity,
            obs.induced_sphere_net_charge_c,
            obs.wire_current_sphere_to_ground_a,
            obs.wire_current_ground_to_sphere_a,
            obs.induced_force_on_source_n,
            obs.induced_mechanical_energy_j,
        ))
    if samples[0].source_position_m != source.position_m:
        # No unnoticed shift at t=0: important for teacher-authored source
        # identity, including arbitrary translated centres.
        raise UnsupportedProblem("authored initial charge position cannot be reproduced exactly")
    return PrescribedSphereTrace(spec, tuple(samples))
