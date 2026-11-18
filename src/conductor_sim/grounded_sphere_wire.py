"""M37a: ideal grounded-wire charge flux from genuine Kelvin sphere equilibria.

A supplied instantaneous velocity is only *kinematics*, not an integrated path.
For actual elapsed-time observations consume M16a's existing motion solver.
This module neither computes a second electrostatic force law nor models
resistance, electron drift, radiation or collision.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import fsum, hypot, isclose, isfinite, ulp

from .external_sphere import ExternalGroundedSphereBackend
from .model import ElectrostaticProblem, Grounded, SolidSphere, Vector3
from .motion_1d import (
    Motion1DResult, Motion1DSpec, MotionEvent, MotionSample,
    _NONRELATIVISTIC_SPEED_LIMIT_M_S,
)
from .observables import force_energy
from .solver import UnsupportedProblem


def _number(value, label: str) -> float:
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError(f"{label} must be a finite real number, not a boolean")
    return float(value)


def _vector(value, label: str) -> Vector3:
    if type(value) not in (list, tuple) or len(value) != 3:
        raise ValueError(f"{label} must be a finite three-vector")
    values = tuple(_number(x, label) for x in value)
    if not isfinite(hypot(*values)):
        raise ValueError(f"{label} vector magnitude is not representable")
    return values


def _dot(a, b):
    return fsum(a[i] * b[i] for i in range(3))


def _matches(actual, expected):
    return (type(actual) in (int, float) and isfinite(actual)
            and isclose(actual, expected, rel_tol=2e-8,
                        abs_tol=128 * ulp(abs(expected))))


@dataclass(frozen=True, slots=True)
class GroundedSphereWireObservation:
    """SI quantities at ONE prescribed instant; no implied time history.

    Positive `wire_current_sphere_to_ground_a` = conventional positive charge
    flowing from the sphere into its ideal ground connection. The opposite
    current is `wire_current_ground_to_sphere_a`.
    """

    sphere_name: str
    sphere_center_m: Vector3
    sphere_radius_m: float
    point_charge_c: float
    permittivity_f_per_m: float
    source_position_m: Vector3
    source_velocity_m_s: Vector3
    distance_from_center_m: float
    radial_velocity_m_s: float
    induced_sphere_net_charge_c: float
    induced_net_charge_rate_c_per_s: float
    wire_current_sphere_to_ground_a: float
    wire_current_ground_to_sphere_a: float
    induced_force_on_source_n: Vector3
    induced_mechanical_energy_j: float

    def as_dict(self):
        return dict(
            schema="conductor_sim.grounded_sphere_wire_instant.v1",
            mode="instantaneous imposed kinematics; no trajectory asserted",
            units={"length": "m", "velocity": "m/s", "charge": "C",
                   "current": "A", "force": "N", "energy": "J"},
            current_sign="positive sphere_to_ground is conventional +charge flux",
            **asdict(self),
        )


def observe_grounded_sphere_wire(
    problem: ElectrostaticProblem,
    source_velocity_m_s: Vector3,
) -> GroundedSphereWireObservation:
    """Observe one signed conventional wire-current derivative, at this instant.

    Uses the *actual* qualified static Kelvin backend and force_energy; only
    differentiates the known grounded-sphere equilibrium charge Q=-q*a/d.
    Tangential motion has I_wire=0 even when local sigma redistributes.
    No statement about the external force maintaining a prescribed velocity.
    """
    if type(problem) is not ElectrostaticProblem or len(problem.charges) != 1 or len(problem.conductors) != 1:
        raise UnsupportedProblem("wire current requires one point charge and one grounded solid sphere")
    sphere, source = problem.conductors[0], problem.charges[0]
    if type(sphere) is not SolidSphere or type(sphere.constraint) is not Grounded or source.charge_c == 0:
        raise UnsupportedProblem("wire current requires a nonzero point charge outside a grounded solid sphere")
    velocity = _vector(source_velocity_m_s, "source velocity")
    if hypot(*velocity) >= _NONRELATIVISTIC_SPEED_LIMIT_M_S:
        raise UnsupportedProblem("ideal quasistatic current requires velocity below the classical speed cutoff")
    dx = tuple(source.position_m[i] - sphere.center_m[i] for i in range(3))
    d = hypot(*dx)
    if not isfinite(d) or not d > sphere.radius_m:
        raise UnsupportedProblem("source must be strictly outside the grounded solid sphere")
    radial_velocity = fsum(dx[i] * velocity[i] / d for i in range(3))

    native = ExternalGroundedSphereBackend().solve(problem)
    force_data = force_energy(native)
    if len(force_data.charges) != 1 or force_data.charges[0].source != source:
        raise UnsupportedProblem("moving source identity is not uniquely retained")
    charge_on_sphere = native.conductor_results[0].net_charge_c
    # If d increases, negative induced charge becomes less negative.
    charge_rate = source.charge_c * (sphere.radius_m / d) * (radial_velocity / d)
    # The ground terminal has the opposite sign of the sphere-charge rate.
    sphere_to_ground = -charge_rate
    values = (d, radial_velocity, charge_on_sphere, charge_rate,
              sphere_to_ground, force_data.energy.total_j,
              *force_data.charges[0].total_force_n)
    if not all(isfinite(x) for x in values):
        raise ValueError("Kelvin charge, current, force or energy overflowed")
    return GroundedSphereWireObservation(
        sphere.name, sphere.center_m, sphere.radius_m, source.charge_c,
        problem.permittivity_f_per_m, source.position_m, velocity, d, radial_velocity, charge_on_sphere,
        charge_rate, sphere_to_ground, charge_rate,
        force_data.charges[0].total_force_n, force_data.energy.total_j,
    )


@dataclass(frozen=True, slots=True)
class GroundedSphereWireTrace:
    """Saved exact-state charge/current witnesses attached to M16a motion."""

    motion: Motion1DResult
    observations: tuple[GroundedSphereWireObservation, ...]

    def as_dict(self):
        return {
            "schema": "conductor_sim.grounded_sphere_wire_motion.v1",
            "mode": "actual elapsed-time M16a motion; instant Kelvin response",
            "current_sign": "sphere_to_ground positive for conventional +charge flux",
            "physics_regime": "ideal quasistatic, classical point charge; no wire impedance or impact",
            "motion": self.motion.as_dict(),
            "observations": [
                dict(time_s=sample.time_s, **obs.as_dict())
                for sample, obs in zip(self.motion.samples, self.observations)
            ],
        }


def observe_grounded_sphere_motion(motion: Motion1DResult) -> GroundedSphereWireTrace:
    """Augment a valid radial grounded-sphere M16a history, not an M13 sweep.

    Check positions, observable physics, K/U and termination of saved samples.
    This does NOT independently re-integrate the trajectory or authenticate
    arbitrarily constructed Python objects; the source solver must be trusted.
    """
    if type(motion) is not Motion1DResult or type(motion.spec) is not Motion1DSpec:
        raise TypeError("expected an actual Motion1DResult with a Motion1DSpec")
    spec = motion.spec
    if len(spec.problem.conductors) != 1 or len(spec.problem.charges) != 1 or spec.charge_index != 0:
        raise UnsupportedProblem("current trace requires the single moving source inventory")
    sphere = spec.problem.conductors[0]
    if type(sphere) is not SolidSphere or type(sphere.constraint) is not Grounded:
        raise UnsupportedProblem("current trace requires a grounded solid sphere")
    if motion.termination not in ("duration", "surface-guard"):
        raise ValueError("unsupported motion termination")
    if not motion.samples or any(type(s) is not MotionSample for s in motion.samples):
        raise ValueError("motion has no authentic typed samples")
    if any(type(e) is not MotionEvent for e in motion.events):
        raise ValueError("motion has malformed events")
    initial = spec.selected_charge
    radius = sphere.radius_m
    guard = spec.surface_guard_m if spec.surface_guard_m is not None else 1e-6 * radius
    charge = initial.charge_c
    from .model import PointCharge
    observations = []
    first_kinetic = first_energy = None
    max_drift = 0.0
    previous_t = -1.0
    for index, row in enumerate(motion.samples):
        values = (row.time_s, row.displacement_m, row.velocity_m_s, row.acceleration_m_s2,
                  row.force_n, row.kinetic_energy_j, row.interaction_energy_j,
                  row.total_energy_j, row.energy_drift_j)
        if any(type(v) not in (int, float) or not isfinite(v) for v in values):
            raise ValueError("motion sample has nonfinite or Boolean numerical data")
        if not (0 <= row.time_s <= spec.duration_s and row.time_s > previous_t):
            raise ValueError("motion samples must have ordered actual elapsed timestamps")
        previous_t = row.time_s
        position = tuple(initial.position_m[i] + spec.axis[i] * row.displacement_m
                         for i in range(3))
        if type(row.position_m) is not tuple or len(row.position_m) != 3 or any(
            type(v) not in (int, float) or not isfinite(v) or v != position[i]
            for i, v in enumerate(row.position_m)
        ):
            raise ValueError("saved motion position contradicts its authored axis and displacement")
        d = hypot(*(position[i] - sphere.center_m[i] for i in range(3)))
        if d <= radius + guard:
            raise ValueError("saved motion crosses the ideal sphere approach guard")
        velocity = tuple(spec.axis[i] * row.velocity_m_s for i in range(3))
        if hypot(*velocity) >= _NONRELATIVISTIC_SPEED_LIMIT_M_S:
            raise ValueError("saved motion violates the nonrelativistic speed cutoff")
        from dataclasses import replace
        problem = replace(spec.problem, charges=(PointCharge(charge, position),))
        obs = observe_grounded_sphere_wire(problem, velocity)
        force = _dot(obs.induced_force_on_source_n, spec.axis)
        kinetic = 0.5 * spec.mass_kg * row.velocity_m_s**2
        total = kinetic + obs.induced_mechanical_energy_j
        if first_kinetic is None:
            first_kinetic = kinetic
            first_energy = obs.induced_mechanical_energy_j
            if row.time_s != 0 or row.displacement_m != 0 or row.velocity_m_s != spec.initial_velocity_m_s:
                raise ValueError("first saved state must preserve the exact authored physical initial state")
        drift = fsum((kinetic - first_kinetic, obs.induced_mechanical_energy_j - first_energy))
        for label, actual, expected in (
            ("force", row.force_n, force),
            ("acceleration", row.acceleration_m_s2, force / spec.mass_kg),
            ("kinetic energy", row.kinetic_energy_j, kinetic),
            ("interaction energy", row.interaction_energy_j, obs.induced_mechanical_energy_j),
            ("total energy", row.total_energy_j, total),
            ("energy drift", row.energy_drift_j, drift),
        ):
            if not _matches(actual, expected):
                raise ValueError(f"saved motion {label} contradicts physical Kelvin state")
        max_drift = max(max_drift, abs(drift))
        observations.append(obs)
    if not _matches(motion.max_abs_energy_drift_j, max_drift):
        raise ValueError("motion maximum energy drift contradicts saved actual states")
    if motion.termination == "duration" and motion.samples[-1].time_s != spec.duration_s:
        raise ValueError("duration termination has no actual final sample")
    for event in motion.events:
        if (event.kind not in ("turning-point", "surface-guard") or
                type(event.time_s) not in (int, float) or not isfinite(event.time_s)
                or not 0 <= event.time_s <= motion.samples[-1].time_s):
            raise ValueError("motion has an invalid event")
    if motion.termination == "surface-guard":
        if not motion.events or motion.events[-1].kind != "surface-guard":
            raise ValueError("surface-guard termination must retain an approach event")
        if not _matches(motion.events[-1].time_s, motion.samples[-1].time_s):
            raise ValueError("surface-guard event time and final saved observation disagree")
    elif any(e.kind == "surface-guard" for e in motion.events):
        raise ValueError("duration-complete history contains an impossible surface guard")
    return GroundedSphereWireTrace(motion, tuple(observations))
