"""M16b: saved M16a elapsed-time trajectory -> existing offline electrostatic viewer.

Numerical dynamics is computed *only* by integrate_motion_1d. We solve the
static electrostatic field at each selected **actual** output sample using the
same analytical backend. No browser-side ODE, interpolated field solutions,
static-position-sweep masquerading as time, or FEMM force fallback is allowed.
"""
from __future__ import annotations

from dataclasses import replace
from math import hypot, isclose, isfinite

from ._resources import cleanup_resources
from .model import ConductingPlane, ElectrostaticProblem, PointCharge
from .motion_1d import Motion1DResult, Motion1DSpec, integrate_motion_1d
from .observables import force_energy
from .solver import UnsupportedProblem
from .visualization import SliceSpec, sample_sweep


def _unit(v):
    length = hypot(*v)
    if not length or not isfinite(length):
        raise ValueError('display direction must be finite and nonzero')
    return tuple(q / length for q in v)


def _perpendicular(v):
    ref = (1., 0., 0.) if abs(v[0]) < .85 else (0., 1., 0.)
    dot = sum(ref[i] * v[i] for i in range(3))
    return _unit(tuple(ref[i] - dot * v[i] for i in range(3)))


def motion_slice(spec: Motion1DSpec, trajectory: Motion1DResult, *, resolution=25,
                 extent_m=None) -> SliceSpec:
    """Fixed meridian including entire trajectory and *all* conductors.

    Never use a tangential coordinate of an infinite-plane's authored anchor
    to choose the view extent. The axis has already been proved physically
    invariant by M16a; no new line constraints are introduced here.
    """
    if trajectory.spec != spec or not trajectory.samples:
        raise ValueError('motion slice requires the matching nonempty M16a result')
    axis = spec.axis
    conductors = spec.problem.conductors
    if isinstance(conductors[0], ConductingPlane):
        p = spec.selected_charge.position_m
        c = conductors[0]
        height = c.signed_distance_m(p)
        origin = tuple(p[i] - height * c.normal[i] for i in range(3))
        extra = [abs(other.signed_distance_m(origin)) for other in conductors]
    else:
        origin = conductors[0].center_m
        extra = [hypot(*(c.center_m[i] - origin[i] for i in range(3))) + c.outer_radius_m
                 for c in conductors]
    reach = max(extra + [hypot(*(s.position_m[i] - origin[i] for i in range(3)))
                         for s in trajectory.samples])
    if extent_m is None:
        extent_m = 1.30 * max(reach, 1e-6)
    if not isfinite(extent_m) or extent_m <= 0:
        raise ValueError('display extent must be finite and positive')
    plane = SliceSpec(origin, _perpendicular(axis), axis,
                      (-extent_m, extent_m), (-extent_m, extent_m), resolution)
    for sample in trajectory.samples:
        x, y, h = plane.project(sample.position_m)
        if abs(h) > 1e-9 * max(1., reach) or not plane.contains((x, y)):
            raise UnsupportedProblem('motion display plane must contain the entire trajectory')
    return plane


def _choose_indices(result: Motion1DResult, cap: int) -> tuple[int, ...]:
    """Bound visual field-map solves while preserving full physics time data."""
    if isinstance(cap, bool) or not isinstance(cap, int) or not 2 <= cap <= 24:
        raise ValueError('visual frame cap must be an integer between 2 and 24')
    n = len(result.samples)
    if n < 2:
        raise ValueError('motion visualization requires at least two actual time samples')
    if n <= cap:
        return tuple(range(n))
    # Uniform time snapshots first: a greedy scan from t=0 can otherwise
    # accidentally spend the entire budget at the *beginning* of playback.
    chosen = {round(i*(n-1)/(cap-1)) for i in range(cap)}
    assert len(chosen) == cap  # n > cap guarantees at least one index spacing.
    times = [s.time_s for s in result.samples]
    for event in result.events:
        nearest = min(range(1, n - 1), key=lambda i: abs(times[i] - event.time_s))
        if nearest in chosen:
            continue
        # Retain endpoints, replace the closest interior frame to ensure the
        # actual turning/guard event is visible without changing frame count.
        movable = sorted(chosen - {0, n - 1}, key=lambda i: abs(times[i]-times[nearest]))
        if movable:
            chosen.remove(movable[0]);chosen.add(nearest)
    return tuple(sorted(chosen))


def sample_motion_1d(spec: Motion1DSpec, backend, *, result=None, plane=None,
                     resolution=25, line_steps=90, seeds=None, profile_samples=35,
                     visual_frames=24):
    """Combine real time-series physics with matching precomputed field frames.

    Every physical M16a time sample (up to 2,001) remains in the graph data;
    at most `visual_frames` selected times get expensive electrostatic field
    rasters. Browser playback shows only those stored, solved snapshots.
    """
    if not isinstance(spec, Motion1DSpec):
        raise TypeError('expected Motion1DSpec')
    if result is None:
        result = integrate_motion_1d(spec, backend)
    if not isinstance(result, Motion1DResult) or result.spec != spec:
        raise ValueError('supplied trajectory must be an M16a result for this exact specification')
    indices = _choose_indices(result, visual_frames)
    if plane is None:
        plane = motion_slice(spec, result, resolution=resolution)
    elif not isinstance(plane, SliceSpec):
        raise TypeError('display plane must be a SliceSpec')
    for row in result.samples:
        x, y, h = plane.project(row.position_m)
        if abs(h) > 1e-9 * max(1., hypot(*row.position_m)) or not plane.contains((x, y)):
            raise UnsupportedProblem('motion trajectory is not entirely in the authored display slice')

    class _CheckedBackend:
        """Prevent attaching an analytical M16a trajectory to a different field solver."""
        def supports(self, problem):
            fn = getattr(backend, 'supports', None)
            return fn(problem) if callable(fn) else True

        def solve(self, problem):
            solution = backend.solve(problem)
            try:
                data = force_energy(solution)  # FEMM does NOT expose this and must refuse.
                if len(data.charges) != 1:
                    raise UnsupportedProblem('motion frame must have exactly one active source')
                selected = data.charges[0]
                row = expected_by_position[tuple(problem.charges[0].position_m)]
                force = sum(selected.total_force_n[i] * spec.axis[i] for i in range(3))
                if (not isclose(force, row.force_n, rel_tol=2e-10, abs_tol=1e-28) or
                        not isclose(data.energy.total_j, row.interaction_energy_j,
                                    rel_tol=2e-10, abs_tol=1e-26)):
                    raise UnsupportedProblem('frame field solution disagrees with integrated trajectory')
            except BaseException as exc:
                close = getattr(solution, 'close', None)
                if callable(close):
                    cleanup_resources((('time-motion field solution', close),), primary=exc)
                raise
            return solution

    selected = [result.samples[i] for i in indices]
    expected_by_position = {tuple(row.position_m): row for row in selected}
    # Same scalar position can recur at different times (e.g., a turning
    # trajectory). The electrostatic force/energy must then be identical.
    models = []
    for row in selected:
        charges = list(spec.problem.charges)
        charges[spec.charge_index] = PointCharge(spec.selected_charge.charge_c, row.position_m)
        problem = replace(spec.problem, charges=tuple(charges))
        models.append((f'Elapsed time {row.time_s:.5g} s', problem))
    bundle = sample_sweep(models, _CheckedBackend(), plane=plane,
                          resolution=resolution, line_steps=line_steps,
                          seeds=seeds, profile_samples=profile_samples)
    physics = result.as_dict()
    physics.update(schema='conductor_sim.motion_viewer.v1',
                   frame_sample_indices=list(indices),
                   field_frame_count=len(indices),
                   graph_sample_count=len(result.samples),
                   fields='Each displayed electrostatic field is solved at the matching actual numerical trajectory time; no between-frame field interpolation.')
    for frame, index in zip(bundle['frames'], indices):
        frame['motion_sample_index'] = index
    bundle['motion_1d'] = physics
    bundle['units'].update(time='s', velocity='m/s', acceleration='m/s^2', kinetic_energy='J',
                           mechanical_interaction_energy='J')
    bundle['playback'] = ('Actual elapsed-time trajectory integrated by M16a; offline browser selects '
                          'precomputed physical-time samples and static matching electrostatic fields; '
                          'no browser ODE and no exact point-charge surface impact')
    return bundle
