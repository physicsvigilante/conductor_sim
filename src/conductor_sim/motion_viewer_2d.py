"""M17b: actual M17a curved trajectories in the existing offline viewer.

All physics comes from integrate_motion_2d and the accepted analytical
force_energy providers. Only selected genuine time samples receive solved
electrostatic field snapshots; the complete trajectory is retained for graphs
and the in-slice path overlay. Never interpolate or invent field solutions.
"""
from __future__ import annotations

from dataclasses import replace
from math import fsum, hypot, isclose, isfinite

from ._resources import cleanup_resources
from .model import ConductingPlane, PointCharge
from .motion_2d import Motion2DResult, Motion2DSpec, integrate_motion_2d
from .motion_viewer import _choose_indices
from .observables import force_energy
from .solver import UnsupportedProblem
from .visualization import SliceSpec, sample_sweep


def _dot(a, b):
    return fsum(x*y for x, y in zip(a, b))


def _sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def _norm(v):
    return hypot(*v)


def _unit(v):
    n = _norm(v)
    if n == 0 or not isfinite(n):
        raise ValueError('slice direction must be finite and nonzero')
    return tuple(x/n for x in v)


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _slice_contains_trajectory(plane, trajectory, *, scale):
    normal = plane.normal
    if abs(_dot(normal, trajectory.plane_normal)) < 1-1e-8:
        raise UnsupportedProblem('display slice must be parallel to the proven invariant motion plane')
    for row in trajectory.samples:
        x, y, h = plane.project(row.position_m)
        if abs(h) > 1e-8*max(scale, _norm(_sub(row.position_m, trajectory.plane_origin_m)), 1e-12):
            raise UnsupportedProblem('motion trajectory is not entirely in the authored display slice')
        if not plane.contains((x, y)):
            raise UnsupportedProblem('motion display slice must include the complete trajectory')


def motion_slice_2d(spec: Motion2DSpec, trajectory: Motion2DResult, *, resolution=25) -> SliceSpec:
    """Fixed physical symmetry slice containing every stored particle location.

    For infinite planes, ignore arbitrary tangential conductor anchors;
    for radial geometry retain all nested conductor extents. The axes
    reflect the initial tangential launch and the conductor normal/radius.
    """
    if not isinstance(spec, Motion2DSpec) or not isinstance(trajectory, Motion2DResult):
        raise TypeError('expected Motion2DSpec and Motion2DResult')
    if trajectory.spec != spec or len(trajectory.samples) < 2:
        raise ValueError('motion slice requires the matching nonempty M17a trajectory')
    conductors = spec.problem.conductors
    initial = spec.selected_charge.position_m
    normal = trajectory.plane_normal
    if isinstance(conductors[0], ConductingPlane):
        v = conductors[0].normal
        height = conductors[0].signed_distance_m(initial)
        origin = tuple(initial[i]-height*v[i] for i in range(3))
        tangent = _sub(spec.initial_velocity_m_s,
                       tuple(_dot(spec.initial_velocity_m_s, v)*x for x in v))
        u = _unit(tangent if _norm(tangent) else _cross(v, normal))
        # Physical intercepts of the authored plane boundaries through the
        # chosen reference, not their arbitrary position along the surface.
        boundary = [-c.signed_distance_m(origin)/_dot(c.normal, v) for c in conductors]
        points = [(_dot(_sub(row.position_m, origin), u),
                   _dot(_sub(row.position_m, origin), v)) for row in trajectory.samples]
        scale = max([abs(b) for b in boundary] + [abs(height), 1e-6])
        xs = [x for x,_ in points]
        ys = [y for _,y in points] + boundary
        xspan = max(xs)-min(xs)
        yspan = max(ys)-min(ys)
        xpad = max(.30*scale, .16*xspan)
        ypad = max(.24*scale, .15*yspan)
        xrange = (min(xs)-xpad, max(xs)+xpad)
        yrange = (min(ys)-ypad, max(ys)+ypad)
    else:
        origin = conductors[0].center_m
        radial = _sub(initial, origin)
        direction = radial if _norm(radial) else spec.initial_velocity_m_s
        if not _norm(direction):
            # The stationary central source has no unique physical azimuth;
            # select a reproducible meridian, still within its recorded plane.
            direction = _cross(normal, (1., 0., 0.))
            if not _norm(direction):
                direction = _cross(normal, (0., 1., 0.))
        v = _unit(direction)
        tangent = _sub(spec.initial_velocity_m_s,
                       tuple(_dot(spec.initial_velocity_m_s, v)*x for x in v))
        u = _unit(tangent if _norm(tangent) else _cross(v, normal))
        reach = max(_norm(_sub(row.position_m, origin)) for row in trajectory.samples)
        outer = max(c.outer_radius_m for c in conductors)
        span = 1.24*max(reach, outer, 1e-6)
        xrange = (-span, span)
        yrange = (-span, span)
        scale = span
    plane = SliceSpec(origin, u, v, xrange, yrange, resolution)
    _slice_contains_trajectory(plane, trajectory, scale=scale)
    return plane


def sample_motion_2d(spec: Motion2DSpec, backend, *, result=None, plane=None,
                     resolution=25, line_steps=90, seeds=None, profile_samples=35,
                     visual_frames=24):
    """Full real M17a trajectory + at most 24 matching solved field snapshots.

    A custom display plane is permitted only if it contains the complete
    physical trajectory; the renderer cannot project a 3D path onto an
    unrelated slice and pretend the force vectors were planar.
    """
    if not isinstance(spec, Motion2DSpec):
        raise TypeError('expected Motion2DSpec')
    if result is None:
        result = integrate_motion_2d(spec, backend)
    if not isinstance(result, Motion2DResult) or result.spec != spec:
        raise ValueError('supplied trajectory must be an M17a result for this exact specification')
    indices = _choose_indices(result, visual_frames)
    if plane is None:
        plane = motion_slice_2d(spec, result, resolution=resolution)
    elif not isinstance(plane, SliceSpec):
        raise TypeError('display plane must be a SliceSpec')
    _slice_contains_trajectory(plane, result, scale=max(
        1e-6, max(_norm(_sub(s.position_m, result.plane_origin_m)) for s in result.samples)))

    selected = [result.samples[i] for i in indices]
    models = []
    for row in selected:
        source = PointCharge(spec.selected_charge.charge_c, row.position_m)
        models.append((f'Elapsed time {row.time_s:.5g} s',
                       replace(spec.problem, charges=(source,))))

    class _CheckedBackend:
        """No FEMM substitution and no matching a field to the wrong time."""
        next_index = 0

        def supports(self, problem):
            fn = getattr(backend, 'supports', None)
            return fn(problem) if callable(fn) else True

        def solve(self, problem):
            if self.next_index >= len(selected):
                raise ValueError('unexpected extra field-frame solve')
            expected = selected[self.next_index]
            if (len(problem.charges) != 1 or
                    problem.charges[0].position_m != expected.position_m or
                    problem.charges[0].charge_c != spec.selected_charge.charge_c):
                raise UnsupportedProblem('field frame must match a genuine ordered M17a time sample')
            solution = backend.solve(problem)
            try:
                readings = force_energy(solution)  # unsupported FEMM must refuse
                if (len(readings.charges) != 1 or
                        readings.charges[0].source != problem.charges[0]):
                    raise UnsupportedProblem('field-frame charge identity does not match motion')
                actual = readings.charges[0].total_force_n
                if (any(not isclose(a, b, rel_tol=2e-10, abs_tol=1e-28)
                        for a,b in zip(actual, expected.force_n)) or
                        not isclose(readings.energy.total_j, expected.interaction_energy_j,
                                    rel_tol=2e-10, abs_tol=1e-26)):
                    raise UnsupportedProblem('field-frame electrostatic force/energy disagrees with integrated trajectory')
            except BaseException as error:
                close = getattr(solution, 'close', None)
                if callable(close):
                    cleanup_resources((('M17b field-frame solution', close),), primary=error)
                raise
            self.next_index += 1
            return solution

    checked = _CheckedBackend()
    bundle = sample_sweep(models, checked, plane=plane, resolution=resolution,
                          line_steps=line_steps, seeds=seeds,
                          profile_samples=profile_samples)
    if checked.next_index != len(selected):
        raise ValueError('field-frame solution count differs from actual motion samples')
    physics = result.as_dict()
    physics.update(schema='conductor_sim.motion_viewer_2d.v1',
                   frame_sample_indices=list(indices),
                   graph_sample_count=len(result.samples),
                   field_frame_count=len(indices),
                   fields='Each field frame solved at the matching actual numerical position/time; no interpolation between field frames.')
    for frame, index in zip(bundle['frames'], indices):
        frame['motion_sample_index'] = index
    bundle['motion_2d'] = physics
    bundle['units'].update(time='s', velocity='m/s', acceleration='m/s^2',
                           kinetic_energy='J', mechanical_interaction_energy='J')
    bundle['playback'] = ('Genuine elapsed-time curved trajectory integrated by M17a; '
                          'offline browser displays matching precomputed electrostatic fields '
                          'at selected actual time samples. No browser ODE, artificial path '
                          'constraint or conductor collision model.')
    return bundle
