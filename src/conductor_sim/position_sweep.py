"""M13b: solver-independent position sweeps with force/energy observations.

An authored charge is tracked by its *component index* in the original model,
not by the sorted grouped-source index returned by an analytical solution.
One physical point charge moves on one straight path; charge inventories,
conductor constraints, geometry and potential gauge remain fixed.
No time integration, browser-time solve, or implicit FEMM analytical fallback.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import hypot, isfinite

from ._resources import cleanup_resources
from .geometry import contains_metal
from .model import (ConductingPlane, ElectrostaticProblem, PointCharge,
                    Vector3, _position)
from .observables import force_energy
from .solver import UnsupportedProblem
from .visualization import (SliceSpec, dot, sample_frame, sub, unit)
from .visual_adapters import context_for_solution


def _add(a, b, scale=1.):
    return tuple(a[i] + scale * b[i] for i in range(3))


def _line_distance(a, b, point):
    ab = sub(b, a)
    length2 = dot(ab, ab)
    if length2 == 0:
        return hypot(*sub(point, a))
    t = max(0., min(1., dot(sub(point, a), ab) / length2))
    return hypot(*sub(point, _add(a, ab, t)))


def _transverse(v, normal):
    return sub(v, tuple(dot(v, normal)*x for x in normal))


def _perpendicular(v):
    ref = (1., 0., 0.) if abs(v[0]) < .85 else (0., 1., 0.)
    return unit(_transverse(ref, v))


@dataclass(frozen=True, slots=True)
class PositionSweepSpec:
    """Track authored `charge_index` from its initial position to `end_m`.

    `steps` counts stored equilibria and is limited by the existing viewer.
    Source identity never silently changes when sorted source order changes.
    """
    problem: ElectrostaticProblem
    charge_index: int
    end_m: Vector3
    steps: int = 11

    def __post_init__(self):
        if not isinstance(self.problem, ElectrostaticProblem):
            raise TypeError('problem must be ElectrostaticProblem')
        if isinstance(self.charge_index, bool) or not isinstance(self.charge_index, int):
            raise TypeError('charge_index must be an authored integer index')
        if not 0 <= self.charge_index < len(self.problem.charges):
            raise IndexError('selected authored charge index is out of bounds')
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or not 3 <= self.steps <= 24:
            raise ValueError('steps must be an integer between 3 and 24')
        end = _position(self.end_m, 'sweep end')
        object.__setattr__(self, 'end_m', end)
        if self.selected_charge.charge_c == 0:
            raise UnsupportedProblem('a zero-charge component cannot be tracked for force/energy')
        start = self.start_m
        length = hypot(*sub(end, start))
        if not isfinite(length) or length <= 0:
            raise ValueError('sweep endpoints must be distinct and finite')
        # Distinct authored components cannot be assigned separate point
        # mechanical energies at coincident sites. Reject even if they cancel.
        for i, other in enumerate(self.problem.charges):
            if i == self.charge_index:
                continue
            if _line_distance(start, end, other.position_m) <= 1e-12*max(length, 1.):
                raise UnsupportedProblem('moving source path intersects another authored point charge')
        for conductor in self.problem.conductors:
            if isinstance(conductor, ConductingPlane):
                if (conductor.signed_distance_m(start) <= 0 or
                        conductor.signed_distance_m(end) <= 0):
                    raise UnsupportedProblem('source path must remain strictly in the plate vacuum')
            elif getattr(conductor, 'inner_radius_m', 0.) == 0:
                if _line_distance(start, end, conductor.center_m) <= conductor.outer_radius_m:
                    raise UnsupportedProblem('moving source path intersects the solid conductor')
            else:
                # Nested conductor/cavity geometries are not supported by the
                # M13a force API; do not imply safe trajectories through them.
                raise UnsupportedProblem('M13b does not support shell/cavity force sweeps')

    @property
    def selected_charge(self):
        return self.problem.charges[self.charge_index]

    @property
    def start_m(self):
        return self.selected_charge.position_m

    @property
    def length_m(self):
        return hypot(*sub(self.end_m, self.start_m))

    @property
    def direction(self):
        return unit(sub(self.end_m, self.start_m))

    def states(self):
        """Deterministically authored models; nothing mutates the input."""
        delta = sub(self.end_m, self.start_m)
        for index in range(self.steps):
            position = _add(self.start_m, delta, index/(self.steps-1))
            charges = list(self.problem.charges)
            charges[self.charge_index] = PointCharge(self.selected_charge.charge_c, position)
            yield (index*self.length_m/(self.steps-1),
                   ElectrostaticProblem(tuple(charges), self.problem.conductors,
                                        self.problem.permittivity_f_per_m))


def sweep_slice(spec: PositionSweepSpec, *, resolution=25, extent_m=None) -> SliceSpec:
    """Fixed 2D slice containing the conductor anchor and the entire path."""
    conductor = spec.problem.conductors[0] if spec.problem.conductors else None
    if conductor is None:
        raise UnsupportedProblem('a conductor is required for the M13b viewer')
    start, end = spec.start_m, spec.end_m
    if isinstance(conductor, ConductingPlane):
        v = conductor.normal
        displacement = _transverse(sub(end, start), v)
        if hypot(*displacement) <= 1e-12*max(1., spec.length_m):
            displacement = _transverse(sub(start, conductor.point_m), v)
        u = unit(displacement) if hypot(*displacement) else _perpendicular(v)
        middle = tuple((start[i]+end[i])/2 for i in range(3))
        height = conductor.signed_distance_m(middle)
        origin = sub(middle, tuple(height*x for x in v))
    else:
        origin = conductor.center_m
        v = unit(sub(start, origin))
        lateral = _transverse(sub(end, origin), v)
        u = unit(lateral) if hypot(*lateral)>1e-12 else _perpendicular(v)
    anchors = [start, end]
    for charge in spec.problem.charges:
        anchors.append(charge.position_m)
    if isinstance(conductor, ConductingPlane):
        anchors.extend(c.point_m for c in spec.problem.conductors
                       if isinstance(c, ConductingPlane))
    reach = max((hypot(*sub(p, origin)) for p in anchors),default=.1)
    if not isinstance(conductor, ConductingPlane):
        reach=max(reach, conductor.outer_radius_m)
    extent = 1.40*max(reach, spec.length_m*.5) if extent_m is None else extent_m
    if not isfinite(extent) or extent<=0:
        raise ValueError('sweep slice extent must be finite and positive')
    plane=SliceSpec(origin,u,v,(-extent,extent),(-extent,extent),resolution)
    for point in (start,end):
        x,y,h=plane.project(point)
        if abs(h)>1e-9*max(1.,spec.length_m) or not plane.contains((x,y)):
            raise UnsupportedProblem('the selected path must lie within the fixed display slice')
    return plane


def sample_position_sweep(spec: PositionSweepSpec, backend, *, plane=None,
                          resolution=25, line_steps=110, seeds=None, profile_samples=45):
    """Build precomputed viewer frames, forces, and mechanical energy plots.

    Solve each state exactly once; force_energy observes the same solution as
    the field-map sampler. Always close any owned solver session. Numerical
    solutions lacking an explicit force provider fail before any field samples.
    """
    if not isinstance(spec, PositionSweepSpec):
        raise TypeError('spec must be PositionSweepSpec')
    if plane is None:
        plane = sweep_slice(spec,resolution=resolution)
    elif not isinstance(plane, SliceSpec):
        raise TypeError('plane must be a SliceSpec')
    for pos in (spec.start_m,spec.end_m):
        x,y,h=plane.project(pos)
        if abs(h)>1e-9*max(spec.length_m,1.) or not plane.contains((x,y)):
            raise UnsupportedProblem('the entire selected path must lie in the fixed slice')
    if seeds is not None:
        seeds=tuple(seeds)
        if len(seeds)>96:
            raise ValueError('at most 96 authored field-line seeds')
    frames=[]
    series=[]
    reference=None
    direction=spec.direction
    for i,(distance,problem) in enumerate(spec.states()):
        supports=getattr(backend,'supports',None)
        if callable(supports) and not supports(problem):
            raise UnsupportedProblem(f'backend does not support sweep state {i}')
        solution=backend.solve(problem)
        try:
            observed=force_energy(solution)
            selected=problem.charges[spec.charge_index]
            matching=[(j,c) for j,c in enumerate(observed.charges)
                      if c.source.position_m == selected.position_m]
            if len(matching)!=1 or matching[0][1].source.charge_c!=selected.charge_c:
                raise UnsupportedProblem('selected authored charge identity is ambiguous or canceled')
            grouped_index,charge=matching[0]
            force=charge.total_force_n
            f_parallel=dot(force,direction)
            total=observed.energy.total_j
            if reference is None:
                reference=total
            work=total-reference
            frame=sample_frame(context_for_solution(problem,solution),plane,
                               label=f'Charge position {distance:.4f} m along path',
                               seeds=seeds,line_steps=line_steps,profile_samples=profile_samples)
            record=dict(index=i,coordinate_m=distance,
                        position_m=list(selected.position_m),charge_c=selected.charge_c,
                        grouped_index=grouped_index,force_n=list(force),
                        force_parallel_n=f_parallel,
                        direct_force_n=list(charge.direct_force_n),
                        induced_force_n=list(charge.induced_force_n),
                        background_force_n=list(charge.background_force_n),
                        energy_j=total,energy_direct_j=observed.energy.direct_j,
                        energy_induced_j=observed.energy.induced_j,
                        energy_background_j=observed.energy.background_j,
                        external_quasistatic_work_j=work,
                        method=observed.method,energy_reference=observed.energy.reference)
            frame['force_energy']=record
            frames.append(frame)
            series.append(record)
        except BaseException as exc:
            close=getattr(solution,'close',None)
            if callable(close):
                cleanup_resources((('position-sweep solution',close),),primary=exc)
            raise
        else:
            close=getattr(solution,'close',None)
            if callable(close):
                close()
    from .potential_display import potential_colour_scale
    from .visualization import _finite
    bundle={'schema':'conductor_sim.visualization.v1',
            'units':{'length':'m','potential':'V','field':'V/m','density':'C/m^2',
                     'force':'N','mechanical_interaction_energy':'J','work':'J'},
            'playback':'Precomputed charge positions, not elapsed time or simulated motion',
            'color_limit_v':max(f['display_clip_v'] for f in frames),
            'potential_colour_scale':potential_colour_scale(frames),
            'frames':frames,
            'position_sweep':{
                'schema':'conductor_sim.position_sweep.v1',
                'authored_charge_index':spec.charge_index,
                'charge_c':spec.selected_charge.charge_c,
                'path_start_m':list(spec.start_m), 'path_end_m':list(spec.end_m),
                'path_length_m':spec.length_m,
                'direction':list(spec.direction),
                'reference_frame_index':0,
                'energy_label':'Mechanical interaction energy (J)',
                'force_label':'Force component along path (N)',
                'work_label':'External quasistatic work (J), relative to starting position',
                'conditions':'One charge moved quasistatically; all charge strengths, '
                             'other positions, conductor geometry, constraints, and voltage gauge fixed.',
                'samples':series}}
    import json
    json.dumps(bundle,allow_nan=False)
    return bundle
