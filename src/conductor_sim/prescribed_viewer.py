"""M19b: genuine M19a test-charge motion in the shared offline time viewer.

Fixed charge distributions are not conductors. A full trajectory is consumed
once; no browser-side integration or moving-source backreaction is introduced.
Ring/disc profiles are evaluated ONLY along their certified symmetry axis.
Non-axial cases plot E(t) and V(t) sampled at actual particle positions.
Neither case fabricates a 2D potential/field-line solution.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from math import fsum, hypot, isclose, isfinite, ulp

from .model import PointCharge
from .motion_1d import _NONRELATIVISTIC_SPEED_LIMIT_M_S
from .motion_viewer import _choose_indices
from .prescribed_fields import (
    UniformRingAxisField, UniformDiskAxisField, UniformRodField,
    FixedPointField, UniformShellField, UniformSphereField,
    UniformElectricField, _dot, _sub, _mul, _unit, _vector,
)
from .prescribed_motion import (
    TestChargeMotionSpec, TestChargeMotionResult, integrate_test_charge,
    _FixedDomain,
)
from .solver import SingularField, UnsupportedProblem
from .visualization import SliceSpec


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1],
            a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0])


def _norm(v):
    return hypot(*v)


def _close(a, b, *, rel=3e-9, absolute=1e-26):
    return isclose(a, b, rel_tol=rel, abs_tol=absolute)


def _point_close(a, b, scale):
    rounding = 32*max(ulp(abs(x)) for x in (*a, *b))
    return _norm(_sub(a,b)) <= 2e-9*scale+rounding


def _validate_trajectory(spec, result):
    """Reject incomplete, altered or physically inconsistent M19a saved rows.

    This is a source-observation check, not authentication of an arbitrary
    trajectory or independent reintegration. It runs before any HTML output.
    """
    if not isinstance(spec, TestChargeMotionSpec):
        raise TypeError('expected TestChargeMotionSpec')
    if not isinstance(result, TestChargeMotionResult) or result.spec != spec:
        raise ValueError('saved test-charge trajectory must match its exact authored specification')
    rows = result.samples
    if not 2 <= len(rows) <= 2001 or not all(isfinite(s.time_s) for s in rows):
        raise ValueError('test-charge trajectory requires 2 to 2001 finite time samples')
    if rows[0].time_s != 0 or any(a.time_s >= b.time_s for a,b in zip(rows,rows[1:])):
        raise ValueError('test-charge trajectory time samples must start at zero and strictly increase')
    if result.termination not in ('duration', 'source-guard'):
        raise ValueError('unsupported fixed-source termination')
    if len(result.events) != (0 if result.termination=='duration' else 1):
        raise ValueError('source-guard termination must match event inventory')
    if result.termination=='duration' and not _close(rows[-1].time_s, spec.duration_s, absolute=1e-14*spec.duration_s):
        raise ValueError('duration termination must end at authored elapsed time')
    if result.termination=='source-guard':
        ev=result.events[0]
        if (ev.kind != 'source-guard' or not _close(ev.time_s,rows[-1].time_s,
               absolute=1e-14*spec.duration_s) or
                not _point_close(ev.position_m,rows[-1].position_m,spec.length_scale_m)):
            raise ValueError('source-guard event must match the final sample')
    charge, mass = spec.particle.charge_c,spec.mass_kg
    p0, v0 = spec.particle.position_m,spec.initial_velocity_m_s
    scale=spec.length_scale_m
    if not _point_close(rows[0].position_m,p0,scale) or not _point_close(rows[0].velocity_m_s,v0,scale):
        raise ValueError('saved initial test-particle state disagrees with authored launch')
    domain=_FixedDomain(spec)
    potential0=spec.field.potential_at(p0)
    kinetic0=.5*mass*_norm(v0)**2
    max_drift=0.
    for row in rows:
        for key in ('position_m','displacement_m','velocity_m_s','acceleration_m_s2',
                    'electric_field_v_m','force_n'):
            value=getattr(row,key)
            if (not isinstance(value,(tuple,list)) or len(value)!=3 or
                    any(type(v) not in (int,float) or not isfinite(v) for v in value)):
                raise ValueError(f'nonfinite/malformed saved test-charge {key}')
        for key in ('potential_v','kinetic_energy_j','potential_energy_j',
                    'total_energy_j','energy_drift_j'):
            v=getattr(row,key)
            if type(v) not in (int,float) or not isfinite(v):
                raise ValueError(f'nonfinite/malformed saved test-charge {key}')
        if _norm(row.velocity_m_s)>=_NONRELATIVISTIC_SPEED_LIMIT_M_S:
            raise ValueError('saved test-charge speed violates the nonrelativistic model')
        if domain.clearance(row.position_m)<=0:
            raise ValueError('test-charge trajectory crosses a singularity or field-jump guard')
        if not _point_close(row.displacement_m,_sub(row.position_m,p0),scale):
            raise ValueError('saved test-charge displacement does not match position')
        if (abs(_dot(_sub(row.position_m,p0),spec.plane_normal))>1e-9*max(scale,_norm(_sub(row.position_m,p0))) or
                abs(_dot(row.velocity_m_s,spec.plane_normal))>max(1e-12,1e-9*_norm(row.velocity_m_s))):
            raise ValueError('saved test-charge position/velocity leaves the certified plane')
        electric=spec.field.electric_field_at(row.position_m)
        potential=spec.field.potential_at(row.position_m)
        if (any(not _close(a,b,absolute=1e-15) for a,b in zip(row.electric_field_v_m,electric)) or
                not _close(row.potential_v,potential,absolute=1e-15)):
            raise ValueError('saved electric field/potential disagrees with fixed sources')
        force=_mul(electric,charge)
        acceleration=_mul(force,1/mass)
        if (any(not _close(a,b,absolute=1e-27) for a,b in zip(force,row.force_n)) or
                any(not _close(a,b,absolute=1e-14) for a,b in zip(acceleration,row.acceleration_m_s2))):
            raise ValueError('test-charge saved force/acceleration violates F=qE')
        kinetic=.5*mass*_norm(row.velocity_m_s)**2
        interaction=charge*potential
        total=fsum((kinetic,interaction))
        drift=fsum((kinetic-kinetic0,interaction-charge*potential0))
        if (not _close(row.kinetic_energy_j,kinetic,absolute=1e-27) or
                not _close(row.potential_energy_j,interaction,absolute=1e-27) or
                not _close(row.total_energy_j,total,absolute=1e-27) or
                not _close(row.energy_drift_j,drift,absolute=2e-24)):
            raise ValueError('saved test-charge energy violates K+qV or numerical drift convention')
        max_drift=max(max_drift,abs(row.energy_drift_j))
    if not _close(result.max_abs_energy_drift_j,max_drift,absolute=1e-27):
        raise ValueError('saved test-charge energy-drift maximum disagrees with rows')


def test_charge_slice(spec, result, *, resolution=17, extent_m=None):
    """One static physical plane containing the complete stored trajectory."""
    if result.spec != spec:
        raise ValueError('display plane requires a matching motion result')
    axis=spec.axis
    normal=spec.plane_normal
    if axis is not None:
        v=axis
        origin=next(s.center_m for s in spec.field.sources if isinstance(s,(
            UniformRingAxisField,UniformDiskAxisField)))
    else:
        origin=spec.particle.position_m
        # A field direction sets a useful local vertical even when a charged
        # particle starts from rest. Otherwise the launch direction defines it.
        candidates=[s.field_v_m for s in spec.field.sources if isinstance(s,UniformElectricField)]
        candidates += [spec.initial_velocity_m_s]
        candidates += [_sub(r.position_m,origin) for r in result.samples]
        candidates += [_sub((s.position_m if isinstance(s,FixedPointField) else s.center_m),origin)
                       for s in spec.field.sources if isinstance(s,(FixedPointField,UniformSphereField,UniformShellField))]
        candidates += [_sub(s.end_m,s.start_m) for s in spec.field.sources if isinstance(s,UniformRodField)]
        direction=next((d for d in candidates if _norm(d)>1e-12*spec.length_scale_m),None)
        if direction is None:
            from .motion_2d import _perpendicular
            direction=_perpendicular(normal)
        direction=_sub(direction,_mul(normal,_dot(direction,normal)))
        if _norm(direction)<=1e-12*spec.length_scale_m:
            raise UnsupportedProblem('cannot construct a physical planar slice for the fixed sources')
        v=_unit(direction,'slice vertical')
    u=_unit(_cross(v,normal),'slice horizontal')
    def position(p):
        delta=_sub(p,origin)
        return _dot(delta,u),_dot(delta,v),_dot(delta,normal)
    points=list(s.position_m for s in result.samples)
    for s in spec.field.sources:
        if isinstance(s,FixedPointField):points.append(s.position_m)
        if isinstance(s,(UniformSphereField,UniformShellField,UniformRingAxisField,UniformDiskAxisField)):
            points.append(s.center_m)
        if isinstance(s,UniformRodField):points.extend((s.start_m,s.end_m))
    projected=[position(p) for p in points]
    source_extent=max([s.radius_m for s in spec.field.sources if isinstance(s,(
        UniformSphereField,UniformShellField,UniformRingAxisField,UniformDiskAxisField))]+[0.])
    lim=max([max(abs(a),abs(b)) for a,b,_ in projected]+[source_extent,.5*spec.length_scale_m])
    extent=1.35*lim if extent_m is None else extent_m
    if type(extent) not in (int,float) or not isfinite(extent) or extent<=0:
        raise ValueError('view extent must be finite and positive')
    plane=SliceSpec(origin,u,v,(-extent,extent),(-extent,extent),resolution)
    for row in result.samples:
        x,y,h=plane.project(row.position_m)
        if abs(h)>1e-9*max(spec.length_scale_m,_norm(_sub(row.position_m,origin))) or not plane.contains((x,y)):
            raise UnsupportedProblem('saved motion leaves the chosen physical view plane')
    return plane


def _axial_profile(spec, plane, *, count=201):
    if type(count) is not int or count<21 or count>501 or count%2!=1:
        raise ValueError('axial profile requires 21 to 501 odd samples')
    origin=plane.origin_m;axis=spec.axis
    if axis is None:
        return None
    radius=plane.y_range_m[1]
    profile=[]
    for j in range(count):
        z=-radius+2*radius*j/(count-1)
        p=tuple(origin[i]+axis[i]*z for i in range(3))
        try:
            potential=spec.field.potential_at(p)
        except SingularField:
            potential=None
        try:
            electric=_dot(spec.field.electric_field_at(p),axis)
        except SingularField:
            electric=None
        profile.append(dict(coordinate_m=z,potential_v=potential,field_parallel_v_m=electric))
    return profile


def _frame_placeholder(plane, row, sample_index, spec):
    # No 2D potential arrays are fabricated for axis-only sources. For other
    # sources M19b shows the source diagram plus exact trajectory observations;
    # a later milestone can add independently validated spatial field maps.
    grid={'x_m':[plane.x_range_m[0],plane.x_range_m[1]],
          'y_m':[plane.y_range_m[0],plane.y_range_m[1]],
          'potential_v':[[None,None],[None,None]],
          'electric_field_v_per_m':[[None,None],[None,None]],
          'region':[['unsampled','unsampled'],['unsampled','unsampled']]}
    return dict(label=f'Elapsed time {row.time_s:.5g} s',
        backend='Analytical prescribed field',plane=asdict(plane),grid=grid,
        contours=[],field_lines=[],line_termination_counts={},profiles=[],
        potential_interpolation={'method':'no-spatial-map','cells':[[False]],'boundary':[]},
        potential_refinements=[],geometry=[],
        charges=[dict(charge_c=spec.particle.charge_c, position_m=list(row.position_m),
                      slice_position_m=list(plane.project(row.position_m)),motion_role='test particle')],
        conductor_results=[],source_masks=[],domain_radius_m=None,domain_cylinder=None,
        display_clip_v=1.,notes=['Prescribed nonresponding sources; test particle is NOT included in field sources.',
         'No off-axis source fields are extrapolated. Spatial scene is an authored geometry diagram, not a potential map.'],
        density_method='Not applicable: no conductor or induced surface density',
        conductor_results_method='Not applicable: prescribed field has no conductors',
        field_line_pool={'kind':'not-sampled','available':0,'seed_budget':0},
        domain_center_m=plane.origin_m,motion_sample_index=sample_index)


def sample_test_charge_viewer(spec:TestChargeMotionSpec, *, result=None, visual_frames=16,
                              resolution=17, profile_samples=201, extent_m=None):
    """Export original numerical time histories and certified-source profiles.

    Only the shared M19a integrator can produce the trajectory. No additional
    force law, browser integrator, sampled off-axis ring/disc field or
    moving-test-charge backreaction is introduced.
    """
    if not isinstance(spec,TestChargeMotionSpec):
        raise TypeError('expected TestChargeMotionSpec')
    if result is None:
        result=integrate_test_charge(spec)
    _validate_trajectory(spec,result)
    indices=_choose_indices(result,visual_frames)
    plane=test_charge_slice(spec,result,resolution=resolution,extent_m=extent_m)
    profile=_axial_profile(spec,plane,count=profile_samples)
    domain=_FixedDomain(spec)
    breaks=[i for i,(a,b) in enumerate(zip(result.samples,result.samples[1:]),start=1)
            if domain.charge_segment_clearance(a.position_m,b.position_m)<=0]
    physics=json.loads(json.dumps(result.as_dict(),allow_nan=False))
    for row in physics['samples']:
        # Compatibility adapter for the shared K/U/K+U graph. Both values are
        # full qV; render_html checks they agree exactly, not inferred work.
        row['interaction_energy_j']=row['potential_energy_j']
    physics.update(schema='conductor_sim.test_charge_viewer.v1',
                   frame_sample_indices=list(indices),graph_sample_count=len(result.samples),
                   field_frame_count=len(indices),trajectory_break_indices=breaks,
                   profile_mode='axis' if spec.axis is not None else 'elapsed-time',
                   profile_samples=profile,
                   display_note='Fixed sources do not respond to the test charge; '
                    'the geometry scene is a schematic slice, not a sampled field map.')
    frames=[_frame_placeholder(plane,result.samples[i],i,spec) for i in indices]
    from .potential_display import potential_colour_scale
    bundle=dict(schema='conductor_sim.visualization.v1',
                units=dict(length='m',potential='V',field='V/m',density='not applicable',
                           time='s',velocity='m/s',energy='J'),
                playback='Real M19a elapsed-time samples; no browser ODE, no '
                         'moving-charge induction, no invented off-axis spatial field.',
                frames=frames,prescribed_motion=physics,
                potential_colour_scale=potential_colour_scale(frames))
    # Browser metadata is canonical JSON-native data, never tuples that
    # differ between Python and its saved HTML/JSON copies.
    bundle=json.loads(json.dumps(bundle,allow_nan=False))
    from .prescribed_viewer_validation import validate_prescribed_viewer
    validate_prescribed_viewer(bundle)
    return bundle
