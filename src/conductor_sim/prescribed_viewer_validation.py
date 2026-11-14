"""Strict saved-data validation for M19b's shared offline renderer.

Ensures samples, fixed-source definitions, energy convention and every display
frame are consistent. It does not authenticate an integration history.
"""
from __future__ import annotations

import json
import re
from math import fsum, hypot, isclose, isfinite

from .motion_1d import _NONRELATIVISTIC_SPEED_LIMIT_M_S
from .prescribed_fields import (PrescribedField, SOURCE_TYPES, _dot, _sub)
from .solver import SingularField
from .visualization import SliceSpec
from .conservative_forces import (restore_gravity, ELECTRIC_ENERGY, COMPOSED_ENERGY,
    validate_composed_sample, scalar_matches, CONTRIBUTION_KEYS)


def _number(x):
    return type(x) in (float,int) and isfinite(x)


def _vector(v):
    return isinstance(v,(list,tuple)) and len(v)==3 and all(_number(x) for x in v)


def _same(a,b,absolute=2e-24):
    return isclose(a,b,rel_tol=3e-9,abs_tol=absolute)


def _position_same(a,b,scale):
    return hypot(*_sub(a,b))<=max(2e-9*scale,1e-13)


def _restore_field(record):
    if (not isinstance(record,dict) or record.get('schema')!='conductor_sim.prescribed_field.v1'
            or not _number(record.get('permittivity_f_m')) or record['permittivity_f_m']<=0
            or not isinstance(record.get('sources'),list)):
        raise ValueError('invalid prescribed fixed-source inventory')
    types={type_.__name__:type_ for type_ in SOURCE_TYPES}
    sources=[]
    for item in record['sources']:
        if not isinstance(item,dict) or item.get('kind') not in types:
            raise ValueError('unsupported saved fixed-source geometry')
        source=dict(item)
        kind=source.pop('kind')
        try:
            restored=types[kind](**source)
        except (TypeError,ValueError) as exc:
            raise ValueError('invalid saved fixed-source geometry') from exc
        if kind in ('UniformRingAxisField','UniformDiskAxisField'):
            # A normalized tilted axis can have norm 1 - 1 ulp. Running the
            # source constructor *again* renormalizes it and changes the last
            # float bit of E/V, including the tiny saved energy drift. Only
            # the verifier's private reconstructed copy is adjusted, after
            # proving the serialized axis is unit and agrees with the freshly
            # validated constructor to roundoff precision. Production fields
            # and the frozen authored source object are never changed.
            axis=source.get('axis')
            if (not _vector(axis) or
                    abs(hypot(*axis)-1.)>2e-15 or
                    any(abs(a-b)>2e-15 for a,b in zip(axis,restored.axis))):
                raise ValueError('saved fixed-source axis is not a valid normalized orientation')
            object.__setattr__(restored,'axis',tuple(axis))
        sources.append(restored)
    field=PrescribedField(tuple(sources),record['permittivity_f_m'])
    if json.loads(json.dumps(field.as_dict(),allow_nan=False)) != record:
        raise ValueError('saved fixed-source inventory has altered or extraneous fields')
    return field


def validate_prescribed_viewer(bundle):
    """Validate all public M19b data and its actual-frame identity."""
    from .prescribed_motion import _FixedDomain, TestChargeMotionSpec
    from .model import PointCharge
    if not isinstance(bundle,dict) or bundle.get('schema')!='conductor_sim.visualization.v1':
        raise ValueError('expected existing shared visualization bundle')
    if any(key in bundle for key in ('motion_1d','motion_2d','position_sweep')):
        raise ValueError('prescribed test-charge playback cannot also be conductor motion or a position sweep')
    m=bundle.get('prescribed_motion');frames=bundle.get('frames')
    if not isinstance(m,dict) or m.get('schema') not in (
            'conductor_sim.test_charge_viewer.v1','conductor_sim.test_charge_viewer.v2'):
        raise ValueError('expected M19b fixed-source motion metadata')
    if not isinstance(frames,list) or not 2<=len(frames)<=24:
        raise ValueError('M19b requires 2 to 24 real-time display frames')
    composed = m['schema']=='conductor_sim.test_charge_viewer.v2'
    if not composed and 'gravity' in m:
        raise ValueError('legacy electric-only metadata cannot carry undeclared gravity')
    gravity=restore_gravity(m.get('gravity')) if composed else None
    field=_restore_field(m.get('field'))
    particle=m.get('particle')
    if (not isinstance(particle,dict) or not _number(particle.get('charge_c')) or
            particle['charge_c']==0 or not _vector(particle.get('position_m'))):
        raise ValueError('saved test-particle identity is invalid')
    for name in ('mass_kg','duration_s','sample_interval_s','source_guard_m','length_scale_m'):
        if not _number(m.get(name)) or m[name]<=0:
            raise ValueError(f'bad prescribed motion {name}')
    if not _vector(m.get('initial_velocity_m_s')) or not _vector(m.get('plane_normal')):
        raise ValueError('prescribed motion requires finite initial velocity and plane normal')
    if not _same(hypot(*m['plane_normal']),1.,absolute=1e-11):
        raise ValueError('prescribed motion plane normal must be unit length')
    if not _number(m.get('max_abs_energy_drift_j')) or m['max_abs_energy_drift_j']<0:
        raise ValueError('invalid maximum saved drift')
    if m.get('energy_convention') != (COMPOSED_ENERGY if composed else ELECTRIC_ENERGY):
        raise ValueError('composed energy convention must include qV+Ug' if composed else
                         'fixed test-charge energy convention must be full qV')
    expected_axis=next((s.axis for s in field.sources if hasattr(s,'_coordinate')),None)
    if ((expected_axis is None and m.get('axis') is not None) or
            (expected_axis is not None and (not _vector(m.get('axis')) or
            not all(_same(x,y) for x,y in zip(expected_axis,m['axis']))))):
        raise ValueError('saved fixed-source axis is not supported by the actual sources')
    orientation=m.get('axis_display','vertical')
    if (orientation not in ('vertical','horizontal') or
            (orientation=='horizontal' and expected_axis is None)):
        raise ValueError('unsupported display axis orientation for fixed source field')
    ann=m.get('source_annotations')
    if ann is not None:
        if not isinstance(ann,list) or len(ann)!=len(field.sources):
            raise ValueError('source annotations must match the original source inventory')
        ids,labels=[],[]
        for index,(source,item) in enumerate(zip(field.sources,ann)):
            if (not isinstance(item,dict) or
                    set(item)!={'source_index','id','label','kind'} or
                    type(item['source_index']) is not int or item['source_index']!=index or
                    item['kind']!=type(source).__name__ or
                    type(item['id']) is not str or
                    re.fullmatch(r'[A-Za-z][A-Za-z0-9:_-]{0,63}',item['id']) is None or
                    type(item['label']) is not str or
                    not 1<=len(item['label'])<=48 or
                    item['label']!=item['label'].strip() or
                    not item['label'].isprintable()):
                raise ValueError('source annotation ID, kind or label does not match the source order')
            ids.append(item['id']);labels.append(item['label'])
        if len(set(ids))!=len(ids) or len(set(labels))!=len(labels):
            raise ValueError('source annotations contain duplicate physical identities or labels')
    if composed and ann and any(a['id']=='gravity' for a in ann):
        raise ValueError('gravity identity conflicts with an electric source')
    spec=TestChargeMotionSpec(field,PointCharge(particle['charge_c'],particle['position_m']),
        m['mass_kg'],m['initial_velocity_m_s'],m['duration_s'],m['sample_interval_s'],
        plane_normal=m['plane_normal'],length_scale_m=m['length_scale_m'],
        source_guard_m=m['source_guard_m'],gravity=gravity)
    domain=_FixedDomain(spec)
    rows=m.get('samples')
    if not isinstance(rows,list) or not 2<=len(rows)<=2001:
        raise ValueError('prescribed motion requires 2 to 2001 full numerical samples')
    if rows[0].get('time_s')!=0:
        raise ValueError('test-charge time must start at zero')
    indices=m.get('frame_sample_indices')
    if (not isinstance(indices,list) or len(indices)!=len(frames) or
            not all(type(i) is int and 0<=i<len(rows) for i in indices) or
            indices[0]!=0 or indices[-1]!=len(rows)-1 or
            any(a>=b for a,b in zip(indices,indices[1:])) or
            m.get('graph_sample_count')!=len(rows) or
            m.get('field_frame_count')!=len(frames)):
        raise ValueError('saved real-time field-frame mapping is invalid')
    if not isinstance(m.get('trajectory_break_indices'),list) or any(
            type(i) is not int or not 1<=i<len(rows) for i in m['trajectory_break_indices']):
        raise ValueError('invalid saved path breaks')
    # Geometry crossings may be hidden by the renderer, never invented as a
    # physically traversed line. All unsafe display chords must be marked.
    expected_breaks=[]
    max_drift=0.
    mass,charge=m['mass_kg'],particle['charge_c']
    initial=particle['position_m'];normal=m['plane_normal']
    initial_energy=charge*field.potential_at(initial)
    initial_k=.5*mass*hypot(*m['initial_velocity_m_s'])**2
    last=-1.
    if composed and (not isinstance(rows[0],dict) or
            not _vector(rows[0].get('position_m')) or not _vector(rows[0].get('velocity_m_s')) or
            not _position_same(rows[0]['position_m'],initial,m['length_scale_m']) or
            not _position_same(rows[0]['velocity_m_s'],m['initial_velocity_m_s'],m['length_scale_m'])):
        raise ValueError('composed trajectory initial state differs from the authored launch')
    for i,r in enumerate(rows):
        if (not isinstance(r,dict) or
                any(not _vector(r.get(k)) for k in ('position_m','displacement_m',
                    'velocity_m_s','acceleration_m_s2','electric_field_v_m','force_n')) or
                any(not _number(r.get(k)) for k in ('time_s','potential_v',
                    'kinetic_energy_j','potential_energy_j','interaction_energy_j',
                    'total_energy_j','energy_drift_j')) or r['time_s']<=last):
            raise ValueError('fixed-source graph rows must have finite full vectors and increasing times')
        last=r['time_s']
        p,v=r['position_m'],r['velocity_m_s']
        if (domain.clearance(p)<=0 or
                hypot(*v)>=_NONRELATIVISTIC_SPEED_LIMIT_M_S or
                not _position_same(r['displacement_m'],_sub(p,initial),m['length_scale_m']) or
                abs(_dot(_sub(p,initial),normal))>2e-9*m['length_scale_m'] or
                abs(_dot(v,normal))>max(1e-12,2e-9*hypot(*v))):
            raise ValueError('saved fixed-source motion crosses a guard or leaves its certified plane')
        E=field.electric_field_at(p);V=field.potential_at(p)
        if (any(not _same(x,y,absolute=1e-14) for x,y in zip(E,r['electric_field_v_m'])) or
                not _same(V,r['potential_v'],absolute=1e-14)):
            raise ValueError('saved field/potential is not from the authored fixed sources')
        if composed:
            validate_composed_sample(spec, r)
        else:
            if any(k in r for k in CONTRIBUTION_KEYS):
                raise ValueError('undeclared gravity contributions in electric-only rows')
            if (any(not _same(a,charge*b,absolute=1e-27) for a,b in zip(r['force_n'],E)) or
                    any(not _same(a,charge*b/mass,absolute=1e-15) for a,b in zip(r['acceleration_m_s2'],E))):
                raise ValueError('saved test-charge force/acceleration is not F=qE')
            K=.5*mass*hypot(*v)**2
            U=charge*V
            drift=fsum((K-initial_k,U-initial_energy))
            if (not _same(r['kinetic_energy_j'],K) or
                    not _same(r['potential_energy_j'],U) or
                    r['interaction_energy_j'] != r['potential_energy_j'] or
                    not _same(r['total_energy_j'],fsum((K,U))) or
                    not _same(r['energy_drift_j'],drift)):
                raise ValueError('fixed-source energy must be K+qV, with saved drift')
        max_drift=max(max_drift,abs(r['energy_drift_j']))
        if i and domain.charge_segment_clearance(rows[i-1]['position_m'],p)<=0:
            expected_breaks.append(i)
    if sorted(set(m['trajectory_break_indices']))!=expected_breaks:
        raise ValueError('fixed-source displayed path skips a required guard break')
    if not (scalar_matches(m['max_abs_energy_drift_j'],max_drift) if composed
            else _same(max_drift,m['max_abs_energy_drift_j'])):
        raise ValueError('max absolute energy drift does not match samples')
    if m.get('termination') not in ('duration','source-guard') or not isinstance(m.get('events'),list):
        raise ValueError('bad fixed-source motion termination')
    if len(m['events']) != (0 if m['termination']=='duration' else 1):
        raise ValueError('fixed-source approach cutoff must have one exact terminal event')
    if m['termination']=='duration' and not _same(last,m['duration_s'],absolute=1e-14*m['duration_s']):
        raise ValueError('duration cutoff disagrees with saved times')
    if m['termination']=='source-guard':
        e=m['events'][0]
        if (e.get('kind')!='source-guard' or not _same(e.get('time_s'),last,absolute=1e-14*m['duration_s']) or
                not _vector(e.get('position_m')) or not _position_same(e['position_m'],rows[-1]['position_m'],m['length_scale_m'])):
            raise ValueError('source-guard event does not match saved last sample')
    p=m.get('profile_samples')
    if m.get('profile_mode')=='axis':
        if (expected_axis is None or not isinstance(p,list) or not 21<=len(p)<=501 or
                len(p)%2!=1):
            raise ValueError('axis-only field requires a finite axial profile')
    elif m.get('profile_mode')=='elapsed-time':
        if expected_axis is not None or p is not None:
            raise ValueError('nonaxial source must show exact trajectory-time observations')
    else:
        raise ValueError('unrecognized fixed-source graph mode')
    first_plane=frames[0].get('plane')
    if not isinstance(first_plane,dict):
        raise ValueError('missing physical display slice')
    try: plane=SliceSpec(**first_plane)
    except (TypeError,ValueError) as err: raise ValueError('invalid fixed-source display plane') from err
    if abs(_dot(plane.normal,normal))<1-1e-8:
        raise ValueError('fixed-source display axes must span certified physical plane')
    if expected_axis is not None:
        display_axis=plane.u if orientation=='horizontal' else plane.v
        if any(not _same(a,b,absolute=1e-11) for a,b in zip(display_axis,expected_axis)):
            raise ValueError('saved display axis orientation contradicts physical ring/disc symmetry axis')
    for frame,i in zip(frames,indices):
        row=rows[i]
        if (frame.get('plane')!=first_plane or frame.get('motion_sample_index')!=i or
                frame.get('conductor_results')!=[] or frame.get('geometry')!=[] or
                frame.get('contours')!=[] or frame.get('field_lines')!=[] or
                frame.get('source_masks')!=[] or len(frame.get('charges',[]))!=1):
            raise ValueError('fixed sources cannot be shown as induced conductors, field maps or extra probe charges')
        q=frame['charges'][0]
        if (q.get('charge_c')!=charge or q.get('motion_role')!='test particle' or
                q.get('position_m')!=row['position_m'] or
                not _vector(q.get('slice_position_m')) or
                not all(_same(x,y,absolute=1e-11) for x,y in zip(q['slice_position_m'],plane.project(row['position_m'])))):
            raise ValueError('display test particle must match exact actual-time position')
        grid=frame.get('grid')
        if (not isinstance(grid,dict) or any(v is not None for row_ in grid.get('potential_v',[]) for v in row_) or
                any(v is not None for row_ in grid.get('electric_field_v_per_m',[]) for v in row_)):
            raise ValueError('M19b schematics cannot invent an unsolved 2D field map')
    if p is not None:
        axis=expected_axis
        prev=-float('inf')
        for item in p:
            if (not isinstance(item,dict) or not _number(item.get('coordinate_m')) or
                    item['coordinate_m']<=prev):
                raise ValueError('axial field profile must increase in real metres')
            prev=item['coordinate_m']
            position=tuple(plane.origin_m[i]+axis[i]*prev for i in range(3))
            for key,func in [('potential_v',lambda:field.potential_at(position)),
                             ('field_parallel_v_m',lambda:_dot(field.electric_field_at(position),axis))]:
                try: expected=func()
                except SingularField: expected=None
                saved=item.get(key)
                if ((expected is None and saved is not None) or
                        (expected is not None and (not _number(saved) or not _same(expected,saved,absolute=1e-14)))):
                    raise ValueError('axial profile must use real source V/E values or explicit null gaps')
    return True
