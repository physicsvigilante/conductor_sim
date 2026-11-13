"""Semantic validation of saved planar motion, without re-integrating physics.

A self-consistent file is not an authenticated integration result. This checks
all graph rows as well as field rows, before solver acquisition or HTML output.
It does not certify arbitrary user-edited trajectories against a force law.
"""
from math import fsum, hypot, isclose, isfinite, ulp


_VECTOR_FIELDS = ('position_m', 'displacement_m', 'velocity_m_s',
                  'acceleration_m_s2', 'force_n')
_SCALAR_FIELDS = ('time_s', 'kinetic_energy_j', 'interaction_energy_j',
                  'total_energy_j', 'energy_drift_j')


def _number(value):
    return type(value) in (int, float) and isfinite(value)


def _vector(value):
    return (isinstance(value, (list, tuple)) and len(value) == 3
            and all(_number(x) for x in value))


def _dot(a, b):
    return fsum(x*y for x, y in zip(a, b))


def _sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def _close(a, b):
    return isclose(a, b, rel_tol=2e-9, abs_tol=1e-30)


def _position_close(a, b, scale):
    # Allow subtraction roundoff for translated models, not a tolerance
    # proportional to distance from the unrelated world-coordinate origin.
    rounding = 16*max(ulp(abs(x)) for x in (*a, *b))
    return hypot(*_sub(a, b)) <= 2e-9*scale + rounding


def validate_planar_motion(motion, *, frames=None):
    """Reject malformed/mismatched SI motion metadata; leave input untouched."""
    if not isinstance(motion, dict) or motion.get('schema') not in (
            'conductor_sim.motion_2d.v1', 'conductor_sim.motion_viewer_2d.v1'):
        raise ValueError('expected a planar motion schema')
    mass, charge = motion.get('mass_kg'), motion.get('charge_c')
    if not _number(mass) or mass <= 0 or not _number(charge) or charge == 0:
        raise ValueError('motion mass must be positive and charge finite and nonzero')
    for key in ('initial_position_m', 'initial_velocity_m_s', 'plane_origin_m', 'plane_normal'):
        if not _vector(motion.get(key)):
            raise ValueError('motion requires finite initial vectors and physical plane')
    if not _number(motion.get('guard_m')) or motion['guard_m'] <= 0:
        raise ValueError('motion surface guard must be finite and positive')
    normal, origin = motion['plane_normal'], motion['plane_origin_m']
    if not isclose(hypot(*normal), 1., rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError('motion plane normal must be a unit vector')
    # Optional M18 source inventory is shared by numerical and viewer data.
    # Original one-source M17 exports remain valid without these fields.
    stationary = motion.get('stationary_charges')
    if stationary is None:
        if any(key in motion for key in ('moving_charge_index', 'charge_guard_m')):
            raise ValueError('incomplete stationary-source motion metadata')
    else:
        moving_index = motion.get('moving_charge_index')
        charge_guard = motion.get('charge_guard_m')
        if (not isinstance(stationary, list) or not 1 <= len(stationary) <= 31 or
                type(moving_index) is not int or not 0 <= moving_index <= len(stationary) or
                not _number(charge_guard) or charge_guard <= 0):
            raise ValueError('invalid M18a stationary-charge identity or guard metadata')
        indices = []
        positions = [motion['initial_position_m']]
        for fixed in stationary:
            if (not isinstance(fixed, dict) or type(fixed.get('charge_index')) is not int or
                    not _number(fixed.get('charge_c')) or fixed['charge_c'] == 0 or
                    not _vector(fixed.get('position_m'))):
                raise ValueError('held point charges must have finite original identities and SI data')
            indices.append(fixed['charge_index'])
            positions.append(fixed['position_m'])
        if (set(indices) != set(range(len(stationary)+1))-{moving_index} or
                len({tuple(p) for p in positions}) != len(positions)):
            raise ValueError('held point-charge indices and locations must be distinct and stable')
    rows = motion.get('samples')
    if not isinstance(rows, list) or not 2 <= len(rows) <= 2001:
        raise ValueError('motion requires 2 to 2001 finite numerical samples')
    last = -1.
    for row in rows:
        if (not isinstance(row, dict) or
                not all(_number(row.get(k)) for k in _SCALAR_FIELDS) or
                not all(_vector(row.get(k)) for k in _VECTOR_FIELDS) or row['time_s'] <= last):
            raise ValueError('planar motion samples require finite SI vectors and strictly increasing time')
        last = row['time_s']
    if rows[0]['time_s'] != 0:
        raise ValueError('planar motion must start at t=0')
    scale = max(1e-12, *(hypot(*_sub(r['position_m'], origin)) for r in rows))
    p0, v0 = motion['initial_position_m'], motion['initial_velocity_m_s']
    if (not _position_close(rows[0]['position_m'], p0, scale) or
            any(not _close(x,y) for x,y in zip(rows[0]['velocity_m_s'],v0))):
        raise ValueError('motion initial state must match its metadata')
    if stationary is not None:
        for fixed in stationary:
            point = fixed['position_m']
            transverse = _dot(_sub(point,origin),normal)
            if abs(transverse)>2e-9*max(scale,hypot(*_sub(point,origin)))+16*max(ulp(abs(v)) for v in point):
                raise ValueError('fixed charge lies outside the proven physical plane')
    for row in rows:
        position, velocity, force = (row[k] for k in ('position_m', 'velocity_m_s', 'force_n'))
        if stationary is not None and any(
                hypot(*_sub(position, fixed['position_m'])) <= charge_guard
                for fixed in stationary):
            raise ValueError('moving charge sample crossed the held point-charge guard')
        if not _position_close(row['displacement_m'], _sub(position,p0), scale):
            raise ValueError('motion displacement disagrees with actual numerical position')
        projected = tuple(position[i]-_dot(_sub(position,origin),normal)*normal[i] for i in range(3))
        if not _position_close(position, projected, scale):
            raise ValueError('all motion graph positions must remain in the physical slice')
        if (abs(_dot(velocity,normal)) > max(1e-12,5e-9*hypot(*velocity)) or
                abs(_dot(force,normal)) > max(1e-20,5e-9*hypot(*force))):
            raise ValueError('motion velocity/force disagrees with physical planar symmetry')
        if any(not _close(a, f/mass) for a,f in zip(row['acceleration_m_s2'], force)):
            raise ValueError('motion acceleration disagrees with force and mass')
        from .motion_1d import _NONRELATIVISTIC_SPEED_LIMIT_M_S
        if hypot(*velocity) >= _NONRELATIVISTIC_SPEED_LIMIT_M_S:
            raise ValueError('saved classical motion speed must be below 0.1c')
        kinetic = .5*mass*hypot(*velocity)**2
        if row['kinetic_energy_j'] < 0 or not _close(row['kinetic_energy_j'], kinetic):
            raise ValueError('motion kinetic energy disagrees with velocity and mass')
        if not _close(row['total_energy_j'], fsum((kinetic,row['interaction_energy_j']))):
            raise ValueError('motion total energy disagrees with K+U')
    max_drift = motion.get('max_abs_energy_drift_j')
    if not _number(max_drift) or not _close(max_drift,max(abs(r['energy_drift_j']) for r in rows)):
        raise ValueError('motion maximum energy drift disagrees with saved samples')
    # Do not recover drift by subtracting large total energies: the integrator
    # intentionally computes it from separate direct/induced/background terms.
    events = motion.get('events')
    termination = motion.get('termination')
    if not isinstance(events,list) or termination not in ('duration','surface-guard','charge-guard'):
        raise ValueError('invalid planar motion event/termination metadata')
    if termination == 'charge-guard' and stationary is None:
        raise ValueError('charge-guard termination requires held sources')
    if len(events) != (1 if termination != 'duration' else 0):
        raise ValueError('guard termination requires its recorded approach event')
    for event in events:
        if (not isinstance(event,dict) or event.get('kind') != termination or
                not _number(event.get('time_s')) or not 0 <= event['time_s'] <= last or
                not _vector(event.get('position_m')) or not _close(event['time_s'],last) or
                not _position_close(event['position_m'],rows[-1]['position_m'],scale)):
            raise ValueError('motion approach event must match the terminal numerical state')
    if termination == 'charge-guard':
        terminal = rows[-1]['position_m']
        near = min(hypot(*_sub(terminal, fixed['position_m'])) for fixed in stationary)
        if near > charge_guard + max(.04*charge_guard,1e-8*scale):
            raise ValueError('charge-guard event is not near a held point charge')
    if frames is None:
        return
    indices = motion.get('frame_sample_indices')
    if (not isinstance(indices,list) or not 2 <= len(indices) <= 24 or len(indices) != len(frames) or
            not all(type(i) is int and 0 <= i < len(rows) for i in indices) or
            indices[0] != 0 or indices[-1] != len(rows)-1 or
            any(a>=b for a,b in zip(indices,indices[1:])) or
            motion.get('graph_sample_count') != len(rows) or
            motion.get('field_frame_count') != len(frames)):
        raise ValueError('motion frame indices/counts must match ordered full numerical samples')
    from .visualization import SliceSpec
    saved = frames[0].get('plane')
    try:
        if not isinstance(saved,dict):
            raise ValueError('missing slice')
        plane = SliceSpec(**saved)
        if any(not isclose(hypot(*saved[k]),1.,rel_tol=1e-10,abs_tol=1e-12) for k in ('u','v')):
            raise ValueError('slice basis must have unit axes')
        if abs(_dot(plane.normal,normal)) < 1-1e-8:
            raise ValueError('slice must match the physical invariant plane')
    except (ValueError,TypeError,KeyError) as error:
        raise ValueError('invalid planar motion display slice') from error
    positions = [row['position_m'] for row in rows]
    positions.extend(q['position_m'] for q in stationary or ())
    for position in positions:
        x,y,h = plane.project(position)
        if not plane.contains((x,y)) or abs(h) > 2e-9*scale + 16*max(ulp(abs(v)) for v in position):
            raise ValueError('every graph row and held charge must be inside the fixed physical display slice')
    for frame, index in zip(frames,indices):
        row = rows[index]
        sources = frame.get('charges',())
        if (frame.get('plane') != saved or frame.get('motion_sample_index') != index or
                not isinstance(sources,(list,tuple)) or len(sources) != 1+len(stationary or ())):
            raise ValueError('each motion field must match its sample, full charge inventory and fixed slice')
        expected = {q['charge_index']: (q['position_m'],q['charge_c'],'held')
                    for q in stationary or ()}
        expected[moving_index if stationary is not None else 0] = (row['position_m'],charge,'moving')
        seen = set()
        for source in sources:
            if not isinstance(source,dict):
                raise ValueError('invalid motion field source')
            authored = source.get('charge_index') if stationary is not None else 0
            if type(authored) is not int or authored not in expected or authored in seen:
                raise ValueError('motion field source identity must be unique and authored')
            seen.add(authored)
            position, value, role = expected[authored]
            if (source.get('charge_c') != value or
                    not _vector(source.get('position_m')) or list(source['position_m']) != list(position) or
                    not _vector(source.get('slice_position_m')) or
                    not _position_close(source['slice_position_m'],plane.project(position),scale) or
                    (stationary is not None and source.get('motion_role') != role)):
                raise ValueError('each motion field must match its source charge identity, role, position and slice')


def planar_path_breaks(motion, frame):
    """Flag display chords crossing metal, not collisions of the numerical path.

    A sparse saved curved trajectory must not be drawn as a straight line
    through a conductor. Reuse the accepted geometric intersection primitives;
    do not fill the gap with an invented continuous-time interpolant.
    """
    from .visualization import SliceSpec
    from .geometry import SphereBoundary, PlaneBoundary
    plane = SliceSpec(**frame['plane'])
    boundaries = []
    geometry = frame.get('geometry',())
    for shape in geometry:
        if not shape.get('visible'):
            continue
        if shape['kind']=='plane':
            boundaries.append(PlaneBoundary((*shape['point_m'],0.),(*shape['normal_xy'],0.)))
        elif shape['kind']=='sphere':
            boundaries.extend(SphereBoundary((*shape['center_m'],0.),radius)
                for radius in (shape['inner_radius_m'],shape['outer_radius_m']) if radius>0)
    # A chord through a held source's approach cutoff is also misleading.
    # These circles are numerical guards, not physical particle radii.
    for source in motion.get('stationary_charges',()):
        boundaries.append(SphereBoundary((*plane.project(source['position_m'])[:2],0.),
                                          motion['charge_guard_m']))
    points = [(*plane.project(row['position_m'])[:2],0.) for row in motion['samples']]
    return [i for i,(a,b) in enumerate(zip(points,points[1:]),1)
            if any(boundary.first_hit(a,b) is not None for boundary in boundaries)]
