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


def _safe_fsum(values):
    try:
        return fsum(values)
    except (OverflowError, ValueError):
        return float('inf')


def _dot(a, b):
    return fsum(x*y for x, y in zip(a, b))


def _sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def _close(a, b):
    return isclose(a, b, rel_tol=2e-9, abs_tol=1e-30)


def reconstructs_position(origin, displacement, position):
    """Each RK axis stores either world position or launch-relative displacement."""
    return all(a + d == p or p - a == d for a, d, p in zip(origin, displacement, position))


def normal_roundoff(position, normal):
    """Maximum normal offset from rounding each integrated world coordinate."""
    return .5 * fsum(ulp(p)*abs(n) for p,n in zip(position, normal))


def _position_close(a, b, scale):
    if not isfinite(scale):
        return False
    dist = hypot(*_sub(a, b))
    if not isfinite(dist):
        return False
    # Allow subtraction roundoff for translated models, not a tolerance
    # proportional to distance from the unrelated world-coordinate origin.
    rounding = 16*max(ulp(abs(x)) for x in (*a, *b))
    return dist <= 2e-9*scale + rounding


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
    gravity_obj = None
    if 'gravity' in motion:
        from .conservative_forces import restore_gravity
        gravity_obj = restore_gravity(motion['gravity'])
        g_vec = gravity_obj.acceleration_m_s2
        g_scale = max(abs(x) for x in g_vec)
        if g_scale > 0.:
            g_scaled = tuple(x / g_scale for x in g_vec)
            if abs(_dot(g_scaled, normal)) > 64.0 * ulp(1.0) * hypot(*g_scaled):
                raise ValueError('saved uniform gravity lies outside the proven physical plane')
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
    if not isfinite(scale):
        raise ValueError('motion sample positions exceed finite distance scale')
    p0, v0 = motion['initial_position_m'], motion['initial_velocity_m_s']
    if (any(rows[0]['displacement_m']) or not _position_close(rows[0]['position_m'], p0, scale) or
            any(not _close(x,y) for x,y in zip(rows[0]['velocity_m_s'],v0))):
        raise ValueError('motion initial state must match its metadata')
    if stationary is not None:
        for fixed in stationary:
            point = fixed['position_m']
            transverse = _dot(_sub(point,origin),normal)
            if gravity_obj is not None:
                point_ulp = max(ulp(abs(v)) for v in point)
                rel = _sub(point, p0)
                rel_dist = hypot(*rel)
                if (abs(transverse) > 64 * ulp(1.0) * hypot(*_sub(point, origin)) + 16 * point_ulp
                        or (rel_dist > 0.0 and abs(_dot(rel, normal)) > 64 * ulp(1.0) * rel_dist + 16 * point_ulp)):
                    raise ValueError('fixed charge lies outside the proven physical plane')
            elif abs(transverse)>2e-9*max(scale,hypot(*_sub(point,origin)))+16*max(ulp(abs(v)) for v in point):
                raise ValueError('fixed charge lies outside the proven physical plane')
    for row in rows:
        position, velocity, force = (row[k] for k in ('position_m', 'velocity_m_s', 'force_n'))
        if stationary is not None and any(
                hypot(*_sub(position, fixed['position_m'])) <= charge_guard
                for fixed in stationary):
            raise ValueError('moving charge sample crossed the held point-charge guard')
        if (not reconstructs_position(p0, row['displacement_m'], position) and
                not _position_close(row['displacement_m'], _sub(position,p0), scale)):
            raise ValueError('motion displacement disagrees with actual numerical position')
        projected = tuple(position[i]-_dot(_sub(position,origin),normal)*normal[i] for i in range(3))
        if (not _position_close(position, projected, scale) or
                abs(_dot(row['displacement_m'],normal)) > 2e-9*scale):
            raise ValueError('all motion graph positions must remain in the physical slice')
        if gravity_obj is None:
            if (abs(_dot(velocity,normal)) > max(1e-12,5e-9*hypot(*velocity)) or
                    abs(_dot(force,normal)) > max(1e-20,5e-9*hypot(*force))):
                raise ValueError('motion velocity/force disagrees with physical planar symmetry')
        else:
            e_force, g_force = row.get('electric_force_n'), row.get('gravitational_force_n')
            e_u, g_u = row.get('electric_interaction_energy_j'), row.get('gravitational_potential_energy_j')
            if not (_vector(e_force) and _vector(g_force) and _number(e_u) and _number(g_u)):
                raise ValueError('gravity-coupled motion samples require separated force and energy contributions')
            f_scale_row = max(
                hypot(*force),
                hypot(*e_force),
                hypot(*g_force),
            )
            if (abs(_dot(velocity, normal)) > 5e-9 * hypot(*velocity)
                    or abs(_dot(force, normal)) > 5e-9 * f_scale_row
                    or abs(_dot(e_force, normal)) > 5e-9 * hypot(*e_force)):
                raise ValueError('motion velocity/force disagrees with physical planar symmetry')
        if gravity_obj is None:
            if any(not _close(a, f/mass) for a,f in zip(row['acceleration_m_s2'], force)):
                raise ValueError('motion acceleration disagrees with force and mass')
        else:
            for a, f in zip(row['acceleration_m_s2'], force):
                exp_a = f / mass
                if not isfinite(exp_a) or not isclose(a, exp_a, rel_tol=2e-9, abs_tol=128 * ulp(abs(exp_a))):
                    raise ValueError('motion acceleration disagrees with force and mass')
        from .motion_1d import _NONRELATIVISTIC_SPEED_LIMIT_M_S
        speed = hypot(*velocity)
        if speed >= _NONRELATIVISTIC_SPEED_LIMIT_M_S:
            raise ValueError('saved classical motion speed must be below 0.1c')
        kinetic = .5*mass*(speed * speed)
        if gravity_obj is None:
            if row['kinetic_energy_j'] < 0 or not _close(row['kinetic_energy_j'], kinetic):
                raise ValueError('motion kinetic energy disagrees with velocity and mass')
        else:
            if (not isfinite(kinetic) or row['kinetic_energy_j'] < 0 or
                    not isclose(row['kinetic_energy_j'], kinetic, rel_tol=2e-9, abs_tol=128 * ulp(abs(kinetic)))):
                raise ValueError('motion kinetic energy disagrees with velocity and mass')
        if gravity_obj is None:
            if any(k in row for k in ('electric_force_n', 'gravitational_force_n',
                                      'electric_interaction_energy_j', 'gravitational_potential_energy_j')):
                raise ValueError('gravity contribution fields require gravity metadata')
            if not _close(row['total_energy_j'], fsum((kinetic, row['interaction_energy_j']))):
                raise ValueError('motion total energy disagrees with K+U')
        else:
            exp_gf = gravity_obj.force_at(position, mass_kg=mass)
            exp_gu = gravity_obj.potential_energy_at(position, mass_kg=mass)
            if any(a != b for a, b in zip(g_force, exp_gf)):
                raise ValueError('saved gravitational force disagrees with UniformGravity')
            for a, b, c in zip(force, e_force, exp_gf):
                exp_f = _safe_fsum((b, c))
                if not isfinite(exp_f) or not isclose(a, exp_f, rel_tol=0.0, abs_tol=8 * ulp(abs(exp_f))):
                    raise ValueError('saved total force disagrees with electric + gravitational sum')
            if g_u != exp_gu:
                raise ValueError('saved gravitational energy disagrees with UniformGravity')
            exp_u = _safe_fsum((e_u, exp_gu))
            if not isfinite(exp_u) or not isclose(row['interaction_energy_j'], exp_u,
                                                  rel_tol=0.0, abs_tol=8 * ulp(abs(exp_u))):
                raise ValueError('saved interaction energy disagrees with electric + gravitational sum')
            exp_e = _safe_fsum((row['kinetic_energy_j'], e_u, exp_gu))
            if not isfinite(exp_e) or not isclose(row['total_energy_j'], exp_e,
                                                  rel_tol=0.0, abs_tol=8 * ulp(abs(exp_e))):
                raise ValueError('motion total energy disagrees with K+U_elec+U_grav')
    if gravity_obj is not None:
        if rows[0]['energy_drift_j'] != 0.0:
            raise ValueError('saved initial energy_drift_j must be zero')
        from .conservative_forces import UniformGravity
        local_gravity = UniformGravity(gravity_obj.acceleration_m_s2, reference_position_m=p0)
        r0 = rows[0]
        seen_positions = {}
        seen_pos_ke = {}
        for row in rows:
            pos_key = tuple(row['position_m'])
            ef_key = tuple(row['electric_force_n'])
            if pos_key in seen_positions:
                prev_eu, prev_ef, prev_k, prev_drift = seen_positions[pos_key]
                if row['electric_interaction_energy_j'] != prev_eu or ef_key != prev_ef:
                    raise ValueError('saved electrostatic force/energy changed at a repeated particle position')
                dk_rep = _safe_fsum((row['kinetic_energy_j'], -prev_k))
                k_rep_err = 0.0 if row['kinetic_energy_j'] == prev_k else 32 * ulp(max(abs(row['kinetic_energy_j']), abs(prev_k)))
                if not isclose(_safe_fsum((row['energy_drift_j'], -prev_drift)), dk_rep,
                               rel_tol=1e-6, abs_tol=k_rep_err):
                    raise ValueError('saved planar energy_drift_j disagrees at a repeated particle position')
            else:
                seen_positions[pos_key] = (
                    row['electric_interaction_energy_j'], ef_key,
                    row['kinetic_energy_j'], row['energy_drift_j'],
                )
            pos_ke_key = (pos_key, row['kinetic_energy_j'])
            if pos_ke_key in seen_pos_ke:
                if row['energy_drift_j'] != seen_pos_ke[pos_ke_key]:
                    raise ValueError('saved planar energy_drift_j changed at identical position and kinetic energy')
            else:
                seen_pos_ke[pos_ke_key] = row['energy_drift_j']
            same_pos = pos_key == tuple(r0['position_m'])
            dk = _safe_fsum((row['kinetic_energy_j'], -r0['kinetic_energy_j']))
            deu = _safe_fsum((row['electric_interaction_energy_j'], -r0['electric_interaction_energy_j']))
            dgu = local_gravity.potential_energy_at(row['position_m'], mass_kg=mass)
            exp_drift = _safe_fsum((dk, deu, dgu))
            k_err = 0.0 if row['kinetic_energy_j'] == r0['kinetic_energy_j'] else 32 * ulp(max(abs(row['kinetic_energy_j']), abs(r0['kinetic_energy_j'])))
            eu_err = 0.0 if same_pos else 32 * ulp(max(abs(row['electric_interaction_energy_j']), abs(r0['electric_interaction_energy_j'])))
            gu_err = 0.0 if dgu == 0.0 else 32 * ulp(abs(dgu))
            if not isfinite(exp_drift) or not isclose(
                    row['energy_drift_j'], exp_drift,
                    rel_tol=1e-6, abs_tol=k_err + eu_err + gu_err):
                raise ValueError('saved planar energy_drift_j disagrees with energy ledger')
    max_drift = motion.get('max_abs_energy_drift_j')
    actual_max_drift = max(abs(r['energy_drift_j']) for r in rows)
    if gravity_obj is not None:
        if not _number(max_drift) or not isclose(max_drift, actual_max_drift, rel_tol=2e-9, abs_tol=128 * ulp(actual_max_drift)):
            raise ValueError('motion maximum energy drift disagrees with saved samples')
    elif not _number(max_drift) or not _close(max_drift, actual_max_drift):
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


def validate_motion_1d_gravity(motion):
    """Validate optional M45 1D gravity metadata and separated sample ledgers.

    Checks every row in ``motion['samples']`` (both field-frame rows and
    graph-only rows) without altering legacy non-gravity M16 exports.
    """
    rows = motion.get('samples')
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError('1D motion requires at least two finite samples')
    gravity_keys = (
        'electric_force_n',
        'gravitational_force_n',
        'electric_interaction_energy_j',
        'gravitational_potential_energy_j',
    )
    if 'gravity' not in motion:
        for row in rows:
            if not isinstance(row, dict) or any(k in row for k in gravity_keys):
                raise ValueError('gravity contribution fields require gravity metadata')
        return
    from .conservative_forces import restore_gravity
    gravity_obj = restore_gravity(motion['gravity'])
    mass = motion.get('mass_kg')
    if not _number(mass) or mass <= 0:
        raise ValueError('gravity-coupled 1D motion requires positive finite mass_kg')
    axis = motion.get('axis')
    if not _vector(axis) or not isclose(hypot(*axis), 1., rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError('gravity-coupled 1D motion requires a unit axis vector')
    p0 = motion.get('initial_position_m')
    if not _vector(p0):
        raise ValueError('gravity-coupled 1D motion requires a finite initial_position_m')
    g_vec = gravity_obj.acceleration_m_s2
    g_scale = max(abs(x) for x in g_vec)
    if g_scale > 0.:
        if any(axis[i] == 0.0 and g_vec[i] != 0.0 for i in range(3)):
            raise ValueError('saved 1D uniform gravity is not collinear with the motion axis')
        g_scaled = tuple(x / g_scale for x in g_vec)
        g_along = _dot(g_scaled, axis)
        g_trans = hypot(*(g_scaled[i] - g_along * axis[i] for i in range(3)))
        if g_trans > 64.0 * ulp(1.0) * hypot(*g_scaled):
            raise ValueError('saved 1D uniform gravity is not collinear with the motion axis')
    for row in rows:
        if not isinstance(row, dict) or not _vector(row.get('position_m')):
            raise ValueError('gravity-coupled 1D motion samples require finite position_m')
    scale = max(1e-12, *(hypot(*_sub(r['position_m'], p0)) for r in rows))
    if not isfinite(scale):
        raise ValueError('saved 1D sample positions exceed finite distance scale')
    for row in rows:
        position = row['position_m']
        if not all(_number(row.get(k)) for k in (
                'displacement_m', 'velocity_m_s', 'acceleration_m_s2', 'force_n',
                'kinetic_energy_j', 'interaction_energy_j', 'total_energy_j', 'energy_drift_j')):
            raise ValueError('gravity-coupled 1D motion samples require finite kinematic and energy fields')
        exp_pos = tuple(p0[i] + row['displacement_m'] * axis[i] for i in range(3))
        if (not all(isfinite(x) for x in exp_pos) or
                any(abs(position[i] - exp_pos[i]) > (
                        0.0 if axis[i] == 0.0
                        else 2e-9 * scale + 16 * ulp(abs(row['displacement_m'] * axis[i])))
                    for i in range(3))):
            raise ValueError('saved 1D sample position disagrees with axis displacement')
        e_force, g_force = row.get('electric_force_n'), row.get('gravitational_force_n')
        e_u, g_u = row.get('electric_interaction_energy_j'), row.get('gravitational_potential_energy_j')
        if not all(_number(v) for v in (e_force, g_force, e_u, g_u)):
            raise ValueError('gravity-coupled 1D motion samples require separated force and energy contributions')
        exp_gf = _safe_fsum(x * y for x, y in zip(gravity_obj.force_at(position, mass_kg=mass), axis))
        exp_gu = gravity_obj.potential_energy_at(position, mass_kg=mass)
        if not isfinite(exp_gf) or g_force != exp_gf:
            raise ValueError('saved 1D gravitational force disagrees with UniformGravity')
        exp_f = _safe_fsum((e_force, exp_gf))
        if not isfinite(exp_f) or not isclose(row['force_n'], exp_f, rel_tol=0.0, abs_tol=8 * ulp(abs(exp_f))):
            raise ValueError('saved 1D total force disagrees with electric + gravitational sum')
        exp_acc = row['force_n'] / mass
        if (not isfinite(exp_acc) or
                not isclose(row['acceleration_m_s2'], exp_acc,
                            rel_tol=0.0, abs_tol=8 * ulp(abs(exp_acc)))):
            raise ValueError('saved 1D acceleration disagrees with force and mass')
        if g_u != exp_gu:
            raise ValueError('saved 1D gravitational energy disagrees with UniformGravity')
        exp_u = _safe_fsum((e_u, exp_gu))
        if not isfinite(exp_u) or not isclose(row['interaction_energy_j'], exp_u,
                                              rel_tol=0.0, abs_tol=8 * ulp(abs(exp_u))):
            raise ValueError('saved 1D interaction energy disagrees with electric + gravitational sum')
        from .motion_1d import _NONRELATIVISTIC_SPEED_LIMIT_M_S
        v_s = row['velocity_m_s']
        if abs(v_s) >= _NONRELATIVISTIC_SPEED_LIMIT_M_S:
            raise ValueError('saved 1D classical motion speed must be below 0.1c')
        kinetic = .5 * mass * (v_s * v_s)
        if (not isfinite(kinetic) or row['kinetic_energy_j'] < 0 or
                not isclose(row['kinetic_energy_j'], kinetic,
                            rel_tol=2e-9, abs_tol=128 * ulp(abs(kinetic)))):
            raise ValueError('saved 1D kinetic energy disagrees with velocity and mass')
        exp_e = _safe_fsum((row['kinetic_energy_j'], e_u, exp_gu))
        if not isfinite(exp_e) or not isclose(row['total_energy_j'], exp_e,
                                              rel_tol=0.0, abs_tol=8 * ulp(abs(exp_e))):
            raise ValueError('saved 1D total energy disagrees with K+U_elec+U_grav')
    if rows[0]['energy_drift_j'] != 0.0:
        raise ValueError('saved 1D initial energy_drift_j must be zero')
    from .conservative_forces import UniformGravity
    local_gravity = UniformGravity(gravity_obj.acceleration_m_s2, reference_position_m=p0)
    r0 = rows[0]
    seen_positions = {}
    seen_pos_ke = {}
    for row in rows:
        pos_key = tuple(row['position_m'])
        if pos_key in seen_positions:
            prev_eu, prev_ef, prev_k, prev_drift = seen_positions[pos_key]
            if row['electric_interaction_energy_j'] != prev_eu or row['electric_force_n'] != prev_ef:
                raise ValueError('saved 1D electrostatic force/energy changed at a repeated particle position')
            dk_rep = _safe_fsum((row['kinetic_energy_j'], -prev_k))
            k_rep_err = 0.0 if row['kinetic_energy_j'] == prev_k else 32 * ulp(max(abs(row['kinetic_energy_j']), abs(prev_k)))
            if not isclose(_safe_fsum((row['energy_drift_j'], -prev_drift)), dk_rep,
                           rel_tol=1e-6, abs_tol=k_rep_err):
                raise ValueError('saved 1D energy_drift_j disagrees at a repeated particle position')
        else:
            seen_positions[pos_key] = (
                row['electric_interaction_energy_j'], row['electric_force_n'],
                row['kinetic_energy_j'], row['energy_drift_j'],
            )
        pos_ke_key = (pos_key, row['kinetic_energy_j'])
        if pos_ke_key in seen_pos_ke:
            if row['energy_drift_j'] != seen_pos_ke[pos_ke_key]:
                raise ValueError('saved 1D energy_drift_j changed at identical position and kinetic energy')
        else:
            seen_pos_ke[pos_ke_key] = row['energy_drift_j']
        same_pos = pos_key == tuple(r0['position_m'])
        dk = _safe_fsum((row['kinetic_energy_j'], -r0['kinetic_energy_j']))
        deu = _safe_fsum((row['electric_interaction_energy_j'], -r0['electric_interaction_energy_j']))
        dgu = local_gravity.potential_energy_at(row['position_m'], mass_kg=mass)
        exp_drift = _safe_fsum((dk, deu, dgu))
        k_err = 0.0 if row['kinetic_energy_j'] == r0['kinetic_energy_j'] else 32 * ulp(max(abs(row['kinetic_energy_j']), abs(r0['kinetic_energy_j'])))
        eu_err = 0.0 if same_pos else 32 * ulp(max(abs(row['electric_interaction_energy_j']), abs(r0['electric_interaction_energy_j'])))
        gu_err = 0.0 if dgu == 0.0 else 32 * ulp(abs(dgu))
        if not isfinite(exp_drift) or not isclose(
                row['energy_drift_j'], exp_drift,
                rel_tol=1e-6, abs_tol=k_err + eu_err + gu_err):
            raise ValueError('saved 1D energy_drift_j disagrees with energy ledger')
    max_drift = motion.get('max_abs_energy_drift_j')
    actual_max = max(abs(r['energy_drift_j']) for r in rows)
    if not _number(max_drift) or not isclose(max_drift, actual_max, rel_tol=2e-9, abs_tol=128 * ulp(actual_max)):
        raise ValueError('saved 1D maximum energy drift disagrees with saved samples')
