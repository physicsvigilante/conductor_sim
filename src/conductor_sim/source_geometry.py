"""M25a: qualified, display-only geometry shared by motion-event viewers.

Build a physical-plane cross-section or an explicitly named 3D orthographic
projection from *existing* immutable field sources. Nothing here evaluates E,
V, motion, conductor response, or an off-axis ring/disc field. In particular,
off-plane points and rods never acquire false in-plane charge markers.
"""
from __future__ import annotations

from math import hypot, isfinite, sqrt, ulp
import re

from .prescribed_fields import (
    FixedPointField, PrescribedField, UniformDiskAxisField,
    UniformElectricField, UniformRingAxisField, UniformRingField, UniformRodField,
    UniformShellField, UniformSphereField, _dot, _sub, _unit, _vector,
)
from .prescribed_motion import TestChargeMotionSpec


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _projection_axis(normal):
    """Match historical event/driven orthographic screen orientation exactly."""
    seed = (1., 0., 0.) if abs(normal[0]) < .9 else (0., 1., 0.)
    transverse = _sub(seed, tuple(normal[i]*_dot(seed, normal) for i in range(3)))
    u = _unit(transverse, 'source geometry screen right')
    return u, _unit(_cross(u, normal), 'source geometry screen up')


def _plane_coordinate(position, origin, u, v, normal):
    point = _vector(position, 'display geometry world position')
    delta = _sub(point, origin)
    values = (_dot(delta, u), _dot(delta, v), _dot(delta, normal))
    if not all(isfinite(x) for x in values):
        raise ValueError('source geometry projection overflowed')
    return values


def _normal_clearance(position, origin, length_scale):
    # Do not widen a slice merely because a charge is far away along the plane.
    magnitude = max(*(abs(x) for x in (*position, *origin)), length_scale)
    return max(1e-11*length_scale, 64*ulp(magnitude))


def _rim_axes(normal):
    seed = (1., 0., 0.) if abs(normal[0]) < .9 else (0., 1., 0.)
    first = _unit(_sub(seed, tuple(normal[i]*_dot(seed, normal) for i in range(3))), 'rim first axis')
    return first, _unit(_cross(normal, first), 'rim second axis')



def _projected_major_axis(a, b, radius):
    """Major projected semiaxis of a physical circular rim, not one rim column.

    The projection matrix is [a b]. Its major squared singular value is the
    largest eigenvalue of a*a.T + b*b.T. Divide by the physical radius before
    squaring to avoid an otherwise unnecessary radius-squared overflow.
    Either orientation is valid; the viewer paints both endpoint signs.
    """
    ax, ay, bx, by = a[0]/radius, a[1]/radius, b[0]/radius, b[1]/radius
    xx, xy, yy = ax*ax+bx*bx, ax*ay+bx*by, ay*ay+by*by
    eigen = (xx+yy+hypot(xx-yy, 2*xy))/2
    # Choose the better-conditioned eigenvector identity in near-axial cases.
    c1, c2 = (xy, eigen-xx), (eigen-yy, xy)
    direction = c1 if hypot(*c1) >= hypot(*c2) else c2
    norm = hypot(*direction)
    if norm == 0:  # Face-on circle: any direction is a major axis.
        direction, norm = (1., 0.), 1.
    semiaxis = radius*sqrt(eigen)
    result = (semiaxis*direction[0]/norm, semiaxis*direction[1]/norm)
    if not all(isfinite(x) for x in result):
        raise ValueError('circular projected semiaxis overflowed')
    return result

def source_geometry(spec: TestChargeMotionSpec, annotations, *, trusted_scenario=None) -> dict:
    """A typed, deterministic visual ledger; unchanged physical model and IDs.

    `point`, `rod` and `sphere/shell` are *physical slice geometry*; rings
    and discs are *3D orthographic projections* of their actual circles.
    Fields are arrow glyphs, not a field-line map. Undrawable shapes keep their
    world positions and signed normal offsets in the visible source ledger.

    Raw (spec, annotations) is an *author assertion*; syntax/uniqueness alone
    cannot authenticate physical source IDs. Integrated viewers supply the
    already validated, immutable PrescribedScenario as trusted_scenario and
    require its exact spec object and annotation binding. HTML is not an
    authenticated import format; model revision IDs encode authored facts.
    """
    if type(spec) is not TestChargeMotionSpec or type(spec.field) is not PrescribedField:
        raise TypeError('geometry requires a qualified built-in particle-motion specification')
    annotations = tuple(annotations)
    if trusted_scenario is not None:
        # Only an explicit immutable authored scenario can bind physical IDs.
        # The independent-array helper does not claim to authenticate labels.
        from .scenarios import PrescribedScenario
        if (type(trusted_scenario) is not PrescribedScenario or
                trusted_scenario.spec is not spec or
                trusted_scenario.source_annotations != annotations):
            raise ValueError('geometry IDs must match the exact retained scenario binding')
    if len(annotations) != len(spec.field.sources):
        raise ValueError('physical source IDs must match the exact source inventory')
    n = _unit(spec.plane_normal, 'physical-plane normal')
    u, v = _projection_axis(n)
    origin = spec.particle.position_m
    entries = []
    used = set()
    used_labels = set()
    source_id = re.compile(r'[A-Za-z][A-Za-z0-9:_-]{0,63}\Z')
    for index, (src, entry) in enumerate(zip(spec.field.sources, annotations)):
        if (type(entry) not in (tuple, list) or len(entry) != 2 or
                type(entry[0]) is not str or source_id.fullmatch(entry[0]) is None or
                type(entry[1]) is not str or not 1 <= len(entry[1]) <= 48 or
                entry[1] != entry[1].strip() or not entry[1].isprintable() or
                entry[0] in used or entry[1] in used_labels):
            raise ValueError('invalid, duplicate, or mismatched physical source ID')
        used.add(entry[0])
        used_labels.add(entry[1])
        kind = type(src).__name__
        e = dict(index=index, source_id=entry[0], label=entry[1], kind=kind,
                 shape=None, status=None, charge_c=None, notes='', bounds_uv=[])
        if type(src) is UniformElectricField:
            # These glyphs show uniform direction, never measured field lines.
            projected = (_dot(src.field_v_m, u), _dot(src.field_v_m, v))
            e.update(shape='field-arrows', status='background-field',
                     field_uv=projected, field_world_v_m=src.field_v_m,
                     notes='Uniform electric field (direction glyphs, not field lines).')
        elif type(src) is FixedPointField:
            p = _plane_coordinate(src.position_m, origin, u, v, n)
            e.update(charge_c=src.charge_c, world_position_m=src.position_m,
                     signed_normal_offset_m=p[2], center_uv=p[:2])
            if abs(p[2]) > _normal_clearance(src.position_m, origin, spec.length_scale_m):
                e.update(shape='omitted', status='omitted-out-of-plane',
                         notes='Point charge omitted from physical slice; full 3D field remains active. No in-plane replacement.')
            else:
                e.update(shape='point', status='in-slice', bounds_uv=[p[:2]],
                         notes='Actual point charge on the physical motion slice.')
        elif type(src) is UniformRodField:
            a = _plane_coordinate(src.start_m, origin, u, v, n)
            b = _plane_coordinate(src.end_m, origin, u, v, n)
            e.update(charge_c=src.charge_c, world_endpoints_m=[src.start_m, src.end_m],
                     signed_endpoint_offsets_m=[a[2], b[2]], endpoints_uv=[a[:2], b[:2]])
            if (abs(a[2]) > _normal_clearance(src.start_m, origin, spec.length_scale_m) or
                    abs(b[2]) > _normal_clearance(src.end_m, origin, spec.length_scale_m)):
                e.update(shape='omitted', status='omitted-out-of-plane',
                         notes='Rod is not wholly in physical slice; omitted, including any projected chord. Full 3D field remains active.')
            else:
                e.update(shape='rod', status='in-slice', bounds_uv=[a[:2], b[:2]],
                         notes='Physical charged finite rod in the motion slice.')
        elif type(src) in (UniformSphereField, UniformShellField):
            c = _plane_coordinate(src.center_m, origin, u, v, n)
            e.update(charge_c=src.charge_c, radius_m=src.radius_m,
                     world_position_m=src.center_m, signed_normal_offset_m=c[2], center_uv=c[:2])
            separation = abs(c[2])
            if separation > src.radius_m:
                # Physical geometric non-intersection is STRICT: the numerical
                # tolerance for an almost invariant motion plane cannot make
                # a genuine off-plane sphere/shell intersect that plane.
                e.update(shape='omitted', status='omitted-out-of-plane',
                         notes='No physical intersection with motion slice; source field still active.')
            elif separation == src.radius_m:
                e.update(shape='slice-tangent', status='slice-tangent',
                         slice_radius_m=0., bounds_uv=[c[:2]],
                         notes='Actual single-point tangent to physical motion slice; no nonzero-area cross-section.')
            else:
                radius = sqrt((src.radius_m-separation)*(src.radius_m+separation))
                e.update(shape='sphere-slice' if type(src) is UniformSphereField else 'shell-slice',
                         status='slice-intersection', slice_radius_m=radius,
                         bounds_uv=[(c[0]-radius,c[1]-radius),(c[0]+radius,c[1]+radius)],
                         notes=('Uniform nonconducting volume-charge sphere cross-section.' if type(src) is UniformSphereField
                                else 'Fixed nonconducting thin-shell surface intersection; field jumps at surface.'))
        elif type(src) in (UniformRingAxisField, UniformRingField, UniformDiskAxisField):
            c = _plane_coordinate(src.center_m, origin, u, v, n)
            p, q = _rim_axes(src.axis)
            a = (src.radius_m*_dot(p,u), src.radius_m*_dot(p,v))
            b = (src.radius_m*_dot(q,u), src.radius_m*_dot(q,v))
            if not all(isfinite(x) for x in (*a,*b)):
                raise ValueError('circular source projection overflowed')
            edge_on = abs((a[0]/src.radius_m)*(b[1]/src.radius_m)-(a[1]/src.radius_m)*(b[0]/src.radius_m)) <= 1e-10
            rx, ry = hypot(a[0],b[0]), hypot(a[1],b[1])
            major = _projected_major_axis(a, b, src.radius_m)
            e.update(shape='ring-projection' if type(src) in (UniformRingAxisField, UniformRingField) else 'disk-projection',
                     status='orthographic-projection', charge_c=src.charge_c,
                     radius_m=src.radius_m, world_position_m=src.center_m,
                     signed_normal_offset_m=c[2], center_uv=c[:2],
                     world_axis=src.axis, rim_basis_uv=[a,b], projected_major_axis_uv=major,
                     edge_on=edge_on,
                     bounds_uv=[(c[0]-rx,c[1]-ry),(c[0]+rx,c[1]+ry)],
                     notes=('Fixed thin charged ring' if type(src) in (UniformRingAxisField, UniformRingField)
                            else 'Fixed uniformly charged nonconducting disc')+
                           ' · '+('edge-on' if edge_on else 'elliptical')+
                           (' orthographic projection ONLY; full 3D off-axis E/V is evaluated in the certified plane.'
                            if type(src) is UniformRingField else
                            ' orthographic projection ONLY; field and potential remain axis-only.'))
            if type(src) is UniformRingField:
                from .ring_motion_symmetry import certify_ring_motion_plane
                e['motion_symmetry_mode'] = certify_ring_motion_plane(
                    src, origin_m=origin, normal=n)
        else:
            raise TypeError('unsupported physical source type: '+kind)
        # Guard display-only computed extents against overflow: the underlying
        # valid physical model may still be usable with a different view.
        if any(not isfinite(x) for p in e['bounds_uv'] for x in p):
            raise ValueError('source geometry display extent overflowed')
        entries.append(e)
    has_full_ring = any(type(src) is UniformRingField for src in spec.field.sources)
    # Old source-geometry ledgers and viewer imports remain byte-identical
    # when the new full-ring component is absent.
    qualification = (
        'Geometry only; new full rings use genuine off-axis 3D E/V under strict '
        'meridian/equatorial physical-plane symmetry; legacy axial rings/discs '
        'remain axis-only. Ring rims are orthographic 3D projections, not '
        'charged rods or particle guides. Off-plane point/rod sources remain '
        'active in the physical field despite their omitted glyphs. '
        'No conductor response or browser physics.'
        if has_full_ring else
        'Geometry only; no off-axis ring/disc field or potential, no conductor response. '
        'Off-plane point/rod objects have no false in-plane markers. '
        'Ring/disc shapes are orthographic 3D projections; spheres/shells are physical slice intersections.')
    return dict(schema='conductor_sim.source_geometry.v1',
                physical_plane='through moving particle at initial position; normal is authored',
                origin_m=origin, normal=n, u=u, v=v, entries=entries,
                qualification=qualification)


# A single browser renderer for conservative-event and drag/scheduled viewers.
# It consumes only *precomputed, typed, Python-owned* shape projections.
SOURCE_GEOMETRY_JS = r'''
function sourceBounds(points,geometry){
 for(const e of geometry.entries)for(const p of e.bounds_uv)points.push(p);
}
function populateSourceGeometry(target,geometry){
 for(const e of geometry.entries){
  const p=document.createElement('p');p.className='source-geometry-entry';
  const props=[];
  if(e.charge_c!==null)props.push('q='+fmt(e.charge_c)+' C');
  if(e.radius_m!==undefined)props.push('radius='+fmt(e.radius_m)+' m');
  if(e.world_axis!==undefined)props.push('axis='+vec(e.world_axis));
  if(e.world_position_m!==undefined)props.push('world position='+vec(e.world_position_m)+' m');
  if(e.world_endpoints_m!==undefined)props.push('world endpoints='+e.world_endpoints_m.map(vec).join(' → ')+' m');
  if(e.signed_normal_offset_m!==undefined)props.push('signed normal offset='+fmt(e.signed_normal_offset_m)+' m');
  if(e.signed_endpoint_offsets_m!==undefined)props.push('signed endpoint offsets='+e.signed_endpoint_offsets_m.map(x=>fmt(x)).join(', ')+' m');
  p.textContent=e.source_id+' — '+e.label+' · '+e.kind+' · '+e.status+'\n'+props.join(' · ')+'\n'+e.notes;
  target.appendChild(p);
 }
 const note=document.createElement('p');note.className='muted';note.id='source-geometry-scope';note.textContent=geometry.qualification;target.appendChild(note);
}
function drawSourceGeometry(ctx,geometry,X,Y,pix,region){
 if(!$('source-geometry').checked)return;
 const colors=q=>q>0?'#b2483c':'#326aab';
 const label=(e,p,postfix='')=>{ctx.fillStyle='#274056';ctx.font='12px system-ui';ctx.fillText(e.source_id+postfix,X(p[0])+7,Y(p[1])-10);};
 const path=(points,close=false)=>{ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(X(p[0]),Y(p[1])):ctx.moveTo(X(p[0]),Y(p[1])));if(close)ctx.closePath();};
 ctx.save();ctx.beginPath();ctx.rect(...region);ctx.clip();
 for(const e of geometry.entries){
  if(e.shape==='omitted')continue;
  ctx.lineWidth=2;ctx.setLineDash([]);ctx.strokeStyle=colors(e.charge_c);ctx.fillStyle=colors(e.charge_c);
  if(e.shape==='field-arrows'){
   const field=e.field_uv,m=Math.hypot(...field);if(!m)continue;
   const [l,t,w,h]=region;
   ctx.strokeStyle='#7182bd';ctx.fillStyle='#7182bd';ctx.lineWidth=1.5;
   for(let ix=1;ix<=3;ix++)for(let iy=1;iy<=2;iy++){
    const x=l+w*ix/4,y=t+h*iy/3,dx=17*field[0]/m,dy=-17*field[1]/m;
    ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x+dx,y+dy);ctx.stroke();
    const th=Math.atan2(dy,dx);ctx.beginPath();ctx.moveTo(x+dx,y+dy);
    ctx.lineTo(x+dx-5*Math.cos(th-.48),y+dy-5*Math.sin(th-.48));
    ctx.lineTo(x+dx-5*Math.cos(th+.48),y+dy-5*Math.sin(th+.48));ctx.fill();
   }
   continue;
  }
  if(e.shape==='point'){
   const p=e.center_uv;ctx.beginPath();ctx.arc(X(p[0]),Y(p[1]),5,0,2*Math.PI);ctx.fill();label(e,p);continue;
  }
  if(e.shape==='rod'){
   ctx.lineWidth=5;path(e.endpoints_uv);ctx.stroke();
   label(e,[(e.endpoints_uv[0][0]+e.endpoints_uv[1][0])/2,(e.endpoints_uv[0][1]+e.endpoints_uv[1][1])/2],' · rod');continue;
  }
  if(e.shape==='slice-tangent'){
   const p=e.center_uv;ctx.lineWidth=2;ctx.beginPath();ctx.arc(X(p[0]),Y(p[1]),4,0,2*Math.PI);ctx.stroke();
   label(e,p,' · tangent point');continue;
  }
  if(e.shape==='sphere-slice'||e.shape==='shell-slice'){
   const p=e.center_uv;ctx.setLineDash(e.shape==='shell-slice'?[3,4]:[8,4]);
   ctx.beginPath();ctx.arc(X(p[0]),Y(p[1]),e.slice_radius_m*pix,0,2*Math.PI);ctx.stroke();
   ctx.setLineDash([]);label(e,p,e.shape==='shell-slice'?' · shell slice':' · sphere slice');continue;
  }
  if(e.shape==='ring-projection'||e.shape==='disk-projection'){
   const c=e.center_uv,a=e.rim_basis_uv[0],b=e.rim_basis_uv[1];
   const circle=Array.from({length:81},(_,j)=>{
    const t=2*Math.PI*j/80;return[c[0]+a[0]*Math.cos(t)+b[0]*Math.sin(t),c[1]+a[1]*Math.cos(t)+b[1]*Math.sin(t)];});
   path(circle,true);
   if(e.shape==='disk-projection'){
    if(e.edge_on){ctx.lineWidth=7;ctx.globalAlpha=.35;ctx.stroke();ctx.globalAlpha=1;}
    else{ctx.globalAlpha=.18;ctx.fill();ctx.globalAlpha=1;ctx.lineWidth=2;ctx.stroke();}
   }else{
    ctx.setLineDash([4,3]);ctx.lineWidth=2;ctx.stroke();ctx.setLineDash([]);
    if(e.edge_on){
     // The end of an edge-on projected circle is the major singular
     // semiaxis of [a b], not necessarily either parameterization column.
     const z=e.projected_major_axis_uv;
     for(const s of [-1,1]){ctx.beginPath();ctx.arc(X(c[0]+s*z[0]),Y(c[1]+s*z[1]),4,0,2*Math.PI);ctx.stroke();}
    }
   }
   label(e,c,(e.shape==='ring-projection'?' · ring':' · disc')+(e.edge_on?' (edge-on)':' (projection)'));
   continue;
  }
  throw new Error('unrecognized qualified geometry shape');
 }
 ctx.restore();
}
'''
