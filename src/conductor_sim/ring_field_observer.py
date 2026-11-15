"""M27a static 3D ring E/V observations projected onto an authored 2D slice.

No responding conductors, off-axis legacy ring eval, browser calculations,
invented particle-time state or physical motion-plane constraint. Every cell
stores true world 3-vectors and the signed out-of-slice E component.
"""
from __future__ import annotations

from dataclasses import asdict
from html import escape
import json
from math import cos, fsum, hypot, isclose, isfinite, pi, sin
from pathlib import Path

from .jee_potential_explorer import _STYLE, _safe_json_script
from .prescribed_fields import (PrescribedField, UniformRingField,
    UniformElectricField, FixedPointField, _dot, _sub, _unit, _mul, _vector, _number)
from .prescribed_viewer_validation import _restore_field
from .jee_potential_explorer import _bind
from .solver import SingularField, UnsupportedProblem
from .visualization import SliceSpec


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _ring_rim(ring, plane):
    """Orthographic projection of the physical 3D ring (NOT a flat charge)."""
    axis = ring.axis
    seed = (1., 0., 0.) if abs(axis[0]) < .9 else (0., 1., 0.)
    first = _unit(_sub(seed, _mul(axis, _dot(seed, axis))), "ring rim first")
    second = _unit(_cross(axis, first), "ring rim second")
    points = []
    for j in range(73):
        angle = 2*pi*j/72
        p = tuple(ring.center_m[i] + ring.radius_m*(
            cos(angle)*first[i] + sin(angle)*second[i]) for i in range(3))
        points.append(list(plane.project(p)))
    return dict(kind="UniformRingField", projection="orthographic 3D rim, not material in slice",
                charge_c=ring.charge_c, radius_m=ring.radius_m,
                world_center_m=list(ring.center_m), world_axis=list(ring.axis),
                center_uvn=list(plane.project(ring.center_m)), projected_rim_uvn=points)


def _check_inventory(field, plane, annotations, mask_m):
    if type(field) is not PrescribedField or type(plane) is not SliceSpec:
        raise TypeError("ring slice requires PrescribedField and SliceSpec")
    ann = _bind(field, annotations)
    if not (1 <= len(field.sources) <= 12 and
            any(type(s) is UniformRingField for s in field.sources) and
            all(type(s) in (UniformRingField, UniformElectricField, FixedPointField)
                for s in field.sources)):
        raise UnsupportedProblem("M27a map supports 1..12 fixed rings, points and uniform E, with at least one off-axis-capable ring")
    mask = _number(mask_m, "ring display exclusion width", positive=True)
    if mask > .03*min(plane.x_range_m[1]-plane.x_range_m[0],
                       plane.y_range_m[1]-plane.y_range_m[0]):
        raise ValueError("source mask may not obscure more than 3% of either view span")
    if plane.resolution > 61:
        raise ValueError("M27a static ring map supports at most 61 grid points per axis")
    return ann, mask


def _masked(field, point, width):
    """Actual 3D source clearance; never mask an off-plane ring's shadow."""
    return any(s.clearance_at(point) <= width
               for s in field.sources if type(s) in (UniformRingField, FixedPointField))


def _projection_metadata(field, plane, annotations):
    return [dict(source_id=ann["id"], label=ann["label"], **_ring_rim(source, plane))
            for source, ann in zip(field.sources, annotations)
            if type(source) is UniformRingField]


def _sample(field, plane, mask, xy):
    point = plane.world(xy)
    values = dict(x_m=xy[0], y_m=xy[1], world_m=list(point))
    if _masked(field, point, mask):
        return dict(**values, region="source-exclusion",
                    potential_v=None, electric_field_v_m=None,
                    field_u_v_m=None, field_v_v_m=None, field_normal_v_m=None)
    potential = field.potential_at(point)
    electric = field.electric_field_at(point)
    return dict(**values, region="physical-observation",
                potential_v=potential, electric_field_v_m=list(electric),
                field_u_v_m=_dot(electric, plane.u),
                field_v_v_m=_dot(electric, plane.v),
                field_normal_v_m=_dot(electric, plane.normal))


def sample_ring_field_slice(field: PrescribedField, plane: SliceSpec, *,
                            source_annotations, exclusion_width_m: float):
    """Compute a complete, finite, masked 2D static slice of full 3D E/V.

    The slice is an observation, not an invariant particle-motion plane.
    Off-plane E remains in full vectors and normal-component readouts.
    """
    annotations, mask = _check_inventory(field, plane, source_annotations, exclusion_width_m)
    n = plane.resolution
    xs = [plane.x_range_m[0] + (plane.x_range_m[1]-plane.x_range_m[0])*i/(n-1)
          for i in range(n)]
    ys = [plane.y_range_m[0] + (plane.y_range_m[1]-plane.y_range_m[0])*j/(n-1)
          for j in range(n)]
    cells = [_sample(field, plane, mask, (x, y)) for y in ys for x in xs]
    record = dict(
        schema="conductor_sim.ring_field_slice.v1",
        physical_mode="static full-3D electric field/potential observed on a 2D plane",
        time_evolution=False, motion_qualified=False,
        field=field.as_dict(), plane=asdict(plane),
        source_annotations=annotations, exclusion_width_m=mask,
        excluded_region="distance to actual ideal charged wire/point in 3D; display mask, not material radius",
        full_vector_convention="in-plane arrows show projected components only; normal E is separately stored",
        ring_projections=_projection_metadata(field, plane, annotations),
        grid=dict(x_m=xs, y_m=ys, rows=n, columns=n, cells=cells))
    result = json.loads(json.dumps(record, allow_nan=False))
    validate_ring_field_slice(result)
    return result


def _real(x):
    return type(x) in (float, int) and isfinite(x)


def _same(a, b):
    return _real(a) and isclose(a, b, rel_tol=3e-9, abs_tol=2e-10)


def validate_ring_field_slice(record):
    """Recompute every grid value and geometry; reject altered/fictional E/V."""
    allowed = {"schema", "physical_mode", "time_evolution", "motion_qualified",
               "field", "plane", "source_annotations", "exclusion_width_m",
               "excluded_region", "full_vector_convention", "ring_projections", "grid"}
    if type(record) is not dict or set(record) != allowed or record.get("schema") != "conductor_sim.ring_field_slice.v1":
        raise ValueError("not a bounded M27a static ring-field record")
    if record["time_evolution"] is not False or record["motion_qualified"] is not False:
        raise ValueError("a static ring observation must not claim particle motion")
    if record["physical_mode"] != "static full-3D electric field/potential observed on a 2D plane":
        raise ValueError("ring physical observation mode changed")
    if record["full_vector_convention"] != "in-plane arrows show projected components only; normal E is separately stored":
        raise ValueError("field projection meaning changed")
    if record["excluded_region"] != "distance to actual ideal charged wire/point in 3D; display mask, not material radius":
        raise ValueError("source-exclusion meaning changed")
    try:
        field = _restore_field(record["field"])
        plane = SliceSpec(**record["plane"])
        incoming = record["source_annotations"]
        if type(incoming) is not list:
            raise ValueError("missing typed source identity ledger")
        annotations, mask = _check_inventory(
            field, plane, [(x["id"], x["label"]) for x in incoming],
            record["exclusion_width_m"])
    except (TypeError, KeyError, UnsupportedProblem, SingularField) as exc:
        raise ValueError("invalid static source, plane or binding") from exc
    if incoming != annotations:
        raise ValueError("source identity or kind differs from actual ring model")
    expected_geometry = _projection_metadata(field, plane, annotations)
    rendered_geometry = record["ring_projections"]
    if type(rendered_geometry) is not list or len(rendered_geometry) != len(expected_geometry):
        raise ValueError("ring source projection inventory differs from actual 3D sources")
    for saved, expected in zip(rendered_geometry, expected_geometry):
        if type(saved) is not dict or set(saved) != set(expected):
            raise ValueError("ring projection has forged or missing physical source metadata")
        # Exact source identities, physical parameters and world orientations
        # remain byte-specific. Only the calculated 2D/normal projection
        # coordinates tolerate unit-vector re-normalization by a few ulps
        # when a tilted SliceSpec is reconstructed from saved JSON.
        for key in ("source_id", "label", "kind", "projection", "charge_c",
                    "radius_m", "world_center_m", "world_axis"):
            if saved[key] != expected[key]:
                raise ValueError("ring projection identity or physical source was modified")
        scale = max(expected["radius_m"], *(abs(v) for v in (
            *expected["world_center_m"], *plane.origin_m)))
        tol = 4e-13*scale
        for key in ("center_uvn", "projected_rim_uvn"):
            a, b = saved[key], expected[key]
            if type(a) is not list or len(a) != len(b):
                raise ValueError("ring projection coordinates are malformed")
            pairs = (zip(a,b) if key == "center_uvn" else
                     (pair for arow,brow in zip(a,b)
                      if type(arow) is list and len(arow) == len(brow) == 3
                      for pair in zip(arow,brow)))
            if key == "projected_rim_uvn" and any(type(row) is not list or len(row)!=3 for row in a):
                raise ValueError("ring projected-rim coordinate shape changed")
            if any(not _real(v) or not isclose(v,ex,rel_tol=4e-13,abs_tol=tol)
                   for v,ex in pairs):
                raise ValueError("saved ring projection differs from actual 3D source geometry")
    grid = record["grid"]
    if type(grid) is not dict or set(grid) != {"x_m","y_m","rows","columns","cells"}:
        raise ValueError("invalid saved ring grid")
    n = plane.resolution
    if grid["rows"] != n or grid["columns"] != n or type(grid["cells"]) is not list or len(grid["cells"]) != n*n:
        raise ValueError("incorrect saved ring grid dimensions")
    x_expected = [plane.x_range_m[0] + (plane.x_range_m[1]-plane.x_range_m[0])*i/(n-1)
                  for i in range(n)]
    y_expected = [plane.y_range_m[0] + (plane.y_range_m[1]-plane.y_range_m[0])*j/(n-1)
                  for j in range(n)]
    for original, expected in ((grid["x_m"],x_expected),(grid["y_m"],y_expected)):
        if type(original) is not list or len(original)!=n or any(not _same(a,b) for a,b in zip(original,expected)):
            raise ValueError("saved physical grid coordinates changed")
    keys = {"x_m", "y_m", "world_m", "region", "potential_v",
            "electric_field_v_m", "field_u_v_m", "field_v_v_m", "field_normal_v_m"}
    for j,y in enumerate(y_expected):
        for i,x in enumerate(x_expected):
            old = grid["cells"][j*n+i]
            if type(old) is not dict or set(old) != keys:
                raise ValueError("malformed saved ring field sample")
            expected = _sample(field, plane, mask, (x,y))
            if not _same(old["x_m"],x) or not _same(old["y_m"],y):
                raise ValueError("saved ring physical position differs from grid")
            if not isinstance(old["world_m"],list) or len(old["world_m"])!=3 or any(
                not _same(a,b) for a,b in zip(old["world_m"],expected["world_m"])):
                raise ValueError("saved ring physical world position was changed")
            if old["region"] != expected["region"]:
                raise ValueError("actual 3D source exclusion was changed")
            for key in ("potential_v","field_u_v_m","field_v_v_m","field_normal_v_m"):
                if expected[key] is None:
                    if old[key] is not None: raise ValueError("source mask was replaced with a fictitious value")
                elif not _same(old[key],expected[key]):
                    raise ValueError("saved ring E/V disagrees with physical source")
            v_old, v_expected = old["electric_field_v_m"],expected["electric_field_v_m"]
            if v_expected is None:
                if v_old is not None: raise ValueError("masked ring E cannot be fabricated")
            elif not isinstance(v_old,list) or len(v_old)!=3 or any(
                not _same(a,b) for a,b in zip(v_old,v_expected)):
                raise ValueError("saved complete ring E vector disagrees with physical source")
    return True


def _heat(v, maximum):
    """Diverging, bounded signed-potential display; no extrapolated physics."""
    magnitude = min(1., abs(v)/maximum) if maximum else 0.
    white=(248,249,250)
    other=(191,76,52) if v>=0 else (64,104,166)
    color=tuple(round(white[i]*(1-magnitude)+other[i]*magnitude) for i in range(3))
    return "#%02x%02x%02x"%color


def render_ring_field_slice(record, *, title="Off-axis electric field of fixed charged rings"):
    """Stand-alone offline SVG using ONLY validated precomputed static states."""
    validate_ring_field_slice(record)
    if type(title) is not str or not 1<=len(title)<=120 or title!=title.strip() or not title.isprintable():
        raise ValueError("invalid static chart title")
    plane=record["plane"]
    grid=record["grid"]
    xs,ys=grid["x_m"],grid["y_m"]
    xmin,xmax=xs[0],xs[-1];ymin,ymax=ys[0],ys[-1]
    n=grid["rows"];w=780.;h=540.;offset_x=55.;offset_y=30.
    X=lambda x: offset_x+w*(x-xmin)/(xmax-xmin)
    Y=lambda y: offset_y+h*(ymax-y)/(ymax-ymin)
    width=w/(n-1);height=h/(n-1)
    cells=grid["cells"]
    potentials=[abs(v["potential_v"]) for v in cells if v["potential_v"] is not None]
    maximum=max(potentials,default=1.)
    svgs=['<svg id="static-ring-map" class="plot" viewBox="0 0 890 630" role="img" aria-label="Signed potential map and in-plane electric field components" xmlns="http://www.w3.org/2000/svg">']
    svgs.append('<rect width="890" height="630" fill="#f8fafc"/>')
    for cell in cells:
        color="#dde3e8" if cell["potential_v"] is None else _heat(cell["potential_v"], maximum)
        svgs.append('<rect class="potential-sample" x="%.5f" y="%.5f" width="%.5f" height="%.5f" fill="%s"/>'%(
            X(cell["x_m"])-width/2,Y(cell["y_m"])-height/2,width+0.3,height+0.3,color))
    # Clear grid boundaries and true world-frame coordinates.
    svgs.append('<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" stroke="#324b61" fill="none" stroke-width="1.2"/>'%(
        offset_x-width/2,offset_y-height/2,w+width,h+height))
    sampled=[cell for cell in cells if cell["potential_v"] is not None]
    max_e=max((hypot(cell["field_u_v_m"],cell["field_v_v_m"]) for cell in sampled), default=0.)
    step=max(1,(n-1)//11)
    for j in range(0,n,step):
        for i in range(0,n,step):
            cell=cells[j*n+i]
            if cell["potential_v"] is None:continue
            u,v=cell["field_u_v_m"],cell["field_v_v_m"]
            mag=hypot(u,v)
            if not mag:continue
            # All arrow lengths are normalized for legibility, not literal
            # field magnitude. Signed full E is preserved numerically.
            length=4+min(13.,12*mag/max_e) if max_e else 0.
            xx,yy=X(cell["x_m"]),Y(cell["y_m"])
            dx,dy=length*u/mag,-length*v/mag
            svgs.append('<path class="field-arrow" d="M%.3f %.3f l%.3f %.3f" stroke="#162a3c" stroke-width="1.5" fill="none"/>'%(xx,yy,dx,dy))
            px,py=xx+dx,yy+dy
            svgs.append('<circle cx="%.3f" cy="%.3f" r="2.4" fill="#162a3c"/>'%(px,py))
    for ring in record["ring_projections"]:
        pts=ring["projected_rim_uvn"]
        path=" ".join(("%.3f,%.3f"%(X(p[0]),Y(p[1]))) for p in pts)
        svgs.append('<polyline class="ring-source-projection" fill="none" stroke="#752aa7" stroke-width="3.2" stroke-dasharray="7,5" points="'+path+'"/>')
        cu,cv,_=ring["center_uvn"]
        svgs.append('<text x="%.2f" y="%.2f" font-size="13" fill="#5f257a">%s</text>'%(
            X(cu)+8,Y(cv)-12,escape(ring["label"]+" ["+ring["source_id"]+"]")))
    for i in range(5):
        x=xmin+(xmax-xmin)*i/4
        y=ymin+(ymax-ymin)*i/4
        svgs.append('<text x="%.2f" y="600" text-anchor="middle" font-size="12">%.3g</text>'%(X(x),x))
        svgs.append('<text x="42" y="%.2f" text-anchor="end" font-size="12">%.3g</text>'%(Y(y)+4,y))
    svgs.append('<text x="445" y="622" text-anchor="middle" font-size="13">Plane u coordinate (m)</text>')
    svgs.append('<text x="28" y="17" font-size="13">Plane v coordinate (m) ↑</text>')
    svgs.append('</svg>')
    ann=record["source_annotations"]
    items=''.join('<tr><td>'+escape(x["label"])+'</td><td><code>'+escape(x["id"])+'</code></td><td>'+
                  escape(x["kind"])+'</td></tr>' for x in ann)
    stats=''.join('<tr><td>'+escape(k)+'</td><td>'+escape(str(val))+'</td></tr>' for k,val in (
        ("Plane origin (m)",plane["origin_m"]),("World u",plane["u"]),("World v",plane["v"]),
        ("Physical normal",tuple(plane["u"][(i+1)%3]*plane["v"][(i+2)%3]-
                                   plane["u"][(i+2)%3]*plane["v"][(i+1)%3] for i in range(3))),
        ("Potential range, V",(-maximum,maximum)),
        ("Display exclusion, m",record["exclusion_width_m"]),
        ("Max |E in plane| (V/m)",max_e),
        ("Max |E normal| (V/m)",max((abs(c["field_normal_v_m"]) for c in sampled),default=0.))
    ))
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>'+escape(title)+'</title>'
        '<style>'+_STYLE+' .static-layout{display:grid;grid-template-columns:minmax(0,2fr) minmax(240px,1fr);gap:18px}'
        ' .static-layout > div{min-width:0} svg{max-width:100%;height:auto}code{font-size:12px}'
        ' @media(max-width:800px){.static-layout{grid-template-columns:1fr}}</style></head>'
        '<body><header><div class="kicker">CONDUCTOR_SIM / STATIC OFF-AXIS RING E/V</div><h1>'+
        escape(title)+'</h1><p class="sub">A real 3D fixed-source field sampled on an authored 2D plane — not constrained particle motion.</p></header>'
        '<main><section class="card"><h2>Signed electric potential and projected electric-field directions</h2>'
        '<div class="static-layout"><div>'+''.join(svgs)+'</div><div><h2>Full physical vectors retained</h2>'
        '<p class="muted">Purple dashed rim: true 3D ring orthographic projection, not a charged line in the observation plane.'
        ' Dark arrows: in-plane components of computed E, normalized for readability (not literal magnitude).'
        ' Cell colours: signed potential relative to this chart only. Grey cells exclude the singular wire vicinity; they are not E=0.'
        ' A nonzero E normal to this plane is not discarded from the saved full 3D vectors.</p>'
        '<table><tbody>'+stats+'</tbody></table></div></div></section>'
        '<section class="card"><h2>Original physical source identities</h2><table><thead>'
        '<tr><th>Label</th><th>Stable ID</th><th>Physical source type</th></tr></thead><tbody>'+items+'</tbody></table>'
        '<p class="muted">All sources are fixed and nonresponding. No conductor induction, guide, time evolution, field-line integration or browser solver is represented.</p></section>'
        '</main><script type="application/json" id="ring-field-data">'+_safe_json_script(record)+'</script></body></html>')


def write_ring_field_slice(path, record, *, title="Off-axis charged ring static field"):
    """Write self-contained HTML and adjacent detached immutable-source JSON."""
    output=Path(path)
    html=render_ring_field_slice(record,title=title)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(html,encoding="utf-8")
    output.with_suffix(".json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return output
