"""M21b: source-faithful static JEE electric-potential/field observations.

Reuse prescribed static field laws and M5's sampling/contouring renderer. These
are *static* observations. No solver, charge movement, conductor induction or
browser-side field evaluation is implemented here. Axis-only ring/disc physics
is never used to paint a two-dimensional field map.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from html import escape
import json
from math import fsum, hypot, isclose, isfinite
from pathlib import Path
import re

from .model import ElectrostaticProblem, PointCharge
from .prescribed_fields import (
    PrescribedField, FixedPointField, UniformElectricField,
    UniformRingAxisField, UniformDiskAxisField, UniformShellField,
    _dot, _sub, _mul, _unit, _number, _vector,
)
from .prescribed_viewer import _source_annotations
from .solver import SingularField, UnsupportedProblem
from .visualization import (
    SliceSpec, SamplingContext, SourceMask, sample_frame, contour_segments,
)
from .potential_display import potential_colour_scale


class _StaticFixedSolution:
    """Bridge existing FieldSolution *observation* methods, not a new solver."""
    conductor_results = ()
    def __init__(self, field):
        self.field = field
    def potential_at(self, point):
        return self.field.potential_at(point)
    def electric_field_at(self, point):
        return self.field.electric_field_at(point)


def _bind(field, annotations):
    if type(field) is not PrescribedField:
        raise TypeError("require a built-in PrescribedField (not subclass physics)")
    bound = _source_annotations(type("_Spec", (), {"field": field})(), annotations)
    if bound is None:
        raise ValueError("explicit stable source IDs and labels are required")
    return bound


def _json_native(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _fixed_plane(field, plane):
    """Certify point-source and uniform-E reflection symmetry of a 2D slice."""
    if not isinstance(plane, SliceSpec):
        raise TypeError("plane must be a physical SliceSpec")
    if not field.sources or len(field.sources) > 12:
        raise UnsupportedProblem("M21b map supports 1 to 12 fixed point/uniform-E sources")
    n = plane.normal
    span = max(plane.x_range_m[1]-plane.x_range_m[0],
               plane.y_range_m[1]-plane.y_range_m[0])
    for source in field.sources:
        if type(source) is FixedPointField:
            displacement = _sub(source.position_m, plane.origin_m)
            if abs(_dot(displacement, n)) > 1e-11*max(span, hypot(*displacement)):
                raise UnsupportedProblem("point charge is outside the certified physical plane")
        elif type(source) is UniformElectricField:
            if abs(_dot(source.field_v_m, n)) > 1e-11*max(1e-30, hypot(*source.field_v_m)):
                raise UnsupportedProblem("external uniform E has an unrepresented normal component")
        else:
            raise UnsupportedProblem("2D M21b field map supports only point sources and uniform E; rings/discs remain axis-only")
    return True


def sample_fixed_potential_map(field: PrescribedField, plane: SliceSpec, *,
                               source_annotations, source_mask_m: float,
                               highlight_potential_v: float = 0.,
                               line_steps: int = 240):
    """Build an honest one-frame M5 potential map from supported fixed sources.

    Charge exclusion circles are *presentation masks*, not physical radii.
    Grid, contour segments, lines and refinements come from the existing M5
    engine; the numerical solver still consists only of source observations.
    """
    annotations = _bind(field, source_annotations)
    _fixed_plane(field, plane)
    radius = _number(source_mask_m, "source_mask_m", positive=True)
    _number(highlight_potential_v, "highlight_potential_v")
    if radius > .03*min(plane.x_range_m[1]-plane.x_range_m[0],
                        plane.y_range_m[1]-plane.y_range_m[0]):
        raise ValueError("source mask may not obscure more than 3% of either view span")
    if type(line_steps) is not int or not 8 <= line_steps <= 700:
        raise ValueError("line_steps must be an integer in 8..700")
    points = tuple(s for s in field.sources if type(s) is FixedPointField)
    # The adapted problem is for existing M5 source markers and line seeds.
    # Its conductors are deliberately empty; no conductor backend is solved.
    display = ElectrostaticProblem(
        charges=tuple(PointCharge(s.charge_c, s.position_m) for s in points),
        conductors=(), permittivity_f_per_m=field.permittivity_f_m)
    context = SamplingContext(
        problem=display, solution=_StaticFixedSolution(field),
        backend="Analytical prescribed fixed sources (not conductors)",
        masks=tuple(SourceMask(s.position_m, radius, "display-exclusion")
                    for s in points),
        notes=("Physical 2D static V/E of fixed, nonresponding sources; no conductor induction.",
               "Dashed circles are arbitrary display exclusions, not physical particle sizes.",
               "Equipotential contours are triangulated finite-resolution observations, not analytic curves.",
               "No time evolution or force on a moving particle is implied.") +
              tuple(f"{a['label']} [{a['id']}]: {a['kind']}" for a in annotations),
        density_method="Not applicable: no conducting surfaces",
        conductor_results_method="Not applicable: prescribed fixed charge inventory")
    frame = sample_frame(
        context, plane, label="One static physical field state",
        line_steps=line_steps, profile_samples=9)
    highlighted = contour_segments(frame["grid"], [float(highlight_potential_v)],
                                   context, plane)[0]
    highlighted["highlight"] = True
    frame["contours"].append(highlighted)
    metadata = dict(
        schema="conductor_sim.fixed_field_observer.v1",
        mode="static physical 2D sampled field, not motion or conductor induction",
        field=field.as_dict(),
        source_annotations=annotations,
        source_exclusion_radius_m=radius,
        highlight_potential_v=float(highlight_potential_v),
        highlighted_segment_count=len(highlighted["segments_m"]),
        qualified_sources="FixedPointField and UniformElectricField, physically in-plane")
    bundle = dict(schema="conductor_sim.visualization.v1",
                  units=dict(length="m", potential="V", field="V/m",
                             density="not applicable"),
                  playback="one sampled static field, no browser ODE or physical time",
                  frames=[frame],
                  static_fixed_source_view=metadata,
                  potential_colour_scale=potential_colour_scale([frame]))
    result = _json_native(bundle)
    validate_fixed_potential_map(result)
    return result


def validate_fixed_potential_map(bundle):
    """Re-evaluate every stored grid observation and the highlighted contour."""
    from .prescribed_viewer_validation import _restore_field
    if (not isinstance(bundle, dict) or bundle.get("schema") != "conductor_sim.visualization.v1"
            or set(bundle) & {"motion_1d", "motion_2d", "prescribed_motion", "position_sweep"}):
        raise ValueError("static fixed-source bundle must not masquerade as motion")
    meta = bundle.get("static_fixed_source_view")
    if not isinstance(meta, dict) or meta.get("schema") != "conductor_sim.fixed_field_observer.v1":
        raise ValueError("missing fixed-source observation provenance")
    if len(bundle.get("frames", [])) != 1:
        raise ValueError("static field observer requires exactly one sampled state")
    field = _restore_field(meta.get("field"))
    ann = _bind(field, [(a["id"], a["label"]) for a in meta["source_annotations"]])
    if _json_native(ann) != meta["source_annotations"]:
        raise ValueError("source annotation inventory is inconsistent")
    frame = bundle["frames"][0]
    try:
        plane = SliceSpec(**frame["plane"])
        _fixed_plane(field, plane)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid fixed-source physical plane") from exc
    grid = frame["grid"]
    xs, ys = grid["x_m"], grid["y_m"]
    if (len(xs) != plane.resolution or len(ys) != plane.resolution
            or any(len(grid[k]) != len(ys) for k in ("potential_v","electric_field_v_per_m","region"))
            or len(frame.get("source_masks", ())) != sum(type(s) is FixedPointField for s in field.sources)):
        raise ValueError("static field grid, mask or resolution is inconsistent")
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            if len(grid["electric_field_v_per_m"][j]) != len(xs) or len(grid["potential_v"][j]) != len(xs):
                raise ValueError("static field grid row width changed")
            point = plane.world((x,y))
            masks = tuple(s.position_m for s in field.sources if type(s) is FixedPointField)
            masked = any(hypot(*_sub(p, point)) <= meta["source_exclusion_radius_m"] for p in masks)
            v, e = grid["potential_v"][j][i], grid["electric_field_v_per_m"][j][i]
            if masked:
                if v is not None or e is not None or grid["region"][j][i] != "source-mask":
                    raise ValueError("point source singularity display mask was modified")
            else:
                if v is None or e is None or grid["region"][j][i] != "vacuum":
                    raise ValueError("valid static field node was removed")
                expected_v = field.potential_at(point)
                expected_e = field.electric_field_at(point)
                if (not isclose(v,expected_v,rel_tol=5e-10,abs_tol=1e-9) or
                    any(not isclose(a,b,rel_tol=5e-10,abs_tol=1e-9) for a,b in zip(e,expected_e))):
                    raise ValueError("saved static V/E disagrees with authored sources")
    highlights = [c for c in frame["contours"] if c.get("highlight") is True]
    if len(highlights) != 1 or highlights[0]["level_v"] != meta["highlight_potential_v"]:
        raise ValueError("highlighted equipotential contour changed")
    for line in frame.get("field_lines", ()):
        for p in line["points_m"]:
            if any(not isfinite(v) for v in p):
                raise ValueError("nonfinite visual field line")
    return True


def _coord_point(origin, axis, coordinate):
    return tuple(origin[i] + axis[i]*coordinate for i in range(3))


def sample_static_profile(field: PrescribedField, *, source_annotations,
                          origin_m, axis, start_m: float, end_m: float,
                          sample_count: int = 241, coordinate="axial"):
    """One-dimensional observations, without artificial motion or E projection.

    At ideal surface jumps/singularities the E observation is None. The
    potential may remain defined. A nonzero transverse E is retained in the
    output, especially for the workbook's horizontally displaced ring probe.
    """
    ann = _bind(field, source_annotations)
    origin = _vector(origin_m, "origin")
    direction = _unit(axis, "axis")
    _number(start_m, "profile start"); _number(end_m, "profile end")
    if not start_m < end_m:
        raise ValueError("profile end must be greater than start")
    if type(sample_count) is not int or not 21 <= sample_count <= 501 or sample_count%2 != 1:
        raise ValueError("profile needs 21..501 odd samples")
    if coordinate not in ("radial","axial"):
        raise ValueError("profile coordinate must be radial or axial")
    if coordinate == "radial":
        if start_m < 0 or len(field.sources) != 1 or type(field.sources[0]) is not UniformShellField:
            raise UnsupportedProblem("M21b radial surface profile requires one centred fixed uniform shell")
        if origin != field.sources[0].center_m:
            raise UnsupportedProblem("radial profile must start at shell centre")
    rows=[]
    for j in range(sample_count):
        s = float(start_m+(end_m-start_m)*j/(sample_count-1))
        point = _coord_point(origin,direction,s)
        try:
            potential = field.potential_at(point)
        except SingularField:
            potential = None
        try:
            E = field.electric_field_at(point)
        except SingularField:
            E = None
        if E is not None:
            parallel = _dot(E,direction)
            transverse = hypot(*_sub(E, _mul(direction,parallel)))
        else:
            parallel,transverse = None,None
        rows.append(dict(coordinate_m=s, point_m=list(point),
                         potential_v=potential, electric_field_v_m=list(E) if E is not None else None,
                         field_parallel_v_m=parallel,
                         field_transverse_abs_v_m=transverse,
                         field_undefined=E is None))
    return _json_native(dict(schema="conductor_sim.fixed_profile.v1",
               mode="static spatial samples; no time integration",
               coordinate=coordinate, field=field.as_dict(), source_annotations=ann,
               origin_m=origin, axis=direction, rows=rows))


def endpoint_field_work(field: PrescribedField, *, source_annotations,
                        probe_charge_c: float, start_m, end_m):
    """The exact electrostatic *field* work between two authored endpoints.

    No mechanical constraint, trajectory, initial mass or transit time implied.
    """
    ann = _bind(field, source_annotations)
    q = _number(probe_charge_c, "probe charge", nonzero=True)
    start,end = _vector(start_m, "start_m"), _vector(end_m,"end_m")
    if start == end:
        raise ValueError("work endpoints must differ")
    components=[]
    for entry,source in zip(ann,field.sources):
        try:
            va = source.potential_at(start, permittivity_f_m=field.permittivity_f_m)
            vb = source.potential_at(end, permittivity_f_m=field.permittivity_f_m)
        except SingularField as err:
            raise UnsupportedProblem("work endpoint lies on an ideal fixed-source singularity") from err
        components.append(dict(id=entry["id"],label=entry["label"],kind=entry["kind"],
                               start_potential_v=va,end_potential_v=vb,
                               field_work_j=q*(va-vb)))
    va = field.potential_at(start)
    vb = field.potential_at(end)
    work = q*fsum(x["start_potential_v"]-x["end_potential_v"] for x in components)
    if not isclose(work,q*(va-vb),rel_tol=5e-12,abs_tol=1e-11):
        raise ValueError("work decomposition contradicts source potentials")
    EA, EB = field.electric_field_at(start),field.electric_field_at(end)
    return _json_native(dict(schema="conductor_sim.fixed_endpoint_work.v1",
        mode="static endpoint work only; no guide, path or elapsed time",
        start_m=start, end_m=end, probe_charge_c=q,field=field.as_dict(),
        source_annotations=ann,
        start_potential_v=va,end_potential_v=vb,
        field_work_j=work, external_quasistatic_work_j=-work,
        source_work_components=components,
        start_electric_field_v_m=EA,end_electric_field_v_m=EB,
        caveat="Conservative static E: W_field=q(V_start-V_end); nonzero transverse force is not discarded or dynamically constrained."))


def _plot_svg(rows, key, *, label, color, surface_coordinate=None):
    """Static sampled polyline with true gaps at None-valued field entries."""
    W,H=760,290
    left,right,top,bottom=76,24,22,52
    values=[r[key] for r in rows if r[key] is not None]
    if not values:
        raise ValueError("profile has no defined observations")
    ymin,ymax=min(values+[0]),max(values+[0])
    if ymin==ymax:
        ymin-=1.;ymax+=1.
    else:
        pad=.08*(ymax-ymin);ymin-=pad;ymax+=pad
    xmin,xmax=rows[0]["coordinate_m"],rows[-1]["coordinate_m"]
    X=lambda x:left+(W-left-right)*(x-xmin)/(xmax-xmin)
    Y=lambda y:H-bottom-(H-top-bottom)*(y-ymin)/(ymax-ymin)
    chunks,one=[],[]
    for r in rows:
        v=r[key]
        if v is None:
            if len(one)>1:chunks.append(one)
            one=[]
        else:
            one.append((X(r["coordinate_m"]),Y(v)))
    if len(one)>1:chunks.append(one)
    paths="".join('<polyline fill="none" stroke="'+color+'" stroke-width="2.8" points="'+
                  " ".join(f"{x:.3f},{y:.3f}" for x,y in chunk)+'"/>' for chunk in chunks)
    grid=''.join(
        '<line x1="'+str(left)+'" y1="'+f"{Y(ymin+(ymax-ymin)*i/4):.2f}"+
        '" x2="'+str(W-right)+'" y2="'+f"{Y(ymin+(ymax-ymin)*i/4):.2f}"+
        '" stroke="#dbe5ef"/><text x="'+str(left-10)+'" y="'+
        f"{Y(ymin+(ymax-ymin)*i/4)+4:.2f}"+'" text-anchor="end" font-size="12">'+
        escape(f"{ymin+(ymax-ymin)*i/4:.3g}")+'</text>'
        for i in range(5))
    ticks=''.join('<text x="'+f"{X(xmin+(xmax-xmin)*i/4):.2f}"+
                  '" y="'+str(H-28)+'" text-anchor="middle" font-size="12">'+
                  escape(f"{xmin+(xmax-xmin)*i/4:.3g}")+'</text>' for i in range(5))
    marker=("" if surface_coordinate is None else
            '<line id="physical-surface" x1="'+f"{X(surface_coordinate):.2f}"+
            '" x2="'+f"{X(surface_coordinate):.2f}"+'" y1="'+str(top)+'" y2="'+str(H-bottom)+
            '" stroke="#d06d26" stroke-width="2" stroke-dasharray="5,5"/>'+
            '<text x="'+f"{X(surface_coordinate)+5:.2f}"+'" y="'+str(top+15)+
            '" font-size="12" fill="#96551a">r = R (E undefined)</text>')
    return ('<svg class="plot" viewBox="0 0 760 290" role="img" aria-label="'+escape(label,quote=True)+
            '" xmlns="http://www.w3.org/2000/svg"><rect width="760" height="290" fill="#fbfdff"/>'+
            grid+'<line x1="'+str(left)+'" y1="'+str(top)+'" x2="'+str(left)+
            '" y2="'+str(H-bottom)+'" stroke="#556a7d"/><line x1="'+str(left)+
            '" y1="'+str(H-bottom)+'" x2="'+str(W-right)+'" y2="'+str(H-bottom)+
            '" stroke="#556a7d"/>'+paths+marker+ticks+
            '<text x="380" y="282" text-anchor="middle" font-size="13">Physical coordinate (m)</text>'+
            '<text x="380" y="15" text-anchor="middle" font-size="14" font-weight="600">'+
            escape(label)+'</text></svg>')


_STYLE = """body{margin:0;background:#f3f6fa;color:#183047;font:15px/1.55 system-ui,sans-serif}
header{background:#183047;color:white;padding:24px max(16px,calc((100vw - 1150px)/2))}
main{max-width:1150px;margin:20px auto;padding:0 16px}
h1{font-size:25px;margin:4px 0}h2{font-size:18px;margin:0 0 14px}
.sub{color:#d6e3ef}.kicker{font-size:12px;letter-spacing:.1em}
.card{background:white;border:1px solid #cbd7e4;border-radius:8px;padding:18px;margin:16px 0}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.plot{width:100%;height:auto;border:1px solid #d7e4ee}
.data{font:13px/1.7 ui-monospace,monospace;white-space:pre-wrap}
table{width:100%;border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #e1e9f1;text-align:left}
p{margin:8px 0}.muted{color:#52677b;font-size:13px}
@media(max-width:850px){.grid{grid-template-columns:1fr}header{padding:18px}h1{font-size:21px}}
"""


def _safe_json_script(value):
    return json.dumps(value,allow_nan=False,separators=(",",":")).replace("&","\\u0026").replace("<","\\u003c").replace(">","\\u003e")


def render_static_profile(profile, *, title, source_note, surface_coordinate=None,
                          work=None, geometry_svg=""):
    """Lightweight non-animated SVG of *sampled* E/V; no browser physics."""
    if profile.get("schema")!="conductor_sim.fixed_profile.v1":
        raise ValueError("invalid fixed profile")
    rows=profile["rows"]
    if work is not None and work.get("schema")!="conductor_sim.fixed_endpoint_work.v1":
        raise ValueError("invalid endpoint-work record")
    chart1=_plot_svg(rows,"field_parallel_v_m",label="E along sampled direction (V/m)",
                     color="#185e9e",surface_coordinate=surface_coordinate)
    chart2=_plot_svg(rows,"potential_v",label="Electrostatic potential V (volts)",
                     color="#aa5d24",surface_coordinate=surface_coordinate)
    note=("No time evolution. Missing E samples are left blank, not set to zero."
          " Profiles are sampled at the saved physical coordinates.")
    work_content=""
    if work:
        cells="".join("<tr><td>"+escape(p["label"])+" ["+escape(p["id"])+"]</td><td>"+
                      escape(f"{p['field_work_j']:.8g}")+" J</td></tr>"
                      for p in work["source_work_components"])
        work_content=('<section class="card" id="work-summary"><h2>Work by the electrostatic field</h2>'+
          '<p><strong>W = q(V_A - V_B) = '+escape(f'{work["field_work_j"]:.8g}')+
          ' J</strong>; external quasistatic work = '+escape(f'{work["external_quasistatic_work_j"]:.8g}')+
          ' J.</p><table><thead><tr><th>Physical source ID</th><th>Field-work contribution</th></tr></thead>'+
          '<tbody>'+cells+'</tbody></table><p class="muted">'+escape(work["caveat"])+
          '</p></section>')
    metadata=dict(schema="conductor_sim.jee_static_explorer.v1",
                  profile=profile, endpoint_work=work,
                  physical_surface_coordinate_m=surface_coordinate)
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'+
            '<meta name="viewport" content="width=device-width,initial-scale=1"><title>'+
            escape(title)+'</title><style>'+_STYLE+'</style></head><body>'+
            '<header><div class="kicker">CONDUCTOR_SIM / JEE STATIC SOURCE OBSERVER</div><h1>'+
            escape(title)+'</h1><p class="sub">'+escape(source_note)+'</p></header><main>'+
            ('<section class="card" id="source-geometry"><h2>Physical source arrangement (schematic)</h2>'+
             geometry_svg+'</section>' if geometry_svg else '')+
            '<section class="card"><h2>Independent static electric-field observations</h2>'+
            '<div class="grid">'+chart1+chart2+'</div><p class="muted">'+note+'</p></section>'+
            work_content+'<section class="card"><h2>Source identities and qualifications</h2>'+
            '<p class="muted">These values are from immutable externally prescribed charge distributions; no conductor induction, collision, guide force or browser ODE is inferred.</p><div class="data">'+
            escape("\n".join(f"{a['label']} [{a['id']}] — {a['kind']}" for a in profile["source_annotations"]))+
            '</div></section></main><script id="observation-data" type="application/json">'+
            _safe_json_script(metadata)+'</script></body></html>')


def write_observation(output, html, audit):
    """Write self-contained offline HTML and detached reproducible JSON."""
    path=Path(output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(html,encoding="utf-8")
    sidecar=path.with_suffix(".json")
    sidecar.write_text(json.dumps(audit,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return path,sidecar
