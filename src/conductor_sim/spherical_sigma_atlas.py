"""M40b: honest offline XY projection and discrete M40a spherical sigma atlas.

No novel field law, 3D E/V map, time integration or topology-only geometry.
All observations come from freshly validated M40a physical solutions.
"""
from __future__ import annotations

from dataclasses import replace
from html import escape
import json
from math import atan2, cos, degrees, hypot, isfinite, pi, sin

from .model import FixedNetCharge, Grounded
from .spherical_multicavity import SphericalSigmaSolution

SCHEMA = "conductor_sim.m40b_spherical_surface_atlas.v1"
ROLE = "teacher_hypothetical_spherical_multicavity_only"
WARNING = (
    "ACTUAL 3D SPHERES SHOWN AS XY ORTHOGRAPHIC PROJECTIONS; z IS LISTED "
    "SEPARATELY. XY OVERLAP DOES NOT PROVE 3D CONTACT OR INTERSECTION."
)
LIMITS = (
    "Signed tiles are distinct surface POINT SAMPLES, not continuous "
    "fields, area averages, integrated fluxes, or time animation. "
    "No irregular cavities, source forces, E(x), V(x), ground-wire "
    "currents, newly solved source JEE problems, or 3D field maps."
)


def _canonical(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":"))


def _pair(isolated, grounded):
    """Refuse altered, mixed or unrelated states before making visual claims."""
    if type(isolated) is not SphericalSigmaSolution or type(grounded) is not SphericalSigmaSolution:
        raise TypeError("M40b needs actual native M40a spherical sigma solutions")
    a, b = isolated.model.gauss_assembly, grounded.model.gauss_assembly
    if type(a.constraint) is not FixedNetCharge or type(b.constraint) is not Grounded:
        raise ValueError("M40b needs isolated then independently grounded equilibria")
    # Invoke the native re-admission and saved-ledger integrity checks.
    one = isolated.as_dict()
    two = grounded.as_dict()
    if (replace(a, constraint=Grounded()) != b
            or isolated.model.geometry != grounded.model.geometry
            or _canonical(one["voids"]) != _canonical(two["voids"])
            or _canonical(one["outer_sphere"]) != _canonical(two["outer_sphere"])
            or one["source_role"] != ROLE or two["source_role"] != ROLE):
        raise ValueError("paired scenes must have identical original sources and actual 3D geometry")
    if (one["original_workbook_items_newly_solved"] != 0
            or two["original_workbook_items_newly_solved"] != 0):
        raise ValueError("no original workbook solution attribution is qualified")
    return one, two


def _rays(directions):
    if directions is None:
        # Four colatitude rows by six longitude columns: 24 genuine point samples.
        return tuple((sin(pi * (i + .5) / 4) * cos(2 * pi * (j + .5) / 6),
                      sin(pi * (i + .5) / 4) * sin(2 * pi * (j + .5) / 6),
                      cos(pi * (i + .5) / 4))
                     for i in range(4) for j in range(6))
    if type(directions) not in (tuple, list) or not 1 <= len(directions) <= 48:
        raise ValueError("at most 48 bounded explicit surface rays are admitted")
    if any(type(n) not in (list, tuple) or len(n) != 3 for n in directions):
        raise ValueError("every spherical atlas query needs three raw components")
    # M40a.as_dict below is authoritative for validating *each raw component*.
    return tuple(tuple(n) for n in directions)


def surface_atlas_manifest(isolated, grounded, *, directions=None):
    """Export paired real surface-probe evidence without uncomputed interpolation."""
    ia, ga = _pair(isolated, grounded)
    normals = _rays(directions)
    rows = []
    witnesses = {"isolated": [], "grounded": []}
    for void, cav in zip(ia["voids"], ia["gauss"]["cavities"]):
        ident = void["cavity_id"]
        probes = [(ident, ray) for ray in normals]
        ar = isolated.as_dict(probes)
        br = grounded.as_dict(probes)
        witnesses["isolated"].append(ar)
        witnesses["grounded"].append(br)
        samples = []
        for left, right in zip(ar["samples"], br["samples"]):
            if (_canonical(left["input_direction"]) != _canonical(right["input_direction"])
                    or _canonical(left["unit_normal"]) != _canonical(right["unit_normal"])
                    or _canonical(left["sigma_c_per_m2"]) != _canonical(right["sigma_c_per_m2"])):
                raise ValueError("inner sigma cannot differ with only isolated/grounded outer constraint")
            raw = left["input_direction"]
            theta = degrees(atan2(hypot(raw[0], raw[1]), raw[2]))
            phi = degrees(atan2(raw[1], raw[0])) % 360.0
            samples.append({
                "input_direction": raw,
                "unit_normal": left["unit_normal"],  # rounded display ONLY
                "theta_deg_display": theta,
                "phi_deg_display": phi,
                "isolated_sigma_c_per_m2": left["sigma_c_per_m2"],
                "grounded_sigma_c_per_m2": right["sigma_c_per_m2"],
            })
        lookup = {p["source_id"]: p["position_m"] for p in void["source_positions"]}
        sources = [{"source_id": s["source_id"], "charge_c": s["charge_c"],
                    "position_m": lookup[s["source_id"]]}
                   for s in cav["authored_sources"]]
        rows.append({
            "cavity_id": ident, "center_m": void["center_m"],
            "radius_m": void["radius_m"], "sources": sources,
            "source_total_charge_c": cav["source_charge_sum_c"],
            "whole_inner_face_charge_c": cav["induced_inner_face_total_charge_c"],
            "samples": samples,
        })
    outer_ray = (1.0, 0.0, 0.0)
    oa = isolated.as_dict((("outer", outer_ray),))
    ob = grounded.as_dict((("outer", outer_ray),))
    witnesses["isolated"].append(oa)
    witnesses["grounded"].append(ob)
    if ob["samples"][0]["sigma_c_per_m2"] != 0.0:
        raise ValueError("the grounded spherical outer surface has zero sigma")
    return {
        "schema": SCHEMA,
        "source_role": ROLE,
        "original_workbook_items_newly_solved": 0,
        "geometry_qualified_spherical_only": True,
        "visual_is_xy_orthographic_projection": True,
        "projection_is_2d_field_or_cross_section": False,
        "sample_tiles_are_point_observations_only": True,
        "time_evolution_defined": False,
        "irregular_3d_fields_qualified": False,
        "warning": WARNING, "limitations": LIMITS,
        "units": {"coordinate": "m", "radius": "m", "charge": "C",
                  "surface_density": "C/m^2"},
        "outer_sphere": ia["outer_sphere"],
        "cavities": rows,
        "outer": {
            "isolated_whole_face_charge_c": ia["gauss"]["outer_face_total_charge_c"],
            "grounded_whole_face_charge_c": ga["gauss"]["outer_face_total_charge_c"],
            "isolated_uniform_sigma_c_per_m2": oa["samples"][0]["sigma_c_per_m2"],
            "grounded_uniform_sigma_c_per_m2": ob["samples"][0]["sigma_c_per_m2"],
        },
        "native_m40a_v2_witness_records": witnesses,
    }


_CSS = """
<style>
:root{color-scheme:light;font:16px/1.48 system-ui,-apple-system,'Segoe UI',sans-serif;
 color:#19334a;background:#f5f8fb}
*{box-sizing:border-box}body{margin:0;min-width:300px}
header{padding:1.8rem max(1rem,calc((100vw - 1100px)/2));background:#15354d;color:white}
h1{font-size:clamp(1.5rem,3vw,2.2rem);margin:.3rem 0}h2{font-size:1.35rem}
h3{font-size:1.08rem}main,footer{max-width:1100px;margin:1.25rem auto;padding:0 1rem}
.notice{background:#fff3db;border-left:4px solid #c18a18;padding:.85rem;margin:.9rem 0}
.geometry{background:white;border:1px solid #c4d5df;border-radius:12px;padding:1rem}
svg{display:block;max-width:100%;height:auto;margin:auto;background:#eaf3f9}
.table-wrap,.atlas-wrap{overflow-x:auto;border:1px solid #cad6df;border-radius:10px;background:white}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:.5rem .65rem;text-align:left;border-bottom:1px solid #dfe7ee;vertical-align:top}
th{background:#eaf1f7}thead th{background:#244d68;color:white}
pre,code{overflow-wrap:anywhere;font-size:.9em}.note{color:#44586c}
.cavity{padding:1rem;margin:1.5rem 0;border:1px solid #c4d5df;border-radius:12px;background:white}
.atlas{display:grid;grid-template-columns:repeat(6,minmax(135px,1fr));gap:.35rem;min-width:825px;padding:.55rem}
.tile{padding:.55rem;border:1px solid #c4d5df;border-radius:7px;font-variant-numeric:tabular-nums}
.positive{background:#dcecfb}.negative{background:#fff0d6}.zero{background:#f0f2f4}
.tile small{display:block}.tile strong{display:block;margin:.3rem 0;font-size:.94rem}
.tag{display:inline-block;border:1px solid #a9bcca;border-radius:8px;padding:.1rem .45rem}
footer{border-top:1px solid #cfdae4;padding-top:1rem;margin-bottom:2rem}
@media(max-width:600px){header{padding:1.2rem 1rem}main{padding:0 .7rem}}
</style>
"""


def _si(value, unit):
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError("cannot show Boolean/nonfinite physical SI")
    if value == 0:
        return "0 " + unit
    return ("+" if value > 0 else "−") + repr(abs(value)) + " " + unit


def _xyz(v):
    return "(" + ", ".join(repr(x) for x in v) + ") m"


def _projection(manifest):
    outer = manifest["outer_sphere"]
    cx, cy, _ = outer["center_m"]
    R = outer["radius_m"]
    scale = 375. / R
    def xy(x, y):
        return 425. + (x - cx) * scale, 425. - (y - cy) * scale
    pieces = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 850 850" '
        'role="img" aria-label="Real spherical metal and cavities projected orthographically onto world XY; depth z is not shown">',
        '<title>World XY orthographic projection: +x right, +y up, depth z in tables</title>',
        '<circle cx="425" cy="425" r="375" fill="#dae8f1" stroke="#285a7d" stroke-width="3"/>',
        '<line x1="425" y1="425" x2="820" y2="425" stroke="#2b526e" stroke-width="2"/>',
        '<line x1="425" y1="425" x2="425" y2="30" stroke="#2b526e" stroke-width="2"/>',
        '<text x="780" y="445" fill="#163d59">+x</text>',
        '<text x="435" y="50" fill="#163d59">+y</text>',
    ]
    for i, cav in enumerate(manifest["cavities"]):
        x, y = xy(cav["center_m"][0], cav["center_m"][1])
        radius = cav["radius_m"] * scale
        pieces.append('<circle id="m40b-void-{}" cx="{}" cy="{}" r="{}" '
                      'fill="#ffffff" stroke="#2e7996" stroke-width="2">'
                      '<title>{}</title></circle>'.format(
                          i, repr(x), repr(y), repr(radius),
                          escape("Cavity "+cav["cavity_id"]+"; 3D center "+_xyz(cav["center_m"]),quote=True)))
        for j, src in enumerate(cav["sources"]):
            px, py = xy(src["position_m"][0], src["position_m"][1])
            color = "#1d5c99" if src["charge_c"] >= 0 else "#9b4e10"
            pieces.append('<circle id="m40b-source-{}-{}" cx="{}" cy="{}" r="6" '
                          'fill="{}" stroke="#fff" stroke-width="1">'
                          '<title>{}</title></circle>'.format(
                              i,j,repr(px),repr(py),color,
                              escape(src["source_id"]+" "+_si(src["charge_c"],"C")+
                                     "; true 3D position "+_xyz(src["position_m"]),quote=True)))
    pieces.append('</svg>')
    return "".join(pieces)


def render_surface_atlas(isolated, grounded, *, directions=None):
    """Render only from live reverified typed physics, never arbitrary JSON."""
    m = surface_atlas_manifest(isolated, grounded, directions=directions)
    e = lambda x: escape(str(x), quote=True)
    sections = []
    for i, cav in enumerate(m["cavities"]):
        source_rows = "".join(
            '<tr><th scope="row"><code>{}</code></th><td>{}</td><td>{}</td></tr>'.format(
                e(s["source_id"]), e(_si(s["charge_c"],"C")),e(_xyz(s["position_m"])))
            for s in cav["sources"]
        ) or '<tr><td colspan="3">No point sources inside this cavity</td></tr>'
        tiles = []
        for j, s in enumerate(cav["samples"]):
            v = s["isolated_sigma_c_per_m2"]
            sign = "positive" if v > 0 else "negative" if v < 0 else "zero"
            angle = "θ≈{:.3f}°, φ≈{:.3f}°".format(
                s["theta_deg_display"],s["phi_deg_display"])
            raw = "(" + ", ".join(repr(x) for x in s["input_direction"]) + ")"
            tiles.append(
                '<div class="tile {}" id="m40b-tile-{}-{}" '
                'aria-label="{}; raw physical ray {}; isolated and grounded sigma {}">'
                '<small>{}</small><small>raw n={}</small>'
                '<strong>{}</strong><small>Grounded: {}</small></div>'.format(
                    sign,i,j,e(angle),e(raw),e(_si(v,"C/m²")),e(angle),e(raw),
                    e(_si(v,"C/m²")),e(_si(s["grounded_sigma_c_per_m2"],"C/m²"))))
        tag = ("empty cavity" if not cav["sources"] else
               "signed-neutral sources, possibly nonzero local σ"
               if cav["source_total_charge_c"] == 0 else "net-charged cavity")
        sections.append(
            '<section class="cavity" id="m40b-cavity-{}">'
            '<h3>Cavity <code>{}</code> <small class="tag">{}</small></h3>'
            '<p>True 3D center: {} · radius: {} · whole-face Gauss Q: {} '
            '(not an atlas-grid sum)</p>'
            '<div class="table-wrap"><table><thead><tr><th>Source identity</th>'
            '<th>Signed q</th><th>True (x, y, z) position</th></tr></thead>'
            '<tbody>{}</tbody></table></div>'
            '<p class="note">Each tile is one native M40a point sample. θ is '
            'colatitude from +z, φ longitude from +x toward +y; approximate '
            'angle labels are display-only. Original raw n sets the physics. '
            'Blue=positive, amber=negative, grey=zero; not area averages.</p>'
            '<div class="atlas-wrap"><div class="atlas" role="group" '
            'aria-label="Discrete spherical surface point samples for cavity {}">{}</div></div>'
            '</section>'.format(
                i,e(cav["cavity_id"]),e(tag),e(_xyz(cav["center_m"])),
                e(_si(cav["radius_m"],"m")),e(_si(cav["whole_inner_face_charge_c"],"C")),
                source_rows,e(cav["cavity_id"]),"".join(tiles)))
    o = m["outer"]
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src &#39;none&#39;; '
        'style-src &#39;unsafe-inline&#39;; script-src &#39;none&#39;; '
        'connect-src &#39;none&#39;; img-src &#39;none&#39;; font-src &#39;none&#39;">'
        '<title>M40b real spherical sigma teaching atlas</title>'+_CSS+
        '</head><body><header><p id="m40b-role">TEACHER HYPOTHETICAL · '
        '0 newly solved original JEE workbook questions</p>'
        '<h1>Local induced charge · spherical cavities</h1>'
        '<p>Actual M40a physics, two independent static equilibria. '
        'No time evolution or general 3D field map.</p></header>'
        '<main><div class="notice" id="m40b-warning">'+e(WARNING)+'</div>'
        '<div class="notice" id="m40b-limits">'+e(LIMITS)+'</div>'
        '<section class="geometry"><h2>Real geometry · XY projection</h2>'
        '<p>One physical metre scale; +x right, +y up. z remains physical '
        'and is listed for every cavity and source. Projection overlap does '
        'not prove 3D overlap.</p>'+_projection(m)+'</section>'
        '<section><h2>Uniform outer sphere · Gauss law</h2>'
        '<p>Outer sphere center: '+e(_xyz(m["outer_sphere"]["center_m"]))+
        '; radius: '+e(_si(m["outer_sphere"]["radius_m"],"m"))+'</p>'
        '<div class="table-wrap"><table><thead><tr><th>Static state</th>'
        '<th>Whole outer face Q</th><th>Uniform outer σ</th></tr></thead><tbody>'
        '<tr><th>Isolated</th><td id="m40b-q-outer-isolated">'+
        e(_si(o["isolated_whole_face_charge_c"],"C"))+'</td><td>'+
        e(_si(o["isolated_uniform_sigma_c_per_m2"],"C/m²"))+'</td></tr>'
        '<tr><th>Grounded</th><td id="m40b-q-outer-grounded">'+
        e(_si(o["grounded_whole_face_charge_c"],"C"))+'</td><td>'+
        e(_si(o["grounded_uniform_sigma_c_per_m2"],"C/m²"))+
        '</td></tr></tbody></table></div></section>'
        '<section><h2>Each real inner spherical surface</h2>'+
        "".join(sections)+'</section>'
        '</main><footer><p>Qualified M40a surface observations only; not '
        'external fields, image-charge source forces, moving charges or '
        'current from grounding. External edits to self-consistent records '
        'are not historically authenticated without a trusted source anchor.</p>'
        '</footer></body></html>'
    )
