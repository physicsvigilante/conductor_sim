"""M40c no-JavaScript presentation of independently solved static source positions.

This renders ONLY a live validated StaticSigmaComparison, never imported JSON.
It does not imply a path, elapsed time, force or grounding-current history.
"""
from __future__ import annotations

from html import escape

from .spherical_sigma_atlas import _CSS, _projection, _si, _xyz
from .spherical_sigma_relocation import StaticSigmaComparison, NO_CHRONOLOGY


def _scene(manifest: dict, key: str) -> dict:
    geom = manifest[key + "_voids"]
    rows = []
    for void, charges in zip(geom, manifest["gauss"]["cavities"]):
        if void["cavity_id"] != charges["cavity_id"]:
            raise ValueError("scene Gauss face and physical geometry diverge")
        positions = {s["source_id"]: s["position_m"] for s in void["source_positions"]}
        sources = [{"source_id": s["source_id"], "charge_c": s["charge_c"],
                    "position_m": positions[s["source_id"]]}
                   for s in charges["authored_sources"]]
        rows.append({"cavity_id": void["cavity_id"], "center_m": void["center_m"],
                     "radius_m": void["radius_m"], "sources": sources})
    return {"outer_sphere": manifest["outer_sphere"], "cavities": rows}


def render_static_sigma_comparison(comparison: StaticSigmaComparison) -> str:
    """HTML is derived only from fresh two-state M40a physical re-execution."""
    if type(comparison) is not StaticSigmaComparison:
        raise TypeError("viewer needs real native M40c comparison")
    m = comparison.as_dict()
    e = lambda v: escape(str(v), quote=True)
    baseline, revised = _scene(m, "baseline"), _scene(m, "revised")
    panels = []
    for index, (label, scene) in enumerate((("Baseline", baseline), ("Revised hypothetical", revised))):
        # M40b projection uses static IDs; two panels must namespace them.
        projection = _projection(scene).replace('id="m40b-', 'id="m40c-panel-{}-'.format(index))
        src_rows = "".join(
            "<tr><th><code>{}</code></th><td><code>{}</code></td><td>{}</td><td>{}</td></tr>".format(
                e(source["source_id"]),e(cav["cavity_id"]),
                e(_si(source["charge_c"],"C")),e(_xyz(source["position_m"])))
            for cav in scene["cavities"] for source in cav["sources"]
        )
        panels.append('<section class="geometry"><h2>'+e(label)+
                      ' — one static equilibrium</h2>'+projection+
                      '<div class="table-wrap"><table><thead><tr><th>Source ID</th>'+
                      '<th>Cavity</th><th>Signed q</th><th>True (x, y, z)</th></tr></thead><tbody>'+
                      src_rows+'</tbody></table></div></section>')
    diff_rows = []
    for cav in m["cavities"]:
        samples = []
        for j,s in enumerate(cav["samples"]):
            status = s["delta_status"]
            delta = ("Unresolved numerical difference" if status == "unresolved_numerical_difference"
                     else _si(s["delta_c_per_m2"],"C/m²"))
            samples.append('<tr id="m40c-{}-{}"><th scope="row">raw n = {}</th>'
                           '<td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>'.format(
                               e(cav["cavity_id"]),j,e(tuple(s["input_direction"])),
                               e(_si(s["sigma_baseline_c_per_m2"],"C/m²")),
                               e(_si(s["sigma_revised_c_per_m2"],"C/m²")),
                               e(delta),e(status.replace("_"," "))))
        diff_rows.append('<section class="cavity"><h3>Cavity <code>'+e(cav["cavity_id"])+
                         '</code></h3><p>Center: '+e(_xyz(cav["center_m"]))+
                         ' · radius: '+e(_si(cav["radius_m"],"m"))+
                         ' · unchanged whole-face Gauss Q: '+
                         e(_si(cav["whole_face_induced_charge_c"],"C"))+
                         '</p><div class="table-wrap"><table><thead><tr><th>Original physical ray</th>'
                         '<th>Baseline σ</th><th>Revised σ</th><th>Static difference</th>'
                         '<th>Qualification</th></tr></thead><tbody>'+"".join(samples)+
                         '</tbody></table></div></section>')
    relocated = m["relocation"]
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src &#39;none&#39;; '
        'style-src &#39;unsafe-inline&#39;; script-src &#39;none&#39;; '
        'connect-src &#39;none&#39;; img-src &#39;none&#39;; font-src &#39;none&#39;">'
        '<title>M40c static spherical-source comparison</title>'+_CSS+
        '</head><body><header><p id="m40c-role">TEACHER HYPOTHETICAL — '
        'ZERO NEW ORIGINAL JEE QUESTIONS SOLVED</p>'
        '<h1>How does a fixed source position change induced surface density?</h1>'
        '<p>Two separately solved static electrostatic equilibria, not motion.</p>'
        '</header><main><div class="notice" id="m40c-no-chronology">'+e(NO_CHRONOLOGY)+'</div>'
        '<div class="notice">Actual 3D spheres shown as XY orthographic projections '+
        'using one metre scale. Overlap in XY is not 3D contact or a cross-section. '
        'Every source depth z is shown in the true (x,y,z) tables.</div>'
        '<p>Repositioned named source: <code id="m40c-source">'+e(relocated["source_id"])+
        '</code>, cavity <code>'+e(relocated["cavity_id"])+
        '</code>; originally '+e(_xyz(relocated["original_position_m"]))+
        '; revised '+e(_xyz(relocated["revised_position_m"]))+'</p>'+
        "".join(panels)+'<h2>Native spherical inner-face point samples</h2>'+
        '<p>Same original raw ray in both states; display unit normals are not '
        'precision physics input. Signed differences below are computed from two '
        'independent native M40a evaluations, not interpolated area averages. '
        'Whole-face Gauss charge is separate from sparse point samples.</p>'+
        "".join(diff_rows)+'<p>Uniform outer spherical surface density is unchanged: '+
        e(_si(m["outer_uniform_sigma_c_per_m2"],"C/m²"))+'</p></main>'
        '<footer>Teacher-authored, geometry-qualified spherical cavities only. '
        'No arbitrary 3D field/potential map, source force, wire current, energy '
        'or time evolution. Exported HTML bytes and detached JSON are not '
        'cryptographically tied to historical teacher input.</footer></body></html>'
    )
