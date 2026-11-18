"""M37b: dependency-free offline SVG playback of Python-computed Kelvin snapshots.

The browser selects an actual stored *prescribed* time, never integrates
Newton's law, recalculates induction/current or interpolates physical fields.
"""
from __future__ import annotations

from html import escape
import json
from math import isfinite

from .prescribed_ground_wire import PrescribedSphereTrace, SCHEMA, sample_prescribed_sphere_radial


def _fmt(value):
    if not isfinite(value):
        raise ValueError("nonfinite current viewer value")
    return format(value, ".11g")


def viewer_manifest(trace: PrescribedSphereTrace, source: dict) -> dict:
    """Bind original-source symbolic statements separately from teacher values."""
    if type(trace) is not PrescribedSphereTrace or not trace.samples:
        raise TypeError("viewer needs a real nonempty prescribed Kelvin trace")
    # A caller can construct an arbitrary dataclass. A scientific viewer must
    # not attach a trusted scene/graph to forged saved state observations.
    # Re-run the existing native static Kelvin observer at every authored
    # prescribed time and require exact typed evidence before drawing.
    expected = sample_prescribed_sphere_radial(trace.spec)
    if trace.samples != expected.samples:
        raise ValueError("viewer refuses altered prescribed Kelvin current/charge/time samples")
    if type(source) is not dict or source.get("role") not in ("original_workbook_teacher_example", "teacher_hypothetical"):
        raise ValueError("source provenance role is required")
    if type(source.get("statement")) is not str or not source["statement"].strip():
        raise ValueError("source statement cannot be empty")
    if type(source.get("numeric_attribution")) is not str or not source["numeric_attribution"].strip():
        raise ValueError("teacher numeric attribution is required")
    # Normalize the complete physical payload to its actual JSON wire form.
    # A Python tuple must not appear to disagree with a saved/browser array.
    physics = json.loads(json.dumps(trace.as_dict(),allow_nan=False))
    if physics["schema"] != SCHEMA:
        raise ValueError("wrong physical prescribed trace")
    sphere = trace.spec.problem.conductors[0]
    max_d = max(row.radial_distance_m for row in trace.samples)
    # A fixed proportional world-length -> display map; no animation
    # exaggeration of the radial path, no per-frame changing scale.
    scale = min(96. / sphere.radius_m, 515. / max_d)
    x0, y0 = 129., 169.
    start_current = max(abs(row.wire_current_sphere_to_ground_a) for row in trace.samples)
    if start_current == 0 or not isfinite(start_current):
        raise ValueError("nonzero physical current is needed for plotted approach")
    width = 487.
    left, top, chart_height = 61., 25., 210.
    rows = []
    n = len(trace.samples)
    for row in trace.samples:
        index = row.index
        current = row.wire_current_sphere_to_ground_a
        chart_x = left+width*index/(n-1)
        # Bipolar plot centred on 140; all values retain physical signs.
        chart_y = 139. - (current/start_current)*100.
        rows.append({
            "index": index,
            "charge_x_px": x0 + scale*row.radial_distance_m,
            "chart_x_px": chart_x,
            "chart_y_px": chart_y,
            "time_label": _fmt(row.time_s)+" s",
            "distance_label": _fmt(row.radial_distance_m)+" m",
            "induced_charge_label": _fmt(row.induced_sphere_net_charge_c)+" C",
            "wire_current_label": _fmt(current)+" A",
            "opposite_current_label": _fmt(row.wire_current_ground_to_sphere_a)+" A",
        })
    return {
        "schema": "conductor_sim.prescribed_ground_wire_viewer.v1",
        "source": dict(source),
        "trace": physics,
        "layout": {
            "fixed_scene": True, "radial_schematic": True,
            "sphere_cx_px": x0, "sphere_cy_px": y0,
            "sphere_radius_px": scale*sphere.radius_m,
            "charge_cy_px": y0, "px_per_meter": scale,
            "chart_center_y_px": 139., "chart_current_peak_abs_a": start_current,
            "scene_positions_precomputed": True,
        },
        "readouts": rows,
        "display_contract": (
            "Only precomputed Python Kelvin equilibrium observations and externally prescribed "
            "constant-speed timestamps; display is a fixed-radial schematic, NOT an M16a "
            "trajectory or browser electromagnetic calculation."
        ),
    }


def render_prescribed_ground_wire(trace: PrescribedSphereTrace, *, source: dict) -> str:
    """Self-contained HTML; no network, physics computed by Python only."""
    m = viewer_manifest(trace, source)
    physics, meta, layout = m["trace"],m["source"],m["layout"]
    original = meta["role"] == "original_workbook_teacher_example"
    header=("PRINTED WORKBOOK p8 · Multiple Correct Q21" if original else
            "TEACHER HYPOTHETICAL · same qualified grounded-sphere physics")
    role_text=(
        "SOURCE-GIVEN: positive q approaches a grounded metallic sphere of radius R "
        "at externally maintained constant small speed v. Workbook answer key: "
        "options (1) and (3), conventional sphere→ground current increasing."
        if original else meta["statement"]
    )
    teacher=(
        "ALL numbers (q, R, d₀, v, duration, ε) displayed here are TEACHER-CHOSEN "
        "illustrations, NOT printed workbook numerical givens."
    )
    rows=m["readouts"]
    first=rows[0]
    r=layout["sphere_radius_px"]
    cx=layout["sphere_cx_px"];cy=layout["sphere_cy_px"]
    start=rows[0]["charge_x_px"]
    polyline=" ".join(f'{_fmt(row["chart_x_px"])},{_fmt(row["chart_y_px"])}' for row in rows)
    dataset=json.dumps(m,ensure_ascii=False,allow_nan=False,separators=(",",":")).replace("<","\\u003c")
    # All source text is HTML-escaped; the JSON script also escapes '<' to
    # prevent user-authored source names from becoming an active </script>.
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Grounded sphere · prescribed speed · source Q21</title>
<style>
:root {{ color-scheme:light; font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Arial; color:#182839; background:#f4f7fa; }}
* {{ box-sizing:border-box }} body {{margin:0}} main {{max-width:1140px;padding:18px;margin:auto}}
header,.pane {{background:#fff;border:1px solid #cad7e1;border-radius:13px;padding:18px;margin-bottom:14px}}
h1 {{font-size:1.45rem;margin:0 0 8px}} h2 {{font-size:1.05rem;margin:0 0 8px}}
p {{line-height:1.5}} .notice {{background:#e9f5f2;border-left:4px solid #157b77;padding:10px 14px}}
.caution {{color:#694a20;background:#fff7e7;padding:9px;border-radius:6px}}
.cols {{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:14px}}
.cols .pane {{min-width:0}} .tags {{display:flex;gap:8px;flex-wrap:wrap;margin:9px 0}}
.small {{font-size:.86rem;color:#516477}} .value {{font-variant-numeric:tabular-nums;font-weight:750}}
.metrics {{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}}
.metric {{background:#f1f5f8;padding:12px;border-radius:8px;min-width:0}}
.metric .value {{display:block;font-size:1.05rem;overflow-wrap:anywhere}}
input[type=range] {{width:100%;accent-color:#127a87}}
svg {{width:100%;height:auto;display:block;max-width:100%}}
button {{border:1px solid #9bb8c7;background:#eaf4f7;color:#143b45;border-radius:8px;padding:7px 14px;cursor:pointer}}
button:focus-visible,input:focus-visible {{outline:3px solid #197b8b}}
hr {{border:0;border-top:1px solid #d7e1e7}}
</style></head><body><main>
<header id="m37b-source">
<h1>{escape(header)}</h1>
<p class="notice" id="m37b-original-statement">{escape(role_text)}</p>
<p class="caution" id="m37b-teacher-attribution">{escape(teacher)}</p>
<p class="small">Original source: {escape(meta["statement"])}. This is a WORKBOOK EXERCISE, not an asserted official JEE past paper.</p>
<p class="small">MODEL: externally maintained constant radial speed, not an electrically free accelerating particle; each state is a freshly solved static Kelvin equilibrium. Conventional current sign is positive <strong>sphere → ground</strong>. This does NOT model a resistive wire, electromagnetic delay, heat or microscopic electron drift.</p>
</header>
<div class="cols">
<section class="pane"><h2>Fixed physical radial schematic</h2>
<svg id="m37b-scene" viewBox="0 0 760 345" role="img" aria-label="Grounded conducting solid sphere with one externally driven point charge on a radial line">
<line x1="{cx}" y1="{cy}" x2="685" y2="{cy}" stroke="#b5cbd3" stroke-dasharray="6 8" stroke-width="2"/>
<circle id="m37b-sphere" cx="{cx}" cy="{cy}" r="{_fmt(r)}" fill="#c5dde3" stroke="#286078" stroke-width="3"/>
<line x1="{cx}" y1="{_fmt(cy+r)}" x2="{cx}" y2="316" stroke="#276777" stroke-width="4"/>
<path d="M95 316h68 M105 324h48 M115 332h28" stroke="#276777" stroke-width="3" fill="none"/>
<circle id="m37b-charge" cx="{_fmt(start)}" cy="{cy}" r="11" fill="#d36428" stroke="#964017" stroke-width="2"/>
<text id="m37b-q-sign" x="{_fmt(start)}" y="{_fmt(cy-19)}" fill="#994315" text-anchor="middle" font-size="17">{'+' if physics["source_charge_c"]>0 else '−'}q</text>
<text x="{cx}" y="{cy+4}" text-anchor="middle" font-size="16" fill="#175065">Grounded</text>
<text x="405" y="25" font-size="14" fill="#375465">Fixed scale; radial schematic (not world x or an EM current animation)</text>
</svg>
<p class="small">One physically solid conductor, one external charge. Only charge position changes by the teacher-supplied protocol; sphere and ground remain fixed. Spatial projection is fixed across frames.</p>
</section>
<section class="pane"><h2>Signed conventional current vs prescribed time</h2>
<svg id="m37b-current-chart" viewBox="0 0 615 295" role="img" aria-label="Python-precomputed conventional sphere-to-ground current versus prescribed physical elapsed seconds">
<rect x="48" y="19" width="508" height="238" fill="#f8fbfc" stroke="#dbe6e9"/>
<line x1="61" y1="139" x2="548" y2="139" stroke="#647787" stroke-width="1"/>
<polyline id="m37b-current-line" points="{polyline}" fill="none" stroke="#168181" stroke-width="3" stroke-linejoin="round"/>
<circle id="m37b-current-marker" cx="{_fmt(first["chart_x_px"])}" cy="{_fmt(first["chart_y_px"])}" r="7" fill="#d36428" stroke="#fff" stroke-width="2"/>
<text x="61" y="282" font-size="14" fill="#365263">t = 0 s</text>
<text x="546" y="282" text-anchor="end" font-size="14" fill="#365263">t = {_fmt(physics["duration_s"])} s</text>
<text x="60" y="14" font-size="14" fill="#365263">I (A), positive sphere → ground</text>
</svg>
<p class="small">Curve and marker are based entirely on stored Python-computed SI current samples. No browser current calculation or interpolated physics.</p>
</section></div>
<section class="pane">
<h2>Selected precomputed physical observation</h2>
<label for="m37b-index">Teacher-prescribed elapsed time (s): <strong id="m37b-time">{first["time_label"]}</strong></label>
<input id="m37b-index" type="range" min="0" max="{len(rows)-1}" value="0" step="1" disabled aria-label="Select precomputed prescribed time snapshot">
<p class="small"><button type="button" id="m37b-start" disabled>Restart</button> <button type="button" id="m37b-play" disabled>Play saved snapshots</button> Playback speed is display compression, NOT physical real-time.</p>
<div class="metrics" aria-live="polite">
<div class="metric">Radial separation d <strong class="value" id="m37b-distance">{first["distance_label"]}</strong></div>
<div class="metric">Net induced sphere Q <strong class="value" id="m37b-Q">{first["induced_charge_label"]}</strong></div>
<div class="metric">Current sphere → ground <strong class="value" id="m37b-current">{first["wire_current_label"]}</strong></div>
<div class="metric">Opposite conventional current <strong class="value" id="m37b-opposite">{first["opposite_current_label"]}</strong></div>
</div>
<noscript><p id="m37b-nojs"><strong>JavaScript is off:</strong> the original first snapshot, symbolic source statement, complete precomputed current curve and starting SI numbers remain visible. No other snapshot or animation is asserted.</p></noscript>
</section>
<script id="m37b-data" type="application/json">{dataset}</script>
<script id="m37b-display-only">
(function(){{
"use strict";
const payload=JSON.parse(document.getElementById("m37b-data").textContent);
const rows=payload.readouts;
const slider=document.getElementById("m37b-index");
const elements={{time:"m37b-time",distance:"m37b-distance",q:"m37b-Q",current:"m37b-current",opposite:"m37b-opposite"}};
let timer=null;
function show(index){{
  if (!Number.isInteger(index)||index<0||index>=rows.length) return;
  const row=rows[index];
  document.getElementById(elements.time).textContent=row.time_label;
  document.getElementById(elements.distance).textContent=row.distance_label;
  document.getElementById(elements.q).textContent=row.induced_charge_label;
  document.getElementById(elements.current).textContent=row.wire_current_label;
  document.getElementById(elements.opposite).textContent=row.opposite_current_label;
  document.getElementById("m37b-charge").setAttribute("cx",String(row.charge_x_px));
  document.getElementById("m37b-q-sign").setAttribute("x",String(row.charge_x_px));
  document.getElementById("m37b-current-marker").setAttribute("cx",String(row.chart_x_px));
  document.getElementById("m37b-current-marker").setAttribute("cy",String(row.chart_y_px));
  slider.value=String(index);
}}
function stop(){{if(timer!==null){{clearInterval(timer);timer=null;}}}}
slider.addEventListener("input",()=>{{stop();show(Number(slider.value));}});
document.getElementById("m37b-start").addEventListener("click",()=>{{stop();show(0);}});
document.getElementById("m37b-play").addEventListener("click",()=>{{
  stop();
  let i=Number(slider.value);
  if(i>=rows.length-1)i=-1;
  timer=setInterval(()=>{{
    i++;
    if(i>=rows.length){{stop();return;}}
    show(i);
  }},100);
}});
slider.disabled=false;
document.getElementById("m37b-start").disabled=false;
document.getElementById("m37b-play").disabled=false;
show(0);
window.m37bPrescribedViewer={{get mode(){{return payload.trace.actual_time_origin;}},get sampleCount(){{return rows.length;}},get selectedIndex(){{return Number(slider.value);}}}};
}})();
</script></main></body></html>
"""
