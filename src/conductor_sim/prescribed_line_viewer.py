"""M38b: offline, source-honest local-plane display of authentic M38a Kelvin time data.

No new electrostatic law, geometry path integration, or browser-side physics.
The 3D line is displayed in its true local u-w plane; full xyz remains visible.
"""
from __future__ import annotations

from html import escape
import json
from math import fsum, hypot, isfinite, ulp

from .prescribed_ground_line import PrescribedSphereLineTrace
from .solver import UnsupportedProblem


SCHEMA = "conductor_sim.prescribed_ground_line_viewer.v1"
HEADING = "TEACHER HYPOTHETICAL · grounded-sphere 3D straight-line passage"
ORIGINAL_WARNING = (
    "Original workbook p8 Multiple Correct Q21 is RADIAL. This teacher "
    "3D straight-line investigation (radial or off-axis) is TEACHER-AUTHORED, "
    "not an authenticated workbook replay or an additional original JEE problem."
)
TEACHER_NOTICE = (
    "ALL numbers, source positions, 3D velocity and duration are "
    "TEACHER-CHOSEN SI values, NOT printed workbook numerical givens."
)
PROTOCOL_NOTICE = (
    "Externally maintained constant CARTESIAN velocity; changing radial speed "
    "and instantaneous Kelvin equilibria. NOT a free Newtonian orbit, "
    "real ground-wire microphysics, radiation or a Maxwell transient."
)
PROJECTION = (
    "Fixed physical local u-w plane through the sphere centre and the actual "
    "3D straight line; NOT world x-y, not a perspective 3D scene."
)


def _fmt(value: float) -> str:
    if type(value) not in (float, int) or not isfinite(value):
        raise ValueError("non-finite physical display number")
    # Signed floating zero is not a negative physical current direction.
    if value == 0:
        return "0"
    return format(value, ".11g")


def _vec(values, suffix="m"):
    return "[" + ", ".join(_fmt(x) for x in values) + "] " + suffix


def _dot(a, b):
    return fsum(a[i] * b[i] for i in range(3))


def _unit(values, label):
    mag = hypot(*values)
    if not isfinite(mag) or mag <= 0:
        raise UnsupportedProblem(label + ": degenerate projection basis")
    return tuple(v / mag for v in values)


def _basis(native):
    centre = native["sphere_center_m"]
    start = native["initial_source_position_m"]
    v = native["constant_cartesian_velocity_m_s"]
    u = _unit(v, "physical Cartesian velocity")
    relative = tuple(start[i] - centre[i] for i in range(3))
    axial = _dot(relative, u)
    perpendicular = tuple(relative[i] - axial * u[i] for i in range(3))
    mag = hypot(*perpendicular)
    resolution = 128 * ulp(max(hypot(*relative), native["sphere_radius_m"]))
    if mag <= resolution:
        # A true radial path has infinitely many equivalent physical planes.
        # Pick a repeatable perpendicular world unit without inventing depth.
        j = min(range(3), key=lambda k: abs(u[k]))
        unit_axis = tuple(1. if k == j else 0. for k in range(3))
        axial2 = _dot(unit_axis, u)
        perpendicular = tuple(unit_axis[k] - axial2*u[k] for k in range(3))
    w = _unit(perpendicular, "physical local transverse direction")
    normal = _unit((
        u[1]*w[2]-u[2]*w[1],
        u[2]*w[0]-u[0]*w[2],
        u[0]*w[1]-u[1]*w[0],
    ), "physical local plane normal")
    if abs(_dot(u, w)) > 2e-12 or abs(hypot(*w)-1.) > 2e-12:
        raise UnsupportedProblem("derived u-w axes do not form an orthonormal physical plane")
    return u, w, normal


def line_view_manifest(trace: PrescribedSphereLineTrace) -> dict:
    """Emit a real full-physics checked manifest; never trust a forged trace."""
    if type(trace) is not PrescribedSphereLineTrace:
        raise TypeError("physical viewer requires genuine typed M38a line history")
    native = json.loads(json.dumps(trace.as_dict(), ensure_ascii=False, allow_nan=False))
    if native["source_role"] != "teacher_hypothetical_off_axis_transfer_not_original_workbook_p8_Q21":
        raise ValueError("M38b does not reproduce the original radial p8 question")
    if len(native["samples"]) < 3:
        raise ValueError("physical line viewer needs multiple saved true time states")

    centre = native["sphere_center_m"]
    u, w, normal = _basis(native)
    xy = []
    for row in native["samples"]:
        displacement = tuple(row["source_position_m"][i]-centre[i] for i in range(3))
        off_plane = abs(_dot(displacement, normal))
        if off_plane > 3e-10 * max(row["radial_distance_m"], native["sphere_radius_m"]):
            raise UnsupportedProblem("3D source position lies outside the qualified local physical plane")
        xy.append((_dot(displacement, u), _dot(displacement, w)))

    radius = native["sphere_radius_m"]
    extent_u = max(radius, *(abs(x) for x, _ in xy))
    extent_w = max(radius, *(abs(y) for _, y in xy))
    scale = min(275. / extent_u, 122. / extent_w, 106. / radius)
    if not isfinite(scale) or scale <= 0 or not isfinite(scale*radius):
        raise UnsupportedProblem("non-representable true local-plane view scale")

    cx, cy = 360., 190.
    peak = max(abs(s["wire_current_sphere_to_ground_a"]) for s in native["samples"])
    if not isfinite(peak) or peak <= 0:
        raise UnsupportedProblem("ground-wire current cannot be resolved for signed current plot")
    rows = []
    duration = native["duration_s"]
    for row, (a, b) in zip(native["samples"], xy):
        i = row["index"]
        sx, sy = cx + scale*a, cy - scale*b
        px = 64. + 506. * row["time_s"] / duration
        py = 151. - 97. * row["wire_current_sphere_to_ground_a"] / peak
        values = (sx, sy, px, py)
        if not all(isfinite(val) for val in values):
            raise UnsupportedProblem("physical projection/current plot is not representable")
        rows.append({
            "index": i,
            "scene_x_px": sx, "scene_y_px": sy,
            "sign_x_px": sx, "sign_y_px": sy - 20.,
            "chart_x_px": px, "chart_y_px": py,
            "time_label": _fmt(row["time_s"]) + " s",
            "distance_label": _fmt(row["radial_distance_m"]) + " m",
            "radial_velocity_label": _fmt(row["radial_velocity_m_s"]) + " m/s",
            "induced_charge_label": _fmt(row["induced_sphere_net_charge_c"]) + " C",
            "current_label": _fmt(row["wire_current_sphere_to_ground_a"]) + " A",
            "opposite_label": _fmt(row["wire_current_ground_to_sphere_a"]) + " A",
            "world_position_label": _vec(row["source_position_m"]),
            "world_velocity_label": _vec(row["source_velocity_m_s"], "m/s"),
            "force_label": _vec(row["induced_force_on_source_n"], "N"),
            "energy_label": _fmt(row["induced_mechanical_energy_j"]) + " J",
        })
    layout = {
        "projection": PROJECTION,
        "local_u_world": list(u), "local_w_world": list(w),
        "local_normal_world": list(normal),
        "u_label": _vec(u, "unit"),
        "w_label": _vec(w, "unit"),
        "normal_label": _vec(normal, "unit"),
        "sphere_cx_px": cx, "sphere_cy_px": cy,
        "sphere_r_px": radius*scale,
        "pixels_per_meter": scale,
        "path_start_x_px": rows[0]["scene_x_px"],
        "path_start_y_px": rows[0]["scene_y_px"],
        "path_end_x_px": rows[-1]["scene_x_px"],
        "path_end_y_px": rows[-1]["scene_y_px"],
        "chart_zero_y_px": 151.,
        "chart_peak_abs_a": peak,
    }
    return {
        "schema": SCHEMA,
        "source": {
            "role": "teacher_hypothetical",
            "heading": HEADING,
            "original_warning": ORIGINAL_WARNING,
            "teacher_notice": TEACHER_NOTICE,
            "protocol_notice": PROTOCOL_NOTICE,
            "newly_solved_original_workbook_items": 0,
        },
        "trace": native,
        "layout": layout,
        "readouts": rows,
        "display_contract": (
            "Only actual precomputed M38a Kelvin states, physical local u-w "
            "projection, saved current/force/energy and teacher-prescribed time. "
            "Polyline connections are DISPLAY interpolation, never a physical solver."
        ),
    }


_STYLE = """
:root{color-scheme:light;font-family:system-ui,Arial,sans-serif;color:#1c3546;background:#f4f7fa}
*{box-sizing:border-box} body{margin:0} main{max-width:1180px;padding:16px;margin:auto}
header,.pane{background:#fff;border:1px solid #cbd9e3;border-radius:12px;padding:17px;margin-bottom:14px}
h1{font-size:1.4rem;margin:0 0 8px}h2{font-size:1.05rem;margin:0 0 8px}
p{line-height:1.5} .notice{border-left:4px solid #ac6527;background:#fff4e7;padding:9px 12px}
.muted{color:#496176;font-size:.88rem}.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,470px),1fr));gap:12px}
.cols .pane{min-width:0}.metrics{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
.metric{background:#eff5f9;padding:10px;border-radius:8px;min-width:0}
.value{display:block;font-weight:700;font-size:.96rem;overflow-wrap:anywhere;font-variant-numeric:tabular-nums}
svg{display:block;width:100%;height:auto;max-width:100%}
input[type=range]{width:100%;accent-color:#147b82}
button{border:1px solid #99b6c7;background:#e9f4f7;border-radius:7px;padding:7px 12px;cursor:pointer}
button:focus-visible,input:focus-visible{outline:3px solid #168198}
.basis{font-family:ui-monospace,monospace;font-size:.79rem;overflow-wrap:anywhere}
@media(max-width:440px){.metrics{grid-template-columns:1fr}h1{font-size:1.22rem}}
"""

_SCRIPT = r"""
(function() {
  "use strict";
  const data = JSON.parse(document.getElementById("m38b-data").textContent);
  const rows = data.readouts;
  const slider = document.getElementById("m38b-index");
  const fields = {
    "m38b-time":"time_label", "m38b-distance":"distance_label",
    "m38b-radial-v":"radial_velocity_label", "m38b-Q":"induced_charge_label",
    "m38b-current":"current_label", "m38b-opposite":"opposite_label",
    "m38b-world":"world_position_label", "m38b-world-v":"world_velocity_label",
    "m38b-force":"force_label", "m38b-energy":"energy_label"
  };
  let timer = null;
  function show(i) {
    if (!Number.isInteger(i) || i < 0 || i >= rows.length) return;
    const r = rows[i];
    for (const [id,key] of Object.entries(fields)) {
      document.getElementById(id).textContent = r[key];
    }
    const charge = document.getElementById("m38b-charge");
    charge.setAttribute("cx", String(r.scene_x_px));
    charge.setAttribute("cy", String(r.scene_y_px));
    const sign = document.getElementById("m38b-q-sign");
    sign.setAttribute("x", String(r.sign_x_px));
    sign.setAttribute("y", String(r.sign_y_px));
    const marker = document.getElementById("m38b-current-marker");
    marker.setAttribute("cx", String(r.chart_x_px));
    marker.setAttribute("cy", String(r.chart_y_px));
    slider.value = String(i);
  }
  function stop() { if (timer !== null) { clearInterval(timer); timer = null; } }
  slider.addEventListener("input", () => {stop(); show(Number(slider.value));});
  document.getElementById("m38b-reset").addEventListener("click", () => {stop(); show(0);});
  document.getElementById("m38b-play").addEventListener("click", () => {
    stop();
    let i = Number(slider.value);
    if (i >= rows.length - 1) i = -1;
    timer = setInterval(() => {
      i++;
      if (i >= rows.length) {stop(); return;}
      show(i);
    }, 100);
  });
  slider.disabled = false;
  document.getElementById("m38b-play").disabled = false;
  document.getElementById("m38b-reset").disabled = false;
  show(0);
  window.m38bViewer = {
    get sampleCount(){return rows.length;},
    get selectedIndex(){return Number(slider.value);},
    get sourceRole(){return data.source.role;},
    get projection(){return data.layout.projection;}
  };
})();
"""


def render_prescribed_line_viewer(trace: PrescribedSphereLineTrace) -> str:
    """Generate complete, independent-of-network HTML and honest JS-off frame."""
    m = line_view_manifest(trace)
    source, native, layout, rows = m["source"], m["trace"], m["layout"], m["readouts"]
    first = rows[0]
    payload = json.dumps(m, ensure_ascii=False, separators=(",", ":"),
                         allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    poly = " ".join(_fmt(x["chart_x_px"])+","+_fmt(x["chart_y_px"]) for x in rows)
    f = _fmt
    qsign = "+q" if native["source_charge_c"] > 0 else "−q"
    # All dynamic text is from validated SI/manifests; keep HTML escaping
    # even for the authored sphere's potentially arbitrary human name.
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Teacher off-axis grounded sphere · verified current</title>
<style>{_STYLE}</style></head><body><main>
<header id="m38b-source"><h1 id="m38b-heading">{escape(source["heading"])}</h1>
<p class="notice" id="m38b-original-warning">{escape(source["original_warning"])}</p>
<p id="m38b-teacher-notice">{escape(source["teacher_notice"])}</p>
<p class="muted" id="m38b-protocol">{escape(source["protocol_notice"])}</p>
<p class="muted" id="m38b-sphere-name">Physical conductor: {escape(native["sphere_name"])} (grounded solid sphere). Conventional positive current is sphere → ground. All 3D world coordinates are SI metres.</p>
</header>
<div class="cols">
<section class="pane"><h2>Genuine source path in its local physical u–w plane</h2>
<p class="muted" id="m38b-projection">{escape(layout["projection"])}</p>
<svg id="m38b-scene" viewBox="0 0 720 390" role="img" aria-label="Genuine straight constant-velocity charge path, real solid grounded-sphere cross-section, one fixed equal-metre scale">
<line x1="32" y1="190" x2="688" y2="190" stroke="#c8d5dd" stroke-dasharray="4 8"/>
<line x1="360" y1="26" x2="360" y2="352" stroke="#c8d5dd" stroke-dasharray="4 8"/>
<circle id="m38b-sphere" cx="{f(layout["sphere_cx_px"])}" cy="{f(layout["sphere_cy_px"])}" r="{f(layout["sphere_r_px"])}" fill="#c5dce4" stroke="#236478" stroke-width="3"/>
<line id="m38b-path" x1="{f(layout["path_start_x_px"])}" y1="{f(layout["path_start_y_px"])}" x2="{f(layout["path_end_x_px"])}" y2="{f(layout["path_end_y_px"])}" stroke="#9e6b30" stroke-width="3" stroke-dasharray="10 6"/>
<circle id="m38b-charge" cx="{f(first["scene_x_px"])}" cy="{f(first["scene_y_px"])}" r="9" fill="#c14b2c" stroke="#fff" stroke-width="2"/>
<text id="m38b-q-sign" x="{f(first["sign_x_px"])}" y="{f(first["sign_y_px"])}" font-size="16" fill="#8e3f20" text-anchor="middle">{escape(qsign)}</text>
<text x="360" y="195" font-size="13" fill="#174c61" text-anchor="middle">Grounded</text>
<text x="676" y="181" font-size="14" fill="#355367" text-anchor="end">+u</text>
<text x="372" y="37" font-size="14" fill="#355367">+w</text>
<text x="28" y="378" font-size="12" fill="#416075">One fixed equal-metre scale in both axes; no world-xy claim</text>
</svg>
<p class="basis">u (world) = <strong id="m38b-u">{escape(layout["u_label"])}</strong></p>
<p class="basis">w (world) = <strong id="m38b-w">{escape(layout["w_label"])}</strong></p>
<p class="basis">normal (world) = <strong id="m38b-normal">{escape(layout["normal_label"])}</strong></p>
<p class="muted">Solid sphere circle, real straight-line external charge path. The grounding connection is an ideal boundary condition, not simulated wire conduction.</p>
</section>
<section class="pane"><h2>Signed conventional current vs prescribed physical time</h2>
<svg id="m38b-current-chart" viewBox="0 0 640 310" role="img" aria-label="All saved Kelvin sphere-to-ground current values against prescribed elapsed seconds">
<rect x="53" y="29" width="529" height="245" fill="#f8fbfc" stroke="#dae6ec"/>
<line id="m38b-zero" x1="64" y1="{f(layout["chart_zero_y_px"])}" x2="570" y2="{f(layout["chart_zero_y_px"])}" stroke="#567185" stroke-width="1.5"/>
<polyline id="m38b-current-line" points="{poly}" fill="none" stroke="#198284" stroke-width="3" stroke-linejoin="round"/>
<circle id="m38b-current-marker" cx="{f(first["chart_x_px"])}" cy="{f(first["chart_y_px"])}" r="7" fill="#c14b2c" stroke="white" stroke-width="2"/>
<text x="60" y="17" font-size="13" fill="#38566a">I (A), positive sphere → ground</text>
<text x="64" y="291" font-size="13" fill="#38566a">t=0 s</text>
<text x="570" y="291" text-anchor="end" font-size="13" fill="#38566a">t={f(native["duration_s"])} s</text>
</svg>
<p class="muted">All {len(rows)} genuine saved SI observations are precomputed in Python. Joined line segments are visual interpolation only; they are NOT additional physical solutions.</p>
</section>
</div>
<section class="pane">
<h2>Selected authentic teacher-prescribed elapsed-time frame</h2>
<label for="m38b-index">Physical t = <strong id="m38b-time">{escape(first["time_label"])}</strong></label>
<input id="m38b-index" type="range" min="0" max="{len(rows)-1}" step="1" value="0" disabled aria-label="Choose saved physical time index">
<p><button id="m38b-reset" type="button" disabled>Restart</button> <button id="m38b-play" type="button" disabled>Play saved frames</button>
<span class="muted">Playback compression is a display control, not new motion physics.</span></p>
<div class="metrics" aria-live="polite">
<div class="metric">Radial distance d<strong class="value" id="m38b-distance">{escape(first["distance_label"])}</strong></div>
<div class="metric">Radial velocity dr/dt<strong class="value" id="m38b-radial-v">{escape(first["radial_velocity_label"])}</strong></div>
<div class="metric">Induced sphere net Q<strong class="value" id="m38b-Q">{escape(first["induced_charge_label"])}</strong></div>
<div class="metric">Sphere → ground current<strong class="value" id="m38b-current">{escape(first["current_label"])}</strong></div>
<div class="metric">Ground → sphere current<strong class="value" id="m38b-opposite">{escape(first["opposite_label"])}</strong></div>
<div class="metric">Real world position (x,y,z)<strong class="value" id="m38b-world">{escape(first["world_position_label"])}</strong></div>
<div class="metric">Real Cartesian velocity<strong class="value" id="m38b-world-v">{escape(first["world_velocity_label"])}</strong></div>
<div class="metric">Induced source force (world xyz)<strong class="value" id="m38b-force">{escape(first["force_label"])}</strong></div>
<div class="metric">Induced mechanical energy<strong class="value" id="m38b-energy">{escape(first["energy_label"])}</strong></div>
</div>
<noscript><p id="m38b-nojs"><strong>JavaScript disabled:</strong> This is the authentic first physical frame. The complete static path and signed current curve remain visible. Controls are disabled; no other time or physics is implied.</p></noscript>
</section>
<script id="m38b-data" type="application/json">{payload}</script>
<script id="m38b-display-only">{_SCRIPT}</script>
</main></body></html>
"""
    return html
