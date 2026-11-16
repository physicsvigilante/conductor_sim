"""M29a: offline comparison of native trajectories with different source inventories.

Draw real particle trajectory-space projections only. This is NOT an
apparatus/source-geometry drawing; per-model source membership and structural
change records are explicit, and a browser never integrates physical dynamics.
"""
from __future__ import annotations

from html import escape
import json
from math import isfinite
from pathlib import Path

from .structural_family import StructuralFamilyRun
from .trajectory_sweep_viewer import _dot, _physical_axes


def _fmt(number):
    if type(number) not in (int, float) or not isfinite(number):
        raise ValueError("nonfinite physical display value")
    return f"{number:.6g}"


def _scenario_track(scenario, native, label, status, change_labels, origin, u, v):
    sources = [
        dict(id=sid, label=display, kind=type(component).__name__)
        for (sid, display), component in zip(
            scenario.source_annotations, scenario.spec.field.sources)]
    samples = []
    if native is not None:
        if native.spec is not scenario.spec:
            raise ValueError("native trajectory is not bound to this physical source inventory")
        for row in native.samples:
            delta = tuple(row.position_m[k]-origin[k] for k in range(3))
            samples.append(dict(
                time_s=row.time_s, x_m=_dot(delta, u), y_m=_dot(delta, v),
                position_m=row.position_m, velocity_m_s=row.velocity_m_s))
    return dict(scenario_id=scenario.scenario_id,
                label=label, status=status,
                termination=None if native is None else native.termination,
                origin=scenario.origin, provenance=scenario.provenance,
                source_inventory=sources,
                has_gravity=scenario.spec.gravity is not None,
                structural_changes=change_labels,
                samples=samples)


def render_structural_family(run: StructuralFamilyRun, *,
                             title="M29a · Structural physical trajectory comparison") -> str:
    """Render a trusted native structural family; arbitrary imported JSON is refused."""
    if type(run) is not StructuralFamilyRun:
        raise TypeError("requires a native StructuralFamilyRun, not untrusted JSON")
    if (type(title) is not str or not 1 <= len(title) <= 120 or
            not title.isprintable() or title.strip() != title):
        raise ValueError("title must be printable, trimmed text")
    baseline = run.plan.baseline
    origin = baseline.spec.particle.position_m
    normal = baseline.spec.plane_normal
    u, v = _physical_axes(normal)
    tracks = [
        _scenario_track(baseline, run.baseline_trajectory, "Original baseline",
                        ("complete" if run.baseline_trajectory.termination == "duration"
                         else "censored"), [], origin, u, v)]
    for entry in run.entries:
        if entry.assembly is None:
            tracks.append(dict(
                scenario_id=f"{run.plan.plan_id}:{entry.case.case_id}",
                label=entry.case.note, status=entry.status, termination=None,
                origin="teacher-hypothetical", provenance=entry.case.note,
                source_inventory=None, has_gravity=None,
                structural_changes=[], reason=entry.reason, samples=[]))
            continue
        child = entry.assembly.scenario
        if child.spec.plane_normal != normal:
            raise ValueError("common-plane trajectory overlay refuses changed physical planes")
        if child.spec.particle.position_m != origin:
            raise ValueError("structural operation unexpectedly revised particle launch")
        track = _scenario_track(child, entry.trajectory, entry.case.note,
                                entry.status, entry.assembly.changes, origin, u, v)
        track["reason"] = entry.reason
        tracks.append(track)
    coords = [(s["x_m"], s["y_m"]) for t in tracks for s in t["samples"]]
    if not coords:
        raise ValueError("there are no native trajectory samples")
    minx, maxx = min(x for x,y in coords), max(x for x,y in coords)
    miny, maxy = min(y for x,y in coords), max(y for x,y in coords)
    dx, dy = max(maxx-minx, 1e-6), max(maxy-miny, 1e-6)
    bounds = [minx-.08*dx, maxx+.08*dx, miny-.08*dy, maxy+.08*dy]
    W,H,L,R,T,B = 860,480,70,26,24,49
    def px(x): return L+(x-bounds[0])/(bounds[1]-bounds[0])*(W-L-R)
    def py(y): return H-B-(y-bounds[2])/(bounds[3]-bounds[2])*(H-T-B)
    colors=("#136a98","#c55c3b","#567e42","#9161a8","#a17628","#148c83")
    lines = []
    for i in range(5):
        x = bounds[0] + (bounds[1]-bounds[0])*i/4
        y = bounds[2] + (bounds[3]-bounds[2])*i/4
        lines.append(f'<line x1="{px(x):.7f}" x2="{px(x):.7f}" y1="{T}" y2="{H-B}" stroke="#e0e8ef"/>')
        lines.append(f'<line x1="{L}" x2="{W-R}" y1="{py(y):.7f}" y2="{py(y):.7f}" stroke="#e0e8ef"/>')
        lines.append(f'<text x="{px(x):.7f}" y="{H-B+19}" text-anchor="middle" fill="#4c6578" font-size="11">{escape(_fmt(x))}</text>')
        lines.append(f'<text x="{L-8}" y="{py(y)+4:.7f}" text-anchor="end" fill="#4c6578" font-size="11">{escape(_fmt(y))}</text>')
    for i, track in enumerate(tracks):
        for row in track["samples"]:
            row["svg_x"]=px(row["x_m"])
            row["svg_y"]=py(row["y_m"])
        points = " ".join(f'{r["svg_x"]:.7f},{r["svg_y"]:.7f}'
                          for r in track["samples"])
        if points:
            lines.append(f'<polyline data-scenario-index="{i}" points="{points}" '
                         f'fill="none" stroke="{colors[i]}" stroke-width="3" '
                         'stroke-linejoin="round" stroke-linecap="round"/>')
            lines.append(f'<circle data-marker-index="{i}" '
                         'r="5.7" fill="'+colors[i]+'" stroke="white" stroke-width="1.5"/>')
    lines.append(f'<text x="{W/2}" y="{H-6}" text-anchor="middle" fill="#294862" font-size="13">Projected world u (m)</text>')
    lines.append(f'<text x="17" y="{H/2}" transform="rotate(-90 17 {H/2})" text-anchor="middle" fill="#294862" font-size="13">Projected world v (m)</text>')
    source_counts=[len(t["source_inventory"]) if t["source_inventory"] is not None else None
                   for t in tracks]
    times=sorted({s["time_s"] for t in tracks for s in t["samples"]})
    data=dict(schema="conductor_sim.structural_family_view.v1",
              plan_id=run.plan.plan_id,
              baseline_revision_id=baseline.revision_id,
              baseline_origin=baseline.origin, baseline_provenance=baseline.provenance,
              origin_m=origin, projection_u=u, projection_v=v,
              frame_times_s=times, source_counts=source_counts, tracks=tracks,
              browser_ode=False, time_interpolation=False, extrapolation=False,
              source_geometry_rendered=False)
    safe=(json.dumps(data, separators=(",", ":"), allow_nan=False)
          .replace("&", r"\u0026").replace("<", r"\u003c").replace(">", r"\u003e"))
    svg = "".join(lines)
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>"""+escape(title)+"""</title><style>
:root{font:15px/1.5 system-ui,sans-serif;color:#142e43;background:#f3f6f9}
*{box-sizing:border-box}body{margin:0}
header{padding:18px max(16px,calc((100vw - 1150px)/2));background:#173b55;color:#fff}
h1{margin:0 0 5px;font-size:24px}header p{margin:0;font-size:13px;color:#d5e6f2}
main{max-width:1150px;margin:auto;padding:16px;display:grid;gap:15px;min-width:0}
section{background:white;border:1px solid #ccd9e4;padding:16px;border-radius:9px;min-width:0;overflow-wrap:anywhere}
h2{font-size:17px;margin:0 0 6px}p{margin:5px 0 10px}
.muted{color:#516b7e;font-size:12px;overflow-wrap:anywhere}
svg{display:block;width:100%;height:auto;border:1px solid #d9e2e9;border-radius:5px;background:#fcfdfe}
.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
button{font:inherit;padding:7px 11px;background:white;color:#173b55;border:1px solid #9db2c5;border-radius:5px;cursor:pointer}
button:focus-visible,input:focus-visible{outline:3px solid #276f9c;outline-offset:2px}
input[type=range]{flex:1;min-width:115px}output{font:12px ui-monospace,monospace}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:12px}
th,td{text-align:left;vertical-align:top;padding:8px;border-bottom:1px solid #e0e7ee;overflow-wrap:anywhere}
th{background:#eef3f8}.scenario-name{font-weight:600}
@media(max-width:520px){h1{font-size:19px}main{padding:10px}section{padding:10px}th,td{padding:6px}}
</style></head><body><header><h1>"""+escape(title)+"""</h1>
<p>Actual native physical trajectories · independently reassembled source inventories · no browser physics</p>
</header><main><section><h2>World-plane trajectory comparison</h2>
<p class="muted">Each colored curve is one separately solved particle path in the
shared physically qualified plane. This is TRAJECTORY SPACE ONLY, NOT an apparatus
geometry, not a map of ring rims or source positions, and not a video of sources
changing during one physical experiment. Paths stop at real solved termination;
no source contact or postguard continuation is inferred.</p>
<svg id="structural-plot" viewBox="0 0 860 480" role="img" aria-label="Compared native particle trajectories in a common world plane">"""+svg+"""</svg>
<p class="muted" id="projection"></p>
<div class="controls"><button type="button" id="play">Play saved states</button>
<button type="button" id="restart">Restart</button>
<label for="saved-time">Saved physical time (union)</label>
<input id="saved-time" type="range" min="0" max="0" step="1" value="0"/>
<output id="clock" for="saved-time">0 s</output></div>
<p class="muted">Controls select exactly original Python-saved physical samples,
without interpolation or extrapolation. A source-guard-censored case is shown as
unavailable beyond its final saved state, while a longer case can continue.
For actual component geometry, use the existing qualified fixed-source viewer.</p>
</section><section><h2>Author-bound source memberships and real states</h2>
<div class="scroll"><table><thead><tr><th>Scenario</th><th>Physically assembled sources</th>
<th>Structural change from baseline</th><th>Native state or refusal</th>
<th>Compared to baseline at the exact saved time</th></tr></thead>
<tbody id="structural-states"></tbody></table></div>
</section><section><h2>Identity, provenance and interpretation</h2>
<p id="provenance"></p>
<p class="muted">Source IDs and names reflect the actual immutable authored models;
their provenance is an author assertion, not independent workbook authentication.
An insertion, removal, replacement or gravity membership change is a **separate
teacher hypothetical** and never a source moving during a single trajectory.
Numeric comparisons use common world coordinates and exactly matching physical
sample times; cross-inventory electrostatic potential energies are not
interpreted as mechanical work.</p>
</section></main>
<script type="application/json" id="m29a-native-data">"""+safe+"""</script>
<script>
'use strict';
const d=JSON.parse(document.getElementById('m29a-native-data').textContent);
const times=d.frame_times_s,slider=document.getElementById('saved-time'),
      clock=document.getElementById('clock'),tbody=document.getElementById('structural-states');
const colors=['#136a98','#c55c3b','#567e42','#9161a8','#a17628','#148c83'];
slider.max=String(times.length-1);
const view={loaded:true,frameIndex:0,physicalTime:times[0],activeCount:0,
            unavailableCount:0,sourceCounts:d.source_counts};
window.m29aStructuralViewer=view;
document.getElementById('provenance').textContent =
  'Original authored baseline ('+d.baseline_origin+'): '+d.baseline_provenance+
  ' · model revision SHA-256: '+d.baseline_revision_id;
document.getElementById('projection').textContent =
  'World origin ['+d.origin_m.join(', ')+'] m · u = ['+d.projection_u.join(', ')+
  '] · v = ['+d.projection_v.join(', ')+']';
const markers=d.tracks.map((track,i)=>document.querySelector(
  '#structural-plot circle[data-marker-index="'+i+'"]'));
const fmt=n=>Number(n).toPrecision(6);
function row(t,content){const el=document.createElement('td');el.textContent=content;t.appendChild(el);return el}
function paint(frame){
  if(!Number.isInteger(frame)||frame<0||frame>=times.length)throw Error('unknown saved time');
  const at=times[frame];slider.value=String(frame);
  clock.textContent=fmt(at)+' s · native '+(frame+1)+'/'+times.length;
  tbody.replaceChildren();let active=0,unavailable=0;
  d.tracks.forEach((track,i)=>{
    const tr=document.createElement('tr');
    const n=row(tr,track.scenario_id+' · '+track.label);
    n.className='scenario-name';n.style.color=colors[i];
    const inv=track.source_inventory;
    row(tr,inv===null?'No qualified source inventory':
      inv.map(s=>s.id+' ('+s.kind+')').join(', ')+(track.has_gravity?' · gravity':''));
    row(tr,track.structural_changes.length?track.structural_changes.map(c=>c.operation+
      (c.before?' ['+(c.before.id||c.component_id)+']':'')+
      (c.after?' ['+(c.after.id||c.component_id)+']':'')).join('; '):'Original inventory');
    const sample=track.samples.find(s=>s.time_s===at);
    let detail;
    if(sample){
      active++;
      detail=(track.status==='censored'?'CENSORED at '+fmt(track.samples.at(-1).time_s)+' s; ':'')+
        'r=['+sample.position_m.map(fmt).join(', ')+'] m, v=['+
        sample.velocity_m_s.map(fmt).join(', ')+'] m/s';
      if(markers[i]){
        markers[i].setAttribute('cx',String(sample.svg_x));
        markers[i].setAttribute('cy',String(sample.svg_y));
        markers[i].style.display='';
      }
    }else{
      unavailable++;
      detail=track.samples.length?'No saved state at this time; '+track.termination+
        ' at '+fmt(track.samples.at(-1).time_s)+' s':
        track.status+': '+(track.reason||'No qualified numerical history');
      if(markers[i])markers[i].style.display='none';
    }
    row(tr,detail);
    const base=d.tracks[0].samples.find(s=>s.time_s===at);
    if(i===0)row(tr,sample?'Baseline reference':'No baseline state');
    else if(sample&&base){
      const dr=sample.position_m.map((x,k)=>x-base.position_m[k]);
      const dv=sample.velocity_m_s.map((x,k)=>x-base.velocity_m_s[k]);
      row(tr,'Δr=['+dr.map(fmt).join(', ')+'] m; Δv=['+dv.map(fmt).join(', ')+'] m/s');
    }else row(tr,'No literal-time comparison; no interpolation');
    tbody.appendChild(tr);
  });
  view.frameIndex=frame;view.physicalTime=at;view.activeCount=active;
  view.unavailableCount=unavailable;
}
let play=null;
function stop(){if(play!==null)clearInterval(play);play=null;document.getElementById('play').textContent='Play saved states'}
function seek(i){stop();paint(i)}
slider.addEventListener('input',()=>seek(Number(slider.value)));
document.getElementById('restart').addEventListener('click',()=>seek(0));
document.getElementById('play').addEventListener('click',()=>{
  if(play!==null){stop();return}
  if(Number(slider.value)===times.length-1)paint(0);
  document.getElementById('play').textContent='Pause';
  play=setInterval(()=>{const i=Number(slider.value)+1;
    if(i>=times.length){stop();return}paint(i)},160);
});
paint(0);
</script></body></html>"""


def write_structural_family_html(run: StructuralFamilyRun, output, **kwargs) -> Path:
    path=Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_structural_family(run, **kwargs), encoding="utf-8")
    return path
