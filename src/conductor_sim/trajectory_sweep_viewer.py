"""M28a: honest offline trajectory-space comparison over native saved states.

This plots projected particle paths, not conductor/ring apparatus geometry.
JavaScript selects already-solved Python samples; it never integrates ODEs.
"""
from __future__ import annotations

from html import escape
import json
from math import hypot, ulp
from pathlib import Path

from .trajectory_sweep import TrajectorySweepRun, _shared_rows


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def _unit(v):
    length = hypot(*v)
    if length < 1e-10:
        raise ValueError("degenerate projection basis")
    return tuple(a/length for a in v)


def _physical_axes(normal):
    """Deterministic orthogonal world-plane directions, x then y preferred."""
    normal = _unit(normal)
    candidates = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
    def orthogonal(candidate, earlier=()):
        v = tuple(candidate[i]-_dot(candidate, normal)*normal[i] for i in range(3))
        for axis in earlier:
            projection = _dot(v, axis)
            v = tuple(v[i]-projection*axis[i] for i in range(3))
        if hypot(*v) < 1e-8:
            return None
        return _unit(v)
    u = next(x for candidate in candidates
             if (x := orthogonal(candidate)) is not None)
    v = next(x for candidate in candidates
             if (x := orthogonal(candidate, (u,))) is not None)
    return u, v


def render_trajectory_sweep(run: TrajectorySweepRun, *,
                            title="M28a · Compared physical trajectories") -> str:
    """Render trusted native sweep results; no external JS, time interpolation or ODE.

    A serialized/untrusted dictionary is *not* accepted as a solved result.
    Censored runs remain shorter; their cursor vanishes at unavailable times.
    """
    if type(run) is not TrajectorySweepRun:
        raise TypeError("requires a native TrajectorySweepRun, not arbitrary JSON")
    if (type(title) is not str or not 1 <= len(title) <= 120 or
            not title.isprintable() or title != title.strip()):
        raise ValueError("invalid trajectory comparison title")
    origin = run.plan.baseline.spec.particle.position_m
    normal = run.plan.baseline.spec.plane_normal
    # Several parallel physical planes can all be separately invariant
    # (e.g. uniform E with a shifted out-of-plane launch). A single 2D
    # projection must NOT silently collapse distinct world planes.
    for entry in run.entries:
        if entry.scenario is None or entry.trajectory is None:
            continue
        revised = entry.scenario.spec
        if hypot(*(a-b for a,b in zip(revised.plane_normal, normal))) > 1e-12:
            raise ValueError("sweep viewer requires the same world physical plane orientation")
        launch = revised.particle.position_m
        world_scale = max(1., *(abs(x) for x in (*origin, *launch)))
        separation = abs(_dot(tuple(launch[i]-origin[i] for i in range(3)), normal))
        if separation > max(1e-12, 64*ulp(world_scale)):
            raise ValueError("sweep viewer refuses different physical planes; "
                             "use the full 3D numerical JSON comparison")
    u, v = _physical_axes(normal)
    all_tracks = []

    def project(sample):
        p = tuple(sample.position_m[i]-origin[i] for i in range(3))
        return dict(time_s=sample.time_s, x_m=_dot(p, u), y_m=_dot(p, v),
                    position_m=sample.position_m, velocity_m_s=sample.velocity_m_s,
                    kinetic_energy_j=sample.kinetic_energy_j)

    all_tracks.append(dict(scenario_id=run.plan.baseline.scenario_id, role="baseline",
                           value=None, status=("complete" if
                           run.baseline_trajectory.termination == "duration" else "censored"),
                           explanation=run.baseline_trajectory.termination,
                           termination=run.baseline_trajectory.termination,
                           samples=[project(s) for s in run.baseline_trajectory.samples],
                           comparisons=[]))
    for entry in run.entries:
        if entry.scenario is not None and entry.trajectory is not None:
            if entry.trajectory.spec is not entry.scenario.spec:
                raise ValueError("sweep trajectory is not bound to its physical scenario")
        all_tracks.append(dict(scenario_id=(
                               entry.scenario.scenario_id if entry.scenario else
                               f"{run.plan.plan_id}:v{entry.index:02d}"),
                               role="variation", value=entry.requested_value,
                               status=entry.status, explanation=entry.explanation,
                               termination=(None if entry.trajectory is None else
                                            entry.trajectory.termination),
                               samples=[] if entry.trajectory is None else
                               [project(s) for s in entry.trajectory.samples],
                               comparisons=[] if entry.trajectory is None else
                               _shared_rows(run.baseline_trajectory, entry.trajectory)))

    coords = [(s["x_m"], s["y_m"]) for track in all_tracks for s in track["samples"]]
    if not coords:
        raise ValueError("no native trajectory samples")
    min_x, max_x = min(x for x, y in coords), max(x for x, y in coords)
    min_y, max_y = min(y for x, y in coords), max(y for x, y in coords)
    dx = max(max_x-min_x, 1e-6)
    dy = max(max_y-min_y, 1e-6)
    bounds = [min_x-.07*dx, max_x+.07*dx, min_y-.07*dy, max_y+.07*dy]
    data = dict(schema="conductor_sim.trajectory_sweep_view.v1",
                plan_id=run.plan.plan_id, axis=dict(
                    component_id=run.plan.axis.component_id,
                    property_name=run.plan.axis.property_name),
                baseline_revision_id=run.plan.baseline.revision_id,
                source_ids=[pair[0] for pair in run.plan.baseline.source_annotations],
                frame_times_s=[s.time_s for s in run.baseline_trajectory.samples],
                bounds=bounds, origin_m=origin, projection_u=u, projection_v=v,
                tracks=all_tracks, time_interpolation=False, browser_ode=False)
    raw=json.dumps(data, separators=(",", ":"), allow_nan=False)
    safe=(raw.replace("&", r"\u0026").replace("<", r"\u003c")
              .replace(">", r"\u003e"))
    heading=escape(title, quote=True)
    axis_text=escape(f"{run.plan.axis.component_id}.{run.plan.axis.property_name}")
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>"""+heading+"""</title>
<style>
:root{color-scheme:light;font:15px/1.5 system-ui,sans-serif;color:#162e42;background:#f5f7fa}
*{box-sizing:border-box}body{margin:0}header{padding:22px max(16px,calc((100vw - 1100px)/2));background:#18364d;color:#fff}
h1{font-size:23px;margin:0 0 7px}header p{margin:0;font-size:13px;color:#d7e7f3}
main{width:100%;max-width:1130px;margin:auto;padding:16px;display:grid;gap:16px}
.card{background:white;border:1px solid #cbd6e1;border-radius:9px;padding:16px;min-width:0;overflow-wrap:anywhere}
h2{font-size:17px;margin:0 0 8px}p{margin:6px 0 12px}.meta{color:#536b7e;font-size:12px;overflow-wrap:anywhere}
.controls{display:flex;flex-wrap:wrap;align-items:center;gap:10px}
button{border:1px solid #a7b9c8;border-radius:5px;background:white;color:#173a55;padding:8px 13px;font:inherit;cursor:pointer}
button:focus-visible,input:focus-visible{outline:3px solid #327aab;outline-offset:2px}
input[type=range]{flex:1;min-width:120px}#clock{font:13px ui-monospace,monospace}
svg{width:100%;height:auto;display:block;background:#fafcfe;border:1px solid #d5e0e8;border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:12px}
th,td{padding:9px;text-align:left;border-bottom:1px solid #e2e8ef;vertical-align:top;overflow-wrap:anywhere}
th{background:#edf3f7}.scroll{overflow-x:auto}td:nth-child(1){font-weight:600}
code{overflow-wrap:anywhere}.notice{font-size:12px;color:#506778}
@media(max-width:520px){h1{font-size:19px}main{padding:10px}.card{padding:10px}
.controls{gap:8px}table{font-size:11px}th,td{padding:6px}}
</style></head><body><header><h1>"""+heading+"""</h1>
<p>Independent Python trajectories · parameter """+axis_text+""" · no browser physics</p></header>
<main><section class="card"><h2>Trajectory-space comparison</h2>
<p class="meta">This graph projects actual world positions into the common physical plane.
It is NOT an apparatus/ring-geometry drawing. Each colored line is one native solved
trajectory, including a terminal source-guard point where present. No path is
extended after termination.</p>
<svg id="sweep-plot" viewBox="0 0 840 490" role="img" aria-label="Compared particle trajectories in a physically qualified plane"></svg>
<p class="meta" id="projection"></p>
<div class="controls"><button id="play" type="button">Play saved states</button>
<button id="reset" type="button">Restart</button><label for="frame">Saved physical time</label>
<input id="frame" type="range" min="0" max="0" step="1" value="0">
<output id="clock" for="frame">0 s</output></div>
<p class="notice">Playback selects discrete Python-saved states at display pace, not physical-time-rate
integration or interpolation. A missing synchronized sample is displayed as unavailable.
For source geometry and field maps use the original qualified M27b viewer.</p></section>
<section class="card"><h2>Actual outcomes at the selected saved time</h2>
<div class="scroll"><table><thead><tr><th>Physical scenario</th><th>Status / saved state</th>
<th>Difference from baseline at identical time</th></tr></thead>
<tbody id="sweep-states"></tbody></table></div></section>
<section class="card"><h2>Model lineage and interpretation</h2>
<p class="meta" id="source-ids"></p><p>All variations are teacher-authored physical hypotheses
from the unchanged baseline; they are not intermediate particle states or literal
experimental transitions. An admissible revised model can still fail or terminate at
a source guard. Comparison of absolute potential energies is intentionally omitted
because electrostatic reference choices can differ between scenarios.</p>
<p class="notice">Source workbook p12 Q40–41 gives ring A/B geometry, E, and source-given probe q=+10 µC.
The M27b numeric mass, shifted launch, elapsed time and these revisions are teacher choices.
The simulator does not solve the source's original symbolic axial Q41 as written.</p>
</section></main><script type="application/json" id="m28-sweep-data">"""+safe+"""</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('m28-sweep-data').textContent);
const ns='http://www.w3.org/2000/svg',plot=document.getElementById('sweep-plot');
const clock=document.getElementById('clock'),slider=document.getElementById('frame');
const table=document.getElementById('sweep-states');
const colors=['#176a99','#c65332','#567a35','#8757a6','#b4770b','#127a78','#b14985'];
const W=840,H=490,L=58,R=25,T=27,B=54,b=data.bounds;
function px(x){return L+(x-b[0])/(b[1]-b[0])*(W-L-R)}
function py(y){return H-B-(y-b[2])/(b[3]-b[2])*(H-T-B)}
function elt(tag,attrs,parent=plot){const node=document.createElementNS(ns,tag);
  for(const [k,v] of Object.entries(attrs))node.setAttribute(k,String(v));
  parent.appendChild(node);return node}
function fmt(n){return Number(n).toPrecision(5)}
for(let i=0;i<=4;i++){const x=b[0]+i*(b[1]-b[0])/4,y=b[2]+i*(b[3]-b[2])/4;
  elt('line',{x1:px(x),x2:px(x),y1:T,y2:H-B,stroke:'#e0e7ef'});
  elt('line',{x1:L,x2:W-R,y1:py(y),y2:py(y),stroke:'#e0e7ef'});
  const xt=elt('text',{x:px(x),y:H-B+19,'text-anchor':'middle',fill:'#445e70','font-size':11});
  xt.textContent=fmt(x);
  const yt=elt('text',{x:L-8,y:py(y)+4,'text-anchor':'end',fill:'#445e70','font-size':11});
  yt.textContent=fmt(y)}
const xlabel=elt('text',{x:W/2,y:H-6,'text-anchor':'middle',fill:'#284b65','font-size':13});
xlabel.textContent='Projected world position u (m)';
const ylabel=elt('text',{x:15,y:H/2,transform:'rotate(-90 15 '+H/2+')','text-anchor':'middle',fill:'#284b65','font-size':13});
ylabel.textContent='Projected world position v (m)';
const markers=[];
data.tracks.forEach((track,i)=>{
  if(track.samples.length){
    const poly=elt('polyline',{points:track.samples.map(s=>px(s.x_m)+','+py(s.y_m)).join(' '),
      fill:'none',stroke:colors[i],'stroke-width':2.8,'stroke-linecap':'round','stroke-linejoin':'round'});
    const marker=elt('circle',{r:5.5,fill:colors[i],stroke:'#fff','stroke-width':1.5});
    markers.push(marker);
  }else{markers.push(null)}
});
document.getElementById('projection').textContent='Origin (m) = ['+
  data.origin_m.map(fmt).join(', ')+'] · world u = ['+data.projection_u.map(fmt).join(', ')+
  '] · world v = ['+data.projection_v.map(fmt).join(', ')+']';
document.getElementById('source-ids').textContent='Author-bound source identities: '+
  data.source_ids.join(', ')+' · parent revision SHA-256: '+data.baseline_revision_id;
slider.max=String(data.frame_times_s.length-1);
const viewer={loaded:true,physicalTime:0,frameIndex:0,activeCount:0,missingCount:0};
window.m28SweepViewer=viewer;
function redraw(index){
  if(!Number.isInteger(index)||index<0||index>=data.frame_times_s.length)throw Error('saved index out of range');
  const t=data.frame_times_s[index];
  slider.value=String(index);clock.textContent=fmt(t)+' s · sample '+(index+1)+'/'+data.frame_times_s.length;
  table.replaceChildren();let active=0,missing=0;
  data.tracks.forEach((track,i)=>{
    const tr=document.createElement('tr'),a=document.createElement('td'),status=document.createElement('td'),
          delta=document.createElement('td');
    a.textContent=track.scenario_id+' · '+(track.role==='baseline'?'baseline':
      data.axis.component_id+'.'+data.axis.property_name+' = '+JSON.stringify(track.value));
    a.style.color=colors[i];
    const row=track.samples.find(s=>s.time_s===t);
    if(row){
      active++;
      if(markers[i]){markers[i].setAttribute('cx',px(row.x_m));markers[i].setAttribute('cy',py(row.y_m));
        markers[i].style.display=''}
      status.textContent=(track.status==='censored'?'CENSORED at '+fmt(track.samples.at(-1).time_s)+' s · ':'')+
        'r = ['+row.position_m.map(fmt).join(', ')+'] m; v = ['+
        row.velocity_m_s.map(fmt).join(', ')+'] m/s';
    }else{
      missing++;
      if(markers[i])markers[i].style.display='none';
      status.textContent=track.status==='rejected-physical-model'||track.status==='integration-failed'?
        track.status+': '+track.explanation :
        'No saved state at this time; '+track.termination+' at '+fmt(track.samples.at(-1).time_s)+' s';
    }
    if(track.role==='baseline')delta.textContent='Reference state';
    else{
      const match=track.comparisons.find(s=>s.time_s===t);
      delta.textContent=match?'|Δr| = '+fmt(match.position_distance_m)+
        ' m; |Δv| = '+fmt(match.velocity_distance_m_s)+' m/s':
        'No exact-time comparison; no interpolation';
    }
    tr.append(a,status,delta);table.append(tr);
  });
  viewer.physicalTime=t;viewer.frameIndex=index;viewer.activeCount=active;viewer.missingCount=missing;
}
let playback=null;
function stop(){if(playback!==null)clearInterval(playback);playback=null;document.getElementById('play').textContent='Play saved states'}
function seek(i){stop();redraw(i)}
slider.addEventListener('input',()=>seek(Number(slider.value)));
document.getElementById('reset').addEventListener('click',()=>seek(0));
document.getElementById('play').addEventListener('click',()=>{
  if(playback!==null){stop();return}
  if(Number(slider.value)===data.frame_times_s.length-1)redraw(0);
  document.getElementById('play').textContent='Pause';
  playback=setInterval(()=>{
    const next=Number(slider.value)+1;
    if(next>=data.frame_times_s.length){stop();return}
    redraw(next);
  },160);
});
redraw(0);
</script></body></html>"""


def write_trajectory_sweep_html(run, path, **kwargs):
    """Write one self-contained offline HTML document."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_trajectory_sweep(run, **kwargs), encoding="utf-8")
    return target
