"""Offline saved-state event viewer. No browser physics or invented interpolation."""
from __future__ import annotations

from dataclasses import asdict
from html import escape
import json

from .jee_potential_explorer import _safe_json_script, _STYLE
from .motion_events import EventRun, validate_event_run
from .source_geometry import source_geometry, SOURCE_GEOMETRY_JS


def render_event_run(run: EventRun, *, title='M24a · Motion event investigations'):
    validate_event_run(run)
    record=run.as_dict()
    record['source_geometry']=source_geometry(run.study.scenario.spec, run.study.scenario.source_annotations, trusted_scenario=run.study.scenario)
    frames={s.time_s:s.as_dict() for s in run.trajectory.samples}
    last=run.trajectory.samples[-1].time_s
    for event in run.events:
        if event.sample.time_s<=last:
            frames[event.sample.time_s]=event.sample.as_dict()
    record['display_states']=[frames[t] for t in sorted(frames)]
    rows=[]
    for event in run.events:
        s=event.sample;lo,hi=event.time_bracket_s
        rows.append('<tr><td>'+escape(event.event_id)+'</td><td>'+escape(event.kind)+
            '</td><td>'+f'{s.time_s:.10g} s<br><small>[{lo:.10g}, {hi:.10g}]</small>'+
            '</td><td>'+f'{event.residual:.4g} {event.residual_unit}'+'</td><td>'+
            ('Full speed within tolerance' if event.full_stop_within_tolerance
             else 'Not total rest')+'</td></tr>')
    outcome=''.join('<p><strong>'+escape(k)+':</strong> '+escape(v)+'</p>'
                    for k,v in record['outcomes'].items())
    notes=''.join('<p>'+escape(str(item))+'</p>' for item in (*run.initial_conditions,*run.unresolved))
    table=(''.join(rows) or '<tr><td colspan="5">No isolated event located in the reported interval.</td></tr>')
    data=_safe_json_script(record)
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>'+escape(title)+
        '</title><style>'+_STYLE+_CSS+'</style></head><body><header>'
        '<div class="kicker">CONDUCTOR_SIM / VERIFIED PHYSICAL-TIME EVENT STUDY</div><h1>'+escape(title)+
        '</h1><p class="sub">'+escape(run.study.scenario.provenance)+'</p></header><main>'
        '<section class="card"><div class="controls"><button id="play">Play</button>'
        '<button id="reset">Reset</button><label>Speed <select id="speed"><option value=".25">0.25x</option>'
        '<option value="1" selected>1x</option><option value="4">4x</option></select></label>'
        '<label><input id="forces" type="checkbox" checked> Resultant force</label>'
        '<label><input id="velocity" type="checkbox" checked> Velocity</label>'
        '<label><input id="path" type="checkbox" checked> Computed path</label>'
        '<label><input id="source-geometry" type="checkbox" checked> Physical source geometry</label></div>'
        '<input id="frame" type="range" min="0" max="'+str(len(frames)-1)+'" value="0" step="1">'
        '<p id="clock"></p><p id="termination"><strong>Termination: '+escape(run.trajectory.termination)+
        '</strong>. A terminal detector ends observation; no impact, absorption, rebound or charge transfer is computed.</p>'
        '<div class="grid"><div><canvas id="scene" width="800" height="480"></canvas>'
        '<p class="muted" id="axes"></p><p class="muted">Detector lines intersect the actual motion plane. '
        'In-slice points and rods are drawn only at their real locations. Sphere/shell outlines are true slice intersections; rings and discs are labelled orthographic 3D projections. Out-of-plane points/rods remain in the full field but have no false in-plane markers. No field-line map is implied.</p></div>'
        '<div><pre id="readout"></pre><h2>Retained physical sources</h2><div id="sources"></div></div></div>'
        '<canvas id="energy" width="1000" height="250"></canvas>'
        '<p class="muted">Only saved numerical states are played. Lines connect observations for display; '
        'event positions are freshly evaluated states, never linear interpolation of saved frames.</p></section>'
        '<section class="card"><h2>Located events and numerical resolution</h2><div class="scroll">'
        '<table id="events"><thead><tr><th>Identity</th><th>Quantity</th><th>Time / bracket</th>'
        '<th>Residual</th><th>Full velocity</th></tr></thead><tbody>'+table+'</tbody></table></div>'
        '<p>Ordering: '+escape(run.ordering)+'</p><p class="muted">Brackets localize roots of the numerical solution; '
        'they are not global exact-ODE error bounds. Event direction and full speed are checked separately.</p></section>'
        '<section class="card" id="outcomes"><h2>Question outcomes</h2>'+outcome+notes+
        '<p class="muted">'+escape(record['qualification'])+'</p></section></main>'
        '<script id="event-data" type="application/json">'+data+'</script><script>'+SOURCE_GEOMETRY_JS+_JS+'</script></body></html>')


_CSS='''canvas{width:100%;height:auto;border:1px solid #cddbe7;background:#fbfdff;box-sizing:border-box}
.controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap}button,select{font:inherit;padding:6px 12px}
#frame{width:100%;margin-top:14px}pre{font:13px/1.75 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere}
.scroll{overflow-x:auto}td{vertical-align:top}#sources{font-size:13px;overflow-wrap:anywhere}#clock{font-weight:650}
@media(max-width:850px){.grid{grid-template-columns:1fr}.card{padding:12px}td{padding:5px}main{padding:0 10px}}
.source-geometry-entry{white-space:pre-wrap;border-left:3px solid #c6d6e5;padding:5px 9px;margin:8px 0}'''

_JS=r'''
'use strict';
const data=JSON.parse(document.getElementById('event-data').textContent),states=data.display_states;
const model=data.study.model,spec=model.spec,questions=data.study.questions;
const $=id=>document.getElementById(id),fmt=(x,n=6)=>Number(x).toPrecision(n),vec=x=>'('+x.map(v=>fmt(v,5)).join(', ')+')';
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0),sub=(a,b)=>a.map((x,i)=>x-b[i]);
const scale=(a,k)=>a.map(x=>x*k),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const normal=spec.plane_normal,origin=spec.particle.position_m;
const geometry=data.source_geometry,u=geometry.u,v=geometry.v;
const project=p=>[dot(sub(p,origin),u),dot(sub(p,origin),v)],canvas=$('scene'),ctx=canvas.getContext('2d');
const positions=states.map(s=>project(s.position_m));
for(const e of data.events)positions.push(project(e.sample.position_m));
for(const q of questions)if(q.kind==='PlaneEvent')positions.push(project(q.point_m));
sourceBounds(positions,geometry);
let xmin=Math.min(...positions.map(p=>p[0])),xmax=Math.max(...positions.map(p=>p[0]));
let ymin=Math.min(...positions.map(p=>p[1])),ymax=Math.max(...positions.map(p=>p[1]));
let span=Math.max(xmax-xmin,ymax-ymin,spec.length_scale_m*.08,1e-20),xc=(xmin+xmax)/2,yc=(ymin+ymax)/2;
const pixels=Math.min(650/(xmax-xmin+.5*span),350/(ymax-ymin+.5*span));
const X=x=>400+(x-xc)*pixels,Y=y=>245-(y-yc)*pixels;
const Fmax=Math.max(...states.map(s=>Math.hypot(...s.force_n)),1e-300),Vmax=Math.max(...states.map(s=>Math.hypot(...s.velocity_m_s)),1e-300);
$('axes').textContent='Screen right: '+vec(u)+'; screen up: '+vec(v)+'. Origin '+vec(origin)+' m; common metre scale.';
populateSourceGeometry($('sources'),geometry);
const authored=document.createElement('pre');authored.id='authored';authored.textContent='q = '+fmt(spec.particle.charge_c)+' C; m = '+fmt(spec.mass_kg)+' kg\nInitial position: '+vec(origin)+' m\nInitial velocity: '+vec(spec.initial_velocity_m_s)+' m/s'+(spec.gravity?'\ng = '+vec(spec.gravity.acceleration_m_s2)+' m/s2':'');$('sources').appendChild(authored);
let index=0,playing=false,startWall=0,startTime=0;
function arrow(a,b,color){ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=color==='#ae473d'?4.5:2.6;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();let th=Math.atan2(b[1]-a[1],b[0]-a[0]);ctx.beginPath();ctx.moveTo(...b);ctx.lineTo(b[0]-9*Math.cos(th-.4),b[1]-9*Math.sin(th-.4));ctx.lineTo(b[0]-9*Math.cos(th+.4),b[1]-9*Math.sin(th+.4));ctx.closePath();ctx.fill();}
function draw(){
 const s=states[index];ctx.clearRect(0,0,800,480);ctx.font='13px system-ui';
 ctx.fillStyle='#53697d';ctx.fillText('Prescribed fields + explicit observation geometry',20,22);
 for(const q of questions){
  if(q.kind!=='PlaneEvent')continue;
  const A=dot(q.normal,u),B=dot(q.normal,v),C=dot(q.normal,sub(q.point_m,origin));
  if(A*A+B*B<1e-20)continue;
  const center=[A*C/(A*A+B*B),B*C/(A*A+B*B)],t=[-B,A],L=span*8/Math.hypot(...t);
  ctx.setLineDash(q.role==='ideal-stop-surface'?[]:[7,5]);ctx.strokeStyle=q.role==='ideal-stop-surface'?'#605b53':'#927747';ctx.lineWidth=q.role==='ideal-stop-surface'?4:2;
  ctx.save();ctx.beginPath();ctx.rect(25,35,750,410);ctx.clip();ctx.beginPath();ctx.moveTo(X(center[0]-L*t[0]),Y(center[1]-L*t[1]));ctx.lineTo(X(center[0]+L*t[0]),Y(center[1]+L*t[1]));ctx.stroke();ctx.restore();ctx.setLineDash([]);
  const qp=project(q.point_m);ctx.fillStyle='#715a30';ctx.fillText(q.event_id,Math.max(28,Math.min(680,X(qp[0])+7)),Math.max(50,Math.min(420,Y(qp[1])-12)));
 }
 drawSourceGeometry(ctx,geometry,X,Y,pixels,[25,35,750,410]);
 if($('path').checked){ctx.strokeStyle='#376a9b';ctx.lineWidth=2;ctx.beginPath();states.slice(0,index+1).forEach((s,j)=>{let p=project(s.position_m);j?ctx.lineTo(X(p[0]),Y(p[1])):ctx.moveTo(X(p[0]),Y(p[1]));});ctx.stroke();}
 for(const e of data.events){const p=project(e.sample.position_m);ctx.strokeStyle='#9b468b';ctx.lineWidth=2;ctx.strokeRect(X(p[0])-5,Y(p[1])-5,10,10);}
 const p=project(s.position_m),xy=[X(p[0]),Y(p[1])];
 if($('forces').checked)arrow(xy,[xy[0]+85*dot(s.force_n,u)/Fmax,xy[1]-85*dot(s.force_n,v)/Fmax],'#ae473d');
 if($('velocity').checked)arrow(xy,[xy[0]+85*dot(s.velocity_m_s,u)/Vmax,xy[1]-85*dot(s.velocity_m_s,v)/Vmax],'#218063');
 ctx.fillStyle='#1a4e7a';ctx.beginPath();ctx.arc(...xy,6,0,2*Math.PI);ctx.fill();
 ctx.fillStyle='#ae473d';ctx.fillText('Resultant force (common scale)',430,455);ctx.fillStyle='#218063';ctx.fillText('Velocity',655,455);
 $('clock').textContent='Physical t = '+fmt(s.time_s,10)+' s · computed state '+(index+1)+'/'+states.length;
 $('readout').textContent='Position (m): '+vec(s.position_m)+'\nVelocity (m/s): '+vec(s.velocity_m_s)+'\nSpeed: '+fmt(Math.hypot(...s.velocity_m_s))+' m/s\nE (V/m): '+vec(s.electric_field_v_m)+'\nForce (N): '+vec(s.force_n)+'\nK: '+fmt(s.kinetic_energy_j)+' J\nU: '+fmt(s.potential_energy_j)+' J\nK+U: '+fmt(s.total_energy_j)+' J\nNumerical energy drift: '+fmt(s.energy_drift_j)+' J';
 $('frame').value=index;energy();
}
function energy(){const c=$('energy'),g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);const keys=['kinetic_energy_j','potential_energy_j','total_energy_j'],cols=['#218063','#a96035','#445c99'];let vals=states.flatMap(s=>keys.map(k=>s[k])),lo=Math.min(...vals),hi=Math.max(...vals);if(hi===lo){let z=Math.max(Math.abs(hi)*.1,1e-30);lo-=z;hi+=z;}const duration=states[states.length-1].time_s||1;const xp=t=>65+905*t/duration,yp=z=>215-170*(z-lo)/(hi-lo);g.font='12px system-ui';g.fillStyle='#354f67';g.fillText('Actual numerical energies vs physical time (s) · J',65,20);g.fillText(fmt(hi,4),4,48);g.fillText(fmt(lo,4),4,215);g.fillText('0',62,239);g.fillText(fmt(duration,5),860,239);keys.forEach((key,k)=>{g.strokeStyle=cols[k];g.lineWidth=2;g.beginPath();states.forEach((s,j)=>j?g.lineTo(xp(s.time_s),yp(s[key])):g.moveTo(xp(s.time_s),yp(s[key])));g.stroke();g.fillStyle=cols[k];g.fillText(['K','U','K+U'][k],640+k*90,20);});g.strokeStyle='#b64283';g.beginPath();g.moveTo(xp(states[index].time_s),30);g.lineTo(xp(states[index].time_s),220);g.stroke();}
function pause(){playing=false;$('play').textContent='Play';}
function tick(now){if(!playing)return;const end=states[states.length-1].time_s;const t=startTime+(now-startWall)/1000*Number($('speed').value)*end/4;while(index+1<states.length&&states[index+1].time_s<=t)index++;draw();if(t>=end){pause();return;}requestAnimationFrame(tick);}
$('play').onclick=()=>{if(playing){pause();return;}if(index===states.length-1)index=0;if(states.length<2)return;playing=true;startWall=performance.now();startTime=states[index].time_s;$('play').textContent='Pause';requestAnimationFrame(tick);};
$('reset').onclick=()=>{pause();index=0;draw();};$('frame').oninput=()=>{pause();index=Number($('frame').value);draw();};$('speed').onchange=()=>{startWall=performance.now();startTime=states[index].time_s;};for(const id of ['forces','velocity','path','source-geometry'])$(id).onchange=draw;
window.conductorEventViewer={get loaded(){return true},get frameIndex(){return index},get frameCount(){return states.length},get playing(){return playing},get physicalTime(){return states[index].time_s}};
draw();
'''
