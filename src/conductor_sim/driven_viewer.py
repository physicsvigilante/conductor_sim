"""Offline physical-time driven motion: actual forces, switches and work ledger.

Reuses established style and projection conventions. Shared geometry is also used by the conservative event viewer, without
conflating its K+U observation with dissipative driven work.
"""
from html import escape
from .jee_potential_explorer import _STYLE, _safe_json_script
from .motion_event_viewer import _CSS
from .driven_motion import validate_driven_run
from .source_geometry import source_geometry, SOURCE_GEOMETRY_JS


def render_driven_run(run, *, title='M24b · Drag and scheduled electric field'):
    validate_driven_run(run);record=run.as_dict()
    record['source_geometry']=source_geometry(run.study.model.base.spec, run.study.model.base.source_annotations)
    frames={s.time_s:s.as_dict() for s in run.samples}
    last=run.samples[-1].time_s
    for e in run.events:
        if e.sample.time_s<=last:
            frames[e.sample.time_s]=e.sample.as_dict()
    record['display_states']=[frames[t] for t in sorted(frames)]
    events=''.join('<tr><td>'+escape(e.event_id)+'</td><td>'+escape(e.kind)+'</td><td>'+f'{e.sample.time_s:.10g} s'+
        '<br><small>'+escape(str(e.time_bracket_s))+'</small></td><td>'+f'{e.residual:.5g} '+escape(e.residual_unit)+
        '</td><td>'+('Speed within tolerance' if e.full_stop_within_tolerance else 'Not full rest')+'</td></tr>' for e in run.events)
    candidate_note=('<p class="muted" id="terminal-candidates">Overlapping terminal brackets retain alternative numerical event states, not post-stop motion. Playback ends at the retained terminal state; their order is unresolved.</p>'
                    if any(e.sample.time_s>last for e in run.events) else '')
    switches=''.join('<tr><td>'+f'{s.time_s:.10g} s'+'</td><td>'+escape(str(s.left.velocity_m_s))+
        '</td><td>'+escape(str(s.left.acceleration_m_s2))+'</td><td>'+escape(str(s.right.acceleration_m_s2))+'</td></tr>' for s in run.switches)
    outcomes=''.join('<p><strong>'+escape(k)+'</strong>: '+escape(v)+'</p>' for k,v in record['outcomes'].items())
    notes=''.join('<p>'+escape(str(x))+'</p>' for x in (*run.initial_conditions,*run.unresolved))
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>'+escape(title)+'</title><style>'+_STYLE+_CSS+'</style></head><body><header>'
        '<div class="kicker">CONDUCTOR_SIM / DRAG + SCHEDULED INPUTS</div><h1>'+escape(title)+'</h1><p class="sub">'+escape(run.study.model.provenance)+
        '</p></header><main><section class="card"><div class="controls"><button id="play">Play</button><button id="reset">Reset</button>'
        '<label>Speed <select id="speed"><option value=".25">0.25x</option><option value="1" selected>1x</option><option value="4">4x</option></select></label>'
        '<label><input type="checkbox" id="forces" checked> Resultant</label><label><input type="checkbox" id="components" checked> Electric / gravity / drag</label>'
        '<label><input type="checkbox" id="velocity" checked> Velocity</label><label><input type="checkbox" id="path" checked> Computed path</label>'
        '<label><input type="checkbox" id="source-geometry" checked> Physical source geometry</label></div>'
        '<input id="frame" type="range" min="0" max="'+str(len(frames)-1)+'" value="0" step="1"><p id="clock"></p>'
        '<p id="termination"><strong>Termination: '+escape(run.termination)+'</strong>. Detector termination is not impact or charge transfer; a source guard is not a detector.</p>'
        '<div class="grid"><div><canvas id="scene" width="800" height="480"></canvas><p id="axes" class="muted"></p>'
        '<p class="muted">Points/rods drawn only in the motion plane; sphere/shell outlines are actual slice intersections, and ring/disc shapes are orthographic 3D projections. Omitted off-plane points/rods retain their signed offsets and full electric fields. Full vectors are retained. No field-line map or responding conductor is implied.</p></div>'
        '<div><pre id="readout"></pre><h2>Retained physical components</h2><div id="sources"></div></div></div>'
        '<canvas id="energy" width="1000" height="250"></canvas><p class="muted">Dissipation is physical energy transfer to the stationary medium, not numerical drift. Curves join saved observations; playback uses computed states only.</p></section>'
        '<section class="card"><h2>Authored switches: continuous position, velocity and work</h2><div class="scroll"><table id="switches"><thead><tr><th>Time</th><th>Velocity, both sides (m/s)</th><th>Left acceleration (m/s²)</th><th>Right acceleration (m/s²)</th></tr></thead><tbody>'+
        (switches or '<tr><td colspan="4">No switch reached in this observed interval.</td></tr>')+'</tbody></table></div></section>'
        '<section class="card"><h2>Located events</h2><div class="scroll"><table id="events"><thead><tr><th>Identity</th><th>Kind</th><th>Time / bracket</th><th>Residual</th><th>Interpretation</th></tr></thead><tbody>'+
        (events or '<tr><td colspan="5">No isolated event located at the requested resolution.</td></tr>')+'</tbody></table></div>'+candidate_note+'<p>Ordering: '+escape(run.ordering)+
        '</p><p class="muted">Brackets localize the numerical trajectory, not exact ODE error. Events within the requested time tolerance of a switch can remain unresolved.</p></section>'
        '<section class="card" id="outcomes"><h2>Question outcomes</h2>'+outcomes+notes+'<p class="muted">'+escape(record['qualification'])+'</p></section>'
        '</main><script id="driven-data" type="application/json">'+_safe_json_script(record)+'</script><script>'+SOURCE_GEOMETRY_JS+_JS+'</script></body></html>')


_JS=r'''
'use strict';
const data=JSON.parse(document.getElementById('driven-data').textContent),states=data.display_states;
const model=data.study.model,base=model.base,spec=base.spec,questions=data.study.questions;
const $=id=>document.getElementById(id),fmt=(x,n=6)=>Number(x).toPrecision(n),vec=x=>'('+x.map(v=>fmt(v,5)).join(', ')+')';
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0),sub=(a,b)=>a.map((x,i)=>x-b[i]),mul=(a,k)=>a.map(x=>x*k);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const origin=spec.particle.position_m,n=spec.plane_normal;
const geometry=data.source_geometry,u=geometry.u,v=geometry.v;
const proj=p=>[dot(sub(p,origin),u),dot(sub(p,origin),v)],canvas=$('scene'),ctx=canvas.getContext('2d');
const positions=states.map(s=>proj(s.position_m));for(const q of questions)if(q.kind==='PlaneEvent')positions.push(proj(q.point_m));
sourceBounds(positions,geometry);
let xmin=Math.min(...positions.map(p=>p[0])),xmax=Math.max(...positions.map(p=>p[0])),ymin=Math.min(...positions.map(p=>p[1])),ymax=Math.max(...positions.map(p=>p[1]));
const span=Math.max(xmax-xmin,ymax-ymin,spec.length_scale_m*.08,1e-20),xc=(xmin+xmax)/2,yc=(ymin+ymax)/2,pix=Math.min(650/(xmax-xmin+.5*span),350/(ymax-ymin+.5*span));
const X=x=>400+(x-xc)*pix,Y=y=>230-(y-yc)*pix;
const forceKeys=['fixed_electric_force_n','applied_electric_force_n','gravitational_force_n','drag_force_n','force_n'];
const fmax=Math.max(...states.flatMap(s=>forceKeys.map(k=>Math.hypot(...s[k]))),1e-300),vmax=Math.max(...states.map(s=>Math.hypot(...s.velocity_m_s)),1e-300);
$('axes').textContent='Screen right '+vec(u)+'; screen up '+vec(v)+'. Origin '+vec(origin)+' m; common metre scale. Force arrows share one scale.';
populateSourceGeometry($('sources'),geometry);
let desc='Authored mechanical components';
desc+='\n'+model.drag.component_id+': b='+fmt(model.drag.coefficient_kg_s)+' kg/s (stationary medium)\n'+model.applied.source_id+': '+model.applied.label;
model.applied.phases.forEach((p,i)=>desc+='\nPhase '+i+' from '+fmt(p.start_s)+' s: E='+vec(p.field_v_m)+' V/m');
desc+='\nq='+fmt(spec.particle.charge_c)+' C; m='+fmt(spec.mass_kg)+' kg'+(spec.gravity?'\ng='+vec(spec.gravity.acceleration_m_s2)+' m/s²':'');
const sourceText=document.createElement('pre');sourceText.textContent=desc;$('sources').appendChild(sourceText);
let index=0,playing=false,wall=0,begin=0;
function arrow(p,w,max,color,width=2.5){if(Math.hypot(...w)===0)return;const a=[X(p[0]),Y(p[1])],b=[a[0]+85*dot(w,u)/max,a[1]-85*dot(w,v)/max];if(Math.hypot(b[0]-a[0],b[1]-a[1])<.1)return;ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();const th=Math.atan2(b[1]-a[1],b[0]-a[0]);ctx.beginPath();ctx.moveTo(...b);ctx.lineTo(b[0]-8*Math.cos(th-.4),b[1]-8*Math.sin(th-.4));ctx.lineTo(b[0]-8*Math.cos(th+.4),b[1]-8*Math.sin(th+.4));ctx.fill();}
function draw(){const s=states[index];ctx.clearRect(0,0,800,480);ctx.font='13px system-ui';ctx.fillStyle='#334e68';ctx.fillText('Phase '+s.phase_index+' · prescribed E = '+vec(model.applied.phases[s.phase_index].field_v_m)+' V/m',20,23);
 for(const q of questions){if(q.kind!=='PlaneEvent')continue;const A=dot(q.normal,u),B=dot(q.normal,v),C=dot(q.normal,sub(q.point_m,origin));if(A*A+B*B<1e-20)continue;const c=[A*C/(A*A+B*B),B*C/(A*A+B*B)],d=[-B,A],L=span*8/Math.hypot(...d);ctx.save();ctx.beginPath();ctx.rect(25,35,750,390);ctx.clip();ctx.strokeStyle='#8a7446';ctx.lineWidth=q.role==='ideal-stop-surface'?4:2;ctx.setLineDash(q.role==='ideal-stop-surface'?[]:[7,5]);ctx.beginPath();ctx.moveTo(X(c[0]-L*d[0]),Y(c[1]-L*d[1]));ctx.lineTo(X(c[0]+L*d[0]),Y(c[1]+L*d[1]));ctx.stroke();ctx.restore();const p=proj(q.point_m);ctx.fillStyle='#80683a';ctx.fillText(q.event_id,Math.max(27,Math.min(680,X(p[0])+8)),Math.max(48,Math.min(410,Y(p[1])-10)));}
 drawSourceGeometry(ctx,geometry,X,Y,pix,[25,35,750,390]);
 if($('path').checked){ctx.strokeStyle='#376a9b';ctx.lineWidth=2;ctx.beginPath();states.slice(0,index+1).forEach((r,j)=>{const p=proj(r.position_m);j?ctx.lineTo(X(p[0]),Y(p[1])):ctx.moveTo(X(p[0]),Y(p[1]));});ctx.stroke();}
 for(const e of data.events){const p=proj(e.sample.position_m);ctx.strokeStyle='#984184';ctx.strokeRect(X(p[0])-5,Y(p[1])-5,10,10);}
 const p=proj(s.position_m);
 if($('components').checked){arrow(p,s.fixed_electric_force_n,fmax,'#6d4fb5');arrow(p,s.applied_electric_force_n,fmax,'#8b51a0');arrow(p,s.gravitational_force_n,fmax,'#bc802a');arrow(p,s.drag_force_n,fmax,'#237c94');}
 if($('velocity').checked)arrow(p,s.velocity_m_s,vmax,'#218063');if($('forces').checked)arrow(p,s.force_n,fmax,'#b4483c',4.2);
 ctx.fillStyle='#193f65';ctx.beginPath();ctx.arc(X(p[0]),Y(p[1]),5,0,2*Math.PI);ctx.fill();
 [['Resultant','#b4483c'],['Electric','#8b51a0'],['Gravity','#bc802a'],['Drag','#237c94'],['Velocity','#218063']].forEach((a,i)=>{ctx.fillStyle=a[1];ctx.fillText(a[0],35+i*145,457);});
 $('clock').textContent='Physical t = '+fmt(s.time_s,10)+' s · phase '+s.phase_index+' · state '+(index+1)+'/'+states.length;
 $('readout').textContent='Position (m): '+vec(s.position_m)+'\nVelocity (m/s): '+vec(s.velocity_m_s)+'\nAcceleration (m/s²): '+vec(s.acceleration_m_s2)+'\nFixed electric force (N): '+vec(s.fixed_electric_force_n)+'\nApplied electric force (N): '+vec(s.applied_electric_force_n)+'\nGravity force (N): '+vec(s.gravitational_force_n)+'\nDrag force (N): '+vec(s.drag_force_n)+'\nResultant (N): '+vec(s.force_n)+'\nK: '+fmt(s.kinetic_energy_j)+' J\nFixed-field work: '+fmt(s.fixed_work_j)+' J\nApplied-field work: '+fmt(s.applied_work_j)+' J\nGravitational work: '+fmt(s.gravitational_work_j)+' J\nDissipated D: '+fmt(s.dissipated_energy_j)+' J\nWork-balance residual: '+fmt(s.work_balance_error_j)+' J';$('frame').value=index;energy();}
function energy(){const c=$('energy'),g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);const k0=states[0].kinetic_energy_j;const values=states.map(s=>[s.kinetic_energy_j-k0,s.fixed_work_j+s.applied_work_j+s.gravitational_work_j-s.dissipated_energy_j,s.dissipated_energy_j]);let lo=Math.min(0,...values.flat()),hi=Math.max(0,...values.flat());if(lo===hi){const d=Math.max(Math.abs(hi)*.1,1e-30);lo-=d;hi+=d;}const end=states[states.length-1].time_s||1,x=t=>70+890*t/end,y=z=>210-165*(z-lo)/(hi-lo);g.font='12px system-ui';g.fillStyle='#354f67';g.fillText('Work accounting (J) vs physical time (s) — no conserved K+U claim',70,20);g.fillText(fmt(hi,4),2,48);g.fillText(fmt(lo,4),2,211);g.fillText('0',68,239);g.fillText(fmt(end,5),850,239);const colors=['#277b63','#9a4f93','#a86526'];for(let k=0;k<3;k++){g.strokeStyle=colors[k];g.setLineDash(k===1?[6,4]:[]);g.lineWidth=2;g.beginPath();states.forEach((s,i)=>i?g.lineTo(x(s.time_s),y(values[i][k])):g.moveTo(x(s.time_s),y(values[i][k])));g.stroke();g.fillStyle=colors[k];g.fillText(['Delta K','Net work','Dissipation'][k],630+k*110,37);}g.setLineDash([3,4]);g.strokeStyle='#8494a4';for(const sw of data.switches){g.beginPath();g.moveTo(x(sw.time_s),43);g.lineTo(x(sw.time_s),215);g.stroke();}g.setLineDash([]);g.strokeStyle='#b64283';g.beginPath();g.moveTo(x(states[index].time_s),43);g.lineTo(x(states[index].time_s),215);g.stroke();}
function pause(){playing=false;$('play').textContent='Play';}function tick(now){if(!playing)return;const end=states[states.length-1].time_s,t=begin+(now-wall)/1000*Number($('speed').value)*end/4;while(index+1<states.length&&states[index+1].time_s<=t)index++;draw();if(t>=end){pause();return;}requestAnimationFrame(tick);}
$('play').onclick=()=>{if(playing){pause();return;}if(index===states.length-1)index=0;if(states.length<2)return;playing=true;wall=performance.now();begin=states[index].time_s;$('play').textContent='Pause';requestAnimationFrame(tick);};$('reset').onclick=()=>{pause();index=0;draw();};$('frame').oninput=()=>{pause();index=Number($('frame').value);draw();};$('speed').onchange=()=>{wall=performance.now();begin=states[index].time_s;};for(const id of ['forces','components','velocity','path','source-geometry'])$(id).onchange=draw;
window.conductorDrivenViewer={get loaded(){return true},get frameIndex(){return index},get playing(){return playing},get physicalTime(){return states[index].time_s}};draw();
'''
