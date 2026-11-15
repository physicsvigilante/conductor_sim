"""Offline fixed-axis dipole playback of verified saved physical-time samples."""
from html import escape
import json
from pathlib import Path

from .dipole_motion import validate_dipole_motion
from .dipole_viewer import _STYLE

_SCRIPT = r"""
const data=JSON.parse(document.getElementById('motion-data').textContent), rows=data.samples, spec=data.spec;
const $=id=>document.getElementById(id), c=$('scene'),ctx=c.getContext('2d'),model=spec.model;
const n=spec.rotation_axis, pivot=model.dipole.pivot_m;
const fmt=x=>Number(x).toPrecision(6),vec=v=>'['+v.map(fmt).join(', ')+']';
const pts=rows.flatMap(r=>[r.observation.positive_position_m,r.observation.negative_position_m]);
for(const named of model.sources){const s=named.source;if(s.position_m)pts.push(s.position_m);
if(s.center_m){pts.push(s.center_m);if(s.radius_m)pts.push([s.center_m[0]+s.radius_m,s.center_m[1]+s.radius_m,0],[s.center_m[0]-s.radius_m,s.center_m[1]-s.radius_m,0])}
if(s.start_m)pts.push(s.start_m,s.end_m);}
const span=Math.max(model.dipole.separation_m*.75,...pts.map(p=>Math.max(Math.abs(p[0]-pivot[0]),Math.abs(p[1]-pivot[1]))))*1.35;
const sc=135/span,X=p=>[300+(p[0]-pivot[0])*sc,190-(p[1]-pivot[1])*sc];
const maxF=Math.max(0,...rows.flatMap(r=>[r.observation.positive_force_n,r.observation.negative_force_n,r.observation.net_force_n].map(f=>Math.hypot(...f))));
function line(a,b,color,width=2){ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();}
function text(s,p,color='#233e54'){ctx.fillStyle=color;ctx.font='12px system-ui';ctx.fillText(s,...p);}
function dot(p,r,color){ctx.fillStyle=color;ctx.beginPath();ctx.arc(...p,r,0,2*Math.PI);ctx.fill();}
function arrow(p,f,color){if(!maxF)return;const a=X(p),v=[68*f[0]/maxF,-68*f[1]/maxF],b=[a[0]+v[0],a[1]+v[1]];
if(Math.hypot(...v)<.01)return;line(a,b,color,2.5);const t=Math.atan2(v[1],v[0]);line(b,[b[0]-6*Math.cos(t-.5),b[1]-6*Math.sin(t-.5)],color);line(b,[b[0]-6*Math.cos(t+.5),b[1]-6*Math.sin(t+.5)],color);}
function scene(r){ctx.clearRect(0,0,c.width,c.height);ctx.fillStyle='#fbfdff';ctx.fillRect(0,0,c.width,c.height);
line([30,338],[100,338],'#778b99');text('+x',[104,342]);line([30,338],[30,280],'#778b99');text('+y',[23,270]);
let u=0;
for(const named of model.sources){const s=named.source;
if(s.kind==='UniformElectricField'){text(named.id+': E='+vec(s.field_v_m)+' V/m',[18,22+u*17]);u++;continue;}
if(s.start_m){line(X(s.start_m),X(s.end_m),'#72869b',4);text(named.id,X(s.start_m));}
else if(s.radius_m){const p=X(s.center_m);ctx.strokeStyle='#72869b';ctx.lineWidth=2;ctx.beginPath();ctx.arc(...p,s.radius_m*sc,0,2*Math.PI);ctx.stroke();text(named.id,[p[0]+5,p[1]-8]);}
else if(s.position_m){const p=X(s.position_m);dot(p,4,'#72869b');text(named.id,[p[0]+7,p[1]-7]);}}
const o=r.observation,a=X(o.positive_position_m),b=X(o.negative_position_m),h=X(pivot);
line(a,b,'#263d55',3);dot(a,6,'#bf4133');dot(b,6,'#2464b2');text('+q',[a[0]+8,a[1]+18],'#bf4133');text('−q',[b[0]+8,b[1]+18],'#2464b2');
line([h[0]-6,h[1]-6],[h[0]+6,h[1]+6],'#804da5');line([h[0]-6,h[1]+6],[h[0]+6,h[1]-6],'#804da5');
if($('forces').checked){arrow(o.positive_position_m,o.positive_force_n,'#bf4133');arrow(o.negative_position_m,o.negative_force_n,'#2464b2');arrow(o.dipole.center_m,o.net_force_n,'#188366');}
// Only the projected axial torque's handedness is drawn; magnitude is read.
const tz=r.axis_torque_n_m*n[2];
if(Math.abs(tz)>0){ctx.strokeStyle='#804da5';ctx.lineWidth=2;ctx.beginPath();ctx.arc(h[0],h[1],40,0,tz>0?-Math.PI/2:Math.PI/2,tz>0);ctx.stroke();
text(tz>0?'τ_axis ↺':'τ_axis ↻',[h[0]+43,h[1]-12],'#804da5');}
text('τ_axis = '+fmt(r.axis_torque_n_m)+' N m (full value, not arc size)',[18,358],'#804da5');
text('Fixed ideal hinge; rotation axis n='+vec(n),[18,377]);
}
function plot(id,keys,colors,unit,index){const ca=$(id),g=ca.getContext('2d');g.clearRect(0,0,ca.width,ca.height);
let v=rows.flatMap(r=>keys.map(k=>r[k])),lo=Math.min(0,...v),hi=Math.max(0,...v);if(lo===hi){lo-=1;hi+=1;}
const pad=(hi-lo)*.08;lo-=pad;hi+=pad;
const xx=t=>80+430*t/rows.at(-1).time_s, yy=v=>205-160*(v-lo)/(hi-lo);
g.font='11px system-ui';g.fillStyle='#334d60';
for(let j=0;j<5;j++){const v=lo+(hi-lo)*j/4,y=yy(v);g.strokeStyle='#dbe5ed';g.beginPath();g.moveTo(80,y);g.lineTo(510,y);g.stroke();g.fillText(v.toExponential(1),4,y+3);}
keys.forEach((k,j)=>{g.fillStyle=colors[j];for(let i=0;i<rows.length;i++){g.beginPath();g.arc(xx(rows[i].time_s),yy(rows[i][k]),i===index?4:1.7,0,2*Math.PI);g.fill();}});
g.fillStyle='#243e54';g.fillText('0 s',70,228);g.fillText(fmt(rows.at(-1).time_s)+' s',410,228);g.fillText('Saved physical time (s) · '+unit,125,250);
}
let index=0,playing=false,last=0,clock=0;
function draw(){const r=rows[index],o=r.observation;$('time').value=index;
$('clock').textContent='Physical t = '+fmt(r.time_s)+' s · sample '+(index+1)+' / '+rows.length;
$('numbers').textContent=`θ (unwrapped rad): ${fmt(r.angle_rad)}\nω (rad/s): ${fmt(r.angular_velocity_rad_s)}\nα (rad/s²): ${fmt(r.angular_acceleration_rad_s2)}\nτ_axis (N m): ${fmt(r.axis_torque_n_m)}\nFull external torque (N m): ${vec(o.torque_n_m)}\nF+ (N): ${vec(o.positive_force_n)}\nF− (N): ${vec(o.negative_force_n)}\nResultant external F (N): ${vec(o.net_force_n)}\nK_rotation (J): ${fmt(r.kinetic_energy_j)}\nU_external (J): ${fmt(r.interaction_energy_j)}\nTotal K+U (J): ${fmt(r.total_energy_j)}\nEnergy drift (J): ${fmt(r.energy_drift_j)}\n+q position (m): ${vec(o.positive_position_m)}\n−q position (m): ${vec(o.negative_position_m)}`;
scene(r);plot('energy',['kinetic_energy_j','interaction_energy_j','total_energy_j'],['#bd6b27','#2464b2','#188366'],'Energy (J)',index);
plot('angleplot',['angle_rad'],['#804da5'],'Unwrapped angle (rad)',index);
$('play').textContent=playing?'Pause':'Play';
window.dipoleMotion={index,playing,sampleCount:rows.length,time_s:r.time_s,mode:'elapsed-time-fixed-axis',termination:data.termination};
}
function pause(){playing=false;draw();}
$('time').max=rows.length-1;$('time').addEventListener('input',()=>{playing=false;index=Number($('time').value);clock=rows[index].time_s;draw();});
$('reset').onclick=()=>{playing=false;index=0;clock=0;draw();};
$('forces').onchange=draw;
$('play').onclick=()=>{if(playing){pause();return;}if(index===rows.length-1){index=0;clock=0;}else clock=rows[index].time_s;playing=true;last=performance.now();draw();};
function tick(now){if(playing){clock+=(now-last)/1000*Number($('speed').value);last=now;let next=index;while(next+1<rows.length&&rows[next+1].time_s<=clock)next++;if(next!==index){index=next;draw();}
if(clock>=rows.at(-1).time_s){index=rows.length-1;pause();}}requestAnimationFrame(tick);}
draw();requestAnimationFrame(tick);
"""

def render_dipole_motion(result, *, title='Rigid dipole · real fixed-axis motion'):
    validate_dipole_motion(result)
    record=result.as_dict()
    if len(record['samples'])<2:
        raise ValueError('no elapsed trajectory to display; retain the numerical guard result')
    spec=record['spec']
    inventory=escape(json.dumps(spec['model'],indent=2))
    payload=json.dumps(record,allow_nan=False,separators=(',',':')).replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e')
    controls=('<section class="card"><h2>Saved physical-time playback</h2><button id="play">Play</button> '
              '<button id="reset">Reset</button> <label>Speed <select id="speed"><option>0.25</option><option>0.5</option><option selected>1</option><option>2</option><option>4</option></select> physical seconds / wall second</label>'
              '<input id="time" type="range" min="0" step="1" value="0"><strong id="clock"></strong>'
              '<p class="muted">Only computed samples are displayed. No browser physics, interpolated pose, or inferred collision.</p></section>')
    status=('Completed requested physical duration.' if result.termination=='duration' else
            'SOURCE GUARD: this trajectory is incomplete. It stops before the singular source or field jump; no impact is simulated.')
    html=('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+
        escape(title)+'</title><style>'+_STYLE+
        'canvas{width:100%;height:auto;background:#fbfdff}button,select{padding:6px 12px;font:inherit}input[type=checkbox]{width:auto} .grid{grid-template-columns:1.1fr 1fr}@media(max-width:850px){.grid{grid-template-columns:1fr}}</style></head><body><header><small>CONDUCTOR_SIM / FIXED-AXIS DIPOLE DYNAMICS</small><h1>'+
        escape(title)+'</h1><p>'+escape(spec['model']['provenance'])+'</p></header><main><section class="card"><p id="termination">'+escape(status)+
        '</p><p>Explicit inertia about the fixed axis: <strong>'+escape(f'{spec["inertia_kg_m2"]:.6g}')+
        ' kg m²</strong>. Ideal frictionless hinge: no free translation, no damping, and no gravity on an unspecified mass distribution.</p></section>'+controls+
        '<div class="grid"><section class="card"><h2>World x–y orthographic projection</h2><p class="muted">Depth is not shown. Full three-dimensional endpoint forces and torque are retained below; projection never defines the physical force.</p><canvas id="scene" width="600" height="390" aria-label="Dipole and sources at computed physical time"></canvas><label><input id="forces" type="checkbox" checked> Show actual projected electric forces</label><p class="muted">Red +q, blue −q, green net external force; common force-arrow scale across every sample. Cross: fixed pivot. Purple arc gives only projected axial torque direction, not magnitude or angular velocity. The support may react other force/torque components without work; support reactions are not calculated.</p></section>'+
        '<section class="card"><h2>Rotational energy exchange</h2><canvas id="energy" width="550" height="265" aria-label="Saved kinetic potential and total energy"></canvas><p class="muted">Orange K, blue U_external, green total. Fixed-length internal binding energy is excluded. No energy correction or normalization is applied.</p><h2>Unwrapped angular motion</h2><canvas id="angleplot" width="550" height="265" aria-label="Computed angular motion versus physical time"></canvas><p class="muted">Dots are actual saved values. Sign-change intervals are brackets, not exact turning times.</p></section></div>'+
        '<section class="card"><h2>Full physical observations in SI</h2><div id="numbers" class="vectors"></div></section><section class="card"><details><summary>Immutable authored model, source identities and physical coordinates</summary><pre class="vectors">'+inventory+
        '</pre></details></section></main><script id="motion-data" type="application/json">'+payload+
        '</script><script>'+_SCRIPT+'</script></body></html>')
    return html, record


def write_dipole_motion(result, output, **kwargs):
    html,record=render_dipole_motion(result,**kwargs)
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(html,encoding='utf-8')
    path.with_suffix('.json').write_text(json.dumps(record,allow_nan=False,indent=2)+'\n',encoding='utf-8')
    return path
