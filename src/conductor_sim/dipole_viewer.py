"""Offline static dipole investigation; no browser field solve or time animation."""
from html import escape
import json
from math import pi
from pathlib import Path

from .dipoles import sample_orientations, uniform_rotational_equilibria
from .solver import UnsupportedProblem
from .prescribed_fields import UniformRingAxisField, UniformDiskAxisField

_STYLE = '''body{margin:0;background:#f2f5f9;color:#183047;font:15px/1.5 system-ui,sans-serif}
header{background:#17334b;color:white;padding:22px max(18px,calc((100vw - 1220px)/2))}
main{max-width:1220px;margin:20px auto;padding:0 16px}h1{font-size:25px;margin:3px 0}h2{font-size:18px;margin:0 0 10px}
.card{background:white;border:1px solid #c8d6e2;border-radius:8px;padding:16px;margin:14px 0}.grid{display:grid;grid-template-columns:1.2fr 1fr;gap:16px}
svg{width:100%;height:auto;background:#fafcff}input{width:100%}table{border-collapse:collapse;width:100%;font-size:13px}
td,th{text-align:left;padding:7px;border-bottom:1px solid #dce5ee;overflow-wrap:anywhere}th{font-weight:600}
.muted{color:#4d6479;font-size:13px}.warning{background:#fff1dc;padding:9px}.vectors{font:12px/1.6 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere}
@media(max-width:850px){.grid{grid-template-columns:1fr}header{padding:18px}h1{font-size:21px}}'''

_SCRIPT = r'''
const data=JSON.parse(document.getElementById('dipole-data').textContent), rows=data.rows;
const $=id=>document.getElementById(id),ns='http://www.w3.org/2000/svg';
function node(tag,attrs,parent,text){let e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(text!==undefined)e.textContent=text;parent.appendChild(e);return e}
const fmt=x=>x===null?'undefined':Number(x).toPrecision(6), vector=v=>v===null?'undefined':v.map(fmt).join(', ');
const valid=rows.filter(r=>r.observation).map(r=>r.observation);
const sourceList=data.model.sources, positions=[];
for(const o of valid)positions.push(o.positive_position_m,o.negative_position_m,o.torque_origin_m);
for(const s of sourceList){const v=s.source;if(v.position_m)positions.push(v.position_m);if(v.center_m){positions.push(v.center_m);if(v.radius_m)positions.push([v.center_m[0]+v.radius_m,v.center_m[1]+v.radius_m,0],[v.center_m[0]-v.radius_m,v.center_m[1]-v.radius_m,0])}if(v.start_m)positions.push(v.start_m,v.end_m)}
let cx=data.model.dipole.center_m[0],cy=data.model.dipole.center_m[1];
let span=Math.max(data.model.dipole.separation_m*3,...positions.map(p=>Math.max(Math.abs(p[0]-cx),Math.abs(p[1]-cy))*1.25));
// Equal metre scale on both axes, not independent distortion.
const scale=Math.min(240,155)/span;const XY=p=>[300+(p[0]-cx)*scale,195-(p[1]-cy)*scale];
const fmax=Math.max(0,...valid.flatMap(o=>[o.positive_force_n,o.negative_force_n,o.net_force_n].filter(Boolean).map(f=>Math.hypot(...f))));
function arrow(svg,p,f,color,tag){if(!f||!fmax)return;let [x,y]=XY(p),dx=70*f[0]/fmax,dy=-70*f[1]/fmax;if(Math.hypot(dx,dy)<.01)return;
node('line',{x1:x,y1:y,x2:x+dx,y2:y+dy,stroke:color,'stroke-width':2.5,'data-force':tag},svg);
let a=Math.atan2(dy,dx);node('path',{d:`M${x+dx-7*Math.cos(a-.5)},${y+dy-7*Math.sin(a-.5)} L${x+dx},${y+dy} L${x+dx-7*Math.cos(a+.5)},${y+dy-7*Math.sin(a+.5)}`,fill:'none',stroke:color,'stroke-width':2.5},svg)}
function drawScene(o){let svg=$('scene');svg.replaceChildren();
node('line',{x1:30,y1:340,x2:120,y2:340,stroke:'#718398'},svg);node('text',{x:125,y:344,'font-size':13},svg,'+x');
node('line',{x1:30,y1:340,x2:30,y2:270,stroke:'#718398'},svg);node('text',{x:25,y:260,'font-size':13},svg,'+y');
for(const named of sourceList){let s=named.source;
 if(s.kind==='UniformElectricField'){node('text',{x:20,y:25+18*sourceList.indexOf(named),'font-size':12},svg,`${named.id}: E=(${vector(s.field_v_m)}) V/m`);continue}
 if(s.start_m){let a=XY(s.start_m),b=XY(s.end_m);node('line',{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:'#72869b','stroke-width':5},svg)}
 let p=s.position_m||s.center_m||s.start_m; if(p){let [x,y]=XY(p);node('circle',{cx:x,cy:y,r:s.radius_m?s.radius_m*scale:5,fill:'none',stroke:'#72869b','stroke-width':2},svg);node('text',{x:x+8,y:y-8,'font-size':12},svg,named.id)}
}
if(!o)return;
let a=XY(o.positive_position_m),b=XY(o.negative_position_m),c=XY(o.torque_origin_m);
node('line',{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:'#263d55','stroke-width':3},svg);
for(const [p,color,label] of [[a,'#bf4133','+q'],[b,'#2464b2','−q']]){node('circle',{cx:p[0],cy:p[1],r:6,fill:color},svg);node('text',{x:p[0]+8,y:p[1]+17,'font-size':14,fill:color},svg,label)}
node('path',{d:`M${c[0]-5},${c[1]-5} l10,10 m0,-10 l-10,10`,stroke:'#7f4ba5','stroke-width':2},svg);
arrow(svg,o.positive_position_m,o.positive_force_n,'#bc3f32','positive');arrow(svg,o.negative_position_m,o.negative_force_n,'#2464b2','negative');arrow(svg,o.dipole.center_m,o.net_force_n,'#1a8569','net');
if(o.torque_n_m&&Math.abs(o.torque_n_m[2])>0){let sign=o.torque_n_m[2]>0?1:-1, x=c[0],y=c[1];
 node('path',{d:`M${x+30},${y} A30,30 0 0 ${sign>0?0:1} ${x},${y-sign*30}`,fill:'none',stroke:'#8a4ba8','stroke-width':2,'data-torque':'z'},svg);
 node('text',{x:x-28,y:y-sign*35,'font-size':12,fill:'#8a4ba8'},svg,sign>0?'τz ↺':'τz ↻')}
}
function drawPlot(index){let svg=$('energy');svg.replaceChildren();let lo=Math.min(...valid.map(o=>o.interaction_energy_j)),hi=Math.max(...valid.map(o=>o.interaction_energy_j));if(hi===lo){lo-=1;hi+=1}
let a=rows[0].angle_rad,b=rows.at(-1).angle_rad,xx=t=>65+440*(t-a)/(b-a),yy=u=>205-160*(u-lo)/(hi-lo);
for(let i=0;i<5;i++){let u=lo+(hi-lo)*i/4;node('line',{x1:65,y1:yy(u),x2:505,y2:yy(u),stroke:'#dfe7ef'},svg);node('text',{x:60,y:yy(u)+4,'font-size':11,'text-anchor':'end'},svg,u.toExponential(2))}
for(const [j,r] of rows.entries())if(r.observation)node('circle',{cx:xx(r.angle_rad),cy:yy(r.observation.interaction_energy_j),r:j===index?5:2.5,fill:j===index?'#bb4c2b':'#2464b2','data-sample':j},svg);
for(const t of [a,(a+b)/2,b])node('text',{x:xx(t),y:228,'font-size':12,'text-anchor':'middle'},svg,(t*180/Math.PI).toFixed(0)+'°');
node('text',{x:280,y:251,'font-size':13,'text-anchor':'middle'},svg,'Selected orientation angle (degrees)');node('text',{x:280,y:18,'font-size':13,'text-anchor':'middle'},svg,'External interaction energy (J) — sampled points only')}
function draw(){let i=Number($('angle').value),r=rows[i],o=r.observation;
$('angle-label').textContent=(r.angle_rad*180/Math.PI).toFixed(1)+'° · independent static pose '+(i+1)+' / '+rows.length;
$('status').textContent=o?'No elapsed time, angular velocity or continuous rotation is implied.':r.reason;
drawScene(o);drawPlot(i);
$('values').textContent=o?`F+ (N): [${vector(o.positive_force_n)}]\nF− (N): [${vector(o.negative_force_n)}]\nNet F (N): [${vector(o.net_force_n)}]\nTorque about marked origin (N m): [${vector(o.torque_n_m)}]\np (C m): [${vector(o.dipole_moment_c_m)}]\nU_external (J): ${fmt(o.interaction_energy_j)}\n+q position (m): [${vector(o.positive_position_m)}]\n−q position (m): [${vector(o.negative_position_m)}]`:'Potential undefined at this configuration.';
const initial=rows[0].observation;
if(o&&initial){let change=o.interaction_energy_j-initial.interaction_energy_j;$('values').textContent+=`\nΔU / external quasistatic endpoint work (J): ${fmt(change)}\nElectric endpoint work (J): ${fmt(-change)}`;}
$('force-scale').textContent=`Projected force arrows: ${fmax?fmt(fmax/70):'zero'} N per SVG unit, shared across all poses.`;
let reasons=o?o.source_contributions.flatMap(s=>s.endpoint_field_unavailable.filter(Boolean)):[];
$('unavailable').hidden=!reasons.length;$('unavailable').textContent='Unavailable field/force/torque: '+reasons.join('; ');
window.dipoleExplorer={mode:'static-orientation',index:i,sampleCount:rows.length,modelDigest:data.model_digest};}
$('angle').max=rows.length-1;$('angle').addEventListener('input',draw);draw();
'''


def render_dipole_explorer(model, angles_rad, *, title='Rigid dipole: static investigation',
                            source_note='', rotation_axis=(0., 0., 1.)):
    """Compute observations from typed model; no arbitrary unverified import path."""
    if any(type(s.source) in (UniformRingAxisField, UniformDiskAxisField) for s in model.sources):
        raise UnsupportedProblem('M23a projected apparatus viewer does not depict axis-only ring/disc sources; use numerical observations')
    record = sample_orientations(model, angles_rad, rotation_axis=rotation_axis)
    if not any(r['observation'] for r in record['rows']):
        raise UnsupportedProblem('no finite potential observations to display')
    try:
        equilibria = uniform_rotational_equilibria(model)
        eq = json.dumps(equilibria)
    except UnsupportedProblem:
        equilibria = None
        eq = 'Nonuniform field: no complete equilibrium/stability search is claimed.'
    record['equilibria'] = equilibria
    inventory = ''.join('<tr><td>'+escape(s.source_id)+'</td><td>'+escape(s.label)+
        '</td><td>'+escape(type(s.source).__name__)+'</td></tr>' for s in model.sources)
    parameters = escape(json.dumps([s.as_dict() for s in model.sources], indent=2))
    payload = json.dumps(record, allow_nan=False, separators=(',', ':')).replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e')
    html = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+escape(title)+
        '</title><style>'+_STYLE+'</style></head><body><header><small>CONDUCTOR_SIM / STATIC DIPOLE OBSERVER</small><h1>'+escape(title)+
        '</h1><p>'+escape(source_note or model.provenance)+'</p></header><main><section class="card"><label for="angle"><strong>Select a saved orientation — not time playback</strong></label><input id="angle" type="range" min="0" step="1" value="0"><div id="angle-label"></div><p id="status" class="muted"></p></section>'+
        '<div class="grid"><section class="card"><h2>World x–y projection of the authored configuration</h2><p class="muted">Depth is not shown. All numerical vectors below retain x, y, z. No projected force is used in calculations.</p><svg id="scene" viewBox="0 0 600 375" role="img" aria-label="Explicit world xy projection of dipole and sources"></svg><p id="force-scale" class="muted"></p><p class="muted">Force arrows: red +q, blue −q, green resultant. One fixed force scale for all poses; perpendicular components are in the readout. Cross: torque origin. Purple arc indicates only the sign of τz, not torque magnitude or motion.</p></section>'+
        '<section class="card"><h2>Energy versus authored orientation</h2><svg id="energy" viewBox="0 0 560 265" role="img" aria-label="Saved static energy samples"></svg><p class="muted">Samples are not joined: unsampled orientations and admissible paths are not certified. Only external interaction energy is shown; fixed-length internal binding energy is excluded.</p><p id="equilibria" class="muted">'+escape(eq)+'</p></section></div>'+
        '<section class="card"><h2>Full-vector observations in SI</h2><div id="values" class="vectors"></div><p id="unavailable" class="warning" hidden></p></section>'+
        '<section class="card"><h2>Unchanged external source inventory</h2><table><tr><th>Identity</th><th>Label</th><th>Component</th></tr>'+inventory+'</table><details><summary>Source parameters (SI; full 3D coordinates)</summary><pre class="vectors">'+parameters+'</pre></details><p class="muted">Sources are held and do not respond to this rigid pair. A pivot is an authored static investigation constraint; an arbitrary pose need not be an equilibrium and may require external holding forces/torques. No joint dynamics, guide force or moment of inertia is inferred.</p></section></main>'+
        '<script id="dipole-data" type="application/json">'+payload+'</script><script>'+_SCRIPT+'</script></body></html>')
    return html, record


def write_dipole_explorer(model, angles_rad, output, **options):
    html, record = render_dipole_explorer(model, angles_rad, **options)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding='utf-8')
    path.with_suffix('.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return path
