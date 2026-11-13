"""Offline, dependency-free HTML canvas viewer for visualization.v1 bundles."""
from html import escape
import json
import re


def render_html(bundle, *, title='Conductor simulation'):
    if bundle.get('schema') != 'conductor_sim.visualization.v1' or not bundle.get('frames'):
        raise ValueError('expected a nonempty conductor_sim.visualization.v1 bundle')
    data = json.dumps(bundle,allow_nan=False,separators=(',',':'))
    # JSON lives inside a script element: HTML escaping alone is insufficient.
    data = data.replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e')
    replacements = {'__TITLE__':escape(str(title),quote=True),'__DATA__':data}
    return re.sub(r'__(?:TITLE|DATA)__',lambda m:replacements[m.group(0)],TEMPLATE)


TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:light;--ink:#142334;--muted:#526579;--line:#d9e2eb;--blue:#165884}*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:var(--ink);font:15px/1.5 system-ui,-apple-system,sans-serif}header{background:#122b40;color:white;padding:24px max(20px,calc((100vw - 1260px)/2));border-bottom:4px solid #3ca5b0}h1{font-size:26px;margin:4px 0}h2{font-size:17px;margin:0 0 14px}p{margin:8px 0}.kicker{font-size:11px;letter-spacing:.16em;text-transform:uppercase;opacity:.75}.sub{color:#cbd9e4;max-width:850px;font-size:13px}main{max-width:1300px;margin:auto;padding:22px;display:grid;grid-template-columns:minmax(440px,1.6fr) minmax(310px,1fr);gap:20px}.card{background:white;border:1px solid var(--line);border-radius:10px;padding:20px;box-shadow:0 4px 16px #14233406}.wide{grid-column:1/-1}.row{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}.chip{font:12px ui-monospace,monospace;background:#e7f3f6;color:#17556a;padding:5px 9px;border-radius:5px}canvas{display:block;width:100%;height:auto}#scene{background:#f9fbfd;border:1px solid var(--line);border-radius:5px}#profile{border:1px solid var(--line);margin-top:12px}label{font-size:13px;cursor:pointer}select,button{padding:7px;border:1px solid #bbcbd7;border-radius:4px;background:white;color:var(--ink)}input[type=range]{flex:1;min-width:160px}input[type=checkbox]{accent-color:#176488}#case{font-weight:650;font-size:18px}#probe{font:12px/1.6 ui-monospace,monospace;min-height:42px;margin-top:9px;color:#344d65}table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:7px 5px;text-align:right;border-bottom:1px solid #e6ebf0}th:first-child,td:first-child{text-align:left}.small{font-size:12px;color:var(--muted)}#scale{height:12px;background:linear-gradient(90deg,#3476ae,#f8f9fb,#d95b43);border:1px solid var(--line);margin-top:10px}.ticks{display:flex;justify-content:space-between;font:11px ui-monospace,monospace;color:var(--muted)}.controls{margin:14px 0 12px;display:flex;gap:18px;flex-wrap:wrap}#notes p{padding-left:12px;border-left:2px solid #d9e6ef;font-size:12px;color:var(--muted)}aside .card{margin-bottom:20px}.error{color:#b52222}@media(max-width:850px){main{grid-template-columns:1fr}.wide{grid-column:auto}header{padding:18px}h1{font-size:22px}.card{padding:15px}}
</style></head><body>
<header><div class="kicker">CONDUCTOR_SIM / M5 VISUALIZATION</div><h1>__TITLE__</h1><p class="sub">Potential, field geometry and induced surface charge from one reusable sampling pipeline. No server or solver is needed to inspect this exported file.</p></header>
<main><section class="card wide"><div class="row"><span id="case"></span><span id="backend" class="chip"></span></div><div class="row" style="margin-top:14px"><label for="frame">Stored state</label><input id="frame" type="range" min="0" step="1" value="0"><span id="counter" class="small"></span></div><p class="small">The slider selects precomputed solutions. It does not recompute FEMM or simulate charge motion in the browser.</p></section>
<section class="card"><div class="row"><h2>Planar cross-section</h2><span class="small">metres · V · V/m</span></div><div class="controls"><label><input type="checkbox" id="heat" checked> Potential</label><label><input type="checkbox" id="contours" checked> Equipotentials</label><label><input type="checkbox" id="lines" checked> Field lines</label></div><canvas id="scene" width="720" height="720" aria-label="Potential map and electric field lines"></canvas><div id="scale"></div><div class="ticks"><span id="lo"></span><span>0 V</span><span id="hi"></span></div><div id="probe">Move the pointer over the cross-section to inspect the nearest stored sample.</div><p class="small">Grey regions: conducting metal. Dashed source discs: excluded regions. Arrows follow E; line count does not represent flux. Contours are sampled interpolations, not exact curves.</p></section>
<aside><section class="card"><h2>Conductor results</h2><div style="overflow:auto"><table><thead><tr><th>Name</th><th>V</th><th>Net nC</th><th>Inner nC</th><th>Outer nC</th></tr></thead><tbody id="conductors"></tbody></table></div><p class="small" id="totals-note"></p></section><section class="card"><h2>Induced surface-charge profile</h2><div class="row"><label for="density">Conductor / face</label><select id="density"></select></div><canvas id="profile" width="450" height="260" aria-label="Induced charge density along a great circle"></canvas><p id="method" class="small"></p><p class="small">For spherical conductors the horizontal axis is polar angle; for planes it is signed distance (m) along the cross-section. These lines are not area integrals.</p></section><section class="card"><h2>Source and sampling details</h2><div id="sources" class="small"></div><div id="notes"></div></section></aside>
</main><script id="simulation-data" type="application/json">__DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('simulation-data').textContent);
const $=id=>document.getElementById(id), canvas=$('scene'),ctx=canvas.getContext('2d');
let frame=data.frames[0],map=null;
const fmt=x=>Number(x)===0?'0':Number(x).toPrecision(5);
const resetProbe=()=>{$('probe').textContent='Move the pointer over the cross-section to inspect the nearest stored sample.'};
const escText=(parent,text,tag='p')=>{const e=document.createElement(tag);e.textContent=text;parent.appendChild(e);return e};
$('frame').max=data.frames.length-1;const lim=data.color_limit_v;
$('lo').textContent=fmt(-lim)+' V';$('hi').textContent=fmt(lim)+' V';
function colour(value){const t=Math.max(-1,Math.min(1,value/lim));const a=[248,249,251], b=t<0?[52,118,174]:[217,91,67];return `rgb(${a.map((x,i)=>Math.round(x+Math.abs(t)*(b[i]-x))).join(',')})`}
function drawScene(){
 const p=frame.plane,g=frame.grid,W=canvas.width,H=canvas.height,pad=46;
 const scale=Math.min((W-2*pad)/(p.x_range_m[1]-p.x_range_m[0]),(H-2*pad)/(p.y_range_m[1]-p.y_range_m[0]));
 const ox=(W-scale*(p.x_range_m[1]-p.x_range_m[0]))/2,oy=(H-scale*(p.y_range_m[1]-p.y_range_m[0]))/2;
 const X=x=>ox+(x-p.x_range_m[0])*scale,Y=y=>H-oy-(y-p.y_range_m[0])*scale;map={X,Y,scale,ox,oy};
 ctx.clearRect(0,0,W,H);ctx.fillStyle='#f9fbfd';ctx.fillRect(0,0,W,H);
 if($('heat').checked){const dx=(g.x_m[1]-g.x_m[0])*scale,dy=(g.y_m[1]-g.y_m[0])*scale;
 for(let j=0;j<g.y_m.length;j++)for(let i=0;i<g.x_m.length;i++){const v=g.potential_v[j][i];if(v!==null){ctx.fillStyle=colour(v);ctx.fillRect(X(g.x_m[i])-dx/2,Y(g.y_m[j])-dy/2,dx+1,dy+1)}}}
 ctx.strokeStyle='#d0dae4';ctx.lineWidth=1;ctx.setLineDash([3,5]);for(const z of [0]){ctx.beginPath();ctx.moveTo(X(z),oy);ctx.lineTo(X(z),H-oy);ctx.moveTo(ox,Y(z));ctx.lineTo(W-ox,Y(z));ctx.stroke()}ctx.setLineDash([]);
 if($('contours').checked){ctx.strokeStyle='#49677988';ctx.lineWidth=1;for(const c of frame.contours){ctx.beginPath();for(const seg of c.segments_m){ctx.moveTo(X(seg[0][0]),Y(seg[0][1]));ctx.lineTo(X(seg[1][0]),Y(seg[1][1]))}ctx.stroke()}}
 if($('lines').checked){ctx.strokeStyle='#183d56';ctx.fillStyle='#183d56';ctx.lineWidth=1.25;for(const l of frame.field_lines){const q=l.points_m;if(q.length<2)continue;ctx.beginPath();q.forEach((v,i)=>i?ctx.lineTo(X(v[0]),Y(v[1])):ctx.moveTo(X(v[0]),Y(v[1])));ctx.stroke();const k=Math.floor((q.length-1)*.58),a=q[k],b=q[Math.min(k+1,q.length-1)];const angle=Math.atan2(Y(b[1])-Y(a[1]),X(b[0])-X(a[0]));ctx.save();ctx.translate(X(a[0]),Y(a[1]));ctx.rotate(angle);ctx.beginPath();ctx.moveTo(4,0);ctx.lineTo(-3,-2.8);ctx.lineTo(-3,2.8);ctx.closePath();ctx.fill();ctx.restore()}}
 for(const s of frame.geometry){
  if(!s.visible)continue;
  if(s.kind==='plane'){
   // Clip the physical conducting half-space to the current view rectangle.
   const [px,py]=s.point_m,[nx,ny]=s.normal_xy;
   const xlo=p.x_range_m[0],xhi=p.x_range_m[1],ylo=p.y_range_m[0],yhi=p.y_range_m[1];
   const corners=[[xlo,ylo],[xhi,ylo],[xhi,yhi],[xlo,yhi]];
   const signed=q=>nx*(q[0]-px)+ny*(q[1]-py);
   let poly=corners;
   const clipped=[];
   for(let i=0;i<poly.length;i++){
    const a=poly[i],b=poly[(i+1)%poly.length],da=signed(a),db=signed(b);
    if(da<=0)clipped.push(a);
    if((da<0&&db>0)||(da>0&&db<0)){
     const t=da/(da-db);clipped.push([a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1])]);
    }
   }
   if(clipped.length){ctx.beginPath();clipped.forEach((q,i)=>i?ctx.lineTo(X(q[0]),Y(q[1])):ctx.moveTo(X(q[0]),Y(q[1])));ctx.closePath();ctx.fillStyle='#afbdc9';ctx.fill()}
   const length=Math.hypot(xhi-xlo,yhi-ylo)*2,norm=Math.hypot(nx,ny),tx=-ny/norm,ty=nx/norm;
   ctx.save();ctx.beginPath();ctx.rect(ox,oy,W-2*ox,H-2*oy);ctx.clip();
   ctx.strokeStyle='#586d7f';ctx.lineWidth=2;ctx.beginPath();
   ctx.moveTo(X(px-length*tx),Y(py-length*ty));ctx.lineTo(X(px+length*tx),Y(py+length*ty));ctx.stroke();ctx.restore();
  }else{
   const [x,y]=s.center_m;ctx.beginPath();ctx.arc(X(x),Y(y),s.outer_radius_m*scale,0,Math.PI*2);
   if(s.inner_radius_m>0)ctx.arc(X(x),Y(y),s.inner_radius_m*scale,0,Math.PI*2,true);
   ctx.fillStyle='#afbdc9';ctx.fill('evenodd');ctx.strokeStyle='#586d7f';ctx.lineWidth=1;ctx.stroke();
  }
 }
 function project(v){const d=v.map((x,i)=>x-p.origin_m[i]);const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);const n=[p.u[1]*p.v[2]-p.u[2]*p.v[1],p.u[2]*p.v[0]-p.u[0]*p.v[2],p.u[0]*p.v[1]-p.u[1]*p.v[0]];return[dot(d,p.u),dot(d,p.v),dot(d,n)]}
 for(const m of frame.source_masks){const [x,y,h]=project(m.center_m);if(Math.abs(h)>=m.radius_m)continue;ctx.beginPath();ctx.arc(X(x),Y(y),Math.sqrt(m.radius_m*m.radius_m-h*h)*scale,0,Math.PI*2);ctx.fillStyle='#fff';ctx.fill();ctx.setLineDash([3,2]);ctx.strokeStyle='#7d8d9c';ctx.stroke();ctx.setLineDash([])}
 for(const q of frame.charges){const [x,y,h]=q.slice_position_m;if(Math.abs(h)>1e-10)continue;ctx.beginPath();ctx.arc(X(x),Y(y),4,0,Math.PI*2);ctx.fillStyle=q.charge_c>0?'#b83121':'#175ba3';ctx.fill();ctx.fillStyle='#10273a';ctx.font='bold 14px system-ui';ctx.fillText(q.charge_c>0?'+':'−',X(x)+7,Y(y)-7)}
 ctx.fillStyle='#f9fbfd';ctx.fillRect(0,0,W,oy-1);ctx.fillRect(0,H-oy+1,W,oy);ctx.fillRect(0,0,ox-1,H);ctx.fillRect(W-ox+1,0,ox,H);
 ctx.strokeStyle='#99acbc';ctx.strokeRect(ox,oy,W-2*ox,H-2*oy);ctx.fillStyle='#4c6274';ctx.font='12px system-ui';ctx.textAlign='center';
 for(let i=0;i<=4;i++){const x=p.x_range_m[0]+i*(p.x_range_m[1]-p.x_range_m[0])/4,y=p.y_range_m[0]+i*(p.y_range_m[1]-p.y_range_m[0])/4;ctx.fillText(x.toFixed(3),X(x),H-oy+18);ctx.textAlign='right';ctx.fillText(y.toFixed(3),ox-7,Y(y)+4);ctx.textAlign='center'}ctx.fillText('slice u (m)',W/2,H-7);ctx.save();ctx.translate(12,H/2);ctx.rotate(-Math.PI/2);ctx.fillText('slice v (m)',0,0);ctx.restore();ctx.textAlign='left';
}
function drawProfile(){
 const pc=$('profile'),c=pc.getContext('2d'),W=pc.width,H=pc.height;
 c.clearRect(0,0,W,H);
 const series=frame.profiles[Number($('density').value)];
 if(!series){$('method').textContent='Surface density unavailable for this state.';c.textAlign='left';c.fillStyle='#536a7e';c.font='12px system-ui';c.fillText('Surface-density API not available.',25,50);return}
 const isPlane=Array.isArray(series.coordinate_m);
 const xs=isPlane?series.coordinate_m:series.theta_rad;
 const left=xs[0],right=xs[xs.length-1],ys=series.sigma_c_per_m2.map(x=>x*1e9);
 let lo=Math.min(0,...ys),hi=Math.max(0,...ys),span=hi-lo||1;lo-=.08*span;hi+=.08*span;
 const X=t=>64+(t-left)/(right-left)*(W-84),Y=v=>H-40-(v-lo)/(hi-lo)*(H-67);
 c.strokeStyle='#c5d1dc';c.beginPath();c.moveTo(64,20);c.lineTo(64,H-40);c.lineTo(W-20,H-40);c.stroke();
 c.font='11px system-ui';c.fillStyle='#536a7e';c.textAlign='right';
 for(let i=0;i<4;i++){const v=lo+i*(hi-lo)/3;c.fillText(v.toPrecision(3),58,Y(v)+4)}
 c.textAlign='center';for(let i=0;i<=4;i++){const x=left+i*(right-left)/4;
 c.fillText(isPlane?x.toPrecision(2):Math.round(x*180/Math.PI)+'°',X(x),H-22)}
 c.textAlign='left';c.fillText('σ  [nC/m²]',65,15);
 c.strokeStyle='#165884';c.lineWidth=2;c.beginPath();
 xs.forEach((x,i)=>i?c.lineTo(X(x),Y(ys[i])):c.moveTo(X(x),Y(ys[i])));c.stroke();
 $('method').textContent=series.method+(isPlane?' · horizontal axis: signed distance (m)':'');
}
function selectFrame(){frame=data.frames[Number($('frame').value)];resetProbe();$('case').textContent=frame.label;$('counter').textContent=(Number($('frame').value)+1)+' / '+data.frames.length;$('backend').textContent=frame.backend;$('conductors').replaceChildren();for(const q of frame.conductor_results){const tr=document.createElement('tr');for(const v of [q.name,fmt(q.potential_v),q.net_charge_c===null?'n/a':fmt(q.net_charge_c*1e9),q.inner_surface_charge_c===null?'n/a':fmt(q.inner_surface_charge_c*1e9),q.outer_surface_charge_c===null?'n/a':fmt(q.outer_surface_charge_c*1e9)])escText(tr,v,'td');$('conductors').appendChild(tr)}$('totals-note').textContent=frame.conductor_results_method||'Supplied solver conductor properties; provenance not specified.';$('density').replaceChildren();frame.profiles.forEach((s,i)=>{const o=document.createElement('option');o.value=i;o.textContent=s.conductor+' / '+s.surface;$('density').appendChild(o)});$('notes').replaceChildren();frame.notes.forEach(n=>escText($('notes'),n));escText($('notes'),'Trace terminations: '+Object.entries(frame.line_termination_counts||{}).map(([k,v])=>k+' '+v).join('; '));$('sources').replaceChildren();frame.charges.forEach(q=>escText($('sources'),fmt(q.charge_c*1e9)+' nC at ['+q.position_m.map(fmt).join(', ')+'] m'+(Math.abs(q.slice_position_m[2])>1e-10?' (outside this slice)':'')));frame.source_masks.forEach(m=>escText($('sources'),m.kind+': radius '+fmt(m.radius_m)+' m'));drawScene();drawProfile()}
$('frame').addEventListener('input',selectFrame);$('density').addEventListener('change',drawProfile);for(const id of ['heat','contours','lines'])$(id).addEventListener('change',drawScene);
canvas.addEventListener('mousemove',e=>{const rect=canvas.getBoundingClientRect(),px=(e.clientX-rect.left)*canvas.width/rect.width,py=(e.clientY-rect.top)*canvas.height/rect.height;const p=frame.plane,g=frame.grid;const x=p.x_range_m[0]+(px-map.ox)/map.scale,y=p.y_range_m[0]+(canvas.height-py-map.oy)/map.scale;const i=Math.round((x-g.x_m[0])/(g.x_m[1]-g.x_m[0])),j=Math.round((y-g.y_m[0])/(g.y_m[1]-g.y_m[0]));if(!frame||px<map.ox||px>canvas.width-map.ox||py<map.oy||py>canvas.height-map.oy||j<0||i<0||j>=g.y_m.length||i>=g.x_m.length){resetProbe();return;}const v=g.potential_v[j][i],f=g.electric_field_v_per_m[j][i];$('probe').textContent='u='+fmt(g.x_m[i])+' m   v='+fmt(g.y_m[j])+' m   '+g.region[j][i]+'\nV='+ (v===null?'masked':fmt(v)+' V')+'   |E|='+(f===null?'unavailable':fmt(Math.hypot(...f))+' V/m')});
canvas.addEventListener('mouseleave',resetProbe);
selectFrame();window.conductorViewer={frameCount:data.frames.length,schema:data.schema};
</script></body></html>'''
