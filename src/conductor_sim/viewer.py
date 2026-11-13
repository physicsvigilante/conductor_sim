"""Offline, dependency-free HTML canvas viewer for visualization.v1 bundles."""
from html import escape
import json
import re


def render_html(bundle, *, title='Conductor simulation'):
    if bundle.get('schema') != 'conductor_sim.visualization.v1' or not bundle.get('frames'):
        raise ValueError('expected a nonempty conductor_sim.visualization.v1 bundle')
    # Older v1 exports get safe display metadata without mutating their data.
    from .potential_display import potential_interpolation, potential_colour_scale
    frames = [dict(f) for f in bundle['frames']]
    for frame in frames:
        if 'potential_interpolation' not in frame:
            frame['potential_interpolation'] = potential_interpolation(frame)
    display = dict(bundle, frames=frames)
    sweep = display.get('position_sweep')
    if sweep is not None:
        if (sweep.get('schema') != 'conductor_sim.position_sweep.v1' or
                len(sweep.get('samples', ())) != len(frames) or
                any(f.get('force_energy') != row for f, row in
                    zip(frames, sweep['samples']))):
            raise ValueError('position sweep samples must match displayed frames exactly')
    display.setdefault('potential_colour_scale', potential_colour_scale(frames))
    data = json.dumps(display,allow_nan=False,separators=(',',':'))
    # JSON lives inside a script element: HTML escaping alone is insufficient.
    data = data.replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e')
    replacements = {'__TITLE__':escape(str(title),quote=True),'__DATA__':data}
    return re.sub(r'__(?:TITLE|DATA)__',lambda m:replacements[m.group(0)],TEMPLATE)


TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:light;--ink:#142334;--muted:#526579;--line:#d9e2eb;--blue:#165884}*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:var(--ink);font:15px/1.5 system-ui,-apple-system,sans-serif}header{background:#122b40;color:white;padding:24px max(20px,calc((100vw - 1260px)/2));border-bottom:4px solid #3ca5b0}h1{font-size:26px;margin:4px 0}h2{font-size:17px;margin:0 0 14px}p{margin:8px 0}.kicker{font-size:11px;letter-spacing:.16em;text-transform:uppercase;opacity:.75}.sub{color:#cbd9e4;max-width:850px;font-size:13px}main{max-width:1300px;margin:auto;padding:22px;display:grid;grid-template-columns:minmax(440px,1.6fr) minmax(310px,1fr);gap:20px}.card{background:white;border:1px solid var(--line);border-radius:10px;padding:20px;box-shadow:0 4px 16px #14233406}.wide{grid-column:1/-1}.row{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}.chip{font:12px ui-monospace,monospace;background:#e7f3f6;color:#17556a;padding:5px 9px;border-radius:5px}canvas{display:block;width:100%;height:auto}#scene{background:#f9fbfd;border:1px solid var(--line);border-radius:5px}#profile{border:1px solid var(--line);margin-top:12px}label{font-size:13px;cursor:pointer}select,button{padding:7px;border:1px solid #bbcbd7;border-radius:4px;background:white;color:var(--ink)}input[type=range]{flex:1;min-width:160px}input[type=checkbox]{accent-color:#176488}#case{font-weight:650;font-size:18px}#probe{font:12px/1.6 ui-monospace,monospace;min-height:42px;margin-top:9px;color:#344d65}table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:7px 5px;text-align:right;border-bottom:1px solid #e6ebf0}th:first-child,td:first-child{text-align:left}.small{font-size:12px;color:var(--muted)}#scale{height:12px;background:linear-gradient(90deg,#3476ae,#f8f9fb,#d95b43);border:1px solid var(--line);margin-top:10px}.ticks{display:flex;justify-content:space-between;font:11px ui-monospace,monospace;color:var(--muted)}.line-density-control{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:8px 0 12px}.line-density-control input{min-width:100px;flex:1 1 120px}.line-density-control output{font-size:12px;min-width:96px}.controls{margin:14px 0 12px;display:flex;gap:18px;flex-wrap:wrap}#notes p{padding-left:12px;border-left:2px solid #d9e6ef;font-size:12px;color:var(--muted)}aside .card{margin-bottom:20px}.error{color:#b52222}[hidden]{display:none!important}.force-plots{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.force-plots canvas{border:1px solid var(--line);border-radius:5px;background:#fbfcfe}.force-plots h3{font-size:14px;margin:4px 0 8px}@media(max-width:850px){main{grid-template-columns:1fr}.wide{grid-column:auto}header{padding:18px}h1{font-size:22px}.card{padding:15px}.force-plots{grid-template-columns:1fr}}
</style></head><body>
<header><div class="kicker">CONDUCTOR_SIM / M5 VISUALIZATION</div><h1>__TITLE__</h1><p class="sub">Potential, field geometry and induced surface charge from one reusable sampling pipeline. No server or solver is needed to inspect this exported file.</p></header>
<main><section class="card wide"><div class="row"><span id="case"></span><span id="backend" class="chip"></span></div><div class="row" style="margin-top:14px"><label for="frame" id="position-label">Stored state</label><input id="frame" type="range" min="0" step="1" value="0"><span id="counter" class="small"></span></div><p class="small">The slider selects precomputed solutions. It does not recompute FEMM or simulate charge motion in the browser.</p></section>
<section class="card"><div class="row"><h2>Planar cross-section</h2><span class="small">metres · V · V/m</span></div><div class="controls"><label><input type="checkbox" id="heat" checked> Potential</label><label><input type="checkbox" id="contours" checked> Equipotentials</label><label><input type="checkbox" id="lines" checked> Field lines</label><label id="force-toggle" hidden><input type="checkbox" id="force-arrow" checked> Selected-charge force arrow</label></div><div class="line-density-control"><label for="line-density">Field-line density</label><input id="line-density" type="range" min="1" max="5" step="1" value="3" aria-describedby="line-density-value"><output id="line-density-value" for="line-density" aria-live="polite"></output></div><canvas id="scene" width="720" height="720" aria-label="Potential map and electric field lines"></canvas><div id="scale"></div><div class="ticks"><span id="lo"></span><span id="lo-mid"></span><span>0 V</span><span id="hi-mid"></span><span id="hi"></span></div><p id="potential-method" class="small">Smooth sampled potential · asinh colour scale, shared across states. No blur. Blank cells are excluded or unresolved.</p><div id="probe">Move the pointer over the cross-section to inspect the nearest stored sample.</div><p class="small">Grey regions: conducting metal. Dashed source discs: excluded regions. Arrows follow E; line count does not represent flux. Contours are sampled interpolations, not exact curves.</p></section>
<aside><section class="card"><h2>Conductor results</h2><div style="overflow:auto"><table><thead><tr><th>Name</th><th>V</th><th>Net nC</th><th>Inner nC</th><th>Outer nC</th></tr></thead><tbody id="conductors"></tbody></table></div><p class="small" id="totals-note"></p></section><section class="card"><h2>Induced surface-charge profile</h2><div class="row"><label for="density">Conductor / face</label><select id="density"></select></div><canvas id="profile" width="450" height="260" aria-label="Induced charge density along a great circle"></canvas><p id="method" class="small"></p><p class="small">For spherical conductors the horizontal axis is polar angle; for planes it is signed distance (m) along the cross-section. These lines are not area integrals.</p></section><section class="card"><h2>Source and sampling details</h2><div id="sources" class="small"></div><div id="notes"></div></section></aside>
<section class="card wide" id="force-panel" hidden><h2>Force and mechanical interaction energy</h2>
<p class="small" id="force-summary" aria-live="polite"></p>
<div class="force-plots"><section><h3>Force along displacement (µN)</h3><canvas id="force-chart" width="560" height="270" aria-label="Force component versus charge position"></canvas></section>
<section><h3>Mechanical interaction energy (nJ)</h3><canvas id="energy-chart" width="560" height="270" aria-label="Interaction energy versus charge position"></canvas></section></div>
<p class="small" id="work-summary" aria-live="polite"></p>
<details><summary>Direct, induced and background contributions</summary><p class="small" id="force-components"></p></details>
<p class="small" id="force-method"></p><p class="small">The position slider selects precomputed static equilibria, not elapsed time or simulated motion. Force arrows show the in-slice component and their lengths are display-normalized; graphs carry the physical scales. Work assumes all other charges, electrode constraints and voltage references are unchanged.</p></section>
</main><script id="simulation-data" type="application/json">__DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('simulation-data').textContent);
const $=id=>document.getElementById(id), canvas=$('scene'),ctx=canvas.getContext('2d');
let frame=data.frames[0],map=null;
const fmt=x=>Number(x)===0?'0':Number(x).toPrecision(5);
const resetProbe=()=>{$('probe').textContent='Move the pointer over the cross-section to inspect the nearest stored sample.'};
const escText=(parent,text,tag='p')=>{const e=document.createElement(tag);e.textContent=text;parent.appendChild(e);return e};
$('frame').max=data.frames.length-1;
const colourSpec=data.potential_colour_scale, lim=colourSpec.max_abs_v, soft=colourSpec.softening_v;
const colourExtent=Math.asinh(lim/soft);
const halfColourValue=soft*Math.sinh(colourExtent/2);
$('lo').textContent=fmt(-lim)+' V';$('hi').textContent=fmt(lim)+' V';
$('lo-mid').textContent=fmt(-halfColourValue);$('hi-mid').textContent=fmt(halfColourValue);
function colourComponents(value){
 const t=Math.max(-1,Math.min(1,Math.asinh(value/soft)/colourExtent));
 const a=[248,249,251],b=t<0?[52,118,174]:[217,91,67];
 return a.map((x,i)=>Math.round(x+Math.abs(t)*(b[i]-x)));
}
function colour(value){return `rgb(${colourComponents(value).join(',')})`}
const interpolationCache=new WeakMap();
function stencilIndex(spec,nx){
 const boundary=new Map();
 for(const item of spec.boundary)boundary.set(item.cell[1]*(nx-1)+item.cell[0],item.triangles);
 return {cells:spec.cells,boundary};
}
function interpolationFor(f){
 let index=interpolationCache.get(f);if(index)return index;
 index=stencilIndex(f.potential_interpolation,f.grid.x_m.length);
 index.patches=(f.potential_refinements||[]).map(p=>({grid:p.grid,
  stencil:stencilIndex(p.potential_interpolation,p.grid.x_m.length)}));
 interpolationCache.set(f,index);return index;
}
function gridSample(g,index,x,y){
 const nx=g.x_m.length,ny=g.y_m.length;
 const fx=(x-g.x_m[0])/(g.x_m[1]-g.x_m[0]),fy=(y-g.y_m[0])/(g.y_m[1]-g.y_m[0]);
 if(fx<0||fy<0||fx>nx-1||fy>ny-1)return null;
 const i=Math.min(nx-2,Math.floor(fx)),j=Math.min(ny-2,Math.floor(fy)),tx=fx-i,ty=fy-j;
 if(index.cells[j][i]){
  const a=g.potential_v[j][i],b=g.potential_v[j][i+1],c=g.potential_v[j+1][i],d=g.potential_v[j+1][i+1];
  return (1-ty)*((1-tx)*a+tx*b)+ty*((1-tx)*c+tx*d);
 }
 for(const [a,b,c] of index.boundary.get(j*(nx-1)+i)||[]){
  const det=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
  if(!det)continue;
  const u=((x-a[0])*(c[1]-a[1])-(y-a[1])*(c[0]-a[0]))/det;
  const v=((b[0]-a[0])*(y-a[1])-(b[1]-a[1])*(x-a[0]))/det;
  if(u>=-1e-12&&v>=-1e-12&&u+v<=1+1e-12)return a[2]*(1-u-v)+b[2]*u+c[2]*v;
 }
 return null;
}
function potentialSample(f,x,y){
 const index=interpolationFor(f),coarse=gridSample(f.grid,index,x,y);
 for(const patch of index.patches){
  const g=patch.grid;
  if(x<g.x_m[0]||x>g.x_m[g.x_m.length-1]||y<g.y_m[0]||y>g.y_m[g.y_m.length-1])continue;
  const fine=gridSample(g,patch.stencil,x,y);
  if(fine===null)continue;
  if(coarse===null)return fine;
  // Blend to the original coarse interpolant near the patch perimeter.
  // This only blends two valid vacuum estimates; masked regions remain null.
  const margin=Math.min(x-g.x_m[0],g.x_m[g.x_m.length-1]-x,
                        y-g.y_m[0],g.y_m[g.y_m.length-1]-y);
  const d=Math.max(f.grid.x_m[1]-f.grid.x_m[0],f.grid.y_m[1]-f.grid.y_m[0]);
  const t=Math.max(0,Math.min(1,margin/d));
  return coarse*(1-t)+fine*t;
 }
 return coarse;
}
let heatCache=null,heatmapBuilds=0;
function drawHeatmap(W,H,p,ox,oy,scale){
 // Keep only the current frame's raster. Density/toggles reuse it immediately.
 if(!heatCache||heatCache.frame!==frame||heatCache.width!==W||heatCache.height!==H){
  const layer=document.createElement('canvas');layer.width=W;layer.height=H;
  const lc=layer.getContext('2d'),img=lc.createImageData(W,H),pixels=img.data;
  for(let py=Math.ceil(oy);py<H-oy;py++){
   const y=p.y_range_m[0]+(H-py-.5-oy)/scale;
   for(let px=Math.ceil(ox);px<W-ox;px++){
    const x=p.x_range_m[0]+(px+.5-ox)/scale,v=potentialSample(frame,x,y);
    if(v===null)continue;
    const rgb=colourComponents(v),k=4*(py*W+px);
    pixels[k]=rgb[0];pixels[k+1]=rgb[1];pixels[k+2]=rgb[2];pixels[k+3]=255;
   }
  }
  lc.putImageData(img,0,0);heatmapBuilds++;
  heatCache={frame,width:W,height:H,layer};
 }
 ctx.drawImage(heatCache.layer,0,0);
}
const lineFractions=[.15,.3,.5,.75,1],lineLabels=['Sparse','Light','Medium','Dense','Full'];
const lineIndex=()=>Math.max(0,Math.min(4,Number($('line-density').value)-1));
function shownFieldLines(){
 const pool=frame.field_lines||[];
 return pool.slice(0,pool.length?Math.max(1,Math.ceil(pool.length*lineFractions[lineIndex()])):0);
}
function updateLineDensityLabel(){
 const pool=frame.field_lines||[],on=$('lines').checked;
 $('line-density').disabled=!pool.length||!on;
 const label=!pool.length?'No traces':(!on?'Field lines off':lineLabels[lineIndex()]+' · '+shownFieldLines().length+' / '+pool.length);
 $('line-density-value').textContent=label;$('line-density').setAttribute('aria-valuetext',label);
}
const sweep=data.position_sweep||null;
if(sweep){
 $('force-panel').hidden=false;$('force-toggle').hidden=false;
 $('position-label').textContent='Charge position';
}
function forceArrow(p,X,Y,project){
 if(!sweep||!$('force-arrow').checked||!frame.force_energy)return;
 const current=frame.force_energy,pos=project(current.position_m);
 if(Math.abs(pos[2])>1e-9)return;
 const F=current.force_n,fu=F.reduce((z,v,i)=>z+v*p.u[i],0),fv=F.reduce((z,v,i)=>z+v*p.v[i],0);
 const n=Math.hypot(fu,fv),max=Math.max(...sweep.samples.map(r=>Math.hypot(...r.force_n)));
 if(n<=Math.max(max*1e-11,1e-30)||max===0)return;
 const size=Math.max(9,Math.min(55,55*Math.sqrt(n/max))),dx=size*fu/n,dy=-size*fv/n;
 const x=X(pos[0]),y=Y(pos[1]);
 ctx.save();ctx.strokeStyle='#7225a4';ctx.fillStyle='#7225a4';ctx.lineWidth=3;
 ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x+dx,y+dy);ctx.stroke();
 const a=Math.atan2(dy,dx);ctx.translate(x+dx,y+dy);ctx.rotate(a);
 ctx.beginPath();ctx.moveTo(0,0);ctx.lineTo(-11,-5);ctx.lineTo(-11,5);ctx.closePath();ctx.fill();
 ctx.restore();
}
function drawSweepPlot(canvasId,key,scale,color){
 if(!sweep)return;
 const canvas=$(canvasId),c=canvas.getContext('2d'),W=canvas.width,H=canvas.height;
 const samples=sweep.samples, values=samples.map(r=>r[key]*scale);
 const lo0=Math.min(...values,0),hi0=Math.max(...values,0),span=hi0-lo0;
 const pad=Math.max(span*.12,Math.max(Math.abs(lo0),Math.abs(hi0))*1e-6,1e-10),lo=lo0-pad,hi=hi0+pad;
 const left=67,right=W-18,top=20,bottom=H-48;
 const x=(s)=>left+(right-left)*(s/sweep.path_length_m);
 const y=(v)=>bottom-(bottom-top)*(v-lo)/(hi-lo);
 c.clearRect(0,0,W,H);c.fillStyle='#fbfcfe';c.fillRect(0,0,W,H);
 c.font='12px system-ui';c.fillStyle='#496074';c.strokeStyle='#d6dfe8';c.lineWidth=1;
 for(let i=0;i<=4;i++){
  const value=lo+(hi-lo)*i/4, yy=y(value);
  c.beginPath();c.moveTo(left,yy);c.lineTo(right,yy);c.stroke();
  c.textAlign='right';const displayed=Math.abs(value)<Math.max(Math.abs(lo),Math.abs(hi))*1e-10?0:value;c.fillText(Number(displayed.toPrecision(3)).toString(),left-7,yy+4);
 }
 c.textAlign='center';for(let i=0;i<=4;i++){
  const s=sweep.path_length_m*i/4;
  c.fillText(Number(s.toPrecision(3)).toString(),x(s),H-26);
 }
 c.fillText('Path position (m)',(left+right)/2,H-7);
 if(lo<0&&hi>0){c.strokeStyle='#a9b5c0';c.setLineDash([4,4]);c.beginPath();c.moveTo(left,y(0));c.lineTo(right,y(0));c.stroke();c.setLineDash([])}
 c.strokeStyle=color;c.lineWidth=2.5;c.beginPath();
 samples.forEach((r,i)=>i?c.lineTo(x(r.coordinate_m),y(values[i])):c.moveTo(x(r.coordinate_m),y(values[i])));c.stroke();
 const index=Number($('frame').value),current=samples[index],xx=x(current.coordinate_m),yy=y(values[index]);
 c.strokeStyle='#263e57';c.setLineDash([3,4]);c.lineWidth=1;c.beginPath();c.moveTo(xx,top);c.lineTo(xx,bottom);c.stroke();c.setLineDash([]);
 c.beginPath();c.arc(xx,yy,5,0,Math.PI*2);c.fillStyle=color;c.fill();
}
function drawSweepObservations(){
 if(!sweep)return;
 const r=sweep.samples[Number($('frame').value)];
 const magnitude=Math.hypot(...r.force_n),axis=[r.force_n[0]*sweep.direction[0]+r.force_n[1]*sweep.direction[1]+r.force_n[2]*sweep.direction[2]];
 $('force-summary').textContent='Path s = '+fmt(r.coordinate_m)+' m  ·  |F| = '+fmt(magnitude*1e6)+' µN  ·  F along path = '+fmt(axis[0]*1e6)+' µN  ·  U = '+fmt(r.energy_j*1e9)+' nJ';
 $('work-summary').textContent='External quasistatic work from the initial position: '+fmt(r.external_quasistatic_work_j*1e9)+' nJ; work by electric force: '+fmt(-r.external_quasistatic_work_j*1e9)+' nJ.';
 $('force-components').textContent='Force along path (µN): direct '+fmt(1e6*r.direct_force_n.reduce((v,f,i)=>v+f*sweep.direction[i],0))+'; induced '+fmt(1e6*r.induced_force_n.reduce((v,f,i)=>v+f*sweep.direction[i],0))+'; prescribed background '+fmt(1e6*r.background_force_n.reduce((v,f,i)=>v+f*sweep.direction[i],0))+'. Energy (nJ): direct '+fmt(r.energy_direct_j*1e9)+'; induced '+fmt(r.energy_induced_j*1e9)+'; background '+fmt(r.energy_background_j*1e9)+'.';
 const p=frame.plane,n=[p.u[1]*p.v[2]-p.u[2]*p.v[1],p.u[2]*p.v[0]-p.u[0]*p.v[2],p.u[0]*p.v[1]-p.u[1]*p.v[0]];
 const omitted=Math.abs(r.force_n.reduce((v,f,i)=>v+f*n[i],0));
 $('force-method').textContent=r.method+' · '+r.energy_reference+(omitted>Math.hypot(...r.force_n)*1e-6?' · Arrow shows in-slice projection; some force is out of plane.':'');
 drawSweepPlot('force-chart','force_parallel_n',1e6,'#7533a9');
 drawSweepPlot('energy-chart','energy_j',1e9,'#087b85');
}
function drawScene(){
 const p=frame.plane,g=frame.grid,W=canvas.width,H=canvas.height,pad=46;
 const scale=Math.min((W-2*pad)/(p.x_range_m[1]-p.x_range_m[0]),(H-2*pad)/(p.y_range_m[1]-p.y_range_m[0]));
 const ox=(W-scale*(p.x_range_m[1]-p.x_range_m[0]))/2,oy=(H-scale*(p.y_range_m[1]-p.y_range_m[0]))/2;
 const X=x=>ox+(x-p.x_range_m[0])*scale,Y=y=>H-oy-(y-p.y_range_m[0])*scale;map={X,Y,scale,ox,oy};
 ctx.clearRect(0,0,W,H);ctx.fillStyle='#f9fbfd';ctx.fillRect(0,0,W,H);
 if($('heat').checked)drawHeatmap(W,H,p,ox,oy,scale);
 updateLineDensityLabel();
 ctx.strokeStyle='#d0dae4';ctx.lineWidth=1;ctx.setLineDash([3,5]);for(const z of [0]){ctx.beginPath();ctx.moveTo(X(z),oy);ctx.lineTo(X(z),H-oy);ctx.moveTo(ox,Y(z));ctx.lineTo(W-ox,Y(z));ctx.stroke()}ctx.setLineDash([]);
 if($('contours').checked){ctx.strokeStyle='#49677988';ctx.lineWidth=1;for(const c of frame.contours){ctx.beginPath();for(const seg of c.segments_m){ctx.moveTo(X(seg[0][0]),Y(seg[0][1]));ctx.lineTo(X(seg[1][0]),Y(seg[1][1]))}ctx.stroke()}}
 if($('lines').checked){ctx.strokeStyle='#183d56';ctx.fillStyle='#183d56';ctx.lineWidth=1.25;for(const l of shownFieldLines()){const q=l.points_m;if(q.length<2)continue;ctx.beginPath();q.forEach((v,i)=>i?ctx.lineTo(X(v[0]),Y(v[1])):ctx.moveTo(X(v[0]),Y(v[1])));ctx.stroke();const k=Math.floor((q.length-1)*.58),a=q[k],b=q[Math.min(k+1,q.length-1)];const angle=Math.atan2(Y(b[1])-Y(a[1]),X(b[0])-X(a[0]));ctx.save();ctx.translate(X(a[0]),Y(a[1]));ctx.rotate(angle);ctx.beginPath();ctx.moveTo(4,0);ctx.lineTo(-3,-2.8);ctx.lineTo(-3,2.8);ctx.closePath();ctx.fill();ctx.restore()}}
 for(const s of frame.geometry){
  if(!s.visible){
   if(s.kind==='plane'&&s.fills_metal){ctx.fillStyle='#afbdc9';ctx.fillRect(ox,oy,W-2*ox,H-2*oy)}
   continue;
  }
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
 forceArrow(p,X,Y,project);
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
 const left=xs[0],right=xs[xs.length-1],ys=series.sigma_c_per_m2.map(x=>x===null?null:x*1e9);
 const available=ys.filter(x=>x!==null);
 if(!available.length){$('method').textContent=series.method+' · no samples inside the numerical domain';c.textAlign='left';c.fillStyle='#536a7e';c.font='12px system-ui';c.fillText('No density samples in this view.',25,50);return}
 let lo=Math.min(0,...available),hi=Math.max(0,...available),span=hi-lo||1;lo-=.08*span;hi+=.08*span;
 const X=t=>64+(t-left)/(right-left)*(W-84),Y=v=>H-40-(v-lo)/(hi-lo)*(H-67);
 c.strokeStyle='#c5d1dc';c.beginPath();c.moveTo(64,20);c.lineTo(64,H-40);c.lineTo(W-20,H-40);c.stroke();
 c.font='11px system-ui';c.fillStyle='#536a7e';c.textAlign='right';
 for(let i=0;i<4;i++){const v=lo+i*(hi-lo)/3;c.fillText(v.toPrecision(3),58,Y(v)+4)}
 c.textAlign='center';for(let i=0;i<=4;i++){const x=left+i*(right-left)/4;
 c.fillText(isPlane?x.toPrecision(2):Math.round(x*180/Math.PI)+'°',X(x),H-22)}
 c.textAlign='left';c.fillText('σ  [nC/m²]',65,15);
 c.strokeStyle='#165884';c.lineWidth=2;c.beginPath();
 let connected=false;
 xs.forEach((x,i)=>{if(ys[i]===null){connected=false;return}if(connected)c.lineTo(X(x),Y(ys[i]));else c.moveTo(X(x),Y(ys[i]));connected=true});c.stroke();
 $('method').textContent=series.method+(isPlane?' · horizontal axis: signed distance (m)':'')+(available.length<ys.length?' · gaps: excluded samples':'');
}
function selectFrame(){frame=data.frames[Number($('frame').value)];resetProbe();$('case').textContent=frame.label;$('counter').textContent=(Number($('frame').value)+1)+' / '+data.frames.length;$('backend').textContent=frame.backend;$('conductors').replaceChildren();for(const q of frame.conductor_results){const tr=document.createElement('tr');for(const v of [q.name,fmt(q.potential_v),q.net_charge_c===null?'n/a':fmt(q.net_charge_c*1e9),q.inner_surface_charge_c===null?'n/a':fmt(q.inner_surface_charge_c*1e9),q.outer_surface_charge_c===null?'n/a':fmt(q.outer_surface_charge_c*1e9)])escText(tr,v,'td');$('conductors').appendChild(tr)}$('totals-note').textContent=frame.conductor_results_method||'Supplied solver conductor properties; provenance not specified.';$('density').replaceChildren();frame.profiles.forEach((s,i)=>{const o=document.createElement('option');o.value=i;o.textContent=s.conductor+' / '+s.surface;$('density').appendChild(o)});$('notes').replaceChildren();frame.notes.forEach(n=>escText($('notes'),n));escText($('notes'),'Trace terminations: '+Object.entries(frame.line_termination_counts||{}).map(([k,v])=>k+' '+v).join('; '));$('sources').replaceChildren();frame.charges.forEach(q=>escText($('sources'),fmt(q.charge_c*1e9)+' nC at ['+q.position_m.map(fmt).join(', ')+'] m'+(Math.abs(q.slice_position_m[2])>1e-10?' (outside this slice)':'')));frame.source_masks.forEach(m=>escText($('sources'),m.kind+': radius '+fmt(m.radius_m)+' m'));drawScene();drawProfile();drawSweepObservations()}
$('frame').addEventListener('input',selectFrame);$('density').addEventListener('change',drawProfile);$('line-density').addEventListener('input',drawScene);$('force-arrow').addEventListener('change',drawScene);for(const id of ['heat','contours','lines'])$(id).addEventListener('change',drawScene);
canvas.addEventListener('mousemove',e=>{const rect=canvas.getBoundingClientRect(),px=(e.clientX-rect.left)*canvas.width/rect.width,py=(e.clientY-rect.top)*canvas.height/rect.height;const p=frame.plane,g=frame.grid;const x=p.x_range_m[0]+(px-map.ox)/map.scale,y=p.y_range_m[0]+(canvas.height-py-map.oy)/map.scale;const i=Math.round((x-g.x_m[0])/(g.x_m[1]-g.x_m[0])),j=Math.round((y-g.y_m[0])/(g.y_m[1]-g.y_m[0]));if(!frame||px<map.ox||px>canvas.width-map.ox||py<map.oy||py>canvas.height-map.oy||j<0||i<0||j>=g.y_m.length||i>=g.x_m.length){resetProbe();return;}const v=g.potential_v[j][i],f=g.electric_field_v_per_m[j][i];$('probe').textContent='u='+fmt(g.x_m[i])+' m   v='+fmt(g.y_m[j])+' m   '+g.region[j][i]+'\nV='+ (v===null?'masked':fmt(v)+' V')+'   |E|='+(f===null?'unavailable':fmt(Math.hypot(...f))+' V/m')});
canvas.addEventListener('mouseleave',resetProbe);
selectFrame();window.conductorViewer={frameCount:data.frames.length,schema:data.schema,get heatmapBuilds(){return heatmapBuilds}};
</script></body></html>'''
