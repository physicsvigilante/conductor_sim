"""Offline before/after schematic for the explicitly approximate M21d model."""
from html import escape
import json
from pathlib import Path
from .distant_spheres import DistantSphereConnectionBackend, DistantSphereConnectionResult


def render_distant_spheres(result, *, title, provenance):
    if type(result) is not DistantSphereConnectionResult:
        raise TypeError('expected a computed DistantSphereConnectionResult')
    check = DistantSphereConnectionBackend().solve(result.plan, approximation=result.approximation)
    if check != result:
        raise ValueError('display result differs from the independently recomputed model')
    record = dict(result.as_dict(), provenance=provenance)
    data = json.dumps(record, allow_nan=False, separators=(',', ':'))
    data = data.replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e')
    return TEMPLATE.replace('__TITLE__',escape(str(title))).replace('__DATA__',data)


def write_distant_spheres(result, path, *, title, provenance):
    html = render_distant_spheres(result, title=title, provenance=provenance)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding='utf-8')
    path.with_suffix('.json').write_text(json.dumps(dict(result.as_dict(),
        provenance=provenance), indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return path


TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
*{box-sizing:border-box}body{margin:0;color:#20364d;background:#f2f6fa;font:15px/1.5 system-ui,sans-serif}
header{padding:24px max(20px,calc((100vw - 1240px)/2));background:#19344c;color:white}
h1{font-size:25px;margin:4px 0}h2{font-size:18px;margin:0 0 12px}p{margin:8px 0}.kicker{font-size:12px;letter-spacing:.12em}
main{max-width:1280px;padding:20px;margin:auto}.card{min-width:0;background:white;border:1px solid #d3dfe9;border-radius:10px;padding:18px;margin-bottom:18px}
.pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.warn{background:#fff5da;border-left:5px solid #aa7422}
canvas{width:100%;height:auto;display:block;border:1px solid #e1e9f0;border-radius:6px}
.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:8px 6px;border-bottom:1px solid #dae3ed;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
.small{font-size:13px;color:#546c81}.metrics{display:flex;gap:24px;flex-wrap:wrap}.metric strong{font-size:21px;display:block;color:#17687d}
button{background:white;color:#244c6a;border:1px solid #8ca3b6;padding:9px 14px;border-radius:5px;cursor:pointer}button[aria-pressed=true]{background:#1b6078;color:white}
#scope{font-weight:600}#provenance p{overflow-wrap:anywhere}summary{cursor:pointer}code{overflow-wrap:anywhere}
@media(max-width:850px){.pair{grid-template-columns:1fr}main{padding:12px}.card{padding:12px}h1{font-size:21px}}
</style></head><body><header><div class="kicker">CONDUCTOR_SIM / M21D</div><h1>__TITLE__</h1>
<p>Charge sharing between far-separated spherical conductors</p></header><main>
<section class="card warn"><p id="scope">Leading self-capacitance approximation — two static equilibria, not a discharge animation.</p>
<p>C = 4πεR; connected members share V, not necessarily Q. Mutual induction and wire capacitance are omitted.</p>
<p class="small">Sphere radii are proportional within each diagram; centre separations are compressed schematically. No field lines or current paths are simulated.</p></section>
<section class="card" id="provenance"><h2>Source and authored choices</h2></section>
<section class="card"><h2>Before / after connection</h2><button id="show-delta" aria-pressed="false">Show charge changes</button><p class="small">This control changes labels only; both equilibria are already computed.</p></section>
<div class="pair"><section class="card"><h2>Before: isolated conductors</h2><div class="scroll"><canvas id="before" width="700" height="270" aria-label="Initial isolated spheres, schematic"></canvas></div><div class="scroll"><table id="before-table"></table></div></section>
<section class="card"><h2>After: linked equilibrium</h2><div class="scroll"><canvas id="after" width="700" height="270" aria-label="Connected sphere groups in final static equilibrium, schematic"></canvas></div><div class="scroll"><table id="after-table"></table></div></section></div>
<section class="card"><h2>Electrostatic energy in this approximation</h2><div id="energy" class="metrics"></div><p class="small">Release is available energy on settling. Its partition into wire heating, radiation or other losses is not modeled. No current magnitude or elapsed time is inferred.</p></section>
<section class="card"><h2>Charge transferred by stable identity</h2><div class="scroll"><table id="transfers"></table></div><p class="small">Positive ΔQ means net positive charge received (electrons lost). Negative ΔQ means electrons gained. No electron animation is implied.</p></section>
<section class="card"><h2>Approximation checks and omitted effects</h2><div id="qualification"></div><details><summary>Per-object mutual-potential estimates and scope limits</summary><div id="diagnostics"></div><div id="limits"></div></details></section>
</main><script id="distant-data" type="application/json">__DATA__</script><script>
'use strict';
const data=JSON.parse(document.getElementById('distant-data').textContent),$=id=>document.getElementById(id);
const fmt=x=>Number(x)===0?'0':Number(x).toPrecision(6);
const add=(el,text,tag='p')=>{const n=document.createElement(tag);n.textContent=text;el.appendChild(n);return n};
let delta=false;
Object.entries(data.provenance).forEach(([k,v])=>add($('provenance'),k.replaceAll('_',' ')+': '+(typeof v==='string'?v:JSON.stringify(v))));
function table(id,rows){const t=$(id);t.replaceChildren();const h=document.createElement('tr');['ID','Q (C)','V (V)','σ outer (C/m²)','E outer (V/m)'].forEach(x=>add(h,x,'th'));t.appendChild(h);rows.forEach(r=>{const tr=document.createElement('tr');[r.name,fmt(r.charge_c),fmt(r.potential_v),fmt(r.mean_outer_sigma_c_m2),fmt(r.outer_field_normal_v_m)].forEach(x=>add(tr,x,'td'));t.appendChild(tr)})}
function draw(id,state,linked){const c=$(id),ctx=c.getContext('2d'),rows=state.conductors,n=rows.length,R=Math.max(...rows.map(r=>r.outer_radius_m));const W=Math.max(700,220*n);c.width=W;c.style.minWidth=n>3?(170*n)+'px':'';ctx.clearRect(0,0,W,270);ctx.fillStyle='#fbfdff';ctx.fillRect(0,0,W,270);
 const X=i=>70+((W-140)*(i+.5)/n),Y=108,unit=Math.min(58,240/n)/R;
 if(linked){ctx.strokeStyle='#516b84';ctx.lineWidth=2;for(const link of data.links){const members=link.members.map(name=>rows.findIndex(r=>r.name===name));const y=195+data.links.indexOf(link)*13;ctx.beginPath();members.forEach(i=>{ctx.moveTo(X(i),Y+rows[i].outer_radius_m*unit);ctx.lineTo(X(i),y)});ctx.moveTo(Math.min(...members.map(X)),y);ctx.lineTo(Math.max(...members.map(X)),y);ctx.stroke()}}
 rows.forEach((r,i)=>{const x=X(i),rad=r.outer_radius_m*unit;ctx.beginPath();ctx.arc(x,Y,rad,0,2*Math.PI);ctx.fillStyle='#c2d3df';ctx.fill();ctx.strokeStyle='#496880';ctx.stroke();if(r.kind==='SphericalShell'){ctx.beginPath();ctx.arc(x,Y,r.inner_radius_m*unit,0,2*Math.PI);ctx.fillStyle='#fbfdff';ctx.fill();ctx.stroke()}
 ctx.fillStyle='#20364d';ctx.font='bold 14px system-ui';ctx.textAlign='center';ctx.fillText(r.name.length>24?r.name.slice(0,21)+'…':r.name,x,25);ctx.font='12px system-ui';ctx.fillText('R = '+fmt(r.outer_radius_m)+' m',x,44);
 const q=delta?data.transfers[i].received_c:r.charge_c;ctx.fillStyle=q<0?'#1960a3':'#a74330';ctx.fillText((delta?'ΔQ = ':'Q = ')+fmt(q)+' C',x,233);ctx.fillStyle='#294d68';ctx.fillText('V = '+fmt(r.potential_v)+' V',x,254)})}
function paint(){draw('before',data.before,false);draw('after',data.after,true)}
table('before-table',data.before.conductors);table('after-table',data.after.conductors);paint();
$('show-delta').onclick=()=>{delta=!delta;$('show-delta').setAttribute('aria-pressed',String(delta));$('show-delta').textContent=delta?'Show net charges':'Show charge changes';paint()};
for(const [label,value] of [['Before',data.before.self_field_energy_j],['After',data.after.self_field_energy_j],['Released',data.released_field_energy_j]]){const div=add($('energy'),'','div');div.className='metric';add(div,label,'span');add(div,fmt(value)+' J','strong')}
const tr=document.createElement('tr');['Conductor ID','ΔQ received (C)'].forEach(x=>add(tr,x,'th'));$('transfers').appendChild(tr);data.transfers.forEach(r=>{const tr=document.createElement('tr');[r.name,fmt(r.received_c)].forEach(x=>add(tr,x,'td'));$('transfers').appendChild(tr)});
add($('qualification'),'max Σ(R_i/d_ij) = '+fmt(Math.max(...data.qualification.geometric_row_sums))+'; required ≤ '+data.qualification.maximum_allowed_row_sum+'. This is NOT a 5% physical-error guarantee.');
add($('qualification'),'Floating groups: '+data.links.map(l=>l.members.join(' + ')).join('; '));
add($('qualification'),'Total charge: '+fmt(data.before.total_charge_c)+' C before; '+fmt(data.after.total_charge_c)+' C after.');
for(const label of ['before','after']){add($('diagnostics'),label+': omitted mutual-monopole potential estimates (V), in ID order: '+data[label].omitted_monopole_potential_estimate_v.map(fmt).join(', '));add($('diagnostics'),label+': omitted pair-energy absolute scale = '+fmt(data[label].omitted_pair_energy_abs_scale_j)+' J. Diagnostic only; NOT added to self-only energies and NOT an exact error bound.')}
data.limitations.forEach(x=>add($('limits'),x));
window.distantSphereViewer={mode:'two-static-approximate-equilibria',sourceIds:data.conductor_ids,get deltaLabels(){return delta},hasTimeEvolution:false};
</script></body></html>'''
