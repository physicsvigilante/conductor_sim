"""Opt-in force/energy presentation on the existing offline motion renderer.

Uses its saved-time controls, scene, graph and projection functions. No new
physics, time interpolation, or renderer copy. Electric-only HTML is untouched.
"""

def append_gravity_presentation(html: str) -> str:
    if '</body>' not in html:
        raise ValueError('shared renderer body boundary is missing')
    return html.replace('</body>', SCRIPT + '</body>', 1)


SCRIPT = r'''<script id="gravity-presentation">
'use strict';
if(!prescribed || !prescribed.gravity)throw Error('Gravity presentation requires validated v2 observations');
document.querySelector('header .kicker').textContent='CONDUCTOR_SIM / M21E COMPOSED FORCES';
document.querySelector('header .sub').textContent='One reusable particle model: prescribed electric force plus independently authored uniform gravity. Stored full-vector trajectories, distinct force contributions, and K + electric qV + gravitational Ug. No browser solver.';
$('motion-panel').querySelector('h2').textContent='Motion under electric force and uniform gravity';
$('motion-scope-note').textContent='F_total = qE + mg. U_e = qV; U_g = -m g·(r-r_ref). The electric-field and voltage graphs remain electrical quantities, not disguised gravity. Fixed sources do not respond. No friction, drag, guide, plate impact or switched fields. The shared solver is Newtonian with speed below 0.1c.';
$('scene-scope').textContent='Schematic source geometry and saved physical trajectory. Force arrows: orange electric qE, blue gravity mg, purple resultant; all three use one common linear scale over this run. Green velocity uses its own display scale. Zero force has no arrow. No mechanical constraint, solved plate boundary or spatial field map is implied.';
$('scene').setAttribute('aria-label','Saved charged-particle path and separate electric, gravity and resultant force arrows');
$('force-toggle').lastChild.textContent=' Electric / gravity / resultant forces';
const gLedger=document.createElement('p');gLedger.id='gravity-force-summary';gLedger.className='small';
gLedger.style.overflowWrap='anywhere';$('scene-panel').appendChild(gLedger);
const gLegend=$('motion-panel').querySelector('.motion-legend');
gLegend.children[1].textContent='● Electric U_e = qV';
gLegend.children[2].textContent='● Total K + U_e + U_g';
const gKey=document.createElement('span');gKey.style.color='#7225a4';gKey.textContent='● Gravitational U_g';gLegend.appendChild(gKey);
const gExtent=Math.max(...motion.samples.flatMap(r=>[Math.abs(r.kinetic_energy_j),Math.abs(r.electric_potential_energy_j),Math.abs(r.gravitational_potential_energy_j),Math.abs(r.total_energy_j)]));
const [gEnergyScale,gEnergyLabel]=gExtent<1e-15?[1e18,'aJ']:gExtent<1e-12?[1e15,'fJ']:gExtent<1e-9?[1e12,'pJ']:gExtent<1e-6?[1e9,'nJ']:gExtent<1e-3?[1e6,'µJ']:gExtent<1?[1e3,'mJ']:[1,'J'];
$('motion-energy-heading').textContent='K, electric U_e, gravitational U_g and total ('+gEnergyLabel+') versus time';
$('motion-energy-chart').setAttribute('aria-label','Distinct kinetic, electric potential, gravitational potential and total mechanical energies');
const gMaxForce=Math.max(...motionFrames.flatMap(r=>['electric_force_n','gravitational_force_n','force_n'].map(k=>Math.hypot(...r[k]))));
forceArrow=function(p,X,Y,project){
 if(!$('force-arrow').checked || gMaxForce===0)return;
 const r=motionFrames[Number($('frame').value)],pos=project(r.position_m);
 if(Math.abs(pos[2])>1e-9)return;
 for(const [key,color] of [['electric_force_n','#ca6a35'],['gravitational_force_n','#1766a2'],['force_n','#7225a4']]){
  const F=r[key],fu=F.reduce((s,x,i)=>s+x*p.u[i],0),fv=F.reduce((s,x,i)=>s+x*p.v[i],0);
  const dx=55*fu/gMaxForce,dy=-55*fv/gMaxForce,L=Math.hypot(dx,dy);
  if(L<1e-8)continue;
  const x=X(pos[0]),y=Y(pos[1]),head=Math.min(9,L*.3);
  ctx.save();ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=2.8;
  ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x+dx,y+dy);ctx.stroke();
  ctx.translate(x+dx,y+dy);ctx.rotate(Math.atan2(dy,dx));ctx.beginPath();
  ctx.moveTo(0,0);ctx.lineTo(-head,-head*.45);ctx.lineTo(-head,head*.45);ctx.closePath();ctx.fill();ctx.restore();
 }
};
const electricOnlyObservations=drawMotionObservations;
drawMotionObservations=function(){
 electricOnlyObservations();
 const r=motionFrames[Number($('frame').value)],vec=v=>'['+v.map(fmt).join(', ')+']';
 gLedger.textContent='Electric qE = '+vec(r.electric_force_n)+' N; gravity mg = '+vec(r.gravitational_force_n)+' N; resultant = '+vec(r.force_n)+' N. Force arrows share a scale; they are not three independent normalizations.';
 $('motion-summary').textContent='t = '+fmt(r.time_s*timeFactor)+' '+timeLabel+' · r = '+vec(r.position_m)+' m · v = '+vec(r.velocity_m_s)+' m/s · a = '+vec(r.acceleration_m_s2)+' m/s² · F_total = '+vec(r.force_n)+' N · K = '+fmt(r.kinetic_energy_j)+' J · U_e=qV = '+fmt(r.electric_potential_energy_j)+' J · U_g = '+fmt(r.gravitational_potential_energy_j)+' J · total = '+fmt(r.total_energy_j)+' J · energy drift = '+fmt(r.energy_drift_j)+' J';
 drawMotionGraph('motion-energy-chart',[
  {value:r=>r.total_energy_j*gEnergyScale,color:'#168875'},
  {value:r=>r.electric_potential_energy_j*gEnergyScale,color:'#ca6a35'},
  {value:r=>r.gravitational_potential_energy_j*gEnergyScale,color:'#7225a4'},
  {value:r=>r.kinetic_energy_j*gEnergyScale,color:'#1965a5'}]);
};
drawScene();drawMotionObservations();
window.conductorViewer.forceComponents=['electric','gravity'];
</script>'''
