"""M20a: compare two independently verified fixed-source trajectories offline.

The browser changes ONLY the selection of existing physical-time snapshots.
There is no field solve, RK integration, arbitrary source modification, or
scenario registry here. Source identities and provenance are authored facts.
"""
from __future__ import annotations

from html import escape
import json
from pathlib import Path
import re

from .prescribed_viewer_validation import validate_prescribed_viewer
from .viewer import render_html

_IDENTIFIER = re.compile(r'[A-Za-z][A-Za-z0-9:_-]{0,63}\Z')


def _text(value, name, length):
    if (type(value) is not str or not 1 <= len(value) <= length or
            not value.isprintable() or value != value.strip()):
        raise ValueError(f'{name} must be nonempty, printable, trimmed text (max {length} characters)')
    return value


def _id(value, name):
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f'{name} must be a bounded stable ASCII identifier')
    return value


def _source_map(motion):
    annotations = motion.get('source_annotations')
    if annotations is None:
        raise ValueError('paired viewers require explicit authored source IDs from M19e')
    return {a['id']: (a, motion['field']['sources'][a['source_index']]) for a in annotations}


def _changes(left, right):
    """Describe actual serialized model differences, never authored claims.

    IDs carry the author's identity assertions. The model values, not labels,
    determine the physically relevant changes displayed to the teacher.
    """
    baseline, variation = _source_map(left), _source_map(right)
    if set(baseline) != set(variation):
        raise ValueError('source identities differ: this bounded comparison permits property revisions, not adding/removing sources')
    changed = []
    for identity, (a, source) in baseline.items():
        b, counterpart = variation[identity]
        if source['kind'] != counterpart['kind']:
            raise ValueError(f'source kind changed for stable physical identity {identity}')
        if a['label'] != b['label']:
            raise ValueError(f'source label changed for stable physical identity {identity}')
        for k in sorted(set(source) | set(counterpart)):
            if source.get(k) != counterpart.get(k):
                changed.append({'component_id': identity, 'property': k,
                                'baseline': source.get(k), 'variation': counterpart.get(k)})
    for k in ('charge_c', 'position_m'):
        a, b = left['particle'][k], right['particle'][k]
        if a != b:
            changed.append({'component_id': 'test-particle', 'property': k,
                            'baseline': a, 'variation': b})
    for k in ('mass_kg', 'initial_velocity_m_s'):
        a, b = left[k], right[k]
        if a != b:
            changed.append({'component_id': 'test-particle', 'property': k,
                            'baseline': a, 'variation': b})
    k = 'permittivity_f_m'
    if left['field'][k] != right['field'][k]:
        changed.append({'component_id': 'medium', 'property': k,
                        'baseline': left['field'][k], 'variation': right['field'][k]})
    if not changed:
        raise ValueError('no physical or potential-reference revision found; two identical models do not form this comparison')
    return changed


def prepare_scenario_comparison(baseline, variation, *,
                                baseline_id='baseline', variation_id='variation',
                                baseline_title='Baseline', variation_title='Variation',
                                baseline_provenance='Original authored model',
                                variation_provenance='Explicit teacher hypothetical'):
    """Return a small, validated comparison manifest; preserve both inputs.

    Each motion bundle must independently pass the same strict M19b/M19e
    source/field/force/energy checks as a standalone export. This does not
    authenticate the underlying RK history, which needs separate references.
    Equal real frame and sample times are required: no browser interpolation.
    """
    if baseline is variation:
        raise ValueError('scenario inputs must be distinct independent bundles')
    if validate_prescribed_viewer(baseline) is not True or validate_prescribed_viewer(variation) is not True:
        raise ValueError('both scenarios must be independently valid prescribed-motion viewer bundles')
    bid, vid = _id(baseline_id, 'baseline_id'), _id(variation_id, 'variation_id')
    if bid == vid:
        raise ValueError('scenario IDs must be distinct')
    titles = (_text(baseline_title,'baseline title',100), _text(variation_title,'variation title',100))
    notes = (_text(baseline_provenance,'baseline provenance',500),
             _text(variation_provenance,'variation provenance',500))
    b, v = baseline['prescribed_motion'], variation['prescribed_motion']
    if (baseline.get('units') != variation.get('units') or
            b.get('energy_convention') != v.get('energy_convention')):
        raise ValueError('comparison requires consistent SI units and energy conventions')
    if b.get('axis_display','vertical') != v.get('axis_display','vertical'):
        raise ValueError('scenarios use different schematic axis orientations')
    if b['axis'] != v['axis']:
        raise ValueError('scenarios lack a common certified physical symmetry axis')
    # Both viewers must render the same axes in world space, not fake a
    # difference by swapping screen projections of the unchanged motion.
    plane0, plane1 = baseline['frames'][0]['plane'], variation['frames'][0]['plane']
    if plane0 != plane1:
        raise ValueError('scenarios require identical physical slice origins, axes, scales and grid resolution; use a shared authored extent')
    rows0,rows1=b['samples'],v['samples']
    times0=[r['time_s'] for r in rows0]
    times1=[r['time_s'] for r in rows1]
    if times0 != times1:
        raise ValueError('comparison requires identical saved physical time grids; no interpolation is supplied')
    index0,index1=b['frame_sample_indices'],v['frame_sample_indices']
    if index0 != index1:
        raise ValueError('comparison requires identical original saved-field frame times')
    if b['termination'] != v['termination'] or b['termination'] != 'duration':
        raise ValueError('comparison requires two complete duration histories; do not hide source-guard censoring')
    differences = _changes(b,v)
    return {'schema':'conductor_sim.scenario_comparison.v1',
            'mode':'two complete prescribed-field trajectories; shared real sample times; no browser physics',
            'scenario_ids':[bid,vid],'titles':list(titles),'provenance':list(notes),
            'source_ids':list(_source_map(b)),
            'source_inventories':[b['field']['sources'],v['field']['sources']],
            'plane':plane0,
            'frame_sample_indices':list(index0),
            'frame_times_s':[times0[i] for i in index0],
            'physical_sample_count':len(times0),
            'revisions':differences,
            'position_m':[ [rows0[i]['position_m'],rows1[i]['position_m']] for i in index0],
            'velocity_m_s':[ [rows0[i]['velocity_m_s'],rows1[i]['velocity_m_s']] for i in index0],
            'kinetic_energy_j':[ [rows0[i]['kinetic_energy_j'],rows1[i]['kinetic_energy_j']] for i in index0],
            'potential_energy_j':[ [rows0[i]['potential_energy_j'],rows1[i]['potential_energy_j']] for i in index0],
            'time_interpolation':False,
            'source_model_changed':True}


def render_scenario_comparison(baseline, variation, *,
                               title='Electrostatic scenario comparison', **kwargs):
    """Self-contained HTML with *original* full viewers side by side.

    Only frame selection, playback clocks and numerical differences are new.
    There is no added physical observation, renderer copy or engine update.
    """
    title=_text(title,'comparison title',120)
    manifest=prepare_scenario_comparison(baseline,variation,**kwargs)
    original=[render_html(baseline,title=manifest['titles'][0]),
              render_html(variation,title=manifest['titles'][1])]
    metadata=json.dumps(manifest,allow_nan=False,separators=(',',':'))
    metadata=metadata.replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e')
    def h(x): return escape(str(x), quote=True)
    changes=''.join('<tr><td>'+h(c['component_id'])+'</td><td>'+h(c['property'])+'</td><td>'+h(json.dumps(c['baseline']))+'</td><td>'+h(json.dumps(c['variation']))+'</td></tr>' for c in manifest['revisions'])
    bodies=''.join('<article class="card"><h2>'+h(manifest['titles'][i])+'</h2><p class="meta">'+h(manifest['scenario_ids'][i])+' · '+h(manifest['provenance'][i])+'</p><p class="readout" id="readout-'+str(i)+'">Waiting for validated viewer</p><iframe id="viewer-'+str(i)+'" title="'+h(manifest['titles'][i])+' verified electrostatic viewer" srcdoc="'+h(original[i])+'" loading="eager"></iframe></article>' for i in range(2))
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'''+h(title)+'''</title><style>
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#183047;font:15px/1.5 system-ui,sans-serif}header{background:#183047;color:white;padding:20px max(16px,calc((100vw - 1450px)/2))}h1{font-size:25px;margin:0 0 5px}.sub{color:#d9e5eb;font-size:13px}main{max-width:1500px;margin:auto;padding:16px}.card{background:white;border:1px solid #cbd8e2;border-radius:9px;padding:15px;min-width:0}.meta{color:#506779;font-size:12px}h2{margin:0 0 6px;font-size:17px}h3{font-size:15px}button,select{font:inherit;border-radius:5px;border:1px solid #abbccc;background:white;color:#183047;padding:7px 12px}button{cursor:pointer}button:focus-visible,input:focus-visible{outline:3px solid #197b9d;outline-offset:2px}.playrow{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.playrow input{flex:1;min-width:170px}.panels{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px;margin-top:16px}iframe{width:100%;height:1040px;border:1px solid #cad9e5;border-radius:5px;background:white}.readout{min-height:60px;background:#edf4f8;padding:9px;border-radius:5px;font:12px/1.6 ui-monospace,monospace;white-space:pre-wrap}table{border-collapse:collapse;width:100%;font-size:12px}th,td{text-align:left;padding:7px;border-bottom:1px solid #dfe7ef;overflow-wrap:anywhere}th{background:#f4f7fa}.overflow{overflow-x:auto}p{margin:7px 0}.notice{font-size:12px;color:#506779}.controls{margin-bottom:16px}@media(max-width:900px){.panels{grid-template-columns:1fr}iframe{height:900px}h1{font-size:20px}}
</style></head><body><header><h1>'''+h(title)+'''</h1><p class="sub">Two separately validated fixed-field scenarios · one shared physical clock · no interpolated physics or browser integration</p></header><main>
<section class="card controls"><div class="playrow"><button id="compare-play" type="button" aria-pressed="false">Play</button><button id="compare-reset" type="button">Restart</button><label for="compare-time">Elapsed physical time</label><input type="range" id="compare-time" min="0" max="'''+str(len(manifest['frame_times_s'])-1)+'''" step="1" value="0"><span id="compare-clock" role="status">0 s</span><label for="compare-speed">Playback</label><select id="compare-speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select></div><p class="notice">Shared slider selects exactly the same <strong>saved physical time</strong> for both simulations. Display times are discrete; no interpolation, new field solve or time stepping takes place in this HTML. Both scenarios must complete the same duration; truncated/source-guard histories are refused. Individual panels contain the original M19e drawings and graphs.</p><p id="compare-delta" class="readout" aria-live="off"></p></section>
<div class="panels">'''+bodies+'''</div><section class="card" style="margin-top:16px"><h3>Verified source and initial-condition revisions</h3><p class="notice">IDs and scenario provenance are authored assertions; differences below are computed from the two complete validated physical model inventories. A numerical comparison at a common saved time does not mean the particle travelled between scenarios. Absolute qV values may differ by voltage-gauge choice.</p><div class="overflow"><table><thead><tr><th>Physical identity</th><th>Changed property</th><th>Baseline</th><th>Variation</th></tr></thead><tbody>'''+changes+'''</tbody></table></div></section></main><script type="application/json" id="comparison-data">'''+metadata+'''</script><script>
'use strict';
const comparison=JSON.parse(document.getElementById('comparison-data').textContent);
const frames=[document.getElementById('viewer-0'),document.getElementById('viewer-1')];
const clock=document.getElementById('compare-time'),clockText=document.getElementById('compare-clock');
const play=document.getElementById('compare-play'),speed=document.getElementById('compare-speed');
let loaded=[false,false],playing=false,raf=null,startClock=null,startTime=0;
function fmt(n){return Number(n).toPrecision(5)}
function pause(){playing=false;if(raf!==null)cancelAnimationFrame(raf);raf=null;startClock=null;play.textContent='Play';play.setAttribute('aria-pressed','false')}
function seek(i){if(!Number.isInteger(i)||i<0||i>=comparison.frame_times_s.length)throw Error('invalid saved time index');
 clock.value=String(i);const t=comparison.frame_times_s[i];clockText.textContent=fmt(t)+' s · saved snapshot '+(i+1)+'/'+comparison.frame_times_s.length;
 for(let side=0;side<2;side++){
  const p=comparison.position_m[i][side],v=comparison.velocity_m_s[i][side];
  document.getElementById('readout-'+side).textContent='Physical t = '+fmt(t)+' s\\nposition (m): ['+p.map(fmt).join(', ')+']\\nvelocity (m/s): ['+v.map(fmt).join(', ')+']\\nK = '+fmt(comparison.kinetic_energy_j[i][side])+' J · qV = '+fmt(comparison.potential_energy_j[i][side])+' J';
  if(!loaded[side])continue;
  const iframe=frames[side].contentWindow,slider=iframe.document.getElementById('frame');
  if(!slider||!iframe.conductorViewer||iframe.conductorViewer.mode!=='elapsed-time-prescribed')throw Error('validated prescribed viewer not loaded');
  slider.value=String(i);slider.dispatchEvent(new iframe.Event('input',{bubbles:true}));
 }
 const a=comparison.position_m[i],b=comparison.velocity_m_s[i];
 const dp=Math.hypot(...a[0].map((x,j)=>x-a[1][j]));
 const dv=Math.hypot(...b[0].map((x,j)=>x-b[1][j]));
 document.getElementById('compare-delta').textContent='At equal saved physical time t = '+fmt(t)+' s: |r(variation) − r(baseline)| = '+fmt(dp)+' m; |v(variation) − v(baseline)| = '+fmt(dv)+' m/s. These are comparisons between independently computed states.';
}
function onFrameLoaded(side){const iframe=frames[side].contentWindow;const v=iframe&&iframe.conductorViewer;
 if(!v||v.mode!=='elapsed-time-prescribed'||v.frameCount!==comparison.frame_times_s.length){document.getElementById('readout-'+side).textContent='Error: expected a complete prescribed-field viewer';return}
 iframe.document.querySelector('header').hidden=true;
 iframe.document.querySelector('main > section.card.wide').hidden=true;
 loaded[side]=true;seek(Number(clock.value));
}
frames.forEach((frame,i)=>frame.addEventListener('load',()=>onFrameLoaded(i)));
clock.addEventListener('input',()=>{pause();seek(Number(clock.value))});
document.getElementById('compare-reset').addEventListener('click',()=>{pause();seek(0)});
speed.addEventListener('change',()=>{if(playing){pause();start()}});
function tick(now){if(!playing)return;if(startClock===null)startClock=now;const times=comparison.frame_times_s,last=times[times.length-1];const goal=Math.min(last,startTime+(now-startClock)*last*Number(speed.value)/4000);
 let i=Number(clock.value);while(i+1<times.length&&times[i+1]<=goal)i++;if(Number(clock.value)!==i)seek(i);
 if(goal>=last){seek(times.length-1);pause();return}raf=requestAnimationFrame(tick)}
function start(){if(!loaded.every(Boolean))return;if(Number(clock.value)===comparison.frame_times_s.length-1)seek(0);
 playing=true;play.textContent='Pause';play.setAttribute('aria-pressed','true');startClock=null;startTime=comparison.frame_times_s[Number(clock.value)];raf=requestAnimationFrame(tick)}
play.addEventListener('click',()=>playing?pause():start());
document.addEventListener('visibilitychange',()=>{if(document.hidden)pause()});
seek(0);window.conductorComparison={schema:comparison.schema,sourceIDs:comparison.source_ids,frameCount:comparison.frame_times_s.length,get loaded(){return loaded.every(Boolean)},get playing(){return playing},get frameIndex(){return Number(clock.value)},get revisions(){return comparison.revisions.length}};
</script></body></html>'''


def write_scenario_comparison(baseline, variation, path, **kwargs):
    """Write one fully self-contained comparison HTML; never mutate inputs."""
    html=render_scenario_comparison(baseline,variation,**kwargs)
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(html,encoding='utf-8')
    return path
