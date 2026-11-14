"""M19f: compare two *already solved* fixed-field teaching scenarios.

No physics evaluation, source modification, trajectory integration or browser
interpolation happens in this adapter. Both saved M19e viewer bundles are
revalidated and their real field-frame times must agree before synchronization.
"""
from __future__ import annotations

from html import escape
import json
from math import hypot
from pathlib import Path

from .prescribed_viewer_validation import validate_prescribed_viewer
from .viewer import render_html

SCHEMA = 'conductor_sim.prescribed_comparison.v1'


def _label(value, what, *, limit=100):
    if type(value) is not str or not 1 <= len(value) <= limit or value != value.strip() or not value.isprintable():
        raise ValueError(f'{what} must be a short printable nonblank label')
    return value


def _changes(left, right):
    """Only actual authored model-data differences; no teacher prose as evidence."""
    old, new = left['prescribed_motion'], right['prescribed_motion']
    old_src, new_src = old['field']['sources'], new['field']['sources']
    old_ann, new_ann = old.get('source_annotations'), new.get('source_annotations')
    if old_ann is None or new_ann is None:
        raise ValueError('comparison requires stable source IDs in both M19e bundles')
    if len(old_ann) != len(new_ann):
        raise ValueError('fixed source inventory changed: this bounded comparison requires the same identities')
    if ([x['id'] for x in old_ann] != [x['id'] for x in new_ann] or
            [x['kind'] for x in old_ann] != [x['kind'] for x in new_ann]):
        raise ValueError('source ID/order/kind differs; do not imply unrelated sources are the same')
    changes=[]
    for a,b,ann in zip(old_src,new_src,old_ann):
        for key in sorted(set(a)|set(b)):
            if a.get(key)!=b.get(key):
                changes.append({'owner':ann['id'], 'property':key,
                                'before':a.get(key), 'after':b.get(key)})
    for owner,record_a,record_b in (
        ('field',old['field'],new['field']),('particle',old['particle'],new['particle']),
        ('motion',old,new)):
        fields=('permittivity_f_m',) if owner=='field' else (
            ('charge_c','position_m') if owner=='particle' else
            ('mass_kg','initial_velocity_m_s','duration_s','sample_interval_s',
             'source_guard_m','length_scale_m','plane_normal'))
        for key in fields:
            if record_a.get(key)!=record_b.get(key):
                changes.append({'owner':owner,'property':key,'before':record_a.get(key),
                                'after':record_b.get(key)})
    return changes


def compare_prescribed_scenarios(left_bundle, right_bundle, *,
                                labels=('Baseline','Teacher hypothetical'),
                                source_context='Two authored fixed-field scenarios'):
    """Validate and compare two compatible full M19e offline viewer bundles.

    Synchronization is accepted only for identical *actual* saved frame times;
    nothing is interpolated or estimated between physics samples in the browser.
    """
    if not isinstance(labels,(tuple,list)) or len(labels)!=2:
        raise ValueError('labels must contain exactly two labels')
    labels=[_label(v,'scenario label') for v in labels]
    if labels[0]==labels[1]:
        raise ValueError('scenario labels must be distinguishable')
    context=_label(source_context,'source context',limit=320)
    # Canonicalize for copy isolation and reject non-JSON floats/unknown types.
    bundles=[]
    for candidate in (left_bundle,right_bundle):
        try:
            copy=json.loads(json.dumps(candidate,allow_nan=False))
        except (ValueError,TypeError) as exc:
            raise ValueError('scenario bundle must be finite JSON-native data') from exc
        validate_prescribed_viewer(copy)
        bundles.append(copy)
    left,right=bundles
    a,b=left['prescribed_motion'],right['prescribed_motion']
    if a.get('profile_mode') != b.get('profile_mode'):
        raise ValueError('spatial axes and time-profile modes cannot be silently compared')
    if (a.get('axis_display','vertical')!=b.get('axis_display','vertical') or
            left['frames'][0]['plane'] != right['frames'][0]['plane']):
        raise ValueError('viewer physical planes/scales differ: a comparison requires matching axes, origin and metre scales')
    indices_a=a['frame_sample_indices'];indices_b=b['frame_sample_indices']
    times_a=[a['samples'][i]['time_s'] for i in indices_a]
    times_b=[b['samples'][i]['time_s'] for i in indices_b]
    if len(times_a)!=len(times_b) or any(x!=y for x,y in zip(times_a,times_b)):
        raise ValueError('saved physical frame times differ; no invented synchronized timestamps')
    changes=_changes(left,right)
    unchanged_sources=[ann['id'] for s0,s1,ann in zip(
        a['field']['sources'],b['field']['sources'],a['source_annotations']) if s0==s1]
    unchanged_probe_properties=[k for k in ('charge_c','position_m')
        if a['particle'][k]==b['particle'][k]]
    unchanged_motion_properties=[k for k in ('mass_kg','initial_velocity_m_s',
        'duration_s','sample_interval_s','source_guard_m','length_scale_m','plane_normal')
        if a[k]==b[k]]
    # Comparable, gauge-labelled physical readouts at *same saved* elapsed time.
    comparisons=[]
    for ia,ib,t in zip(indices_a,indices_b,times_a):
        row_a,row_b=a['samples'][ia],b['samples'][ib]
        dpos=[y-x for x,y in zip(row_a['position_m'],row_b['position_m'])]
        comparisons.append(dict(time_s=t,
            left_sample_index=ia,right_sample_index=ib,
            delta_position_m=dpos,delta_position_norm_m=hypot(*dpos),
            delta_speed_m_s=hypot(*row_b['velocity_m_s'])-hypot(*row_a['velocity_m_s']),
            delta_kinetic_energy_j=row_b['kinetic_energy_j']-row_a['kinetic_energy_j'],
            delta_potential_energy_j=row_b['potential_energy_j']-row_a['potential_energy_j'],
            delta_total_energy_j=row_b['total_energy_j']-row_a['total_energy_j']))
    result={'schema':SCHEMA,'source_context':context,'labels':labels,
            'scope':'Two independently solved fixed-source test-particle motions; no new force law, no browser integration, no implied causal proof.',
            'limitations':'U=qV and total energy are gauge/reference-dependent across scenarios; graph differences compare saved values at identical elapsed times, not work between scenarios.',
            'source_ids':[x['id'] for x in a['source_annotations']],
            'changes':changes,'unchanged_source_identities':True,
            'unchanged_source_values':unchanged_sources,
            'unchanged_particle_properties':unchanged_probe_properties,
            'unchanged_motion_properties':unchanged_motion_properties,
            'time_s':times_a,'samples':comparisons,
            'scenario_bundles':bundles}
    return result


def sample_prescribed_scenario_comparison(left_spec, right_spec, *,
                                          result_left=None, result_right=None,
                                          source_annotations,
                                          labels=('Baseline','Teacher hypothetical'),
                                          source_context='Two authored fixed-field scenarios',
                                          axis_orientation='horizontal',
                                          visual_frames=12, profile_samples=101,
                                          resolution=15):
    """Run each accepted M19a spec once and export two metrically aligned views.

    A single common slice extent is selected from the two actual saved paths.
    Both views must use identical world origin/basis, or comparison refuses.
    """
    from .prescribed_motion import TestChargeMotionSpec, integrate_test_charge
    from .prescribed_viewer import sample_test_charge_viewer, test_charge_slice
    if not isinstance(left_spec,TestChargeMotionSpec) or not isinstance(right_spec,TestChargeMotionSpec):
        raise TypeError('comparison requires two TestChargeMotionSpec objects')
    if result_left is None:
        result_left=integrate_test_charge(left_spec)
    if result_right is None:
        result_right=integrate_test_charge(right_spec)
    a=test_charge_slice(left_spec,result_left,resolution=resolution,
                        axis_orientation=axis_orientation)
    b=test_charge_slice(right_spec,result_right,resolution=resolution,
                        axis_orientation=axis_orientation)
    extent=max(a.x_range_m[1],b.x_range_m[1],a.y_range_m[1],b.y_range_m[1])
    options=dict(source_annotations=source_annotations,axis_orientation=axis_orientation,
                 visual_frames=visual_frames,profile_samples=profile_samples,
                 resolution=resolution,extent_m=extent)
    left=sample_test_charge_viewer(left_spec,result=result_left,**options)
    right=sample_test_charge_viewer(right_spec,result=result_right,**options)
    return compare_prescribed_scenarios(left,right,labels=labels,source_context=source_context)


def validate_comparison(comparison):
    if not isinstance(comparison,dict) or comparison.get('schema')!=SCHEMA:
        raise ValueError('expected prescribed comparison v1')
    source=comparison.get('scenario_bundles')
    if not isinstance(source,list) or len(source)!=2:
        raise ValueError('comparison must have two verified source bundles')
    rebuilt=compare_prescribed_scenarios(*source,labels=comparison.get('labels'),
                                         source_context=comparison.get('source_context'))
    if comparison!=rebuilt:
        raise ValueError('comparison metadata or derived physics was altered')
    return rebuilt


def render_scenario_comparison_html(comparison, *, title='Fixed-field scenario comparison'):
    """One standalone offline HTML with two existing viewers and one honest cursor."""
    comparison=validate_comparison(comparison)
    title=_label(title,'comparison title')
    a,b=comparison['scenario_bundles']
    frame_html=[render_html(a,title=comparison['labels'][0]),
                render_html(b,title=comparison['labels'][1])]
    # The exact same recorded frame times are mandatory, so frames can be
    # addressed by their REAL saved indices without inventing interpolation.
    metadata={key:comparison[key] for key in ('labels','source_context','scope','limitations','changes','time_s','samples','source_ids',
                            'unchanged_source_values','unchanged_particle_properties','unchanged_motion_properties')}
    data=json.dumps(metadata,ensure_ascii=True,allow_nan=False,separators=(',',':'))
    data=data.replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    f_a=escape(frame_html[0],quote=True)
    f_b=escape(frame_html[1],quote=True)
    h=escape(title,quote=True)
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{h}</title>
<style>
:root{{color-scheme:light;--ink:#15283a;--muted:#53647b;--line:#dbe3ea}}
*{{box-sizing:border-box}}body{{margin:0;background:#f2f5f9;font:15px/1.5 system-ui,sans-serif;color:var(--ink)}}
header{{background:#102c42;color:#fff;padding:22px max(20px,calc((100vw - 1400px)/2))}}
h1{{font-size:25px;margin:3px 0}}p{{margin:6px 0}}main{{max-width:1480px;margin:auto;padding:18px}}
.card{{border:1px solid var(--line);background:white;border-radius:10px;padding:17px;margin-bottom:16px}}
.controls{{display:flex;align-items:center;gap:12px;flex-wrap:wrap}}
button,select{{font:inherit;padding:7px 11px;border:1px solid #bac8d4;background:#fff;border-radius:5px}}
button:focus-visible,input:focus-visible{{outline:3px solid #008a9c;outline-offset:3px}}
input[type=range]{{flex:1;min-width:180px}}.small{{font-size:12px;color:var(--muted)}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}
iframe{{width:100%;height:1160px;border:1px solid var(--line);background:white;border-radius:6px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border-bottom:1px solid var(--line);padding:8px;text-align:left}}
.number{{font:13px ui-monospace,monospace}}strong{{font-weight:700}}.warning{{color:#775327}}
@media(max-width:1020px){{.grid{{grid-template-columns:1fr}}iframe{{height:920px}}}}
</style></head><body><header><div class="small" style="color:#a9d3d8">CONDUCTOR_SIM / M19F · TWO VERIFIED SCENARIOS</div>
<h1>{h}</h1><p>{escape(comparison['source_context'])}</p>
<p style="font-size:13px;color:#c9d9e4">Prescribed sources and real saved physical trajectories. Shared controls select only matching saved times; neither viewer computes forces or motion.</p></header>
<main><section class="card"><h2>Actual model revisions</h2><p id="changes"></p><p class="small" id="unchanged"></p>
<p class="small warning">Potential energy qV and total energy depend on the potential reference; their differences across scenarios are not automatically work.</p></section>
<section class="card"><div class="controls"><button id="compare-play" type="button" aria-pressed="false">Play both</button>
<button id="compare-reset" type="button">Restart</button><label for="compare-frame">Shared elapsed-time frame</label>
<input id="compare-frame" type="range" min="0" max="{len(comparison['time_s'])-1}" step="1" value="0"><output id="compare-time" role="status"></output></div>
<p class="small">Both renderers show the same physical elapsed time. Playback selects saved samples, with no interpolation of the field, particle position, or time. Pause before using the individual viewer controls.</p>
<table><tbody><tr><th>At selected time</th><th>Difference (right − left)</th></tr>
<tr><td>Particle position difference magnitude</td><td class="number" id="delta-pos"></td></tr>
<tr><td>Signed difference in speed</td><td class="number" id="delta-speed"></td></tr>
<tr><td>Kinetic energy difference</td><td class="number" id="delta-kinetic"></td></tr>
<tr><td>Potential energy difference (gauge-dependent)</td><td class="number" id="delta-potential"></td></tr></tbody></table></section>
<section class="grid"><div class="card"><h2>{escape(comparison['labels'][0])}</h2><iframe id="scenario-left" title="{escape(comparison['labels'][0],quote=True)}" srcdoc="{f_a}"></iframe></div>
<div class="card"><h2>{escape(comparison['labels'][1])}</h2><iframe id="scenario-right" title="{escape(comparison['labels'][1],quote=True)}" srcdoc="{f_b}"></iframe></div></section>
<section class="card small">Each simulation retains its own K, qV, E and position graphs. Stable source IDs identify physical sources, not a claim that physical parameter values are unchanged. This is an offline educational comparison, not time-dependent source evolution or a fully coupled system.</section></main>
<script id="comparison-data" type="application/json">{data}</script>
<script>
'use strict';
const model=JSON.parse(document.getElementById('comparison-data').textContent);
const $=id=>document.getElementById(id);const frames=model.time_s.length;
let running=false,raf=null,start=0,base=0;
const format=(v,unit)=>Number(v).toPrecision(6)+' '+unit;
const change=model.changes;
$('changes').textContent=change.length?change.map(x=>x.owner+'.'+x.property+': '+JSON.stringify(x.before)+' → '+JSON.stringify(x.after)).join(' · '):'No physical or numerical source values changed';
$('unchanged').textContent='Source identities: '+model.source_ids.join(', ')+'. Unchanged source physical values: '+(model.unchanged_source_values.join(', ')||'none')+'. Unchanged particle properties: '+model.unchanged_particle_properties.join(', ')+'. Unchanged motion inputs: '+model.unchanged_motion_properties.join(', ')+'. Original saved models are immutable.';
function update(){{
 const i=Number($('compare-frame').value),row=model.samples[i];
 $('compare-time').textContent=format(row.time_s,'s')+' · saved frame '+(i+1)+'/'+frames;
 $('delta-pos').textContent=format(row.delta_position_norm_m,'m');
 $('delta-speed').textContent=format(row.delta_speed_m_s,'m/s');
 $('delta-kinetic').textContent=format(row.delta_kinetic_energy_j,'J');
 $('delta-potential').textContent=format(row.delta_potential_energy_j,'J');
 for(const id of ['scenario-left','scenario-right']){{
  const frame=$(id);try{{const d=frame.contentDocument;const input=d&&d.getElementById('frame');
  if(input&&d.defaultView?.conductorViewer){{const play=d.getElementById('motion-play');
   if(play?.getAttribute('aria-pressed')==='true')play.click();
   input.value=String(i);input.dispatchEvent(new Event('input',{{bubbles:true}}));
  }} }}catch(err){{console.error('Comparison could not control an embedded viewer',err)}}
 }}
}}
function pause(){{running=false;if(raf!==null)cancelAnimationFrame(raf);raf=null;$('compare-play').textContent='Play both';$('compare-play').setAttribute('aria-pressed','false')}}
function tick(now){{if(!running)return;const elapsed=(now-start)/1000;
 const full=model.time_s[frames-1]-model.time_s[0],goal=model.time_s[base]+elapsed*full/5;
 let i=base;while(i+1<frames&&model.time_s[i+1]<=goal)i++;
 $('compare-frame').value=String(i);update();
 if(i===frames-1){{pause();return}}raf=requestAnimationFrame(tick);
}}
$('compare-play').addEventListener('click',()=>{{if(running){{pause();return}}if(Number($('compare-frame').value)===frames-1)$('compare-frame').value='0';base=Number($('compare-frame').value);running=true;start=performance.now();$('compare-play').textContent='Pause both';$('compare-play').setAttribute('aria-pressed','true');raf=requestAnimationFrame(tick)}});
$('compare-reset').addEventListener('click',()=>{{pause();$('compare-frame').value='0';update()}});
$('compare-frame').addEventListener('input',()=>{{pause();update()}});
for(const id of ['scenario-left','scenario-right'])$(id).addEventListener('load',update);
document.addEventListener('visibilitychange',()=>{{if(document.hidden)pause()}});
update();
window.scenarioComparison={{mode:'two-prescribed-scenarios',get frameCount(){{return frames}},get playing(){{return running}},get index(){{return Number($('compare-frame').value)}}}};
</script></body></html>'''


def write_scenario_comparison(comparison, path, *, title='Fixed-field scenario comparison'):
    output=Path(path)
    rendered=render_scenario_comparison_html(comparison,title=title)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(rendered,encoding='utf-8')
    return output
