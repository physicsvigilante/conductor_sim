"""Real offline Chromium acceptance for M19b prescribed-field playback.

Tests rendered diagrams, time controls, source/particle separation, actual
painted force/velocity vectors, axial-profile gaps and mobile layout. This
checks the browser independently from the Python trajectory producer.
"""
from __future__ import annotations

import argparse
import json
from math import hypot
from pathlib import Path
import shutil

from playwright.sync_api import sync_playwright, Error as PlaywrightError


def run(path,screenshot=None):
    assert path.is_file(),path
    errors,requests=[],[]
    with sync_playwright() as playwright:
        try:
            browser=playwright.chromium.launch(headless=True)
        except PlaywrightError:
            executable=shutil.which('chromium') or shutil.which('chromium-browser')
            if executable is None:
                raise
            browser=playwright.chromium.launch(headless=True,executable_path=executable,
                                               args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1240,'height':920},device_scale_factor=1)
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('request',lambda r:requests.append(r.url))
        html=path.read_text(encoding='utf-8')
        page.set_content(html,wait_until='load')
        data=page.evaluate('data');motion=data['prescribed_motion']
        original=json.dumps(data,sort_keys=True)
        assert data['schema']=='conductor_sim.visualization.v1'
        assert page.evaluate('window.conductorViewer.mode')=='elapsed-time-prescribed'
        assert not any(k in data for k in ('position_sweep','motion_1d','motion_2d'))
        assert page.evaluate('window.conductorViewer.motionSampleCount')==len(motion['samples'])
        assert page.locator('#motion-controls').is_visible()
        assert page.locator('#motion-panel').is_visible()
        assert page.locator('#prescribed-panel').is_visible()
        assert page.locator('#scene').is_visible()
        assert page.locator('#conductor-panel').is_hidden()
        assert page.locator('#density-panel').is_hidden()
        assert page.locator('#potential-key').is_hidden()
        assert page.locator('#line-density-control').is_hidden()
        assert page.locator('#heat').locator('..').is_hidden()
        assert page.locator('#contours').locator('..').is_hidden()
        assert page.locator('#lines').locator('..').is_hidden()
        assert page.locator('#force-toggle').is_visible()
        assert page.locator('#velocity-toggle').is_visible()
        assert page.locator('#trajectory-toggle').is_visible()
        assert page.locator('#fixed-field-chart').is_visible()
        assert page.locator('#fixed-potential-chart').is_visible()
        assert page.locator('#motion-energy-chart').is_visible()
        assert 'U=qV' in page.locator('#motion-summary').inner_text()
        assert 'test particle' in page.locator('#sources').inner_text()
        assert 'Fixed source' in page.locator('#sources').inner_text()
        assert 'No conductor' in page.locator('#sources').inner_text()
        assert 'schematic' in page.locator('#scene-heading').inner_text().lower()
        assert motion['field_frame_count']==len(data['frames'])==page.evaluate('window.conductorViewer.frameCount')
        assert motion['graph_sample_count']==len(motion['samples'])
        assert all(a['time_s']<b['time_s'] for a,b in zip(motion['samples'],motion['samples'][1:]))
        assert all(a['grid']['potential_v']==[[None,None],[None,None]] for a in data['frames'])
        assert all(a['grid']['electric_field_v_per_m']==[[None,None],[None,None]] for a in data['frames'])
        assert all(not a['conductor_results'] and not a['field_lines'] and not a['contours'] for a in data['frames'])
        assert all(a['charges'][0]['position_m']==motion['samples'][i]['position_m']
                   for a,i in zip(data['frames'],motion['frame_sample_indices']))
        assert motion['profile_mode'] in ('axis','elapsed-time')
        if motion['profile_mode']=='axis':
            assert motion['profile_samples'] is not None
            if any(src['kind']=='UniformDiskAxisField' for src in motion['field']['sources']):
                assert any(point['field_parallel_v_m'] is None and point['potential_v'] is not None
                           for point in motion['profile_samples'])
        else:
            assert motion['profile_samples'] is None
        assert not errors and not requests,(errors,requests)

        vectors=0
        for i,index in enumerate(motion['frame_sample_indices']):
            page.locator('#frame').fill(str(i))
            page.locator('#frame').dispatch_event('input')
            row=motion['samples'][index]
            assert 'U=qV' in page.locator('#motion-summary').inner_text()
            assert 'E = ' in page.locator('#motion-summary').inner_text()
            assert 'V = ' in page.locator('#motion-summary').inner_text()
            assert page.locator('#motion-clock').inner_text().startswith('t = ')
            for name,key in (('forceArrow','force_n'),('velocityArrow','velocity_m_s')):
                ops=page.evaluate('''name=>{
                  const ops=[],m=ctx.moveTo,l=ctx.lineTo,p=frame.plane;
                  ctx.moveTo=function(x,y){ops.push(['m',x,y]);return m.call(this,x,y)};
                  ctx.lineTo=function(x,y){ops.push(['l',x,y]);return l.call(this,x,y)};
                  const project=x=>{const d=x.map((v,i)=>v-p.origin_m[i]);
                    return [p.u,p.v].map(a=>a.reduce((s,v,i)=>s+v*d[i],0)).concat([0])};
                  try{(name==='forceArrow'?forceArrow:velocityArrow)(p,x=>x,y=>-y,project)}
                  finally{ctx.moveTo=m;ctx.lineTo=l}
                  return ops.slice(0,2);
                }''',name)
                plane=data['frames'][i]['plane'];v=row[key]
                expected=(sum(v[j]*plane['u'][j] for j in range(3)),
                          -sum(v[j]*plane['v'][j] for j in range(3)))
                if hypot(*expected)<=1e-28:
                    continue
                assert len(ops)==2,(name,index,ops)
                origin=(sum((row['position_m'][j]-plane['origin_m'][j])*plane['u'][j]
                            for j in range(3)),
                        -sum((row['position_m'][j]-plane['origin_m'][j])*plane['v'][j]
                            for j in range(3)))
                assert hypot(ops[0][1]-origin[0],ops[0][2]-origin[1])<1e-10
                actual=(ops[1][1]-ops[0][1],ops[1][2]-ops[0][2])
                assert sum(a*b for a,b in zip(actual,expected))/(hypot(*actual)*hypot(*expected))>1-1e-10
                vectors+=1
        assert vectors>=len(data['frames'])
        labels=page.evaluate('''()=>{
          const labels=[],orig=ctx.fillText;
          ctx.fillText=function(t,...args){labels.push(String(t));return orig.call(this,t,...args)};
          try{drawScene()}finally{ctx.fillText=orig}return labels;
        }''')
        assert any('test' in name for name in labels)
        assert any('S1' in name or 'fixed rod' in name for name in labels) or any(
            s['kind']=='UniformElectricField' for s in motion['field']['sources'])

        scene=page.locator('#scene')
        before=scene.evaluate('(c)=>c.toDataURL()')
        page.locator('#trajectory').uncheck()
        after=scene.evaluate('(c)=>c.toDataURL()')
        assert before!=after,'saved trajectory must be painted'
        page.locator('#trajectory').check()
        for control in ('velocity-arrow','force-arrow'):
            old=scene.evaluate('(c)=>c.toDataURL()')
            page.locator('#'+control).uncheck()
            assert old!=scene.evaluate('(c)=>c.toDataURL()'), control
            page.locator('#'+control).check()
        old=page.locator('#fixed-potential-chart').evaluate('(c)=>c.toDataURL()')
        page.locator('#frame').fill('0')
        page.locator('#frame').dispatch_event('input')
        assert old!=page.locator('#fixed-potential-chart').evaluate('(c)=>c.toDataURL()'), 'cursor not synchronized'
        page.locator('#motion-reset').click()
        assert page.locator('#frame').input_value()=='0'
        page.locator('#motion-speed').select_option('0.5')
        page.locator('#motion-play').click()
        page.wait_for_timeout(120)
        page.locator('#motion-play').click()
        assert not page.evaluate('window.conductorViewer.playing')
        index=page.locator('#frame').input_value()
        page.wait_for_timeout(100)
        assert page.locator('#frame').input_value()==index
        page.locator('#motion-speed').select_option('4')
        page.locator('#motion-play').click()
        page.wait_for_function('!window.conductorViewer.playing',timeout=10000)
        assert page.locator('#frame').input_value()==str(len(data['frames'])-1)
        assert ('Termination: '+motion['termination']) in page.locator('#motion-event').inner_text()
        page.locator('#frame').focus()
        page.keyboard.press('ArrowLeft')
        assert page.locator('#frame').input_value()==str(len(data['frames'])-2)
        assert json.dumps(page.evaluate('data'),sort_keys=True)==original
        if screenshot:
            screenshot.parent.mkdir(parents=True,exist_ok=True)
            page.screenshot(path=str(screenshot),full_page=True)
        page.set_viewport_size({'width':390,'height':780})
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1')
        assert page.locator('#motion-controls').is_visible()
        assert page.locator('#prescribed-panel').is_visible()
        assert page.locator('#motion-energy-chart').is_visible()
        assert not errors,errors
        assert not requests,requests
        browser.close()
    return dict(case=path.name,frames=len(motion['frame_sample_indices']),
                actual_numerical_samples=len(motion['samples']),
                profile_mode=motion['profile_mode'],vector_origin_direction_checks=vectors,
                chronological_time=True,source_geometry=True,source_probe_separation=True,
                no_fictitious_spatial_field_map=True,controls='passed',
                mobile_390px='passed',saved_data_unchanged=True,
                javascript_errors=errors,external_requests=requests)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('html',type=Path)
    parser.add_argument('--screenshot',type=Path)
    args=parser.parse_args()
    print(json.dumps(run(args.html,args.screenshot),indent=2))


if __name__=='__main__':
    main()
