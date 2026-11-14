"""Actual offline Chromium checks of M21c trajectory and source visibility."""
from __future__ import annotations

import argparse
import json
from math import hypot
from pathlib import Path
import shutil

from playwright.sync_api import sync_playwright, Error as PlaywrightError


def check(path, screenshot=None):
    path=Path(path)
    errors, requests, console=[],[],[]
    with sync_playwright() as p:
        try:
            browser=p.chromium.launch(headless=True)
        except PlaywrightError:
            binary=shutil.which('chromium') or shutil.which('chromium-browser')
            if not binary:
                raise
            browser=p.chromium.launch(headless=True,executable_path=binary,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1360,'height':1000})
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('request',lambda r:requests.append(r.url))
        page.on('console',lambda m:console.append(m.text) if m.type=='error' else None)
        page.set_content(path.read_text(encoding='utf-8'),wait_until='load')
        page.wait_for_function("window.conductorViewer && window.conductorViewer.mode==='elapsed-time-prescribed'")
        data=page.evaluate('data'); original=json.dumps(data,sort_keys=True)
        m=data['prescribed_motion']; case=data['m21c_teaching']['case']
        assert m['termination']=='duration'
        assert len(m['field']['sources'])==2
        assert [s['id'] for s in m['source_annotations']]==['charge:upper','charge:lower']
        assert m['axis'] is None  # Not invented ring/guide coordinates.
        assert len(data['frames'])==16
        assert page.locator('#motion-controls').is_visible()
        assert page.locator('#scene-world-axes').is_visible()
        assert page.locator('#conductor-panel').is_hidden()
        assert page.locator('#potential-key').is_hidden()
        source_text=page.locator('#sources').inner_text()
        assert 'charge:upper' in source_text and 'charge:lower' in source_text
        # Capture calls from the real fixed-source draw routine alone. This
        # distinguishes painted sources from the independently drawn probe.
        painted=page.evaluate('''() => {
          const points=[],texts=[],arc=ctx.arc,text=ctx.fillText,pl=frame.plane;
          ctx.arc=function(x,y,r,...rest){points.push([x,y,r]);return arc.call(this,x,y,r,...rest)};
          ctx.fillText=function(t,...rest){texts.push(String(t));return text.call(this,t,...rest)};
          const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
          const project=p=>[pl.u,pl.v,cross(pl.u,pl.v)].map(a=>a.reduce((s,v,i)=>s+v*(p[i]-pl.origin_m[i]),0));
          try{drawPrescribedSources(pl,x=>x,y=>y,project)}finally{ctx.arc=arc;ctx.fillText=text;drawScene()}
          return {points,texts};
        }''')
        if case=='mirror-plane':
            assert painted['points']==[] and painted['texts']==[]
            notes=page.locator('#notes').inner_text()
            assert notes.count('signed normal offset')==2
            assert 'No in-plane replacement source' in notes
        else:
            assert len(painted['points'])==2
            assert hypot(*(a-b for a,b in zip(painted['points'][0][:2],painted['points'][1][:2])))==.2
            assert any('Upper fixed +Q' in t for t in painted['texts'])
            assert any('Lower fixed +Q' in t for t in painted['texts'])
        scene=page.locator('#scene')
        for index in (0,7,15):
            page.locator('#frame').fill(str(index))
            page.locator('#frame').dispatch_event('input')
            assert page.evaluate('frame.motion_sample_index')==m['frame_sample_indices'][index]
            assert 'U=qV' in page.locator('#motion-summary').inner_text()
        for control in ('trajectory','force-arrow','velocity-arrow'):
            before=scene.evaluate('c=>c.toDataURL()')
            page.locator('#'+control).uncheck()
            assert scene.evaluate('c=>c.toDataURL()')!=before, control
            page.locator('#'+control).check()
        page.locator('#motion-reset').click()
        assert page.locator('#frame').input_value()=='0'
        page.locator('#motion-speed').select_option('4')
        page.locator('#motion-play').click()
        page.wait_for_function("Number(document.getElementById('frame').value)>=2",timeout=5000)
        if page.evaluate('window.conductorViewer.playing'):
            page.locator('#motion-play').click()
        paused=page.locator('#frame').input_value()
        page.wait_for_timeout(100)
        assert page.locator('#frame').input_value()==paused
        assert not page.evaluate('window.conductorViewer.playing')
        page.locator('#motion-reset').click()
        assert page.locator('#frame').input_value()=='0'
        page.locator('#frame').fill('7')
        page.locator('#frame').dispatch_event('input')
        assert json.dumps(page.evaluate('data'),sort_keys=True)==original
        if screenshot:
            screenshot=Path(screenshot);screenshot.parent.mkdir(parents=True,exist_ok=True)
            page.screenshot(path=str(screenshot),full_page=True)
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1')
        assert page.locator('#motion-controls').is_visible()
        assert not errors and not console and not requests,(errors,console,requests)
        browser.close()
    return dict(case=case,physical_samples=len(m['samples']),snapshots=16,
                real_source_markers=len(painted['points']),
                off_slice_sources_omitted_and_named=case=='mirror-plane',
                saved_clock_and_play_pause_reset=True, vector_and_trajectory_pixels=True,
                mobile_width_px=390,javascript_errors=errors,console_errors=console,
                external_requests=requests)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('html',type=Path)
    parser.add_argument('--screenshot',type=Path)
    args=parser.parse_args()
    print(json.dumps(check(args.html,args.screenshot),indent=2))


if __name__=='__main__':
    main()
