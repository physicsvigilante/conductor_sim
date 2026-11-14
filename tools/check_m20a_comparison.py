"""Real Chromium acceptance for two synced *saved* physical time histories.

Uses a standalone HTML export without local network. The canonical M19e
viewers are embedded as srcdoc, never redrawn or reintegrated in the parent.
"""
from __future__ import annotations

import argparse
import json
from math import hypot, isclose
from pathlib import Path
import shutil

from playwright.sync_api import sync_playwright, Error as PlaywrightError


def check(html_path, screenshot=None, *, case='axial-field'):
    html_path=Path(html_path)
    if not html_path.is_file():
        raise FileNotFoundError(html_path)
    page_errors, external_requests, console_errors=[],[],[]
    with sync_playwright() as p:
        try:
            browser=p.chromium.launch(headless=True)
        except PlaywrightError:
            exe=shutil.which('chromium') or shutil.which('chromium-browser')
            if not exe: raise
            browser=p.chromium.launch(headless=True,executable_path=exe,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1440,'height':1000})
        page.on('pageerror',lambda e:page_errors.append(str(e)))
        page.on('request',lambda r:external_requests.append(r.url))
        page.on('console',lambda m:console_errors.append(m.text) if m.type=='error' else None)
        page.set_content(html_path.read_text(encoding='utf-8'),wait_until='load')
        page.wait_for_function('window.conductorComparison && window.conductorComparison.loaded',timeout=20000)
        m=page.evaluate('JSON.parse(document.getElementById("comparison-data").textContent)')
        assert m['schema']=='conductor_sim.scenario_comparison.v1'
        assert len(m['frame_times_s'])==12
        assert m['frame_times_s'][0]==0. and m['frame_times_s'][-1]==.25
        assert m['physical_sample_count']==126
        assert set(m['source_ids'])=={'ring:A','ring:B','field:external'}
        assert len(m['revisions'])==1
        assert 'Teacher hypothetical' in ' '.join(m['provenance'])
        assert page.evaluate('window.conductorComparison.frameCount')==12
        assert page.locator('#compare-time').input_value()=='0'
        assert page.locator('#compare-delta').inner_text().find('0.0000 m')>=0
        # Actual iframe globals expose the retained M19 viewer and remain
        # usable on their own; their timeline controls are hidden in compare.
        sources=page.evaluate('''() => [...document.querySelectorAll('iframe')].map(f=> {
          const w=f.contentWindow,d=w.document;
          return {mode:w.conductorViewer.mode,frames:w.conductorViewer.frameCount,
            index:Number(d.getElementById('frame').value),
            sampleCount:w.conductorViewer.motionSampleCount,hidden:d.querySelector('main > section.card.wide').hidden,
            sourceNotes:d.getElementById('sources').innerText};
        })''')
        assert len(sources)==2
        assert sources[0]['mode']==sources[1]['mode']=='elapsed-time-prescribed'
        assert sources[0]['frames']==sources[1]['frames']==12
        assert sources[0]['sampleCount']==sources[1]['sampleCount']==126
        assert m['plane']['u']==[1.,0.,0.]
        assert sources[0]['hidden'] and sources[1]['hidden']
        for s in sources:
            assert 'Ring A [ring:A]' in s['sourceNotes']
            assert 'Ring B [ring:B]' in s['sourceNotes']
        # Compare actual painted geometry: both panels use the same metric
        # world-space scale and preserve the independently drawn ring dots.
        paint=page.evaluate('''() => [...document.querySelectorAll('iframe')].map(f=> {
           const d=f.contentWindow.document,c=d.getElementById('scene'),g=c.getContext('2d'),p=JSON.parse(document.getElementById('comparison-data').textContent).plane;
           const W=c.width,H=c.height,pad=46;
           const sc=Math.min((W-2*pad)/(p.x_range_m[1]-p.x_range_m[0]),(H-2*pad)/(p.y_range_m[1]-p.y_range_m[0]));
           const X=x=>(W-sc*(p.x_range_m[1]-p.x_range_m[0]))/2+(x-p.x_range_m[0])*sc;
           const Y=y=>H-(H-sc*(p.y_range_m[1]-p.y_range_m[0]))/2-(y-p.y_range_m[0])*sc;
           const all=JSON.parse(document.getElementById('comparison-data').textContent);
           const side=f.id==='viewer-0'?0:1,s=all.source_inventories[side];
           return [s[0],s[1]].map(r=>Array.from(g.getImageData(
             Math.round(X(r.center_m[0]-p.origin_m[0])),Math.round(Y(r.radius_m)),1,1).data));
        })''')
        for panel in paint:
            assert panel[0][0]>130 and panel[0][0]>1.4*panel[0][2],panel
            assert panel[1][2]>115 and panel[1][2]>1.4*panel[1][0],panel
        for index in (1,6,11):
            page.locator('#compare-time').fill(str(index))
            t=m['frame_times_s'][index]
            assert page.evaluate('window.conductorComparison.frameIndex')==index
            panel_indices=page.evaluate("[...document.querySelectorAll('iframe')].map(f=>Number(f.contentWindow.document.getElementById('frame').value))")
            assert panel_indices==[index,index]
            assert str(index+1)+'/12' in page.locator('#compare-clock').inner_text()
            assert 'Physical t = ' in page.locator('#readout-0').inner_text()
            assert 'Physical t = ' in page.locator('#readout-1').inner_text()
            r=m['position_m'][index]; v=m['velocity_m_s'][index]
            dr=hypot(*(r[1][k]-r[0][k] for k in range(3)))
            dv=hypot(*(v[1][k]-v[0][k] for k in range(3)))
            assert f'{dr:.5g}' in page.locator('#compare-delta').inner_text() or isclose(dr,0,abs_tol=1e-12)
            assert f'{dv:.5g}' in page.locator('#compare-delta').inner_text() or isclose(dv,0,abs_tol=1e-12)
            assert isclose(float(page.locator('#compare-clock').inner_text().split()[0]),t,rel_tol=1e-4,abs_tol=1e-6)
        if case=='gauge-shift':
            assert all(x==y for x,y in m['position_m'])
            assert all(x==y for x,y in m['velocity_m_s'])
        else:
            assert m['position_m'][-1][0]!=m['position_m'][-1][1]
        page.locator('#compare-reset').click()
        assert page.locator('#compare-time').input_value()=='0'
        page.locator('#compare-speed').select_option('4')
        page.locator('#compare-play').click()
        page.wait_for_timeout(310)
        assert page.evaluate('window.conductorComparison.playing') is True
        current=page.evaluate('window.conductorComparison.frameIndex')
        assert current>=1,('time playback did not progress',current)
        assert page.evaluate("[...document.querySelectorAll('iframe')].map(f=>f.contentWindow.document.getElementById('frame').value)")==[str(current)]*2
        page.locator('#compare-play').click()
        assert page.evaluate('window.conductorComparison.playing') is False
        assert page.locator('#compare-play').inner_text()=='Play'
        if screenshot:
            Path(screenshot).parent.mkdir(parents=True,exist_ok=True)
            page.screenshot(path=str(screenshot),full_page=False)
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
        assert page.locator('#viewer-0').is_visible() and page.locator('#viewer-1').is_visible()
        assert not page_errors,page_errors
        assert not console_errors,console_errors
        assert not external_requests,external_requests
        browser.close()
        return dict(case=case,filename=html_path.name,
                    independent_viewer_count=2,physical_samples=m['physical_sample_count'],
                    synchronized_saved_frames=len(m['frame_times_s']),
                    source_ids=m['source_ids'],source_scene_pixels=paint,
                    source_revisions=m['revisions'],
                    actual_browser_play_pause_scrub_reset='passed',
                    world_axis_and_scale='matched',mobile_width_px=390,
                    javascript_errors=page_errors,external_requests=external_requests)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('html',type=Path)
    parser.add_argument('--case',choices=['axial-field','double-mass','ring-b-shift','gauge-shift'],default='axial-field')
    parser.add_argument('--screenshot',type=Path)
    args=parser.parse_args()
    print(json.dumps(check(args.html,args.screenshot,case=args.case),indent=2))


if __name__=='__main__':main()
