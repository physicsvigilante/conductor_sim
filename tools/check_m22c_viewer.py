"""M22c: real offline Chromium acceptance of certified off-plane mirrored rods."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from playwright.sync_api import sync_playwright, Error as PlaywrightError


def check(path, *, case, screenshot=None):
    path=Path(path)
    errors=[];console=[];requests=[]
    with sync_playwright() as p:
        try:
            browser=p.chromium.launch(headless=True)
        except PlaywrightError:
            binary=shutil.which('chromium') or shutil.which('chromium-browser')
            if not binary: raise
            browser=p.chromium.launch(headless=True,executable_path=binary,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1360,'height':950})
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('console',lambda e:console.append(e.text) if e.type=='error' else None)
        page.on('request',lambda e:requests.append(e.url))
        page.set_content(path.read_text(encoding='utf-8'),wait_until='load')
        page.wait_for_function("window.conductorViewer && window.conductorViewer.mode==='elapsed-time-prescribed'")
        record=page.evaluate('data')
        initial=json.dumps(record,sort_keys=True)
        m=record['prescribed_motion']
        assert m['termination']=='duration'
        assert m['mirrored_rod_source_indices']==([0,1,2,3] if case=='R' else [0,1])
        assert m['axis'] is None
        assert len(record['frames'])==9
        assert len(m['samples'])==(13 if case=='R' else 15)
        assert [a['kind'] for a in m['source_annotations'][:2]]==['UniformRodField']*2
        assert m['field']['sources'][0]['charge_c']==m['field']['sources'][1]['charge_c']
        assert page.locator('#motion-controls').is_visible()
        assert page.locator('#conductor-panel').is_hidden()
        sources=page.locator('#sources').inner_text()
        assert 'rod:' in sources
        note=page.locator('#notes').inner_text()
        assert note.count('signed endpoint normal offsets')==(4 if case=='R' else 2)
        assert 'Not drawn here' in note
        assert 'full physical 3D field is included' in note
        assert 'not projected or replaced' in note
        drawn=page.evaluate('''() => {
          const texts=[],lines=[],origText=ctx.fillText,origLine=ctx.lineTo,pl=frame.plane;
          const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
          const project=p=>[pl.u,pl.v,cross(pl.u,pl.v)].map(d=>d.reduce((v,n,i)=>v+n*(p[i]-pl.origin_m[i]),0));
          ctx.fillText=function(s,...a){texts.push(String(s));return origText.call(this,s,...a)};
          ctx.lineTo=function(x,y){lines.push([x,y]);return origLine.call(this,x,y)};
          try{drawPrescribedSources(pl,x=>x,y=>y,project)}
          finally{ctx.fillText=origText;ctx.lineTo=origLine;drawScene()}
          return {texts,lines};
        }''')
        # The new off-plane rods must NOT be drawn as in-plane projected rods.
        if case!='R':
            assert not any('fixed rod' in t for t in drawn['texts']),drawn['texts']
        if case=='P1':
            assert not drawn['texts'] and not drawn['lines']
        elif case=='P2':
            assert any('Held in-plane point' in t for t in drawn['texts'])
            assert len(drawn['lines'])>=9
        else:
            assert len([t for t in drawn['texts'] if 'fixed rod' in t])==1
            assert any('rod inplane' in t for t in drawn['texts'])
            assert len(drawn['lines'])>=10 # genuine applied in-plane electric field arrows
        canvas=page.locator('#scene')
        assert canvas.is_visible()
        for i in (0,4,8):
            page.locator('#frame').fill(str(i))
            page.locator('#frame').dispatch_event('input')
            assert page.evaluate('frame.motion_sample_index')==m['frame_sample_indices'][i]
        for control in ('force-arrow','velocity-arrow','trajectory'):
            page.locator('#frame').fill('4')
            page.locator('#frame').dispatch_event('input')
            before=canvas.evaluate('c=>c.toDataURL()')
            page.locator('#'+control).uncheck()
            assert canvas.evaluate('c=>c.toDataURL()')!=before,control
            page.locator('#'+control).check()
        page.locator('#motion-reset').click()
        assert page.locator('#frame').input_value()=='0'
        page.locator('#motion-speed').select_option('4')
        page.locator('#motion-play').click()
        page.wait_for_function("Number(document.getElementById('frame').value)>=2",timeout=6000)
        if page.evaluate('window.conductorViewer.playing'):
            page.locator('#motion-play').click()
        position=page.locator('#frame').input_value()
        page.wait_for_timeout(85)
        assert page.locator('#frame').input_value()==position
        assert not page.evaluate('window.conductorViewer.playing')
        page.locator('#motion-reset').click()
        assert page.locator('#frame').input_value()=='0'
        assert json.dumps(page.evaluate('data'),sort_keys=True)==initial
        if screenshot:
            screenshot=Path(screenshot)
            screenshot.parent.mkdir(parents=True,exist_ok=True)
            page.screenshot(path=str(screenshot),full_page=True)
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1')
        assert page.locator('#motion-controls').is_visible()
        assert not errors and not console and not requests,(errors,console,requests)
        browser.close()
    return dict(case=case,source_ids=[a['id'] for a in m['source_annotations']],
                actual_browser='Chromium',physical_samples=len(m['samples']),
                frames=len(record['frames']),omitted_rod_ids=[m['source_annotations'][i]['id']
                                                 for i in m['mirrored_rod_source_indices']],
                canvas_rod_projection='not painted',
                actual_clock_scrub_play_pause_reset='passed',
                force_velocity_trajectory_toggles='passed',mobile_width_px=390,
                javascript_errors=errors,console_errors=console,external_requests=requests)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('html',type=Path)
    parser.add_argument('--case',choices=['P1','P2','R'],required=True)
    parser.add_argument('--screenshot',type=Path)
    args=parser.parse_args()
    print(json.dumps(check(args.html,case=args.case,screenshot=args.screenshot),indent=2))

if __name__=='__main__':main()
