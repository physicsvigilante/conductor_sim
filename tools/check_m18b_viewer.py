"""Actual offline Chromium acceptance of held-source motion in the shared viewer."""
from __future__ import annotations
import argparse
import json
from math import hypot
from pathlib import Path
import shutil
from playwright.sync_api import sync_playwright, Error as PlaywrightError


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('html',type=Path)
    parser.add_argument('--screenshot',type=Path)
    args=parser.parse_args()
    errors=[];requests=[];directions=0
    with sync_playwright() as playwright:
        try: browser=playwright.chromium.launch(headless=True)
        except PlaywrightError:
            binary=shutil.which('chromium') or shutil.which('chromium-browser')
            if binary is None:raise
            browser=playwright.chromium.launch(headless=True,executable_path=binary,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1240,'height':920})
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('request',lambda r:requests.append(r.url))
        page.set_content(args.html.read_text(encoding='utf-8'),wait_until='load')
        data=page.evaluate('data');motion=data['motion_2d']
        assert page.evaluate('window.conductorViewer.mode')=='elapsed-time-held-sources'
        assert page.locator('#motion-panel').is_visible() and page.locator('#force-panel').is_hidden()
        assert len(motion['stationary_charges'])>=1
        original=json.dumps(data,sort_keys=True)
        fixed={q['charge_index']:q for q in motion['stationary_charges']}
        moving=motion['moving_charge_index'];indices=motion['frame_sample_indices']
        assert indices[0]==0 and indices[-1]==len(motion['samples'])-1
        assert all(a['time_s']<b['time_s'] for a,b in zip(motion['samples'],motion['samples'][1:]))
        for i,idx in enumerate(indices):
            frame=data['frames'][i];row=motion['samples'][idx]
            inventory={q['charge_index']:q for q in frame['charges']}
            assert len(inventory)==len(fixed)+1
            assert inventory[moving]['position_m']==row['position_m']
            assert inventory[moving]['motion_role']=='moving'
            for identity,source in fixed.items():
                assert inventory[identity]['position_m']==source['position_m']
                assert inventory[identity]['charge_c']==source['charge_c']
                assert inventory[identity]['motion_role']=='held'
            page.locator('#frame').fill(str(i));page.locator('#frame').dispatch_event('input')
            assert f'q{moving} · moving' in page.locator('#sources').inner_text()
            for identity in fixed:assert f'q{identity} · held' in page.locator('#sources').inner_text()
            # Instrument actual arrow canvas calls: both direction and origin
            # must refer to the selected moving site, not sorted source zero.
            for name,key in [('forceArrow','force_n'),('velocityArrow','velocity_m_s')]:
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
                p=frame['plane'];v=row[key]
                expected=(sum(v[j]*p['u'][j] for j in range(3)), -sum(v[j]*p['v'][j] for j in range(3)))
                if hypot(*expected)<=1e-30:continue
                assert len(ops)==2,(name,i,ops)
                origin=(sum((row['position_m'][j]-p['origin_m'][j])*p['u'][j] for j in range(3)),
                        -sum((row['position_m'][j]-p['origin_m'][j])*p['v'][j] for j in range(3)))
                assert hypot(ops[0][1]-origin[0],ops[0][2]-origin[1])<1e-10
                actual=(ops[1][1]-ops[0][1],ops[1][2]-ops[0][2])
                assert sum(a*b for a,b in zip(actual,expected))/(hypot(*actual)*hypot(*expected))>1-1e-10
                directions+=1
        # Distinct marker roles must be painted, not only stored in metadata.
        labels=page.evaluate('''()=>{
          const labels=[],f=ctx.fillText;
          ctx.fillText=function(t,...args){labels.push(String(t));return f.call(this,t,...args)};
          try{drawScene()}finally{ctx.fillText=f}
          return labels;
        }''')
        assert any(f'q{moving} (moving)' in label for label in labels)
        assert all(any(f'q{i} (held)' in label for label in labels) for i in fixed)
        before=page.locator('#scene').evaluate('(c)=>c.toDataURL()')
        page.locator('#trajectory').uncheck()
        assert before!=page.locator('#scene').evaluate('(c)=>c.toDataURL()')
        page.locator('#trajectory').check()
        page.locator('#motion-reset').click()
        assert page.locator('#frame').input_value()=='0'
        page.locator('#motion-speed').select_option('0.5')
        page.locator('#motion-play').click();page.wait_for_timeout(120)
        page.locator('#motion-play').click()
        assert not page.evaluate('window.conductorViewer.playing')
        index=page.locator('#frame').input_value();page.wait_for_timeout(100)
        assert page.locator('#frame').input_value()==index
        page.locator('#motion-speed').select_option('4')
        page.locator('#motion-play').click()
        page.wait_for_function("!window.conductorViewer.playing",timeout=10000)
        assert page.locator('#frame').input_value()==str(len(indices)-1)
        assert 'Termination: '+motion['termination'] in page.locator('#motion-event').inner_text()
        page.locator('#frame').focus();page.keyboard.press('ArrowLeft')
        assert page.locator('#frame').input_value()==str(len(indices)-2)
        assert json.dumps(page.evaluate('data'),sort_keys=True)==original
        if args.screenshot:
            args.screenshot.parent.mkdir(parents=True,exist_ok=True)
            page.screenshot(path=str(args.screenshot),full_page=True)
        page.set_viewport_size({'width':390,'height':780})
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1')
        assert page.locator('#motion-controls').is_visible()
        assert page.locator('#motion-energy-chart').is_visible()
        assert not errors,errors
        assert not requests,requests
        browser.close()
    print(json.dumps(dict(frames=len(indices),held_sources=len(fixed),full_samples=len(motion['samples']),
        vector_origin_direction_checks=directions,source_labels='passed',play_pause_reset_keyboard='passed',
        physical_data_unchanged=True,mobile_390='passed',javascript_errors=errors,external_requests=requests),indent=2))


if __name__=='__main__':main()
