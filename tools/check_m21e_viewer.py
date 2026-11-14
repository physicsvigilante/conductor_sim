"""Real Chromium acceptance of composed forces on the reused motion viewer."""
from __future__ import annotations
import argparse
import json
from math import hypot, isclose
from pathlib import Path
import shutil
from playwright.sync_api import sync_playwright, Error as PlaywrightError


def _panel(page):
    page.wait_for_function("window.conductorViewer && window.conductorViewer.forceComponents")
    data=page.evaluate('data');m=data['prescribed_motion']
    assert m['schema']=='conductor_sim.test_charge_viewer.v2'
    assert m['gravity']['kind']=='UniformGravity' and m['termination']=='duration'
    assert page.locator('#gravity-force-summary').is_visible()
    assert 'gravity mg' in page.locator('#gravity-force-summary').inner_text()
    assert 'F_total = ' in page.locator('#motion-summary').inner_text()
    assert 'U_e=qV' in page.locator('#motion-summary').inner_text()
    assert 'U_g' in page.locator('#motion-summary').inner_text()
    assert 'electric U_e' in page.locator('#motion-energy-heading').inner_text()
    assert 'gravitational U_g' in page.locator('#motion-energy-heading').inner_text()
    assert page.locator('#conductor-panel').is_hidden()
    assert page.locator('#scene-world-axes').is_visible()
    # Observe the actual existing-canvas arrow calls, not just saved metadata.
    arrows=page.evaluate('''() => {
      const out=[],old=ctx.stroke,pl=frame.plane;
      ctx.stroke=function(){out.push({color:this.strokeStyle,width:this.lineWidth});return old.call(this)};
      const project=p=>[pl.u,pl.v, [0,0,0]].map(a=>a.reduce((s,v,i)=>s+v*(p[i]-pl.origin_m[i]),0));
      try{forceArrow(pl,x=>x,y=>y,project)}finally{ctx.stroke=old;drawScene()}
      return out;
    }''')
    first=m['samples'][0]
    expected=[]
    for key,color in [('electric_force_n','#ca6a35'),('gravitational_force_n','#1766a2'),('force_n','#7225a4')]:
        if hypot(*first[key])>0:expected.append(color)
    assert [a['color'] for a in arrows]==expected
    image=page.evaluate("document.getElementById('scene').toDataURL()")
    page.locator('#force-arrow').uncheck()
    assert image!=page.evaluate("document.getElementById('scene').toDataURL()")
    page.locator('#force-arrow').check()
    assert image==page.evaluate("document.getElementById('scene').toDataURL()")
    return data,dict(actual_force_arrows=arrows,source_ids=[a['id'] for a in m['source_annotations']],
                     physical_samples=len(m['samples']),schema=m['schema'])


def check(path, *, case, screenshot=None):
    path=Path(path);errors=[];requests=[];console=[]
    with sync_playwright() as p:
        try:browser=p.chromium.launch(headless=True)
        except PlaywrightError:
            executable=shutil.which('chromium') or shutil.which('chromium-browser')
            if not executable:raise
            browser=p.chromium.launch(headless=True,executable_path=executable,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1400,'height':1000})
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('request',lambda r:requests.append(r.url))
        page.on('console',lambda m:console.append(m.text) if m.type=='error' else None)
        page.set_content(path.read_text(encoding='utf-8'),wait_until='load')
        panels=[]
        if case=='mass-revision':
            page.wait_for_function('window.conductorComparison && window.conductorComparison.loaded')
            children=page.frames[1:];assert len(children)==2
            originals=[]
            for child in children:
                data,record=_panel(child);originals.append(data);panels.append(record)
            manifest=page.evaluate('comparison')
            assert manifest['potential_energy_label']=='U_e + U_g'
            assert manifest['revisions']==[{'component_id':'test-particle','property':'mass_kg','baseline':2e-6,'variation':4e-6}]
            for i in (0,6,12):
                page.locator('#compare-time').fill(str(i))
                assert [child.locator('#frame').input_value() for child in children]==[str(i),str(i)]
                assert 'U_e + U_g' in page.locator('#readout-0').inner_text()
                assert 'U_e + U_g' in page.locator('#readout-1').inner_text()
            page.locator('#compare-reset').click();page.locator('#compare-speed').select_option('4')
            page.locator('#compare-play').click();page.wait_for_timeout(310)
            assert page.evaluate('window.conductorComparison.frameIndex')>0
            page.locator('#compare-play').click();page.locator('#compare-reset').click()
            assert [c.evaluate('data') for c in children]==originals
        else:
            original,record=_panel(page);panels.append(record)
            m=original['prescribed_motion'];idx=m['frame_sample_indices']
            if case=='jee2012-proton':
                assert 'screen right = world +x; screen up = world +y' in page.locator('#scene-world-axes').inner_text()
                row=m['samples'][-1]
                assert isclose(row['position_m'][0],.5*10*.024**2,rel_tol=1e-12)
                assert abs(row['position_m'][0])<.005
                assert '1 cm' in page.locator('#notes').inner_text()
            for i in (0,6,12):
                page.locator('#frame').fill(str(i))
                row=m['samples'][idx[i]]
                shown=page.locator('#motion-summary').inner_text()
                assert 'U_e=qV' in shown and 'U_g' in shown
                expected=page.evaluate("r=>r.map(x=>fmt(x)).join(', ')",row['gravitational_force_n'])
                assert expected in page.locator('#gravity-force-summary').inner_text()
            page.locator('#motion-reset').click();page.locator('#motion-speed').select_option('4')
            page.locator('#motion-play').click();page.wait_for_timeout(310)
            assert int(page.locator('#frame').input_value())>0
            page.locator('#motion-play').click()
            assert page.evaluate('window.conductorViewer.playing') is False
            page.locator('#motion-reset').click()
            assert page.evaluate('data')==original
        if screenshot:
            Path(screenshot).parent.mkdir(parents=True,exist_ok=True)
            page.screenshot(path=str(screenshot),full_page=True)
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
        assert not errors,errors
        assert not requests,requests
        assert not console,console
        result=dict(case=case,browser=browser.version,panels=panels,play_pause_scrub_reset='passed',
                    immutable_saved_data=True,mobile_width=390,javascript_errors=errors,
                    external_requests=requests,console_errors=console)
        browser.close();return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('html',type=Path)
    ap.add_argument('--case',required=True);ap.add_argument('--screenshot',type=Path)
    a=ap.parse_args();print(json.dumps(check(a.html,case=a.case,screenshot=a.screenshot),indent=2))

if __name__=='__main__':main()
