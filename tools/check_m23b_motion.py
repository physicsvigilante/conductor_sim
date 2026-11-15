"""Real Chromium acceptance of M23b saved physical-time dipole motion."""
from __future__ import annotations
import argparse
import json
from math import isclose
from pathlib import Path
import shutil
from playwright.sync_api import sync_playwright, Error as PlaywrightError


def check(path, screenshot=None):
    path=Path(path);errors=[];console=[];requests=[]
    with sync_playwright() as p:
        try:
            browser=p.chromium.launch(headless=True)
        except PlaywrightError:
            exe=shutil.which('chromium') or shutil.which('chromium-browser')
            if exe is None:raise
            browser=p.chromium.launch(headless=True,executable_path=exe,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1380,'height':1000})
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('console',lambda m:console.append(m.text) if m.type=='error' else None)
        page.on('request',lambda r:requests.append(r.url))
        page.set_content(path.read_text(encoding='utf-8'),wait_until='load')
        page.wait_for_function("window.dipoleMotion && window.dipoleMotion.mode==='elapsed-time-fixed-axis'")
        data=page.evaluate("JSON.parse(document.getElementById('motion-data').textContent)")
        rows=data['samples'];original=json.dumps(data,sort_keys=True)
        assert len(rows)>2 and rows[0]['time_s']==0
        assert 'orthographic projection' in page.locator('main').inner_text()
        assert 'Teacher hypothetical' in page.locator('header').inner_text()
        assert 'ideal frictionless hinge' in page.locator('main').inner_text().lower()
        assert page.locator('#play').is_visible() and page.locator('#reset').is_visible()
        canvas=page.locator('#scene')
        painted=canvas.evaluate("""c=>{const a=c.getContext('2d').getImageData(0,0,c.width,c.height).data;
          let ink=0;for(let i=0;i<a.length;i+=4)if(a[i]+a[i+1]+a[i+2]<550)ink++;return ink;}""")
        assert painted>100
        assert page.locator('#energy').is_visible() and page.locator('#angleplot').is_visible()
        for i in (0,len(rows)//3,len(rows)-1):
            page.locator('#time').fill(str(i));page.locator('#time').dispatch_event('input')
            assert page.evaluate('window.dipoleMotion.index')==i
            assert page.evaluate('window.dipoleMotion.time_s')==rows[i]['time_s']
            numbers=page.locator('#numbers').inner_text()
            expected=page.evaluate('rows[window.dipoleMotion.index].kinetic_energy_j.toPrecision(6)')
            assert 'K_rotation (J): '+expected in numbers
            expected=page.evaluate('rows[window.dipoleMotion.index].angular_velocity_rad_s.toPrecision(6)')
            assert 'ω (rad/s): '+expected in numbers
            assert 'Physical t = ' in page.locator('#clock').inner_text()
        # Inspect actual apparatus forces using a nonzero-force orientation.
        page.locator('#time').fill(str(len(rows)//3));page.locator('#time').dispatch_event('input')
        before=canvas.evaluate('c=>c.toDataURL()')
        page.locator('#forces').uncheck()
        assert canvas.evaluate('c=>c.toDataURL()') != before
        page.locator('#forces').check()
        assert canvas.evaluate('c=>c.toDataURL()') == before
        # Actual play/pause for the very short guard run: deliberately slow it.
        page.locator('#reset').click()
        assert page.locator('#time').input_value()=='0'
        page.locator('#speed').select_option('0.25')
        page.locator('#play').click()
        page.wait_for_function("window.dipoleMotion.index>0",timeout=8000)
        if page.evaluate('window.dipoleMotion.playing'):
            page.locator('#play').click()
        i=page.evaluate('window.dipoleMotion.index')
        page.wait_for_timeout(100)
        assert page.evaluate('window.dipoleMotion.index')==i
        assert page.evaluate('window.dipoleMotion.playing') is False
        page.locator('#reset').click()
        assert page.evaluate('window.dipoleMotion.index')==0
        assert json.dumps(page.evaluate('data'),sort_keys=True)==original
        if data['termination']=='source-guard':
            assert 'incomplete' in page.locator('#termination').inner_text()
            assert rows[-1]['time_s']<data['spec']['duration_s']
        else:
            assert rows[-1]['time_s']==data['spec']['duration_s']
        page.locator('#time').fill(str(len(rows)//4));page.locator('#time').dispatch_event('input')
        if screenshot:
            screenshot=Path(screenshot);screenshot.parent.mkdir(parents=True,exist_ok=True)
            page.screenshot(path=str(screenshot),full_page=True)
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
        assert page.locator('#play').is_visible() and page.locator('#time').is_visible()
        assert not errors and not console and not requests,(errors,console,requests)
        browser.close()
    return dict(case=data['spec']['model']['model_id'],browser='Chromium',
        samples=len(rows),termination=data['termination'],physical_end_s=rows[-1]['time_s'],
        painted_scene_pixels=painted,controls='scrub/play/pause/reset/speed passed',
        model_unchanged=True,force_toggle='passed',mobile_px=390,
        javascript_errors=errors,console_errors=console,external_requests=requests)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('html',type=Path);ap.add_argument('--screenshot',type=Path)
    a=ap.parse_args()
    print(json.dumps(check(a.html,a.screenshot),indent=2))
if __name__=='__main__':main()
