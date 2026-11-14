"""Actual offline Chromium acceptance for M19f two-scenario comparison.

Tests independent viewer frames and a *shared actual-time cursor*, not an ODE.
Both viewers must preserve source names, the same physical plane/scale, and
all output graphs. No HTTP request or unhandled script error is tolerated.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from playwright.sync_api import sync_playwright


def check(path, screenshot=None):
    html=Path(path).read_text(encoding='utf8')
    errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'],
            **({'executable_path':'/usr/bin/chromium'} if shutil.which('chromium') else {}))
        try:
            page=browser.new_page(viewport={'width':1460,'height':920},device_scale_factor=1)
            page.on('pageerror',lambda e:errors.append('js: '+str(e)))
            page.on('request',lambda r:errors.append('network: '+r.url) if r.url.startswith(('http:','https:')) else None)
            # This also works in locked-down environments that forbid file://.
            page.set_content(html,wait_until='load')
            page.wait_for_function('window.scenarioComparison && '+
                'document.querySelector("#scenario-left").contentWindow.conductorViewer && '+
                'document.querySelector("#scenario-right").contentWindow.conductorViewer',timeout=30000)
            assert page.evaluate('window.scenarioComparison.mode')=='two-prescribed-scenarios'
            n=page.evaluate('window.scenarioComparison.frameCount')
            assert n==12
            assert page.locator('#changes').inner_text().count('field:external.field_v_m')==1
            assert 'ring:A, ring:B' in page.locator('#unchanged').inner_text()
            assert 'unchanged source physical values' in page.locator('#unchanged').inner_text().lower()
            assert 'gauge-dependent' in page.locator('main').inner_text().lower()
            for pane in ('scenario-left','scenario-right'):
                f=page.frame_locator('#'+pane)
                assert f.locator('#motion-panel').is_visible()
                assert f.locator('#prescribed-panel').is_visible()
                assert f.locator('#scene').get_attribute('width')=='720'
                assert 'Ring A' in f.locator('#scene').get_attribute('aria-label') or 'Fixed electrostatic' in f.locator('#scene').get_attribute('aria-label')
                assert 'screen right = world +x' in f.locator('#scene-world-axes').inner_text()
                assert f.locator('#frame').input_value()=='0'
                assert f.locator('#motion-clock').inner_text().startswith('t = 0 ')
            for i in (1,5,9,11):
                page.locator('#compare-frame').fill(str(i))
                values=[]
                for pane in ('scenario-left','scenario-right'):
                    f=page.frame_locator('#'+pane)
                    assert f.locator('#frame').input_value()==str(i)
                    values.append(f.locator('#motion-clock').inner_text())
                assert values[0]==values[1]
                assert float(page.locator('#delta-pos').inner_text().split()[0])>=0
            page.locator('#compare-reset').click()
            assert page.evaluate('window.scenarioComparison.index')==0
            page.locator('#compare-play').click()
            page.wait_for_timeout(700)
            assert page.evaluate('window.scenarioComparison.index')>0
            assert page.evaluate('window.scenarioComparison.playing')
            page.locator('#compare-play').click()
            assert not page.evaluate('window.scenarioComparison.playing')
            page.locator('#compare-reset').click()
            if screenshot:
                page.screenshot(path=str(screenshot),full_page=True)
            # Actual responsive layout, not an HTML/CSS string comparison.
            page.set_viewport_size({'width':390,'height':844})
            page.wait_for_timeout(300)
            assert page.evaluate('document.documentElement.scrollWidth <= 390')
            panels=page.locator('.grid>.card')
            assert panels.count()==2
            box0,box1=panels.nth(0).bounding_box(),panels.nth(1).bounding_box()
            assert box1['y']>box0['y']+box0['height']-1
            page.locator('#compare-frame').fill('6')
            assert page.frame_locator('#scenario-left').locator('#frame').input_value()=='6'
            assert page.frame_locator('#scenario-right').locator('#frame').input_value()=='6'
            assert not errors,errors
            return {'schema':'conductor_sim.m19f_browser_acceptance.v1',
                    'path':str(path),'frame_count':n,
                    'synchronized_actual_time':True,'source_identifiers_present':True,
                    'matching_world_axes':True,'matching_metre_scale':True,
                    'both_independent_motion_graphs':True,'mobile_width_px':390,
                    'offline_no_network':True,'unhandled_javascript_errors':errors}
        finally:
            browser.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('html',type=Path)
    p.add_argument('--screenshot',type=Path)
    args=p.parse_args()
    result=check(args.html,args.screenshot)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
