"""Actual offline Chromium check; static orientation controls, no time playback."""
import argparse
import json
from math import isclose
from pathlib import Path
import shutil
from playwright.sync_api import sync_playwright, Error


def check(path, screenshot=None):
    errors=[];requests=[]
    with sync_playwright() as p:
        try:browser=p.chromium.launch(headless=True)
        except Error:
            binary=shutil.which('chromium') or shutil.which('chromium-browser')
            if binary is None:raise
            browser=p.chromium.launch(headless=True,executable_path=binary,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1400,'height':1000})
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('console',lambda e:errors.append(e.text) if e.type=='error' else None)
        page.on('request',lambda e:requests.append(e.url))
        page.set_content(Path(path).read_text(encoding='utf-8'),wait_until='load')
        page.wait_for_function("window.dipoleExplorer && window.dipoleExplorer.mode==='static-orientation'")
        record=page.evaluate("JSON.parse(document.getElementById('dipole-data').textContent)")
        assert len(record['rows'])==73
        assert 'not time playback' in page.locator('main').inner_text()
        assert 'World x–y projection' in page.locator('main').inner_text()
        assert page.locator('#scene circle').count()>=2
        assert page.locator('#energy circle').count()==73
        original=page.locator('#scene').inner_html()
        for index in (8,18,23,36,72):
            page.locator('#angle').fill(str(index));page.locator('#angle').dispatch_event('input')
            assert page.evaluate('window.dipoleExplorer.index')==index
            obs=record['rows'][index]['observation']
            assert obs is not None
            text=page.locator('#values').inner_text()
            assert 'Torque about marked origin' in text and 'Electric endpoint work' in text
            shown=float(text.split('U_external (J): ')[1].split('\n')[0])
            assert isclose(shown,obs['interaction_energy_j'],rel_tol=1e-5,abs_tol=1e-20)
            assert page.locator(f'#energy circle[data-sample="{index}"]').get_attribute('r')=='5'
            assert page.locator('#scene line[data-force="positive"]').count()==1
        page.locator('#angle').fill('8');page.locator('#angle').dispatch_event('input')
        assert page.locator('#scene').inner_html()!=original
        assert page.locator('#scene path[data-torque="z"]').count()==1
        if screenshot:
            Path(screenshot).parent.mkdir(parents=True,exist_ok=True)
            page.screenshot(path=str(screenshot),full_page=True)
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1')
        assert page.locator('#angle').is_visible()
        assert not errors and not requests,(errors,requests)
        browser.close()
    return dict(model_id=record['model']['model_id'],static_samples=73,
        real_browser='Chromium',orientation_selector='passed',force_and_torque_drawing='passed',
        numerical_energy_readout='passed',mobile_width=390,javascript_errors=errors,external_requests=requests)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('html',type=Path);p.add_argument('--screenshot',type=Path)
    args=p.parse_args();print(json.dumps(check(args.html,args.screenshot),indent=2))
