"""Actual Chromium checks of saved-time events, not browser physics."""
import argparse
import json
from pathlib import Path
import shutil
from playwright.sync_api import sync_playwright, Error as PlaywrightError


def check(path,screenshot=None):
    errors=[];logs=[];requests=[]
    with sync_playwright() as p:
        try:browser=p.chromium.launch(headless=True)
        except PlaywrightError:
            binary=shutil.which('chromium') or shutil.which('chromium-browser')
            if not binary:raise
            browser=p.chromium.launch(headless=True,executable_path=binary,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1440,'height':1050})
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('console',lambda e:logs.append(e.text) if e.type=='error' else None)
        page.on('request',lambda r:requests.append(r.url))
        html=Path(path).read_text(encoding='utf-8')
        page.set_content(html,wait_until='load')
        page.wait_for_function('window.conductorEventViewer && window.conductorEventViewer.loaded')
        data=page.evaluate('data');snapshot=json.dumps(data,sort_keys=True)
        states=data['display_states'];last=len(states)-1
        assert data['schema']=='conductor_sim.event_run.v1'
        assert data['study']['model']['revision_id']==data['study']['model_revision_id']
        assert page.locator('#events tbody tr').count()==max(1,len(data['events']))
        assert page.locator('#sources').inner_text()
        assert 'not total rest' in page.locator('#events').inner_text().lower() or not data['events'] or any(e['full_stop_within_tolerance'] for e in data['events'])
        assert data['trajectory']['termination'] in page.locator('#termination').inner_text()
        for i in sorted(set((0,last//2,last))):
            page.locator('#frame').fill(str(i));page.locator('#frame').dispatch_event('input')
            assert page.evaluate('window.conductorEventViewer.frameIndex')==i
            assert page.evaluate('window.conductorEventViewer.physicalTime')==states[i]['time_s']
            assert 'K+U:' in page.locator('#readout').inner_text()
        # Current force is nonzero in the preselected cases; actual painted
        # arrows must change on toggling, not merely checkbox metadata.
        page.locator('#frame').fill('0');page.locator('#frame').dispatch_event('input')
        image=page.locator('#scene').evaluate('c=>c.toDataURL()')
        page.locator('#forces').uncheck()
        assert page.locator('#scene').evaluate('c=>c.toDataURL()')!=image
        page.locator('#forces').check()
        page.locator('#frame').fill(str(last));page.locator('#frame').dispatch_event('input')
        image=page.locator('#scene').evaluate('c=>c.toDataURL()')
        page.locator('#path').uncheck()
        assert page.locator('#scene').evaluate('c=>c.toDataURL()')!=image
        page.locator('#path').check()
        if screenshot:
            Path(screenshot).parent.mkdir(parents=True,exist_ok=True)
            page.screenshot(path=str(screenshot),full_page=True)
        page.locator('#reset').click();assert page.locator('#frame').input_value()=='0'
        page.locator('#speed').select_option('4');page.locator('#play').click()
        page.wait_for_function('window.conductorEventViewer.frameIndex>=1',timeout=6000)
        if page.evaluate('window.conductorEventViewer.playing'):page.locator('#play').click()
        i=page.locator('#frame').input_value();page.wait_for_timeout(60)
        assert page.locator('#frame').input_value()==i
        page.locator('#reset').click();assert page.locator('#frame').input_value()=='0'
        assert json.dumps(page.evaluate('data'),sort_keys=True)==snapshot
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1')
        assert not errors and not logs and not requests,(errors,logs,requests)
        browser.close()
    return dict(case=data['study']['model']['scenario_id'],browser='Chromium',events=len(data['events']),
        termination=data['trajectory']['termination'],saved_states=len(states),
        play_scrub_pause_reset='passed',force_and_path_pixels='passed',mobile_width=390,
        javascript_errors=errors,console_errors=logs,external_requests=requests)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('html',type=Path);ap.add_argument('--screenshot',type=Path)
    a=ap.parse_args();print(json.dumps(check(a.html,a.screenshot),indent=2))
