"""Real offline Chromium: work ledger, actual schedule states, controls and drawing."""
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
            exe=shutil.which('chromium') or shutil.which('chromium-browser')
            if not exe:raise
            browser=p.chromium.launch(headless=True,executable_path=exe,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1440,'height':1050})
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('console',lambda e:logs.append(e.text) if e.type=='error' else None)
        page.on('request',lambda r:requests.append(r.url))
        page.set_content(Path(path).read_text(encoding='utf-8'),wait_until='load')
        page.wait_for_function('window.conductorDrivenViewer && window.conductorDrivenViewer.loaded')
        data=page.evaluate('data');before=json.dumps(data,sort_keys=True);rows=data['display_states'];last=len(rows)-1
        assert data['schema']=='conductor_sim.driven_run.v1'
        assert data['study']['model']['revision_id']==data['study']['model_revision_id']
        assert page.locator('#switches tbody tr').count()==max(1,len(data['switches']))
        assert page.locator('#events tbody tr').count()==max(1,len(data['events']))
        assert 'stationary medium' in page.locator('#sources').inner_text()
        assert data['termination'] in page.locator('#termination').inner_text()
        for i in sorted(set((0,last//2,last))):
            page.locator('#frame').fill(str(i));page.locator('#frame').dispatch_event('input')
            assert page.evaluate('window.conductorDrivenViewer.physicalTime')==rows[i]['time_s']
            assert 'Dissipated D:' in page.locator('#readout').inner_text()
            assert 'Work-balance residual:' in page.locator('#readout').inner_text()
            assert 'Numerical energy drift' not in page.locator('#readout').inner_text()
        for sw in data['switches']:
            assert sw['left']['position_m']==sw['right']['position_m']
            assert sw['left']['velocity_m_s']==sw['right']['velocity_m_s']
            i=next(j for j,r in enumerate(rows) if r['time_s']==sw['time_s'])
            page.locator('#frame').fill(str(i));page.locator('#frame').dispatch_event('input')
            assert 'phase '+str(sw['right']['phase_index']) in page.locator('#clock').inner_text()
        # Search actual saved states for a visible net force, not a false
        # assertion that a balanced initial state must draw a net-force arrow.
        for control in ('forces','components','path'):
            if control=='path':
                # Other arrows can physically occlude a short path. Isolate
                # this overlay rather than inventing a larger displacement.
                for other in ('forces','components','velocity'):page.locator('#'+other).uncheck()
            index=last if control=='path' else max(range(len(rows)),key=lambda j:sum(x*x for x in rows[j]['force_n']))
            page.locator('#frame').fill(str(index));page.locator('#frame').dispatch_event('input')
            image=page.locator('#scene').evaluate('c=>c.toDataURL()')
            page.locator('#'+control).uncheck()
            assert page.locator('#scene').evaluate('c=>c.toDataURL()')!=image,control
            page.locator('#'+control).check()
        for other in ('forces','components','velocity'):page.locator('#'+other).check()
        page.locator('#frame').fill(str(last));page.locator('#frame').dispatch_event('input')
        if screenshot:
            Path(screenshot).parent.mkdir(parents=True,exist_ok=True)
            page.screenshot(path=str(screenshot),full_page=True)
        page.locator('#reset').click();assert page.locator('#frame').input_value()=='0'
        page.locator('#speed').select_option('4');page.locator('#play').click()
        page.wait_for_function('window.conductorDrivenViewer.frameIndex>=1',timeout=6000)
        if page.evaluate('window.conductorDrivenViewer.playing'):page.locator('#play').click()
        i=page.locator('#frame').input_value();page.wait_for_timeout(70)
        assert page.locator('#frame').input_value()==i
        page.locator('#reset').click();assert page.locator('#frame').input_value()=='0'
        assert json.dumps(page.evaluate('data'),sort_keys=True)==before
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1')
        assert not errors and not logs and not requests,(errors,logs,requests)
        browser.close()
    return dict(model=data['study']['model']['model_id'],browser='Chromium',saved_states=len(rows),
        switches=len(data['switches']),events=len(data['events']),termination=data['termination'],
        real_force_component_path_pixels='passed',work_and_schedule_readouts='passed',
        play_pause_scrub_reset_speed='passed',mobile_px=390,
        javascript_errors=errors,console_errors=logs,external_requests=requests)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('html',type=Path);ap.add_argument('--screenshot',type=Path)
    a=ap.parse_args();print(json.dumps(check(a.html,a.screenshot),indent=2))
