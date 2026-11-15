"""Post-M24b: real Chromium regression of uncertain terminal playback.

The new two public-API trial scenes use the retained driven-browser checker.
No new solver, physics projection, or requirement for nonzero net force is used.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil

from playwright.sync_api import sync_playwright, Error as PlaywrightError
from conductor_sim import PrescribedField, PointCharge, TestChargeMotionSpec
from conductor_sim.scenarios import PrescribedScenario
from conductor_sim.driven_motion import (DrivenModel, DrivenStudy, LinearDrag,
    UniformFieldSchedule, FieldPhase, EventPolicy, PlaneEvent, investigate_driven_motion)
from conductor_sim.driven_viewer import render_driven_run


def check(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    model=DrivenModel(PrescribedScenario('review:coincident',TestChargeMotionSpec(
        PrescribedField(()),PointCharge(1e-9,(0.,0.,0.)),1e-9,(1.,0.,0.),
        duration_s=.5,sample_interval_s=.1,plane_normal=(0.,1.,0.)),(),
        'Teacher numerical-resolution demonstration, not a finite plate model.'),
        LinearDrag(0.),UniformFieldSchedule((FieldPhase(0.,(0.,0.,0.)),)),
        'review:driven','Two detector planes separated by 1e-10 m; temporal order is unresolved. '
        'No post-stop motion is displayed; detector spacing is below visual pixel resolution.')
    queries=(PlaneEvent('first',(.11,0.,0.),(1.,0.,0.),'First detector',1,True),
             PlaneEvent('second',(.11+1e-10,0.,0.),(1.,0.,0.),'Second detector',1,True))
    run=investigate_driven_motion(DrivenStudy(model,queries,EventPolicy(.025,1e-8,1e-10,1e-9)))
    html=render_driven_run(run,title='Corrected near-coincident terminal detectors')
    (output/'near-coincident.html').write_text(html,encoding='utf-8')
    (output/'near-coincident.json').write_text(json.dumps(run.as_dict(),indent=2)+'\n',encoding='utf-8')
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
        page.set_content(html,wait_until='load')
        page.wait_for_function('window.conductorDrivenViewer && window.conductorDrivenViewer.loaded')
        data=page.evaluate('data');end=run.samples[-1].time_s
        assert data['ordering']=='coincident-or-order-unresolved-within-time-resolution'
        assert any(e['sample']['time_s']>end for e in data['events'])
        assert max(s['time_s'] for s in data['display_states'])==end
        assert 'not post-stop motion' in page.locator('#terminal-candidates').inner_text()
        assert page.locator('#events tbody tr').count()==2
        last=len(data['display_states'])-1
        for i in range(last+1):
            page.locator('#frame').fill(str(i));page.locator('#frame').dispatch_event('input')
            assert page.evaluate('window.conductorDrivenViewer.physicalTime')<=end
        # With zero resultant force, test the real nonzero velocity/path instead.
        page.locator('#velocity').uncheck()
        image=page.locator('#scene').evaluate('c=>c.toDataURL()')
        page.locator('#path').uncheck()
        assert page.locator('#scene').evaluate('c=>c.toDataURL()')!=image
        page.locator('#path').check();page.locator('#velocity').check()
        page.screenshot(path=str(output/'near-coincident.png'),full_page=True)
        page.locator('#reset').click();page.locator('#speed').select_option('4');page.locator('#play').click()
        page.wait_for_function('!window.conductorDrivenViewer.playing && window.conductorDrivenViewer.frameIndex>0',timeout=6000)
        assert page.evaluate('window.conductorDrivenViewer.physicalTime')==end
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1')
        assert not errors and not logs and not requests
        browser.close()
    report=dict(actual_browser='Chromium',events=len(run.events),displayed_end_s=end,
        alternative_event_after_end_not_played=True,unresolved_order_visible=True,
        physical_time_playback='passed',javascript_errors=errors,console_errors=logs,external_requests=requests)
    (output/'near-coincident-browser.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path('output/post-m24b'))
    a=ap.parse_args();print(json.dumps(check(a.output),indent=2))
