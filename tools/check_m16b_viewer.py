"""Chromium M16b acceptance: real elapsed-time frames, charts, and offline controls.

Run with playwright installed: python tools/check_m16b_viewer.py output/motion.html
"""
from __future__ import annotations

import argparse
import json
import shutil
from playwright.sync_api import Error as PlaywrightError
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('html',type=Path)
    parser.add_argument('--screenshot',type=Path)
    args=parser.parse_args()
    assert args.html.is_file(), args.html
    html=args.html.read_text(encoding='utf-8')
    requests=[];errors=[]
    with sync_playwright() as playwright:
        try:
            browser=playwright.chromium.launch(headless=True)
        except PlaywrightError:
            executable=shutil.which('chromium') or shutil.which('chromium-browser')
            if executable is None:
                raise
            browser=playwright.chromium.launch(headless=True,executable_path=executable,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1240,'height':910},device_scale_factor=1)
        page.on('pageerror',lambda err:errors.append(str(err)))
        page.on('request',lambda request:requests.append(request.url))
        # Equivalent to opening a standalone offline HTML; no web server and
        # no browser network dependency (also works in restricted environments).
        page.set_content(html,wait_until='load')
        data=page.evaluate('data')
        assert data['schema']=='conductor_sim.visualization.v1'
        assert 'position_sweep' not in data
        motion=data['motion_1d']
        assert motion['schema']=='conductor_sim.motion_viewer.v1'
        indices=motion['frame_sample_indices']
        assert len(indices)==len(data['frames'])==page.evaluate('window.conductorViewer.frameCount')
        assert page.evaluate('window.conductorViewer.mode')=='elapsed-time'
        assert page.locator('#motion-panel').is_visible()
        assert page.locator('#motion-controls').is_visible()
        assert page.locator('#force-panel').is_hidden()
        assert page.locator('#static-playback-note').is_hidden()
        assert page.locator('#position-label').inner_text()=='Elapsed time'
        assert page.locator('#motion-play').inner_text()=='Play'
        assert page.locator('#motion-summary').inner_text().startswith('t = 0 ')
        assert 'energy drift' in page.locator('#motion-summary').inner_text()
        assert page.locator('#motion-position-chart').is_visible()
        assert page.locator('#motion-velocity-chart').is_visible()
        assert page.locator('#motion-energy-chart').is_visible()
        assert all(motion['samples'][i]['position_m']==frame['charges'][0]['position_m']
                   for i,frame in zip(indices,data['frames']))
        assert all(motion['samples'][i]['time_s']<motion['samples'][j]['time_s']
                   for i,j in zip(indices,indices[1:]))
        assert motion['samples'][-1]['time_s']==motion['samples'][indices[-1]]['time_s']
        assert page.locator('#motion-event').inner_text().startswith('Termination: ')
        assert not page.locator('#line-density').is_disabled() or not data['frames'][0]['field_lines']
        first=page.locator('#scene').evaluate('(c)=>c.toDataURL()')
        last=str(len(indices)-1)
        page.locator('#frame').fill(last)
        page.locator('#frame').dispatch_event('input')
        summary=page.locator('#motion-summary').inner_text()
        assert 'displacement' in summary and 'v =' in summary
        assert 'energy drift' in summary
        assert page.locator('#scene').evaluate('(c)=>c.toDataURL()')!=first or (
            motion['samples'][0]['position_m']==motion['samples'][-1]['position_m'])
        page.locator('#frame').focus()
        page.keyboard.press('ArrowLeft')
        assert page.locator('#frame').input_value()==str(len(indices)-2)
        # Distinct drawn force/velocity glyphs do not mutate underlying physics.
        canvas_before=page.locator('#scene').evaluate('(c)=>c.toDataURL()')
        page.locator('#velocity-arrow').uncheck()
        canvas_after=page.locator('#scene').evaluate('(c)=>c.toDataURL()')
        row=motion['samples'][indices[-2]]
        if abs(row['velocity_m_s'])>1e-13:
            assert canvas_before!=canvas_after
        page.locator('#velocity-arrow').check()
        page.locator('#heat').uncheck()
        assert page.locator('#scene').evaluate('(c)=>c.toDataURL()')!=canvas_before
        page.locator('#heat').check()
        # Actual wall-clock UI playback is intentionally *scaled*; physical
        # time labels are read from the frozen numeric trajectory.
        page.locator('#motion-reset').click()
        assert page.locator('#frame').input_value()=='0'
        page.locator('#motion-speed').select_option('4')
        page.locator('#motion-play').click()
        assert page.evaluate('window.conductorViewer.playing') is True
        assert page.locator('#motion-play').inner_text()=='Pause'
        page.wait_for_timeout(1200)
        assert page.locator('#frame').input_value()==last
        assert page.evaluate('window.conductorViewer.playing') is False
        assert page.locator('#motion-play').get_attribute('aria-pressed')=='false'
        page.locator('#motion-reset').click()
        assert page.locator('#frame').input_value()=='0'
        assert page.locator('#motion-clock').inner_text().startswith('t = 0 ')
        page.locator('#frame').fill(str(len(indices)//2))
        page.locator('#frame').dispatch_event('input')
        if args.screenshot:
            args.screenshot.parent.mkdir(parents=True,exist_ok=True)
            page.screenshot(path=str(args.screenshot),full_page=True)
        page.set_viewport_size({'width':390,'height':780})
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 1')
        assert page.locator('#motion-play').is_visible()
        assert page.locator('#motion-energy-chart').is_visible()
        assert not errors,errors
        assert not requests,requests
        browser.close()
    print(json.dumps({'frames':len(indices),'full_samples':len(motion['samples']),
                      'mode':'real elapsed-time', 'charts':'passed', 'controls':'passed',
                      'velocity_arrow':'passed','metadata_match':'passed','mobile_390':'passed',
                      'offline':'passed','javascript_errors':errors,'external_requests':requests},indent=2))


if __name__=='__main__':
    main()
