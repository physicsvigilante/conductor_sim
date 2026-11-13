"""Chromium M17b acceptance for genuine curved, offline time playback.

Run with Playwright installed:
  python tools/check_m17b_viewer.py output/m17b-plane.html
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('html',type=Path)
    parser.add_argument('--screenshot',type=Path)
    args=parser.parse_args()
    assert args.html.is_file(),args.html
    html=args.html.read_text(encoding='utf-8')
    requests,errors=[],[]
    with sync_playwright() as playwright:
        try:
            browser=playwright.chromium.launch(headless=True)
        except PlaywrightError:
            executable=shutil.which('chromium') or shutil.which('chromium-browser')
            if executable is None:
                raise
            browser=playwright.chromium.launch(headless=True,
                executable_path=executable,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1240,'height':920},device_scale_factor=1)
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.on('request',lambda request:requests.append(request.url))
        page.set_content(html,wait_until='load')
        data=page.evaluate('data')
        assert data['schema']=='conductor_sim.visualization.v1'
        assert 'position_sweep' not in data and 'motion_1d' not in data
        motion=data['motion_2d']
        assert motion['schema']=='conductor_sim.motion_viewer_2d.v1'
        indices=motion['frame_sample_indices']
        assert len(indices)==len(data['frames'])==page.evaluate('window.conductorViewer.frameCount')
        assert page.evaluate('window.conductorViewer.mode')=='elapsed-time-2d'
        assert page.evaluate('window.conductorViewer.motionSampleCount')==len(motion['samples'])
        assert motion['graph_sample_count']==len(motion['samples'])
        assert indices[0]==0 and indices[-1]==len(motion['samples'])-1
        assert len(motion['samples'])>=len(indices)>=2
        assert page.locator('#motion-panel').is_visible()
        assert page.locator('#motion-controls').is_visible()
        assert page.locator('#trajectory-toggle').is_visible()
        assert page.locator('#motion-vector-legend').is_visible()
        assert page.locator('#force-panel').is_hidden()
        assert page.locator('#static-playback-note').is_hidden()
        assert page.locator('#position-label').inner_text()=='Elapsed time'
        assert page.locator('#motion-summary').inner_text().startswith('t = 0 ')
        assert 'energy drift' in page.locator('#motion-summary').inner_text()
        assert 'components' in page.locator('#motion-position-chart').locator('..').locator('h3').inner_text()
        assert 'components' in page.locator('#motion-velocity-chart').locator('..').locator('h3').inner_text()
        assert page.locator('#motion-energy-chart').is_visible()
        assert all(frame['charges'][0]['position_m']==motion['samples'][idx]['position_m']
                   and frame['motion_sample_index']==idx
                   for idx,frame in zip(indices,data['frames']))
        assert all(frame['plane']==data['frames'][0]['plane'] for frame in data['frames'])
        assert all(a['time_s']<b['time_s']
                   for a,b in zip(motion['samples'],motion['samples'][1:]))
        assert not requests and not errors,(requests,errors)

        initial=page.locator('#scene').evaluate('(canvas)=>canvas.toDataURL()')
        last=str(len(indices)-1)
        page.locator('#frame').fill(last)
        page.locator('#frame').dispatch_event('input')
        assert page.locator('#motion-clock').inner_text().startswith('t = ')
        assert '[' in page.locator('#motion-summary').inner_text()
        assert 'energy drift' in page.locator('#motion-summary').inner_text()
        assert initial!=page.locator('#scene').evaluate('(canvas)=>canvas.toDataURL()')

        page.locator('#frame').focus()
        page.keyboard.press('ArrowLeft')
        assert page.locator('#frame').input_value()==str(len(indices)-2)
        scene=page.locator('#scene')
        path_before=scene.evaluate('(c)=>c.toDataURL()')
        page.locator('#trajectory').uncheck()
        path_after=scene.evaluate('(c)=>c.toDataURL()')
        assert path_before!=path_after, 'real curved path overlay must be painted'
        page.locator('#trajectory').check()

        def image():
            return scene.evaluate('(c)=>c.toDataURL()')
        before=image()
        page.locator('#velocity-arrow').uncheck()
        assert before!=image(), 'vector velocity arrow should change scene'
        page.locator('#velocity-arrow').check()
        before=image()
        page.locator('#force-arrow').uncheck()
        # A source at a center may have zero force, but these acceptance
        # cases use an off-center particle with nonzero conductor force.
        assert before!=image(), 'vector force arrow should change scene'
        page.locator('#force-arrow').check()

        page.locator('#motion-reset').click()
        assert page.locator('#frame').input_value()=='0'
        assert page.locator('#motion-clock').inner_text().startswith('t = 0 ')
        page.locator('#motion-speed').select_option('4')
        page.locator('#motion-play').click()
        assert page.evaluate('window.conductorViewer.playing') is True
        assert page.locator('#motion-play').inner_text()=='Pause'
        page.wait_for_timeout(1450)
        assert page.locator('#frame').input_value()==last
        assert page.evaluate('window.conductorViewer.playing') is False
        assert page.locator('#motion-play').get_attribute('aria-pressed')=='false'
        page.locator('#motion-reset').click()
        page.locator('#frame').fill(str(len(indices)//2))
        page.locator('#frame').dispatch_event('input')
        if args.screenshot:
            args.screenshot.parent.mkdir(parents=True,exist_ok=True)
            page.screenshot(path=str(args.screenshot),full_page=True)
        page.set_viewport_size({'width':390,'height':780})
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth + 1')
        assert page.locator('#motion-play').is_visible()
        assert page.locator('#trajectory-toggle').is_visible()
        assert page.locator('#motion-energy-chart').is_visible()
        assert not errors,errors
        assert not requests,requests
        browser.close()
    print(json.dumps({'frames':len(indices),'full_samples':len(motion['samples']),
                      'mode':'M17a planar physical time','curved_path':'passed',
                      'velocity_and_force_vectors':'passed','charts':'passed',
                      'playback':'passed','mobile_390':'passed','offline':'passed',
                      'javascript_errors':errors,'external_requests':requests},indent=2))


if __name__=='__main__':
    main()
