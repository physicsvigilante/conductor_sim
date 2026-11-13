"""Headless Chromium acceptance for the offline M13b position explorer."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('html',type=Path)
    parser.add_argument('--screenshot',type=Path)
    args=parser.parse_args()
    errors=[]
    requests=[]
    with sync_playwright() as p:
        chromium=os.environ.get('CHROMIUM_PATH')
        if chromium is None and Path('/usr/bin/chromium').exists():
            chromium='/usr/bin/chromium'
        kwargs={'headless':True}
        if chromium:
            kwargs['executable_path']=chromium
        browser=p.chromium.launch(**kwargs)
        page=browser.new_page(viewport={'width':1280,'height':900},device_scale_factor=1)
        page.on('pageerror',lambda error: errors.append(str(error)))
        page.route('https://**',lambda route: (requests.append(route.request.url),route.abort()))
        page.route('http://**',lambda route: (requests.append(route.request.url),route.abort()))
        page.set_content(args.html.read_text(encoding='utf-8'),wait_until='load')
        assert page.locator('#force-panel').is_visible()
        assert page.locator('#force-toggle').is_visible()
        assert page.locator('#position-label').inner_text()=='Charge position'
        metadata=page.evaluate('data.position_sweep')
        assert metadata['schema']=='conductor_sim.position_sweep.v1'
        assert len(metadata['samples'])>=3
        assert page.locator('#force-summary').inner_text()
        assert page.locator('#work-summary').inner_text().find('0 nJ') >= 0
        assert 'Mechanical interaction' in page.locator('#force-panel').inner_text()
        count=page.evaluate('window.conductorViewer.heatmapBuilds')
        initial=page.locator('#scene').evaluate('(e)=>e.toDataURL()')
        page.locator('#force-arrow').uncheck()
        without=page.locator('#scene').evaluate('(e)=>e.toDataURL()')
        assert initial != without, 'force arrow was not visibly drawn'
        assert page.evaluate('window.conductorViewer.heatmapBuilds')==count
        page.locator('#force-arrow').check()
        assert page.locator('#scene').evaluate('(e)=>e.toDataURL()')==initial
        # Force and energy charts must use separate scales and move markers.
        chart0=page.locator('#force-chart').evaluate('(e)=>e.toDataURL()')
        energy0=page.locator('#energy-chart').evaluate('(e)=>e.toDataURL()')
        summary0=page.locator('#force-summary').inner_text()
        slider=page.locator('#frame')
        slider.fill(str(len(metadata['samples'])-1))
        slider.dispatch_event('input')
        assert page.locator('#force-summary').inner_text() != summary0
        assert page.locator('#work-summary').inner_text().find('External quasistatic work')>=0
        assert page.locator('#force-chart').evaluate('(e)=>e.toDataURL()')!=chart0
        assert page.locator('#energy-chart').evaluate('(e)=>e.toDataURL()')!=energy0
        # Check that slider and graphs agree with embedded *numerical* data.
        state=page.evaluate("""()=>({
            frameIndex:Number(document.getElementById('frame').value),
            f:frame.force_energy.force_parallel_n,
            U:frame.force_energy.energy_j,
            s:frame.force_energy.coordinate_m,
            position:frame.force_energy.position_m,
            work:frame.force_energy.external_quasistatic_work_j})""")
        last=metadata['samples'][-1]
        assert state['frameIndex']==len(metadata['samples'])-1
        assert state['f']==last['force_parallel_n'] and state['U']==last['energy_j']
        assert state['s']==last['coordinate_m'] and state['position']==last['position_m']
        assert state['work']==last['external_quasistatic_work_j']
        # Persist frame selection when changing field-line budget. Work and U
        # cannot change because tracing is visualization-only.
        work=page.locator('#work-summary').inner_text()
        density=page.locator('#line-density')
        density.fill('5')
        density.dispatch_event('input')
        assert page.locator('#work-summary').inner_text()==work
        assert page.locator('#frame').input_value()==str(len(metadata['samples'])-1)
        slider.focus()
        slider.press('ArrowLeft')
        assert page.locator('#frame').input_value()==str(len(metadata['samples'])-2)
        # Screenshot at midframe; fine mesh, source and field traces retained.
        slider.fill(str(len(metadata['samples'])//2))
        slider.dispatch_event('input')
        if args.screenshot:
            args.screenshot.parent.mkdir(parents=True,exist_ok=True)
            page.screenshot(path=str(args.screenshot),full_page=True)
        page.set_viewport_size({'width':390,'height':780})
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth+1')
        assert page.locator('#force-chart').is_visible()
        assert page.locator('#energy-chart').is_visible()
        assert not errors,errors
        assert not requests,requests
        browser.close()
    print(json.dumps({'frames':len(metadata['samples']),
                      'physics_graphs':'passed','force_arrow':'passed',
                      'position_and_work_sync':'passed','legacy_controls':'passed',
                      'mobile_390':'passed','javascript_errors':errors,
                      'external_requests':requests},indent=2))


if __name__=='__main__':
    main()
