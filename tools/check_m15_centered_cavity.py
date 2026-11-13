"""Chromium acceptance: real zero-force centre versus nonzero off-centre arrow.

A force-arrow toggle must not invent an arrow at a true equilibrium. Once the
charge moves off centre, the same viewer draws a force arrow without changing
stored mechanical energy or requiring browser-time electrostatics.
"""
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
    errors=[];requests=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,**(
            {'executable_path':os.environ['CHROMIUM_PATH']}
            if 'CHROMIUM_PATH' in os.environ else
            {'executable_path':'/usr/bin/chromium'} if Path('/usr/bin/chromium').exists() else {}))
        page=browser.new_page(viewport={'width':1200,'height':850})
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.route('http://**',lambda r:(requests.append(r.request.url),r.abort()))
        page.route('https://**',lambda r:(requests.append(r.request.url),r.abort()))
        page.set_content(args.html.read_text(encoding='utf-8'),wait_until='load')
        data=page.evaluate('data.position_sweep')
        assert data['samples'][0]['position_m']==[0,0,0]
        assert data['samples'][0]['force_n']==[0,0,0]
        assert len(data['samples'])>=5
        arrow=page.locator('#force-arrow')
        scene=page.locator('#scene')
        original=scene.evaluate('(x)=>x.toDataURL()')
        arrow.uncheck()
        assert scene.evaluate('(x)=>x.toDataURL()')==original,'zero force must not have an arrow'
        arrow.check()
        assert scene.evaluate('(x)=>x.toDataURL()')==original
        slider=page.locator('#frame')
        slider.fill('1');slider.dispatch_event('input')
        assert data['samples'][1]['force_n']!=[0,0,0]
        force_on=scene.evaluate('(x)=>x.toDataURL()')
        arrow.uncheck()
        assert scene.evaluate('(x)=>x.toDataURL()')!=force_on,'off-centre force arrow must be drawn'
        arrow.check()
        assert scene.evaluate('(x)=>x.toDataURL()')==force_on
        assert page.locator('#energy-chart').is_visible()
        assert page.locator('#force-chart').is_visible()
        assert page.locator('#work-summary').inner_text()
        slider.fill(str(len(data['samples'])-1));slider.dispatch_event('input')
        assert page.evaluate('frame.force_energy.energy_j')==data['samples'][-1]['energy_j']
        assert page.evaluate('frame.force_energy.force_parallel_n')==data['samples'][-1]['force_parallel_n']
        if args.screenshot:
            args.screenshot.parent.mkdir(parents=True,exist_ok=True)
            page.screenshot(path=str(args.screenshot),full_page=True)
        page.set_viewport_size({'width':390,'height':780})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth+1')
        assert not errors,errors
        assert not requests,requests
        browser.close()
    print(json.dumps({'source_at_center':'passed','force_arrow_zero':'passed',
                      'force_arrow_offcenter':'passed','energy_and_force_sync':'passed',
                      'mobile':'passed','javascript_errors':errors,'external_requests':requests},indent=2))


if __name__=='__main__':
    main()
