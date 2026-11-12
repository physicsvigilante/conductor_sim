"""Optional browser acceptance check; requires Playwright and Chromium.

python tools/check_viewer.py output/conductors.html --screenshot output/view.png
"""
import argparse
import json
from pathlib import Path
import shutil
from playwright.sync_api import sync_playwright


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('html',type=Path)
    parser.add_argument('--screenshot',type=Path)
    args=parser.parse_args()
    with sync_playwright() as pw:
        exe=shutil.which('chromium') or shutil.which('google-chrome')
        browser=pw.chromium.launch(executable_path=exe,headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1360,'height':1120},device_scale_factor=1)
        errors=[];requests=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        def route(r):
            requests.append(r.request.url);r.abort()
        page.route('https://**',route);page.route('http://**',route)
        page.set_content(args.html.read_text(encoding='utf-8'),wait_until='load')
        assert page.evaluate('window.conductorViewer.schema')=='conductor_sim.visualization.v1'
        count=page.evaluate('window.conductorViewer.frameCount')
        first=page.locator('#scene').evaluate('(c)=>c.toDataURL()')
        if count>1:
            page.locator('#frame').evaluate('(e)=>{e.value=e.max;e.dispatchEvent(new Event("input",{bubbles:true}))}')
            assert page.locator('#counter').inner_text()==f'{count} / {count}'
            last=page.locator('#scene').evaluate('(c)=>c.toDataURL()')
            assert first!=last,'sweep controls did not change rendered pixels'
        baseline=page.locator('#scene').evaluate('(c)=>c.toDataURL()')
        page.locator('#lines').uncheck()
        assert page.locator('#scene').evaluate('(c)=>c.toDataURL()')!=baseline
        page.locator('#lines').check()
        before=page.locator('#profile').evaluate('(c)=>c.toDataURL()')
        if page.locator('#density option').count()>1:
            page.locator('#density').select_option('1')
            assert page.locator('#profile').evaluate('(c)=>c.toDataURL()')!=before
            page.locator('#density').select_option('0')
        page.locator('#scene').hover(position={'x':300,'y':300})
        assert 'u=' in page.locator('#probe').inner_text()
        assert page.evaluate('window.PWNED===undefined')
        assert not errors,errors
        assert not requests,requests
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
        if args.screenshot:
            args.screenshot.parent.mkdir(parents=True,exist_ok=True)
            page.screenshot(path=str(args.screenshot),full_page=True)
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
        print(json.dumps({'browser':'Chromium','frames':count,'javascript_errors':errors,
                          'external_requests':requests,'slider':'passed','toggles':'passed',
                          'profile_selector':'passed','hover':'passed','mobile_overflow':False}))
        browser.close()

if __name__=='__main__':main()
