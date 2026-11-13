"""Optional browser acceptance check; requires Playwright and Chromium.

python tools/check_viewer.py output/conductors.html --screenshot output/view.png
"""
import argparse
import json
import re
from copy import deepcopy
from conductor_sim.viewer import render_html
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
        page.locator('#scene').hover(position={'x':300,'y':300})
        assert 'u=' in page.locator('#probe').inner_text()
        if count>1:
            page.locator('#frame').evaluate('(e)=>{e.value=e.max;e.dispatchEvent(new Event("input",{bubbles:true}))}')
            assert page.locator('#counter').inner_text()==f'{count} / {count}'
            assert page.locator('#probe').inner_text().startswith('Move the pointer'), 'stale values from previous state'
            last=page.locator('#scene').evaluate('(c)=>c.toDataURL()')
            assert first!=last,'sweep controls did not change rendered pixels'
        # Exercise the density slider on the actual exported numerical/analytical
        # data, without retracing or changing stored SI values.
        raw=page.evaluate('JSON.stringify(frame)')
        builds=page.evaluate('window.conductorViewer.heatmapBuilds')
        if page.evaluate('frame.field_lines.length')>1:
            page.locator('#line-density').evaluate('(e)=>{e.value=1;e.dispatchEvent(new Event("input"))}')
            sparse=page.evaluate('shownFieldLines().map(l=>JSON.stringify(l.points_m))')
            page.locator('#line-density').evaluate('(e)=>{e.value=5;e.dispatchEvent(new Event("input"))}')
            full=page.evaluate('shownFieldLines().map(l=>JSON.stringify(l.points_m))')
            assert len(full)>len(sparse) and full[:len(sparse)]==sparse
            page.locator('#line-density').evaluate('(e)=>{e.value=3;e.dispatchEvent(new Event("input"))}')
        assert page.evaluate('window.conductorViewer.heatmapBuilds')==builds
        assert page.evaluate('JSON.stringify(frame)')==raw
        baseline=page.locator('#scene').evaluate('(c)=>c.toDataURL()')
        has_traces=page.evaluate('shownFieldLines().some(l=>l.points_m.length>=2)')
        page.locator('#lines').uncheck()
        after_toggle=page.locator('#scene').evaluate('(c)=>c.toDataURL()')
        # A grounded, charge-free final equilibrium may have no field lines.
        # A toggled checkbox cannot remove pixels that were never drawn.
        if has_traces:
            assert after_toggle!=baseline
        else:
            assert after_toggle==baseline
        page.locator('#lines').check()
        before=page.locator('#profile').evaluate('(c)=>c.toDataURL()')
        if page.locator('#density option').count()>1:
            # Different conductor faces can legitimately have identical sigma
            # arrays (e.g. both zero after a wire is connected). Prefer a
            # distinct actual profile when there is one; never demand changed
            # pixels merely because a dropdown option changed.
            changed=page.evaluate('''(()=>{
                const profiles=frame.profiles;
                const base=JSON.stringify(profiles[0].sigma_c_per_m2);
                for(let i=1;i<profiles.length;i++)
                    if(JSON.stringify(profiles[i].sigma_c_per_m2)!==base)return i;
                return null;
            })()''')
            index=changed if changed is not None else 1
            page.locator('#density').select_option(str(index))
            assert page.locator('#density').input_value()==str(index)
            if changed is not None:
                assert page.locator('#profile').evaluate('(c)=>c.toDataURL()')!=before
            page.locator('#density').select_option('0')
        page.locator('#scene').hover(position={'x':300,'y':300})
        assert 'u=' in page.locator('#probe').inner_text()
        # Leaving the plot must not leave a previous point's values visible.
        page.locator('#scene').hover(position={'x':2,'y':2})
        assert page.locator('#probe').inner_text().startswith('Move the pointer')
        page.locator('#scene').hover(position={'x':300,'y':300})
        assert 'u=' in page.locator('#probe').inner_text()
        page.mouse.move(0,0)
        assert page.locator('#probe').inner_text().startswith('Move the pointer')
        assert page.evaluate('Number(fmt(1e-12))') == 1e-12
        assert page.evaluate('Number(fmt(-1e-12))') == -1e-12
        assert page.evaluate('window.PWNED===undefined')
        assert not errors,errors
        assert not requests,requests
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
        if args.screenshot:
            args.screenshot.parent.mkdir(parents=True,exist_ok=True)
            page.screenshot(path=str(args.screenshot),full_page=True)
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
        # Exercise mixed-metadata states without calling another physics solver.
        # The last UI-only fixture has no density API and custom-solver provenance.
        match=re.search(r'<script id="simulation-data" type="application/json">(.*?)</script>',
                        args.html.read_text(encoding='utf-8'), re.S)
        bundle=json.loads(match.group(1))
        fixture=deepcopy(bundle)
        first_frame=deepcopy(fixture['frames'][0])
        no_profile=deepcopy(first_frame)
        no_profile['profiles']=[]
        no_profile['conductor_results_method']='Custom test solver conductor properties.'
        fixture['frames']=[first_frame,no_profile]
        page.goto('about:blank')  # fresh JS realm, as a newly opened exported file
        page.set_content(render_html(fixture),wait_until='load')
        page.locator('#frame').evaluate('(e)=>{e.value=e.max;e.dispatchEvent(new Event("input",{bubbles:true}))}')
        assert page.locator('#method').inner_text()=='Surface density unavailable for this state.'
        assert page.locator('#totals-note').inner_text()=='Custom test solver conductor properties.'
        assert not errors,errors
        assert not requests,requests
        print(json.dumps({'browser':'Chromium' ,'frames':count,'javascript_errors':errors,
                          'external_requests':requests,'slider':'passed','toggles':'passed',
                          'profile_selector':'passed','hover':'passed','mobile_overflow':False,
                          'state_probe_reset':'passed','missing_profile_reset':'passed','explicit_provenance':'passed',
                          'plot_exit_probe_reset':'passed','small_nonzero_values':'passed',
                          'field_line_density':'passed','potential_cache':'passed'}))
        browser.close()

if __name__=='__main__':main()
