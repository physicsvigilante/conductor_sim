"""Actual offline browser checks of immutable approximate before/after states."""
import argparse
import json
import shutil
from math import isclose
from pathlib import Path
from playwright.sync_api import sync_playwright, Error as PlaywrightError


def check(path, screenshot=None):
    errors=[];requests=[]
    with sync_playwright() as p:
        try:
            browser=p.chromium.launch(headless=True)
        except PlaywrightError:
            binary=shutil.which("chromium") or shutil.which("chromium-browser")
            if binary is None: raise
            browser=p.chromium.launch(headless=True,executable_path=binary,args=["--no-sandbox"])
        page=browser.new_page(viewport={'width':1440,'height':1100})
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('console',lambda e:errors.append(e.text) if e.type=='error' else None)
        page.on('request',lambda r:requests.append(r.url))
        page.set_content(Path(path).read_text(encoding='utf-8'),wait_until='load')
        page.wait_for_function('window.distantSphereViewer')
        data=page.evaluate('JSON.parse(document.getElementById("distant-data").textContent)')
        assert data['approximation']=='self-capacitance-leading-order'
        assert not page.evaluate('window.distantSphereViewer.hasTimeEvolution')
        assert 'not a discharge animation' in page.locator('#scope').inner_text()
        assert 'NOT a 5%' in page.locator('#qualification').inner_text()
        for id in ('before','after'):
            assert page.locator('#'+id).is_visible()
            table=page.locator('#'+id+'-table').inner_text()
            for index,row in enumerate(data[id]['conductors']):
                assert row['name'] in table
                displayed=page.locator('#'+id+'-table tr').nth(index+1).locator('td').all_text_contents()
                expected=[row[k] for k in ('charge_c','potential_v','mean_outer_sigma_c_m2','outer_field_normal_v_m')]
                assert all(isclose(float(a),b,rel_tol=6e-6,abs_tol=1e-25) for a,b in zip(displayed[1:],expected))
            ink=page.evaluate('''id=>{const c=document.getElementById(id),d=c.getContext('2d').getImageData(0,0,c.width,c.height).data;let n=0;for(let i=0;i<d.length;i+=4)if(d[i]+d[i+1]+d[i+2]<550)n++;return n}''',id)
            assert ink>150
        # Labels toggle without changing saved physics or inventing intermediate states.
        saved=page.locator('#distant-data').text_content()
        image=page.locator('#after').evaluate('(c)=>c.toDataURL()')
        page.locator('#show-delta').click()
        assert page.evaluate('window.distantSphereViewer.deltaLabels')
        assert page.locator('#after').evaluate('(c)=>c.toDataURL()')!=image
        page.locator('#show-delta').click()
        assert page.locator('#after').evaluate('(c)=>c.toDataURL()')==image
        assert page.locator('#distant-data').text_content()==saved
        if screenshot: page.screenshot(path=str(screenshot),full_page=True)
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        assert not errors,errors
        assert not requests,requests
        browser.close()
    return dict(offline=True,mode='static approximate equilibria',source_ids=data['conductor_ids'],
                label_toggle=True,painted=True,mobile_width_px=390,errors=errors,requests=requests)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('html');ap.add_argument('--screenshot')
    args=ap.parse_args();print(json.dumps(check(args.html,args.screenshot),indent=2))
