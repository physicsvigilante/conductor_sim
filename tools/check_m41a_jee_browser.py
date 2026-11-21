"""Actual offline Chromium evidence for newly authored M41a source p61 card.

The other route viewers have their own already released, separate source
browser checkers. This new probe attests the p61 static/source/teacher numbers
in *four actual Chromium modes*, not whether students learned anything.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from math import isclose

from tools.m41a_jee_trial_reference import check as independent_p61


def check(directory:Path)->dict:
    from playwright.sync_api import sync_playwright

    report=json.loads((directory/'m41a-trial.json').read_text(encoding='utf-8'))
    assert report['actual_browser_verified_count_in_this_run']==0
    native=json.loads((directory/'p61/p61-native.json').read_text(encoding='utf-8'))
    oracle=independent_p61(native)
    if not oracle['qualified_static_grounded_sphere']:
        raise AssertionError('physical source p61 failed separate reference')
    html=directory/'p61/p61-static-teacher.html'
    folder=directory/'p61/browser'
    folder.mkdir(parents=True,exist_ok=True)
    results=[]
    with sync_playwright() as p:
        chromium=p.chromium.launch(headless=True)
        try:
            for enabled in (True,False):
                for width,height in ((1280,820),(390,760)):
                    context=chromium.new_context(viewport={'width':width,'height':height},java_script_enabled=enabled)
                    page=context.new_page()
                    http=[]
                    errors=[]
                    page.on('request',lambda r:http.append(r.url) if r.url.startswith(('http://','https://')) else None)
                    page.on('pageerror',lambda e:errors.append(str(e)))
                    page.goto(html.resolve().as_uri(),wait_until='load',timeout=45000)
                    body=page.locator('body').inner_text()
                    for source_word in ('original notes p61 Illustration 1','Teacher-authored example, NOT source numbers',
                                        'no physical switch current','−QR/r','Q=+1e−9 C','R=0.10 m','r=0.32 m',
                                        'Six signed native samples','Externally edited HTML is NOT authenticated'):
                        if source_word not in body:
                            raise AssertionError('misleading or absent original/source/teacher visible text: '+source_word)
                    charge=page.locator('#p61-final-charge').inner_text().strip()
                    if charge!=format(native['native_sphere_net_charge_c'],'.17g')+' C':
                        raise AssertionError('visible final charge disagrees with separate SI-checked native charge')
                    if page.locator('svg').count()!=1 or not page.locator('svg').is_visible():
                        raise AssertionError('missing actual grounded-sphere source schematic')
                    circle=page.locator('svg circle')
                    if circle.count()!=2:
                        raise AssertionError('source diagram has wrong physical elements')
                    sphere= circle.nth(0)
                    charge_dot=circle.nth(1)
                    for el,attr,expected in ((sphere,'cx',360.),(sphere,'cy',100.),(sphere,'r',65.),
                                             (charge_dot,'cx',152.),(charge_dot,'cy',100.)):
                        if not isclose(float(el.get_attribute(attr)),expected,rel_tol=0,abs_tol=1e-12):
                            raise AssertionError('source 2D schematic coordinates wrong')
                    if not isclose((360.-152.)/65.,native['teacher_numeric_values_not_printed']['r_m']/native['teacher_numeric_values_not_printed']['R_m'],rel_tol=1e-13):
                        raise AssertionError('drawn true teacher r/R false')
                    rows=page.locator('tbody tr')
                    if rows.count()!=6:
                        raise AssertionError('missing pointwise signed sigma samples')
                    for i,point in enumerate(native['native_signed_sigma_samples']):
                        row=rows.nth(i)
                        if row.locator('td').count()!=2:
                            raise AssertionError('missing angle or signed density')
                        if row.locator('td').nth(0).inner_text()!=format(point['theta_rad'],'.12g') or row.locator('td').nth(1).inner_text()!=format(point['sigma_c_per_m2'],'.12g'):
                            raise AssertionError('visible source sigma differs from independent scientific evidence')
                    if http or errors:
                        raise AssertionError('unexpected external request/JS error')
                    if width<500 and page.locator('body').evaluate('(el) => el.scrollWidth')>width+30:
                        raise AssertionError('narrow viewport clips actual source/teacher label')
                    if page.locator('script').count()!=0:
                        raise AssertionError('source p61 static card falsely contains JavaScript')
                    path=folder/f'grounded-p61-{width}-{("js" if enabled else "nojs")}.png'
                    page.screenshot(path=str(path),full_page=True,animations='disabled')
                    results.append({'js_enabled':enabled,'viewport_px':width,'source_text_visible':True,
                                    'source_real_physics_ok':True,'http_requests':0,'page_errors':0,
                                    'screenshot':str(path.relative_to(directory))})
                    context.close()
        finally:
            chromium.close()
    out={'schema':'conductor_sim.m41a_real_source_browser.v1',
         'source_case_id':'p61-external-grounded-sphere',
         'actual_chromium_modes_verified':len(results),
         'verified_new_original_jEE_questions':0,
         'actual_teaching_learning_evaluated':False,
         'separate_native_reference_verified':True,
         'results':results}
    (folder/'browser-evidence.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',required=True,type=Path)
    a=p.parse_args()
    x=check(a.output_dir)
    print('M41a actual source p61 browser modes:',x['actual_chromium_modes_verified'])

if __name__=='__main__':
    main()
