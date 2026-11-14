"""M19e actual Chromium check: authored source identities and world-aligned scene.

This checks painted canvas geometry, not only JSON or DOM metadata. The M19b
checker separately covers physics-time graphs, arrow directions and playback.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from playwright.sync_api import sync_playwright, Error as PlaywrightError


def run(html_path, screenshot=None, *, disc=False):
    html_path=Path(html_path)
    assert html_path.is_file()
    errors,requests=[],[]
    with sync_playwright() as p:
        try:
            browser=p.chromium.launch(headless=True)
        except PlaywrightError:
            executable=shutil.which('chromium') or shutil.which('chromium-browser')
            if executable is None:raise
            browser=p.chromium.launch(headless=True,executable_path=executable,
                                      args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1240,'height':920})
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.on('request',lambda req:requests.append(req.url))
        page.set_content(html_path.read_text(encoding='utf-8'),wait_until='load')
        source=page.evaluate('data')
        m=source['prescribed_motion']
        assert m['axis_display']=='horizontal'
        assert m['axis']==[1.,0.,0.]
        assert m['field']['sources'][0]['axis']==[1.,0.,0.]
        assert (m['field']['sources'][0]['kind']=='UniformDiskAxisField') is disc
        assert source['frames'][0]['plane']['u']==[1.,0.,0.]
        assert source['frames'][0]['plane']['v']==[0.,1.,0.]
        expected_ids=['disc:D'] if disc else ['ring:A','ring:B','field:external']
        expected_labels=['Disc D'] if disc else ['Ring A','Ring B','External field']
        assert [a['id'] for a in m['source_annotations']]==expected_ids
        assert [a['label'] for a in m['source_annotations']]==expected_labels
        if disc:
            centre=m['profile_samples'][len(m['profile_samples'])//2]
            assert abs(centre['coordinate_m'])<1e-14
            assert centre['field_parallel_v_m'] is None
            assert centre['potential_v'] is not None
        assert page.locator('#scene-world-axes').is_visible()
        orientation=page.locator('#scene-world-axes').inner_text()
        assert 'screen right = world +x' in orientation
        assert 'screen up = world +y' in orientation
        assert 'symmetry axis = world +x (shown right)' in orientation
        text=page.locator('#sources').inner_text()
        entries=('Disc D [disc:D]','no conductor') if disc else (
            'Ring A [ring:A]','Ring B [ring:B]',
            'External field [field:external]','no conductor')
        for snippet in entries:
            assert snippet.casefold() in text.casefold(),snippet
        assert page.locator('#conductor-panel').is_hidden()
        assert page.locator('#density-panel').is_hidden()
        assert page.locator('#prescribed-panel').is_visible()
        assert page.locator('#motion-controls').is_visible()
        assert page.locator('#motion-energy-chart').is_visible()
        assert 'U=qV' in page.locator('#motion-summary').inner_text()
        # Capture what canvas text painting really draws. Each source label
        # should be painted in the schematic, not merely listed in metadata.
        labels=page.evaluate('''() => {
          const c=document.getElementById('scene').getContext('2d');
          const old=c.fillText.bind(c), labels=[];
          c.fillText=(...args)=>{labels.push(String(args[0]));return old(...args)};
          selectFrame();
          c.fillText=old;
          return labels;
        }''')
        if disc:
            assert any('Disc D disc' in v for v in labels),labels
        else:
            assert any('Ring A' in v for v in labels),labels
            assert any('Ring B' in v for v in labels),labels
        # Meridian intersections of rings perpendicular to screen +x should
        # be two coloured dots at (x, y) = (0,+/-.30),(.40,+/-.30).
        # A copy/paste of the old axis-vertical renderer would fail this.
        paint=page.evaluate('''() => {
          const canvas=document.getElementById('scene'),c=canvas.getContext('2d');
          const p=data.frames[0].plane,W=canvas.width,H=canvas.height,pad=46;
          const scale=Math.min((W-2*pad)/(p.x_range_m[1]-p.x_range_m[0]),
                               (H-2*pad)/(p.y_range_m[1]-p.y_range_m[0]));
          const ox=(W-scale*(p.x_range_m[1]-p.x_range_m[0]))/2;
          const oy=(H-scale*(p.y_range_m[1]-p.y_range_m[0]))/2;
          const X=x=>ox+(x-p.x_range_m[0])*scale;
          const Y=y=>H-oy-(y-p.y_range_m[0])*scale;
          const points=data.prescribed_motion.field.sources[0].kind==='UniformDiskAxisField'?
              [[0,.075],[0,-.075]]:[[0,.3],[0,-.3],[.4,.3],[.4,-.3]];
          return points.map(([x,y])=>
            Array.from(c.getImageData(Math.round(X(x)),Math.round(Y(y)),1,1).data));
        }''')
        for pixel in paint[:2]:
            assert pixel[0]>145 and pixel[0]>pixel[2]*1.4,('positive source pixel',pixel)
        if not disc:
            for pixel in paint[2:]:
                assert pixel[2]>115 and pixel[2]>pixel[0]*1.4,('ring B pixel',pixel)
        # Actual motion and time-varying graph cursor are still retained.
        slider=page.locator('#frame')
        slider.fill(str(len(source['frames'])-1))
        assert ('Disc D' if disc else 'Ring A') in page.locator('#sources').inner_text()
        assert 'U=qV' in page.locator('#motion-summary').inner_text()
        assert m['graph_sample_count']==len(m['samples'])
        assert len(m['samples'])>=len(source['frames'])
        if not disc: assert len(m['samples'])>len(source['frames'])
        if screenshot:
            Path(screenshot).parent.mkdir(parents=True,exist_ok=True)
            page.screenshot(path=str(screenshot),full_page=True)
        page.locator('#motion-reset').click()
        assert page.locator('#frame').input_value()=='0'
        page.locator('#motion-play').click()
        page.wait_for_timeout(240)
        assert page.evaluate('window.conductorViewer.playing') is True
        page.locator('#motion-play').click()
        assert page.evaluate('window.conductorViewer.playing') is False
        page.set_viewport_size({'width':390,'height':844})
        assert page.locator('#scene-world-axes').is_visible()
        assert 'world +x' in page.locator('#scene-world-axes').inner_text()
        assert page.locator('#scene').is_visible()
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
        assert not errors,errors
        assert not requests,requests
        browser.close()
        return {
            'file':html_path.name,'world_horizontal_axis':True,'source_identities':True,
            'painted_source_labels':labels,'source_geometry_pixels':paint,
            'source_semantics':('axis-normal disc diameter with surface field gap' if disc else
                                '2D schematic of axis-normal rings, not off-axis electric field'),
            'actual_numerical_samples':len(m['samples']),'frames':len(source['frames']),
            'playback_and_mobile':'passed','javascript_errors':errors,'external_requests':requests,
        }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('html',type=Path)
    parser.add_argument('--screenshot',type=Path)
    parser.add_argument('--disc',action='store_true',help='check an axis-horizontal disc demo')
    options=parser.parse_args()
    print(json.dumps(run(options.html,options.screenshot,disc=options.disc),indent=2))


if __name__=='__main__':main()
