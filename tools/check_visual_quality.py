"""Offline Chromium acceptance for smooth potential and nested density controls."""
from copy import deepcopy
import json
import shutil
from pathlib import Path
import argparse

from playwright.sync_api import sync_playwright
from conductor_sim import (ConductingPlane, ElectrostaticProblem, FixedPotential,
                           ParallelPlateBackend, PointCharge)
from conductor_sim.visualization import SliceSpec, SamplingContext, sample_frame, sample_sweep, SourceMask
from conductor_sim.viewer import render_html


class Linear:
    conductor_results=()
    def potential_at(self,p): return 2*p[0]+3*p[2]
    def electric_field_at(self,p): return (-2.,0.,-3.)


def slab(charges=(), bias=0.):
    return ElectrostaticProblem(tuple(charges), (
        ConductingPlane(name='lower'),
        ConductingPlane((0.,0.,.2),(0.,0.,-1.),FixedPotential(bias),'upper')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screenshot',type=Path)
    args=parser.parse_args()
    spec=SliceSpec(x_range_m=(-.3,.3),y_range_m=(-.1,.3),resolution=21)
    models=[('capacitor',slab(bias=20.)),
            ('two charges',slab((PointCharge(1e-9,(0,0,.065)),PointCharge(-.5e-9,(0,0,.14))))),
            ('zero',slab())]
    bundle=sample_sweep(models,ParallelPlateBackend(image_order=24),plane=spec,line_steps=180)
    with sync_playwright() as pw:
        exe=shutil.which('chromium') or shutil.which('google-chrome')
        browser=pw.chromium.launch(executable_path=exe,headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1360,'height':1120})
        errors=[];requests=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        def block(route):
            requests.append(route.request.url);route.abort()
        page.route('http://**',block);page.route('https://**',block)
        def load(b):
            page.goto('about:blank');page.set_content(render_html(b),wait_until='load')
        def slider(value,which='line-density'):
            page.locator('#'+which).evaluate('(e,v)=>{e.value=v;e.dispatchEvent(new Event("input",{bubbles:true}))}',str(value))
        load(bundle)
        slider(1,'frame')
        # A previously blank coarse-grid halo is filled only by finer
        # samples from the same solution; the charge core itself stays null.
        near=page.evaluate("""()=>{
          const index=interpolationFor(frame), y=.065;
          const gap=[];
          for(let k=1;k<=20;k++){
            const x=.001*k;
            const coarse=gridSample(frame.grid,index,x,y);
            const fine=potentialSample(frame,x,y);
            if(coarse===null && fine!==null)gap.push([x,fine]);
          }
          return {gap,core:potentialSample(frame,0,.065), patches:frame.potential_refinements?.length};
        }""")
        assert near['patches']==2 and near['gap'],near
        assert near['core'] is None
        raw=page.evaluate('JSON.stringify(data.frames)')
        builds=page.evaluate('window.conductorViewer.heatmapBuilds')
        prev=[];counts=[];pixels=[]
        for level in range(1,6):
            slider(level)
            current=page.evaluate('shownFieldLines().map(l=>JSON.stringify(l.points_m))')
            assert current[:len(prev)]==prev,'density moved/replaced an existing trace'
            assert len(current)>len(prev)
            counts.append(len(current));prev=current
            pixels.append(page.locator('#scene').evaluate('(c)=>c.toDataURL()'))
            assert f'{len(current)} /' in page.locator('#line-density-value').inner_text()
        assert len(set(pixels))==5
        assert page.evaluate('window.conductorViewer.heatmapBuilds')==builds,'density rebuilt potential raster'
        assert page.evaluate('JSON.stringify(data.frames)')==raw,'control changed exported SI data'
        slider(3)
        page.locator('#line-density').focus();page.keyboard.press('ArrowRight')
        assert page.locator('#line-density').input_value()=='4'
        slider(0,'frame');assert page.locator('#line-density').input_value()=='4'
        page.locator('#lines').uncheck();assert page.locator('#line-density').is_disabled()
        assert page.locator('#line-density-value').inner_text()=='Field lines off'
        page.locator('#lines').check();assert not page.locator('#line-density').is_disabled()
        slider(2,'frame');assert page.locator('#line-density').is_disabled()
        assert page.locator('#line-density-value').inner_text()=='No traces'
        slider(1,'frame');assert not page.locator('#line-density').is_disabled()
        # The legend uses the inverse of the SAME nonlinear map as colours.
        assert page.evaluate('Math.abs(Math.asinh(halfColourValue/soft)/colourExtent-.5)<1e-12')
        assert page.evaluate('colourComponents(0)')==[248,249,251]
        values=page.evaluate('[.1,.5,1,2,10].map(f=>Math.asinh(f*soft/soft)/colourExtent)')
        assert values==sorted(values) and len(set(values))==5
        assert 'asinh' in page.locator('#potential-method').inner_text()
        if args.screenshot:
            args.screenshot.parent.mkdir(parents=True,exist_ok=True)
            page.screenshot(path=str(args.screenshot),full_page=True)
        for width in (390,320):
            page.set_viewport_size({'width':width,'height':844})
            assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
        # Independent linear field: interpolation is exact, not a colour blur.
        empty=ElectrostaticProblem()
        context=SamplingContext(empty,Linear(),'linear')
        linear=sample_frame(context,SliceSpec(resolution=9),seeds=[])
        fixture={'schema':bundle['schema'],'color_limit_v':1.,'frames':[linear]}
        page.set_viewport_size({'width':1360,'height':1120});load(fixture)
        for x,y in ((.08,.12),(-.19,.07),(.21,-.13)):
            value=page.evaluate('([x,y])=>potentialSample(frame,x,y)',[x,y])
            assert abs(value-(2*x+3*y))<1e-12
        # Raster pixels match scalar interpolation followed by the colour map.
        assert page.evaluate('''()=>{const [x,y]=[411,275];
            const wx=frame.plane.x_range_m[0]+(x+.5-map.ox)/map.scale;
            const wy=frame.plane.y_range_m[0]+(canvas.height-y-.5-map.oy)/map.scale;
            const expected=colourComponents(2*wx+3*wy);
            const actual=Array.from(heatCache.layer.getContext('2d').getImageData(x,y,1,1).data);
            return expected.every((c,i)=>c===actual[i])&&actual[3]===255}''')
        # Null/source/domain cells remain unavailable; no nearest-valid fill.
        context.masks=(SourceMask((0.,0.,0.),.04,'numerical-core'),)
        masked=sample_frame(context,SliceSpec(resolution=9),seeds=[])
        fixture['frames']=[masked];load(fixture)
        assert page.evaluate('potentialSample(frame,0,0)') is None
        assert page.evaluate('heatCache.layer.getContext("2d").getImageData(360,360,1,1).data[3]')==0
        context.masks=();context.domain_radius_m=.15
        fixture['frames']=[sample_frame(context,SliceSpec(resolution=9),seeds=[])];load(fixture)
        assert page.evaluate('potentialSample(frame,.2,0)') is None
        # Boundary-clipped triangles reproduce a source-free capacitor to each face.
        load({'schema':bundle['schema'],'color_limit_v':20.,'frames':[bundle['frames'][0]]})
        for y in (.001,.199):
            value=page.evaluate('(y)=>potentialSample(frame,.081,y)',y)
            assert abs(value-100*y)<1e-10,(y,value)
        # Legacy v1 bundle: all existing data remains unchanged in Python.
        old=deepcopy(bundle);old.pop('potential_colour_scale')
        for f in old['frames']:
            f.pop('potential_interpolation');f.pop('field_line_pool')
        raw=json.dumps(old);load(old)
        assert json.dumps(old)==raw
        assert not errors,errors
        assert not requests,requests
        browser.close()
    print(json.dumps({'density_counts':counts,'nested_lines':'passed','cached_potential':'passed',
                      'unchanged_data':'passed','keyboard':'passed','state_persistence':'passed',
                      'mask_and_domain':'passed','linear_interpolation':'passed','plate_boundary':'passed',
                      'nonlinear_legend':'passed','mobile_widths':[390,320],
                      'legacy_bundle':'passed','near_charge_refinement':'passed',
                      'javascript_errors':errors,'external_requests':requests}))


if __name__=='__main__':main()
