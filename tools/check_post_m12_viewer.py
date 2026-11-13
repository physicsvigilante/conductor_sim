"""Browser regressions for missing density samples and plane-parallel slices."""
from copy import deepcopy
import json
import shutil

from playwright.sync_api import sync_playwright
from conductor_sim import ConductingPlane, ElectrostaticProblem, FixedPotential, ParallelPlateBackend
from conductor_sim.visualization import SliceSpec, sample_sweep
from conductor_sim.viewer import render_html


def problem():
    return ElectrostaticProblem(conductors=(
        ConductingPlane(name='lower'),
        ConductingPlane((0.,0.,.2),(0.,0.,-1.),FixedPotential(20.),'upper'),
    ))


def main():
    model = problem()
    bundle = sample_sweep([('automatic capacitor',model)],ParallelPlateBackend(),
                          resolution=17,line_steps=120)
    with sync_playwright() as pw:
        executable = shutil.which('chromium') or shutil.which('google-chrome')
        browser = pw.chromium.launch(executable_path=executable,headless=True,args=['--no-sandbox'])
        page = browser.new_page(viewport={'width':1360,'height':1120})
        errors = []; requests = []
        page.on('pageerror',lambda e:errors.append(str(e)))
        def block(route):
            requests.append(route.request.url)
            route.abort()
        page.route('http://**',block);page.route('https://**',block)
        def load(data):
            page.goto('about:blank')
            page.set_content(render_html(data),wait_until='load')
        load(bundle)
        before = page.locator('#scene').evaluate('(c)=>c.toDataURL()')
        page.locator('#lines').uncheck()
        assert page.locator('#scene').evaluate('(c)=>c.toDataURL()') != before

        sparse = deepcopy(bundle)
        # A real gap must not be plotted as sigma=0 or joined across.
        sparse['frames'][0]['profiles'] = [{
            'conductor':'lower','surface':'outer','coordinate_m':list(range(9)),
            'sigma_c_per_m2':[1e-9,1e-9,None,None,None,None,None,1e-9,1e-9],
            'method':'explicit excluded samples',
        }]
        load(sparse)
        assert 'gaps: excluded samples' in page.locator('#method').inner_text()
        # The constant 1 nC/m2 line lies near y=40.24; interior x=247 is a gap.
        count_blue = '''([x0,x1])=>{const c=document.getElementById('profile'),ctx=c.getContext('2d');
          const d=ctx.getImageData(x0,30,x1-x0,25).data;let n=0;
          for(let i=0;i<d.length;i+=4)if(d[i]<50&&d[i+1]>60&&d[i+1]<130&&d[i+2]>100&&d[i+2]<170&&d[i+3]>0)n++;return n}'''
        assert page.evaluate(count_blue,[80,100]) > 0
        assert page.evaluate(count_blue,[220,280]) == 0
        sparse['frames'][0]['profiles'][0]['sigma_c_per_m2'] = [None]*9
        load(sparse)
        assert 'no samples inside the numerical domain' in page.locator('#method').inner_text()

        for height in (-.1,.1,.3):
            spec=SliceSpec(origin_m=(0.,0.,height),u=(1.,0.,0.),v=(0.,1.,0.),resolution=9)
            slab=sample_sweep([('parallel slice',model)],ParallelPlateBackend(),plane=spec,seeds=[])
            load(slab)
            pixel=page.locator('#scene').evaluate('(c)=>Array.from(c.getContext("2d").getImageData(300,300,1,1).data)')
            assert (pixel[:3]==[175,189,201]) == (height<0 or height>.2)
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
        assert not errors,errors
        assert not requests,requests
        browser.close()
    print(json.dumps({'automatic_capacitor_lines':'passed','null_density_gaps':'passed',
                      'all_null_density':'passed','parallel_slice_metal_fill':'passed',
                      'javascript_errors':errors,'external_requests':requests,'mobile_overflow':False}))


if __name__ == '__main__':
    main()
