"""Actual Chromium checks of vector directions and sparse-path metal gaps.

The stored SI vectors, not a pixel snapshot alone, determine expected arrows.
A coarse half-orbit has two vacuum samples whose display chord crosses metal:
that chord must be omitted, never shown as a physical conductor crossing.
"""
import argparse
import json
from math import hypot, pi, sqrt
from pathlib import Path
import shutil
from playwright.sync_api import sync_playwright, Error as PlaywrightError
from conductor_sim import (ElectrostaticProblem, PointCharge, SolidSphere, Grounded,
    ExternalSphereBackend, Motion2DSpec, force_energy, sample_motion_2d)
from conductor_sim.viewer import render_html


def sparse_orbit():
    p=ElectrostaticProblem((PointCharge(1e-9,(.1,0,0)),),(SolidSphere(.07,Grounded()),))
    backend=ExternalSphereBackend()
    F=force_energy(backend.solve(p)).charges[0].total_force_n[0]
    speed=sqrt(-F*.1/1e-6);duration=pi*.1/speed
    spec=Motion2DSpec(p,0,1e-6,(0,speed,0),duration,duration)
    return sample_motion_2d(spec,backend,resolution=13,seeds=[],line_steps=8,profile_samples=9)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('html',type=Path)
    parser.add_argument('--screenshot',type=Path)
    args=parser.parse_args()
    errors=[];requests=[];checked=0
    with sync_playwright() as playwright:
        try:browser=playwright.chromium.launch(headless=True)
        except PlaywrightError:
            binary=shutil.which('chromium') or shutil.which('chromium-browser')
            if binary is None:raise
            browser=playwright.chromium.launch(headless=True,executable_path=binary,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1240,'height':920})
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('request',lambda r:requests.append(r.url))
        page.set_content(args.html.read_text(encoding='utf-8'))
        data=page.evaluate('data');motion=data['motion_2d'];plane=data['frames'][0]['plane']
        for index,sample_index in enumerate(motion['frame_sample_indices']):
            page.locator('#frame').fill(str(index));page.locator('#frame').dispatch_event('input')
            for function,field in [('forceArrow','force_n'),('velocityArrow','velocity_m_s')]:
                lines=page.evaluate('''name=>{
                    const operations=[],m=ctx.moveTo,l=ctx.lineTo;
                    ctx.moveTo=function(x,y){operations.push(['move',x,y]);return m.call(this,x,y)};
                    ctx.lineTo=function(x,y){operations.push(['line',x,y]);return l.call(this,x,y)};
                    const p=frame.plane;
                    const project=v=>{const d=v.map((x,i)=>x-p.origin_m[i]);
                        const n=[p.u[1]*p.v[2]-p.u[2]*p.v[1],p.u[2]*p.v[0]-p.u[0]*p.v[2],p.u[0]*p.v[1]-p.u[1]*p.v[0]];
                        return [p.u,p.v,n].map(axis=>axis.reduce((s,x,i)=>s+x*d[i],0))};
                    try{(name==='forceArrow'?forceArrow:velocityArrow)(p,x=>x,y=>-y,project)}
                    finally{ctx.moveTo=m;ctx.lineTo=l}
                    return operations.slice(0,2);
                }''',function)
                vector=motion['samples'][sample_index][field]
                expected=(sum(vector[i]*plane['u'][i] for i in range(3)),
                          -sum(vector[i]*plane['v'][i] for i in range(3)))
                if hypot(*expected)<1e-30:continue
                assert len(lines)==2 and [x[0] for x in lines]==['move','line'],(function,lines)
                drawn=(lines[1][1]-lines[0][1],lines[1][2]-lines[0][2])
                alignment=sum(a*b for a,b in zip(drawn,expected))/(hypot(*drawn)*hypot(*expected))
                assert alignment>1-1e-10,(function,index,alignment)
                checked+=1
        orbit=sparse_orbit()
        assert orbit['motion_2d']['trajectory_break_indices']==[1]
        page.close()
        page=browser.new_page(viewport={'width':1240,'height':920})
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('request',lambda r:requests.append(r.url))
        page.set_content(render_html(orbit,title='Sparse saved orbit — no chord through metal'))
        page.locator('#frame').fill('1');page.locator('#frame').dispatch_event('input')
        count=page.evaluate('''()=>{
            let calls=0;const old=ctx.lineTo,p=frame.plane;
            ctx.lineTo=function(...args){calls++;return old.apply(this,args)};
            const project=v=>{const d=v.map((x,i)=>x-p.origin_m[i]);
                return [p.u,p.v].map(axis=>axis.reduce((s,x,i)=>s+x*d[i],0)).concat([0])};
            try{drawTrajectory(p,x=>x,y=>y,project)}finally{ctx.lineTo=old}
            return calls;
        }''')
        assert count==0,'a sparse chord was drawn through conducting metal'
        assert 'Gaps omit chords' in page.locator('#motion-scope-note').inner_text()
        if args.screenshot:
            args.screenshot.parent.mkdir(parents=True,exist_ok=True)
            page.screenshot(path=str(args.screenshot),full_page=True)
        assert not errors,errors
        assert not requests,requests
        browser.close()
    print(json.dumps(dict(vector_direction_checks=checked,sparse_metal_chord='omitted',
                          javascript_errors=errors,external_requests=requests),indent=2))

if __name__=='__main__':main()
