"""First post-freeze M25a R2: developer-exposed, independently checked transfer.

Exact inputs are frozen in docs/m25a_reserved_card.md. The reference RHS and
fixed-step integration below do not import the production RHS or integrator.
Run only after committing the complete candidate and report any failure as-is.
"""
from __future__ import annotations

import argparse
from math import hypot, pi, sqrt
import json
from pathlib import Path
import shutil

from conductor_sim import (
    EPSILON_0, PointCharge, PrescribedField, TestChargeMotionSpec,
    UniformElectricField, UniformGravity, UniformRingAxisField,
)
from conductor_sim.scenarios import PrescribedScenario
from conductor_sim.driven_motion import (
    DrivenModel, DrivenStudy, FieldPhase, LinearDrag, UniformFieldSchedule,
    investigate_driven_motion,
)
from conductor_sim.motion_events import EventPolicy, PlaneEvent
from conductor_sim.driven_viewer import render_driven_run
from playwright.sync_api import sync_playwright, Error as PlaywrightError


def frozen_model():
    spec=TestChargeMotionSpec(
        PrescribedField((UniformRingAxisField(2e-12,.15,center_m=(0.,0.,0.),axis=(0.,0.,1.)),
                         UniformElectricField((0.,0.,.01)))),
        PointCharge(-1e-9,(0.,0.,.12)),1e-9,(0.,0.,-.01),
        duration_s=.7,sample_interval_s=.05,plane_normal=(0.,1.,0.),
        gravity=UniformGravity((0.,0.,-.03)),length_scale_m=.3,
        relative_tolerance=1e-10)
    base=PrescribedScenario('m25a:reserved',spec,
        (('ring:reserved','Nonresponding thin charged ring'),('field:fixed','Held uniform electric field')),
        'Precommitted R2 teacher hypothetical; no assertion of source-given JEE facts.')
    return DrivenModel(base,LinearDrag(3e-10),UniformFieldSchedule((
        FieldPhase(0.,(0.,0.,.04)),FieldPhase(.23,(0.,0.,-.07)))),
        'm25a:R2','Teacher authored: independent z-axis ring projection, drag, gravity and scheduled E.')


def reference_at(t):
    """Independent actual Coulomb-ring RHS + four work integrals, fixed RK4."""
    q=-1e-9;m=1e-9;b=3e-10;Q=2e-12;r=.15
    k=1/(4*pi*EPSILON_0)
    y=(.12,-.01,0.,0.,0.,0.) # z, v, Wfixed, Wapplied, Wgravity, D
    def deriv(state, phase):
        z,v=state[:2]
        er=k*Q*z/(r*r+z*z)**1.5
        ff=q*(er+.01)
        fa=q*(.04 if phase==0 else -.07)
        fg=m*(-.03)
        drag=-b*v
        return (v,(ff+fa+fg+drag)/m,ff*v,fa*v,fg*v,b*v*v)
    def step(state,h,phase):
        a=deriv(state,phase)
        b0=deriv(tuple(y+h*x/2 for y,x in zip(state,a)),phase)
        c=deriv(tuple(y+h*x/2 for y,x in zip(state,b0)),phase)
        d=deriv(tuple(y+h*x for y,x in zip(state,c)),phase)
        return tuple(y+h*(i+2*j+2*k+l)/6 for y,i,j,k,l in zip(state,a,b0,c,d))
    if t<0 or t>.7:raise ValueError('outside committed reference time')
    last=0.
    for end,phase in ((min(t,.23),0),(t,1)):
        if end<=last:continue
        n=int((end-last)/.0001)+1
        h=(end-last)/n
        for _ in range(n):y=step(y,h,phase)
        last=end
    return y


def reference_event():
    """Bisection on independent integrated z(t), without using event locator."""
    previous_t=0.;previous_z=reference_at(previous_t)[0]-.10
    intervals=[]
    for j in range(1,701):
        t=min(.7,j*.001)
        z=reference_at(t)[0]-.10
        if previous_z*z<0:intervals.append((previous_t,t))
        previous_t,previous_z=t,z
    if not intervals:return None
    a,b=intervals[0]
    for _ in range(30):
        mid=(a+b)/2
        if (reference_at(a)[0]-.10)*(reference_at(mid)[0]-.10)<=0:b=mid
        else:a=mid
    mid=(a+b)/2
    if reference_at(mid)[1]>=0:raise AssertionError('first independent detector crossing not downward')
    return mid


def check(output:Path):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    model=frozen_model()
    initial=model.as_dict()
    questions=(PlaneEvent('detector:z',(.0,.0,.10),(0.,0.,1.),
                          'Downward physical detector',-1,False),)
    run=investigate_driven_motion(DrivenStudy(model,questions,
        EventPolicy(.01,1e-9,1e-10,1e-8)))
    assert model.as_dict()==initial
    assert all(abs(s.position_m[0])+abs(s.position_m[1])==0 for s in run.samples)
    errors=[]
    for s in run.samples:
        z,v,wf,wa,wg,d=reference_at(s.time_s)
        errors.append((abs(s.position_m[2]-z),abs(s.velocity_m_s[2]-v),
                       max(abs(s.fixed_work_j-wf),abs(s.applied_work_j-wa),
                           abs(s.gravitational_work_j-wg),abs(s.dissipated_energy_j-d))))
    max_pos=max(x[0] for x in errors);max_vel=max(x[1] for x in errors);max_work=max(x[2] for x in errors)
    assert max_pos<2e-6 and max_vel<2e-5 and max_work<2e-14, (max_pos,max_vel,max_work)
    independent_t=reference_event()
    assert independent_t is not None and run.events,('frozen first observation',independent_t,run.events)
    actual_t=run.events[0].sample.time_s
    assert run.events[0].event_id=='detector:z'
    assert abs(actual_t-independent_t)<2e-5,(actual_t,independent_t)
    assert len(run.switches)==1 and run.switches[0].time_s==.23
    assert run.switches[0].left.position_m==run.switches[0].right.position_m
    assert run.switches[0].left.velocity_m_s==run.switches[0].right.velocity_m_s
    html=render_driven_run(run,title='M25a R2 · frozen rotated-axis ring + drag + switching')
    (output/'m25a-reserved.html').write_text(html,encoding='utf-8')
    (output/'m25a-reserved.json').write_text(json.dumps(run.as_dict(),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    js_errors=[];console_errors=[];requests=[]
    with sync_playwright() as p:
        try:browser=p.chromium.launch(headless=True)
        except PlaywrightError:
            exe=shutil.which('chromium') or shutil.which('chromium-browser')
            if not exe:raise
            browser=p.chromium.launch(headless=True,executable_path=exe,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1440,'height':1050})
        page.on('pageerror',lambda e:js_errors.append(str(e)))
        page.on('console',lambda e:console_errors.append(e.text) if e.type=='error' else None)
        page.on('request',lambda r:requests.append(r.url))
        page.set_content(html,wait_until='load')
        page.wait_for_function('window.conductorDrivenViewer?.loaded')
        data=page.evaluate('data')
        geom=data['source_geometry'];a,b=geom['entries']
        assert a['shape']=='ring-projection' and a['edge_on']
        assert a['world_axis']==[0.,0.,1.] and a['radius_m']==.15
        assert b['shape']=='field-arrows'
        assert page.locator('#sources').inner_text().count('axis-only')>=1
        off=page.locator('#scene').evaluate('c=>c.toDataURL()')
        page.locator('#source-geometry').uncheck()
        assert page.locator('#scene').evaluate('c=>c.toDataURL()')!=off
        page.locator('#source-geometry').check()
        assert page.locator('#scene').evaluate('c=>c.toDataURL()')==off
        page.locator('#frame').fill(str(len(data['display_states'])-1))
        page.locator('#frame').dispatch_event('input')
        assert page.evaluate('window.conductorDrivenViewer.physicalTime')==run.samples[-1].time_s
        assert page.locator('#events tbody tr').count()==1
        page.screenshot(path=str(output/'m25a-reserved.png'),full_page=True)
        page.locator('#reset').click();page.locator('#play').click()
        page.wait_for_function('window.conductorDrivenViewer.frameIndex>=1',timeout=7000)
        if page.evaluate('window.conductorDrivenViewer.playing'):page.locator('#play').click()
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1')
        assert not js_errors and not console_errors and not requests
        browser.close()
    report=dict(case='M25a reserved transfer R2',exposure='same developer; not blinded or independent LLM authoring',
        first_execution='must be post-freeze; timestamp supplied by CI logs',
        model_unchanged=True,source_inventory=['ring:reserved','field:fixed'],
        observed_detector_t_s=actual_t,independent_detector_t_s=independent_t,
        error_detector_s=abs(actual_t-independent_t),max_position_difference_m=max_pos,
        max_velocity_difference_m_s=max_vel,max_work_difference_j=max_work,
        browser='real Chromium',mobile_width_px=390,js_errors=js_errors,console_errors=console_errors,
        network_requests=requests)
    (output/'m25a-reserved-report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,default=Path('output/m25a-reserved'))
    args=a.parse_args();print(json.dumps(check(args.output),indent=2))
