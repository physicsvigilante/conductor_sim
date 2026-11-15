"""M25a Chromium acceptance: actual source geometry, omissions and playback.

R2 reserved transfer is handled by a separate post-freeze tool; this runner
contains only already-exposed developmental/preselected scenarios.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import replace
from math import sqrt, hypot
import shutil

from playwright.sync_api import sync_playwright, Error as PlaywrightError

from conductor_sim import (
    FixedPointField, PointCharge, PrescribedField, TestChargeMotionSpec,
    UniformElectricField, UniformGravity, UniformRodField,
    UniformSphereField, UniformDiskAxisField, UniformRingAxisField,
)
from conductor_sim.scenarios import PrescribedScenario
from conductor_sim.motion_events import EventStudy, EventPolicy, PlaneEvent, investigate_motion
from conductor_sim.motion_event_viewer import render_event_run
from conductor_sim.driven_motion import (DrivenModel, DrivenStudy, FieldPhase, LinearDrag,
    UniformFieldSchedule, investigate_driven_motion)
from conductor_sim.driven_viewer import render_driven_run
from examples.public_api_event_trial import author
from examples.mirrored_rods import make_case

CASES=('coaxial-rings','mirror-rod','offplane-rods','hybrid-sources','ring-disk','event-rings',
       'tilted-rings','tilted-event-rings')


def prepare(case):
    policy=EventPolicy(.01,1e-9,1e-10,1e-9)
    if case in ('tilted-rings','tilted-event-rings'):
        original, queries = author('coaxial-rings')
        normal=(0.,1/sqrt(2),1/sqrt(2))
        spec=replace(original.spec,plane_normal=normal)
        base=PrescribedScenario('m25a:tilted-ring-plane', spec,original.base.source_annotations,
            'Teacher hypothetical: the true x-axis trajectory stays in the rotated motion plane.')
        if case=='tilted-event-rings':
            run=investigate_motion(EventStudy(base,(queries[1],),policy))
            return render_event_run(run,title='M25a F-01 corrected 45-degree edge-on ring geometry'),run.as_dict()
        driven=DrivenModel(base,original.drag,original.applied,'m25a:tilted-ring-driven',
            'Teacher hypothetical; true physical rotated plane with unchanged source laws.')
        run=investigate_driven_motion(DrivenStudy(driven,queries,policy))
        return render_driven_run(run,title='M25a F-01 corrected driven ring geometry'),run.as_dict()
    if case in ('coaxial-rings','mirror-rod','event-rings'):
        model, queries=author('coaxial-rings' if case=='event-rings' else case)
        if case=='event-rings':
            result=investigate_motion(EventStudy(model.base,(queries[1],),policy))
            return render_event_run(result,title='M25a shared geometry · conservative rings'),result.as_dict()
        result=investigate_driven_motion(DrivenStudy(model,queries,policy))
    elif case=='offplane-rods':
        model=make_case('P2')
        q=(PlaneEvent('detector',(.2,0.,0.),(1.,0.,0.),'Passive detector',0),)
        result=investigate_motion(EventStudy(model,q,policy))
        return render_event_run(result,title='M25a off-plane rods, unchanged physical model'),result.as_dict()
    else:
        if case=='hybrid-sources':
            srcs=(UniformSphereField(2e-11,.07,center_m=(.36,0.,0.)),
                  UniformRodField(-2e-11,(-.25,0.,-.06),(-.25,0.,.06)),
                  FixedPointField(1e-11,(-.3,0.,.16)),
                  UniformElectricField((.2,0.,0.)))
            labels=(('sphere:volume','Fixed volume sphere'),('rod:line','Held finite rod'),
                    ('point:held','Held point charge'),('field:static','Uniform fixed field'))
            p=(0.,0.,.2);v=(.02,0.,0.);g=(0.,0.,-.1)
        else:
            srcs=(UniformRingAxisField(4e-11,.12,center_m=(0.,0.,0.),axis=(1.,0.,0.)),
                  UniformDiskAxisField(-5e-11,.14,center_m=(.42,0.,0.),axis=(1.,0.,0.)))
            labels=(('ring:left','Thin charged ring'),('disk:right','Uniform fixed disc'))
            p=(.2,0.,0.);v=(.02,0.,0.);g=(.04,0.,0.)
        base=PrescribedScenario('m25a:'+case,TestChargeMotionSpec(
            PrescribedField(srcs),PointCharge(-1e-9,p),2e-9,v,.3,.05,
            plane_normal=(0.,1.,0.),gravity=UniformGravity(g)),labels,
            'Teacher-authored M25a display check; fixed sources never respond to test charge.')
        model=DrivenModel(base,LinearDrag(1e-10),UniformFieldSchedule((FieldPhase(0.,(.1,0.,0.)),)),
            'm25a:driven:'+case,'No new field law; draw only honest 3D projections or in-slice geometry.')
        queries=(PlaneEvent('detector',(0.,0.,0.),(1.,0.,0.),'Passive plane',0),)
        result=investigate_driven_motion(DrivenStudy(model,queries,policy))
    return render_driven_run(result,title='M25a shared source geometry · '+case),result.as_dict()


def check(case,out_dir:Path):
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    html,physical=prepare(case)
    (out_dir/(case+'.html')).write_text(html,encoding='utf-8')
    (out_dir/(case+'.json')).write_text(json.dumps(physical,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    browser_errors=[];console_errors=[];external_requests=[]
    with sync_playwright() as p:
        try:browser=p.chromium.launch(headless=True)
        except PlaywrightError:
            executable=shutil.which('chromium') or shutil.which('chromium-browser')
            if not executable:raise
            browser=p.chromium.launch(headless=True,executable_path=executable,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1450,'height':1050})
        page.on('pageerror',lambda e:browser_errors.append(str(e)))
        page.on('console',lambda e:console_errors.append(e.text) if e.type=='error' else None)
        page.on('request',lambda r:external_requests.append(r.url))
        page.set_content(html,wait_until='load')
        viewer='conductorEventViewer' if case in ('event-rings','offplane-rods','tilted-event-rings') else 'conductorDrivenViewer'
        page.wait_for_function('window.'+viewer+' && window.'+viewer+'.loaded')
        data=page.evaluate('data');geometry=data['source_geometry'];entries=geometry['entries']
        full_model=(data['study']['model'] if viewer=='conductorEventViewer'
            else data['study']['model']['base'])
        assert [(e['source_id'],e['label']) for e in entries]==[tuple(a) for a in full_model['source_annotations']]
        assert len(entries)==len(full_model['spec']['field']['sources'])
        assert page.locator('.source-geometry-entry').count()==len(entries)
        assert page.locator('#source-geometry-scope').count()==1
        assert page.locator('#source-geometry').is_checked()
        # Instrument ONLY redraw calls: real canvas methods, not a DOM-only claim.
        page.evaluate("""() => {
          window.__geomCalls=[];
          for(const name of ['arc','lineTo','fillText']){
           const original=CanvasRenderingContext2D.prototype[name];
           CanvasRenderingContext2D.prototype[name]=function(...args){
             if(this.canvas.id==='scene')window.__geomCalls.push({name,args});
             return original.apply(this,args);
           };
          }
        }""")
        page.locator('#source-geometry').uncheck()
        pixels_off=page.locator('#scene').evaluate('c=>c.toDataURL()')
        page.evaluate('window.__geomCalls=[]')
        page.locator('#source-geometry').check()
        pixels_on=page.locator('#scene').evaluate('c=>c.toDataURL()')
        painted=page.evaluate('window.__geomCalls')
        assert pixels_on!=pixels_off,('toggle did not change painted source geometry',case)
        painted_labels=[str(c['args'][0]) for c in painted if c['name']=='fillText']
        for e in entries:
            label_found=any(e['source_id'] in v for v in painted_labels)
            if e['shape']=='omitted':
                assert not label_found,(case,'ghost source marker',e['source_id'])
                assert e['status']=='omitted-out-of-plane'
                assert len(e['bounds_uv'])==0
                assert 'full 3d field' in e['notes'].lower() or 'source field still active' in e['notes'].lower()
                assert 'omitted' in page.locator('#sources').inner_text().lower()
            elif e['shape'] in ('point','rod','sphere-slice','shell-slice','slice-tangent','ring-projection','disk-projection'):
                assert label_found,(case,'missing visible source label',e['source_id'])
        if case in ('coaxial-rings','event-rings','tilted-rings','tilted-event-rings'):
            assert [e['shape'] for e in entries]==['ring-projection']*2
            assert all(e['edge_on'] for e in entries)
            assert all(e['radius_m']==.3 for e in entries)
            assert any('ring (edge-on)' in x for x in painted_labels)
            assert not any(' · rod' in x for x in painted_labels)
            assert len([x for x in painted if x['name']=='lineTo'])>130
            if case in ('tilted-rings','tilted-event-rings'):
                # Fresh CanvasRenderingContext2D arc instrumentation, not a
                # DOM-only label test. Physical ring axis is world x, and
                # the qualified 45-degree normal is (0,1/sqrt(2),1/sqrt(2)).
                # Hence the true projected rim endpoints are at world-plane
                # coordinates center_uv +/- (0,R), NOT either R/sqrt(2) rim column.
                arcs=[c['args'] for c in painted if c['name']=='arc' and abs(c['args'][2]-4)<1e-8]
                expected=page.evaluate('''()+>'''.replace('+>','=>')+'''{
                  const e=data.source_geometry.entries;
                  return e.flatMap(r=>[[X(r.center_uv[0]),Y(r.center_uv[1]-r.radius_m)],
                                      [X(r.center_uv[0]),Y(r.center_uv[1]+r.radius_m)]]);
                }''')
                assert len(arcs)==len(expected),(case,'ring marker count',len(arcs),len(expected))
                for x,y in expected:
                    assert min(hypot(c[0]-x,c[1]-y) for c in arcs)<1e-5,(
                        case,'edge-on canvas endpoint deviates from true projected rim',(x,y),arcs)
                assert all(abs(hypot(*e['projected_major_axis_uv'])-.3)<1e-13 for e in entries)
                assert any(max(hypot(*b) for b in e['rim_basis_uv'])<.22 for e in entries)
        elif case=='mirror-rod':
            assert [e['shape'] for e in entries]==['omitted','omitted','rod']
            assert entries[0]['signed_normal_offset_m']==.18
            assert entries[1]['signed_normal_offset_m']==-.18
            assert any('rod:left · rod' in x for x in painted_labels)
        elif case=='offplane-rods':
            assert [e['shape'] for e in entries[:2]]==['omitted','omitted']
            assert entries[2]['shape']=='point'
        elif case=='ring-disk':
            assert [e['shape'] for e in entries]==['ring-projection','disk-projection']
            assert any('disc (edge-on)' in x for x in painted_labels)
        elif case=='hybrid-sources':
            assert [e['shape'] for e in entries]==['sphere-slice','rod','point','field-arrows']
        last=len(data['display_states'])-1
        page.locator('#frame').fill(str(last));page.locator('#frame').dispatch_event('input')
        assert page.evaluate('window.'+viewer+'.physicalTime')==data['display_states'][-1]['time_s']
        assert page.locator('#frame').input_value()==str(last)
        page.screenshot(path=str(out_dir/(case+'.png')),full_page=True)
        page.locator('#reset').click();assert page.locator('#frame').input_value()=='0'
        page.locator('#play').click()
        page.wait_for_function('window.'+viewer+'.frameIndex>=1',timeout=7000)
        if page.evaluate('window.'+viewer+'.playing'):page.locator('#play').click()
        assert page.locator('#source-geometry').is_checked()
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1')
        assert not browser_errors and not console_errors and not external_requests
        browser.close()
    return dict(case=case,browser='Chromium',source_inventory=len(entries),
        statuses=[{'source_id':e['source_id'],'shape':e['shape'],'status':e['status']} for e in entries],
        painted_source_geometry='passed',offplane_omissions='passed',saved_state_playback='passed',
        mobile_px=390,errors=browser_errors,console_errors=console_errors,requests=external_requests)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--case',choices=CASES,default='coaxial-rings')
    a.add_argument('--output',type=Path,default=Path('output/m25a'))
    args=a.parse_args();print(json.dumps(check(args.case,args.output),indent=2))
