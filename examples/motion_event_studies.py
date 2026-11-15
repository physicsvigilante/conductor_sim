"""M24a: source-checked sheet/plate events and reusable teacher investigations.

No case-specific production physics or event detector. R is a separately
preselected, developer-exposed transfer executed only after candidate freeze.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from math import sqrt
from pathlib import Path

from conductor_sim import (EPSILON_0, PrescribedField, UniformElectricField,
    UniformGravity, UniformSphereField, FixedPointField, PointCharge,
    TestChargeMotionSpec)
from conductor_sim.scenarios import PrescribedScenario
from conductor_sim.motion_events import (PlaneEvent, VelocityReversal, EventPolicy,
                                         EventStudy, investigate_motion)

CASES=('sheet-stop','sheet-encounter','plate-transit','oblique-gravity','source-guard')
ELECTRON_Q=1.602176634e-19
ELECTRON_M=9.1093837139e-31


def make_case(case):
    if case in ('sheet-stop','sheet-encounter'):
        # Original p2 Q19 givens: 100 eV electron, metal surface sigma=-2e-6 C/m2.
        # One-sided ideal metal surface field is sigma/epsilon0, NOT sigma/2epsilon0.
        d=.001 if case=='sheet-stop' else .0002
        field=PrescribedField((UniformElectricField((-2e-6/EPSILON_0,0,0)),))
        particle=PointCharge(-ELECTRON_Q,(d,0,0));mass=ELECTRON_M
        velocity=(-sqrt(2*100*ELECTRON_Q/mass),0,0);duration=4e-10;interval=2e-11
        gravity=None
        questions=(VelocityReversal('stop:x',(1,0,0),'Electron stopping/reversal',1,case=='sheet-stop'),
            PlaneEvent('sheet',(0,0,0),(1,0,0),'Ideal metal surface x=0',-1,True,'ideal-stop-surface'))
        pol=EventPolicy(1e-11,1e-18,1e-12,1e-3)
        provenance=(f'Workbook p2 Q19: given 100 eV electron, sigma=-2e-6 C/m2; '
            f'ideal prescribed metal-surface field. Starting distance {d} m is teacher-chosen. '
            'No image-force response. Stopping distance is measured from the located event, not supplied as input.')
    elif case=='plate-transit':
        # Original p3 Q30 has symbolic d,V; these numerical choices are illustrative.
        d=.01;voltage=100.
        field=PrescribedField((UniformElectricField((0,0,-voltage/d)),))
        particle=PointCharge(-ELECTRON_Q,(0,0,0));mass=ELECTRON_M
        velocity=(0,0,0);duration=1e-8;interval=5e-10;gravity=None
        questions=(PlaneEvent('lower',(0,0,0),(0,0,1),'Lower launch plate',-1,True,'ideal-stop-surface'),
            PlaneEvent('upper',(0,0,d),(0,0,1),'Upper plate first encounter',1,True,'ideal-stop-surface'),
            VelocityReversal('vertical',(0,0,1),'Velocity reversal'))
        pol=EventPolicy(2e-10,1e-17,1e-10,1e-3)
        provenance=('Workbook p3 Q30: electron released from lower plate; source d and V symbolic. '
            'Illustrative d=.01 m,V=100 V, initial rest. Prescribed uniform capacitor approximation, '
            'no fringe field, induction, emission mechanism or post-encounter physics.')
    elif case=='oblique-gravity':
        field=PrescribedField((UniformElectricField((2000,0,0)),))
        particle=PointCharge(1e-9,(.1,0,.2));mass=1e-6
        velocity=(.2,0,1.2);duration=1.;interval=.04;gravity=UniformGravity((0,0,-3))
        questions=(VelocityReversal('apex',(0,0,1),'Vertical apex, NOT total rest',-1),
            PlaneEvent('right',(.5,0,0),(1,0,0),'Right passive detector',1,True),
            PlaneEvent('lower',(0,0,0),(0,0,1),'Lower passive detector',-1,True))
        pol=EventPolicy(.02,1e-9,1e-9,1e-8)
        provenance=('Teacher hypothetical: uniform E=(2000,0,0) V/m plus independent gravity '
            '(0,0,-3) m/s2, oblique launch. Several event questions share one unchanged model; '
            'detectors exert no forces and termination only ends observation.')
    elif case=='source-guard':
        field=PrescribedField((FixedPointField(-1e-9),))
        particle=PointCharge(1e-9,(.1,0,0));mass=1e-6
        velocity=(-1,0,0);duration=.5;interval=.01;gravity=None
        questions=(PlaneEvent('behind',(.005,0,0),(1,0,0),'Detector behind numerical guard',-1,True),)
        pol=EventPolicy(.005,1e-9,1e-9,1e-8)
        provenance='Teacher hypothetical: attractive fixed point with detector inside source guard; no contact or unseen event assertion.'
    elif case=='reserved-transfer':
        # Frozen in docs/m24a_charter.md BEFORE implementation; do not execute in development.
        field=PrescribedField((UniformSphereField(1e-9,1.),UniformElectricField((.3,0,-.2))))
        particle=PointCharge(-1e-9,(.2,0,.1));mass=1e-9
        velocity=(.1,0,-.03);duration=.8;interval=.1;gravity=UniformGravity((0,0,-.1))
        questions=(VelocityReversal('turn:x',(1,0,0),'x component reversal',-1),
            PlaneEvent('downward',(0,0,.02),(0,0,1),'First downward z=.02 crossing',-1,True))
        pol=EventPolicy(.01,1e-9,1e-9,1e-8)
        provenance='Frozen reserved M24a transfer: fixed volume sphere + uniform E + gravity; same-author exposed, not blind.'
    else:raise ValueError('unknown event study')
    spec=TestChargeMotionSpec(field,particle,mass,velocity,duration,interval,
        plane_normal=(0,1,0),length_scale_m=.01 if 'sheet' in case or 'plate' in case else 1.,
        relative_tolerance=1e-9,source_guard_m=.01 if case=='source-guard' else 1e-6,
        gravity=gravity)
    scenario=PrescribedScenario('m24a:'+case,spec,
        tuple((f'source:{i}',type(source).__name__) for i,source in enumerate(field.sources)),provenance)
    return EventStudy(scenario,questions,pol)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case',choices=CASES+('reserved-transfer',),default='sheet-stop')
    ap.add_argument('--output',type=Path,default=Path('output/m24a-events.html'))
    args=ap.parse_args()
    from conductor_sim.motion_event_viewer import render_event_run
    result=investigate_motion(make_case(args.case))
    data=result.as_dict()
    if args.case=='sheet-stop' and result.events:
        data['derived_stopping_distance_m']=result.study.scenario.spec.particle.position_m[0]-result.events[0].sample.position_m[0]
        data['inference_scope']='Constant ideal surface field only: numerical stopping displacement gives limiting launch distance; equality is not impact.'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(render_event_run(result,title='M24a · '+args.case),encoding='utf-8')
    args.output.with_suffix('.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(args.case,result.trajectory.termination,[(e.event_id,e.sample.time_s) for e in result.events])

if __name__=='__main__':main()
