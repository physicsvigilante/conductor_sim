"""M19a prescribed-field trajectories; this milestone exports physics JSON.

There is no conductor back-reaction. Rings/discs are axis-only; a uniform
field is a prescribed idealization, not a simulation of finite electrodes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from conductor_sim import (PointCharge, PrescribedField, UniformElectricField,
    UniformRingAxisField, UniformDiskAxisField, UniformRodField,
    UniformSphereField, UniformShellField, TestChargeMotionSpec, integrate_test_charge)


CASES = ('ring-turning','ring-crossing','two-rings','uniform-deflection',
         'sphere-transit','shell-interior','disc-approach','rod-deflection')


def make_case(name):
    if name not in CASES:
        raise ValueError(f'unknown case: {name}')
    charge,position,velocity,duration = 1e-9,(0.,0.,.2),(0.,0.,-.2),1.4
    if name.startswith('ring'):
        sources=(UniformRingAxisField(1e-9,.1),)
        if name=='ring-crossing':
            charge=-1e-9
    elif name=='two-rings':
        sources=(UniformRingAxisField(1e-9,.1),
                 UniformRingAxisField(-.4e-9,.08,(0.,0.,-.15)))
    elif name=='uniform-deflection':
        sources=(UniformElectricField((0.,0.,-100.)),)
        position,velocity,duration=(0.,0.,0.),(.2,0.,.03),.4
    elif name=='sphere-transit':
        sources=(UniformSphereField(1e-9,.1),)
        charge,position,velocity=-1e-9,(-.15,0.,0.),(.3,0.,0.)
    elif name=='shell-interior':
        sources=(UniformShellField(1e-9,.1),)
        position,velocity,duration=(0.,0.,0.),(.1,0.,0.),.5
    elif name=='disc-approach':
        sources=(UniformDiskAxisField(1e-9,.1),)
        charge,position,velocity,duration=-1e-9,(0.,0.,.1),(0.,0.,-.1),1.
    else:
        sources=(UniformRodField(1e-9,(-.05,0.,0.),(.05,0.,0.)),)
        charge,position,velocity,duration=-1e-9,(0.,0.,.15),(.2,0.,0.),.2
    return TestChargeMotionSpec(PrescribedField(sources),PointCharge(charge,position),
                               1e-6,velocity,duration,.01,length_scale_m=.1,source_guard_m=1e-4)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=CASES,default='ring-turning')
    parser.add_argument('--output',type=Path,default=Path('output/m19a-ring.json'))
    args=parser.parse_args()
    result=integrate_test_charge(make_case(args.case))
    data=result.as_dict()
    data['example']=args.case
    text=json.dumps(data,indent=2,allow_nan=False)+'\n'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(text,encoding='utf-8')
    print(f'{args.case}: {len(result.samples)} physical-time samples; '
          f'termination={result.termination}; max energy drift={result.max_abs_energy_drift_j:.6g} J')
    print(f'Fixed sources; no conductor induction. Wrote {args.output}')


if __name__=='__main__':
    main()
