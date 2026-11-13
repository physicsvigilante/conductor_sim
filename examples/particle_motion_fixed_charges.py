"""M18a reusable time trajectories: one charge moves, all others are held fixed.

The original model and existing analytical force/energy backend determine the
physics. Only JSON numerical time samples are exported; M18b will add a viewer
that can display *all* authored sources without dropping held charge sites.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from conductor_sim import (
    ConductingPlane, ElectrostaticProblem, ExternalSphereBackend, FixedNetCharge,
    FixedPotential, Grounded, GroundedPlaneBackend, Motion2DSpec,
    MultiImageShellBackend, NestedShellBackend, ParallelPlateBackend,
    PointCharge, SolidSphere, SphericalShell, integrate_motion_2d,
)


def compose(case):
    """Compose a new arrangement from unchanged public physics components."""
    if case == 'plane':
        charges=(PointCharge(-.5e-9,(.05,0.,.12)), PointCharge(1e-9,(0.,0.,.1)))
        conductors=(ConductingPlane(),)
        backend=GroundedPlaneBackend()
        moving_index,velocity,duration=1,(.12,0.,-.01),.03
    elif case == 'plates':
        charges=(PointCharge(1e-10,(0.,0.,.1)),PointCharge(1e-10,(.04,0.,.12)))
        conductors=(ConductingPlane(name='lower'),
                    ConductingPlane((0.,0.,.2),(0.,0.,-1.),FixedPotential(20.),'upper'))
        backend=ParallelPlateBackend(image_order=32)
        moving_index,velocity,duration=0,(.02,0.,.01),.03
    elif case == 'sphere':
        charges=(PointCharge(-1e-9,(.18,0.,0.)),PointCharge(1e-9,(.23,.05,0.)))
        conductors=(SolidSphere(.07,FixedNetCharge(1.6e-9)),)
        backend=ExternalSphereBackend()
        moving_index,velocity,duration=0,(.02,.04,0.),.03
    elif case == 'cavity':
        charges=(PointCharge(1e-9,(.025,0.,0.)),PointCharge(-.5e-9,(-.03,.02,0.)))
        conductors=(SphericalShell(.1,.12,Grounded()),)
        backend=MultiImageShellBackend()
        moving_index,velocity,duration=0,(.02,.03,0.),.03
    elif case == 'nested':
        charges=(PointCharge(1e-9,(.025,0.,0.)),PointCharge(-.5e-9,(-.03,.02,0.)))
        conductors=(SphericalShell(.1,.12,FixedNetCharge(.2e-9),name='inner'),
                    SphericalShell(.2,.22,Grounded(),name='outer'))
        backend=NestedShellBackend()
        moving_index,velocity,duration=0,(.02,.03,0.),.03
    else:
        charges=(PointCharge(-1e-10,(0.,0.,.1)),PointCharge(1e-10,(.03,0.,.1)))
        conductors=(ConductingPlane(),)
        backend=GroundedPlaneBackend()
        moving_index,velocity,duration=0,(.1,0.,0.),.5
    kwargs={'charge_guard_m':.005} if case == 'guard' else {}
    problem=ElectrostaticProblem(charges,conductors)
    spec=Motion2DSpec(problem,moving_index,1e-6,velocity,duration,.005,**kwargs)
    return spec,backend


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',default='plane',
                        choices=('plane','plates','sphere','cavity','nested','guard'))
    parser.add_argument('--output',type=Path,
                        default=Path('output/m18a-fixed-charges.json'))
    args=parser.parse_args()
    spec,backend=compose(args.case)
    result=integrate_motion_2d(spec,backend)
    data=result.as_dict()
    data['example']=args.case
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(f'{args.case}: moving index {spec.charge_index}; '
          f'{len(spec.stationary_charges)} held charges; '
          f'{len(result.samples)} physical-time samples; termination={result.termination}; '
          f'output={args.output}')


if __name__=='__main__':
    main()
