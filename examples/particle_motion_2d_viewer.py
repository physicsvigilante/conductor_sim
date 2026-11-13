"""M17b standalone offline time playback of a true curved M17a trajectory.

No browser-side force calculation or integration. Every displayed electrostatic
field is freshly solved at one recorded M17a trajectory time and position.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from conductor_sim import (
    ConductingPlane, ElectrostaticProblem, ExternalSphereBackend, FixedPotential,
    Grounded, GroundedPlaneBackend, Motion2DSpec, MultiImageShellBackend,
    ParallelPlateBackend, PointCharge, SolidSphere, SphericalShell,
    integrate_motion_2d, sample_motion_2d,
)
from conductor_sim.visualization import write_visualization


def make_case(name, charge_c, mass_kg, duration_s, sample_interval_s):
    if name == 'plane':
        problem = ElectrostaticProblem((PointCharge(charge_c,(0.,0.,.1)),),
                                       (ConductingPlane(),))
        backend, velocity = GroundedPlaneBackend(),(.2,0.,-.03)
    elif name == 'plates':
        problem = ElectrostaticProblem((PointCharge(charge_c,(0.,0.,.1)),),
            (ConductingPlane(name='lower'),
             ConductingPlane((0.,0.,.2),(0.,0.,-1.),
                             FixedPotential(20.),'upper')))
        backend, velocity = ParallelPlateBackend(image_order=48),(.2,0.,.03)
    elif name == 'sphere':
        problem = ElectrostaticProblem((PointCharge(charge_c,(.18,0.,0.)),),
                                       (SolidSphere(.07,Grounded()),))
        backend, velocity = ExternalSphereBackend(),(.015,.13,.07)
    elif name == 'cavity':
        problem = ElectrostaticProblem((PointCharge(charge_c,(.03,0.,.01)),),
                                       (SphericalShell(.1,.12,Grounded()),))
        backend, velocity = MultiImageShellBackend(),(0.,.09,.03)
    elif name == 'center':
        problem = ElectrostaticProblem((PointCharge(charge_c,(0.,0.,0.)),),
                                       (SphericalShell(.1,.12,Grounded()),))
        backend, velocity = MultiImageShellBackend(),(0.,0.,0.)
    else:
        problem = ElectrostaticProblem((PointCharge(charge_c,(0.,0.,.02)),),
                                       (ConductingPlane(),))
        backend, velocity = GroundedPlaneBackend(),(.02,0.,-.5)
    spec = Motion2DSpec(problem,0,mass_kg,velocity,duration_s,sample_interval_s,
                        surface_guard_m=1e-4 if name=='guard' else None)
    return spec, backend


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('plane','plates','sphere','cavity','center','guard'),
                        default='plates')
    parser.add_argument('--charge-c',type=float,default=1e-9)
    parser.add_argument('--mass-kg',type=float,default=1e-6)
    parser.add_argument('--duration-s',type=float,default=.04)
    parser.add_argument('--sample-interval-s',type=float,default=.005)
    parser.add_argument('--visual-frames',type=int,default=16)
    parser.add_argument('--resolution',type=int,default=25)
    parser.add_argument('--line-steps',type=int,default=60)
    parser.add_argument('--output',type=Path,default=Path('output/m17b-curved.html'))
    args=parser.parse_args()
    spec,backend=make_case(args.case,args.charge_c,args.mass_kg,
                           args.duration_s,args.sample_interval_s)
    trajectory=integrate_motion_2d(spec,backend)
    bundle=sample_motion_2d(spec,backend,result=trajectory,
                            visual_frames=args.visual_frames,
                            resolution=args.resolution,line_steps=args.line_steps)
    bundle['motion_2d']['example']=args.case
    output=args.output
    output.parent.mkdir(parents=True,exist_ok=True)
    write_visualization(bundle,output,title='M17b — '+args.case+' curved electrostatic motion')
    output.with_suffix('.json').write_text(json.dumps(bundle,indent=2,allow_nan=False)+'\n',
                                          encoding='utf-8')
    print(f'{args.case}: {len(trajectory.samples)} physical-time numerical samples, '
          f'{len(bundle["frames"])} solved field snapshots, termination={trajectory.termination}; '
          f'wrote {output} and {output.with_suffix(".json")}')


if __name__=='__main__':
    main()
