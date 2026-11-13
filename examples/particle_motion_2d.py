"""Generate a standalone M17a classical vector trajectory as JSON (no HTML)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from conductor_sim import (
    ConductingPlane, ElectrostaticProblem, ExternalSphereBackend,
    FixedPotential, Grounded, GroundedPlaneBackend, Motion2DSpec,
    MultiImageShellBackend, ParallelPlateBackend, PointCharge, SolidSphere,
    SphericalShell, integrate_motion_2d,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('plane','plates','sphere','cavity'),
                        default='plates')
    parser.add_argument('--output', type=Path,
                        default=Path('output/m17a-trajectory.json'))
    args = parser.parse_args()

    if args.case == 'plane':
        problem = ElectrostaticProblem((PointCharge(1e-9,(0.,0.,.1)),),
                                       (ConductingPlane(),))
        backend, velocity = GroundedPlaneBackend(),(.2,0.,-.03)
    elif args.case == 'plates':
        problem = ElectrostaticProblem((PointCharge(1e-9,(0.,0.,.1)),),
            (ConductingPlane(),
             ConductingPlane((0.,0.,.2),(0.,0.,-1.),
                             FixedPotential(20.),'upper')))
        backend, velocity = ParallelPlateBackend(),(.2,0.,.03)
    elif args.case == 'sphere':
        problem = ElectrostaticProblem((PointCharge(1e-9,(.18,0.,0.)),),
                                       (SolidSphere(.07,Grounded()),))
        backend, velocity = ExternalSphereBackend(),(.015,.13,.07)
    else:
        problem = ElectrostaticProblem((PointCharge(1e-9,(.03,0.,.01)),),
                                       (SphericalShell(.1,.12,Grounded()),))
        backend, velocity = MultiImageShellBackend(),(0.,.09,.03)
    spec = Motion2DSpec(problem, 0, 1e-6, velocity, .04, .005)
    result = integrate_motion_2d(spec, backend)
    data = result.as_dict()
    data['example'] = args.case
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',
                           encoding='utf-8')
    print(f'{args.case}: {len(result.samples)} numerical samples, '
          f'termination={result.termination}; wrote {args.output}')


if __name__ == '__main__':
    main()
