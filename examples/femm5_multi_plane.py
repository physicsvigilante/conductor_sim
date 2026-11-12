"""M11: several point charges along one normal above a grounded plane.

The exact image backend supports arbitrary 3D locations. FEMM requires that
all active sites share a normal-line footpoint, but handles each signed source
with an independent finite charged sphere. The HTML slider selects stored
static equilibria; it never launches a solver in the browser.
"""
from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path

from conductor_sim import (ConductingPlane, ElectrostaticProblem, Femm5PlaneBackend,
                           GroundedPlaneBackend, PointCharge)
from conductor_sim.visualization import SliceSpec, sample_sweep, write_visualization


def cases():
    plane=ConductingPlane(name='grounded conducting half-space')
    scenarios=(
        ('Two positive sources', (1e-9,.5e-9),(.12,.26)),
        ('Opposite signs, nonzero total', (1e-9,-.5e-9),(.12,.26)),
        ('Zero-net signed dipole', (1e-9,-1e-9),(.12,.26)),
    )
    return [(label,ElectrostaticProblem(
        tuple(PointCharge(q,(0.,0.,h)) for q,h in zip(qs,heights)),(plane,)))
            for label,qs,heights in scenarios]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('analytical','femm'),default='analytical')
    parser.add_argument('--resolution',type=int,default=31)
    parser.add_argument('--output',type=Path,default=Path('output/multi_plane.html'))
    args=parser.parse_args()
    states=cases()
    backend=Femm5PlaneBackend() if args.backend=='femm' else GroundedPlaneBackend()
    plane=SliceSpec.meridian(states[0][1],extent_m=.52,resolution=args.resolution)
    bundle=sample_sweep(states,backend,plane=plane,line_steps=300)
    bundle['source']='M11: independent source cores above one grounded infinite conducting plane'
    write_visualization(bundle,args.output,title='Multi-source grounded plane · '+args.backend)
    args.output.with_suffix('.json').write_text(json.dumps(bundle,allow_nan=False),encoding='utf-8')
    settings=backend.settings if args.backend=='femm' else None
    for (label,problem),frame in zip(states,bundle['frames']):
        q_total=sum(s.charge_c for s in problem.charges)
        output=frame['conductor_results'][0]
        print(label)
        print(' ',args.backend,'measured/reference physical conductor Q (nC)',
              f'{output["net_charge_c"]*1e9:+.6f}','V (V)',f'{output["potential_v"]:+.6f}')
        if settings is not None:
            R=max(s.position_m[2] for s in problem.charges)*settings.outer_radius_factor
            disk=-sum(s.charge_c*(1-s.position_m[2]/sqrt(s.position_m[2]**2+R**2))
                      for s in problem.charges)
            print('  independent ideal image-charge integral on ONLY modeled disk:',
                  f'{disk*1e9:+.6f} nC; R={R:.3f} m')
        print('  exact INFINITE plane total induced charge:',f'{-q_total*1e9:+.6f} nC')
    print('Exported',len(bundle['frames']),'stored equilibria:',args.output)


if __name__=='__main__':
    main()
