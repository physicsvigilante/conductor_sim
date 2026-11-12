"""M10: three heights of a point charge over an ideal grounded infinite plane.

Default uses the independent exact mirror-charge reference. --backend femm
solves a *truncated upper half-space* with actual FEMM 5; no silent fallback.
Each HTML frame is a separately precomputed equilibrium, not particle motion.
"""
from __future__ import annotations

import argparse
import json
from math import hypot
from pathlib import Path

from conductor_sim import (ConductingPlane, ElectrostaticProblem, Grounded,
                           PointCharge, GroundedPlaneBackend, Femm5PlaneBackend)
from conductor_sim.visualization import SliceSpec, sample_sweep, write_visualization


def cases():
    conductor=ConductingPlane(point_m=(0,0,0),normal=(0,0,1),
                              constraint=Grounded(),name="grounded plane")
    return [(f"+1 nC at {h*100:.0f} cm above grounded plane",
             ElectrostaticProblem((PointCharge(1e-9,(0.,0.,h)),),(conductor,)))
            for h in (.12,.18,.27)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('analytical','femm'),default='analytical')
    parser.add_argument('--resolution',type=int,default=31)
    parser.add_argument('--output',type=Path,default=Path('output/grounded_plane.html'))
    args=parser.parse_args()
    problems=cases()
    backend=GroundedPlaneBackend() if args.backend=='analytical' else Femm5PlaneBackend()
    # Explicit sweep extent covers the furthest state, not merely the first.
    plane=SliceSpec.meridian(problems[0][1],extent_m=.40,resolution=args.resolution)
    bundle=sample_sweep(problems,backend,plane=plane,line_steps=330)
    bundle['source']='M10: grounded infinite conducting plane with positive source'
    write_visualization(bundle,args.output,title='Grounded plane · mirror charge · '+args.backend)
    args.output.with_suffix('.json').write_text(json.dumps(bundle,allow_nan=False),encoding='utf-8')
    for (label,problem),frame in zip(problems,bundle['frames']):
        exact=GroundedPlaneBackend().solve(problem)
        measured=frame['conductor_results'][0]
        print(label)
        h=problem.charges[0].position_m[2]
        R=h*8.0  # matches default Femm5Settings.outer_radius_factor
        truncated_exact=-1e-9*(1-h/hypot(h,R))
        print('  backend',args.backend,'plane reported Q (nC)',
              f"{measured['net_charge_c']*1e9:.6f}",
              'V (V)',f"{measured['potential_v']:.6f}")
        if args.backend=='femm':
            print('  EXACT charge on only the modeled disk (rho <= R), nC:',
                  f'{truncated_exact*1e9:.6f}', 'R (m):', f'{R:.4f}')
        print('  ideal infinite-plane total Q (nC)',
              f'{exact.conductor_results[0].net_charge_c*1e9:.6f}')
        print('  exact force on the source (N)',exact.induced_force_on_charge_n)
    print('export:',args.output)


if __name__=='__main__':main()
