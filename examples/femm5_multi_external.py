"""M9: two external point charges around one grounded or isolated solid sphere.

The slider plays stored equilibrium solutions; it is NOT time integration.
Default analytical output supports more general 3D source positions. FEMM is
strictly axisymmetric: all source sites must lie on a common central axis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from conductor_sim import (ElectrostaticProblem, ExternalSphereBackend,
                           Femm5ExternalSphereBackend, FixedNetCharge,
                           Grounded, PointCharge, SolidSphere)
from conductor_sim.visualization import SliceSpec, sample_sweep, write_visualization


def cases():
    charges=(PointCharge(1e-9,(0,0,.27)),PointCharge(-.5e-9,(0,0,-.32)))
    return [
        ('Two sources · grounded sphere',
         ElectrostaticProblem(charges,(SolidSphere(.12,Grounded(),name='metal'),))),
        ('Two sources · isolated neutral sphere',
         ElectrostaticProblem(charges,(SolidSphere(.12,FixedNetCharge(0),name='metal'),))),
        ('Two sources · isolated +1.2 nC sphere',
         ElectrostaticProblem(charges,(SolidSphere(.12,FixedNetCharge(1.2e-9),name='metal'),))),
        ('Two sources · isolated -0.8 nC sphere',
         ElectrostaticProblem(charges,(SolidSphere(.12,FixedNetCharge(-.8e-9),name='metal'),))),
    ]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('analytical','femm'),default='analytical')
    parser.add_argument('--resolution',type=int,default=41)
    parser.add_argument('--output',type=Path,default=Path('output/multi_external.html'))
    args=parser.parse_args()
    backend=Femm5ExternalSphereBackend() if args.backend=='femm' else ExternalSphereBackend()
    cases_to_render=cases()
    plane=SliceSpec.meridian(cases_to_render[0][1],extent_m=.46,resolution=args.resolution)
    bundle=sample_sweep(cases_to_render,backend,plane=plane,line_steps=320)
    bundle['source']='M9: two distinct external charges near grounded and isolated solid spheres'
    write_visualization(bundle,args.output,title=f'Multi-source conductor · {args.backend} · stored equilibria')
    args.output.with_suffix('.json').write_text(json.dumps(bundle,allow_nan=False),encoding='utf-8')
    reference=ExternalSphereBackend()
    print(f'{args.output}: {len(bundle["frames"])} stored equilibria; backend={args.backend}')
    for (label,problem),frame in zip(cases_to_render,bundle['frames']):
        ref=reference.solve(problem).conductor_results[0]
        measured=frame['conductor_results'][0]
        print(label)
        print(f"  {args.backend} conductor Q (nC): {measured['net_charge_c']*1e9:+.6f}; V (V): {measured['potential_v']:+.5f}")
        print(f'  Exact reference Q (nC): {ref.net_charge_c*1e9:+.6f}; V (V): {ref.potential_v:+.5f}')


if __name__=='__main__':
    main()
