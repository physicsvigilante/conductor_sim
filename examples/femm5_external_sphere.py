"""M7: external point charge approaching a grounded solid conducting sphere.

Default analytical view requires no FEMM install. --backend femm runs actual
FEMM 5; both export separately-labelled, PRECOMPUTED equilibria to offline HTML.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from conductor_sim import (
    ElectrostaticProblem, Grounded, PointCharge, SolidSphere,
    ExternalGroundedSphereBackend, Femm5ExternalSphereBackend,
)
from conductor_sim.visualization import SliceSpec, sample_sweep, write_visualization


def cases():
    sphere=SolidSphere(.12,Grounded(),name='grounded solid sphere')
    return [(f'+1 nC at {d:.2f} m from grounded sphere centre',
             ElectrostaticProblem(
                 charges=(PointCharge(1e-9,(0.,0.,d)),),conductors=(sphere,)))
            for d in (.21,.27,.34)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('analytical','femm'),default='analytical')
    parser.add_argument('--resolution',type=int,default=41)
    parser.add_argument('--output',type=Path,default=Path('output/external_sphere.html'))
    args=parser.parse_args()
    backend=(ExternalGroundedSphereBackend() if args.backend=='analytical'
             else Femm5ExternalSphereBackend())
    problems=cases()
    plane=SliceSpec.meridian(problems[0][1],extent_m=.48,resolution=args.resolution)
    bundle=sample_sweep(problems,backend,plane=plane,line_steps=350)
    bundle['source']='M7: external charge near grounded solid sphere'
    write_visualization(bundle,args.output,
                        title='Grounded sphere · external charge · '+args.backend)
    args.output.with_suffix('.json').write_text(json.dumps(bundle,allow_nan=False),encoding='utf-8')
    reference=ExternalGroundedSphereBackend()
    print('Exported',len(bundle['frames']),'precomputed frames to',args.output)
    for (label,problem),frame in zip(problems,bundle['frames']):
        exact=reference.solve(problem)
        induced_exact=exact.conductor_results[0].net_charge_c
        induced_computed=frame['conductor_results'][0]['net_charge_c']
        potential_computed=frame['conductor_results'][0]['potential_v']
        print(label)
        print(' ',args.backend,'sphere net charge (nC)',f'{induced_computed*1e9:.9f}',
              'potential (V)',f'{potential_computed:.6f}')
        print('  analytical expected net charge (nC)',f'{induced_exact*1e9:.9f}',
              'exact force on charge along +axis (N)',
              f'{exact.induced_force_on_charge_n[2]:.9e}')


if __name__=='__main__':
    main()
