"""M8: compare a grounded sphere with neutral and net-charged isolated spheres.

Four separately solved static equilibria. --backend analytical (default) is
self-contained; --backend femm requires a genuine FEMM 5 TCP installation.
Neither branch replaces a failed FEMM solve with analytical values.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from conductor_sim import (
    ElectrostaticProblem, ExternalSphereBackend, Femm5ExternalSphereBackend,
    FixedNetCharge, Grounded, PointCharge, SolidSphere,
)
from conductor_sim.visualization import SliceSpec, sample_sweep, write_visualization


def cases():
    radius=.12
    source=PointCharge(1e-9,(0.,0.,.27))
    constraints=(
        ('Grounded (V=0; Q unknown)', Grounded()),
        ('Isolated neutral (Q=0 nC)', FixedNetCharge(0.)),
        ('Isolated +2 nC (repulsion can win)', FixedNetCharge(2e-9)),
        ('Isolated −0.6 nC (extra attraction)', FixedNetCharge(-.6e-9)),
    )
    return [(label,ElectrostaticProblem(
        charges=(source,),
        conductors=(SolidSphere(radius,constraint,name='physical solid sphere'),),
    )) for label,constraint in constraints]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('analytical','femm'),default='analytical')
    parser.add_argument('--resolution',type=int,default=31)
    parser.add_argument('--output',type=Path,default=Path('output/isolated_sphere.html'))
    args=parser.parse_args()
    samples=cases()
    backend=ExternalSphereBackend() if args.backend=='analytical' else Femm5ExternalSphereBackend()
    plane=SliceSpec.meridian(samples[0][1],extent_m=.47,resolution=args.resolution)
    bundle=sample_sweep(samples,backend,plane=plane,line_steps=260)
    bundle['source']='M8: external +1 nC and grounded/isolated solid sphere with different Q'
    write_visualization(bundle,args.output,title='Grounded vs isolated conducting sphere · '+args.backend)
    args.output.with_suffix('.json').write_text(json.dumps(bundle,allow_nan=False),encoding='utf-8')
    print(f'{args.output}: {len(bundle["frames"])} stored equilibria; backend={args.backend}')
    reference=ExternalSphereBackend()
    for (label,problem),frame in zip(samples,bundle['frames']):
        conductor=frame['conductor_results'][0]
        exact=reference.solve(problem)
        print(label)
        print('  '+args.backend+' sphere net charge (nC)',
              f'{conductor["net_charge_c"]*1e9:+.6f}',
              'potential (V)',f'{conductor["potential_v"]:+.4f}')
        print('  analytical reference net charge (nC)',
              f'{exact.conductor_results[0].net_charge_c*1e9:+.6f}',
              'potential (V)',f'{exact.conductor_results[0].potential_v:+.4f}')
        print('  exact force on source along +axis (N)',
              f'{exact.force_on_charge_n[2]:+.8e}')


if __name__=='__main__':
    main()
