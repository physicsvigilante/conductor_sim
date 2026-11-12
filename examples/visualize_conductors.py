"""Export offline visualization: python examples/visualize_conductors.py --output output/view.html

Default: three analytical states, no FEMM needed. --backend femm uses actual
FEMM 5 and requires FEMM5_BINDING. Sliders play precomputed states, not solves.
"""
import argparse
import json
from pathlib import Path
from conductor_sim import (ElectrostaticProblem,FixedNetCharge,Grounded,PointCharge,
                           SphericalShell,NestedShellBackend,Femm5Backend)
from conductor_sim.visualization import SliceSpec,sample_sweep,write_visualization


def cases():
    shells=(SphericalShell(.1,.12,FixedNetCharge(0),name='inner'),
            SphericalShell(.2,.22,Grounded(),name='outer'))
    return [(f'Charge displacement {d*100:g} cm · neutral inner / grounded outer',
             ElectrostaticProblem((PointCharge(1e-9,(0,0,d)),PointCharge(.45e-9,(0,0,-.025))),shells))
            for d in (.0,.025,.05)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['analytical','femm'],default='analytical')
    parser.add_argument('--resolution',type=int,default=61)
    parser.add_argument('--output',type=Path,default=Path('output/conductors.html'))
    args=parser.parse_args()
    backend=Femm5Backend() if args.backend=='femm' else NestedShellBackend()
    states=cases()
    plane=SliceSpec.meridian(states[0][1],resolution=args.resolution,extent_m=.3)
    bundle=sample_sweep(states,backend,plane=plane)
    title='Nested conductors · '+('FEMM 5 numerical samples' if args.backend=='femm' else 'analytical reference')
    write_visualization(bundle,args.output,title=title)
    args.output.with_suffix('.json').write_text(json.dumps(bundle,allow_nan=False),encoding='utf-8')
    print(f'{args.output}: {len(bundle["frames"])} states; backend={args.backend}')
    for frame in bundle['frames']:
        print(frame['label'], 'lines=',len(frame['field_lines']), 'terminations=',frame['line_termination_counts'])


if __name__=='__main__':main()
