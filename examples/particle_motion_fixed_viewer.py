"""M18b offline playback: one moving charge and every externally held source.

Compose with the M18a public-model examples; use the existing motion exporter.
No extra integrator, browser force law, or omission of held-source fields.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
from particle_motion_fixed_charges import compose
from conductor_sim import sample_motion_2d
from conductor_sim.visualization import write_visualization


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('plane','plates','sphere','cavity','nested','guard'),default='plane')
    parser.add_argument('--duration-s',type=float)
    parser.add_argument('--sample-interval-s',type=float)
    parser.add_argument('--resolution',type=int,default=25)
    parser.add_argument('--visual-frames',type=int,default=16)
    parser.add_argument('--line-steps',type=int,default=60)
    parser.add_argument('--output',type=Path,default=Path('output/m18b-held-sources.html'))
    args=parser.parse_args()
    spec,backend=compose(args.case)
    overrides={}
    if args.duration_s is not None:overrides['duration_s']=args.duration_s
    if args.sample_interval_s is not None:overrides['sample_interval_s']=args.sample_interval_s
    spec=replace(spec,**overrides)
    bundle=sample_motion_2d(spec,backend,resolution=args.resolution,
                            visual_frames=args.visual_frames,line_steps=args.line_steps)
    bundle['motion_2d']['example']=args.case
    write_visualization(bundle,args.output,title='M18b — '+args.case+' with held charges')
    args.output.with_suffix('.json').write_text(json.dumps(bundle,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(f'{args.output}: moving q{spec.charge_index}, {len(spec.stationary_charges)} held sources; '
          f'{len(bundle["motion_2d"]["samples"])} numerical samples; {len(bundle["frames"])} solved fields')


if __name__=='__main__':
    main()
