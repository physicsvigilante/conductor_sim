"""M19b offline fixed-source test-charge viewer (HTML + matching JSON).

Examples:
  python examples/test_charge_viewer.py --case ring-turning --output output/m19b-ring.html
  python examples/test_charge_viewer.py --case uniform-deflection --output output/m19b-uniform.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from conductor_sim import sample_test_charge_viewer
from conductor_sim.visualization import write_visualization
from test_charge_motion import CASES, make_case


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=CASES,default='ring-turning')
    parser.add_argument('--visual-frames',type=int,default=12)
    parser.add_argument('--profile-samples',type=int,default=201)
    parser.add_argument('--resolution',type=int,default=17)
    parser.add_argument('--output',type=Path,default=Path('output/m19b-ring.html'))
    args=parser.parse_args()
    spec=make_case(args.case)
    bundle=sample_test_charge_viewer(spec,visual_frames=args.visual_frames,
                                     profile_samples=args.profile_samples,
                                     resolution=args.resolution)
    bundle['prescribed_motion']['example']=args.case
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_visualization(bundle,args.output,title='M19b — '+args.case.replace('-',' '))
    path=args.output.with_suffix('.json')
    path.write_text(json.dumps(bundle,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(f'{args.case}: {len(bundle["prescribed_motion"]["samples"])} real physics samples, '
          f'{len(bundle["frames"])} synced display frames, '
          f'{bundle["prescribed_motion"]["profile_mode"]} E/V graphs, '
          f'termination={bundle["prescribed_motion"]["termination"]}')
    print(f'Wrote standalone offline {args.output} and {path}; no induced conductor charge or browser ODE.')


if __name__=='__main__':
    main()
