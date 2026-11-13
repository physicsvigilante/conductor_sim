"""M16b standalone offline HTML time playback from real M16a numerical dynamics.

No browser-side physics integration: the actual M16a trajectory is computed
once, field snapshots are solved independently at selected *saved* time rows.
Examples are single charged point particles; labels do not claim exact impact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from conductor_sim import Motion1DSpec, integrate_motion_1d
from conductor_sim.motion_viewer import sample_motion_1d
from conductor_sim.visualization import write_visualization
from particle_motion_1d import scenario


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('plane', 'plates', 'cavity', 'sphere-exterior', 'turning'),
                        default='cavity')
    parser.add_argument('--mass-kg', type=float, default=1e-6)
    parser.add_argument('--charge-c', type=float, default=1e-9)
    parser.add_argument('--duration-s', type=float, default=.15)
    parser.add_argument('--sample-interval-s', type=float, default=.015)
    parser.add_argument('--initial-velocity-m-s', type=float, default=None)
    parser.add_argument('--surface-guard-m', type=float, default=None)
    parser.add_argument('--resolution', type=int, default=25)
    parser.add_argument('--line-steps', type=int, default=75)
    parser.add_argument('--visual-frames', type=int, default=24)
    parser.add_argument('--output', type=Path, default=Path('output/particle_motion.html'))
    args = parser.parse_args()
    problem, backend, axis, initial_v = scenario(args.case, args.charge_c)
    spec = Motion1DSpec(problem, 0, args.mass_kg, axis,
                        initial_v if args.initial_velocity_m_s is None
                        else args.initial_velocity_m_s,
                        args.duration_s, args.sample_interval_s,
                        surface_guard_m=args.surface_guard_m)
    result = integrate_motion_1d(spec, backend)
    bundle = sample_motion_1d(spec, backend, result=result,
                              resolution=args.resolution, line_steps=args.line_steps,
                              visual_frames=args.visual_frames)
    bundle['motion_1d']['example'] = args.case
    file = write_visualization(bundle, args.output,
                               title='Elapsed-time charged particle · ' + args.case)
    file.with_suffix('.json').write_text(json.dumps(bundle, indent=2, allow_nan=False)+'\n',
                                         encoding='utf-8')
    last = result.samples[-1]
    print('Offline M16b time viewer:',file)
    print(f'Actual numerical samples: {len(result.samples)}; displayed field snapshots: {len(bundle["frames"])}')
    print(f'End t={last.time_s:.6g} s; v={last.velocity_m_s:.6g} m/s; '+
          f'termination={result.termination}; events={[e.kind for e in result.events]}')
    print('Full data:',file.with_suffix('.json'))
    print('Offline playback is indexed by physical time; no browser-side ODE or exact impact claim.')


if __name__=='__main__':
    main()
