"""M19e teaching preview: physical ring A/B identity and world-horizontal axis.

Inspired by the geometry of 02_conductors_half_size.pdf page 12, Q40–41.
These are explicitly TEACHER HYPOTHETICALS, not a reconstruction of free
motion under the source's upward field and unspecified particle mass.
The rings carry fixed uniform source charges, not conducting-ring induction.

Examples:
  python examples/ring_source_orientation.py --case field-off --output output/rings.html
  python examples/ring_source_orientation.py --case axial-field --output output/rings-E.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from conductor_sim import (
    PointCharge, PrescribedField, TestChargeMotionSpec,
    UniformElectricField, UniformRingAxisField, integrate_test_charge,
    sample_test_charge_viewer,
)
from conductor_sim.visualization import write_visualization


# These source IDs belong to the teaching author; the viewer displays and
# validates their bindings but does not infer identities or scenario lineage.
SOURCE_ANNOTATIONS = (
    ('ring:A', 'Ring A'), ('ring:B', 'Ring B'),
    ('field:external', 'External field'),
)


def make_case(case='field-off'):
    if case not in ('field-off', 'axial-field'):
        raise ValueError('unknown teacher hypothetical')
    e_x = 0. if case == 'field-off' else 100000.
    sources = (
        UniformRingAxisField(+10e-6, .30, axis=(1., 0., 0.)),
        UniformRingAxisField(-20e-6, .30, center_m=(.40, 0., 0.), axis=(1., 0., 0.)),
        UniformElectricField((e_x, 0., 0.)),
    )
    return TestChargeMotionSpec(
        PrescribedField(sources), PointCharge(+10e-6, (0., 0., 0.)),
        .10, (0., 0., 0.), .25, .002, length_scale_m=.30,
        source_guard_m=1e-5,
    )


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=('field-off','axial-field'),default='field-off')
    p.add_argument('--axis-orientation',choices=('horizontal','vertical'),default='horizontal')
    p.add_argument('--output',type=Path,default=Path('output/rings-horizontal.html'))
    args=p.parse_args()
    spec=make_case(args.case)
    result=integrate_test_charge(spec)
    bundle=sample_test_charge_viewer(
        spec, result=result, visual_frames=12, profile_samples=101,
        resolution=15, axis_orientation=args.axis_orientation,
        source_annotations=SOURCE_ANNOTATIONS,
    )
    # All physical source values are supplied to the existing M19a backend;
    # these notes are narrative provenance only and do not modify the physics.
    note=('Teacher hypothetical: original workbook has an upward field and '
          'symbolic particle mass, so unconstrained axial motion is unsupported. '
          'Here the mass is set to 0.10 kg and the external field is '+
          ('switched off.' if args.case=='field-off' else 'applied along the x-axis.'))
    for frame in bundle['frames']:
        frame['notes'].append(note)
    bundle['prescribed_motion']['example']=args.case
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_visualization(bundle,args.output,
                        title='M19e: Ring A/B — teacher hypothetical: '+args.case)
    sidecar=args.output.with_suffix('.json')
    sidecar.write_text(json.dumps(bundle,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(f'{args.case}: {len(result.samples)} physical-time samples; '
          f'{len(bundle["frames"])} real display frames; '
          f'termination {result.termination}')
    print(f'Wrote {args.output} and {sidecar} (standalone, offline, no new physics).')


if __name__=='__main__':
    main()
