"""M20a: compare two explicitly authored ring A/B teaching hypotheticals.

One original physical inventory, independently integrated two times with the
existing M19a solver and rendered by the unchanged M19e single-viewer.
This does NOT simulate the original workbook's off-axis free motion; its
upward applied field and unspecified mass are deliberately excluded.

From the repository root:
  python examples/compare_prescribed_scenarios.py --case axial-field --output output/compare-ring-fields.html
  python examples/compare_prescribed_scenarios.py --case double-mass --output output/compare-ring-mass.html
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from conductor_sim import (PrescribedField, integrate_test_charge,
                           sample_test_charge_viewer, write_scenario_comparison)
# As a module, `examples` is importable from the repo root. When this file is
# executed directly, Python puts only this examples/ directory on sys.path;
# use the sibling import without requiring an unstated PYTHONPATH setting.
if __package__:
    from examples.ring_source_orientation import make_case, SOURCE_ANNOTATIONS
else:
    from ring_source_orientation import make_case, SOURCE_ANNOTATIONS


def make_pair(case='axial-field'):
    """Two independent specimens with author-preserved source identities."""
    baseline=make_case('field-off')
    if case=='axial-field':
        variation=make_case('axial-field')
        note='Teacher hypothetical: change prescribed external E from zero to +100000 V/m along world x.'
    elif case=='double-mass':
        variation=replace(baseline,mass_kg=2*baseline.mass_kg)
        note='Teacher hypothetical: double the test particle mass while retaining every source and the initial state.'
    elif case=='ring-b-shift':
        originals=list(baseline.field.sources)
        originals[1]=replace(originals[1],center_m=(.45,0.,0.))
        variation=replace(baseline,field=PrescribedField(tuple(originals)))
        note='Teacher hypothetical: shift Ring B from world x=0.40 m to x=0.45 m.'
    elif case=='gauge-shift':
        originals=list(baseline.field.sources)
        originals[2]=replace(originals[2],reference_potential_v=17.)
        variation=replace(baseline,field=PrescribedField(tuple(originals)))
        note='Teacher hypothetical: change reference potential by +17 V; no electric field, force or trajectory change.'
    else:
        raise ValueError('unsupported example case')
    return baseline,variation,note


def make_bundles(case='axial-field', *,frames=12,profile_samples=101,resolution=15):
    baseline,variation,note=make_pair(case)
    def viewer(spec):
        result=integrate_test_charge(spec)
        return sample_test_charge_viewer(spec,result=result,visual_frames=frames,
                                         profile_samples=profile_samples,
                                         resolution=resolution,extent_m=2.25,axis_orientation='horizontal',
                                         source_annotations=SOURCE_ANNOTATIONS)
    return viewer(baseline),viewer(variation),note


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('axial-field','double-mass','ring-b-shift','gauge-shift'),default='axial-field')
    parser.add_argument('--output',type=Path,default=Path('output/m20a-ring-comparison.html'))
    parser.add_argument('--frames',type=int,default=12)
    args=parser.parse_args()
    left,right,note=make_bundles(args.case,frames=args.frames)
    write_scenario_comparison(left,right,args.output,
        title='M20a · Ring A/B physical scenarios: '+args.case,
        baseline_id='ring-AB-base',variation_id='ring-AB-'+args.case,
        baseline_title='Baseline · external field off',
        variation_title='Teacher revision · '+args.case,
        baseline_provenance='Teacher hypothetical: rings of page 12 Q40–41; explicit mass 0.10 kg, axial motion, no transverse applied field.',
        variation_provenance=note)
    print('Wrote '+str(args.output))
    print('Both original trajectories retain exact verified numerical samples, '
          'physical source IDs and common elapsed time. No browser physics.')


if __name__=='__main__':main()
