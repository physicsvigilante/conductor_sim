"""M20b: author once, revise by stable identity, then independently simulate.

Every case is a teacher hypothetical, not the workbook's transverse-field
problem. Source fields remain prescribed/nonresponding, not conducting rings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from conductor_sim import write_scenario_comparison
from conductor_sim.scenarios import PrescribedScenario, revise_scenario, sample_scenario_viewer

if __package__:
    from examples.ring_source_orientation import make_case, SOURCE_ANNOTATIONS
else:
    from ring_source_orientation import make_case, SOURCE_ANNOTATIONS

CASES = ('double-ring-a-charge', 'ring-b-shift', 'double-mass',
         'reverse-field', 'launch-speed', 'gauge-shift')


def make_scenarios(case='double-ring-a-charge'):
    baseline = PrescribedScenario('ring-AB-base', make_case('axial-field'), SOURCE_ANNOTATIONS,
        'Rings inspired by page 12 Q40-41; explicit mass 0.10 kg and axial external field, '
        'not the original transverse-field problem.')
    if case == 'double-ring-a-charge':
        patches = dict(source_changes={'ring:A': {'charge_c': 20e-6}})
        note = 'Double Ring A charge from +10 to +20 microcoulombs.'
    elif case == 'ring-b-shift':
        patches = dict(source_changes={'ring:B': {'center_m': (.45, 0., 0.)}})
        note = 'Move Ring B along the common world x axis, from 0.40 to 0.45 m.'
    elif case == 'double-mass':
        patches = dict(motion_changes={'mass_kg': .20})
        note = 'Double test-particle mass, retaining every source and initial state.'
    elif case == 'reverse-field':
        patches = dict(source_changes={'field:external': {'field_v_m': (-100000., 0., 0.)}})
        note = 'Reverse the applied axial electric field from +100000 to -100000 V/m.'
    elif case == 'launch-speed':
        patches = dict(motion_changes={'initial_velocity_m_s': (.5, 0., 0.)})
        note = 'Launch the particle at +0.5 m/s along world x instead of releasing it from rest.'
    elif case == 'gauge-shift':
        patches = dict(source_changes={'field:external': {'reference_potential_v': 17.}})
        note = 'Shift the external-field potential reference by +17 V, without changing E.'
    else:
        raise ValueError('unsupported revision case')
    variation = revise_scenario(baseline, scenario_id='ring-AB-' + case, note=note, **patches)
    return baseline, variation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES, default=CASES[0])
    parser.add_argument('--output', type=Path, default=Path('output/m20b-ring-revision.html'))
    args = parser.parse_args()
    baseline, variation = make_scenarios(args.case)
    options = dict(visual_frames=12, profile_samples=101, resolution=15,
                   extent_m=2.25, axis_orientation='horizontal')
    left, right = (sample_scenario_viewer(s, **options) for s in (baseline, variation))
    # M20a checks completeness, shared physical axes, scale and exact time grid;
    # it independently derives the change ledger from the solved inventories.
    write_scenario_comparison(left, right, args.output,
        title='M20b - Reusable Ring A/B model: ' + args.case,
        baseline_id=baseline.scenario_id, variation_id=variation.scenario_id,
        baseline_title='Baseline - axial external field',
        variation_title='Teacher revision - ' + args.case,
        baseline_provenance=baseline.viewer_provenance,
        variation_provenance=variation.viewer_provenance)
    audit = args.output.with_suffix('.json')
    audit.write_text(json.dumps({'baseline': baseline.as_dict(), 'variation': variation.as_dict()},
                               indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('Wrote ' + str(args.output) + ' and ' + str(audit))
    print(variation.required_action + '; two independently solved histories; no browser physics.')


if __name__ == '__main__':
    main()
