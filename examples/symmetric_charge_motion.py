"""M21c: source-anchored turning/escape and composite reflection-plane motion.

POTENTIAL_CONDUCTORS1.pdf physical p13, Illustration 7, is a teaching problem,
not a claimed official past-paper question. Q, q, m and a are symbolic there;
all numerical quantities here are explicit illustrative SI choices.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from math import hypot, pi, sqrt
from pathlib import Path

from conductor_sim import (EPSILON_0, FixedPointField, PointCharge, PrescribedField,
                           TestChargeMotionSpec, integrate_test_charge,
                           sample_test_charge_viewer)
from conductor_sim.scenarios import PrescribedScenario, revise_scenario
from conductor_sim.source_symmetry import certify_point_reflection
from conductor_sim.visualization import write_visualization

CASES = ('turning', 'escape-threshold', 'above-escape', 'mirror-plane')
SOURCE_SHA256 = '8d0d9b5cf821219058cfc4b7685a39b777e460944df44c3f9cffcb68157a60e5'
ANNOTATIONS = (('charge:upper', 'Upper fixed +Q'), ('charge:lower', 'Lower fixed +Q'))
K = 1 / (4*pi*EPSILON_0)
Q = 1e-9
Q_PROBE = -1e-9
A = .1
MASS = 1e-6


def make_scenario(case='turning'):
    if case not in CASES:
        raise ValueError('unknown M21c example')
    field = PrescribedField((FixedPointField(Q, (0., A, 0.)),
                             FixedPointField(Q, (0., -A, 0.))))
    v_turn = sqrt(2*K*Q*abs(Q_PROBE)/(MASS*A))
    base_spec = TestChargeMotionSpec(
        field, PointCharge(Q_PROBE), MASS, (v_turn, 0., 0.), 2., .005,
        plane_normal=(0., 0., 1.), length_scale_m=A)
    base = PrescribedScenario('m21c-turning', base_spec, ANNOTATIONS,
        'Teaching notes p13 Illustration 7: symbolic Q,q,m,a instantiated explicitly '
        'as Q=q=1 nC, m=1 milligram (1e-6 kg), a=0.10 m; launch for x_turn=sqrt(3)a.')
    if case == 'turning':
        return base
    if case == 'mirror-plane':
        # Intentionally new teaching variation, not the printed axial launch.
        # Each charge lies outside the chosen xz motion plane; the pair as a
        # whole preserves it. Both force components in this plane are kept.
        spec = TestChargeMotionSpec(
            field, PointCharge(Q_PROBE, (.04, 0., .02)), MASS,
            (.03, 0., .06), .18, .003, plane_normal=(0., 1., 0.), length_scale_m=A)
        return PrescribedScenario('m21c-mirror-plane', spec, ANNOTATIONS,
            'Teacher hypothetical: oblique launch in the xz reflection plane; '
            'two fixed +Q sources remain outside this slice, not projected into it.')
    factor = 1. if case == 'escape-threshold' else 1.1
    speed = factor*sqrt(4*K*Q*abs(Q_PROBE)/(MASS*A))
    return revise_scenario(base, scenario_id='m21c-'+case,
        note=('Source part (ii): escape threshold from conservation of energy. '
              'Finite-time export does not reach infinity.' if factor == 1. else
              'Teacher hypothetical: launch at 1.1 times the analytic escape speed. '
              'Positive energy implies escape for this source geometry, not arrival at infinity.'),
        motion_changes={'initial_velocity_m_s': (speed, 0., 0.)})


def independent_reference(scenario, result):
    """Coulomb/energy equations independent of the production field provider."""
    spec = scenario.spec
    rows = result.samples
    # All examples keep exactly the two original stationary source positions.
    errors = []
    energies = []
    for row in rows:
        x, y, z = row.position_m
        r1, r2 = hypot(x, y-A, z), hypot(x, y+A, z)
        potential = K*Q*(1/r1+1/r2)
        E = (K*Q*x*(1/r1**3+1/r2**3),
             K*Q*((y-A)/r1**3+(y+A)/r2**3),
             K*Q*z*(1/r1**3+1/r2**3))
        errors.append(abs(row.potential_v-potential))
        errors.extend(abs(a-b) for a, b in zip(row.electric_field_v_m, E))
        energies.append(.5*spec.mass_kg*hypot(*row.velocity_m_s)**2 + spec.particle.charge_c*potential)
    out = dict(analytic_turning_launch_m_s=sqrt(2*K*Q*abs(Q_PROBE)/(MASS*A)),
               analytic_escape_speed_m_s=sqrt(4*K*Q*abs(Q_PROBE)/(MASS*A)),
               max_reference_E_V_error=max(errors),
               max_energy_change_j=max(abs(e-energies[0]) for e in energies),
               max_out_of_plane_m=max(abs(row.position_m[1]) for row in rows),
               field_law='Independent Coulomb sum; U=qV; fixed-source self/mutual energy is constant',
               time_scope='Finite physical times only; no claim of reaching infinity')
    if scenario.scenario_id == 'm21c-turning':
        out['analytic_turning_x_m'] = sqrt(3)*A
        out['largest_saved_x_m'] = max(row.position_m[0] for row in rows)
        out['turning_velocity_bracket_s'] = next(
            [a.time_s, b.time_s] for a, b in zip(rows, rows[1:])
            if a.velocity_m_s[0] > 0 >= b.velocity_m_s[0])
        out['turning_scope'] = 'Bracket between saved samples; not an exact event-time solve'
    return out


def build(case='turning'):
    scenario = make_scenario(case)
    before = scenario.as_dict()
    result = integrate_test_charge(scenario.spec)
    if result.termination != 'duration':
        raise ValueError('example did not complete its independently integrated duration')
    bundle = sample_test_charge_viewer(
        scenario.spec, result=result, source_annotations=scenario.source_annotations,
        visual_frames=16, resolution=11, extent_m={'turning':.28, 'mirror-plane':.02}.get(case, .9))
    certificate = certify_point_reflection(scenario.spec.field.sources,
        origin_m=scenario.spec.particle.position_m, normal=(0., 1., 0.))
    reference = independent_reference(scenario, result)
    note = ('M21c composite reflection: equal signed point sources are paired one-to-one. '
            'Full vector F=qE is integrated without an artificial guide. '
            'Display orientation is stated in world coordinates and may differ from the printed figure.')
    for frame in bundle['frames']:
        frame['notes'].extend([scenario.viewer_provenance, note, reference['time_scope']])
        if case == 'turning':
            frame['notes'].append('Source target: turning at x=sqrt(3)a. Saved velocity bracket: '+
                                  str(reference['turning_velocity_bracket_s'])+' s; not an exact event time.')
    bundle['m21c_teaching'] = dict(case=case, source_file='POTENTIAL_CONDUCTORS1.pdf',
        sha256=SOURCE_SHA256, physical_pdf_page=13, anchor='Illustration 7',
        provenance='Teaching illustration; not asserted to be an official JEE paper',
        numerical_quantities='Illustrative Q=q=1e-9 C, m=1e-6 kg, a=.1 m',
        source_fidelity=('Teacher hypothetical' if case in ('above-escape','mirror-plane') else
                         'Source symbolic relationship; explicit numerical instantiation'),
        point_reflection=asdict(certificate), reference=reference)
    if scenario.as_dict() != before:
        raise AssertionError('source scenario mutated during integration or export')
    return scenario, result, bundle


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case', choices=CASES, default=CASES[0])
    ap.add_argument('--output', type=Path, default=Path('output/m21c-turning.html'))
    args = ap.parse_args()
    pdf = Path(__file__).resolve().parents[1] / 'POTENTIAL_CONDUCTORS1.pdf'
    if sha256(pdf.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('source PDF differs from the reviewed p13 illustration')
    scenario, result, bundle = build(args.case)
    write_visualization(bundle, args.output, title='M21c · Symmetric fixed charges · '+args.case)
    audit = dict(scenario=scenario.as_dict(), observations=bundle,
                 validation='Source/force/energy checks and independent Coulomb references; finite-time numerical trajectory')
    args.output.with_suffix('.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(args.case, result.termination, len(result.samples), 'saved physical samples')
    print(json.dumps(bundle['m21c_teaching']['reference'], indent=2))


if __name__ == '__main__':
    main()
