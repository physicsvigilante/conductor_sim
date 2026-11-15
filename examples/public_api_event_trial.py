"""Post-M24b public-API teaching cases; no arrangement-specific production code.

These are same-developer, preselected transfer checks, not a blinded authoring
benchmark. Point/rod and coaxial-ring models retain their true qualifications.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from conductor_sim import (FixedPointField, UniformRodField, UniformRingAxisField,
    PrescribedField, PointCharge, TestChargeMotionSpec, UniformGravity)
from conductor_sim.scenarios import PrescribedScenario
from conductor_sim.driven_motion import (LinearDrag, FieldPhase, UniformFieldSchedule,
    DrivenModel, DrivenStudy, EventPolicy, PlaneEvent, VelocityReversal,
    investigate_driven_motion, revise_driven_model)
from conductor_sim.driven_viewer import render_driven_run

CASES = ('mirror-rod', 'coaxial-rings')


def author(case):
    if case == 'mirror-rod':
        sources = (FixedPointField(2e-11, (0., .18, .1)),
                   FixedPointField(2e-11, (0., -.18, .1)),
                   UniformRodField(-1e-11, (-.15, 0., -.1), (-.15, 0., .1)))
        labels = (('point:upper', 'Upper held point'), ('point:lower', 'Lower held point'),
                  ('rod:left', 'Left stationary charged rod'))
        q, mass, position, velocity = 1e-9, 1e-9, (.05, 0., .25), (.15, 0., .02)
        gravity, drag, duration = (0., 0., -.08), 4e-10, .9
        phases = (FieldPhase(0., (.1, 0., .05)), FieldPhase(.23, (-1., 0., .25)),
                  FieldPhase(.53, (.2, 0., 0.)))
        plane = PlaneEvent('return', (.05, 0., 0.), (1., 0., 0.),
                           'Return through launch x plane', -1, True)
    elif case == 'coaxial-rings':
        sources = (UniformRingAxisField(4e-11, .3, axis=(1.,0.,0.)),
                   UniformRingAxisField(-3e-11, .3, center_m=(.4,0.,0.), axis=(1.,0.,0.)))
        labels = (('ring:A', 'Stationary positive ring A'), ('ring:B', 'Stationary negative ring B'))
        q, mass, position, velocity = -1e-9, 2e-9, (.18,0.,0.), (-.04,0.,0.)
        gravity, drag, duration = (.05,0.,0.), 6e-10, 1.2
        phases = (FieldPhase(0., (-.15,0.,0.)), FieldPhase(.31, (.18,0.,0.)))
        plane = PlaneEvent('detector', (.17,0.,0.), (1.,0.,0.), 'Axial detector', 0)
    else:
        raise ValueError('unknown public authoring card')
    base = PrescribedScenario('trial:'+case, TestChargeMotionSpec(
        PrescribedField(sources), PointCharge(q,position), mass, velocity,
        duration_s=duration, sample_interval_s=.05, plane_normal=(0.,1.,0.),
        gravity=UniformGravity(gravity), relative_tolerance=1e-10), labels,
        'Teacher hypothetical; original source values are explicitly authored, not an exam reproduction.')
    model = DrivenModel(base, LinearDrag(drag), UniformFieldSchedule(phases),
        'driven:'+case, 'Public-interface transfer: fixed sources, scheduled field, gravity and drag. '
        'Numerical events do not imply contact. Named sources may be omitted from this projection; '
        'all remain in the full physical computation.')
    queries = (VelocityReversal('turn:x', (1.,0.,0.), 'Axial component reversal'), plane)
    return model, queries


def run_trial(case):
    model, queries = author(case)
    policy = EventPolicy(.01, 1e-9, 1e-10, 1e-9)
    snapshot = model.as_dict()
    combined = investigate_driven_motion(DrivenStudy(model, queries, policy))
    reversal_only = investigate_driven_motion(DrivenStudy(model, queries[:1], policy))
    child, changes = revise_driven_model(model, model_id=model.model_id+'-more-drag',
        note='Teacher variation: double only the declared drag coefficient.',
        drag=LinearDrag(2*model.drag.coefficient_kg_s, model.drag.component_id, model.drag.label))
    revision = investigate_driven_motion(DrivenStudy(child, queries, policy))
    assert combined.study.model is reversal_only.study.model is model
    assert model.as_dict() == snapshot and child.parent_revision_id == model.revision_id
    return combined, dict(case=case, exposure='same developer; not blinded or independent authoring',
        source='teacher hypothetical', production_changes_for_arrangement=0,
        same_model_retained=True, combined=combined.as_dict(),
        reversal_only=reversal_only.as_dict(), revision=revision.as_dict(), change_record=changes)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', choices=CASES, default=CASES[0])
    p.add_argument('--output', type=Path, default=Path('output/public-api-trial.html'))
    args=p.parse_args()
    result, record=run_trial(args.case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_driven_run(result,title='Public API trial · '+args.case), encoding='utf-8')
    args.output.with_suffix('.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n', encoding='utf-8')
    print(args.case, result.termination, [(e.event_id,e.sample.time_s) for e in result.events])


if __name__=='__main__':main()
