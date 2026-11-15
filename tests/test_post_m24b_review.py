"""Bounded post-M24b review: numerical coincidence and saved-observation integrity.

Counterexamples were selected and exercised on the unchanged M24b baseline.
No new physical laws, source regimes or numerical tolerances are introduced.
"""
from dataclasses import replace
from math import hypot
import json
import re

import pytest

from conductor_sim import (PrescribedField, UniformElectricField, UniformGravity,
    PointCharge, TestChargeMotionSpec as MotionSpec)
from conductor_sim.scenarios import PrescribedScenario
from conductor_sim.motion_events import (EventStudy, EventPolicy, PlaneEvent,
    investigate_motion, validate_event_run)
from conductor_sim.driven_motion import (DrivenModel, DrivenStudy, LinearDrag,
    FieldPhase, UniformFieldSchedule, investigate_driven_motion, validate_driven_run)
from conductor_sim.motion_event_viewer import render_event_run
from conductor_sim.driven_viewer import render_driven_run


def scenario(*, gravity=False):
    return PrescribedScenario('review:base', MotionSpec(
        PrescribedField((UniformElectricField((0., 0., .2)),) if gravity else ()),
        PointCharge(1e-9, (0., 0., 0.)), 1e-9, (1., 0., 0.),
        duration_s=2., sample_interval_s=.125, plane_normal=(0., 1., 0.),
        gravity=UniformGravity((0., 0., -.1)) if gravity else None),
        (('field:fixed', 'Fixed E'),) if gravity else (), 'Review teacher hypothetical')


def driven(base):
    return DrivenModel(base, LinearDrag(0.), UniformFieldSchedule((
        FieldPhase(0., (0., 0., 0.)),)), 'review:driven', 'Review: zero-drag limit')


def policy():
    return EventPolicy(.025, 1e-8, 1e-10, 1e-9)


def payload(html, identifier):
    return json.loads(re.search(r'<script id="'+identifier+r'" type="application/json">(.*?)</script>', html, re.S).group(1))


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('second_terminal', [False, True])
def test_near_coincident_driven_detectors_preserve_uncertainty_and_end_time(reverse, second_terminal):
    base = scenario()
    # Keep the original reproduction's output grid; roots have overlapping
    # numerical brackets although their representative samples differ slightly.
    base = replace(base, spec=replace(base.spec, duration_s=.5, sample_interval_s=.1))
    qs = (PlaneEvent('first', (.11, 0., 0.), (1., 0., 0.), 'First', 1, True),
          PlaneEvent('second', (.11+1e-10, 0., 0.), (1., 0., 0.), 'Second', 1, second_terminal))
    if reverse:
        qs = qs[::-1]
    original = driven(base)
    snapshot = original.as_dict()
    run = investigate_driven_motion(DrivenStudy(original, qs, policy()))
    assert validate_driven_run(run)
    assert run.termination == 'terminal-event'
    assert run.ordering == 'coincident-or-order-unresolved-within-time-resolution'
    assert {e.event_id for e in run.events} == {'first', 'second'}
    end = run.samples[-1].time_s
    assert any(e.sample.time_s > end for e in run.events)
    assert all(e.sample.time_s <= end+policy().time_tolerance_s for e in run.events)
    assert end == min(e.sample.time_s for e in run.events if e.terminal)
    html = render_driven_run(run)
    record = payload(html, 'driven-data')
    assert max(s['time_s'] for s in record['display_states']) == end
    assert 'not post-stop motion' in html
    assert original.as_dict() == snapshot
    # A changed conclusion must not turn alternative terminal candidates into
    # a claim of resolved order, or become accepted post-stop physical motion.
    with pytest.raises(ValueError):
        validate_driven_run(replace(run, ordering='resolved-separated-brackets'))


def test_well_separated_driven_detector_does_not_extend_terminal_run():
    m = driven(scenario())
    qs = (PlaneEvent('first', (.11, 0., 0.), (1., 0., 0.), 'First', 1, True),
          PlaneEvent('later', (.15, 0., 0.), (1., 0., 0.), 'Later', 1, True))
    r = investigate_driven_motion(DrivenStudy(m, qs, policy()))
    assert [e.event_id for e in r.events] == ['first']
    assert r.as_dict()['outcomes']['later'] == 'search-truncated'
    assert r.ordering == 'resolved-separated-brackets'


@pytest.fixture(scope='module')
def conservative():
    return investigate_motion(EventStudy(scenario(gravity=True), (
        PlaneEvent('detector', (.25, 0., 0.), (1., 0., 0.), 'Detector', 1, True),), policy()))


@pytest.mark.parametrize('field,value', [
    ('displacement_m', (9., 8., 7.)), ('displacement_m', ()),
    ('potential_v', 12345.), ('potential_v', False),
    ('electric_force_n', (0., 0., 0.)), ('electric_force_n', ()),
    ('gravitational_force_n', (0., 0., 0.)), ('gravitational_force_n', None),
    ('electric_potential_energy_j', 12345.), ('electric_potential_energy_j', False),
    ('gravitational_potential_energy_j', 12345.), ('gravitational_potential_energy_j', None),
])
@pytest.mark.parametrize('location', ['trajectory', 'event'])
def test_all_saved_conservative_observations_are_checked(conservative, field, value, location):
    r = conservative
    if location == 'trajectory':
        rows = (*r.trajectory.samples[:-1], replace(r.trajectory.samples[-1], **{field:value}))
        bad = replace(r, trajectory=replace(r.trajectory, samples=rows))
    else:
        bad = replace(r, events=(replace(r.events[0], sample=replace(r.events[0].sample, **{field:value})),))
    with pytest.raises(ValueError):
        validate_event_run(bad)
    with pytest.raises(ValueError):
        render_event_run(bad)


@pytest.mark.parametrize('which', ['conservative','driven'])
@pytest.mark.parametrize('field,value', [
    ('residual', False), ('time_bracket_s', (True, True)),
    ('time_bracket_s', (1., True)),
])
def test_event_numeric_metadata_rejects_booleans(which, field, value):
    base = scenario()
    q = PlaneEvent('exact', (1.,0.,0.), (1.,0.,0.), 'Exactly representable event', 1, True)
    p = EventPolicy(.03125, 1e-8, 1e-10, 1e-9)
    if which == 'conservative':
        r = investigate_motion(EventStudy(base, (q,), p))
        validate = validate_event_run
    else:
        r = investigate_driven_motion(DrivenStudy(driven(base), (q,), p))
        validate = validate_driven_run
    assert r.events[0].sample.time_s == 1. and r.events[0].residual == 0.
    bad = replace(r, events=(replace(r.events[0], **{field:value}),))
    with pytest.raises(ValueError):
        validate(bad)


def test_no_gravity_contributions_cannot_be_injected_into_electric_only_event():
    r = investigate_motion(EventStudy(scenario(), (
        PlaneEvent('p',(.25,0.,0.),(1.,0.,0.),'Plane',1,True),),policy()))
    bad = replace(r, events=(replace(r.events[0], sample=replace(r.events[0].sample,
        gravitational_force_n=(0.,0.,1.))),))
    with pytest.raises(ValueError):
        validate_event_run(bad)


def test_valid_conservative_observations_and_original_state_unchanged(conservative):
    before = conservative.as_dict()
    assert validate_event_run(conservative)
    html = render_event_run(conservative)
    assert payload(html, 'event-data')['trajectory']['termination'] == 'terminal-event'
    assert conservative.as_dict() == before
    s = conservative.events[0].sample
    assert s.gravitational_force_n == pytest.approx((0., 0., -1e-10), rel=1e-14, abs=0.)
    assert s.electric_force_n == pytest.approx((0., 0., 2e-10), rel=1e-14, abs=0.)


@pytest.mark.parametrize('case', ['mirror-rod', 'coaxial-rings'])
def test_public_api_transfer_with_independent_source_sums(case):
    from math import pi, sqrt
    from conductor_sim import EPSILON_0
    from examples.public_api_event_trial import run_trial
    run, report = run_trial(case)
    assert report['same_model_retained']
    assert report['production_changes_for_arrangement'] == 0
    assert report['change_record']['child']['parent_revision_id'] == run.study.model.revision_id
    assert report['combined']['termination'] == report['reversal_only']['termination'] == 'duration'
    for record in (report['combined'],report['reversal_only'],report['revision']):
        assert record['termination'] == 'duration'
        physical = record['study']['model']; spec=physical['base']['spec']
        q, mass=spec['particle']['charge_c'],spec['mass_kg']
        for row in record['samples']:
            p=row['position_m']; expected=[0.,0.,0.]; k=1/(4*pi*EPSILON_0)
            for source in spec['field']['sources']:
                if source['kind']=='FixedPointField':
                    d=[p[i]-source['position_m'][i] for i in range(3)]
                    r=sqrt(sum(x*x for x in d))
                    for i in range(3):expected[i]+=k*source['charge_c']*d[i]/r**3
                elif source['kind']=='UniformRingAxisField':
                    x=sum((p[i]-source['center_m'][i])*source['axis'][i] for i in range(3))
                    for i in range(3):
                        expected[i]+=k*source['charge_c']*x*source['axis'][i]/(x*x+source['radius_m']**2)**1.5
                else:
                    # Independently integrate Coulomb line elements, not the
                    # production finite-rod antiderivative.
                    assert source['kind']=='UniformRodField'
                    N=1024
                    for j in range(N):
                        d=[p[i]-source['start_m'][i]-(j+.5)/N*(source['end_m'][i]-source['start_m'][i]) for i in range(3)]
                        r=sqrt(sum(x*x for x in d))
                        for i in range(3):expected[i]+=k*source['charge_c']/N*d[i]/r**3
            phase=physical['applied']['phases'][row['phase_index']]['field_v_m']
            gravity=spec['gravity']['acceleration_m_s2']
            force=[q*(expected[i]+phase[i])+mass*gravity[i]-physical['drag']['coefficient_kg_s']*row['velocity_m_s'][i] for i in range(3)]
            assert row['force_n']==pytest.approx(force,rel=2e-6,abs=2e-16)
            assert row['dissipated_energy_j']>=0
            assert abs(row['work_balance_error_j'])<1e-16


def test_public_api_ring_revision_keeps_unsupported_transverse_phase_refused():
    from examples.public_api_event_trial import author
    from conductor_sim.driven_motion import revise_driven_model
    from conductor_sim import UnsupportedProblem
    model,_=author('coaxial-rings'); before=model.as_dict()
    bad=UniformFieldSchedule((FieldPhase(0.,(-.15,0.,0.)),FieldPhase(.31,(.18,0.,.1))))
    with pytest.raises(UnsupportedProblem):
        revise_driven_model(model,model_id='trial:invalid',note='Deliberate unsupported phase',applied=bad)
    assert model.as_dict()==before
