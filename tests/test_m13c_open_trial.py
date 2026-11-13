"""M13c open problem-authoring trial, independently scored and replayable."""
from __future__ import annotations

from dataclasses import replace
import inspect
import json
from math import log, pi
from pathlib import Path
import subprocess
import sys

import pytest

from conductor_sim import ParallelPlateBackend
from m13c.attempts import SUBMISSIONS
from m13c.evaluate import (export_one, grade_all, grade_one, source_inventory,
                           submissions_from)
from m13c.protocol import Submission
from m13c.questions import BY_ID, QUESTIONS
from m13c.reference import EPS, KEYS, K, reference_observation


@pytest.mark.parametrize('question_id',[q.id for q in QUESTIONS])
def test_first_pass_authored_new_problem_or_explicit_refusal(question_id):
    result=grade_one(question_id,SUBMISSIONS[question_id])
    assert result['passed'],result['issues']
    if result['expected_capability']=='analytical-only':
        assert len(result['observations'])==KEYS[question_id].steps
    else:
        assert not result['observations']


def test_open_trial_is_separate_from_prior_m6_and_pdf_source_inventory():
    from conductor_sim.pdf_benchmark import CASES
    from m6b.questions import QUESTIONS as M6B
    ids={q.id for q in QUESTIONS}
    assert len(ids)==len(QUESTIONS)==18
    assert ids==set(KEYS)==set(SUBMISSIONS)
    assert ids.isdisjoint({c.id for c in CASES})
    assert ids.isdisjoint({q.id for q in M6B})
    assert sum(k.backend is not None for k in KEYS.values())==10
    assert sum(k.backend is None for k in KEYS.values())==8
    assert all(len(q.prompt)>90 for q in QUESTIONS)
    author_source=inspect.getsource(sys.modules['m13c.attempts'])
    assert 'm13c.reference' not in author_source
    assert 'from .reference import' not in author_source
    assert 'GroundedPlaneBackend' not in author_source
    assert 'ExternalSphereBackend' not in author_source
    assert 'ParallelPlateBackend' not in author_source
    report=source_inventory()
    assert all('present' in x and 'source_used_in_m13c' in x for x in report['documents'])
    old=next(x for x in report['documents'] if x['filename']=='POTENTIAL_CONDUCTORS1.pdf')
    assert old['present'] and old['sha256']=='8d0d9b5cf821219058cfc4b7685a39b777e460944df44c3f9cffcb68157a60e5'
    # The user's new document is not fabricated into a source citation:
    assert not any(x['source_used_in_m13c'] for x in report['documents'])


def test_exact_plane_and_grounded_slabs_have_independent_closed_form_anchors():
    key=KEYS['plane-approach']
    f,ud,ui,ub=reference_observation(key,0.)
    q=key.charges[0][0];h=key.charges[0][1][2]
    assert ud==ub==0
    assert ui==pytest.approx(-K*q*q/(4*h),rel=1e-12)
    assert f[2]==pytest.approx(-K*q*q/(4*h*h),rel=1e-12)
    slab=KEYS['plates-grounded']
    d=.2
    # At the grounded midpoint, the image-series value is exact.
    f,ud,ui,ub=reference_observation(slab,.05)
    assert abs(f[2])<1e-17
    assert ui==pytest.approx(-K*1e-18*log(2)/d,rel=4e-6)


def test_wrong_model_voltage_source_sign_and_path_are_rejected_without_grading_its_graph():
    good=SUBMISSIONS['plates-biased-negative']
    wrong_backend=replace(good,backend='plane')
    assert not grade_one('plates-biased-negative',wrong_backend)['passed']
    spec=good.sweep
    bad=replace(spec,steps=9)
    model=replace(good,sweep=bad)
    failure=grade_one('plates-biased-negative',model)
    assert not failure['passed']
    assert any('geometry' in msg for msg in failure['issues'])
    wrong=replace(good,capability='unsupported',sweep=None,backend=None,
                  reason='we have no support')
    assert not grade_one('plates-biased-negative',wrong)['passed']
    other=replace(good,sweep=SUBMISSIONS['plates-dual-bias'].sweep)
    assert not grade_one('plates-biased-negative',other)['passed']


def test_false_refusal_is_not_credited_and_unsupported_cannot_fabricate_solver():
    key='dipole-torque'
    good=SUBMISSIONS[key]
    assert grade_one(key,good)['passed']
    for attempted in (
        replace(good,reason='no'),
        replace(good,capability='analytical-only',backend='plates',
                sweep=SUBMISSIONS['plates-grounded'].sweep),
        replace(good,sweep=SUBMISSIONS['plates-grounded'].sweep),
    ):
        assert not grade_one(key,attempted)['passed']
    assert 'femm' in SUBMISSIONS['offaxis-femm-force'].reason.lower()


def test_full_first_pass_report_is_finite_and_discloses_missing_pdf(tmp_path):
    report=grade_all(SUBMISSIONS)
    assert (report['question_count'],report['passed'],report['failed'])==(18,18,0)
    assert (report['supported_passes'],report['correct_refusals'])==(10,8)
    assert report['missing']==report['extra']==[]
    assert report['new_pdf_questions_graded']==0
    assert 'not blind' in report['method'].lower()
    assert json.loads(json.dumps(report,allow_nan=False))['passed']==18
    assert sum(len(r['observations']) for r in report['results'])==sum(
        key.steps for key in KEYS.values() if key.backend)


def test_render_one_independent_model_and_refuse_blocked(tmp_path):
    case='plates-grounded'
    html=tmp_path/'m13c.html'
    result=export_one(case,SUBMISSIONS[case],html,resolution=13,image_order=48)
    assert result['states']==9 and html.is_file()
    payload=json.loads(html.with_suffix('.json').read_text(encoding='utf-8'))
    assert payload['authorability_source']['question_id']==case
    assert payload['authorability_source']['classification']=='analytical-only'
    assert len(payload['position_sweep']['samples'])==9
    assert payload['position_sweep']['samples'][4]['force_parallel_n']==pytest.approx(0,abs=1e-18)
    assert all(row['force_energy'] for row in payload['frames'])
    for bad in ['dipole-torque','ring-field','time-trajectory']:
        target=tmp_path/(bad+'.html')
        with pytest.raises(Exception):
            export_one(bad,SUBMISSIONS[bad],target)
        assert not target.exists()


def test_cli_works_with_trusted_submission_loader_and_fail_closed(tmp_path):
    assert submissions_from('m13c.attempts') is SUBMISSIONS
    with pytest.raises(Exception):
        submissions_from('m13c.questions')
    cmd=[sys.executable,'-m','m13c.evaluate']
    output=tmp_path/'m13c.json'
    grade=subprocess.run(cmd+['grade','--output',str(output)],
                         cwd=Path(__file__).parents[1],capture_output=True,text=True,timeout=90)
    assert grade.returncode==0,grade.stderr+grade.stdout
    result=json.loads(output.read_text(encoding='utf-8'))
    assert result['passed']==18
    denied=tmp_path/'bad.html'
    blocked=subprocess.run(cmd+['render','time-trajectory','--output',str(denied)],
                           cwd=Path(__file__).parents[1],capture_output=True,text=True,timeout=20)
    assert blocked.returncode==2 and not denied.exists()
