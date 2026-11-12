"""M6b: independently check authored *new* questions, not just program output."""
from dataclasses import replace
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest

from conductor_sim import Femm5Backend, NestedShellBackend
from conductor_sim.solver import UnsupportedProblem
from m6b.attempts import SUBMISSIONS
from m6b.evaluate import AuthoringError, export_one, grade_all, grade_one, submissions_from
from m6b.protocol import Submission
from m6b.questions import QUESTIONS
from m6b.reference import KEYS
from conductor_sim.pdf_benchmark import CASES


@pytest.mark.parametrize('case_id',[q.id for q in QUESTIONS])
def test_heldout_submissions_solve_or_refuse_as_required(case_id):
    result=grade_one(case_id,SUBMISSIONS[case_id])
    assert result['passed'], result


def test_heldout_questions_are_new_distinct_and_trust_boundary_is_explicit():
    ids=[q.id for q in QUESTIONS]
    assert len(ids)==len(set(ids))==15
    assert all(q.prompt and len(q.prompt)>75 for q in QUESTIONS)
    assert set(ids).isdisjoint({c.id for c in CASES})
    assert set(ids)==set(KEYS)==set(SUBMISSIONS)
    assert sum(KEYS[i].capability=='both' for i in ids)==9
    assert sum(KEYS[i].capability=='analytical-only' for i in ids)==1
    assert sum(KEYS[i].capability=='unsupported' for i in ids)==5
    # Authored file imports the public models + protocol, not the secret keys.
    authored_code=inspect.getsource(sys.modules['m6b.attempts'])
    assert 'from .reference import' not in authored_code
    assert 'MultiImageShellBackend' not in authored_code
    assert 'NestedShellBackend' not in authored_code
    assert 'Femm5Backend' not in authored_code


def test_full_grade_provides_provenance_compatible_machine_report():
    grade=grade_all(SUBMISSIONS)
    assert grade['passed']==15 and grade['failed']==0
    assert (grade['analytical_and_femm'],grade['analytical_only'],grade['correct_refusals'])==(9,1,5)
    assert grade['missing']==grade['extra']==[]
    assert sum(len(x['observations']) for x in grade['results'])==29
    assert all('issues' in x for x in grade['results'])
    assert json.loads(json.dumps(grade,allow_nan=False))['passed']==15


def test_wrong_model_or_missing_observation_does_not_get_credit():
    original=SUBMISSIONS['two-shell-negative-net']
    missing=replace(original,probes=original.probes[:-1])
    r=grade_one('two-shell-negative-net',missing)
    assert not r['passed'] and any('probe' in x for x in r['issues'])
    impostor=replace(original,states=SUBMISSIONS['grounded-outer-shield'].states)
    r=grade_one('two-shell-negative-net',impostor)
    assert not r['passed'] and any('geometry' in x for x in r['issues'])
    wrong_class=replace(original,capability='analytical-only')
    assert not grade_one('two-shell-negative-net',wrong_class)['passed']


def test_unsupported_must_avoid_geometry_fabrication():
    case='charged-ring-on-axis'
    original=SUBMISSIONS[case]
    wrong=replace(original,states=SUBMISSIONS['two-shell-negative-net'].states)
    assert not grade_one(case,wrong)['passed']
    wrong=replace(original,reason='not supported')
    assert not grade_one(case,wrong)['passed']
    wrong=replace(original,capability='both')
    assert not grade_one(case,wrong)['passed']


def test_noncollinear_source_is_analytical_only_and_femm_explicitly_rejects():
    s=SUBMISSIONS['noncollinear-math-only']
    problem=s.states[0][1]
    assert NestedShellBackend().supports(problem)
    assert not Femm5Backend().supports(problem)
    assert grade_one('noncollinear-math-only',s)['passed']
    with pytest.raises(UnsupportedProblem):
        Femm5Backend().solve(problem)


def test_sweep_html_and_json_and_no_fabricated_recomputation(tmp_path):
    candidate='grounded-two-state-sweep'
    dest=tmp_path/'generated.html'
    data=export_one(candidate,SUBMISSIONS[candidate],dest,resolution=21)
    assert data['states']==2 and data['backend']=='analytical'
    assert dest.is_file() and dest.with_suffix('.json').is_file()
    raw=json.loads(dest.with_suffix('.json').read_text(encoding='utf-8'))
    assert len(raw['frames'])==2
    assert raw['authorability_source']['question_id']==candidate
    assert raw['authorability_source']['backend_requested']=='analytical'
    assert 'precomputed' in raw['playback']
    assert raw['frames'][0]['conductor_results'][0]['potential_v'] != raw['frames'][1]['conductor_results'][0]['potential_v']


def test_renderer_rejects_blocked_cases_and_disallowed_backend_before_writing(tmp_path):
    for case,backend in [('charged-ring-on-axis','analytical'),
                         ('noncollinear-math-only','femm')]:
        path=tmp_path/f'{case}.html'
        with pytest.raises(UnsupportedProblem):
            export_one(case,SUBMISSIONS[case],path,backend_name=backend)
        assert not path.exists()
        assert not path.with_suffix('.json').exists()


def test_cli_grades_and_renders_and_rejects_invalid_request(tmp_path):
    cmd=[sys.executable,'-m','m6b.evaluate']
    working=Path(__file__).resolve().parents[1]
    report=tmp_path/'m6b.json'
    out=subprocess.run(cmd+['grade','--output',str(report)],cwd=working,
                       capture_output=True,text=True,timeout=25)
    assert out.returncode==0,out.stderr
    assert json.loads(report.read_text())['passed']==15
    blocked=tmp_path/'blocked.html'
    bad=subprocess.run(cmd+['render','charge-between-conductors',
                            '--output',str(blocked)],cwd=working,
                       capture_output=True,text=True,timeout=25)
    assert bad.returncode==2
    assert not blocked.exists()
    html=tmp_path/'view.html'
    ok=subprocess.run(cmd+['render','grounded-two-state-sweep',
                           '--output',str(html),'--resolution','21'],
                      cwd=working,capture_output=True,text=True,timeout=25)
    assert ok.returncode==0,ok.stderr
    assert html.is_file()


def test_custom_submission_module_loader_is_reusable():
    assert submissions_from('m6b.attempts') is SUBMISSIONS
    with pytest.raises(AuthoringError):
        submissions_from('m6b.questions')
