"""Independent source-anchored acceptance for the 17-page scanned PDF."""
from math import hypot, pi
import json
from pathlib import Path
import subprocess
import sys

import pytest

from conductor_sim import ExternalSphereBackend, force_energy
from m13c.pdf_cases import (CASES, BY_ID, EXECUTABLE, SHA256, K, authored,
                            source_status, grade_pdf, check_one, export_pdf)
from m13c.reference import reference_observation
from m13c.pdf_cases import key_for


@pytest.mark.parametrize('anchor',CASES,ids=lambda x:x.id)
def test_selected_page_anchor_is_explicit_and_not_fabricated(anchor):
    assert 1<=anchor.page<=17 and anchor.question and anchor.topic and anchor.note
    assert anchor.category in ('supported-explorer','static-only','unsupported')
    check=check_one(anchor.id)
    if anchor.id in EXECUTABLE:
        assert check['status']=='checked' and check['passed'] is True,check['issues']
        assert check['frames']>=3
    else:
        assert check['status']=='classification-only' and check['passed'] is None


def test_pinned_scanned_source_and_missing_source_does_not_pass(tmp_path):
    good=source_status()
    assert good['present'] and good['sha256']==SHA256 and good['matches_pinned_source']
    assert good['pages_visually_reviewed']==17
    assert not source_status(root=tmp_path)['matches_pinned_source']
    (tmp_path/good['filename']).write_bytes(b'fake source')
    assert not source_status(root=tmp_path)['matches_pinned_source']


def test_pdf_tangent_neutral_sphere_printed_answers_and_independent_energies():
    spec=authored('p09-multiple-27')
    assert spec.selected_charge.charge_c==1e-6
    assert hypot(*spec.start_m)==5
    sol=ExternalSphereBackend().solve(spec.problem)
    assert sol.conductor_results[0].net_charge_c==0
    v=sol.conductor_results[0].potential_v
    vind=sol.potential_at((3.,0.,0.))-K*1e-6/4.
    assert v==pytest.approx(K*1e-6/5.,rel=1e-12)
    assert vind==pytest.approx(-K*1e-6/20.,rel=1e-12)
    assert v==pytest.approx(1800,rel=.002) and vind==pytest.approx(-450,rel=.002)
    ref=reference_observation(key_for('p09-multiple-27'),0.)
    observed=force_energy(sol)
    assert observed.charges[0].total_force_n==pytest.approx(ref[0],rel=1e-12,abs=1e-14)
    assert observed.energy.total_j==pytest.approx(sum(ref[1:]),rel=1e-12)


def test_approaching_grounded_sphere_printed_current_direction_with_no_time_claim():
    spec=authored('p08-multiple-21')
    assert spec.selected_charge.charge_c==1e-9
    assert spec.problem.conductors[0].radius_m==.1
    initial=ExternalSphereBackend().solve(spec.problem)
    assert initial.conductor_results[0].net_charge_c==pytest.approx(-1e-9*.1/.32)
    checked=check_one('p08-multiple-21')
    assert checked['passed'] and checked['source_observations']['conventional_current_direction']=='sphere to ground'
    assert 'not simulated in time' in checked['source_observations']['caveat']


def test_audited_pdf_fidelity_and_no_invented_additional_solutions():
    report=grade_pdf()
    assert report['anchors_reviewed']==18
    assert report['executed']==2 and report['passed']==2 and report['failed']==0
    assert report['static_only']==2 and report['unsupported']==14
    assert report['classification_only']==16
    assert 'Not a blind' in report['method']
    assert json.loads(json.dumps(report,allow_nan=False))['source']['sha256']==SHA256


@pytest.mark.parametrize('case_id,frame_count',[('p08-multiple-21',7),('p09-multiple-27',5)])
def test_pdf_authored_viewer_html_keeps_source_page_and_force_energy(tmp_path,case_id,frame_count):
    target=tmp_path/(case_id+'.html')
    result=export_pdf(case_id,target,resolution=11)
    assert result['frames']==frame_count
    payload=json.loads(target.with_suffix('.json').read_text(encoding='utf-8'))
    assert payload['authorability_source']['filename']=='02_conductors_half_size.pdf'
    assert payload['authorability_source']['sha256']==SHA256
    assert payload['authorability_source']['page']==BY_ID[case_id].page
    assert 'extensions' in payload['authorability_source']['caveat']
    assert len(payload['position_sweep']['samples'])==frame_count
    assert all(frame['force_energy'] for frame in payload['frames'])
    html=target.read_text(encoding='utf-8')
    assert 'id="force-chart"' in html and 'id="energy-chart"' in html


def test_unsupported_source_cannot_fake_explorer(tmp_path):
    for case_id in ('p03-single-27','p05-single-47','p10-linked-13','p11-linked-28',
                    'p14-numerical-07','p15-archive-02'):
        with pytest.raises(ValueError,match='not a supported'):
            authored(case_id)
        path=tmp_path/(case_id+'.html')
        with pytest.raises(ValueError,match='unsupported PDF'):
            export_pdf(case_id,path)
        assert not path.exists()


def test_pdf_cli_grade_and_fail_closed(tmp_path):
    root=Path(__file__).resolve().parents[1]
    prefix=[sys.executable,'-m','m13c.pdf_cases']
    report=tmp_path/'report.json'
    finished=subprocess.run(prefix+['grade','--output',str(report)],cwd=root,
                            capture_output=True,text=True,timeout=80)
    assert finished.returncode==0,finished.stderr
    assert json.loads(report.read_text(encoding='utf-8'))['passed']==2
    invalid=tmp_path/'invalid.html'
    denied=subprocess.run(prefix+['render','p10-linked-13','--output',str(invalid)],
                          cwd=root,capture_output=True,text=True,timeout=20)
    assert denied.returncode==2 and not invalid.exists()
