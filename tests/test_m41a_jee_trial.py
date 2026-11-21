"""M41a: source and denominator integrity, native science and hostile evidence."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pytest

from tools.m41a_jee_trial import COHORT, REFUSALS, SCHEMA, SOURCE_PDFS, check_corpus, check_trial, generate
from tools.m41a_jee_trial_reference import check as check_p61


@pytest.fixture(scope='module')
def evidence(tmp_path_factory):
    directory = tmp_path_factory.mktemp('m41a_trial')
    report=generate(directory)
    return directory,report


def test_frozen_corpus_provenance():
    assert check_corpus()==SOURCE_PDFS
    assert len(COHORT)==14 and len({x[0] for x in COHORT})==14


def test_native_and_scoped_reference_routes(evidence):
    directory,report=evidence
    check_trial(report,directory)
    assert report['registered_source_anchor_denominator']==14
    assert report['physically_native_and_independently_referenced_scoped_routes']==12
    assert report['explicit_unsupported_original_geometry_count']==2
    assert report['measured_educational_success_cases']==0
    assert report['actual_browser_verified_count_in_this_run']==0
    assert all(x['educational_usefulness']=='not_evaluated_with_students' for x in report['cases'])
    assert all(x['real_chromium_status']=='not_measured_by_this_trial' for x in report['cases'])
    assert all((directory/x['native_route']).is_dir() for x in report['cases'] if x['native_route']!='refuse')


def test_unsupported_real_irregular_conductors_not_replaced_by_spheres(evidence):
    _,r=evidence
    rows={x['source_id']:x for x in r['cases']}
    assert set(REFUSALS)=={'p11-linked-28','p70-cavity-offset'}
    for ident in REFUSALS:
        assert rows[ident]['native_physical_status']=='unsupported_original_geometry'
        assert rows[ident]['visual_html_status']=='not_generated'
        assert rows[ident]['qualified_refusal']


def test_2021_official_crosswalk_preserves_actually_printed_workbook_caption(evidence):
    _,report=evidence
    case=next(x for x in report['cases'] if x['source_id'].startswith('jee2021'))
    assert case['pdf_visible_caption']=='Paragraph for Questions 3 and 4'
    assert 'historical' in case['official_paper_crosswalk_status']
    assert 'one apparatus' in case['source_kind']
    assert case['textbook_answer_completeness']=='not_claimed_as_entire_original_answer'


def test_p61_static_source_givens_are_not_teacher_numbers(evidence):
    directory,r=evidence
    x=json.loads((directory/'p61/p61-native.json').read_text())
    out=check_p61(x)
    assert out['qualified_static_grounded_sphere'] is True
    assert out['independent_teacher_final_charge_c']==pytest.approx(-3.125e-10,rel=1e-14)
    assert x['native_sphere_net_charge_c']==pytest.approx(-3.125e-10,rel=1e-14)
    assert x['source_symbolic_final_charge']=='-Q*R/r'
    assert x['not_physical_time_evolution'] is True
    page=(directory/'p61/p61-static-teacher.html').read_text()
    assert 'Teacher-authored example, NOT source numbers' in page
    assert 'no physical switch current' in page
    assert "default-src 'none'" in page
    assert 'id="p61-final-charge"' in page


@pytest.mark.parametrize('field,value',[
    ('source_id','p02-single-19'),('pdf_page',False),('pdf','POTENTIAL_CONDUCTORS1.pdf'),
    ('printed_anchor','Question 999'),('native_route','m36c'),
    ('native_physical_status','verified_scoped_native_faked'),
    ('native_reference_status','not_run'),('real_chromium_status','PASS'),
    ('visual_html_status','actual_browser_pass'),
    ('educational_usefulness','improved_jee_scores'),
    ('textbook_answer_completeness','full_original_answer_verified'),
    ('source_kind','official exam question'),('source_givens_vs_teacher','all numbers printed'),
    ('pdf_visible_caption','wrong'),('official_paper_crosswalk_status','new official exam evidence'),
])
def test_forged_single_case_refused(evidence,field,value):
    directory,orig=evidence
    record=deepcopy(orig)
    record['cases'][0][field]=value
    with pytest.raises(ValueError):
        check_trial(record,directory)


@pytest.mark.parametrize('field,bad',[
    ('frozen_charter_main_commit','invented'),
    ('registered_source_anchor_denominator',True),
    ('registered_source_anchor_denominator',13),
    ('physically_native_and_independently_referenced_scoped_routes',14),
    ('explicit_unsupported_original_geometry_count',False),
    ('measured_educational_success_cases',True),
    ('actual_browser_verified_count_in_this_run',1),
    ('automatic_full_original_jee_question_solved_score','12/14 = 86% JEE success'),
])
def test_denominator_or_score_forgery_refused(evidence,field,bad):
    directory,orig=evidence
    record=deepcopy(orig)
    record[field]=bad
    with pytest.raises(ValueError):
        check_trial(record,directory)


def test_missing_or_duplicated_source_case_refused(evidence):
    directory,orig=evidence
    for change in ['deleted','duplicate','swapped']:
        data=deepcopy(orig)
        if change=='deleted': data['cases'].pop()
        if change=='duplicate': data['cases'][-1]=deepcopy(data['cases'][-2])
        if change=='swapped': data['cases'][0],data['cases'][1]=data['cases'][1],data['cases'][0]
        with pytest.raises(ValueError): check_trial(data,directory)


def test_unsupported_source_renaming_refused(evidence):
    directory,orig=evidence
    for ident in REFUSALS:
        data=deepcopy(orig)
        row=next(x for x in data['cases'] if x['source_id']==ident)
        row['native_physical_status']='verified_scoped_native'
        with pytest.raises(ValueError): check_trial(data,directory)


def test_substituted_artifact_digest_and_removed_artifact_refused(evidence):
    directory,orig=evidence
    for route in ['m36b','m36c','m36d','m37b','p61']:
        for key in ['native_sha256','separate_reference_sha256','native_path']:
            data=deepcopy(orig)
            data['evidence_files'][route][key]=('0'*64 if key.endswith('sha256') else '../not-real.json')
            with pytest.raises(ValueError): check_trial(data,directory)


@pytest.mark.parametrize('mutation',[
  ('source_id','p60-fake-grounded'),
  ('source_pdf_page',False), ('source_pdf_page',61.0),
  ('original_asked','What is the switch current at t=1 s?'),
  ('not_physical_time_evolution',False),
  ('source_symbolic_final_charge','-Q*R/(r-R)'),
  ('native_sphere_net_charge_c',False),
  ('native_sphere_potential_v',0.01),
])
def test_forged_p61_source_and_si_refused(evidence,mutation):
    directory,_=evidence
    p=json.loads((directory/'p61/p61-native.json').read_text())
    field,value=mutation
    p[field]=value
    with pytest.raises(ValueError): check_p61(p)


def test_p61_forged_signed_density_and_bool_q_refused(evidence):
    directory,_=evidence
    native=json.loads((directory/'p61/p61-native.json').read_text())
    for case in ['charge','sigma','angle','source-position','boolean-coordinate']:
        data=deepcopy(native)
        if case=='boolean-coordinate': data['teacher_numeric_values_not_printed']['sphere_center_m'][2]=False
        if case=='charge': data['teacher_numeric_values_not_printed']['Q_c']=True
        if case=='sigma': data['native_signed_sigma_samples'][0]['sigma_c_per_m2']*=-1
        if case=='angle': data['native_signed_sigma_samples'][0]['theta_rad']=False
        if case=='source-position': data['teacher_numeric_values_not_printed']['charge_position_m'][0]=-.31
        with pytest.raises(ValueError): check_p61(data)


def test_native_file_cannot_stand_in_for_reference(evidence):
    directory, original = evidence
    report = deepcopy(original)
    files = report['evidence_files']['p61']
    files['separate_reference_path'] = files['native_path']
    files['separate_reference_sha256'] = files['native_sha256']
    with pytest.raises(ValueError):
        check_trial(report, directory)


def test_rehashed_wrong_signed_native_charge_rejected(evidence, tmp_path):
    import shutil

    directory, original = evidence
    report = deepcopy(original)
    files = report['evidence_files']
    for record in files.values():
        for field in ('native_path', 'separate_reference_path'):
            path = Path(record[field])
            (tmp_path / path.parent).mkdir(exist_ok=True)
            shutil.copyfile(directory / path, tmp_path / path)
    path = tmp_path / files['p61']['native_path']
    data = json.loads(path.read_text(encoding='utf-8'))
    data['native_sphere_net_charge_c'] *= -1
    path.write_text(json.dumps(data), encoding='utf-8')
    files['p61']['native_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='induced Kelvin net charge'):
        check_trial(report, tmp_path)


def test_tiny_opposite_sign_teacher_example_rejected(evidence):
    directory, _ = evidence
    case = json.loads((directory / 'p61/p61-native.json').read_text(encoding='utf-8'))
    case['teacher_numeric_values_not_printed']['Q_c'] = 1e-31
    for field in ('native_sphere_net_charge_c', 'native_image_charge_c'):
        case[field] *= -1e-22
    for point in case['native_signed_sigma_samples']:
        point['sigma_c_per_m2'] *= -1e-22
    with pytest.raises(ValueError, match='teacher example parameters'):
        check_p61(case)
