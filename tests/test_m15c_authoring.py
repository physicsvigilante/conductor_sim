"""M15c open authoring trial and independent scoring safeguards."""
from dataclasses import replace
from math import pi
import ast
import json
from pathlib import Path
import pytest

from conductor_sim import (ElectrostaticProblem, Grounded, PointCharge,
                           PositionSweepSpec, SphericalShell)
from conductor_sim.solver import UnsupportedProblem
from m15c import evaluate
from m15c.attempts import SUBMISSIONS
from m15c.questions import BY_ID
from m15c.reference import KEYS, observe, work_integral, K


@pytest.mark.parametrize('case_id',tuple(BY_ID))
def test_open_authored_case_or_specific_refusal(case_id):
    result=evaluate.grade_one(case_id,SUBMISSIONS[case_id])
    assert result['passed'],result['issues']


def test_report_discloses_denominators_and_open_method():
    report=evaluate.grade_all()
    assert (report['total'],report['passed'],report['analytical_cases'],report['refusals'])==(12,12,8,4)
    assert not report['missing'] and not report['extra']
    assert 'not blind' in report['method']
    json.dumps(report,allow_nan=False)
    bad=evaluate.grade_all({**SUBMISSIONS,'unexpected':SUBMISSIONS['centre-outward']})
    assert bad['extra']==['unexpected']


@pytest.mark.parametrize('change', ['charge','constraint','units','endpoint','backend','identity'])
def test_wrong_authored_model_is_rejected_before_sampling(change,monkeypatch):
    original=SUBMISSIONS['negative-inward'];spec=original.sweep;p=spec.problem
    if change=='charge':
        bad=replace(original,sweep=replace(spec,problem=replace(p,charges=(replace(p.charges[0],charge_c=1e-9),))))
    elif change=='constraint':
        from conductor_sim import FixedNetCharge
        bad=replace(original,sweep=replace(spec,problem=replace(p,
            conductors=(replace(p.conductors[0],constraint=FixedNetCharge(0.)),))))
    elif change=='units':
        bad=replace(original,sweep=replace(spec,problem=replace(p,permittivity_f_per_m=2*p.permittivity_f_per_m)))
    elif change=='endpoint':
        bad=replace(original,sweep=replace(spec,end_m=(.03,0,0)))
    elif change=='backend':
        bad=replace(original,backend='nested')
    else:
        bad=replace(original,sweep=None)
    def no_sampling(*a,**k):raise AssertionError('wrong model must not be sampled')
    monkeypatch.setattr(evaluate,'sample_position_sweep',no_sampling)
    assert not evaluate.grade_one('negative-inward',bad)['passed']


@pytest.mark.parametrize('field',['force_n','energy_induced_j','external_quasistatic_work_j'])
def test_numerical_grader_detects_corrupted_observations(field,monkeypatch):
    real=evaluate.sample_position_sweep
    def altered(*a,**k):
        b=real(*a,**k)
        row=b['position_sweep']['samples'][-1]
        row[field]=[1.,2.,3.] if field=='force_n' else 123.
        return b
    monkeypatch.setattr(evaluate,'sample_position_sweep',altered)
    assert not evaluate.grade_one('negative-inward',SUBMISSIONS['negative-inward'])['passed']


def test_reference_has_closed_form_anchor_and_independent_work_quadrature():
    key=KEYS['centre-outward'];q=key.sources[0][0];a=key.shells[0][0]
    start=observe(key,0.)
    assert start['force_n']==(0.,0.,0.)
    assert start['energy_j']==pytest.approx(-K*q*q/(2*a),rel=1e-12)
    r=key.end[2]
    end=observe(key,1.)
    assert end['force_n'][2]==pytest.approx(K*q*q*a*r/(a*a-r*r)**2,rel=1e-12)
    assert end['energy_j']==pytest.approx(-K*q*q*a/(2*(a*a-r*r)),rel=1e-12)
    assert work_integral(key)==pytest.approx(end['energy_j']-start['energy_j'],rel=2e-7)


def test_reference_never_imports_engine_and_authorship_never_imports_grading_key():
    base=Path(__file__).parents[1]/'m15c'
    for file,banned in [('reference.py','conductor_sim'),('attempts.py','reference')]:
        tree=ast.parse((base/file).read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):
                assert banned not in (node.module or '')
            elif isinstance(node,ast.Import):
                assert all(banned not in a.name for a in node.names)


def test_invalid_vacuum_paths_are_really_rejected_not_just_labelled():
    shell=SphericalShell(.1,.12,Grounded())
    p=ElectrostaticProblem((PointCharge(1e-9,(0,0,.02)),),(shell,))
    with pytest.raises(UnsupportedProblem,match='cavity'):
        PositionSweepSpec(p,0,(0,0,.14),5)
    p=replace(p,charges=(PointCharge(1e-9,(0,0,-.04)),PointCharge(-1e-9,(0,0,0))))
    with pytest.raises(UnsupportedProblem,match='intersects'):
        PositionSweepSpec(p,0,(0,0,.04),5)


def test_linked_then_disconnected_case_uses_resolved_conductor_charges():
    p=SUBMISSIONS['connect-disconnect'].sweep.problem
    assert [c.constraint.charge_c for c in p.conductors]==pytest.approx([-.6e-9,.8e-9])
    assert p.charges[0].charge_c==.6e-9
    assert evaluate.grade_one('connect-disconnect',SUBMISSIONS['connect-disconnect'])['passed']


def test_export_reuses_existing_schema_and_keeps_prompt_metadata(tmp_path):
    p=tmp_path/'negative.html'
    bundle=evaluate.export('negative-inward',p,resolution=11)
    assert len(bundle['frames'])==5
    assert bundle['authorability_source']['id']=='negative-inward'
    assert bundle['schema']=='conductor_sim.visualization.v1'
    assert all(f['force_energy'] for f in bundle['frames'])
    assert json.loads(p.with_suffix('.json').read_text(encoding='utf-8'))['position_sweep']
    for case in ('cross-metal','source-collision','cavity-time-motion','numerical-cavity-force'):
        with pytest.raises(ValueError):
            evaluate.export(case,tmp_path/(case+'.html'))
        assert not (tmp_path/(case+'.html')).exists()
