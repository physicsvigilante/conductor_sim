"""Independent checks and mutation tests for the original M17c authoring trial."""
import ast
from dataclasses import replace
import json
from pathlib import Path
import pytest

from conductor_sim import (Grounded, FixedNetCharge, PointCharge, UnsupportedProblem)
from m17c import evaluate
from m17c.attempts import SUBMISSIONS, Submission
from m17c.questions import BY_ID
from m17c.reference import KEYS, K, force_energy, trajectory, convergence, norm, sub


@pytest.mark.parametrize('case_id',tuple(BY_ID))
def test_authored_problem_matches_independent_physics_or_specific_refusal(case_id):
    r=evaluate.grade_one(case_id,SUBMISSIONS[case_id])
    assert r['passed'],r['issues']


def test_open_trial_report_discloses_true_denominators_and_scope():
    r=evaluate.grade_all()
    assert (r['total'],r['passed'],r['trajectory_cases'],r['refusals'])==(12,12,8,4)
    assert not r['missing'] and not r['extra']
    assert 'not blind' in r['method'] and 'no new PDF' in r['source']
    assert sum(x.get('samples_checked',0) for x in r['results'])==168
    json.dumps(r,allow_nan=False)
    missing=dict(SUBMISSIONS);missing.pop('tilted-plane')
    assert evaluate.grade_all(missing)['missing']==['tilted-plane']
    extra={**SUBMISSIONS,'unknown':SUBMISSIONS['tilted-plane']}
    assert evaluate.grade_all(extra)['extra']==['unknown']


@pytest.mark.parametrize('mutation',['mass','charge','velocity','duration','radius','constraint','backend','permittivity'])
def test_bad_authoring_is_rejected_before_integration(mutation,monkeypatch):
    original=SUBMISSIONS['isolated-sphere'];s=original.motion;p=s.problem
    if mutation=='mass':s=replace(s,mass_kg=1000*s.mass_kg)
    elif mutation=='charge':s=replace(s,problem=replace(p,charges=(replace(p.charges[0],charge_c=-p.charges[0].charge_c),)))
    elif mutation=='velocity':s=replace(s,initial_velocity_m_s=(.04,.11,0.))
    elif mutation=='duration':s=replace(s,duration_s=.1)
    elif mutation=='radius':s=replace(s,problem=replace(p,conductors=(replace(p.conductors[0],radius_m=.05),)))
    elif mutation=='constraint':s=replace(s,problem=replace(p,conductors=(replace(p.conductors[0],constraint=Grounded()),)))
    elif mutation=='permittivity':s=replace(s,problem=replace(p,permittivity_f_per_m=2*p.permittivity_f_per_m))
    bad=replace(original,motion=s,backend='plane' if mutation=='backend' else original.backend)
    def no_integration(*args,**kwargs):raise AssertionError('model mismatch must fail before solving')
    monkeypatch.setattr(evaluate,'integrate_motion_2d',no_integration)
    assert not evaluate.grade_one('isolated-sphere',bad)['passed']


@pytest.mark.parametrize('field',['time_s','position_m','velocity_m_s','force_n','acceleration_m_s2',
                                 'kinetic_energy_j','interaction_energy_j','total_energy_j','energy_drift_j','nan'])
def test_numerical_grader_detects_tampering_not_just_energy_conservation(field,monkeypatch):
    original=evaluate.integrate_motion_2d
    def altered(*args,**kwargs):
        r=original(*args,**kwargs);row=r.samples[7]
        if field=='nan':row=replace(row,force_n=(float('nan'),0.,0.))
        else:
            value=(99.,0.,0.) if field in ('position_m','velocity_m_s','force_n','acceleration_m_s2') else 99.
            row=replace(row,**{field:value})
        return replace(r,samples=(*r.samples[:7],row,*r.samples[8:]))
    monkeypatch.setattr(evaluate,'integrate_motion_2d',altered)
    assert not evaluate.grade_one('grounded-sphere',SUBMISSIONS['grounded-sphere'])['passed']


def test_references_have_no_engine_import_and_authors_have_no_grading_import():
    folder=Path(__file__).resolve().parents[1]/'m17c'
    for name,forbidden in [('reference.py',('conductor_sim','attempts')),
                            ('attempts.py',('reference','evaluate'))]:
        tree=ast.parse((folder/name).read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            modules=([node.module or ''] if isinstance(node,ast.ImportFrom) else
                     [a.name for a in node.names] if isinstance(node,ast.Import) else [])
            assert all(not any(word in module for word in forbidden) for module in modules)


def test_reference_force_is_independent_energy_gradient_and_verlet_converges():
    for cid in ('tilted-plane','grounded-sphere','isolated-sphere','cavity-oblique','nested-cavity'):
        key=KEYS[cid];p=key.position;force,U=force_energy(key,p)
        for i in range(3):
            delta=1e-6
            plus=tuple(p[j]+(delta if i==j else 0) for j in range(3))
            minus=tuple(p[j]-(delta if i==j else 0) for j in range(3))
            derivative=-(force_energy(key,plus)[1]-force_energy(key,minus)[1])/(2*delta)
            assert derivative==pytest.approx(force[i],rel=2e-6,abs=1e-17)
        c=convergence(cid)
        assert c['position_difference_m']<1e-9 and c['velocity_difference_m_s']<1e-8
    key=replace(KEYS['cavity-oblique'],position=(0.,0.,0.))
    F,U=force_energy(key,key.position)
    assert F==(0.,0.,0.)
    assert U==pytest.approx(-K*key.charge**2/(2*.12),rel=1e-13)
    assert trajectory('centre-crossing')[-1]['position_m'][0]<0


def test_refusal_is_not_replaced_by_a_wrong_supported_problem():
    assert not evaluate.grade_one('finite-plate-edge',SUBMISSIONS['electron-plates'])['passed']
    assert not evaluate.grade_one('numerical-femm',Submission(None,reason='unsupported'))['passed']
    s=SUBMISSIONS['grounded-sphere'].motion
    with pytest.raises(UnsupportedProblem,match='share the physically invariant plane'):
        replace(s,problem=replace(s.problem,charges=(*s.problem.charges,PointCharge(1e-9,(.5,0,0)))))


def test_render_preserves_full_production_physics_and_source_prompt(tmp_path):
    for cid in ('tilted-plane','rotated-radial'):
        output=tmp_path/(cid+'.html')
        b=evaluate.export(cid,output,resolution=9,visual_frames=4)
        assert b['authorability_source']['id']==cid
        assert b['authorability_source']['prompt']==BY_ID[cid].prompt
        assert len(b['frames'])==4 and len(b['motion_2d']['samples'])==21
        assert 'motion_1d' not in b and 'position_sweep' not in b
        html=output.read_text(encoding='utf-8')
        data=json.loads(html.split('<script id="simulation-data" type="application/json">')[1].split('</script>')[0])
        assert data==json.loads(output.with_suffix('.json').read_text(encoding='utf-8'))
    for cid in ('two-moving','finite-plate-edge','varying-voltage','numerical-femm'):
        output=tmp_path/(cid+'.html')
        with pytest.raises(ValueError):evaluate.export(cid,output)
        assert not output.exists()
