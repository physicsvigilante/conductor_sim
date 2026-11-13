"""Original multi-source authoring, independent references and negative controls."""
import ast
from dataclasses import replace
import json
from math import fsum
from pathlib import Path
import pytest

from conductor_sim import (FixedPotential, Grounded, GroundedPlaneBackend,
    PointCharge, UnsupportedProblem, integrate_motion_2d)
from m18c import evaluate
from m18c.attempts import SUBMISSIONS, Submission
from m18c.questions import BY_ID
from m18c.reference import KEYS, force_energy, convergence, norm, sub, K

SUPPORTED=tuple(k for k,v in KEYS.items() if v.backend)

@pytest.mark.parametrize('case_id',tuple(BY_ID))
def test_original_prompt_composition_matches_independent_physics(case_id):
    report=evaluate.grade_one(case_id,SUBMISSIONS[case_id])
    assert report['passed'],report['issues']


def test_trial_denominators_and_unexpected_inventory_fail_closed():
    report=evaluate.grade_all()
    assert (report['total'],report['passed'],report['trajectory_cases'],report['refusals'])==(12,12,8,4)
    assert report['accepted'] and not report['missing'] and not report['extra']
    assert sum(r.get('samples_checked',0) for r in report['results'])==168
    assert 'not blind' in report['method'] and 'no new PDF' in report['source']
    json.dumps(report,allow_nan=False)
    missing=dict(SUBMISSIONS);missing.pop('plane-three')
    r=evaluate.grade_all(missing)
    assert not r['accepted'] and r['passed']==11 and r['missing']==['plane-three']
    r=evaluate.grade_all({**SUBMISSIONS,'extra':SUBMISSIONS['plane-three']})
    assert not r['accepted'] and r['extra']==['extra']


@pytest.mark.parametrize('mutation',('mass','moving-index','moving-sign','held-sign','held-position',
    'missing-held','velocity','duration','radius','constraint','backend','permittivity'))
def test_bad_authored_models_fail_before_integration(mutation,monkeypatch):
    submission=SUBMISSIONS['isolated-sphere'];s=submission.motion;p=s.problem
    if mutation=='mass':s=replace(s,mass_kg=s.mass_kg*1000)
    elif mutation=='moving-index':s=replace(s,charge_index=0)
    elif mutation in ('moving-sign','held-sign'):
        i=1 if mutation=='moving-sign' else 0;charges=list(p.charges)
        charges[i]=replace(charges[i],charge_c=-charges[i].charge_c)
        s=replace(s,problem=replace(p,charges=tuple(charges)))
    elif mutation=='held-position':
        s=replace(s,problem=replace(p,charges=(replace(p.charges[0],position_m=(.23,-.05,0)),*p.charges[1:])))
    elif mutation=='missing-held':s=replace(s,problem=replace(p,charges=p.charges[:2]))
    elif mutation=='velocity':s=replace(s,initial_velocity_m_s=(.03,.10,0))
    elif mutation=='duration':s=replace(s,duration_s=.07)
    elif mutation=='radius':s=replace(s,problem=replace(p,conductors=(replace(p.conductors[0],radius_m=.05),)))
    elif mutation=='constraint':s=replace(s,problem=replace(p,conductors=(replace(p.conductors[0],constraint=Grounded()),)))
    elif mutation=='permittivity':s=replace(s,problem=replace(p,permittivity_f_per_m=2*p.permittivity_f_per_m))
    bad=replace(submission,motion=s,backend='plane' if mutation=='backend' else submission.backend)
    def forbidden(*a,**kw):raise AssertionError('bad model reached production integration')
    monkeypatch.setattr(evaluate,'integrate_motion_2d',forbidden)
    assert not evaluate.grade_one('isolated-sphere',bad)['passed']


@pytest.mark.parametrize('field',('time_s','position_m','displacement_m','velocity_m_s','force_n',
    'acceleration_m_s2','kinetic_energy_j','interaction_energy_j','total_energy_j','energy_drift_j','nan','max-drift'))
def test_grader_catches_corrupted_non_frame_rows(field,monkeypatch):
    original=evaluate.integrate_motion_2d
    def altered(*args,**kwargs):
        result=original(*args,**kwargs)
        if field=='max-drift':return replace(result,max_abs_energy_drift_j=1.)
        row=result.samples[7]
        if field=='nan':row=replace(row,force_n=(float('nan'),0,0))
        else:
            value=(99.,0,0) if field in ('position_m','displacement_m','velocity_m_s','force_n','acceleration_m_s2') else 99.
            row=replace(row,**{field:value})
        return replace(result,samples=(*result.samples[:7],row,*result.samples[8:]))
    monkeypatch.setattr(evaluate,'integrate_motion_2d',altered)
    assert not evaluate.grade_one('plane-three',SUBMISSIONS['plane-three'])['passed']


@pytest.mark.parametrize('case_id',SUPPORTED)
def test_independent_reference_gradient_and_step_halving(case_id):
    key=KEYS[case_id];p=key.sources[key.moving][1];F,U=force_energy(key,p)
    for i in range(3):
        h=1e-6;plus=list(p);minus=list(p);plus[i]+=h;minus[i]-=h
        gradient=-(force_energy(key,plus)[1]-force_energy(key,minus)[1])/(2*h)
        assert gradient==pytest.approx(F[i],rel=3e-6,abs=1e-17)
    c=convergence(case_id)
    assert c['position_difference_m']<1e-9 and c['velocity_difference_m_s']<1e-8
    if case_id=='biased-plates':
        assert c['image_force_difference_n']<1e-14
        assert c['image_energy_difference_j']<1e-15


def test_reference_known_single_source_limits_do_not_depend_on_production():
    plane=replace(KEYS['plane-three'],sources=((1e-9,(0,0,.1)),),moving=0)
    F,U=force_energy(plane,(0,0,.1))
    assert F==pytest.approx((0,0,-K*1e-18/(4*.1**2)),rel=1e-13)
    assert U==pytest.approx(-K*1e-18/(4*.1),rel=1e-13)
    cavity=replace(KEYS['centre-release'],sources=((1e-9,(0,0,0)),),moving=0)
    F,U=force_energy(cavity,(0,0,0))
    assert F==(0.,0.,0.) and U==pytest.approx(-K*1e-18/(2*.12),rel=1e-13)


def test_reference_and_submission_dependencies_are_separated():
    root=Path(__file__).resolve().parents[1]/'m18c'
    for name,forbidden in [('reference.py',('conductor_sim','attempts','m17c')),
                            ('attempts.py',('reference','evaluate','m17c'))]:
        for node in ast.walk(ast.parse((root/name).read_text(encoding='utf-8'))):
            modules=([node.module or ''] if isinstance(node,ast.ImportFrom) else
                     [v.name for v in node.names] if isinstance(node,ast.Import) else [])
            assert all(not any(x in m for x in forbidden) for m in modules)


def test_reordered_authored_sources_keep_physics_and_stable_moving_identity():
    s=SUBMISSIONS['plane-three'].motion;p=s.problem
    other=replace(s,problem=replace(p,charges=(p.charges[2],p.charges[0],p.charges[1])),charge_index=2)
    a=integrate_motion_2d(s,GroundedPlaneBackend());b=integrate_motion_2d(other,GroundedPlaneBackend())
    assert len(a.samples)==len(b.samples)==21
    for x,y in zip(a.samples,b.samples):
        assert x.position_m==pytest.approx(y.position_m,abs=1e-12)
        assert x.velocity_m_s==pytest.approx(y.velocity_m_s,abs=1e-12)
        assert x.force_n==pytest.approx(y.force_n,rel=1e-12,abs=1e-25)
        assert x.interaction_energy_j==pytest.approx(y.interaction_energy_j,rel=1e-12,abs=1e-25)
    bmeta=b.as_dict()
    assert bmeta['moving_charge_index']==2
    assert [q['charge_index'] for q in bmeta['stationary_charges']]==[0,1]


def test_rigid_rotation_translation_preserves_full_held_source_motion():
    s=SUBMISSIONS['plane-three'].motion
    def R(v):return (v[0],.8*v[1]-.6*v[2],.6*v[1]+.8*v[2])
    def T(v):return tuple(x+y for x,y in zip(R(v),(.3,-.4,.2)))
    p=replace(s.problem,charges=tuple(replace(q,position_m=T(q.position_m)) for q in s.problem.charges),
        conductors=tuple(replace(c,point_m=T(c.point_m),normal=R(c.normal)) for c in s.problem.conductors))
    rotated=replace(s,problem=p,initial_velocity_m_s=R(s.initial_velocity_m_s))
    a=integrate_motion_2d(s,GroundedPlaneBackend());b=integrate_motion_2d(rotated,GroundedPlaneBackend())
    for x,y in zip(a.samples,b.samples):
        assert T(x.position_m)==pytest.approx(y.position_m,abs=1e-11)
        assert R(x.velocity_m_s)==pytest.approx(y.velocity_m_s,abs=1e-11)
        assert R(x.force_n)==pytest.approx(y.force_n,rel=1e-10,abs=1e-23)
        assert x.interaction_energy_j==pytest.approx(y.interaction_energy_j,rel=1e-11)


def test_voltage_gauge_shift_changes_energy_offset_not_motion():
    s=SUBMISSIONS['biased-plates'].motion;shift=17.
    other=replace(s,problem=replace(s.problem,conductors=tuple(
        replace(c,constraint=FixedPotential(c.constraint.potential_v+shift)) for c in s.problem.conductors)))
    backend=evaluate.BACKENDS['plates']()
    a=integrate_motion_2d(s,backend);b=integrate_motion_2d(other,backend)
    offset=shift*fsum(q.charge_c for q in s.problem.charges)
    for x,y in zip(a.samples,b.samples):
        assert x.position_m==pytest.approx(y.position_m,abs=1e-12)
        assert x.force_n==pytest.approx(y.force_n,rel=1e-12,abs=1e-25)
        assert y.interaction_energy_j-x.interaction_energy_j==pytest.approx(offset,rel=1e-11)
        assert x.energy_drift_j==pytest.approx(y.energy_drift_j,abs=1e-23)


def test_held_charge_changes_real_trajectory_not_only_display():
    s=SUBMISSIONS['plane-three'].motion
    lone=replace(s,problem=replace(s.problem,charges=(s.selected_charge,)),charge_index=0)
    a=integrate_motion_2d(s,GroundedPlaneBackend());b=integrate_motion_2d(lone,GroundedPlaneBackend())
    assert norm(sub(a.samples[-1].position_m,b.samples[-1].position_m))>1e-5
    assert s.problem.charges[0].position_m==(-.055,0.,.14)


def test_refusals_are_specific_and_do_not_replace_requested_physics():
    for cid in ('two-moving','nonplanar','numerical-femm','varying-voltage'):
        assert not evaluate.grade_one(cid,Submission(None,reason='unsupported'))['passed']
        assert not evaluate.grade_one(cid,SUBMISSIONS['plane-three'])['passed']
    s=SUBMISSIONS['grounded-sphere'].motion
    bad=replace(s.problem,charges=(s.selected_charge,PointCharge(1e-10,(.43,-.02,.33))))
    with pytest.raises(UnsupportedProblem,match='invariant plane'):replace(s,problem=bad)


@pytest.mark.parametrize('case_id',('plane-three','nested-cavity','centre-release'))
def test_export_uses_production_only_and_preserves_all_identities(case_id,tmp_path,monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('reference calculation used to render production data')
    monkeypatch.setattr(evaluate,'trajectory',forbidden);monkeypatch.setattr(evaluate,'force_energy',forbidden)
    path=tmp_path/(case_id+'.html');b=evaluate.export(case_id,path,resolution=9,visual_frames=4)
    s=SUBMISSIONS[case_id].motion;meta=b['motion_2d'];fixed=[i for i in range(len(s.problem.charges)) if i!=s.charge_index]
    assert len(meta['samples'])==21 and len(b['frames'])==4
    assert meta['moving_charge_index']==s.charge_index
    assert b['authorability_source']['prompt']==BY_ID[case_id].prompt
    for frame,idx in zip(b['frames'],meta['frame_sample_indices']):
        sources={q['charge_index']:q for q in frame['charges']}
        assert sources[s.charge_index]['position_m']==meta['samples'][idx]['position_m']
        for i in fixed:
            assert sources[i]['position_m']==list(s.problem.charges[i].position_m)
            assert sources[i]['motion_role']=='held'
    embedded=path.read_text(encoding='utf-8').split('<script id="simulation-data" type="application/json">')[1].split('</script>')[0]
    assert json.loads(embedded)==json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))


def test_unsupported_export_preserves_existing_file(tmp_path):
    path=tmp_path/'existing.html';path.write_text('retained',encoding='utf-8')
    with pytest.raises(ValueError):evaluate.export('two-moving',path)
    assert path.read_text(encoding='utf-8')=='retained'
    assert not path.with_suffix('.json').exists()


def test_independent_green_kernels_satisfy_grounded_boundaries_and_reciprocity():
    from m18c.reference import _kernel
    for cid,r,s,boundary in (
        ('plane-three',(.01,0,.10),(-.055,0,.14),(.03,.02,0)),
        ('grounded-sphere',(.17,0,0),(.23,.08,0),(.06,0,0)),
        ('cavity-three',(.01,.035,0),(-.045,.02,0),(0,.12,0)),
    ):
        key=KEYS[cid]
        assert _kernel(key,r,s,128)[0]==pytest.approx(_kernel(key,s,r,128)[0],rel=1e-13)
        assert 1/norm(sub(boundary,s))+_kernel(key,boundary,s,128)[0]==pytest.approx(0.,abs=1e-12)


def test_grader_rejects_conserved_energy_with_held_potential_omitted():
    submission=SUBMISSIONS['plane-three'];r=integrate_motion_2d(submission.motion,GroundedPlaneBackend())
    # A constant offset preserves forces, trajectories and energy drift, but
    # contradicts the explicitly documented full-configuration U convention.
    rows=tuple(replace(x,interaction_energy_j=x.interaction_energy_j+1e-8,
        total_energy_j=x.total_energy_j+1e-8) for x in r.samples)
    report=evaluate.grade_result('plane-three',submission,replace(r,samples=rows))
    assert not report['passed']
    assert any('interaction_energy_j mismatch' in issue for issue in report['issues'])


def test_grader_rejects_fake_held_charge_metadata_over_single_charge_trajectory():
    submission=SUBMISSIONS['plane-three'];s=submission.motion
    lone=replace(s,problem=replace(s.problem,charges=(s.selected_charge,)),charge_index=0)
    r=integrate_motion_2d(lone,GroundedPlaneBackend())
    report=evaluate.grade_result('plane-three',submission,replace(r,spec=s))
    assert not report['passed']
    assert any('independent trajectory mismatch' in issue for issue in report['issues'])
