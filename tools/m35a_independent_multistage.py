"""Standard-library-only checker of actual M35a native exported evidence.

No conductor_sim imports, production field callbacks or integrator as oracle.
These developer-authored equations do not replace a separate-context review.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from math import sqrt, hypot, isfinite, isclose
from pathlib import Path

# Separate teacher-input reference table, not read from a production callback.
# E, mass, source IDs active, seen, full graph-root world path.
B=('field:base',); S=B+('field:shared',); T=S+('field:temp',)
F=S+('field:final',); SEEN=T+('field:final',)
EXPECTED={
 'baseline':((180.,-30.,210.),2e-6,B,B,('baseline',)),
 'old-a':((220.,-25.,280.),2e-6,S,S,('baseline','old-a')),
 'old-b':((160.,-5.,230.),2e-6,S,S,('baseline','old-b')),
 'mass-a':((220.,-25.,280.),3e-6,S,S,('baseline','old-a','mass-a')),
 'extra-a':((250.,-25.,325.),3e-6,T,T,('baseline','old-a','mass-a','extra-a')),
 'replace-temp':((300.,-45.,360.),3e-6,F,SEEN,('baseline','old-a','mass-a','extra-a','replace-temp')),
 'mass-b':((160.,-5.,230.),4e-6,S,S,('baseline','old-b','mass-b')),
 'remove-final':((220.,-25.,280.),3e-6,S,SEEN,('baseline','old-a','mass-a','extra-a','replace-temp','remove-final')),
 'heavier':((160.,-5.,230.),5e-6,S,S,('baseline','old-b','mass-b','heavier')),
 'fresh-temp':((190.,-5.,275.),5e-6,T,T,('baseline','old-b','mass-b','heavier','fresh-temp')),
}
NAMES=('root','stage1','stage2','stage3','inquiry')


def check(condition,message):
    if not condition:raise AssertionError(message)


def number(value):
    check(type(value) in (int,float) and isfinite(value),'not a finite native physical number')
    return float(value)


def near(actual,expected,tol=1e-8):
    check(isclose(number(actual),expected,rel_tol=0.,abs_tol=tol),f'{actual!r} != {expected!r}')


def vector(value):
    check(type(value) in (tuple,list) and len(value)==3,'missing physical 3-vector')
    return tuple(number(x) for x in value)


def vnear(actual,expected,tol=1e-8):
    for a,b in zip(vector(actual),expected):near(a,b,tol)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                    allow_nan=False,ensure_ascii=True).encode('ascii')).hexdigest()


def state(name,t):
    electric,mass,*_=EXPECTED[name]
    force=tuple(1.2e-9*electric[k]+mass*(.03,-.02,.005)[k] for k in range(3))
    a=tuple(f/mass for f in force)
    pos=tuple((.003,.004,.0125)[k]+(.12,.03,.24)[k]*t+.5*a[k]*t*t for k in range(3))
    vel=tuple((.12,.03,.24)[k]+a[k]*t for k in range(3))
    return pos,vel,force,a


def root_time(name,x):
    ax=state(name,0.)[3][0]
    distance=x-.003
    return 2*distance/(.12+sqrt(.12*.12+2*ax*distance))


def model_check(name,model,cert,models):
    E,mass,active,seen,path=EXPECTED[name]
    spec=model['spec']
    check(all(s['kind']=='UniformElectricField' for s in spec['field']['sources']),'wrong source kind')
    actual=tuple(sum(number(s['field_v_m'][k]) for s in spec['field']['sources']) for k in range(3))
    vnear(actual,E,1e-12)
    near(spec['mass_kg'],mass,1e-16);near(spec['particle']['charge_c'],1.2e-9,1e-20)
    vnear(spec['particle']['position_m'],(.003,.004,.0125),1e-15)
    vnear(spec['initial_velocity_m_s'],(.12,.03,.24),1e-15)
    vnear(spec['gravity']['acceleration_m_s2'],(.03,-.02,.005),1e-15)
    near(spec['duration_s'],.7,1e-15)
    vnear(spec['plane_normal'],tuple(v/sqrt(29) for v in (3.,4.,-2.)),1e-15)
    check(tuple(x[0] for x in model['source_annotations'])==active,'wrong active source IDs')
    check(cert['world_id']==name and cert['model_revision_id']==model['revision_id'],'source identity mismatch')
    check(tuple(cert['world_path'])==path,'source graph-root path reset or replaced')
    check(tuple(cert['active_source_ids'])==active and tuple(cert['seen_source_ids'])==seen,'source lifetime history mismatch')
    check(tuple(cert['retired_source_ids'])==tuple(x for x in seen if x not in active),'retired source resurrection')
    check(cert['gravity_present'] is True,'gravity lost')
    if len(path)>1:
        check(cert['parent_world_id']==path[-2], 'wrong immediate parent world')
        check(cert['parent_revision_id']==models[path[-2]]['revision_id'],'wrong actual parent revision')


def native_check(name,native,model,x):
    check(native['schema']=='conductor_sim.event_run.v1','not a genuine event record')
    check(native['study']['model_revision_id']==model['revision_id'],'wrong observed model')
    check(native['trajectory']['termination']=='duration','must not infer a whole horizon from truncated data')
    samples=native['trajectory']['samples'];check(len(samples)>=20,'missing native samples')
    worst=0.;times=[]
    for sample in samples:
        check(type(sample['time_s']) is float,'Boolean/non-native saved SI clock')
        t=number(sample['time_s']); times.append(t)
        p,v,f,_=state(name,t)
        vnear(sample['position_m'],p,5e-8);vnear(sample['velocity_m_s'],v,5e-8)
        vnear(sample['force_n'],f,2e-12)
        worst=max(worst,hypot(*(sample['position_m'][k]-p[k] for k in range(3))))
    check(times[0]==0. and abs(times[-1]-.7)<1e-12 and all(b>a for a,b in zip(times,times[1:])),'invalid native horizon')
    events=native['events'];check(len(events)==1,'missing/ambiguous actual event')
    event=events[0]
    check(event['direction']==1 and native['outcomes'][event['event_id']]=='located','not a located positive root')
    check(type(event['sample']['time_s']) is float,'forged event time type')
    near(event['sample']['time_s'],root_time(name,x),2e-7)
    near(event['sample']['position_m'][0],x,2e-7)
    lo,hi=event['time_bracket_s'];check(lo<=event['sample']['time_s']<=hi,'root outside numerical bracket')
    return dict(samples=len(samples),times=times,numeric_root=event['sample']['time_s'],
                analytic_root=root_time(name,x),bracket=(lo,hi),max_position_error_m=worst)


def pair_check(pair,observed,models,x):
    left,right=pair['left_world_id'],pair['right_world_id']
    check(pair['numerical_orientation']=='right minus left','incorrect sign convention')
    if left=='resurrect-temp':
        check(pair['status']=='unpaired' and pair['left_revision_id'] is None,'refused source falsely compared')
        check(not pair['exact_time_trajectory_comparisons'],'refused world fabricated trajectory')
        return 0
    check(pair['status']=='compared','expected real common observations')
    check(pair['left_revision_id']==models[left]['revision_id'] and pair['right_revision_id']==models[right]['revision_id'],'pair model identities wrong')
    ltimes=set(observed[left]['times'])
    times=[t for t in observed[right]['times'] if t in ltimes]
    rows=pair['exact_time_trajectory_comparisons']
    check(len(rows)==len(times),'comparison did not use literal native clocks')
    for row,t in zip(rows,times):
        check(type(row['time_s']) is float and row['time_s']==t,'invented/Boolean comparison clock')
        lp,lv,*_=state(left,t);rp,rv,*_=state(right,t)
        vnear(row['delta_position_m'],tuple(rp[k]-lp[k] for k in range(3)),1e-7)
        vnear(row['delta_velocity_m_s'],tuple(rv[k]-lv[k] for k in range(3)),1e-7)
    event=pair['event_time_comparisons'][0]
    check(event['status']=='one-resolved-pair' and event['direction']==1,'incompatible event pair')
    numeric=observed[right]['numeric_root']-observed[left]['numeric_root']
    near(event['estimated_numerical_shift_s'],numeric,1e-14)
    near(event['estimated_numerical_shift_s'],root_time(right,x)-root_time(left,x),4e-7)
    bounds=(observed[right]['bracket'][0]-observed[left]['bracket'][1],
            observed[right]['bracket'][1]-observed[left]['bracket'][0])
    for a,b in zip(event['numerical_bracket_shift_interval_s'],bounds):near(a,b,1e-14)
    return len(rows)


def verify(data):
    root=data['root'];models={'baseline':root['plan']['original']};observed={}
    check(root['schema']=='conductor_sim.unified_observation_graph.v2','wrong root schema')
    model_check('baseline',models['baseline'],root['original_source_ancestry'],models)
    observed['baseline']=native_check('baseline',root['original_native_observation'],models['baseline'],.05)
    for row in root['nodes']:
        name=row['node']['node_id'];model=row['admitted_scenario']
        models[name]=model;model_check(name,model,row['source_ancestry'],models)
        observed[name]=native_check(name,row['native_observation'],model,.05)
    previous=root;stage_history=[];row_count=0
    expected_names=(('mass-a','extra-a'),('replace-temp','mass-b'),
                    ('resurrect-temp','remove-final','heavier','fresh-temp'))
    for i,stage in enumerate(('stage1','stage2','stage3'),1):
        run=data[stage]
        check(run['schema']==('conductor_sim.graph_world_fork.v1' if i==1 else 'conductor_sim.graph_world_fork.v2'),'wrong stage schema')
        check(run['source_graph_sha256']==digest(previous),'immediate source evidence digest mismatch')
        check(run['plan']['source_graph_sha256']==digest(previous),'plan source binding mismatch')
        if i>1:
            check(run['stage_number']==i and run['original_graph_sha256']==digest(root),'root history lost')
            check(run['prior_stages']==stage_history,'ordered stage hash chain broken')
        check(tuple(r['node']['node_id'] for r in run['nodes'])==expected_names[i-1],'changed stage world order')
        for row in run['nodes']:
            name=row['node']['node_id'];parent=row['node']['parent_id']
            check(row['parent_revision_id']==models[parent]['revision_id'],'wrong immediate source model')
            if name=='resurrect-temp':
                check(row['status']=='rejected-structural-model' and row['admitted_scenario'] is None
                      and row['native_observation'] is None and row['source_ancestry'] is None,'retired source was revived')
                continue
            check(row['status']=='complete','expected admitted whole source world')
            model=row['admitted_scenario'];models[name]=model
            model_check(name,model,row['source_ancestry'],models)
            observed[name]=native_check(name,row['native_observation'],model,.05)
        for pair in run['cross_world_comparisons']:
            row_count+=pair_check(pair,observed,models,.05)
        stage_history.append(dict(fork_id=run['plan']['fork_id'],run_sha256=digest(run),
                                  world_ids=list(expected_names[i-1])))
        previous=run
    inquiry=data['inquiry']
    check(inquiry['schema']=='conductor_sim.graph_event_inquiry.v2','inquiry not fork-capable v2')
    check(inquiry['plan']['source_graph_sha256']==digest(previous),'inquiry detached from latest native history')
    check(inquiry['plan']['original_graph_sha256']==digest(root),'inquiry original graph binding lost')
    check(inquiry['plan']['source_stages']==stage_history,'inquiry stage history differs')
    queried={}
    for row in inquiry['nodes']:
        name=row['world_id'];check(row['model_revision_id']==models[name]['revision_id'],'query changed physical model')
        queried[name]=native_check(name,row['native_event_run'],models[name],.07)
    for pair in inquiry['cross_world_comparisons']:
        row_count+=pair_check(pair,queried,models,.07)
    return dict(verdict='PASS',native_worlds=len(models),fresh_inquiries=len(queried),
        verified_saved_samples=sum(o['samples'] for o in observed.values()),
        literal_comparison_rows=row_count,max_position_error_m=max(o['max_position_error_m'] for o in observed.values()),
        conductor_sim_imports=0)


def negative_controls(data):
    edits=(
        lambda d:d['stage2'].__setitem__('source_graph_sha256','fake'),
        lambda d:d['stage2']['nodes'][0].__setitem__('parent_revision_id','fake'),
        lambda d:d['stage1']['nodes'][0]['native_observation']['trajectory']['samples'][0].__setitem__('time_s',False),
        lambda d:d['stage3']['nodes'][1]['source_ancestry'].__setitem__('retired_source_ids',[]),
        lambda d:d['stage3']['nodes'][0].__setitem__('status','complete'),
        lambda d:d['inquiry']['nodes'][1]['native_event_run']['events'][0]['sample'].__setitem__('time_s',.123),
        lambda d:d['stage3']['prior_stages'][0].__setitem__('run_sha256','fake'),
    )
    for edit in edits:
        changed=copy.deepcopy(data);edit(changed)
        try:verify(changed)
        except (AssertionError,ValueError,TypeError,KeyError):continue
        raise AssertionError('independent checker failed to detect deliberate data tampering')
    return len(edits)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=Path('output/m35a'))
    parser.add_argument('--output',type=Path,default=Path('output/m35a/independent.json'))
    args=parser.parse_args()
    data={name:json.loads((args.input/(name+'.json')).read_bytes()) for name in NAMES}
    result=verify(data);result['tamper_controls_rejected']=negative_controls(data)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
