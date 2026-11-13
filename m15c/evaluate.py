"""Replay an open authoring trial; no hidden-test or arbitrary-LLM claim."""
import argparse
from dataclasses import asdict
from math import fsum, isclose
from pathlib import Path
import json
from conductor_sim import (FixedNetCharge, Grounded, MultiImageShellBackend,
                           NestedShellBackend, sample_position_sweep)
from conductor_sim.visualization import write_visualization
from .questions import BY_ID
from .reference import KEYS, observe, work_integral
from .attempts import SUBMISSIONS, Submission

BACKENDS={'single':MultiImageShellBackend,'nested':NestedShellBackend}
ENERGIES=('energy_direct_j','energy_induced_j','energy_background_j')

def _close(x,y,abs_tol=1e-20):
    return isclose(x,y,rel_tol=2e-10,abs_tol=abs_tol)

def _same_vector(a,b,abs_tol=1e-13):
    return len(a)==len(b) and all(_close(x,y,abs_tol) for x,y in zip(a,b))

def model_matches(submission,key):
    """Validate authored units, geometry and constraints before scoring physics."""
    try:
        spec=submission.sweep;p=spec.problem
        if (spec.charge_index!=0 or spec.steps!=5 or len(p.charges)!=len(key.sources) or
            len(p.conductors)!=len(key.shells) or submission.backend!=key.backend or
            not _close(p.permittivity_f_per_m,8.8541878128e-12,1e-25)):
            return False
        world=lambda local:tuple(x+c for x,c in zip(local,key.center))
        if not _same_vector(spec.end_m,world(key.end)):return False
        for actual,(q,x) in zip(p.charges,key.sources):
            if not _close(actual.charge_c,q) or not _same_vector(actual.position_m,world(x)):
                return False
        for actual,(a,b,Q) in zip(p.conductors,key.shells):
            if (not _close(actual.inner_radius_m,a) or not _close(actual.outer_radius_m,b) or
                not _same_vector(actual.center_m,key.center)):
                return False
            if Q is None:
                if not isinstance(actual.constraint,Grounded):return False
            elif (not isinstance(actual.constraint,FixedNetCharge) or
                  not _close(actual.constraint.charge_c,Q)):
                return False
        return True
    except (AttributeError,TypeError,ValueError):
        return False

def grade_one(case_id,submission):
    key=KEYS[case_id]
    result={'id':case_id,'kind':'analytical' if key.backend else 'refusal',
            'passed':False,'issues':[]}
    issues=result['issues']
    if not isinstance(submission,Submission):
        issues.append('Missing or invalid submission');return result
    if not key.backend:
        if submission.backend is not None or submission.sweep is not None:
            issues.append('Unsupported request replaced by a fabricated solver/model')
        if not all(w in submission.reason.lower() for w in key.refusal_words):
            issues.append('Refusal lacks the relevant physical limitation')
        result['passed']=not issues
        return result
    if not model_matches(submission,key):
        issues.append('Authored charge, geometry, constraint, endpoint or identity differs from prompt')
        return result
    bundle=sample_position_sweep(submission.sweep,BACKENDS[key.backend](),
        resolution=9,seeds=[],line_steps=5,profile_samples=9)
    rows=bundle['position_sweep']['samples']
    if len(rows)!=5 or len(bundle['frames'])!=5:
        issues.append('Missing static states');return result
    reference0=observe(key,0.)
    for i,row in enumerate(rows):
        ref=observe(key,i/4)
        if not _same_vector(row['force_n'],ref['force_n'],1e-15):
            issues.append(f'Frame {i}: selected force disagrees with independent Kelvin images')
        for field in (*ENERGIES,'energy_j','force_parallel_n'):
            if not _close(row[field],ref[field],1e-18):
                issues.append(f'Frame {i}: {field} disagrees with independent reference')
        expected_work=fsum(ref[name]-reference0[name] for name in ENERGIES)
        if not _close(row['external_quasistatic_work_j'],expected_work,1e-18):
            issues.append(f'Frame {i}: wrong displacement work')
    integral=work_integral(key)
    if not isclose(rows[-1]['external_quasistatic_work_j'],integral,rel_tol=3e-5,abs_tol=1e-15):
        issues.append('Endpoint work disagrees with independent force quadrature')
    result.update(passed=not issues,states_checked=5,force_integral_work_j=integral,
                  reported_work_j=rows[-1]['external_quasistatic_work_j'])
    return result

def grade_all(submissions=SUBMISSIONS):
    missing=sorted(set(BY_ID)-set(submissions));extra=sorted(set(submissions)-set(BY_ID))
    results=[grade_one(name,submissions.get(name)) for name in BY_ID]
    return {'schema':'conductor_sim.m15c_open_trial.v1',
            'method':'One open assistant-authored engineering trial, not blind LLM accuracy.',
            'source':'12 original prompts; no additional PDF questions claimed.',
            'total':len(results),'passed':sum(r['passed'] for r in results),
            'analytical_cases':sum(r['kind']=='analytical' for r in results),
            'refusals':sum(r['kind']=='refusal' for r in results),
            'missing':missing,'extra':extra,'results':results}

def export(case_id,path,resolution=25):
    submission=SUBMISSIONS[case_id];key=KEYS[case_id]
    if not key.backend or not model_matches(submission,key):
        raise ValueError('Only a valid analytical authored case can be rendered')
    bundle=sample_position_sweep(submission.sweep,BACKENDS[key.backend](),
                                 resolution=resolution,line_steps=100)
    bundle['authorability_source']={'id':case_id,'prompt':BY_ID[case_id].prompt,
        'method':'M15c original open prompt; not a PDF transcription or blind benchmark'}
    path=Path(path)
    write_visualization(bundle,path,title='Cavity authoring explorer · '+case_id)
    path.with_suffix('.json').write_text(json.dumps(bundle,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return bundle

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='action',required=True)
    sub.add_parser('list')
    grade=sub.add_parser('grade');grade.add_argument('--output',type=Path)
    render=sub.add_parser('render');render.add_argument('case_id',choices=tuple(BY_ID))
    render.add_argument('--output',type=Path,default=Path('output/m15c.html'))
    render.add_argument('--resolution',type=int,default=25)
    args=ap.parse_args()
    if args.action=='list':
        for q in BY_ID.values():print(q.id+': '+q.prompt)
        return 0
    if args.action=='grade':
        report=grade_all()
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print(f"M15c {report['passed']}/{report['total']}: "
              f"{report['analytical_cases']} analytical cases and {report['refusals']} explicit refusals")
        for r in report['results']:
            if r['issues']:print(r['id'],r['issues'])
        return 0 if report['passed']==report['total'] and not report['missing'] and not report['extra'] else 1
    try:
        export(args.case_id,args.output,args.resolution)
    except ValueError as exc:
        print(str(exc));return 2
    print(args.output)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
