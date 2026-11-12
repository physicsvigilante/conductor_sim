"""M6b open held-out authorability scorer and offline simulation exporter.

No physics is implemented here. Numerical observations are sampled from the
already-accepted analytical/FEMM backends. Independent grading values live
in m6b.reference; submission authors import only m6b.protocol and the public
conductor_sim models. This is NOT a secure sandbox for untrusted Python.
"""
from __future__ import annotations

import argparse
from importlib import import_module
import json
from math import isclose, isfinite
from pathlib import Path

from conductor_sim import Femm5Backend, NestedShellBackend
from conductor_sim.model import EPSILON_0, FixedNetCharge, Grounded
from conductor_sim.solver import UnsupportedProblem
from conductor_sim.visualization import SliceSpec, sample_sweep, write_visualization

from .protocol import Submission, Probe
from .questions import QUESTIONS, BY_ID
from .reference import KEYS


class AuthoringError(ValueError):
    """A candidate does not meet the prompt or backend contract."""


def submissions_from(module: str):
    """Import a trusted author-written Python module; arbitrary Python executes."""
    obj = getattr(import_module(module), 'SUBMISSIONS', None)
    if not isinstance(obj, dict):
        raise AuthoringError(f'{module} must provide a SUBMISSIONS dictionary')
    return obj


def fingerprint(problem):
    """Describe actual immutable input, not results from any physics solver."""
    shells = []
    for item in problem.conductors:
        if isinstance(item.constraint, FixedNetCharge):
            constraint, q = 'fixed', item.constraint.charge_c
        elif isinstance(item.constraint, Grounded):
            constraint, q = 'grounded', None
        else:
            raise AuthoringError('unrecognized conductor constraint')
        shells.append((item.name,item.inner_radius_m,item.outer_radius_m,
                       constraint,q,tuple(item.center_m)))
    sources = [(item.charge_c,tuple(item.position_m)) for item in problem.charges]
    return (tuple(shells),tuple(sources))


def same_model(actual, expected):
    """Exact structure + SI values with only floating representation tolerance."""
    a_shells,a_sources=actual
    e_shells,e_sources=expected
    if len(a_shells)!=len(e_shells) or len(a_sources)!=len(e_sources):
        return False
    def numeric(x,y):
        if x is None or y is None:
            return x is None and y is None
        return isclose(x,y,rel_tol=1e-12,abs_tol=1e-22)
    for a,e in zip(a_shells,e_shells):
        if a[0]!=e[0] or a[3]!=e[3]:
            return False
        if not (numeric(a[1],e[1]) and numeric(a[2],e[2]) and numeric(a[4],e[4])):
            return False
        if any(not numeric(x,y) for x,y in zip(a[5],e[5])):
            return False
    for a,e in zip(a_sources,e_sources):
        if not numeric(a[0],e[0]) or len(a[1])!=3 or len(e[1])!=3:
            return False
        if any(not numeric(x,y) for x,y in zip(a[1],e[1])):
            return False
    return True


def observe(solution, probe: Probe):
    if probe.kind in ('potential','field_x','field_y','field_z'):
        if probe.point_m is None or probe.conductor is not None:
            raise AuthoringError('spatial probe requires point and no conductor')
        if probe.kind=='potential':
            return solution.potential_at(probe.point_m)
        return solution.electric_field_at(probe.point_m)[
            {'field_x':0,'field_y':1,'field_z':2}[probe.kind]]
    if probe.conductor is None or probe.point_m is not None:
        raise AuthoringError('conductor probe requires a name and no point')
    results={x.name:x for x in solution.conductor_results}
    if probe.conductor not in results:
        raise AuthoringError(f'unknown conductor {probe.conductor!r}')
    c=results[probe.conductor]
    return {
        'shell_voltage':c.potential_v,
        'shell_net_charge':c.net_charge_c,
        'inner_total':c.inner_surface_charge_c,
        'outer_total':c.outer_surface_charge_c,
    }[probe.kind]


def grade_one(question_id: str, submission: Submission) -> dict:
    if question_id not in KEYS:
        raise AuthoringError('unknown question')
    key=KEYS[question_id]
    findings=[]
    if not isinstance(submission, Submission):
        return {'id':question_id,'passed':False,'issues':['not a Submission']}
    if submission.capability!=key.capability:
        findings.append('wrong capability classification')
    if key.capability=='unsupported':
        if submission.states or submission.probes:
            findings.append('unsupported case must not fabricate a solve or observation')
        reason=submission.reason.lower()
        if len(reason)<30:
            findings.append('missing specific limitation explanation')
        if not any(word in reason for word in key.required_reason_words):
            findings.append('limitation explanation does not identify necessary physics')
        return {'id':question_id,'classification':submission.capability,
                'passed':not findings,'issues':findings,'observations':[]}
    if not submission.states or len(submission.states)!=len(key.models):
        findings.append('wrong number of authored equilibrium states')
        return {'id':question_id,'classification':submission.capability,
                'passed':False,'issues':findings,'observations':[]}
    for i,((label,problem),expected) in enumerate(zip(submission.states,key.models)):
        if not isinstance(label,str) or not label:
            findings.append(f'state {i}: missing caption')
        if not same_model(fingerprint(problem),expected) or not isclose(
                problem.permittivity_f_per_m, EPSILON_0, rel_tol=1e-12):
            findings.append(f'state {i}: wrong geometry/charge/constraint, medium or ordering')
    actual_signatures=[p.signature for p in submission.probes]
    expected_signatures=[p.signature for p,value in key.observations]
    if len(set(actual_signatures))!=len(actual_signatures):
        findings.append('duplicate observation requests')
    if set(actual_signatures)!=set(expected_signatures):
        findings.append('missing or unrequested probes')
    if findings:
        return {'id':question_id,'classification':submission.capability,'passed':False,
                'issues':findings,'observations':[]}
    analytical=NestedShellBackend()
    numerical=Femm5Backend()
    for i,(label,problem) in enumerate(submission.states):
        if not analytical.supports(problem):
            findings.append(f'state {i}: exact analytical backend rejects model')
        if numerical.supports(problem)!=(key.capability=='both'):
            findings.append(f'state {i}: FEMM capability classification inconsistent')
    observed=[]
    if not findings:
        cache={i:analytical.solve(problem) for i,(_,problem) in enumerate(submission.states)}
        expected_by_signature={probe.signature:value for probe,value in key.observations}
        for probe in submission.probes:
            try:
                value=observe(cache[probe.state],probe)
                target=expected_by_signature[probe.signature]
                if not (isfinite(value) and isclose(
                    value,target,rel_tol=2e-9,
                    abs_tol=2e-18 if 'total' in probe.kind or 'charge' in probe.kind else 1e-9)):
                    findings.append(f'{probe.label}: independent physics disagreement')
                observed.append({'state':probe.state,'label':probe.label,
                                 'kind':probe.kind,'actual':value,'expected':target})
            except (ValueError,TypeError,IndexError,KeyError,ZeroDivisionError) as exc:
                findings.append(f'{probe.label}: observation failed: {exc}')
    return {'id':question_id,'classification':submission.capability,
            'passed':not findings,'issues':findings,'observations':observed,
            'authored_states':len(submission.states)}


def grade_all(submissions:dict) -> dict:
    expected_ids={q.id for q in QUESTIONS}
    extras=sorted(set(submissions)-expected_ids)
    missing=sorted(expected_ids-set(submissions))
    results=[]
    for q in QUESTIONS:
        if q.id not in submissions:
            results.append({'id':q.id,'passed':False,'issues':['missing attempt']})
            continue
        try:
            results.append(grade_one(q.id,submissions[q.id]))
        except Exception as exc:
            results.append({'id':q.id,'passed':False,'issues':[f'{type(exc).__name__}: {exc}']})
    passed=sum(r['passed'] for r in results)
    return {'benchmark':'m6b-open-heldout-v1',
            'note':'One openly documented assistant-authoring trial; not a blind independent model benchmark.',
            'cases':len(QUESTIONS),'passed':passed,'failed':len(QUESTIONS)-passed,
            'analytical_and_femm':sum(x['passed'] and x.get('classification')=='both' for x in results),
            'analytical_only':sum(x['passed'] and x.get('classification')=='analytical-only' for x in results),
            'correct_refusals':sum(x['passed'] and x.get('classification')=='unsupported' for x in results),
            'missing':missing,'extra':extras,'results':results}


def export_one(question_id,submission: Submission,output: Path,*,backend_name='analytical',resolution=31):
    graded=grade_one(question_id,submission)
    if not graded['passed']:
        raise AuthoringError(f'failed physics/authoring checks: {graded["issues"]}')
    if submission.capability=='unsupported':
        raise UnsupportedProblem('this task requires additional physics; no substituted geometry')
    if backend_name=='femm' and submission.capability!='both':
        raise UnsupportedProblem('FEMM supports only axial charge sites; do not fall back silently')
    backend=NestedShellBackend() if backend_name=='analytical' else Femm5Backend()
    if any(not backend.supports(p) for _,p in submission.states):
        raise UnsupportedProblem('selected backend cannot solve the authored configuration')
    max_radius=max(s.outer_radius_m for _,problem in submission.states for s in problem.conductors)
    plane=SliceSpec.meridian(submission.states[0][1],extent_m=1.6*max_radius,resolution=resolution)
    bundle=sample_sweep(submission.states,backend,plane=plane)
    bundle['authorability_source']={
        'benchmark':'m6b-open-heldout-v1','question_id':question_id,
        'prompt':BY_ID[question_id].prompt,
        'backend_requested':backend_name,
        'authoring_notes':'precomputed static equilibria; no browser-time recomputation',
    }
    write_visualization(bundle,output,title=f'M6b: {question_id}')
    json_path=output.with_suffix('.json')
    json_path.write_text(json.dumps(bundle,allow_nan=False,ensure_ascii=False),encoding='utf-8')
    return {'html':str(output),'json':str(json_path),'states':len(submission.states),
            'backend':backend_name,'question_id':question_id}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission',default='m6b.attempts',
                        help='Trusted importable Python module defining SUBMISSIONS')
    actions=parser.add_subparsers(dest='action',required=True)
    listp=actions.add_parser('list',help='List fresh JEE prompts')
    listp.add_argument('--json',action='store_true')
    gradep=actions.add_parser('grade',help='Grade authored models and observations')
    gradep.add_argument('--output',type=Path)
    renderp=actions.add_parser('render',help='Export accepted authored model to offline HTML+JSON')
    renderp.add_argument('question_id')
    renderp.add_argument('--backend',choices=['analytical','femm'],default='analytical')
    renderp.add_argument('--output',type=Path)
    renderp.add_argument('--resolution',type=int,default=31)
    args=parser.parse_args(argv)
    if args.action=='list':
        data=[{'id':q.id,'prompt':q.prompt} for q in QUESTIONS]
        if args.json:
            print(json.dumps(data,ensure_ascii=False,indent=2))
        else:
            for d in data:
                print(d['id']+'\n  '+d['prompt']+'\n')
        return 0
    try:
        attempts=submissions_from(args.submission)
        if args.action=='grade':
            result=grade_all(attempts)
            text=json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2)
            if args.output:
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(text,encoding='utf-8')
            print(f'M6b: {result["passed"]}/{result["cases"]} passed, '
                  f'{result["analytical_and_femm"]} numerical-capable, '
                  f'{result["analytical_only"]} analytical-only, '
                  f'{result["correct_refusals"]} refusals')
            return 1 if result['failed'] or result['missing'] or result['extra'] else 0
        if args.question_id not in attempts:
            raise AuthoringError('unknown or missing authoring submission')
        out=args.output or Path('output/m6b')/(args.question_id+'.html')
        result=export_one(args.question_id,attempts[args.question_id],out,
                          backend_name=args.backend,resolution=args.resolution)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0
    except (AuthoringError,UnsupportedProblem,ValueError) as exc:
        parser.exit(2,f'M6b: {exc}\n')


if __name__=='__main__':
    raise SystemExit(main())
