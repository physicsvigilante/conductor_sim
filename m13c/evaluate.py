"""M13c: independent, replayable first-pass authorability and scope grading.

A trusted attempt module must contain SUBMISSIONS[id] -> Submission. Importing
an untrusted module executes its Python code; this is NOT a sandbox for public
web submissions. The baseline candidate is m13c.attempts. The independent
reference never imports that module or the conductor solver's force code.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
from importlib import import_module
import json
from math import hypot, isclose
from pathlib import Path

from conductor_sim import (ConductingPlane, FixedNetCharge, FixedPotential,
                           Grounded, GroundedPlaneBackend, ExternalSphereBackend,
                           ParallelPlateBackend, SolidSphere)
from conductor_sim.position_sweep import sample_position_sweep
from conductor_sim.solver import UnsupportedProblem
from conductor_sim.visualization import write_visualization

from .protocol import Submission
from .questions import BY_ID, QUESTIONS
from .reference import KEYS, reference_observation, sub

BACKENDS = {
    'plane': GroundedPlaneBackend,
    'sphere': ExternalSphereBackend,
    'plates': lambda: ParallelPlateBackend(image_order=96),
}
ROOT = Path(__file__).resolve().parent.parent
NEW_PDF = '02_conductors_half_size.pdf'
OLD_PDF = 'POTENTIAL_CONDUCTORS1.pdf'
OLD_SHA256 = '8d0d9b5cf821219058cfc4b7685a39b777e460944df44c3f9cffcb68157a60e5'


class AuthoringError(ValueError):
    pass


def source_inventory():
    """Record only observed repository content; missing PDFs are not scored."""
    items = []
    for name in (NEW_PDF,OLD_PDF):
        path = ROOT/name
        present = path.is_file()
        digest = sha256(path.read_bytes()).hexdigest() if present else None
        items.append({'filename':name,'present':present,'sha256':digest,
                      'size_bytes':path.stat().st_size if present else None,
                      'source_used_in_m13c':False})
    return {'requested_new_pdf_available':items[0]['present'],
            'documents':items,'policy':
            'No page or question from an unavailable PDF is represented as read or graded.'}


def submissions_from(module_name):
    submissions=getattr(import_module(module_name),'SUBMISSIONS',None)
    if not isinstance(submissions,dict):
        raise AuthoringError('trusted submission module must export a SUBMISSIONS dictionary')
    return submissions


def _constraint(c):
    if isinstance(c,Grounded):return ('grounded',0.)
    if isinstance(c,FixedPotential):return ('voltage',c.potential_v)
    if isinstance(c,FixedNetCharge):return ('charge',c.charge_c)
    raise AuthoringError('unexpected conductor constraint')


def _fingerprint(spec):
    conductors=[]
    for c in spec.problem.conductors:
        ct,val=_constraint(c.constraint)
        if isinstance(c,ConductingPlane):
            conductors.append(('plane',c.point_m,c.normal,ct,val,c.name))
        elif isinstance(c,SolidSphere):
            conductors.append(('sphere',c.radius_m,c.center_m,ct,val,c.name))
        else:
            conductors.append(('unsupported shape',type(c).__name__))
    return dict(charges=tuple((c.charge_c,c.position_m) for c in spec.problem.charges),
                conductors=tuple(conductors),index=spec.charge_index,
                end=spec.end_m,steps=spec.steps,eps=spec.problem.permittivity_f_per_m)


def _same(a,b):
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):
        return isclose(a,b,rel_tol=1e-12,abs_tol=1e-16)
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
        return len(a)==len(b) and all(_same(x,y) for x,y in zip(a,b))
    return a==b


def _expected_matches(spec,key):
    try:
        got=_fingerprint(spec)
    except (TypeError,AttributeError,ValueError,AuthoringError):
        return False
    return (_same(got['charges'],key.charges) and
            _same(got['conductors'],key.conductors) and
            got['index']==key.selected and got['steps']==key.steps and
            _same(got['end'],key.end) and
            isclose(got['eps'],8.8541878128e-12,rel_tol=1e-12))


def _check_numeric(got,expected,kind):
    # All measured SI values are checked against independent image oracles;
    # the plate comparison is against a much longer series, not the solver's
    # finite N=96 output. Both tolerances below are acceptance thresholds,
    # not guarantees of global accuracy for arbitrary source arrangements.
    absolute=1.5e-10 if kind=='force' else 2e-12
    return isclose(got,expected,rel_tol=0.001,abs_tol=absolute)


def grade_one(case_id,submitted,*,resolution=9):
    key=KEYS[case_id]
    issues=[]
    observations=[]
    category='unsupported' if key.backend is None else 'analytical-only'
    result={'id':case_id,'expected_capability':category,
            'submitted_capability':getattr(submitted,'capability',None),
            'passed':False,'issues':issues,'observations':observations}
    if not isinstance(submitted,Submission):
        issues.append('not a valid Submission object')
        return result
    if submitted.capability!=category:
        issues.append('incorrect capability classification')
        return result
    if key.backend is None:
        if submitted.sweep is not None or submitted.backend is not None:
            issues.append('unsupported case must not fabricate a model/backend')
        reason=submitted.reason.lower().strip()
        if len(reason)<55 or not all(term in reason for term in key.refusal_terms):
            issues.append('capability refusal lacks case-specific physical explanation')
        result['passed']=not issues
        return result
    if submitted.backend!=key.backend:
        issues.append('wrong backend family; silent FEMM/geometry substitution forbidden')
    if submitted.sweep is None or not _expected_matches(submitted.sweep,key):
        issues.append('authored geometry, charge identity, constraints or path differ from prompt')
    if issues:return result
    try:
        bundle=sample_position_sweep(submitted.sweep,BACKENDS[submitted.backend](),
                                     resolution=resolution,line_steps=12,seeds=[],profile_samples=9)
        samples=bundle['position_sweep']['samples']
        if len(samples)!=key.steps:
            issues.append('incorrect number of physical equilibria')
            return result
        beginning=None
        for i,row in enumerate(samples):
            position_m=row['position_m']
            distance=row['coordinate_m']
            reference=reference_observation(key,distance)
            f_ref,u_direct,u_induced,u_background=reference
            u_ref=u_direct+u_induced+u_background
            if beginning is None:beginning=u_ref
            f_observed=row['force_n']
            component_names=('direct_j','induced_j','background_j')
            for j,(measured,expected) in enumerate(zip(f_observed,f_ref)):
                if not _check_numeric(measured,expected,'force'):
                    issues.append(f'frame {i} force component {j} differs from independent image reference')
            for field,expected in zip(('energy_direct_j','energy_induced_j',
                                       'energy_background_j','energy_j',
                                       'external_quasistatic_work_j'),
                                      (u_direct,u_induced,u_background,u_ref,u_ref-beginning)):
                if not _check_numeric(row[field],expected,'energy'):
                    issues.append(f'frame {i}: {field} differs from independent energy reference')
            if not _same(position_m,tuple(key.charges[key.selected][1][j]+(
                    key.end[j]-key.charges[key.selected][1][j])*i/(key.steps-1)
                    for j in range(3))):
                issues.append(f'frame {i}: selected authored charge position changed unexpectedly')
            direction=sub(key.end,key.charges[key.selected][1])
            length=hypot(*direction)
            expected_component=sum(f_ref[j]*direction[j]/length for j in range(3))
            if not _check_numeric(row['force_parallel_n'],expected_component,'force'):
                issues.append(f'frame {i} path-projected force differs from independent reference')
            observations.append({'coordinate_m':distance,
                                 'reference_force_n':list(f_ref),
                                 'reference_mechanical_energy_j':u_ref,
                                 'reported_force_n':list(f_observed),
                                 'reported_mechanical_energy_j':row['energy_j']})
        if len(bundle['frames'])!=len(samples):
            issues.append('viewer frames and physical observation records disagree')
        if key.backend=='plates' and case_id=='plates-grounded':
            midpoint=samples[len(samples)//2]
            if abs(midpoint['force_parallel_n'])>1e-15:
                issues.append('grounded two-plate midpoint should be a zero-force normal equilibrium')
        if not bundle['playback'].startswith('Precomputed charge positions'):
            issues.append('position sweep was mislabeled as physical motion')
    except Exception as exc:
        issues.append(f'authored solve/observation failed: {type(exc).__name__}: {exc}')
    result['passed']=not issues
    return result


def grade_all(submissions):
    ids=[q.id for q in QUESTIONS]
    missing=sorted(set(ids)-set(submissions))
    extra=sorted(set(submissions)-set(ids))
    results=[grade_one(i,submissions[i]) for i in ids if i in submissions]
    passed=sum(r['passed'] for r in results)
    supported=sum(r['passed'] and r['expected_capability']=='analytical-only' for r in results)
    refusals=sum(r['passed'] and r['expected_capability']=='unsupported' for r in results)
    return {'schema':'conductor_sim.m13c_open_authoring.v1',
            'method':'One open, assistant-authored first attempt; not blind or generalization accuracy',
            'question_count':len(ids),'passed':passed,'failed':len(ids)-passed,
            'supported_passes':supported,'correct_refusals':refusals,
            'missing':missing,'extra':extra,'source_inventory':source_inventory(),
            'new_pdf_questions_graded':0,'results':results}


def export_one(case_id,submitted,path,*,resolution=21,image_order=96):
    if case_id not in BY_ID:raise AuthoringError('unknown M13c question')
    key=KEYS[case_id]
    if key.backend is None:
        raise UnsupportedProblem('cannot export a blocked question as a fabricated simulation')
    if not isinstance(submitted,Submission) or submitted.backend!=key.backend or not _expected_matches(submitted.sweep,key):
        raise AuthoringError('authored model does not match this question')
    backend=(ParallelPlateBackend(image_order=image_order) if submitted.backend=='plates'
             else BACKENDS[submitted.backend]())
    bundle=sample_position_sweep(submitted.sweep,backend,resolution=resolution,
                                 line_steps=70)
    bundle['authorability_source']={'question_id':case_id,
                                  'prompt':BY_ID[case_id].prompt,
                                  'classification':'analytical-only',
                                  'source':'M13c original open JEE-style prompt; not attributed to an unreviewed PDF',
                                  'reference_key':'independently authored static geometry and image formulas'}
    write_visualization(bundle,path,title=f'M13c force/energy · {case_id}')
    Path(path).with_suffix('.json').write_text(
        json.dumps(bundle,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return {'states':len(bundle['frames']),'html':str(path),
            'json':str(Path(path).with_suffix('.json')),
            'capability':'analytical-only'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--submission',default='m13c.attempts',help='trusted module with SUBMISSIONS')
    sub=ap.add_subparsers(dest='operation',required=True)
    sub.add_parser('list')
    sub.add_parser('sources')
    grad=sub.add_parser('grade');grad.add_argument('--output',type=Path)
    render=sub.add_parser('render');render.add_argument('question_id',choices=tuple(BY_ID))
    render.add_argument('--output',type=Path,default=Path('output/m13c.html'))
    render.add_argument('--resolution',type=int,default=21)
    args=ap.parse_args()
    if args.operation=='list':
        for q in QUESTIONS:print(f'{q.id}: {q.prompt}')
        return 0
    if args.operation=='sources':
        print(json.dumps(source_inventory(),indent=2))
        return 0
    attempts=submissions_from(args.submission)
    if args.operation=='grade':
        result=grade_all(attempts)
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print(f"M13c: {result['passed']}/{result['question_count']} graded, "
              f"{result['supported_passes']} supported, {result['correct_refusals']} correct refusals; "
              f"new PDF available: {result['source_inventory']['requested_new_pdf_available']}")
        for r in result['results']:
            if r['issues']:print(r['id'],r['issues'])
        return 0 if result['failed']==0 and not result['missing'] and not result['extra'] else 1
    try:
        print(json.dumps(export_one(args.question_id,attempts[args.question_id],args.output,
                                    resolution=args.resolution),indent=2))
    except (AuthoringError,UnsupportedProblem,ValueError,KeyError) as exc:
        print(f'Cannot export: {exc}',file=__import__('sys').stderr)
        return 2
    return 0


if __name__=='__main__':
    raise SystemExit(main())
