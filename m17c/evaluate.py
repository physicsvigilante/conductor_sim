"""Grade original curved-motion authoring against independent SI references.

This is an open engineering trial, not a blind accuracy measurement. Production
Motion2DSpec/integrate_motion_2d/sample_motion_2d are used without per-case solvers.
"""
import argparse
from math import fsum, isclose, isfinite, ulp
from pathlib import Path
import json
from conductor_sim import (ConductingPlane, SolidSphere, SphericalShell,
    Grounded, FixedPotential, FixedNetCharge, GroundedPlaneBackend, ParallelPlateBackend,
    ExternalSphereBackend, MultiImageShellBackend, NestedShellBackend,
    integrate_motion_2d, sample_motion_2d)
from conductor_sim.visualization import write_visualization
from .questions import BY_ID
from .attempts import SUBMISSIONS, Submission
from .reference import (KEYS, EPS0, trajectory, convergence, force_energy,
                        dot, sub, norm, cross)

BACKENDS={'plane':GroundedPlaneBackend,'plates':ParallelPlateBackend,
          'sphere':ExternalSphereBackend,'cavity':MultiImageShellBackend,'nested':NestedShellBackend}


def _same(a,b):
    return isclose(a,b,rel_tol=1e-12,abs_tol=1e-40)


def _same_vector(a,b):
    return len(a)==len(b) and norm(sub(a,b))<1e-12


def _voltage(c):
    return 0. if isinstance(c,Grounded) else c.potential_v if isinstance(c,FixedPotential) else None


def model_matches(submission,key):
    """Check authored SI conversion, inventory and geometry before integrating."""
    try:
        spec=submission.motion;p=spec.problem
        if (submission.backend!=key.backend or spec.charge_index!=0 or len(p.charges)!=1 or
                not _same(p.permittivity_f_per_m,EPS0) or not _same(spec.mass_kg,key.mass) or
                not _same(p.charges[0].charge_c,key.charge) or
                not _same_vector(p.charges[0].position_m,key.position) or
                not _same_vector(spec.initial_velocity_m_s,key.velocity) or
                not _same(spec.duration_s,key.duration) or not _same(spec.sample_interval_s,key.interval) or
                spec.relative_tolerance>2e-7 or spec.surface_guard_m is not None):
            return False
        if key.backend in ('plane','plates'):
            expected_count=1 if key.backend=='plane' else 2
            if len(p.conductors)!=expected_count:return False
            for i,c in enumerate(p.conductors):
                if not isinstance(c,ConductingPlane):return False
                normal=tuple((-1 if i else 1)*x for x in key.normal)
                point=tuple(key.center[j]+i*key.gap*key.normal[j] for j in range(3))
                if (not _same_vector(c.normal,normal) or
                        abs(dot(sub(c.point_m,point),normal))>1e-12 or
                        _voltage(c.constraint)!=key.voltages[i]):return False
            return True
        if len(p.conductors)!=len(key.shells):return False
        for c,(a,b,Q) in zip(p.conductors,key.shells):
            if not _same_vector(c.center_m,key.center):return False
            if key.backend=='sphere':
                if not isinstance(c,SolidSphere) or not _same(c.radius_m,a):return False
            elif (not isinstance(c,SphericalShell) or not _same(c.inner_radius_m,a)
                    or not _same(c.outer_radius_m,b)):return False
            if Q is None:
                if not isinstance(c.constraint,Grounded):return False
            elif not isinstance(c.constraint,FixedNetCharge) or not _same(c.constraint.charge_c,Q):return False
        return True
    except (AttributeError,TypeError,ValueError,IndexError):
        return False


def grade_one(case_id,submission):
    key=KEYS[case_id]
    report={'id':case_id,'kind':'trajectory' if key.backend else 'refusal',
            'passed':False,'issues':[]}
    issues=report['issues']
    if not isinstance(submission,Submission):
        issues.append('Missing or invalid authored submission');return report
    if not key.backend:
        if submission.backend is not None or submission.motion is not None:
            issues.append('Unsupported request was replaced by a fabricated supported model')
        if not all(word in submission.reason.lower() for word in key.refusal_words):
            issues.append('Refusal does not identify the physical capability limitation')
        report['passed']=not issues
        return report
    if not model_matches(submission,key):
        issues.append('Authored mass, charge, initial conditions, geometry, constraints or time differs from prompt')
        return report
    reference=trajectory(case_id)
    conv=convergence(case_id)
    report['reference_convergence']=conv
    if conv['position_difference_m']>1e-9 or conv['velocity_difference_m_s']>1e-8:
        issues.append('Independent reference failed its own step-halving convergence gate')
        return report
    result=integrate_motion_2d(submission.motion,BACKENDS[key.backend]())
    if (result.spec!=submission.motion or result.termination!='duration' or result.events
            or len(result.samples)!=len(reference) or not isfinite(result.max_abs_energy_drift_j)):
        issues.append('Unexpected termination, events or missing output samples')
        return report
    speed_scale=max(norm(row['velocity_m_s']) for row in reference)
    force_scale=max(norm(row['force_n']) for row in reference)
    energy_scale=max(max(abs(row['interaction_energy_j']),row['kinetic_energy_j']) for row in reference)
    max_position_error=0.;max_velocity_error=0.
    reference_momentum=tuple(key.mass*x for x in cross(sub(key.position,key.center),key.velocity))
    momentum_roundoff=64*ulp(key.mass*norm(sub(key.position,key.center))*speed_scale)
    for i,(row,expected) in enumerate(zip(result.samples,reference)):
        if not all(isfinite(x) for x in (row.time_s,*row.position_m,*row.velocity_m_s,
                *row.force_n,*row.acceleration_m_s2,row.kinetic_energy_j,row.interaction_energy_j,
                row.total_energy_j,row.energy_drift_j)):
            issues.append(f'Sample {i}: nonfinite numerical output')
            continue
        dp=norm(sub(row.position_m,expected['position_m']))
        dv=norm(sub(row.velocity_m_s,expected['velocity_m_s']))
        max_position_error=max(max_position_error,dp);max_velocity_error=max(max_velocity_error,dv)
        if abs(row.time_s-expected['time_s'])>1e-11*key.duration:
            issues.append(f'Sample {i}: wrong physical elapsed time')
        if dp>1e-9 or dv>1e-9+2e-7*speed_scale:
            issues.append(f'Sample {i}: trajectory differs from independent time evolution')
        F,U=force_energy(key,row.position_m)
        # The electron case deliberately tests the stated uniform-field limit.
        # Its bounded induction error is much smaller than these tolerances.
        if norm(sub(row.force_n,F))>1e-30+2e-7*force_scale:
            issues.append(f'Sample {i}: force differs from independent conductor reference')
        if norm(sub(row.acceleration_m_s2,tuple(x/key.mass for x in F)))>1e-20+2e-7*force_scale/key.mass:
            issues.append(f'Sample {i}: acceleration differs from F/m')
        K=.5*key.mass*norm(row.velocity_m_s)**2
        for field,value in [('kinetic_energy_j',K),('interaction_energy_j',U),('total_energy_j',fsum((K,U)))]:
            if abs(getattr(row,field)-value)>1e-30+2e-7*energy_scale:
                issues.append(f'Sample {i}: {field} differs from independent SI reference')
        if key.backend in ('sphere','cavity','nested'):
            L=tuple(key.mass*x for x in cross(sub(row.position_m,key.center),row.velocity_m_s))
            if norm(sub(L,reference_momentum))>max(1e-30,momentum_roundoff)+2e-6*norm(reference_momentum):
                issues.append(f'Sample {i}: angular momentum vector is not conserved')
        else:
            tangential=sub(row.velocity_m_s,tuple(dot(row.velocity_m_s,key.normal)*x for x in key.normal))
            start=sub(key.velocity,tuple(dot(key.velocity,key.normal)*x for x in key.normal))
            if norm(sub(tangential,start))>1e-9+2e-7*speed_scale:
                issues.append(f'Sample {i}: unforced tangential velocity changed')
        if abs(row.energy_drift_j)>1e-30+2e-6*energy_scale:
            issues.append(f'Sample {i}: excessive numerical mechanical-energy drift')
    if not isclose(result.max_abs_energy_drift_j,max(abs(r.energy_drift_j) for r in result.samples),
                   rel_tol=1e-12,abs_tol=1e-30):
        issues.append('Maximum drift does not match samples')
    if case_id=='centre-crossing' and result.samples[-1].position_m[0]>=0:
        issues.append('Particle did not traverse the nonsingular cavity centre')
    report.update(passed=not issues,samples_checked=len(reference),
        max_position_error_m=max_position_error,max_velocity_error_m_s=max_velocity_error,
        max_abs_energy_drift_j=result.max_abs_energy_drift_j,
        reference_method=('Exact uniform-field limit with negligible-induction bound'
            if key.backend=='plates' else 'Independent Kelvin/mirror forces and converged velocity-Verlet'))
    return report


def grade_all(submissions=SUBMISSIONS):
    results=[grade_one(cid,submissions.get(cid)) for cid in BY_ID]
    return dict(schema='conductor_sim.m17c_open_trial.v1',
        method='One open assistant-authored engineering trial, not blind LLM accuracy or an automatic parser.',
        source='12 original prompts; no new PDF coverage claimed.',
        total=len(results),passed=sum(r['passed'] for r in results),
        trajectory_cases=sum(r['kind']=='trajectory' for r in results),
        refusals=sum(r['kind']=='refusal' for r in results),
        missing=sorted(set(BY_ID)-set(submissions)),extra=sorted(set(submissions)-set(BY_ID)),results=results)


def export(case_id,path,resolution=19,visual_frames=12):
    submission=SUBMISSIONS[case_id];key=KEYS[case_id]
    if not key.backend or not model_matches(submission,key):
        raise ValueError('Only a valid supported trajectory case can be exported')
    # This path uses production integration and rendering only. Test-only
    # reference paths are never passed to the viewer or substituted as physics.
    bundle=sample_motion_2d(submission.motion,BACKENDS[key.backend](),
        resolution=resolution,line_steps=50,visual_frames=visual_frames)
    bundle['authorability_source']={'id':case_id,'prompt':BY_ID[case_id].prompt,
        'method':'M17c original open prompt; not blind accuracy or PDF transcription'}
    path=Path(path)
    write_visualization(bundle,path,title='M17c electrostatic motion · '+case_id)
    path.with_suffix('.json').write_text(json.dumps(bundle,allow_nan=False,indent=2)+'\n',encoding='utf-8')
    return bundle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='action',required=True)
    commands.add_parser('list')
    grade=commands.add_parser('grade');grade.add_argument('--output',type=Path)
    render=commands.add_parser('render');render.add_argument('case_id',choices=tuple(BY_ID))
    render.add_argument('--output',type=Path,default=Path('output/m17c.html'))
    render.add_argument('--resolution',type=int,default=19)
    render.add_argument('--visual-frames',type=int,default=12)
    args=parser.parse_args()
    if args.action=='list':
        for question in BY_ID.values():print(question.id+': '+question.prompt)
        return 0
    if args.action=='grade':
        report=grade_all()
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,allow_nan=False,indent=2)+'\n',encoding='utf-8')
        print(f"M17c {report['passed']}/{report['total']}: {report['trajectory_cases']} trajectories, {report['refusals']} refusals")
        for row in report['results']:
            if row['issues']:print(row['id'],row['issues'])
        return 0 if report['passed']==report['total'] and not report['missing'] and not report['extra'] else 1
    try:export(args.case_id,args.output,args.resolution,args.visual_frames)
    except ValueError as error:
        print(str(error));return 2
    print(args.output)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
