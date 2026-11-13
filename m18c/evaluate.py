"""Grade original held-source authoring without using production as its own key."""
from __future__ import annotations
import argparse
import json
from math import fsum, isclose, isfinite
from pathlib import Path
from conductor_sim import (ConductingPlane, SolidSphere, SphericalShell, Grounded,
    FixedPotential, FixedNetCharge, GroundedPlaneBackend, ParallelPlateBackend,
    ExternalSphereBackend, MultiImageShellBackend, NestedShellBackend,
    integrate_motion_2d, sample_motion_2d)
from conductor_sim.visualization import write_visualization
from .attempts import SUBMISSIONS, Submission
from .questions import BY_ID
from .reference import KEYS, EPS0, trajectory, convergence, force_energy, norm, sub, dot

BACKENDS={'plane':GroundedPlaneBackend,'sphere':ExternalSphereBackend,
    'cavity':MultiImageShellBackend,'nested':NestedShellBackend,
    'plates':lambda:ParallelPlateBackend(image_order=256)}


def _same(a,b):return type(a) in (int,float) and isclose(a,b,rel_tol=1e-12,abs_tol=1e-40)
def _vector(a,b):return len(a)==len(b)==3 and all(_same(x,y) for x,y in zip(a,b))
def _voltage(c):return 0. if isinstance(c,Grounded) else c.potential_v if isinstance(c,FixedPotential) else None


def model_matches(submission,key):
    """Prompt identity and SI conversions are checked before any solve."""
    try:
        s=submission.motion;p=s.problem
        if (submission.backend!=key.backend or s.charge_index!=key.moving or
            len(p.charges)!=len(key.sources) or not _same(s.mass_kg,key.mass) or
            not _vector(s.initial_velocity_m_s,key.velocity) or not _same(s.duration_s,key.duration) or
            not _same(s.sample_interval_s,key.duration/20) or s.relative_tolerance>2e-7 or
            s.surface_guard_m is not None or s.charge_guard_m is not None or
            not _same(p.permittivity_f_per_m,EPS0)):
            return False
        for source,(q,xyz) in zip(p.charges,key.sources):
            if not _same(source.charge_c,q) or not _vector(source.position_m,xyz):return False
        if key.backend in ('plane','plates'):
            if len(p.conductors)!=(1 if key.backend=='plane' else 2):return False
            for i,c in enumerate(p.conductors):
                normal=tuple((-1 if i else 1)*x for x in key.normal)
                point=tuple(key.center[j]+i*key.gap*key.normal[j] for j in range(3))
                if (not isinstance(c,ConductingPlane) or not _vector(c.normal,normal) or
                    abs(dot(sub(c.point_m,point),normal))>1e-12 or _voltage(c.constraint)!=key.voltages[i]):
                    return False
        else:
            if len(p.conductors)!=len(key.shells):return False
            for c,(a,b,Q) in zip(p.conductors,key.shells):
                if not _vector(c.center_m,key.center):return False
                if key.backend=='sphere':
                    if not isinstance(c,SolidSphere) or not _same(c.radius_m,a):return False
                elif (not isinstance(c,SphericalShell) or not _same(c.inner_radius_m,a) or
                      not _same(c.outer_radius_m,b)):return False
                if Q is None:
                    if not isinstance(c.constraint,Grounded):return False
                elif not isinstance(c.constraint,FixedNetCharge) or not _same(c.constraint.charge_c,Q):return False
        return True
    except (AttributeError,TypeError,ValueError,IndexError):return False


def grade_result(case_id,submission,result):
    """Compare every real output row, not just a plotted subset or conserved sum."""
    key=KEYS[case_id];ref=trajectory(case_id);conv=convergence(case_id);issues=[]
    report=dict(id=case_id,kind='trajectory',passed=False,issues=issues,reference_convergence=conv)
    if conv['position_difference_m']>1e-9 or conv['velocity_difference_m_s']>1e-8:
        issues.append('Independent reference failed step-halving convergence');return report
    if (result.spec!=submission.motion or result.termination!='duration' or result.events or
        len(result.samples)!=len(ref) or not isfinite(result.max_abs_energy_drift_j)):
        issues.append('Unexpected source/spec identity, termination or sample count');return report
    speed=max(norm(r['velocity_m_s']) for r in ref)
    force_scale=max(norm(r['force_n']) for r in ref)
    energy_scale=max(max(abs(r['interaction_energy_j']),r['kinetic_energy_j']) for r in ref)
    if (conv.get('image_force_difference_n',0)>2e-6*force_scale or
        conv.get('image_energy_difference_j',0)>2e-6*energy_scale):
        issues.append('Independent plate-image reference failed order-doubling convergence');return report
    dpmax=dvmax=0.;first=result.samples[0]
    for i,(row,expected) in enumerate(zip(result.samples,ref)):
        if not all(isfinite(x) for x in (row.time_s,*row.position_m,*row.velocity_m_s,
            *row.displacement_m,*row.acceleration_m_s2,*row.force_n,row.kinetic_energy_j,
            row.interaction_energy_j,row.total_energy_j,row.energy_drift_j)):
            issues.append(f'Sample {i}: nonfinite numerical data');continue
        dp=norm(sub(row.position_m,expected['position_m']));dv=norm(sub(row.velocity_m_s,expected['velocity_m_s']))
        dpmax=max(dpmax,dp);dvmax=max(dvmax,dv)
        if abs(row.time_s-expected['time_s'])>1e-11*key.duration:issues.append(f'Sample {i}: elapsed time mismatch')
        if dp>2e-9 or dv>1e-9+3e-7*speed:issues.append(f'Sample {i}: independent trajectory mismatch')
        F,U=force_energy(key,row.position_m)
        if norm(sub(row.force_n,F))>1e-26+2e-6*force_scale:issues.append(f'Sample {i}: independent force mismatch')
        if norm(sub(row.acceleration_m_s2,tuple(x/key.mass for x in F)))>1e-14+2e-6*force_scale/key.mass:
            issues.append(f'Sample {i}: F/m mismatch')
        if norm(sub(row.displacement_m,sub(row.position_m,key.sources[key.moving][1])))>1e-12:
            issues.append(f'Sample {i}: displacement mismatch')
        kinetic=.5*key.mass*norm(row.velocity_m_s)**2
        for field,value in [('kinetic_energy_j',kinetic),('interaction_energy_j',U),('total_energy_j',fsum((kinetic,U)))]:
            if abs(getattr(row,field)-value)>1e-25+2e-6*energy_scale:issues.append(f'Sample {i}: {field} mismatch')
        if abs(row.energy_drift_j)>1e-25+2e-6*energy_scale:issues.append(f'Sample {i}: excessive energy drift')
        # These bounded benchmark energies have no enormous gauge/held-source
        # offset. This independent balance does not replace componentwise
        # drift in the production engine (which is needed for large offsets).
        measured=fsum((row.kinetic_energy_j-first.kinetic_energy_j,
                       row.interaction_energy_j-first.interaction_energy_j))
        if abs(row.energy_drift_j-measured)>1e-25+1e-10*energy_scale:
            issues.append(f'Sample {i}: reported drift disagrees with energy balance')
    if not isclose(result.max_abs_energy_drift_j,max(abs(r.energy_drift_j) for r in result.samples),rel_tol=1e-12,abs_tol=1e-30):
        issues.append('Maximum drift does not match rows')
    report.update(passed=not issues,samples_checked=len(ref),held_sources=len(key.sources)-1,
        moving_charge_index=key.moving,max_position_error_m=dpmax,max_velocity_error_m_s=dvmax,
        max_abs_energy_drift_j=result.max_abs_energy_drift_j,
        reference_method='Independent boundary Green functions + step-halved velocity-Verlet; no engine imports')
    return report


def grade_one(case_id,submission):
    key=KEYS[case_id]
    report=dict(id=case_id,kind='trajectory' if key.backend else 'refusal',passed=False,issues=[])
    if not isinstance(submission,Submission):report['issues'].append('Missing or invalid submission');return report
    if not key.backend:
        if submission.motion is not None or submission.backend is not None:
            report['issues'].append('Unsupported problem replaced by a supported but different arrangement')
        if not isinstance(submission.reason,str) or not all(w in submission.reason.lower() for w in key.refusal_words):
            report['issues'].append('Refusal must explain the specific physical limitation')
        report['passed']=not report['issues'];return report
    if not model_matches(submission,key):
        report['issues'].append('Authored inventory, moving identity, units, constraints or launch differs from prompt');return report
    try:
        result=integrate_motion_2d(submission.motion,BACKENDS[key.backend]())
        return grade_result(case_id,submission,result)
    except (ValueError,TypeError,RuntimeError,ArithmeticError) as exc:
        report['issues'].append(f'Trajectory/reference evaluation failed: {type(exc).__name__}: {exc}')
        return report


def grade_all(submissions=None):
    submissions=SUBMISSIONS if submissions is None else submissions
    results=[grade_one(cid,submissions.get(cid)) for cid in BY_ID]
    missing=sorted(set(BY_ID)-set(submissions));extra=sorted(set(submissions)-set(BY_ID))
    return dict(schema='conductor_sim.m18c_open_trial.v1',
        method='One open assistant-authored engineering trial, not blind LLM accuracy or an automatic parser.',
        source='12 original prompts; no new PDF coverage claimed.',
        total=len(results),passed=sum(r['passed'] for r in results),
        trajectory_cases=sum(r['kind']=='trajectory' for r in results),
        refusals=sum(r['kind']=='refusal' for r in results),missing=missing,extra=extra,
        accepted=all(r['passed'] for r in results) and not missing and not extra,results=results)


def export(case_id,path,resolution=19,visual_frames=12):
    submission=SUBMISSIONS[case_id];key=KEYS[case_id]
    if not key.backend or not model_matches(submission,key):raise ValueError('Only matching supported authored problems can be exported')
    # Sole production data path. Neither reference trajectories nor grading
    # force functions are called, even to populate display-only metadata.
    bundle=sample_motion_2d(submission.motion,BACKENDS[submission.backend](),resolution=resolution,
        visual_frames=visual_frames,line_steps=35,profile_samples=25)
    bundle['authorability_source']=dict(id=case_id,prompt=BY_ID[case_id].prompt,
        scope='Original open M18c authoring trial; production dynamics and fields only.')
    path=Path(path);write_visualization(bundle,path,title='M18c — '+case_id+' / moving and held charges')
    path.with_suffix('.json').write_text(json.dumps(bundle,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return bundle


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('list')
    grade=commands.add_parser('grade');grade.add_argument('--output',type=Path,default=Path('output/m18c-grade.json'))
    render=commands.add_parser('render');render.add_argument('case',choices=tuple(BY_ID));render.add_argument('--output',type=Path,required=True)
    render.add_argument('--resolution',type=int,default=19);render.add_argument('--visual-frames',type=int,default=12)
    args=parser.parse_args()
    if args.command=='list':
        for q in BY_ID.values():print(q.id+': '+q.prompt)
    elif args.command=='grade':
        report=grade_all();args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print(f'M18c: {report["passed"]}/{report["total"]}; {report["trajectory_cases"]} trajectories, {report["refusals"]} refusals; accepted={report["accepted"]}')
        if not report['accepted']:raise SystemExit(1)
    else:
        b=export(args.case,args.output,args.resolution,args.visual_frames)
        print(f'{args.case}: {len(b["motion_2d"]["samples"])} real time samples; {len(b["frames"])} complete-inventory field frames; {args.output}')

if __name__=='__main__':main()
