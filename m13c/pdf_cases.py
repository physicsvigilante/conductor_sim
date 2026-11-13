"""Selected, visually verified anchors from the scanned conductor PDF.

18 selected references are classified, not misrepresented as 18 solved cases.
Two Kelvin-image calculations independently validate existing analytical
conductor configurations, plus explicitly labelled position-sweep extensions.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import hypot, isclose
from pathlib import Path
import json

from conductor_sim import (ElectrostaticProblem, ExternalSphereBackend,
                           FixedNetCharge, Grounded, PointCharge, SolidSphere)
from conductor_sim.position_sweep import PositionSweepSpec, sample_position_sweep
from conductor_sim.visualization import write_visualization
from .reference import K, s, supported, reference_observation

NAME = '02_conductors_half_size.pdf'
SHA256 = '9dff3c02a9f1848d2c3d99cb1a6e7a44dc7f5a51f7a3a2b8cc92d10ce1fbca5d'
SIZE_BYTES = 20685051
PAGES = 17

@dataclass(frozen=True)
class PdfCase:
    id: str
    page: int
    question: str
    topic: str
    category: str
    note: str = ''

CASES = (
    PdfCase('p01-single-08',1,'Single Correct Q8','Electron work in a uniform free electric field','unsupported','No conductor is authored'),
    PdfCase('p02-single-15',2,'Single Correct Q15','Connecting separated spherical conductors','unsupported','Equal-potential wire constraint is not modelled'),
    PdfCase('p02-single-19',2,'Single Correct Q19','Electron launched at charged metal sheet','unsupported','Stopping time/distance needs dynamics and charge-density boundary'),
    PdfCase('p03-single-27',3,'Single Correct Q27','Energy released by connecting a metal sphere to a surrounding shell','unsupported','Changing connectivity and total conductor self-energy'),
    PdfCase('p03-single-29',3,'Single Correct Q29','Charge/potential of concentric grounded conductors','static-only','Related radial potential problems supported; not an M13b force sweep'),
    PdfCase('p03-single-30',3,'Single Correct Q30','Electron flight time between voltage-biased metal plates','unsupported','Time dynamics not modelled'),
    PdfCase('p05-single-47',5,'Single Correct Q47','Rotating dipole and orientation energy','unsupported','No rigid-dipole torque or rotation primitive'),
    PdfCase('p07-multiple-13',7,'Multiple Correct Q13','Static potential outside a shell with an internal charge','static-only','Cavity potential models exist, not cavity force/energy'),
    PdfCase('p08-multiple-21',8,'Multiple Correct Q21','Positive charge approaches grounded metal sphere','supported-explorer','Static induced force and charge, not current waveform'),
    PdfCase('p09-multiple-27',9,'Multiple Correct Q27','Neutral sphere with tangent AB=4 m, radius BC=3 m, +1 microcoulomb at A','supported-explorer','Exact conductor and induced potential; optional radial sweep extends the question'),
    PdfCase('p10-linked-13',10,'Linked Comprehension Q13','Stored energy of charged isolated concentric shells','unsupported','Not the M13a moving-point mechanical interaction energy'),
    PdfCase('p10-linked-14',10,'Linked Comprehension Q14','Wiring shells and energy converted to heat','unsupported','No connected-conductor energy and heat'),
    PdfCase('p11-linked-28',11,'Linked Comprehension Q28','Conductor with two nonconcentric cavities','unsupported','Multi-cavity geometry unsupported'),
    PdfCase('p12-linked-40',12,'Linked Comprehension Q40','Work for charged rings and particle','unsupported','Charged ring primitive not present'),
    PdfCase('p13-matrix-05',13,'Matrix Matching Q5','Connected conducting shells and redistribution energy','unsupported','Connected constraints and total field energy unavailable'),
    PdfCase('p14-numerical-07',14,'Numerical Q7','Concentric shells connected by a wire','unsupported','Wire constraint unsupported'),
    PdfCase('p15-archive-02',15,'JEE Advanced 2012 Q2','Proton released between biased parallel plates','unsupported','Release acceleration requires dynamics'),
    PdfCase('p16-archive-numerical-01',16,'JEE Advanced Numerical Q1','Oil drops moving between finite conducting discs','unsupported','Finite discs, gravity, drag and motion unsupported'),
)
BY_ID = {case.id:case for case in CASES}
EXECUTABLE = ('p08-multiple-21','p09-multiple-27')


def source_status(root=None):
    root=Path(root) if root is not None else Path(__file__).resolve().parent.parent
    path=root/NAME
    digest=sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return {'filename':NAME,'present':path.is_file(),'sha256':digest,
            'expected_sha256':SHA256,'matches_pinned_source':digest==SHA256,
            'size_bytes':path.stat().st_size if path.is_file() else None,
            'pages_visually_reviewed':PAGES}


def verify_source():
    status=source_status()
    if not status['matches_pinned_source'] or status['size_bytes']!=SIZE_BYTES:
        raise ValueError('M13c requires the original pinned 02_conductors_half_size.pdf')
    return status


def authored(case_id):
    """Parameter choices beyond the printed static source are labelled."""
    if case_id=='p08-multiple-21':
        p=ElectrostaticProblem((PointCharge(1e-9,(0.,0.,.32)),),
                               (SolidSphere(.1,Grounded(),name='sphere'),))
        return PositionSweepSpec(p,0,(0.,0.,.18),7)
    if case_id=='p09-multiple-27':
        p=ElectrostaticProblem((PointCharge(1e-6,(3.,4.,0.)),),
                               (SolidSphere(3.,FixedNetCharge(0.),name='sphere'),))
        return PositionSweepSpec(p,0,(4.2,5.6,0.),5)
    raise ValueError('not a supported PDF force-and-energy sweep')


def key_for(case_id):
    if case_id=='p08-multiple-21':
        return supported('sphere',[s(.1)],[(1e-9,(0.,0.,.32))],(0.,0.,.18),7)
    if case_id=='p09-multiple-27':
        return supported('sphere',[s(3.,fixed=0.)],[(1e-6,(3.,4.,0.))],(4.2,5.6,0.),5)
    raise ValueError('PDF case not executable')


def check_one(case_id):
    anchor=BY_ID[case_id]
    result={'id':case_id,'page':anchor.page,'question':anchor.question,
            'category':anchor.category,'status':'classification-only','passed':None,'issues':[]}
    if case_id not in EXECUTABLE:
        return result
    try:
        spec=authored(case_id)
        key=key_for(case_id)
        sol=ExternalSphereBackend().solve(spec.problem)
        observed=sample_position_sweep(spec,ExternalSphereBackend(),
                                       resolution=9,line_steps=12,seeds=[],profile_samples=9)
        rows=observed['position_sweep']['samples']
        begin=None
        for i,row in enumerate(rows):
            force,direct,induced,background=reference_observation(key,row['coordinate_m'])
            energy=direct+induced+background
            if begin is None:begin=energy
            for measured,expected in zip(row['force_n'],force):
                if not isclose(measured,expected,rel_tol=2e-8,abs_tol=1e-13):
                    result['issues'].append(f'frame {i}: Kelvin force differs')
            for prop,expected in [('energy_j',energy),
                                  ('energy_direct_j',direct),
                                  ('energy_induced_j',induced),
                                  ('energy_background_j',background),
                                  ('external_quasistatic_work_j',energy-begin)]:
                if not isclose(row[prop],expected,rel_tol=2e-8,abs_tol=2e-13):
                    result['issues'].append(f'frame {i}: {prop} differs')
            r=hypot(*row['position_m'])
            q=spec.selected_charge.charge_c
            rad=spec.problem.conductors[0].radius_m
            expected_Q=0. if case_id=='p09-multiple-27' else -q*rad/r
            measured_Q=observed['frames'][i]['conductor_results'][0]['net_charge_c']
            if not isclose(measured_Q,expected_Q,rel_tol=2e-11,abs_tol=1e-15):
                result['issues'].append(f'frame {i}: induced total charge differs')
        if case_id=='p09-multiple-27':
            # Printed source: AB tangent 4 m, BC=R=3 m, AC=5 m.
            potential=sol.conductor_results[0].potential_v
            induced_B=sol.potential_at((3.,0.,0.))-K*1e-6/4.
            if not isclose(potential,K*1e-6/5.,rel_tol=2e-12):
                result['issues'].append('conductor potential disagrees with tangent geometry')
            if not isclose(induced_B,-K*1e-6/20.,rel_tol=2e-12):
                result['issues'].append('induced-only tangent potential disagrees')
            result['source_observations']={'conductor_potential_v':potential,
                                           'induced_potential_at_B_v':induced_B}
        else:
            q0=sol.conductor_results[0].net_charge_c
            if not q0<0:
                result['issues'].append('grounded sphere should gain negative induced charge')
            result['source_observations']={'initial_induced_charge_c':q0,
                'conventional_current_direction':'sphere to ground',
                'caveat':'Current direction inferred from static induction, not simulated in time'}
        result['frames']=len(rows)
    except Exception as e:
        result['issues'].append(f'{type(e).__name__}: {e}')
    result['status']='checked' if not result['issues'] else 'failed'
    result['passed']=not result['issues']
    return result


def grade_pdf():
    source=verify_source()
    results=[check_one(case.id) for case in CASES]
    tested=[r for r in results if r['passed'] is not None]
    return {'schema':'conductor_sim.m13c_pdf.v1','source':source,
            'method':'18 selected visually reviewed page anchors; two independent analytical checks. Not a blind LLM accuracy score.',
            'anchors_reviewed':len(CASES),'executed':len(tested),
            'passed':sum(r['passed'] is True for r in tested),
            'failed':sum(r['passed'] is False for r in tested),
            'static_only':sum(c.category=='static-only' for c in CASES),
            'unsupported':sum(c.category=='unsupported' for c in CASES),
            'classification_only':len(CASES)-len(tested),'results':results}


def export_pdf(case_id,path,resolution=19):
    verify_source()
    if case_id not in EXECUTABLE:
        raise ValueError('unsupported PDF case cannot be exported as a force-energy sweep')
    spec=authored(case_id)
    bundle=sample_position_sweep(spec,ExternalSphereBackend(),resolution=resolution,line_steps=80)
    ref=BY_ID[case_id]
    bundle['authorability_source']={'filename':NAME,'sha256':SHA256,'page':ref.page,
       'question':ref.question,'description':ref.topic,
       'caveat':'The physical conductor setup is source-anchored; selected numerical path and stored force/energy equilibria are pedagogical extensions.'}
    path=Path(path)
    write_visualization(bundle,path,title='M13c source-anchored · '+case_id)
    path.with_suffix('.json').write_text(json.dumps(bundle,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return {'html':str(path),'json':str(path.with_suffix('.json')),'frames':len(bundle['frames'])}


def main():
    import argparse
    import sys
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='action',required=True)
    sub.add_parser('list')
    grade=sub.add_parser('grade')
    grade.add_argument('--output',type=Path)
    render=sub.add_parser('render')
    render.add_argument('case_id',choices=tuple(BY_ID))
    render.add_argument('--output',type=Path,default=Path('output/m13c-pdf.html'))
    render.add_argument('--resolution',type=int,default=19)
    args=ap.parse_args()
    if args.action=='list':
        for c in CASES:print(f'p{c.page}, {c.question} [{c.category}] {c.topic}')
        return 0
    if args.action=='grade':
        report=grade_pdf()
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print(f"PDF: {report['anchors_reviewed']} anchors, {report['passed']}/{report['executed']} independently checked")
        return 0 if report['failed']==0 else 1
    try:
        print(json.dumps(export_pdf(args.case_id,args.output,args.resolution),indent=2))
        return 0
    except ValueError as exc:
        print(f'Cannot export: {exc}',file=sys.stderr)
        return 2

if __name__=='__main__':
    raise SystemExit(main())
