"""M41a: frozen original-JEE-source trial, not a solved-question score.

Runs existing independently-qualified JEE source experiments and detached
stdlib physical references. Adds exactly one teacher numerical example for the
printed p61 static grounded-sphere question, checked independently elsewhere.
Source roles, answer completeness and real browser certification are separate.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from conductor_sim import (ElectrostaticProblem, ExternalSphereBackend, Grounded,
                           PointCharge, SolidSphere)

from tools.m36b_jee_reference import check_native as check_m36b
from tools.m36c_jee_conductor_reference import verify as check_m36c
from tools.m36d_solid_reference import check as check_m36d
from tools.m37b_source_reference import check as check_m37b
from tools.m41a_jee_trial_reference import check as check_p61

REFERENCE_CHECKERS = {
    'm36b': check_m36b, 'm36c': check_m36c, 'm36d': check_m36d,
    'm37b': check_m37b, 'p61': check_p61,
}

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PDFS = {
    '02_conductors_half_size.pdf': '9dff3c02a9f1848d2c3d99cb1a6e7a44dc7f5a51f7a3a2b8cc92d10ce1fbca5d',
    'POTENTIAL_CONDUCTORS1.pdf': '8d0d9b5cf821219058cfc4b7685a39b777e460944df44c3f9cffcb68157a60e5',
}
BOOK = '02_conductors_half_size.pdf'
NOTES = 'POTENTIAL_CONDUCTORS1.pdf'
# Frozen, non-cherry-picked registration. One row per physical apparatus.
# (ID, source PDF, 1-indexed PDF page, source printed anchor, native route)
COHORT = (
    ('p02-single-15', BOOK, 2, 'Single Correct Q15', 'm36b'),
    ('p02-single-19', BOOK, 2, 'Single Correct Q19', 'm36b'),
    ('p03-single-27', BOOK, 3, 'Single Correct Q27', 'm36c'),
    ('p03-single-30', BOOK, 3, 'Single Correct Q30', 'm36b'),
    ('p05-single-47', BOOK, 5, 'Single Correct Q47', 'm36b'),
    ('p08-multiple-21', BOOK, 8, 'Multiple Correct Q21', 'm37b'),
    ('p11-linked-28', BOOK, 11, 'Linked Comprehension Q28', 'refuse'),
    ('p12-linked-40', BOOK, 12, 'Linked Comprehension Q40', 'm36b'),
    ('p15-archive-02', BOOK, 15, 'Archive IIT-JEE 2012 Q2', 'm36b'),
    ('jee2021-p1-q09-q10-equipotential', BOOK, 16, 'Paper 1 Questions 9–10 (one two-charge apparatus)', 'm36b'),
    ('p26-conductor-potential', NOTES, 26, 'Illustration 1', 'm36d'),
    ('p61-external-grounded-sphere', NOTES, 61, 'Illustration 1', 'p61'),
    ('p69-cavity-central-vs-shifted', NOTES, 69, 'Illustration 7', 'm36c'),
    ('p70-cavity-offset', NOTES, 70, 'Illustration 8(i)/(ii)', 'refuse'),
)
SOURCE_KINDS = {
    BOOK: 'original workbook exercise or source-labeled JEE archive',
    NOTES: 'original teaching notes illustration, NOT official JEE paper',
}
REFUSALS = {
    'p11-linked-28': ('irregular outer metal boundary with two cavities and an external charge q',
        'Whole-face Gauss charge is available, but its original irregular exterior local sigma/uniformity in the presence of external charge q is not qualified.'),
    'p70-cavity-offset': ('irregular original cavities/exterior conductor',
        'Neither original non-spherical local density nor the irregular exterior can be replaced by a spherical solver.'),
}
NATIVE_FILES = {
    'm36b': 'native-replays.json',
    'm36c': 'native-conductors.json',
    'm36d': 'native-solid-sphere.json',
    'm37b': 'p8-ground-current.json',
    'p61': 'p61-native.json',
}


def source_kind(ident, pdf):
    if ident.startswith('jee2021'):
        return 'workbook p16 paragraph, historically mapped to JEE Advanced 2021 one apparatus'
    return SOURCE_KINDS[pdf]


def teacher_attribution(route):
    if route == 'p61':
        return 'all p61 numerical values teacher-chosen, source symbolic'
    if route == 'refuse':
        return 'original geometry refused; no teacher numerical substitute'
    return 'native runner maintains explicit printed-source and teacher-number distinctions'


BASE_SHA = '337001f2e0d1ce62c87fa4352993c4a074cfbde6'
SCHEMA = 'conductor_sim.m41a_14_jee_source_trial.v1'


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def check_corpus() -> dict:
    for pdf, expected in SOURCE_PDFS.items():
        if file_sha(ROOT/pdf) != expected:
            raise ValueError(f'original source PDF bytes changed: {pdf}')
    original = json.loads((ROOT/'docs/m36a_jee30_coverage.json').read_text(encoding='utf-8'))
    old = {case['id']: case for case in original['cases']}
    if len(old) != 30 or len(COHORT) != 14 or len({entry[0] for entry in COHORT}) != 14:
        raise ValueError('original 30-case or preregistered 14-case denominator changed')
    for ident, pdf, page, anchor, _ in COHORT:
        if ident not in old:
            raise ValueError('source ID missing from earlier 30-case inventory: '+ident)
        case = old[ident]
        if case['pdf_page'] != page or case['printed_anchor'] != anchor:
            raise ValueError('source page or printed anchor changed: '+ident)
        if ident == 'jee2021-p1-q09-q10-equipotential':
            if 'JEE Advanced 2021' not in case['source'] or 'one' not in original['cohort_rule'].lower():
                raise ValueError('official grouped 2021 source or denominator was misrepresented')
        elif case['source'] != pdf:
            raise ValueError('source file mismatch: '+ident)
    return {pdf: SOURCE_PDFS[pdf] for pdf in SOURCE_PDFS}


def run_script(module: str, arguments: list[str], logpath: Path) -> None:
    env = os.environ.copy()
    # Reproducible source-root import rather than relying on user-local pip state.
    env['PYTHONPATH'] = os.pathsep.join((str(ROOT/'src'),str(ROOT),env.get('PYTHONPATH','')))
    cmd = [sys.executable,'-m',module]+arguments
    proc = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=120, check=False)
    logpath.parent.mkdir(parents=True, exist_ok=True)
    logpath.write_text('Command: '+repr(cmd)+'\n'+proc.stdout,encoding='utf-8')
    if proc.returncode:
        raise RuntimeError(f'{module} failed with exit {proc.returncode}; original log: {logpath}')


def grounded_p61(out_dir: Path) -> dict:
    # Printed p61 illustration has SYMBOLIC Q, sphere R and centre distance r;
    # all concrete SI numbers here are teacher-chosen and explicitly labeled.
    teacher_q = +1e-9
    radius = .10
    centre_distance = .32
    source_xyz = (-centre_distance,0.,0.)
    problem = ElectrostaticProblem(
        (PointCharge(teacher_q,source_xyz),),
        (SolidSphere(radius,Grounded(),name='p61 source-faithful solid sphere'),),
    )
    solution=ExternalSphereBackend().solve(problem)
    angles=(0.,math.pi/6,math.pi/3,math.pi/2,2*math.pi/3,math.pi)
    charges=solution.conductor_results
    result = {
        'schema': 'conductor_sim.m41a_p61_grounded_static_teacher_example.v1',
        'source_id': 'p61-external-grounded-sphere',
        'source_pdf_page': 61, 'source_anchor': 'Illustration 1',
        'original_symbolic_givens': ['external charge +Q','grounded conducting sphere radius R','centre distance r>R'],
        'original_asked': 'static final sphere charge after switch closure; no current-versus-time asked',
        'teacher_numeric_values_not_printed': {'Q_c':teacher_q,'R_m':radius,'r_m':centre_distance,
                                               'charge_position_m':list(source_xyz),'sphere_center_m':[0.,0.,0.]},
        'not_physical_time_evolution': True,
        'native_sphere_net_charge_c':charges[0].net_charge_c,
        'native_sphere_potential_v':charges[0].potential_v,
        'native_image_charge_c':solution.image_charge_c,
        'native_signed_sigma_samples':[
            {'theta_rad':angle,'sigma_c_per_m2':solution.surface_charge_density('outer',angle)}
            for angle in angles
        ],
        'source_symbolic_final_charge': '-Q*R/r',
    }
    out_dir.mkdir(parents=True,exist_ok=True)
    path=out_dir/'p61-native.json'
    path.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    # A simple static schematic; NO 3D E field, charge motion, wire current or
    # unqualified area integration. Labels are original-vs-teacher explicit.
    native_charge=charges[0].net_charge_c
    rings=''.join('<tr><td>'+format(a['theta_rad'],'.12g')+'</td><td>'+format(a['sigma_c_per_m2'],'.12g')+'</td></tr>'
                  for a in result['native_signed_sigma_samples'])
    page=("<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
          "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'\">"
          "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
          "<title>Source p61: grounded-sphere static final charge</title>"
          "<style>body{font:16px system-ui;margin:2rem auto;padding:0 1rem;max-width:56rem;line-height:1.5}"
          "table{border-collapse:collapse;width:100%}td,th{padding:.45rem;border:1px solid #aaa;text-align:left}"
          "svg{width:100%;max-width:35rem;height:auto} .limit{border-left:4px solid #555;padding-left:1rem}"
          "</style><h1>Static final charge: original notes p61 Illustration 1</h1>"
          "<p>Original source provides symbols Q, R, r only. A grounded sphere next to +Q has "
          "<strong>Q<sub>sphere</sub> = −QR/r</strong> for the final equilibrium.</p>"
          "<p class=\"limit\">Teacher-authored example, NOT source numbers: Q=+1e−9 C, "
          "R=0.10 m, r=0.32 m; no physical switch current, elapsed-time charge "
          "transfer, electron motion or numerical time simulation is claimed.</p>"
          "<svg viewBox=\"0 0 560 200\" role=\"img\" aria-label=\"Point charge left of grounded solid sphere\">"
          "<line x1=\"60\" y1=\"100\" x2=\"480\" y2=\"100\" stroke=\"#bbb\" stroke-dasharray=\"4 4\"/>"
          "<circle cx=\"360\" cy=\"100\" r=\"65\" fill=\"#dde\" stroke=\"#334\" stroke-width=\"3\"/>"
          "<circle cx=\"152\" cy=\"100\" r=\"7\" fill=\"#a42\"/>"
          "<text x=\"85\" y=\"85\">+Q</text><text x=\"340\" y=\"105\">V=0</text>"
          "<text x=\"230\" y=\"135\">r=0.32 m</text></svg>"
          '<h2>Native final sphere net charge</h2><p id="p61-final-charge">'+format(native_charge,'.17g')+' C</p>'
          '<h2>Six signed native samples — pointwise, not a surface integral</h2>'
          '<table><thead><tr><th>Angle θ (radians)</th><th>σ (C/m²)</th></tr></thead><tbody>'+rings+'</tbody></table>'
          '<p>Source PDF: '+html.escape(NOTES)+' · page 61 · Illustration 1. '
          'Externally edited HTML is NOT authenticated by the separate JSON checker.</p></html>')
    (out_dir/'p61-static-teacher.html').write_text(page,encoding='utf-8')
    return result


def execution_checkout():
    """Record the checked-out revision and whether tracked sources have local edits."""
    def git(*args):
        return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()

    try:
        head = git('rev-parse', '--verify', 'HEAD')
        dirty = bool(git('status', '--porcelain', '--untracked-files=no'))
    except (OSError, subprocess.CalledProcessError):
        return {'status': 'unavailable'}
    return {'status': 'tracked checkout', 'commit': head, 'tracked_edits': dirty}


def generate(output_dir: Path) -> dict[str,Any]:
    checksums=check_corpus()
    output_dir.mkdir(parents=True,exist_ok=True)
    commands=(
      ('m36b','tools.m36b_jee_replay','tools.m36b_jee_reference','native-replays.json'),
      ('m36c','tools.m36c_jee_conductor_replay','tools.m36c_jee_conductor_reference','native-conductors.json'),
      ('m36d','tools.m36d_true_solid_sphere','tools.m36d_solid_reference','native-solid-sphere.json'),
      ('m37b','tools.m37b_source_approach','tools.m37b_source_reference','p8-ground-current.json'),
    )
    references={}
    for name, runner, checker, manifest in commands:
        dest=output_dir/name
        run_script(runner,['--output-dir',str(dest)],dest/'native.log')
        if not (dest/manifest).is_file(): raise RuntimeError('missing authentic native output: '+manifest)
        ref=dest/'separate-reference.json'
        run_script(checker,['--input',str(dest/manifest),'--output',str(ref)],dest/'reference.log')
        references[name]=json.loads(ref.read_text(encoding='utf-8'))
    grounded_p61(output_dir/'p61')
    run_script('tools.m41a_jee_trial_reference',
               ['--input',str(output_dir/'p61/p61-native.json'),
                '--output',str(output_dir/'p61/separate-reference.json')],
               output_dir/'p61/reference.log')
    references['p61']=json.loads((output_dir/'p61/separate-reference.json').read_text(encoding='utf-8'))

    case_evidence={}
    for ident,pdf,page,anchor,route in COHORT:
        if route=='m36b':
            if ident not in references['m36b']['independently_checked_cases']:
                raise RuntimeError('missing independently executed M36b case '+ident)
        if route=='m36c':
            if ident not in references['m36c']['source_independent_reference_results']:
                raise RuntimeError('missing independently executed M36c case '+ident)
        if route=='m36d' and not (references['m36d']['qualified_original_jee_solid_sphere_apparatus']==1
                                and references['m36d']['actual_solid_source_states_independently_checked']==3):
            raise RuntimeError('missing independently executed solid sphere case')
        if route=='m37b' and (references['m37b']['original_answer_options'] != [1,3]
                              or references['m37b']['verified_python_kelvin_states'] != 31):
            raise RuntimeError('M37b source options/current independent physics not checked')
        if route=='p61' and not references['p61']['qualified_static_grounded_sphere']:
            raise RuntimeError('p61 source-specific Kelvin reference not qualified')
        verified = route!='refuse'
        route_dir=output_dir/route if verified else None
        case_evidence[ident]={
            'source_id':ident, 'pdf':pdf, 'pdf_page':page, 'printed_anchor':anchor,
            'source_kind': source_kind(ident, pdf),
            'pdf_visible_caption': ('Paragraph for Questions 3 and 4' if ident.startswith('jee2021') else anchor),
            'official_paper_crosswalk_status': ('historical M36a mapping to JEE Advanced 2021 Paper 1 Q9-10; not independently reauthenticated from official paper in M41a'
                                                if ident.startswith('jee2021') else 'not_official_paper_claimed'),
            'native_route':route,
            'native_physical_status': ('verified_scoped_native' if verified else 'unsupported_original_geometry'),
            'native_reference_status': ('passed_separate_source_specific_reference' if verified
                                        else 'not_simulated_refusal'),
            'visual_html_status': ('example_generated_not_a_browser_pass' if verified else 'not_generated'),
            'real_chromium_status': 'not_measured_by_this_trial',
            'textbook_answer_completeness': ('source_symbolic_final_charge_verified' if route=='p61'
                else 'not_claimed_as_entire_original_answer'),
            'educational_usefulness': 'not_evaluated_with_students',
            'qualified_refusal': REFUSALS[ident][1] if not verified else None,
            'source_givens_vs_teacher': teacher_attribution(route),
        }
    artifacts={}
    for route, native_name in NATIVE_FILES.items():
        artifacts[route]={
            'native_path':route+'/'+native_name,
            'native_sha256':file_sha(output_dir/route/native_name),
            'separate_reference_path':route+'/separate-reference.json',
            'separate_reference_sha256':file_sha(output_dir/route/'separate-reference.json'),
        }
    report={
      'schema': SCHEMA,
      'method':'source PDFs SHA-verified, released native source runners executed, separate SI references executed',
      'frozen_charter_main_commit': BASE_SHA,
      'execution_checkout': execution_checkout(),
      'source_pdf_sha256': checksums,
      'registered_source_anchor_denominator':len(COHORT),
      'physically_native_and_independently_referenced_scoped_routes':sum(x['native_physical_status']=='verified_scoped_native' for x in case_evidence.values()),
      'explicit_unsupported_original_geometry_count':sum(x['native_physical_status']=='unsupported_original_geometry' for x in case_evidence.values()),
      'measured_educational_success_cases':0,
      'automatic_full_original_jee_question_solved_score': 'NOT MEASURED; NOT CLAIMED',
      'actual_browser_verified_count_in_this_run':0,
      'actual_source_pdf_visual_reinspection':'manually performed on original source pages; independent reviewer must recheck images',
      'note':'source-backed native routes are NOT full original answer or empirically verified teaching effectiveness',
      'evidence_files':artifacts,
      'cases':[case_evidence[x[0]] for x in COHORT],
    }
    path=output_dir/'m41a-trial.json'
    path.write_text(json.dumps(report,indent=2,allow_nan=False,ensure_ascii=False)+'\n',encoding='utf-8')
    check_trial(report,output_dir)
    return report


def check_trial(report:dict[str,Any],evidence_dir:Path) -> None:
    if type(report) is not dict or report.get('schema')!=SCHEMA:
        raise ValueError('invalid M41a schema')
    if report.get('source_pdf_sha256')!=check_corpus():
        raise ValueError('source PDF digests not original')
    if report.get('frozen_charter_main_commit') != BASE_SHA:
        raise ValueError('incorrect frozen charter baseline')
    checkout = report.get('execution_checkout')
    if checkout != {'status': 'unavailable'}:
        if (type(checkout) is not dict or set(checkout) != {'status', 'commit', 'tracked_edits'}
                or checkout['status'] != 'tracked checkout' or type(checkout['tracked_edits']) is not bool
                or type(checkout['commit']) is not str or len(checkout['commit']) != 40
                or any(c not in '0123456789abcdef' for c in checkout['commit'])):
            raise ValueError('invalid execution checkout record')
    rows=report.get('cases')
    if type(rows) is not list or len(rows)!=len(COHORT):
        raise ValueError('source cohort denominator was changed')
    if type(report.get('registered_source_anchor_denominator')) is not int or report['registered_source_anchor_denominator']!=14:
        raise ValueError('denominator inconsistent')
    for row,expected in zip(rows,COHORT):
        ident,pdf,page,anchor,route=expected
        if type(row) is not dict or (row.get('source_id'),row.get('pdf'),row.get('pdf_page'),
                                     row.get('printed_anchor'),row.get('native_route'))!=expected:
            raise ValueError('source attribution/order/route misrepresented: '+ident)
        if type(row['pdf_page']) is not int:
            raise ValueError('Boolean-as-page number')
        verified=route!='refuse'
        if row.get('native_physical_status')!=('verified_scoped_native' if verified else 'unsupported_original_geometry'):
            raise ValueError('native/refusal outcome misrepresented: '+ident)
        if row.get('native_reference_status')!=('passed_separate_source_specific_reference' if verified else 'not_simulated_refusal'):
            raise ValueError('independent reference outcome misrepresented: '+ident)
        if row.get('real_chromium_status')!='not_measured_by_this_trial':
            raise ValueError('browser verification cannot be inferred from HTML: '+ident)
        if row.get('educational_usefulness')!='not_evaluated_with_students':
            raise ValueError('unsupported teaching usefulness claim')
        if row.get('textbook_answer_completeness')!=('source_symbolic_final_charge_verified' if route=='p61'
                         else 'not_claimed_as_entire_original_answer'):
            raise ValueError('unsupported original-JEE full answer claim')
        if row.get('visual_html_status')!=('example_generated_not_a_browser_pass' if verified else 'not_generated'):
            raise ValueError('HTML existence/browser conflation')
        if row.get('qualified_refusal')!=(REFUSALS[ident][1] if not verified else None):
            raise ValueError('invalid physical refusal')
        if ident.startswith('jee2021') and ('one apparatus' not in row.get('source_kind','') or
                row.get('pdf_visible_caption')!='Paragraph for Questions 3 and 4' or
                row.get('official_paper_crosswalk_status')!='historical M36a mapping to JEE Advanced 2021 Paper 1 Q9-10; not independently reauthenticated from official paper in M41a'):
            raise ValueError('workbook printed label versus historical official apparatus crosswalk incorrect')
        if not ident.startswith('jee2021') and (row.get('pdf_visible_caption')!=anchor or
                row.get('official_paper_crosswalk_status')!='not_official_paper_claimed'):
            raise ValueError('false official-source attribution')
        if row.get('source_kind') != source_kind(ident, pdf):
            raise ValueError('incorrect source kind: '+ident)
        if row.get('source_givens_vs_teacher') != teacher_attribution(route):
            raise ValueError('teacher numerical attribution missing')
    for field,value in [('physically_native_and_independently_referenced_scoped_routes',12),
                        ('explicit_unsupported_original_geometry_count',2),
                        ('actual_browser_verified_count_in_this_run',0),
                        ('measured_educational_success_cases',0)]:
        if type(report.get(field)) is not int or report[field]!=value:
            raise ValueError('unsupported metric or Boolean-as-zero: '+field)
    if report.get('automatic_full_original_jee_question_solved_score')!='NOT MEASURED; NOT CLAIMED':
        raise ValueError('fabricated JEE full-question score')
    paths=report.get('evidence_files')
    if type(paths) is not dict or set(paths)!={'m36b','m36c','m36d','m37b','p61'}:
        raise ValueError('missing native/reference evidence route')
    for route,record in paths.items():
        if type(record) is not dict:
            raise ValueError('bad detached evidence route')
        for stem in ('native','separate_reference'):
            rel=record.get(stem+'_path')
            digest=record.get(stem+'_sha256')
            if type(rel) is not str or type(digest) is not str or len(digest)!=64:
                raise ValueError('invalid evidence path/digest')
            expected_name = NATIVE_FILES[route] if stem == 'native' else 'separate-reference.json'
            if rel != f'{route}/{expected_name}':
                raise ValueError('incorrect evidence file for '+stem+' in '+route)
            file=evidence_dir/rel
            if not file.is_file() or file.resolve().parent != (evidence_dir/route).resolve():
                raise ValueError('missing or relocated actual evidence file')
            if file_sha(file)!=digest:
                raise ValueError('actual native or independent SI evidence modified: '+route)
        native = json.loads((evidence_dir / record['native_path']).read_text(encoding='utf-8'))
        expected_reference = REFERENCE_CHECKERS[route](native)
        saved_reference = json.loads((evidence_dir / record['separate_reference_path']).read_text(encoding='utf-8'))
        if json.dumps(saved_reference, sort_keys=True) != json.dumps(expected_reference, sort_keys=True):
            raise ValueError('saved reference differs from independent replay: '+route)
    # Digest equality certifies these detached bytes relative to this report,
    # NOT a trusted teacher history or arbitrary HTML published elsewhere.


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--verify-existing',action='store_true')
    args=p.parse_args()
    if args.verify_existing:
        report=json.loads((args.output_dir/'m41a-trial.json').read_text(encoding='utf-8'))
        check_trial(report,args.output_dir)
    else:
        report=generate(args.output_dir)
    print('M41a registered source anchors:',report['registered_source_anchor_denominator'])
    print('Scoped native+independent reference routes:',report['physically_native_and_independently_referenced_scoped_routes'])
    print('Honest unsupported original geometries:',report['explicit_unsupported_original_geometry_count'])
    print('Verified full JEE solution percentage: NOT MEASURED')

if __name__=='__main__':
    main()
