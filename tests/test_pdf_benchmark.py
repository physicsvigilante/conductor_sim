"""M6 PDF-based authorability benchmark; no FEMM process required."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from conductor_sim import Femm5Backend, NestedShellBackend
from conductor_sim.pdf_benchmark import (
    CASES, PDF_NAME, PDF_PAGES, PDF_SHA256, export_case,
    load_case, report, run_checks,
)
from conductor_sim.solver import UnsupportedProblem


ROOT = Path(__file__).resolve().parents[1]


def test_source_pdf_provenance_is_exact():
    """If PDF is replaced/revised, reconfirm every page anchor before accepting."""
    pdf = ROOT / PDF_NAME
    assert pdf.is_file(), f"checked-in benchmark source missing: {pdf}"
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    assert digest == PDF_SHA256
    assert PDF_PAGES == 71


def test_catalog_has_19_unique_page_anchored_problems():
    assert len(CASES) == 19
    assert len({case.id for case in CASES}) == len(CASES)
    assert len([case for case in CASES if case.runnable]) == 13
    assert len([case for case in CASES if not case.runnable]) == 6
    assert all(1 <= case.page <= PDF_PAGES and case.source_anchor and case.summary and case.note for case in CASES)
    assert all((case.classification == "runnable") == case.runnable for case in CASES)
    assert all(case.blocker for case in CASES if not case.runnable)


@pytest.mark.parametrize('case_id',[case.id for case in CASES if case.runnable])
def test_real_authored_models_work_without_engine_changes(case_id):
    case = load_case(case_id)
    authored = case.author()
    assert authored.states and authored.checks
    assert len({id(p) for label,p in authored.states}) == len(authored.states)
    analytical = NestedShellBackend()
    numerical = Femm5Backend()
    for label,p in authored.states:
        assert analytical.supports(p), (case_id,label)
        assert numerical.supports(p), (case_id,label)
    evaluated = run_checks(case)
    assert evaluated['result'] == 'passed', evaluated
    assert len(evaluated['checks']) >= 1


def test_unsupported_stays_explicit_and_fails_closed(tmp_path):
    for case in CASES:
        if case.runnable:
            continue
        result = run_checks(case)
        assert result['result'] == 'unsupported'
        assert result['reason'] == case.blocker
        with pytest.raises(UnsupportedProblem, match='PDF'):
            export_case(case.id, tmp_path / f'{case.id}.html')
        assert not (tmp_path / f'{case.id}.html').exists()


def test_report_json_is_machine_readable_and_error_free():
    r=report()
    assert r['runnable']==13 and r['blocked']==6 and r['failures']==0
    raw=json.dumps(r,allow_nan=False)
    decoded=json.loads(raw)
    assert decoded['sha256'] == PDF_SHA256
    assert sum(len(c.get('checks',())) for c in r['results']) >= 26


def test_case_pdf_anchor_and_analytical_sweep_export(tmp_path):
    html=tmp_path/'p29.html'
    output=export_case('p29-outer-charge-change',html,resolution=21)
    assert output['states']==2
    assert html.is_file()
    assert 'simulation-data' in html.read_text(encoding='utf-8')
    bundle=json.loads(html.with_suffix('.json').read_text(encoding='utf-8'))
    assert bundle['benchmark_source']['page']==29
    assert bundle['benchmark_source']['pdf_sha256']==PDF_SHA256
    assert len(bundle['frames'])==2
    assert all(len(frame['grid']['x_m'])==21 for frame in bundle['frames'])
    assert all(frame['backend']=='NestedShellSolution' for frame in bundle['frames'])
    assert bundle['frames'][0]['conductor_results'][0]['name']=='inner'


def test_nonuniform_induction_export_preserves_density_and_state(tmp_path):
    html=tmp_path/'p69.html'
    export_case('p69-cavity-central-vs-shifted',html,resolution=21)
    bundle=json.loads(html.with_suffix('.json').read_text(encoding='utf-8'))
    assert len(bundle['frames'])==2
    for frame in bundle['frames']:
        body=frame['conductor_results'][0]
        assert body['inner_surface_charge_c']==pytest.approx(-1e-9)
        assert body['outer_surface_charge_c']==pytest.approx(4e-9)
    before=next(p for p in bundle['frames'][0]['profiles'] if p['surface']=='inner')
    after=next(p for p in bundle['frames'][1]['profiles'] if p['surface']=='inner')
    assert abs(before['sigma_c_per_m2'][0]/before['sigma_c_per_m2'][-1]-1)<1e-12
    assert abs(after['sigma_c_per_m2'][0]) > 5*abs(after['sigma_c_per_m2'][-1])


def test_cli_check_and_unsupported_render(tmp_path):
    checked=tmp_path/'report.json'
    base=[sys.executable,'-m','conductor_sim.pdf_benchmark']
    proc=subprocess.run(base+['check','--output',str(checked)],cwd=ROOT,
                        capture_output=True,text=True,timeout=30)
    assert proc.returncode==0,proc.stderr
    assert json.loads(checked.read_text())['failures']==0
    rendered=tmp_path/'blocked.html'
    proc=subprocess.run(base+['render','p61-external-grounded-sphere','--output',str(rendered)],
                        cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert proc.returncode==2
    assert 'Historical M6a baseline' in proc.stderr
    assert not rendered.is_file()
