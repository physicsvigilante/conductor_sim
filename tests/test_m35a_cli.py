"""Checkout-only tools must work in CI without a developer's PYTHONPATH."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from tools.check_m35a_references import commands


ROOT = Path(__file__).resolve().parents[1]


def _run_module(module: str, *args: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    result = subprocess.run(
        [sys.executable, '-m', module, *args], cwd=ROOT, env=environment,
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_reference_launcher_uses_checkout_module_for_native_audit(tmp_path):
    invocation = commands(tmp_path)[-1]
    assert invocation[:2] == ['-m', 'tools.m35a_audit']


def test_audit_module_without_pythonpath_checks_candidate_and_open_findings(tmp_path):
    output = tmp_path / 'audit.json'
    _run_module('tools.m35a_audit', '--expect', 'candidate', '--output', str(output))
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['result'] == 'AUDIT_COMPLETE_WITH_DOCUMENTED_OPEN_FINDINGS'
    assert Path(report['imported_production_root']) == ROOT / 'src/conductor_sim'
    assert Path(report['imported_example_path']).is_relative_to(ROOT / 'examples')
    fixes = report['reproduced_corrections']
    assert fixes['mutated_self_pair']['rejected'] is True
    assert fixes['malformed_observation']['rejected'] is True
    assert 'observation' in fixes['malformed_observation']['error']
    assert fixes['subnormal_normal']['unit_within_ulp'] is True
    assert type(report['open_legacy_findings']['event_sweep']['accepts_complete_without_native']) is bool
    assert type(report['open_legacy_findings']['trajectory_sweep']['accepts_complete_without_native']) is bool


def test_legacy_snapshot_module_without_pythonpath_writes_five_native_records(tmp_path):
    _run_module('tools.m35a_legacy_snapshot', '--output', str(tmp_path))
    names = {'graph-v1', 'graph-v2', 'graph-v3', 'fork-v1', 'inquiry-v1'}
    assert {p.stem for p in tmp_path.glob('*.json')} == names
    for name in names:
        value = json.loads((tmp_path / (name + '.json')).read_text(encoding='utf-8'))
        assert value['schema'].startswith('conductor_sim.')
