"""Conservative transitive CI coverage for the newer scientific consumers."""
from fnmatch import fnmatchcase
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
# This regression is also runnable against an explicitly supplied baseline.
import os
ROOT = Path(os.environ.get('CONDUCTOR_AUDIT_ROOT', ROOT))

DEPENDENCIES = {
    'm37b-source-current.yml': ('src/conductor_sim/model.py', 'tools/m37a_ground_wire_reference.py', 'tools/check_m37b_nojs_hostility.py'),
    'm38a-prescribed-line.yml': ('src/conductor_sim/model.py', 'src/conductor_sim/external_sphere.py', 'tools/m37a_ground_wire_reference.py'),
    'm38b-line-viewer.yml': ('src/conductor_sim/model.py', 'src/conductor_sim/grounded_wire_current.py', 'tools/m38a_line_reference.py'),
    'm39a-gauss-cavities.yml': ('src/conductor_sim/model.py', 'pyproject.toml'),
    'm39b-topology-viewer.yml': ('src/conductor_sim/cavity_gauss.py', 'tools/m39a_gauss_reference.py'),
    'm40a-spherical-multicavity.yml': ('src/conductor_sim/cavity_gauss.py', 'src/conductor_sim/model.py', 'tools/m39a_gauss_reference.py'),
    'm40b-spherical-atlas.yml': ('src/conductor_sim/spherical_multicavity.py', 'src/conductor_sim/cavity_gauss.py', 'tools/m40a_spherical_reference.py', 'tools/m39a_gauss_reference.py'),
}

@pytest.mark.parametrize('workflow,dependency', [
    (workflow, dependency)
    for workflow, dependencies in DEPENDENCIES.items()
    for dependency in dependencies
])
def test_scientific_consumers_trigger_on_upstream_changes(workflow, dependency):
    text = (ROOT / '.github/workflows' / workflow).read_text(encoding='utf-8')
    pr = text.split('  pull_request:', 1)[1].split('  push:', 1)[0]
    patterns = re.findall(r"^\s+- ['\"]([^'\"]+)['\"]\s*$", pr, re.MULTILINE)
    assert any(fnmatchcase(dependency, pattern) for pattern in patterns), (workflow, dependency)
