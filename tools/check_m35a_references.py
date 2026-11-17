"""Dependency-aware source scientific regression suite; no new solver dependencies.

Run from repository root. Every command is a fresh native execution and a
separately expressed reference; failure exits nonzero and keeps partial logs.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys


def commands(out):
    def path(name): return str(out / name)
    pairs = [
        ['-m', 'examples.m32a_unified_observations', '--case', 'uniform-events', '--output', path('m32a.json')],
        ['tools/m32a_independent_uniform.py', '--native', path('m32a.json'), '--output', path('m32a-reference.json')],
        ['-m', 'examples.m32b_mixed_ancestry', '--output', path('m32b.json')],
        ['tools/m32b_independent_mixed.py', '--native', path('m32b.json'), '--output', path('m32b-reference.json')],
        ['-m', 'examples.m33a_cross_world', '--output', path('m33a.json')],
        ['tools/m33a_independent_crossworld.py', '--native', path('m33a.json'), '--output', path('m33a-reference.json')],
        ['-m', 'examples.m33b_event_inquiry', '--source-graph-output', path('m33b-source.json'), '--output', path('m33b-inquiry.json')],
        ['tools/m33b_independent_inquiry.py', '--source-graph', path('m33b-source.json'), '--inquiry', path('m33b-inquiry.json'), '--output', path('m33b-reference.json')],
        ['-m', 'examples.m34a_oblique_worlds', '--output', path('m34a.json')],
        ['tools/m34a_independent_oblique.py', '--native', path('m34a.json'), '--output', path('m34a-reference.json')],
        ['-m', 'examples.m34b_incremental_forks', '--original-output', path('m34b-source.json'), '--output', path('m34b-fork.json')],
        ['tools/m34b_independent_forks.py', '--original', path('m34b-source.json'), '--fork', path('m34b-fork.json'), '--output', path('m34b-reference.json')],
        ['-m', 'examples.m35a_multistage_exploration', '--output', path('m35a')],
        ['tools/m35a_independent_multistage.py', '--input', path('m35a'), '--output', path('m35a-reference.json')],
        ['tools/m35a_audit.py', '--expect', 'candidate', '--output', path('audit-with-open-findings.json')],
    ]
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('output/m35a-references'))
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = []
    try:
        for index, cmd in enumerate(commands(out), 1):
            label = f'{index:02d}-{Path(cmd[1] if cmd[0] == "-m" else cmd[0]).name}'
            print('EXECUTE', label, flush=True)
            with (out / (label + '.log')).open('w', encoding='utf-8') as log:
                result = subprocess.run([sys.executable, *cmd], stdout=log,
                                        stderr=subprocess.STDOUT, timeout=180)
            report.append(dict(command=cmd, returncode=result.returncode))
            if result.returncode:
                raise RuntimeError(f'{label} failed; retained original log at {out}')
    finally:
        (out / 'commands.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        files = sorted(p for p in out.rglob('*') if p.is_file() and p.name != 'sha256.txt')
        (out / 'sha256.txt').write_text(''.join(
            sha256(p.read_bytes()).hexdigest() + '  ' + str(p.relative_to(out)) + '\n'
            for p in files), encoding='utf-8')
    print(json.dumps(dict(verdict='PASS', commands=len(report),
                         caveat='Audit retains documented open legacy findings; not global certification.'), sort_keys=True))


if __name__ == '__main__':
    main()
