"""M19f: same two fixed rings, two teacher hypotheticals, synchronized offline views.

Source inspiration: 02_conductors_half_size.pdf page 12 Q40-41.
The original source has an UPWARD field and does not specify a numerical mass.
These simulations are explicitly modified teacher hypotheses: no field versus
an AXIAL field with an illustrative 0.10 kg particle mass. The original
freely moving axial scenario is unsupported and not silently approximated.

PowerShell:
    python examples/compare_fixed_scenarios.py --output output/m19f-rings.html
    Start-Process .\\output\\m19f-rings.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from conductor_sim import sample_prescribed_scenario_comparison, write_scenario_comparison
from examples.ring_source_orientation import make_case, SOURCE_ANNOTATIONS


def build_comparison():
    return sample_prescribed_scenario_comparison(
        make_case('field-off'),make_case('axial-field'),
        source_annotations=SOURCE_ANNOTATIONS,
        labels=('H1 · External field off','H4 · Field along world +x'),
        source_context=('Teacher hypotheticals from charged-ring workbook geometry. '
                        'Original upward applied field would invalidate free axial motion; '
                        '0.10 kg is an explicitly adopted illustrative particle mass.'),
        axis_orientation='horizontal',visual_frames=12,profile_samples=101,
        resolution=15,
    )


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('output/m19f-rings.html'))
    args=p.parse_args(argv)
    bundle=build_comparison()
    output=write_scenario_comparison(bundle,args.output,
                   title='Two charged rings — compare verified teacher scenarios')
    sidecar=output.with_suffix('.json')
    sidecar.write_text(json.dumps(bundle,allow_nan=False,indent=2)+'\n',encoding='utf8')
    print(f"Two real M19a trajectories; {len(bundle['time_s'])} exactly shared physical frame times; "
          f"{len(bundle['changes'])} actual model changes; source IDs {bundle['source_ids']}")
    print(f'Wrote {output} and {sidecar}. Both simulations retain their original full saved histories.')


if __name__=='__main__':
    main()
