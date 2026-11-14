"""M22c: reusable reflected uniform-rod source motion, not an exam-specific solver.

Physical source values for P1 come from the frozen M22b diagnostic;
P2 is a distinct JEE-style teacher hypothetical. All held rods remain
nonresponding fixed uniform line charges. No mechanical guide is present.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from conductor_sim import (FixedPointField, PointCharge, PrescribedField,
    TestChargeMotionSpec, UniformElectricField, UniformGravity,
    UniformRodField, integrate_test_charge)
from conductor_sim.scenarios import PrescribedScenario, sample_scenario_viewer
from conductor_sim.source_symmetry import certify_rod_reflection
from conductor_sim.viewer import render_html

CASES=('P1','P2')


def make_case(case: str) -> PrescribedScenario:
    if case == 'P1':
        field=PrescribedField((
            UniformRodField(+1.5e-9,(-.05,+.07,-.02),(.05,+.07,-.02)),
            UniformRodField(+1.5e-9,(-.05,-.07,-.02),(.05,-.07,-.02)),
        ))
        particle=PointCharge(-4e-10,(0.,0.,.1))
        mass=3e-6
        velocity=(.03,0.,0.)
        gravity=None
        annotations=(('rod:upper','Upper stationary rod'),
                     ('rod:lower','Lower stationary rod'))
        provenance=('Teacher hypothetical from M22b frozen geometric boundary: '
                    'two equal +1.5 nC fixed rods mirrored at y=+/-0.07 m. '
                    'Particle charge -0.4 nC and mass 3 mg are illustrative; '
                    'initial velocity is source-given; no source motion.')
    elif case == 'P2':
        field=PrescribedField((
            UniformRodField(-.8e-9,(-.06,.085,-.03),(.055,.12,.04)),
            UniformRodField(-.8e-9,(.055,-.12,.04),(-.06,-.085,-.03)),
            FixedPointField(.55e-9,(.16,0.,.09)),
            UniformElectricField((40.,0.,-30.)),
        ))
        particle=PointCharge(+5e-10,(.02,0.,.13))
        mass=4e-6
        velocity=(.03,0.,-.025)
        gravity=UniformGravity((0.,0.,-.3))
        annotations=(('rod:tilted-1','Tilted stationary rod A'),
                     ('rod:tilted-2','Mirrored stationary rod B'),
                     ('point:fixed','Held in-plane point'),
                     ('field:external','External in-plane E'))
        provenance=('Teacher hypothetical: two negatively charged tilted finite rods with reversed '
                    'endpoint order on mirrored partner, plus separately fixed in-plane '
                    'point source, in-plane external field and uniform gravity. '
                    'No prescribed guide, source response, or off-plane proxy.')
    else:
        raise ValueError('unknown mirrored-rod case')
    return PrescribedScenario(
        scenario_id='m22c-'+case,
        spec=TestChargeMotionSpec(field, particle, mass, velocity,
            duration_s=.14, sample_interval_s=.01, plane_normal=(0.,1.,0.),
            length_scale_m=.4, source_guard_m=.002, gravity=gravity),
        source_annotations=annotations, provenance=provenance)


def create_case(case):
    scenario=make_case(case)
    result=integrate_test_charge(scenario.spec)
    assert result.termination=='duration'
    bundle=sample_scenario_viewer(scenario,visual_frames=9,extent_m=.3,resolution=11)
    rods=tuple(s for s in scenario.spec.field.sources if type(s) is UniformRodField)
    cert=certify_rod_reflection(rods,origin_m=scenario.spec.particle.position_m,
                                normal=scenario.spec.plane_normal)
    audit=dict(schema='conductor_sim.m22c_example.v1',case=case,
               source_fidelity='teacher hypothetical with explicit authored physics',
               scenario=scenario.as_dict(),certificate=asdict(cert),
               full_motion=result.as_dict(),viewer=bundle)
    return render_html(bundle,title=f'M22c — {case}: actual mirrored-rod motion'),audit


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case',choices=CASES,default='P1')
    ap.add_argument('--output',type=Path,default=Path('output/m22c-rods.html'))
    a=ap.parse_args()
    html,audit=create_case(a.case)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(html,encoding='utf-8')
    p=a.output.with_suffix('.json')
    p.write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(a.case, a.output, p, 'duration',audit['full_motion']['samples'][-1]['time_s'])

if __name__=='__main__': main()
