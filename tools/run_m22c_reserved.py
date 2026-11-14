"""One-time reserved M22c multi-pair transfer acceptance AFTER candidate freeze.

The model is read verbatim from the pre-implementation frozen JSON. This tool
is deliberately not invoked in developmental tests and has no production
model-law special cases. The first execution outcome must be retained.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from math import fsum, hypot, pi
from pathlib import Path

from conductor_sim import (EPSILON_0,FixedPointField,PointCharge,PrescribedField,
    TestChargeMotionSpec,UniformElectricField,UniformGravity,UniformRodField,
    integrate_test_charge)
from conductor_sim.scenarios import PrescribedScenario,sample_scenario_viewer
from conductor_sim.source_symmetry import certify_rod_reflection
from conductor_sim.viewer import render_html


def run(config_path, output):
    config=json.loads(Path(config_path).read_text(encoding='utf-8'))
    assert config['schema']=='conductor_sim.m22c.reserved.v1'
    assert config['case_id']=='R-multipair-mixed'
    src=[]
    for record in config['sources']:
        k=record['kind']
        if k=='rod': source=UniformRodField(record['charge_c'],record['start_m'],record['end_m'])
        elif k=='point':source=FixedPointField(record['charge_c'],record['position_m'])
        elif k=='uniform_E':source=UniformElectricField(record['field_v_m'])
        else: raise AssertionError('unsupported reserved source kind')
        src.append(source)
    p=config['particle']; g=config['gravity_acceleration_m_s2']
    spec=TestChargeMotionSpec(
        field=PrescribedField(tuple(src)),
        particle=PointCharge(p['charge_c'],p['position_m']),
        mass_kg=p['mass_kg'],initial_velocity_m_s=p['velocity_m_s'],
        duration_s=config['duration_s'],sample_interval_s=config['sample_interval_s'],
        plane_normal=config['plane_normal'],length_scale_m=config['length_scale_m'],
        gravity=UniformGravity(g),source_guard_m=.002)
    scenario=PrescribedScenario(
        scenario_id=config['case_id'],spec=spec,
        source_annotations=tuple((r['id'],r['id'].replace(':',' ')) for r in config['sources']),
        provenance='Reserved M22c teacher hypothetical, frozen before implementation; not an official JEE past-paper simulation.')
    rods=tuple(s for s in spec.field.sources if type(s) is UniformRodField)
    certificate=certify_rod_reflection(rods,origin_m=spec.particle.position_m,
                                       normal=spec.plane_normal)
    assert len(certificate.pairs)==2 and len(certificate.invariant_indices)==1
    # Independent Coulomb line quadrature at five distinct, non-singular
    # actual physical locations. Do NOT call source.E/V in the reference.
    K=1/(4*pi*EPSILON_0)
    errors=[]
    for test_p in ((0.,0.,-.02),(.04,0.,.02),(-.04,0.,.12),(.025,0.,.07),(-.08,0.,-.05)):
        e=[70.,0.,-25.];v=-70.*test_p[0]+25.*test_p[2]
        for s in rods:
            count=4096
            for j in range(count):
                u=(j+.5)/count
                site=tuple(s.start_m[i]+u*(s.end_m[i]-s.start_m[i]) for i in range(3))
                d=tuple(test_p[i]-site[i] for i in range(3))
                r=hypot(*d)
                dv=K*s.charge_c/count/r
                v+=dv
                for i in range(3):e[i]+=dv*d[i]/r**2
        point=src[5]
        d=tuple(test_p[i]-point.position_m[i] for i in range(3))
        r=hypot(*d)
        dv=K*point.charge_c/r
        v+=dv
        for i in range(3):e[i]+=dv*d[i]/r**2
        exact_v=spec.field.potential_at(test_p)
        exact_e=spec.field.electric_field_at(test_p)
        verr=abs(v-exact_v);eerr=hypot(*(a-b for a,b in zip(e,exact_e)))
        errors.append(dict(point_m=test_p,potential_abs_error_v=verr,field_abs_error_v_m=eerr))
        assert verr<5e-5, errors[-1]
        assert eerr<5e-3,errors[-1]
        assert abs(exact_e[1])<1e-9,errors[-1]
    result=integrate_test_charge(spec)
    assert result.termination=='duration' and len(result.samples)==13
    assert result.max_abs_energy_drift_j<2e-11
    for row in result.samples:
        assert abs(row.position_m[1])<1e-10
        assert abs(row.force_n[1])<1e-13
        assert abs(row.velocity_m_s[1])<1e-10
    bundle=sample_scenario_viewer(scenario,visual_frames=9,resolution=11,extent_m=.32)
    assert bundle['prescribed_motion']['mirrored_rod_source_indices']==[0,1,2,3]
    html=render_html(bundle,title='M22c reserved transfer: two rod pairs and in-plane sources')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(html,encoding='utf-8')
    audit=dict(schema='conductor_sim.m22c_reserved_evidence.v1',
        original_frozen_case=config,model=scenario.as_dict(),certificate=asdict(certificate),
        independent_reference=errors,trajectory=result.as_dict(),viewer=bundle,
        report='Executed on the candidate after preselection; not blind or independent.')
    output.with_suffix('.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return dict(case=config['case_id'],rods=len(rods),pairs=len(certificate.pairs),
       invariant_rods=len(certificate.invariant_indices),
       max_reference_e_error=max(x['field_abs_error_v_m'] for x in errors),
       max_reference_v_error=max(x['potential_abs_error_v'] for x in errors),
       termination=result.termination,energy_drift_j=result.max_abs_energy_drift_j,
       omitted_in_view=bundle['prescribed_motion']['mirrored_rod_source_indices'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=Path('tests/m22c_reserved_case.json'))
    ap.add_argument('--output',type=Path,default=Path('output/m22c-reserved.html'))
    a=ap.parse_args()
    print(json.dumps(run(a.config,a.output),indent=2))

if __name__=='__main__':main()
