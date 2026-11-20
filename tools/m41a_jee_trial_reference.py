"""Separate STD-LIB ONLY analytic Kelvin reference for source p61 static final charge.

NO conductor_sim imports, NO reuse of production image-sphere equations or
source model; this is an independent normal-field and total-charge oracle.
"""
from __future__ import annotations

import argparse
import json
from math import cos,isclose,isfinite,pi
from pathlib import Path


def check(document:dict)->dict:
    if type(document) is not dict or document.get('schema')!='conductor_sim.m41a_p61_grounded_static_teacher_example.v1':
        raise ValueError('not an original p61 teacher model')
    if document.get('source_id')!='p61-external-grounded-sphere' or document.get('source_pdf_page')!=61 or document.get('source_anchor')!='Illustration 1':
        raise ValueError('false original p61 attribution')
    v=document['teacher_numeric_values_not_printed']
    if type(v) is not dict or set(v)!={'Q_c','R_m','r_m','charge_position_m','sphere_center_m'}:
        raise ValueError('teacher source must be fully attributed')
    q,R,d=(v[k] for k in ('Q_c','R_m','r_m'))
    if any(type(x) is not float or not isfinite(x) for x in (q,R,d)) or q<=0 or not 0<R<d:
        raise ValueError('unqualified actual original grounded sphere example')
    if v['charge_position_m']!=[-d,0.0,0.0] or v['sphere_center_m']!=[0.0,0.0,0.0]:
        raise ValueError('teacher geometry inconsistent with distance')
    if document['original_symbolic_givens']!=['external charge +Q','grounded conducting sphere radius R','centre distance r>R']:
        raise ValueError('source givens were changed')
    if document['not_physical_time_evolution'] is not True or document['source_symbolic_final_charge']!='-Q*R/r':
        raise ValueError('fabricated time current or false p61 formula')
    expected_charge = -q*R/d
    for key in ('native_sphere_net_charge_c','native_image_charge_c'):
        got=document.get(key)
        if type(got) is not float or not isfinite(got) or not isclose(got,expected_charge,rel_tol=5e-14,abs_tol=1e-24):
            raise ValueError('induced Kelvin net charge does not match original p61')
    if type(document.get('native_sphere_potential_v')) is not float or document['native_sphere_potential_v']!=0.0:
        raise ValueError('grounded sphere must have zero potential')
    samples=document.get('native_signed_sigma_samples')
    if type(samples) is not list or len(samples)!=6:
        raise ValueError('original six static local density samples required')
    angles=(0.,pi/6,pi/3,pi/2,2*pi/3,pi)
    for i,(sample,angle) in enumerate(zip(samples,angles)):
        if type(sample) is not dict or set(sample)!={'theta_rad','sigma_c_per_m2'}:
            raise ValueError('local angular density witness shape changed')
        theta,sigma=sample['theta_rad'],sample['sigma_c_per_m2']
        if type(theta) is not float or type(sigma) is not float or not isfinite(theta) or not isfinite(sigma):
            raise ValueError('Boolean/nonfinite density or theta')
        if theta!=angle:
            raise ValueError('original surface ray was changed')
        # Independent Kelvin-image field normal on the physical solid-sphere surface.
        # Distinct denominator expression to the production's stable hypot route.
        distance = (d*d+R*R-2*d*R*cos(theta))**1.5
        analytic = -q*(d*d-R*R)/(4*pi*R*distance)
        if not isclose(sigma,analytic,rel_tol=1.5e-13,abs_tol=1e-25):
            raise ValueError(f'signed p61 local density mismatch at {i}')
    return {
        'schema':'conductor_sim.m41a_p61_kelvin_reference.v1',
        'qualified_static_grounded_sphere':True,
        'exact_original_symbolic_asked_charge':'-Q*R/r',
        'independent_teacher_final_charge_c':expected_charge,
        'independent_signed_density_angles':6,
        'not_time_current':True,
        'not_original_source_numerical_values':True,
    }


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    r=check(json.loads(a.input.read_text(encoding='utf-8')))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('M41a independent external Kelvin p61 final Q and 6 local sigma points PASS')

if __name__=='__main__':
    main()
