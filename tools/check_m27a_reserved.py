"""M27a predeclared reserved R1: first CI execution AFTER candidate freeze.

The numerical parameters were frozen in docs/m27a_charter.md before any
production implementation. This script independently re-integrates the
Coulomb circular line with a midpoint rule; it does not use Carlson RF/RD
for its expected answer. It must NOT run as a developmental retuning loop.
"""
from __future__ import annotations

import argparse
import json
from math import cos, hypot, pi, sin
from pathlib import Path

from conductor_sim import EPSILON_0, UniformRingField


def independent_coulomb(q, a, center, axis, p, eps, n=16384):
    axis_length=hypot(*axis)
    w=tuple(t/axis_length for t in axis)
    seed=(1.,0.,0.) if abs(w[0])<.8 else (0.,1.,0.)
    dot=sum(seed[i]*w[i] for i in range(3))
    u=tuple(seed[i]-dot*w[i] for i in range(3))
    un=hypot(*u);u=tuple(t/un for t in u)
    v=(w[1]*u[2]-w[2]*u[1],w[2]*u[0]-w[0]*u[2],w[0]*u[1]-w[1]*u[0])
    potential=0.
    field=[0.,0.,0.]
    for i in range(n):
        t=2*pi*(i+.5)/n
        x=tuple(center[j]+a*(cos(t)*u[j]+sin(t)*v[j]) for j in range(3))
        d=tuple(p[j]-x[j] for j in range(3))
        r=hypot(*d)
        potential+=1/r
        for k in range(3):
            field[k]+=d[k]/r**3
    k=q/(4*pi*eps*n)
    return k*potential,tuple(k*t for t in field)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("output/m27a-reserved.json"))
    args=parser.parse_args()
    q=-2.3e-9;a=.17
    c=(.08,-.06,.11);axis=(2.,-1.,3.)
    p=(.155,.047,.227);eps=1.35*EPSILON_0
    source=UniformRingField(q,a,c,axis)
    observed_v=source.potential_at(p,permittivity_f_m=eps)
    observed_e=source.electric_field_at(p,permittivity_f_m=eps)
    reference_v,reference_e=independent_coulomb(q,a,c,axis,p,eps)
    relative_v=abs(observed_v-reference_v)/max(abs(reference_v),1e-30)
    relative_e=max(abs(x-y)/max(abs(y),1e-30) for x,y in zip(observed_e,reference_e))
    step=2e-6
    gradient=[]
    for j in range(3):
        plus=list(p);minus=list(p);plus[j]+=step;minus[j]-=step
        gradient.append(-(source.potential_at(plus,permittivity_f_m=eps)-
                          source.potential_at(minus,permittivity_f_m=eps))/(2*step))
    gradient_error=max(abs(x-y)/max(abs(y),1e-30) for x,y in zip(observed_e,gradient))
    report=dict(case="M27a frozen reserved R1",
        provenance="Teacher-generated reserved card, frozen before production implementation; not blinded independent authoring",
        source=source.as_dict(),sample_point_m=p,permittivity_f_m=eps,
        observed_potential_v=observed_v,reference_coulomb_potential_v=reference_v,
        observed_field_v_m=observed_e,reference_coulomb_field_v_m=reference_e,
        independent_gradient_v_m=gradient,
        relative_potential_error=relative_v,
        max_relative_component_field_error=relative_e,
        max_relative_gradient_error=gradient_error,
        source_clearance_m=source.clearance_at(p))
    assert relative_v<3e-11,report
    assert relative_e<3e-10,report
    assert gradient_error<2e-8,report
    report["result"]="pass"
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("result","relative_potential_error",
         "max_relative_component_field_error","max_relative_gradient_error")},sort_keys=True))


if __name__=="__main__":
    main()
