"""M27b reserved noncoaxial full-ring trajectory, frozen in docs/m27b_reserved_card.md.

Developer-exposed first execution occurs only after candidate freeze. This
independent reference uses direct Coulomb midpoint line integration and a
test-only velocity-Verlet scheme, not the production Carlson/ODE controller.
"""
from __future__ import annotations

import argparse
import json
from math import cos, hypot, pi, sin
from pathlib import Path

from conductor_sim import (EPSILON_0, PointCharge, PrescribedField,
    TestChargeMotionSpec, UniformElectricField, UniformGravity, UniformRingField,
    integrate_test_charge)


def _unit(v):
    h=hypot(*v)
    return tuple(x/h for x in v)


def _cross(a,b):
    return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])


def _sites(charge, radius, center, axis, *, count=512):
    w=_unit(axis)
    seed=(1.,0.,0.) if abs(w[0])<.8 else (0.,1.,0.)
    proj=sum(seed[j]*w[j] for j in range(3))
    u=_unit(tuple(seed[j]-proj*w[j] for j in range(3)))
    v=_cross(w,u)
    points=[]
    for i in range(count):
        a=2*pi*(i+.5)/count
        points.append(tuple(
            center[j]+radius*(cos(a)*u[j]+sin(a)*v[j]) for j in range(3)))
    return points, charge/(4*pi*EPSILON_0*count)


def _direct_force(p, sources, charge, mass, gravity):
    e=[.5,0.,-.3]
    for sites,amplitude in sources:
        for ring_point in sites:
            d=tuple(p[i]-ring_point[i] for i in range(3))
            r=hypot(*d)
            for i in range(3):
                e[i]+=amplitude*d[i]/r**3
    return tuple(charge*e[i]+mass*gravity[i] for i in range(3))


def run_reserved():
    sources=(
        UniformRingField(+3e-9,.16,center_m=(.05,0.,-.07),axis=(0.,0.,1.)),
        UniformRingField(-1.5e-9,.11,center_m=(-.1,0.,.12),axis=(1.,0.,0.)),
        UniformElectricField((.5,0.,-.3)),
    )
    particle=PointCharge(-4e-10,(.08,0.,.04))
    mass=3e-7
    gravity=(0.,0.,-.12)
    spec=TestChargeMotionSpec(
        PrescribedField(sources),particle,mass,(.1,0.,-.015),
        .07,.007,plane_normal=(0.,1.,0.),length_scale_m=.4,
        source_guard_m=.008,relative_tolerance=1e-9,
        gravity=UniformGravity(gravity))
    before=spec.field.as_dict()
    observed=integrate_test_charge(spec)
    assert observed.termination=="duration"
    assert observed.samples[0].time_s==0.
    assert observed.samples[-1].time_s==spec.duration_s

    # Real 3D circular midpoint samples, fixed before reference integration.
    independent=(
        _sites(+3e-9,.16,(.05,0.,-.07),(0.,0.,1.)),
        _sites(-1.5e-9,.11,(-.1,0.,.12),(1.,0.,0.)),
    )
    q=particle.charge_c
    def acceleration(p):
        f=_direct_force(p,independent,q,mass,gravity)
        return tuple(a/mass for a in f)
    first_force=_direct_force(particle.position_m,independent,q,mass,gravity)
    first_force_error=max(abs(x-y) for x,y in zip(first_force,observed.samples[0].force_n))
    assert all(abs(x-y)<=max(1e-12,5e-8*abs(x))
               for x,y in zip(first_force,observed.samples[0].force_n))

    dt=spec.duration_s/700
    p=list(particle.position_m)
    v=list(spec.initial_velocity_m_s)
    a=acceleration(p)
    for _ in range(700):
        pp=[p[i]+v[i]*dt+.5*a[i]*dt*dt for i in range(3)]
        aa=acceleration(pp)
        v=[v[i]+.5*(a[i]+aa[i])*dt for i in range(3)]
        p,a=pp,aa

    last=observed.samples[-1]
    err_p=max(abs(x-y) for x,y in zip(last.position_m,p))
    err_v=max(abs(x-y) for x,y in zip(last.velocity_m_s,v))
    assert err_p<3e-6,err_p
    assert err_v<2e-5,err_v
    assert all(abs(s.position_m[1])<1e-10 and
               abs(s.velocity_m_s[1])<1e-10 and
               abs(s.force_n[1])<1e-10 for s in observed.samples)
    assert observed.max_abs_energy_drift_j<5e-12
    assert before==spec.field.as_dict()

    return dict(schema="conductor_sim.m27b_reserved_first_run.v1",
        case="predeclared, author-exposed noncoaxial full-ring transfer",
        provenance="Teacher hypothesis, not a sourced exam problem or blinded model trial",
        sources=[dict(source_id=id,component=s.as_dict()) for id,s in zip(
            ("ring:alpha","ring:beta","field:held"),sources)],
        physical_motion_plane_normal=[0,1,0],
        initial_position_m=particle.position_m,
        mass_kg=mass,charge_c=q,gravity_m_s2=gravity,
        source_guard_m=spec.source_guard_m,
        termination=observed.termination,
        true_final_position_m=last.position_m,
        true_final_velocity_m_s=last.velocity_m_s,
        independent_verlet_position_m=p,
        independent_verlet_velocity_m_s=v,
        position_absolute_component_error_m=err_p,
        velocity_absolute_component_error_m_s=err_v,
        first_force_absolute_component_error_n=first_force_error,
        max_abs_energy_drift_j=observed.max_abs_energy_drift_j,
        sample_times_s=[s.time_s for s in observed.samples],
        result="pass")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("output/m27b-reserved.json"))
    args=parser.parse_args()
    out=run_reserved()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({k:out[k] for k in (
        "result","termination","position_absolute_component_error_m",
        "velocity_absolute_component_error_m_s","max_abs_energy_drift_j")},sort_keys=True))


if __name__=="__main__":
    main()
