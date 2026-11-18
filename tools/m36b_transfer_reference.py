"""M36b transfer-variation independent SI check; ZERO production/example imports.

Checks actually saved native event/static/trajectory outputs from three NEW
teacher-authored cases, separately from the seven original source apparatus.
"""
from __future__ import annotations

import argparse
import json
from math import cos, isclose, pi, sin, sqrt, isfinite
from pathlib import Path

EPSILON0=8.8541878128e-12
ELECTRON_Q=1.602176634e-19
ELECTRON_M=9.1093837139e-31


def assert_near(a,b,*,atol=0.,rtol=1e-10,label=""):
    if type(a) not in (int,float) or not isfinite(a) or not isclose(a,b,rel_tol=rtol,abs_tol=atol):
        raise AssertionError(f"{label}: saved native {a!r} vs independent {b!r}")


def verify(data:dict)->dict:
    if data["schema"]!="conductor_sim.m36b_teacher_transfer.v1":
        raise AssertionError("wrong transfer source schema")
    if data["original_JEE_case_denominator_increment"]!=0 or data["source_original_values_changed"] is not False:
        raise AssertionError("teacher variation cannot rewrite source-given case count")
    a=data["teacher_variations"]
    if set(a)!={"charged_sheet","rigid_dipole","electric_plus_gravity"}:
        raise AssertionError("three distinct physical families required")

    sheet=a["charged_sheet"]
    if "teacher-hypothetical" not in sheet["kind"] or sheet["native_termination"]!="terminal-event":
        raise AssertionError("no genuine source-preserving changed sheet event")
    assert_near(sheet["kinetic_eV"],175,label="sheet reviewer variation source")
    assert_near(sheet["sigma_c_m2"],-3.4e-6,label="metal sigma")
    assert_near(sheet["launch_m"],.0011,label="teacher launch clearance")
    speed=sqrt(2*175*ELECTRON_Q/ELECTRON_M)
    acceleration=ELECTRON_Q*3.4e-6/(EPSILON0*ELECTRON_M)
    t=speed/acceleration
    ds=speed*speed/(2*acceleration)
    assert_near(sheet["native_event_t_s"],t,atol=2e-12,rtol=0,label="sheet native event root")
    assert_near(sheet["native_event_position_m"][0],.0011-ds,
                atol=3e-12,rtol=0,label="sheet native turn location")
    for v in sheet["native_event_position_m"][1:]:
        assert_near(v,0,atol=1e-12,rtol=0,label="sheet invariant transverse plane")

    dipole=a["rigid_dipole"]
    if "NOT rotational dynamics" not in dipole["kind"]:
        raise AssertionError("dipole static observer mislabeled as dynamics")
    for name,expected in [("charge_c",1.2e-9),("separation_m",.06),
                          ("field_v_m",200.),("angle_deg",65)]:
        assert_near(dipole[name],expected,label="dipole teacher parameter "+name)
    angle=65*pi/180.
    delta=1.2e-9*.06*200*(1-cos(angle))
    torque=-1.2e-9*.06*200*sin(angle)
    assert_near(dipole["native_work_j"],delta,atol=1e-19,label="native static dipole ΔU")
    assert_near(dipole["native_torque_n_m"][2],torque,atol=1e-19,label="native finite dipole torque")
    assert_near(dipole["independent_delta_u_j"],delta,atol=1e-19,label="independent ΔU")
    assert_near(dipole["independent_torque_z_n_m"],torque,atol=1e-19,label="independent torque")

    proton=a["electric_plus_gravity"]
    if "teacher-hypothetical" not in proton["kind"] or proton["real_sample_count"]<15:
        raise AssertionError("new native time trajectory not supplied")
    for name,expected in [
        ("mass_kg",2e-27),("charge_c",1.4e-19),
        ("electric_field_v_m",1.04375e-7),("gravity_m_s2",9.81),
        ("duration_s",.017),
    ]:
        assert_near(proton[name],expected,label="proton new physical input "+name)
    acc=1.4e-19*1.04375e-7/2e-27
    t=.017
    p=(.5*acc*t*t,-.5*9.81*t*t,0.)
    v=(acc*t,-9.81*t,0.)
    for i in range(3):
        assert_near(proton["native_end_position_m"][i],p[i],
                    atol=3e-14,rtol=1e-12,label=f"real native proton position {i}")
        assert_near(proton["native_end_velocity_m_s"][i],v[i],
                    atol=2e-12,rtol=1e-12,label=f"real native proton velocity {i}")
    return {
        "schema":"conductor_sim.m36b_teacher_transfer_reference.v1",
        "genuinely_recomputed_teacher_physical_families":3,
        "original_source_apparatus_count_added":0,
        "sheet_t_s":speed/acceleration,
        "dipole_delta_u_j":delta,
        "proton_end_position_m":list(p),
        "meaning":"Changed teacher hypotheticals not additional original-JEE problem passes",
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    data=json.loads(args.input.read_text(encoding="utf-8"))
    result=verify(data)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("M36b independently referenced changed teacher-native families:",
          result["genuinely_recomputed_teacher_physical_families"])


if __name__=="__main__":
    main()
