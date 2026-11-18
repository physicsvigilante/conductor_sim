"""M36b reviewer-exposed *teacher* variations executed by genuine released APIs.

These are *not* new original JEE source statements and do not change the
seven-apparatus pilot denominator. Independent original review still required.
The numerical targets are separately derived simple Newton/Coulomb equations;
no production source callbacks are used to derive the reference numbers.
"""
from __future__ import annotations

from dataclasses import replace
import argparse
import json
from math import cos, hypot, pi, sin, sqrt
from pathlib import Path

from conductor_sim import EPSILON_0, PointCharge, PrescribedField, UniformElectricField
from conductor_sim.conservative_forces import UniformGravity
from conductor_sim.dipoles import dipole_work, observe_dipole, orient_dipole
from conductor_sim.motion_events import EventStudy, investigate_motion
from conductor_sim.scenarios import PrescribedScenario
from conductor_sim.prescribed_motion import integrate_test_charge
from examples.motion_event_studies import (
    ELECTRON_M, ELECTRON_Q, make_case as sheet_case,
)
from examples.electric_gravity_motion import build_case as gravity_case
from examples.rigid_dipoles import make_model as dipole_case


def close(a: float, b: float, *, tolerance: float, label: str) -> None:
    if abs(a - b) > tolerance:
        raise AssertionError(f"{label}: independently expected {b}, actual {a}")


def transfer_sheet() -> dict:
    original=sheet_case("sheet-stop")
    spec=original.scenario.spec
    kinetic_eV,sigma,launch=175.,-3.4e-6,.0011
    v0=-sqrt(2*kinetic_eV*ELECTRON_Q/ELECTRON_M)
    modified=replace(spec,
        field=PrescribedField((UniformElectricField((sigma/EPSILON_0,0.,0.)),)),
        particle=PointCharge(-ELECTRON_Q,(launch,0.,0.)),
        initial_velocity_m_s=(v0,0.,0.))
    scenario=PrescribedScenario(
        "m36b-review-sheet-175eV",modified,original.scenario.source_annotations,
        "Teacher-hypothetical source-preserving variation: 175 eV, metal sigma=-3.4e-6 C/m2, initial clearance 1.1mm.",
    )
    run=investigate_motion(EventStudy(scenario,original.questions,original.policy))
    if len(run.events)!=1 or run.events[0].event_id!="stop:x":
        raise AssertionError("teacher source variation did not locate native electron reversal")
    e=run.events[0]
    # Independently Newton's equation for a nonresponding ideal conductor sheet.
    a=(-ELECTRON_Q)*(sigma/EPSILON_0)/ELECTRON_M
    t=-v0/a
    displacement=v0*v0/(2*a)
    close(e.sample.time_s,t,tolerance=2e-12,label="changed one-sided-sheet reversal time")
    close(e.sample.position_m[0],launch-displacement,tolerance=3e-12,
          label="changed source electron reversal world position")
    close(e.sample.velocity_m_s[0],0.,tolerance=1e-3,label="changed source reversal velocity")
    if run.trajectory.termination!="terminal-event":
        raise AssertionError("teacher sheet native event must be terminal")
    return dict(
        kind="teacher-hypothetical native full-vector electron event; NOT source case #8",
        kinetic_eV=kinetic_eV,sigma_c_m2=sigma,launch_m=launch,
        native_event_t_s=e.sample.time_s,native_event_position_m=list(e.sample.position_m),
        native_termination=run.trajectory.termination,
        independently_derived_t_s=t,independently_derived_reversal_displacement_m=displacement,
        reference_verified=True,
    )


def transfer_dipole() -> dict:
    original,_=dipole_case("workbook-q47")
    angle=65*pi/180
    q,length,E=1.2e-9,.06,200.
    modified=replace(
        original,
        dipole=replace(original.dipole,charge_c=q,separation_m=length),
        sources=(replace(original.sources[0],
                         source=UniformElectricField((E,0.,0.))),),
        model_id="m36b-review-dipole-65",
        provenance="Teacher-hypothetical 1.2nC x 6cm in 200 V/m, 65deg static pose; no time rotation or inferred inertia.",
    )
    obs=observe_dipole(orient_dipole(modified,angle))
    work=dipole_work(modified,0.,angle)
    expected_delta_u=q*length*E*(1-cos(angle))
    expected_torque=-q*length*E*sin(angle)
    close(work["external_quasistatic_work_j"],expected_delta_u,
          tolerance=5e-20,label="teacher changed dipole work")
    close(obs["torque_n_m"][2],expected_torque,
          tolerance=5e-20,label="teacher changed dipole torque")
    return dict(
        kind="teacher-hypothetical static finite-dipole observer; NOT rotational dynamics",
        charge_c=q,separation_m=length,field_v_m=E,angle_deg=65,
        native_work_j=work["external_quasistatic_work_j"],
        native_torque_n_m=list(obs["torque_n_m"]),
        independent_delta_u_j=expected_delta_u,independent_torque_z_n_m=expected_torque,
        reference_verified=True,
    )


def transfer_proton() -> dict:
    source,_,_=gravity_case("jee2012-proton")
    m,q,E,g=2e-27,1.4e-19,1.04375e-7,9.81
    end=.017
    changed=replace(
        source.spec,
        mass_kg=m,
        particle=replace(source.spec.particle,charge_c=q),
        field=PrescribedField((UniformElectricField((E,0.,0.)),)),
        gravity=UniformGravity((0.,-g,0.)),
        duration_s=end,
        sample_interval_s=.001,
    )
    scenario=PrescribedScenario(
        "m36b-review-proton-oblique",changed,source.source_annotations,
        "Teacher-hypothetical changed proton mass/charge/E/g and 17ms preview; not IIT-JEE 2012 original values.",
    )
    native=integrate_test_charge(scenario.spec)
    ax=q*E/m
    if len(native.samples)<15:
        raise AssertionError("actual new real elapsed-time native samples missing")
    for index,row in enumerate(native.samples):
        t=row.time_s
        expected_position=(ax*t*t/2,-g*t*t/2,0.)
        expected_velocity=(ax*t,-g*t,0.)
        for i in range(3):
            close(row.position_m[i],expected_position[i],tolerance=3e-14,
                  label=f"teacher proton real position [{index},{i}]")
            close(row.velocity_m_s[i],expected_velocity[i],tolerance=2e-12,
                  label=f"teacher proton real velocity [{index},{i}]")
    final=native.samples[-1]
    return dict(
        kind="teacher-hypothetical genuine vector E+g time evolution; NOT source IIT-JEE 2012 answer",
        mass_kg=m,charge_c=q,electric_field_v_m=E,gravity_m_s2=g,
        duration_s=end,real_sample_count=len(native.samples),
        native_end_position_m=list(final.position_m),
        native_end_velocity_m_s=list(final.velocity_m_s),
        independent_end_position_m=[ax*end*end/2,-g*end*end/2,0.],
        independent_end_velocity_m_s=[ax*end,-g*end,0.],
        reference_verified=True,
    )


def run_transfers() -> dict:
    return {
        "schema":"conductor_sim.m36b_teacher_transfer.v1",
        "original_JEE_case_denominator_increment":0,
        "source_original_values_changed":False,
        "teacher_variations":{
            "charged_sheet":transfer_sheet(),
            "rigid_dipole":transfer_dipole(),
            "electric_plus_gravity":transfer_proton(),
        },
        "reviewer_independence":"developer-executed teacher variations inspired by independent review; requires independent reviewer verification",
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    report=run_transfers()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("M36b actual teacher transfer families:",len(report["teacher_variations"]))
    print("Additional original JEE cases certified:",report["original_JEE_case_denominator_increment"])


if __name__=="__main__":
    main()
