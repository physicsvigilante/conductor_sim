"""Developer source-anchored arithmetic for original workbook Q8, Q19, Q27.

Reads the frozen workbook and reuses released physical APIs and source-specific
M36b/M36c replay+independent references. It does NOT invent source givens,
claim a new independent JEE12 verdict, simulate an unconstrained electron path
for the static work question, or invent wire-current chronology for Q27.
"""
from __future__ import annotations

from hashlib import sha256
from math import isclose, pi, sqrt
from pathlib import Path
import json
from tempfile import TemporaryDirectory

from conductor_sim import PrescribedField, UniformElectricField
from conductor_sim.jee_potential_explorer import endpoint_field_work
from tools.m36b_jee_replay import run_all as m36b_replay
from tools.m36b_jee_reference import check_native as m36b_reference
from tools.m36c_jee_conductor_replay import _connected
from tools.m36c_jee_conductor_reference import connection

WORKBOOK_SHA = "9dff3c02a9f1848d2c3d99cb1a6e7a44dc7f5a51f7a3a2b8cc92d10ce1fbca5d"
EPSILON0 = 8.8541878128e-12
ELEMENTARY_CHARGE = 1.602176634e-19
K = 1 / (4*pi*EPSILON0)


def check(root:Path)->dict:
    original=root/"02_conductors_half_size.pdf"
    if sha256(original.read_bytes()).hexdigest()!=WORKBOOK_SHA:
        raise AssertionError("original workbook frozen bytes changed")
    historical=json.loads((root/"docs/m36a_jee30_coverage.json").read_text(encoding="utf-8"))
    by={x["id"]:x for x in historical["cases"]}
    expected=(("p01-single-08",1,"Single Correct Q8"),
              ("p02-single-19",2,"Single Correct Q19"),
              ("p03-single-27",3,"Single Correct Q27"))
    for ident,page,anchor in expected:
        case=by[ident]
        if (case["source"]!=original.name or case["pdf_page"]!=page
                or case["printed_anchor"]!=anchor):
            raise AssertionError("original printed workbook Q identity changed")

    # p1 Q8: E=10 V/m to the right. AB=5m and BC=3m gives +4m
    # rightward A -> B. External work on the NEGATIVE electron is +40eV.
    # This is controlled static displacement, never a free electron trajectory.
    dx=sqrt(5.**2-3.**2)
    field=PrescribedField((UniformElectricField((10.,0.,0.)),))
    work=endpoint_field_work(
        field,source_annotations=(("sheet","source-given uniform +x electric field"),),
        probe_charge_c=-ELEMENTARY_CHARGE,
        start_m=(0.,0.,0.),end_m=(dx,3.,0.),
    )
    external_j=work["external_quasistatic_work_j"]
    if not isclose(external_j,40*ELEMENTARY_CHARGE,rel_tol=1e-13,abs_tol=1e-28):
        raise AssertionError("real source p1 electron external work sign or magnitude wrong")
    if not isclose(work["field_work_j"],-40*ELEMENTARY_CHARGE,rel_tol=1e-13,abs_tol=1e-28):
        raise AssertionError("original p1 work by E vs work by us conflated")

    # p2 Q19: ONE-SIDED ideal metal field |sigma|/eps0, not the
    # TWO-SIDED insulating sheet field |sigma|/(2*eps0). E=100eV stopping
    # work is 100 volts per elementary charge. Textbook option 0.44mm.
    sigma=-2e-6
    Eabs=abs(sigma)/EPSILON0
    stop_m=100./Eabs
    if not isclose(stop_m,.00044270939064,rel_tol=3e-14):
        raise AssertionError("original ideal metal-sheet stopping distance wrong")
    with TemporaryDirectory(prefix="jee12_source_replays_") as tmp:
        folder=Path(tmp)
        m36b=m36b_replay(folder/"m36b")
        ref=m36b_reference(m36b)
        card=m36b["source_apparatus"]["p02-single-19"]
        independent_stopping=ref["independently_checked_cases"]["p02-single-19"]["stopping_displacement_m"]
        if not isclose(independent_stopping,stop_m,rel_tol=1e-11,abs_tol=1e-16):
            raise AssertionError("source-specific M36b real event and independent physics disagree")
        if not (card["termination"]=="terminal-event" and
                card["kind"]=="actual_elapsed_time_trajectory_and_event"):
            raise AssertionError("p2 Q19 needs a genuine resolved stopping event")
        if "sheet" not in card["viewer_file"]:
            raise AssertionError("real electron stopping viewer absent")

        # p3 Q27: genuine STATIC before/after connected conductors.
        # Source q=20uC; inner solid radius=.1m; outer metal shell radius=.2m.
        # Printed mathematical zero-thickness energy is k*q^2/2*(1/.1-1/.2)
        # approx 9 J (multiple choice). The native model has a disclosed
        # 1 micrometre finite shell thickness and gives slightly different SI.
        m36c=_connected("p03-single-27","pdf-p03-q27",folder/"m36c")
        verified=connection(m36c,"p03-single-27")
        original_ideal_release=.5*K*(20e-6)**2*(1/.1-1/.2)
        actual_release=m36c["released_energy_j"]
        if not isclose(actual_release,verified["independent_gap_energy_release_j"],
                       rel_tol=3e-12,abs_tol=1e-10):
            raise AssertionError("real M36c native field-energy gap release wrong")
        if not (abs(actual_release-9.)<.02 and abs(original_ideal_release-9.)<.02):
            raise AssertionError("source multiple-choice 9J is inconsistent with physical model")
        if m36c["kind"]!="two_physical_static_equilibria_not_wire_current_time":
            raise AssertionError("invented discharge-time/current in printed p3 Q27")

    return dict(
        schema="conductor_sim.jee12_workbook_numeric_developer_probe.v1",
        status="DEVELOPER SOURCE-PRINTED ANSWER CHECK, NOT NEW INDEPENDENT JEE12 VERDICT",
        frozen_pdf_sha256=WORKBOOK_SHA,
        p01_single_08=dict(printed_page=1,printed_anchor="Single Correct Q8",
            source_given_E_v_m=10., source_given_AB_m=5.,source_given_BC_m=3.,
            derived_horizontal_delta_m=dx, electron_charge_c=-ELEMENTARY_CHARGE,
            original_asked="work ON electron BY US",native_external_work_j=external_j,
            independently_expected_external_work_eV=40.,
            real_field_work_eV=work["field_work_j"]/ELEMENTARY_CHARGE,
            original_multiple_choice_40eV=True,
            physical_path_trajectory_or_time_claimed=False),
        p02_single_19=dict(printed_page=2,printed_anchor="Single Correct Q19",
            source_given_initial_energy_eV=100.,source_given_metal_sigma_c_per_m2=sigma,
            one_sided_metal_field_v_m=Eabs, independent_stopping_distance_m=stop_m,
            released_M36b_native_reference_stopping_m=independent_stopping,
            original_multiple_choice_option="0.44 mm", teacher_initial_clearance_m=.001,
            native_real_electron_stopping_event=True,
            additional_early_contact_variant_is_teacher_authored=True),
        p03_single_27=dict(printed_page=3,printed_anchor="Single Correct Q27",
            source_given_inner_solid_radius_m=.1,
            source_given_outer_shell_radius_m=.2,source_given_charge_c=20e-6,
            ideal_zero_thickness_release_j=original_ideal_release,
            native_finite_metal_release_j=actual_release,
            source_multiple_choice_option="9 J",
            finite_shell_thickness_is_teacher_authored=True,
            no_physical_wire_current_or_dissipation_time=True),
        original_numeric_source_anchors_reinspected_by_developer=3,
        original_answers_newly_independently_reviewed=0,
        real_browser_validation_for_jee12_not_claimed=True,
    )


if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path)
    args=ap.parse_args()
    result=check(Path(__file__).resolve().parents[1])
    output=json.dumps(result,indent=2,allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(output,encoding="utf-8")
    else:
        print(output)
