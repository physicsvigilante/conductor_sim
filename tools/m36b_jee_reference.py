"""M36b: standard-library-only, independently expressed JEE source physics.

This module intentionally imports NO conductor_sim/physics production symbols,
no JEE example modules, no numeric solver and no browser. It checks the ACTUAL
saved native M36b source-run observations against separately derived equations
and the preregistered original givens and teacher assumptions.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Fixed external SI reference literals; never read production epsilon to
# compute our own expected electric field/potential.
EPSILON0 = 8.8541878128e-12
ELECTRON_Q = 1.602176634e-19
ELECTRON_M = 9.1093837139e-31
K = 1.0 / (4.0 * math.pi * EPSILON0)
EXPECTED = frozenset({
    "p02-single-19", "p03-single-30", "p05-single-47", "p12-linked-40",
    "p15-archive-02", "p02-single-15",
    "jee2021-p1-q09-q10-equipotential",
})


def demand(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def near(actual: float, expected: float, *, rel: float = 1e-9,
         abs_tol: float = 0.0, label: str = "") -> None:
    demand(type(actual) in (float, int) and math.isfinite(actual), "nonfinite/non-numeric " + label)
    demand(math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_tol),
           f"{label} expected {expected:.16g}, actual {actual!r}")


def near3(actual: list, expected: tuple, *, rel: float = 1e-9,
          abs_tol: float = 1e-12, label: str = "") -> None:
    demand(isinstance(actual, list) and len(actual) == 3, "missing SI 3-vector " + label)
    for i, (a, b) in enumerate(zip(actual, expected)):
        near(a, b, rel=rel, abs_tol=abs_tol, label=f"{label}[{i}]")


def reference_sheet(card: dict) -> dict:
    # Original p2 Q19 fixes 100 electron-volt kinetic energy and metal
    # surface sigma=-2e-6 C/m2. Teacher chooses a 1mm initial clearance.
    demand(card["kind"] == "actual_elapsed_time_trajectory_and_event", "sheet not native event")
    demand(card["native"] is True and card["case_id"] == "p02-single-19", "sheet identity")
    demand(card["source_given"] == {
        "kinetic_energy_eV": 100., "surface_sigma_c_m2": -2e-6}, "sheet source facts")
    near(card["teacher_launch_distance_m"], .001, label="sheet initial teacher clearance")
    field_x = -2e-6 / EPSILON0  # conductor one-sided, not insulating half-field!
    near3(card["source_field_v_m"], (field_x,0,0), rel=1e-11, label="metal E")
    near(card["particle_charge_c"], -ELECTRON_Q, rel=1e-13, label="electron signed q")
    near(card["mass_kg"], ELECTRON_M, rel=1e-13, label="electron m")
    v0 = -math.sqrt(2*100*ELECTRON_Q/ELECTRON_M)
    acceleration = ELECTRON_Q * (2e-6/EPSILON0)/ELECTRON_M
    t_stop = -v0/acceleration
    ds = v0*v0/(2*acceleration)
    near3(card["initial_position_m"], (.001,0,0), abs_tol=0, label="sheet start")
    near3(card["initial_velocity_m_s"], (v0,0,0), rel=2e-12,abs_tol=1e-14,label="source launch")
    demand(len(card["events"]) == 1, "sheet exactly one resolved reversal")
    stop = card["events"][0]
    demand(stop["id"] == "stop:x" and stop["full_stop"] is True, "real complete stop event missing")
    near(stop["time_s"], t_stop, rel=0, abs_tol=2e-12, label="native reversal time")
    near3(stop["position_m"], (.001-ds,0.,0.), rel=0, abs_tol=3e-12,
          label="real stopped electron position vector")
    near3(stop["velocity_m_s"], (0.,0.,0.), rel=0, abs_tol=1e-3,
          label="stopped electron full velocity vector")
    # Source and launch have no transverse force/velocity. A nominal full
    # stop cannot secretly carry vy/vz, even below the x-reversal tolerance.
    near(stop["velocity_m_s"][1],0.,rel=0,abs_tol=1e-9,label="no transverse vy at full stop")
    near(stop["velocity_m_s"][2],0.,rel=0,abs_tol=1e-9,label="no transverse vz at full stop")
    speed_sq=sum(v*v for v in stop["velocity_m_s"])
    demand(speed_sq <= (1e-3)**2 * 1.05,
           "full_stop=True contradicts the magnitude of the actual velocity vector")
    expected_K=0.5*ELECTRON_M*speed_sq
    near(stop["kinetic_energy_j"],expected_K,rel=1e-7,abs_tol=1e-42,
         label="actual electron stopping kinetic energy from full native velocity")
    demand(stop["kinetic_energy_j"] <= .5*ELECTRON_M*(1e-3)**2*1.05,
           "stopped electron kinetic energy exceeds event velocity tolerance")
    demand(card["termination"] == "terminal-event", "sheet observation must end at stop")
    early = card["early_surface_variant"]
    near(early["teacher_launch_distance_m"], .0002, label="early teacher clearance")
    demand(early["event_id"] == "sheet" and early["termination"] == "terminal-event",
           "early surface encounter did not censor before reversal")
    near3(early["initial_position_m"],(.0002,0.,0.),rel=0,abs_tol=1e-15,
          label="early variant genuinely revised launch position")
    near3(early["initial_velocity_m_s"],(v0,0.,0.),rel=2e-12,abs_tol=1e-14,
          label="early variant same source electron initial velocity")
    near3(early["source_field_v_m"],(field_x,0.,0.),rel=1e-11,
          label="early variant same physical one-sided metal E")
    near(early["mass_kg"],ELECTRON_M,rel=1e-13,label="early electron mass")
    near(early["particle_charge_c"],-ELECTRON_Q,rel=1e-13,label="early electron charge")
    d_early=.0002
    discr=(-v0)**2 - 2*acceleration*d_early
    demand(discr>0,"early surface is not reached before physical reversal")
    # Stable first positive quadratic root, rather than merely t < t_stop.
    first_root=2*d_early/(-v0+math.sqrt(discr))
    near(early["event_time_s"], first_root, rel=0, abs_tol=2e-14,
         label="independently computed FIRST metal-surface root time")
    demand(0 < early["event_time_s"] < t_stop,
           "early terminal surface must chronologically preempt reversal")
    contact=early["actual_native_contact_event"]
    demand(contact["id"]=="sheet" and contact["full_stop"] is False,
           "early terminal event must be a surface, not an invented full rest")
    near(contact["time_s"], early["event_time_s"], rel=0, abs_tol=1e-22,
         label="early event time mirrors true native sample")
    near3(contact["position_m"],(0.,0.,0.),rel=0,abs_tol=2e-9,
          label="real first surface root world position")
    contact_v=v0+acceleration*first_root
    near3(contact["velocity_m_s"],(contact_v,0.,0.),rel=2e-8,abs_tol=.1,
          label="native impact-threshold velocity, NOT bounce/contact physics")
    near(contact["kinetic_energy_j"],.5*ELECTRON_M*contact_v**2,
         rel=3e-8,abs_tol=1e-23,label="early event physical kinetic energy")
    return {"time_s": t_stop, "stopping_displacement_m": ds,
            "early_surface_first_root_time_s": first_root,
            "early_surface_event_checked": True}


def reference_plate(card: dict) -> dict:
    demand(card["case_id"] == "p03-single-30" and card["native"] is True, "plate native")
    demand(card["kind"] == "actual_elapsed_time_trajectory_and_event", "plate elapsed motion")
    d,V=.01,100.
    near(card["teacher_gap_m"], d,label="teacher plate d")
    near(card["teacher_potential_difference_v"], V,label="teacher plate V")
    near3(card["source_field_v_m"], (0,0,-V/d),label="imposed uniform gap")
    near3(card["initial_position_m"], (0,0,0),label="plate launch")
    near3(card["initial_velocity_m_s"], (0,0,0),label="plate rest")
    near(card["particle_charge_c"], -ELECTRON_Q,label="plate electron q")
    near(card["mass_kg"], ELECTRON_M,label="plate electron mass")
    near_expected=math.sqrt(2*ELECTRON_M*d*d/(ELECTRON_Q*V))
    demand(len(card["events"]) == 1 and card["events"][0]["id"]=="upper",
           "upper terminal real event absent")
    arrival=card["events"][0]
    near(arrival["time_s"], near_expected,rel=0,abs_tol=3e-17,label="arrival elapsed t")
    near3(arrival["position_m"], (0,0,d),abs_tol=2e-9,label="upper plate terminal location")
    near(arrival["kinetic_energy_j"], ELECTRON_Q*V,rel=1e-7,abs_tol=1e-23,label="arrival K")
    demand(card["termination"] == "terminal-event", "no contact continuation")
    demand(card["initial_conditions"].get("lower")=="initial-on-plane",
           "original launch-plane initial condition lost")
    return {"arrival_time_s":near_expected,"arrived_at_source_upper_plate":True}


def reference_dipole(card: dict) -> dict:
    demand(card["case_id"]=="p05-single-47" and card["native"] is True,"dipole source identity")
    demand(card["kind"]=="static_orientation_observations_not_time_evolution",
           "static dipole work misrepresented as dynamics")
    near(card["teacher_charge_c"],1e-9,label="illustrative dipole charge")
    near(card["teacher_separation_m"],.04,label="illustrative separation")
    near3(card["teacher_field_v_m"], (120.,0.,0.),label="illustrative uniform E")
    demand(card["illustrative_angles_deg"]==[40,90,115,180],"qualitative poses altered")
    changes=card["observed_delta_u_j"]
    demand(len(changes)==4,"four qualitative poses required")
    pE=1e-9*.04*120
    for i,angle in enumerate((40,90,115,180)):
        expected=pE*(1-math.cos(math.radians(angle)))
        near(changes[i],expected,rel=2e-12,abs_tol=1e-20,label=f"dipole ΔU({angle})")
    demand(all(0 < x < y for x,y in zip(changes,changes[1:])), "dipole energy ordering")
    demand("NO numerical angles" in card["provenance"], "teacher/source distinctions missing")
    return {"orientation_energy_order":"acute < perpendicular < obtuse < antiparallel",
            "static_not_rotation":True}


def reference_ring(card: dict) -> dict:
    demand(card["case_id"]=="p12-linked-40" and card["native"] is True, "ring source identity")
    demand(card["kind"]=="static_endpoint_work_not_time_evolution", "invented ring motion")
    anchor=card["source_anchor"]
    demand(anchor["pdf_page"]==12 and anchor["original_mass"].startswith("symbolic"),
           "ring source anchor or symbolic mass was substituted")
    near3(anchor["original_external_field_v_m"],(0,1e5,0),label="upward external E")
    q=+10e-6
    Q1=+10e-6
    Q2=-20e-6
    R=.3
    d=.4
    r_far=math.hypot(R,d)
    expected_work=K*q*(Q1*(1/R-1/r_far)+Q2*(1/r_far-1/R))
    near(card["observed_field_work_j"],expected_work,rel=2e-12,label="two-ring independent axial work")
    near(card["observed_field_work_j"],3.5950207169,rel=2e-9,label="ring joules")
    # Independently compute the FULL vector field at each source centre,
    # including the signed axial field from the opposite charged ring.
    # Reusing the recorded Ex as the expected Ex would accept forged fields.
    axial_at_a=K*Q2*(-d)/(r_far**3)
    axial_at_b=K*Q1*(d)/(r_far**3)
    near3(card["start_electric_field_v_m"],(axial_at_a,1e5,0.),
          rel=2e-12,abs_tol=1e-8,label="independent E vector at ring centre A")
    near3(card["end_electric_field_v_m"],(axial_at_b,1e5,0.),
          rel=2e-12,abs_tol=1e-8,label="independent E vector at ring centre B")
    near(card["transverse_profile_max_abs_v_m"],1e5,abs_tol=1e-7,label="nonzero transverse force")
    parts=card["observed_source_work_components"]
    demand(len(parts)==3 and [p["id"] for p in parts]==["ring:A","ring:B","field:vertical"],
           "lost ring and applied-field source identity")
    # Verify each source's own signed potential-work contribution,
    # not just the total and a conveniently unchanged transverse component.
    expected_a=q*K*Q1*(1/R-1/r_far)
    expected_b=q*K*Q2*(1/r_far-1/R)
    for part,expected,source in zip(parts,(expected_a,expected_b,0.),
                                    ("ring A","ring B","uniform upward field")):
        near(part["field_work_j"],expected,rel=2e-12,abs_tol=1e-13,
             label="independent signed work from "+source)
    near(sum(part["field_work_j"] for part in parts),expected_work,
         rel=2e-12,abs_tol=1e-12,label="source-resolved work sums to total")
    return {"work_j":expected_work,"ring_a_work_j":expected_a,
            "ring_b_work_j":expected_b,"axial_field_a_v_m":axial_at_a,
            "axial_field_b_v_m":axial_at_b,
            "transverse_force_n":q*1e5,"elapsed_time_motion_claimed":False}


def reference_proton(card: dict) -> dict:
    demand(card["case_id"]=="p15-archive-02" and card["native"] is True,"2012 source identity")
    demand(card["kind"]=="actual_elapsed_time_trajectory","proton not real motion")
    demand(card["source_anchor"]["pdf_page"]==15,"incorrect workbook source page")
    m,q,g,d=1.67e-27,1.6e-19,10.,.01
    E=m*g/q
    near(card["source_anchor"]["voltage_v"],E*d,rel=1e-13,label="source-derived voltage")
    near(card["source_anchor"]["plate_gap_m"],d,label="proton actual gap")
    near3(card["source_electric_field_v_m"],(E,0,0),rel=2e-13,abs_tol=1e-40,label="horizontal E")
    near3(card["source_gravity_m_s2"],(0,-g,0),label="vertical g")
    near(card["mass_kg"],m,label="proton mass")
    near(card["charge_c"],q,label="proton charge")
    near3(card["initial_velocity_m_s"],(0,0,0),label="rest release")
    demand(card["termination"]=="duration","proton preview before plates")
    samples=card["saved_samples"]
    demand(len(samples)>=20,"actual saved motion samples required")
    for i, row in enumerate(samples):
        t=row["time_s"]
        expected_p=(.5*g*t*t,-.5*g*t*t,0.)
        expected_v=(g*t,-g*t,0.)
        near3(row["position_m"],expected_p,rel=0,abs_tol=3e-15,label=f"p[{i}]")
        near3(row["velocity_m_s"],expected_v,rel=0,abs_tol=3e-13,label=f"v[{i}]")
        near3(row["electric_force_n"],(m*g,0,0),abs_tol=1e-42,label=f"Fe[{i}]")
        near3(row["gravitational_force_n"],(0,-m*g,0),abs_tol=1e-42,label=f"Fg[{i}]")
        near(row["kinetic_energy_j"],m*g*g*t*t,rel=3e-11,abs_tol=1e-45,label=f"K[{i}]")
        demand(abs(row["position_m"][0])<d/2,"motion illegally crosses plate preview")
    return {"voltage_v":E*d,"saved_real_timestep_rows":len(samples)}


def reference_distant(card: dict) -> dict:
    demand(card["case_id"]=="p02-single-15" and card["native"] is True,"distant-sphere source")
    demand(card["kind"]=="approximate_static_before_and_after_equilibria",
           "far-separated static sharing presented as dynamic")
    qualification=card["model_qualification"]
    demand(qualification["is_relative_error_bound"] is False,
           "5% geometry admission heuristically mislabeled physical error guarantee")
    init,fin=card["before"],card["after"]
    demand(len(init)==len(fin)==2,"two connected spheres required")
    for c0,c1 in zip(init,fin):
        demand(c0["name"]==c1["name"],"source identities changed")
    radii=[c["radius_m"] for c in init]
    charges=[c["charge_c"] for c in init]
    near(radii[0],.1,label="illustrative R1")
    near(radii[1],.2,label="illustrative R2")
    near(charges[0],2e-9,label="illustrative initial Q1")
    near(charges[1],4e-9,label="illustrative initial Q2")
    total=sum(charges)
    expectation=[total*x/sum(radii) for x in radii]
    for i,c in enumerate(fin):
        near(c["charge_c"],expectation[i],rel=1e-12,abs_tol=1e-23,label=f"Qfinal[{i}]")
    near(sum(c["charge_c"] for c in fin),total,rel=1e-12,label="charge conservation")
    ratio=fin[0]["outer_field_v_m"]/fin[1]["outer_field_v_m"]
    near(ratio,radii[1]/radii[0],rel=1e-12,label="E1/E2")
    near(card["release_energy_j"],0,rel=0,abs_tol=1e-27,label="already equal potential")
    # Separate teacher variation on the SAME qualified two-sphere apparatus.
    # It starts non-equipotential: actual redistribution is -3/+3 nC.
    # It is NOT the originally given p2 Q15 initial state, or an eighth case.
    variation=card["teacher_transfer_variant"]
    demand("teacher-authored" in variation["origin"],"source/teacher sphere provenance")
    demand(variation["model_qualification"]["is_relative_error_bound"] is False,
           "teacher transfer falsely promises relative error bound")
    before,after=variation["before"],variation["after"]
    demand(len(before)==len(after)==2,"teacher transfer two source identities")
    for left,right,original in zip(before,after,init):
        demand(left["name"]==right["name"]==original["name"],
               "teacher variation unexpectedly changes physical source identity")
    for i,c in enumerate(before):
        near(c["radius_m"],radii[i],label=f"teacher variant radius {i}")
        near(c["charge_c"],(5e-9,1e-9)[i],label=f"teacher changed source charge {i}")
    teacher_q=(5e-9,1e-9)
    expected_after=[sum(teacher_q)*r/sum(radii) for r in radii]
    expect_transfer=[a-b for a,b in zip(expected_after,teacher_q)]
    demand(len(variation["charge_transfers_c"])==2,"missing signed charge-transfer ledger")
    for i,c in enumerate(after):
        near(c["radius_m"],radii[i],label=f"teacher after radius {i}")
        near(c["charge_c"],expected_after[i],rel=2e-12,abs_tol=1e-24,
             label=f"teacher true redistributing Q{i}")
        near(variation["charge_transfers_c"][i],expect_transfer[i],
             rel=2e-12,abs_tol=1e-24,label=f"independent signed Q transfer {i}")
        near(c["outer_field_v_m"],K*expected_after[i]/radii[i]**2,
             rel=2e-12,label=f"teacher final outward surface E{i}")
    near(sum(c["charge_c"] for c in after),sum(teacher_q),
         rel=2e-12,label="teacher non-equilibrium group conservation")
    self_release=.5*K*(
        sum(teacher_q[i]**2/radii[i] for i in range(2))-
        sum(expected_after[i]**2/radii[i] for i in range(2))
    )
    near(variation["released_field_energy_j"],self_release,
         rel=2e-12,abs_tol=1e-20,label="independent positive released field energy")
    demand(self_release>0,"chosen teacher variant must actually transfer charge")
    return {"surface_field_ratio":ratio,"teacher_variant_signed_transfer_c":expect_transfer,
            "teacher_variant_release_energy_j":self_release,
            "approximation_not_exact_induction":True}


def reference_circle(card: dict) -> dict:
    demand(card["case_id"]=="jee2021-p1-q09-q10-equipotential" and card["native"] is True,
           "2021 original apparatus missing")
    demand(card["kind"]=="sampled_static_two_charge_zero_potential_contour",
           "2021 incorrectly presented as trajectory")
    demand(card["has_motion"] is False, "V=0 circle must not fake particle motion")
    anchor=card["source_anchor"]
    near(anchor["illustrative_Q_c"],1e-9,label="teacher visualization Q")
    src=card["sources"]
    demand(len(src)==2,"source pair absent")
    near(src[0]["charge_c"],-1e-9,label="negative fixed charge")
    near(src[1]["charge_c"],1e-9/math.sqrt(3),rel=2e-13,label="positive fixed charge")
    near3(src[0]["position_m"],(0,0,0),label="negative charge position")
    near3(src[1]["position_m"],(2,0,0),label="positive charge position")
    near(card["actual_highlighted_level_v"],0,abs_tol=0,label="zero contour level")
    demand(card["actual_segment_count"]>100 and card["source_mask_count"]==2,
           "real sampled V=0 contour or masks missing")
    # The count alone does not authenticate source-exclusion placement.
    masks=card["source_masks"]
    demand(isinstance(masks,list) and len(masks)==2,
           "authoritative 2021 source-mask geometry missing")
    for i,(mask,source) in enumerate(zip(masks,src)):
        demand(mask["kind"]=="display-exclusion",
               "source singularities cannot masquerade as conductor boundaries")
        near3(mask["center_m"],tuple(source["position_m"]),rel=0,abs_tol=1e-12,
              label=f"source mask centre {i} physically covers charge")
        near(mask["radius_m"],.11,rel=2e-12,
             label=f"source display-only exclusion radius {i}")
    points=card["sampled_contour_world_xy_m"]
    demand(len(points)>=40,"actual contour samples missing")
    radius=math.sqrt(3.)
    worst=0.
    for x,y in points:
        r1=math.hypot(x,y)
        r2=math.hypot(x-2,y)
        # Independent algebra from potential equality, no sampling callback.
        demand(r1>0.01 and r2>0.01,"singular source appears on contour")
        analytic_v=K*(-1e-9/r1+(1e-9/math.sqrt(3))/r2)
        dist=math.hypot(x-3,y)
        worst=max(worst,abs(dist-radius))
        demand(abs(dist-radius)<.085,"actual V=0 contour not near analytic circle")
        demand(abs(analytic_v)<.12,"potential on sampled contour unphysically high")
    return {"independent_circle_center_m":[3.,0.],"radius_m":radius,
            "checked_sampled_contour_points":len(points),"maximum_radial_error_m":worst}


def check_native(report: dict) -> dict:
    demand(report["schema"]=="conductor_sim.m36b_source_replays.v1","schema mismatch")
    data=report["source_apparatus"]
    demand(set(data)==EXPECTED and report["source_apparatus_count"]==7,
           "must preserve all seven preregistered apparatus with honest denominator")
    demand(report["scientific_verification_status"]=="pending independent source-free reference checker",
           "runner should not self-certify science")
    checks={
        "p02-single-19":reference_sheet,
        "p03-single-30":reference_plate,
        "p05-single-47":reference_dipole,
        "p12-linked-40":reference_ring,
        "p15-archive-02":reference_proton,
        "p02-single-15":reference_distant,
        "jee2021-p1-q09-q10-equipotential":reference_circle,
    }
    verified={key:fn(data[key]) for key,fn in checks.items()}
    denied=report["explicit_refusal"]
    demand(report["unsupported_refusal_count"]==1 and
           denied["case_id"]=="p70-three-cavities" and denied["native"] is False and
           denied["viewer_file"] is None and "not qualified" in denied["reason"],
           "must explicitly refuse unsupported local three-cavity field")
    return {
        "schema":"conductor_sim.m36b_source_reference.v1",
        "actual_preregistered_source_apparatus_executed":7,
        "independent_math_verified_scoped_apparatus":len(verified),
        "independently_checked_cases":verified,
        "justified_local_3d_refusals_not_simulations":1,
        "original_jee_apparatus_coverage_out_of_30":"not evaluated as a 30-case cohort",
        "browser_inspection":"separate real Chromium gate required",
        "textbook_answer_completeness":"not attempted",
    }


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    source=json.loads(args.input.read_text(encoding="utf-8"))
    result=check_native(source)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("M36b source-specific independent physical checks:",len(result["independently_checked_cases"]))
    print("M36b justified 3D local-field refusal (not simulated):",result["justified_local_3d_refusals_not_simulations"])


if __name__ == "__main__":
    main()
