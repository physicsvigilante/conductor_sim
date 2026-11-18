"""M36c: source-free SI references for true original-JEE conductor observations.

No conductor_sim import and no production callbacks. Reads detached native
source-model observations and checks Gauss/Kelvin/static-field energy directly.
"""
from __future__ import annotations
import argparse
import json
from math import isclose, isfinite, pi
from pathlib import Path

EPSILON0=8.8541878128e-12
K=1/(4*pi*EPSILON0)
CASES=(
    "p03-single-27","p10-linked-13+p10-linked-14","p14-numerical-07",
    "p26-conductor-potential","p27-cavity-two-shells",
    "p69-cavity-central-vs-shifted","p09-multiple-27",
)
ANCHORS=(
    "p03-single-27","p10-linked-13","p10-linked-14","p14-numerical-07",
    "p26-conductor-potential","p27-cavity-two-shells",
    "p69-cavity-central-vs-shifted","p09-multiple-27",
)


def demand(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual,expected,*,tol=1e-10,abs_tol=1e-18,label=""):
    demand(type(actual) in (float,int) and isfinite(actual),"nonnumeric/nonfinite "+label)
    demand(isclose(actual,expected,rel_tol=tol,abs_tol=abs_tol),
           f"{label}: saved={actual!r}, Gauss/Kelvin reference={expected!r}")


def record(actual,name,charge,inner,outer,potential):
    demand(actual["name"]==name,"mismatched physical conductor identity "+name)
    close(actual["net_charge_c"],charge,label=name+" net C")
    close(actual["inner_surface_charge_c"],inner,label=name+" true inner face C")
    close(actual["outer_surface_charge_c"],outer,label=name+" outer face C")
    close(actual["potential_v"],potential,abs_tol=1e-11,label=name+" surface V")


def connection(card,case_id):
    demand(card["native"] is True and card["source_case_id"]==case_id,
           "connection source identity")
    demand(card["kind"]=="two_physical_static_equilibria_not_wire_current_time",
           "connection is not a time-dependent current simulation")
    demand(len(card["links"])==1 and card["links"][0]["mode"]=="floating",
           "qualified floating connected group, not ground")
    bodies=card["source_input_bodies"]
    before,after=card["before"],card["after"]
    demand(len(bodies)==len(before)==len(after)==2,"two true named conductors")
    q=20e-6 if case_id=="p03-single-27" else 1e-9
    if case_id=="p10-linked-13+p10-linked-14":
        q=4*pi*EPSILON0*.1**2*1.8e6
    outer_inner=.195 if case_id=="p14-numerical-07" else .199999
    close(bodies[0]["outer_radius_m"],.10,label="inner conductor radius")
    close(bodies[1]["outer_radius_m"],.20,label="outer conductor radius")
    close(bodies[1]["inner_radius_m"],outer_inner,label="real shell finite thickness")
    if case_id=="p03-single-27":
        demand(bodies[0]["kind"]=="SolidSphere" and bodies[0]["inner_radius_m"]==0.,
               "solid conductor has no fake inner cavity")
        demand(bodies[1]["kind"]=="SphericalShell","true outer shell")
        names=("solid","shell")
        qb=(q,0.);qa=(0.,q)
        outer_flux=q
    elif case_id=="p10-linked-13+p10-linked-14":
        demand(all(x["kind"]=="SphericalShell" for x in bodies),
               "p10 two thin spherical shells")
        names=("inner","outer")
        qb=(q,-q);qa=(0.,0.)
        outer_flux=0.
    else:
        demand(case_id=="p14-numerical-07","unexpected connection type")
        demand(all(x["kind"]=="SphericalShell" for x in bodies),
               "p14 two spherical shells")
        names=("inner","outer")
        qb=(q,2*q);qa=(0.,3*q)
        outer_flux=3*q
    demand(tuple(x["name"] for x in bodies)==names,"source names replaced")
    gap_term=1/.1-1/outer_inner
    gap_U=.5*K*q*q*gap_term
    ext_U=.5*K*outer_flux*outer_flux/.2
    close(card["radial_energy_before_j"],gap_U+ext_U,tol=2e-9,abs_tol=1e-12,
          label="full before-field energy gap+outside")
    close(card["radial_energy_after_j"],ext_U,tol=2e-9,abs_tol=1e-12,
          label="after-field exterior energy")
    close(card["released_energy_j"],gap_U,tol=2e-9,abs_tol=1e-12,
          label="real connection energy released")
    close(card["gap_radius_m"],.15,label="selected real vacuum gap")
    for key,x in (("gap_field_before_v_m",K*q/.15**2),
                  ("gap_field_after_v_m",0.)):
        field=card[key]
        demand(isinstance(field,list) and len(field)==3,"actual 3D field vector")
        close(field[0],x,tol=2e-9,abs_tol=1e-8,label=key+" x")
        close(field[1],0,abs_tol=1e-8,label=key+" y")
        close(field[2],0,abs_tol=1e-8,label=key+" z")
    Vouter=K*outer_flux/.2
    Vinner_before=Vouter+K*q*gap_term
    for i in range(2):
        close(before[i]["net_charge_c"],qb[i],label="source before net charge")
        close(after[i]["net_charge_c"],qa[i],label="source after net charge")
    if case_id=="p03-single-27":
        record(before[0],"solid",q,0.,q,Vinner_before)
        record(before[1],"shell",0.,-q,q,Vouter)
        record(after[0],"solid",0.,0.,0.,Vouter)
        record(after[1],"shell",q,0.,q,Vouter)
    elif case_id=="p10-linked-13+p10-linked-14":
        record(before[0],"inner",q,0.,q,Vinner_before)
        record(before[1],"outer",-q,-q,0.,0.)
        record(after[0],"inner",0.,0.,0.,0.)
        record(after[1],"outer",0.,0.,0.,0.)
    else:
        record(before[0],"inner",q,0.,q,Vinner_before)
        record(before[1],"outer",2*q,-q,3*q,Vouter)
        record(after[0],"inner",0.,0.,0.,Vouter)
        record(after[1],"outer",3*q,0.,3*q,Vouter)
    transfers=card["transfers"]
    demand(len(transfers)==2,"both conductor transfer records required")
    for i,x in enumerate(transfers):
        demand(x["name"]==names[i],"transfer source identity changed")
        close(x["before_c"],qb[i],label="before transfer C")
        close(x["after_c"],qa[i],label="after transfer C")
        close(x["received_c"],qa[i]-qb[i],label="signed actual transfer C")
    close(sum(qb),sum(qa),label="floating group charge conservation")
    return {"kind":"two_true_static_equilibria",
            "independent_gap_energy_release_j":gap_U,
            "independent_outer_energy_j":ext_U}


def original_p27_AB(card):
    """Independently reproduce the ORIGINAL centred q / shell V(A), V(B).

    Printed mathematical zero-thickness a,b shell relation differs slightly
    from the genuine authored 1-micrometre-thick metal solutions. Both are
    computed from independent SI Gauss-law terms rather than source callbacks.
    """
    demand(card["source_case_id"]=="p27-cavity-two-shells" and card["native"] is True,
           "original p27 source identity")
    demand(card["source_anchor"]=="Illustration 4" and card["source_pdf_page"]==27,
           "original p27 printed Illustration 4 attribution changed")
    demand(card["kind"]=="true_static_conductor_field_and_surface_observations",
           "p27 original source needs static conductor observers")
    demand(card["original_asked_observables"]==["V(A)","V(B)"],
           "actual printed p27 asks for BOTH original A and B potentials")
    demand("CENTRAL q" in card["source_givens"] and "symbolic" in card["source_givens"],
           "p27 actual source has central symbolic q, not displaced q")
    demand("separate shifted teacher extension" in card["source_shape_fidelity"],
           "do not silently substitute off-centre teacher point for printed source")
    teacher=card["teacher_numerical_values"]
    for label,v in [
        ("q_c",1e-9),("q1_c",2e-9),("q2_c",3e-9),
        ("radius_a_m",.10),("radius_b_m",.20),
        ("authored_metal_thickness_m",1e-6),
    ]:
        close(teacher[label],v,tol=1e-12,abs_tol=1e-22,
              label="teacher numerical model scalar "+label)
    demand(teacher["A_world_m"]==[.04,0.,0.] and
           teacher["B_world_m"]==[.15,0.,0.] and
           teacher["displaced_teacher_source_m"]==[0.,0.,.02],
           "teacher probes A/B must be in the physical source regions")
    q,q1,q2=1e-9,2e-9,3e-9
    a,b=.10,.20
    a_in,b_in=a-1e-6,b-1e-6
    r_A,r_B=.04,.15
    ideal_A=K*(q/r_A+q1/a+q2/b)
    ideal_B=K*((q+q1)/r_B+q2/b)
    formula=card["original_formula_limit"]
    demand("zero-thickness" in formula["scope"] and
           "NOT the saved numerical observation" in formula["scope"],
           "original ideal limit and actual finite metal geometry conflated")
    close(formula["V_A_zero_thickness_v"],ideal_A,tol=2e-12,
          abs_tol=1e-10,label="printed symbolic V(A) formula")
    close(formula["V_B_zero_thickness_v"],ideal_B,tol=2e-12,
          abs_tol=1e-10,label="printed symbolic V(B) formula")
    # For two finite-thickness conducting shells, charge q gives -q on
    # inner face, +q+q1 on first outer face, -(q+q1) on second inner
    # face and q+q1+q2 on second outer face. A and B are in vacuum.
    actual_A=K*(q/r_A-q/a_in+(q+q1)/a-(q+q1)/b_in+(q+q1+q2)/b)
    actual_B=K*((q+q1)/r_B-(q+q1)/b_in+(q+q1+q2)/b)
    demand(abs(actual_A-ideal_A)>.0 and abs(actual_B-ideal_B)>.0,
           "finite-thickness and ideal thin-shell outputs are distinct")
    demand(abs(actual_A-ideal_A)<.002 and abs(actual_B-ideal_B)<.002,
           "thin model approximation must be quantitatively limited")
    states=card["authored_states"]
    demand(len(states)==2,"original source and distinct teacher extension required")
    for i,state in enumerate(states):
        demand(state["index"]==i and len(state["conductors"])==2
               and len(state["charges"])==1,
               "original two metal bodies and one source per static state")
        expected_pos=[0.,0.,0.] if i==0 else [0.,0.,.02]
        demand(state["charges"][0]["position_m"]==expected_pos,
               "printed central q must be FIRST source state")
        close(state["charges"][0]["charge_c"],q,label="actual q in cavity")
        c0,c1=state["conductors"]
        demand((c0["name"],c1["name"])==("inner","outer"),
               "actual named shell identity changed")
        close(c0["inner_radius_m"],a_in,label="first thin actual inner face")
        close(c0["outer_radius_m"],a,label="first thin actual outer face")
        close(c1["inner_radius_m"],b_in,label="outer thin actual inner face")
        close(c1["outer_radius_m"],b,label="outer thin actual outer face")
        result=state["native_conductor_results"]
        Vouter=K*(q+q1+q2)/b
        Vinner=K*((q+q1)*(1/a-1/b_in)+(q+q1+q2)/b)
        record(result[0],"inner",q1,-q,q+q1,Vinner)
        record(result[1],"outer",q2,-(q+q1),q+q1+q2,Vouter)
    demand("PRINTED ORIGINAL" in states[0]["title"] and
           "TEACHER EXTENSION ONLY" in states[1]["title"],
           "original and supplementary conductor models not distinguished")
    obs=card["native_observations"]
    checks=[
        ("Printed original V(A), central q in cavity","potential",0,
         [.04,0.,0.],None,None,None,actual_A),
        ("Printed original V(B), intershell gap","potential",0,
         [.15,0.,0.],None,None,None,actual_B),
        ("Printed original inner shell total","shell_net_charge",0,
         None,"inner",None,None,q1),
        ("Printed original outer shell total","shell_net_charge",0,
         None,"outer",None,None,q2),
        ("TEACHER off-centre extension, inner face -q","inner_total",1,
         None,"inner",None,None,-q),
    ]
    demand(len(obs)==len(checks),"p27 source A/B observation ledger altered")
    for i,(row,expected) in enumerate(zip(obs,checks)):
        label,kind,state,point,conductor,other,theta,value=expected
        demand(row["label"]==label and row["kind"]==kind
               and row["state_index"]==state,
               f"p27 observation/source label mismatch #{i}")
        demand(row["point_m"]==point and row["conductor"]==conductor
               and row["other_conductor"]==other and row["theta_rad"]==theta,
               f"p27 wrong conductor/angle/position attribution #{i}")
        close(row["native_value"],value,tol=2e-10,abs_tol=2e-12,
              label=f"source p27 native V(A), V(B), shell observations #{i}")
    return {"kind":"genuine_source_central_A_B_plus_disclosed_teacher_extension",
            "source_V_A_actual_v":actual_A,
            "source_V_B_actual_v":actual_B,
            "printed_ideal_V_A_v":ideal_A,
            "printed_ideal_V_B_v":ideal_B,
            "checked_source_observations":len(obs)}


def notes(card,id_):
    demand(card["source_case_id"]==id_ and card["native"] is True,"notes source identity")
    demand(card["kind"]=="true_static_conductor_field_and_surface_observations",
           "conductor source observes static V/E, not particle motion")
    page={"p26-conductor-potential":26,"p27-cavity-two-shells":27,
          "p69-cavity-central-vs-shifted":69}[id_]
    demand(card["source_pdf_page"]==page,"wrong pinned physical PDF page")
    states=card["authored_states"]
    obs=card["native_observations"]
    if id_=="p27-cavity-two-shells":
        return original_p27_AB(card)
    require_source={"p26-conductor-potential":"Illustration 1",
                    "p69-cavity-central-vs-shifted":"Illustration 7"}
    demand(card["source_anchor"]==require_source[id_],
           "source original illustration anchor attribution changed")
    Q=1e-9
    if id_=="p26-conductor-potential":
        demand(len(states)==1 and len(obs)==3,"shell potential/field scope")
        s=states[0]
        demand(len(s["conductors"])==1 and s["charges"]==[],"one shell/no internal point")
        demand("SOURCE SOLID sphere; HOLLOW shell" in card["source_shape_fidelity"]
               and "PROXY" in card["source_shape_fidelity"],
               "p26 source shape is a SOLID sphere; hidden hollow proxy must be disclosed")
        shell=s["conductors"][0]
        close(shell["inner_radius_m"],.09,label="proxy hollow metal thickness inner")
        close(shell["outer_radius_m"],.10,label="source conductor R")
        record(s["native_conductor_results"][0],"sphere",Q,0.,Q,K*Q/.1)
        expected=[("potential",K*Q/.1,0,(.05,0.,0.)),
                  ("potential",K*Q/.15,0,(.15,0.,0.)),
                  ("field_x",0.,0,(.05,0.,0.))]
    else:
        demand(len(states)==2 and len(obs)==7,"central and shifted are two STATIC states")
        for i,s in enumerate(states):
            demand(len(s["conductors"])==1 and len(s["charges"])==1,
                   "one true metal shell and one cavity charge")
            shell=s["conductors"][0]
            close(shell["inner_radius_m"],.08,label="cavity boundary radius")
            close(shell["outer_radius_m"],.10,label="outer conducting radius")
            charge=s["charges"][0]
            close(charge["charge_c"],Q,label="cavity positive point charge")
            demand(charge["position_m"]==([0.,0.,0.] if i==0 else [0.,0.,.04]),
                   "genuine centre versus shifted source states")
            record(s["native_conductor_results"][0],"body",3*Q,-Q,4*Q,K*4*Q/.1)
        a,d=.08,.04
        centered=-Q/(4*pi*a*a)
        near=-Q*(a*a-d*d)/(4*pi*a*(a-d)**3)
        far=-Q*(a*a-d*d)/(4*pi*a*(a+d)**3)
        demand(abs(near)>abs(centered)>abs(far),
               "off-centre image charge induces angular inner density")
        expected=[("inner_total",-Q,0,None),("outer_total",4*Q,0,None),
                  ("inner_total",-Q,1,None),("outer_total",4*Q,1,None),
                  ("inner_sigma",centered,0,None),
                  ("inner_sigma",near,1,None),("inner_sigma",far,1,None)]
    for i,(row,(kind,value,state,point)) in enumerate(zip(obs,expected)):
        demand(row["kind"]==kind and row["state_index"]==state,
               f"original source kind/state #{i} changed")
        if point is not None:
            demand(isinstance(row["point_m"],list) and len(row["point_m"])==3,
                   f"original probe world vector #{i} missing")
            for axis,(observed_m,expected_m) in enumerate(zip(row["point_m"],point)):
                close(observed_m,expected_m,tol=2e-14,abs_tol=2e-15,
                      label=f"actual source world probe #{i} axis {axis}")
        # Valid scientific evidence attributes each value to the correct
        # physical surface and angle; numerically correct values with swapped
        # conductor names or theta labels are NOT valid observations.
        if id_=="p26-conductor-potential":
            names=("potential at R/2","potential at 3R/2",
                   "electric field in hollow interior")
            conductor=None
            theta=None
        else:
            names=("central charge induces -q inner",
                   "outer charge is 4q",
                   "off-center inner total remains -q",
                   "outer total remains 4q",
                   "centered inner sigma uniform",
                   "near-side enhanced inner sigma",
                   "far-side weaker inner sigma")
            conductor="body"
            theta=(0.,None,None,None,0.,0.,pi)[i]
        demand(row["label"]==names[i],
               f"source original observable label attribution #{i} changed")
        demand(row["conductor"]==conductor and row["other_conductor"] is None,
               f"wrong physical conductor association #{i}")
        if theta is None:
            demand(row["theta_rad"] is None,
                   f"unexpected physical angle for observer #{i}")
        else:
            close(row["theta_rad"],theta,tol=2e-14,abs_tol=1e-15,
                  label=f"physical Kelvin sigma angle #{i}")
        if point is None:
            demand(row["point_m"] is None,
                   f"surface datum must not be relabelled as spatial sample #{i}")
        close(row["native_value"],value,tol=5e-9,abs_tol=2e-14,
              label=f"independent {id_} physical native observable #{i}")
    return {"kind":"static_spherical_conductor","independent_observations":len(expected)}


def external(card):
    demand(card["source_case_id"]=="p09-multiple-27" and card["native"] is True,
           "held point near isolated neutral conductor")
    demand(card["kind"]=="static_external_neutral_sphere_kelvin_induction_not_approach_time",
           "do not infer elapsed approach current from fixed Kelvin position")
    require=card["source_givens"]
    demand(card["source_page"]==9,"source workbook p9")
    close(require["sphere_radius_m"],3.,label="physical sphere radius 3m")
    close(require["sphere_net_charge_c"],0.,label="isolated conductor charge ZERO")
    close(require["point_source_charge_c"],1e-6,label="given +1 microC")
    demand(require["point_source_position_m"]==[3.,4.,0.],
           "actual 3-4-5 source geometry")
    close(require["tangent_segment_ab_m"],4.,label="actual tangent AB")
    V=K*1e-6/5.
    induced=-K*1e-6/20.
    close(card["native_sphere_potential_v"],V,tol=1e-11,
          label="neutral Kelvin metal equipotential")
    close(card["native_induced_only_at_tangent_v"],induced,tol=1e-11,
          label="Kelvin induced-only tangent potential")
    close(card["native_total_potential_at_tangent_v"],K*1e-6/4+induced,
          tol=1e-11,label="direct plus induced surface potential")
    demand("Additional" in card["teacher_position_sweep"],"extra static sweep is teacher-authored")
    return {"kind":"static_neutral_kelvin","sphere_v":V,"induced_only_v":induced}


def verify(manifest):
    demand(manifest["schema"]=="conductor_sim.m36c_conductor_source_replays.v1",
           "source manifest schema")
    sources=manifest["native_source_apparatus"]
    demand(tuple(sources)==CASES and manifest["source_case_count"]==7,
           "seven fixed physical apparatus, no double-counting")
    demand(tuple(manifest["historical_anchors_used"])==ANCHORS,
           "linked Q13+Q14 two question anchors but ONE physical setup")
    result={}
    for id_ in CASES[:3]:
        result[id_]=connection(sources[id_],id_)
    for id_ in CASES[3:6]:
        result[id_]=notes(sources[id_],id_)
    result[CASES[6]]=external(sources[CASES[6]])
    controls=manifest["non_success_controls"]
    demand(len(controls)==2,"two partial/unsupported honesty controls")
    demand(controls[0]["id"]=="p08-multiple-21" and controls[0]["kind"]=="PARTIAL"
           and controls[0]["native"] is False,
           "static induction is NOT a prescribed approach-time electric current")
    demand(controls[1]["id"]=="p70-three-cavities"
           and controls[1]["kind"]=="UNSUPPORTED" and controls[1]["native"] is False,
           "three irregular 3D cavities local field cannot be fabricated")
    return {
        "schema":"conductor_sim.m36c_jee_conductor_si_reference.v1",
        "checked_distinct_source_apparatus":len(result),
        "source_question_anchors":len(ANCHORS),
        "source_independent_reference_results":result,
        "honest_partial_or_refused_controls":2,
        "actual_chromium":"separate browser acceptance required",
        "automatic_jee_30_solved_percentage":"NOT claimed; selected source-backed conductor pilot",
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    data=json.loads(args.input.read_text(encoding="utf-8"))
    report=verify(data)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("M36c independent original conductor-apparatus references:",
          report["checked_distinct_source_apparatus"])
    print("M36c original historical question anchors:",report["source_question_anchors"])


if __name__=="__main__":
    main()
