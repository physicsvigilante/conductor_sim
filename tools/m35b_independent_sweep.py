"""M35b separately expressed real-native M28a/M28b SI physics/evidence reference.

NO conductor_sim imports. Inputs are JSON exports of actual historical
uniform-mass sweep native integration, not an external signature or proof
of the original textbook. Analytic F=qE, r(t), v(t), signed detector roots,
actual saved clock comparisons, source ancestry and hostile detached tamper
controls are independently checked, rather than trusting production callbacks.
"""
from __future__ import annotations

import argparse
import copy
from hashlib import sha256
import json
from math import isclose, isfinite, sqrt
from pathlib import Path

Q = 1e-6
E = 1000.
MASSES = (.001, .0005, .002)
DET = .05
DURATION = .5


def require(test, reason):
    if not test:
        raise AssertionError(reason)


def num(v):
    require(type(v) in (int,float) and isfinite(v),
            "invalid native numeric type including Boolean or nonfinite SI data")
    return float(v)


def close(v, expectation, label, tolerance=2e-8):
    require(isclose(num(v), expectation, rel_tol=0., abs_tol=tolerance),
            f"{label}: {v!r} differs from independent {expectation!r}")


def triple(values):
    require(type(values) in (list,tuple) and len(values)==3,
            "native record must contain a real three-component SI vector")
    return tuple(num(x) for x in values)


def require_state(samples, mass, label):
    require(type(samples) is list and len(samples)>=3, label+" lacks actual samples")
    acceleration=Q*E/mass
    prev=None
    err_pos=err_vel=err_force=0.
    for row in samples:
        require(type(row["time_s"]) is float,
                label+" native physical time type changed (False != 0.0)")
        t=num(row["time_s"])
        require(0<=t<=DURATION and (prev is None or t>prev),
                label+" native physical clock is nonmonotonic/outside horizon")
        prev=t
        p,v,f=triple(row["position_m"]),triple(row["velocity_m_s"]),triple(row["force_n"])
        expect_p=(.5*acceleration*t*t,0.,0.)
        expect_v=(acceleration*t,0.,0.)
        expect_f=(Q*E,0.,0.)
        err_pos=max(err_pos,*(abs(p[k]-expect_p[k]) for k in range(3)))
        err_vel=max(err_vel,*(abs(v[k]-expect_v[k]) for k in range(3)))
        err_force=max(err_force,*(abs(f[k]-expect_f[k]) for k in range(3)))
        for k in range(3):
            close(p[k],expect_p[k],label+" x/y/z position",5e-8)
            close(v[k],expect_v[k],label+" x/y/z velocity",5e-8)
            close(f[k],expect_f[k],label+" x/y/z force",1e-12)
        close(row["potential_v"],-E*expect_p[0],label+" true V(x)",5e-5)
        close(row["kinetic_energy_j"],.5*mass*(acceleration*t)**2,
              label+" kinetic energy",5e-12)
        close(row["total_energy_j"],0.,label+" conservative mechanical total",5e-10)
    close(samples[0]["time_s"],0.,label+" true initial physical time",1e-15)
    close(samples[-1]["time_s"],DURATION,label+" full observation end",1e-12)
    return dict(saved_states=len(samples),max_position_error_m=err_pos,
                max_velocity_error_m_s=err_vel,max_force_error_n=err_force)


def scenario(s, mass, expected_id=None, parent=None):
    require(type(s) is dict and s["schema"]=="conductor_sim.prescribed_scenario.v1",
            "missing actual model native scenario")
    if expected_id is not None:
        require(s["scenario_id"]==expected_id,"wrong child physical source identity")
    if parent is not None:
        require(s["parent_revision_id"]==parent,"lost true source model parent SHA")
    spec=s["spec"]
    near=(spec["field"]["sources"][0])
    require(len(spec["field"]["sources"])==1 and
            near["kind"]=="UniformElectricField","original fixed source type lost")
    require(tuple(triple(near["field_v_m"]))==(E,0.,0.),
            "source E differs from teacher fixed physical model")
    require(tuple(x[0] for x in s["source_annotations"])==("field:E",),
            "source identity or inventory changed")
    close(spec["particle"]["charge_c"],Q,"physical q",1e-16)
    close(spec["mass_kg"],mass,"physical mass",1e-15)
    close(spec["duration_s"],DURATION,"duration",1e-15)
    require(tuple(triple(spec["particle"]["position_m"]))==(0.,0.,0.) and
            tuple(triple(spec["initial_velocity_m_s"]))==(0.,0.,0.),
            "initial physical state changed from independently specified experiment")
    return s["revision_id"]


def native_event(er,mass,revision,label):
    require(er["schema"]=="conductor_sim.event_run.v1","not genuine M24a native event run")
    require(er["study"]["model_revision_id"]==revision,
            "event observation not bound to current physical source SHA")
    qs=er["study"]["questions"]
    require(len(qs)==1 and qs[0]["event_id"]=="x:detector" and
            qs[0]["kind"]=="PlaneEvent" and qs[0]["direction"]==1 and
            tuple(triple(qs[0]["point_m"]))==(DET,0.,0.) and
            tuple(triple(qs[0]["normal"]))==(1.,0.,0.),
            "typed native event-plane question geometry/direction was changed")
    traj=er["trajectory"]
    require(traj["termination"]=="duration","event horizon was not really complete")
    state=require_state(traj["samples"],mass,label)
    require(er["outcomes"]["x:detector"]=="located","event is not uniquely located")
    roots=er["events"]
    require(len(roots)==1 and roots[0]["event_id"]=="x:detector" and
            roots[0]["direction"]==1,"missing single correct signed M24a root")
    root=roots[0]
    t=num(root["sample"]["time_s"])
    require(type(root["sample"]["time_s"]) is float,
            "Boolean root time is not real native SI evidence")
    lo,hi=(num(x) for x in root["time_bracket_s"])
    require(lo<=t<=hi and hi-lo<=1.01e-8,
            "root point left genuine numerical M24a bracket")
    true_t=sqrt(2*DET*mass/(Q*E))
    close(t,true_t,label+" independent positive signed detector time",4e-8)
    close(root["sample"]["position_m"][0],DET,label+" root lies on detector",2e-8)
    return dict(**state,root_time_s=t,true_root_time_s=true_t,
                root_bracket_s=[lo,hi],times=[r["time_s"] for r in traj["samples"]],
                samples=traj["samples"])


def assert_pairs(rec,old,new,label,event=False):
    rows=(rec["exact_time_trajectory_comparisons"] if event else rec["comparisons"])
    oldtimes={x["time_s"]:x for x in old["samples"]}
    exact=[x for x in new["samples"] if x["time_s"] in oldtimes]
    require(len(rows)==len(exact) and bool(rows),
            label+" has invented/interpolated or missing real same-time SI samples")
    ma,mb=old["mass"],new["mass"]
    for row,r in zip(rows,exact):
        require(type(row["time_s"]) is float and row["time_s"]==r["time_s"],
                label+" native shared timestamp is not literally identical")
        t=row["time_s"]
        ax=Q*E/ma;bx=Q*E/mb
        close(row["delta_position_m"][0],.5*(bx-ax)*t*t,
              label+" right minus left position",4e-8)
        close(row["delta_velocity_m_s"][0],(bx-ax)*t,
              label+" right minus left velocity",4e-8)
        for k in (1,2):
            close(row["delta_position_m"][k],0.,label+" transverse position",1e-12)
            close(row["delta_velocity_m_s"][k],0.,label+" transverse velocity",1e-12)
    if event:
        shifts=rec["event_time_comparisons"]
        require(len(shifts)==1 and shifts[0]["status"]=="one-resolved-pair" and
                shifts[0]["direction"]==1 and shifts[0]["event_id"]=="x:detector",
                label+" event roots should be one compatible same-direction pair")
        sh=shifts[0]
        independent=sqrt(2*DET*mb/(Q*E))-sqrt(2*DET*ma/(Q*E))
        close(sh["estimated_numerical_shift_s"],independent,
              label+" native M24a right-minus-left event shift",5e-8)
        lo=new["root_bracket_s"][0]-old["root_bracket_s"][1]
        hi=new["root_bracket_s"][1]-old["root_bracket_s"][0]
        close(sh["numerical_bracket_shift_interval_s"][0],lo,
              label+" independent bracket lower",2e-15)
        close(sh["numerical_bracket_shift_interval_s"][1],hi,
              label+" independent bracket upper",2e-15)
    return len(rows)


def verify(motion,events):
    require(motion["schema"]=="conductor_sim.trajectory_sweep.v1" and
            events["schema"]=="conductor_sim.event_sweep.v1",
            "original M28a/M28b exported schemas were changed")
    require(motion["plan_id"]==events["plan_id"]=="m28b:uniform-mass",
            "native old and event model plan IDs differ")
    require(motion["axis"]==events["axis"],"old and event axis not the same physics")
    require(len(motion["entries"])==len(events["entries"])==2,
            "wrong number of actual independent physical mass revisions")
    original=motion["baseline_scenario"]
    base_sha=scenario(original,MASSES[0],"m28b:uniform-base")
    require(events["baseline_revision_id"]==base_sha and
            events["baseline_scenario"]["revision_id"]==base_sha,
            "event baseline not the identical original physical revision")
    motion_native=motion["baseline_trajectory"]
    require(motion_native["termination"]=="duration",
            "baseline no-event physical trajectory incomplete")
    m0=require_state(motion_native["samples"],MASSES[0],"baseline motion")
    m0.update(mass=MASSES[0],samples=motion_native["samples"])
    e0=native_event(events["baseline_event_run"],MASSES[0],base_sha,
                    "baseline event")
    e0.update(mass=MASSES[0])
    world_count=2; states=m0["saved_states"]+e0["saved_states"];paired=0;root_shifts=0
    for i,mass in enumerate(MASSES[1:]):
        mr,er=motion["entries"][i],events["entries"][i]
        require(mr["status"]==er["status"]=="complete",
                "legitimate native complete result marked refused")
        child=f"m28b:uniform-mass:v{i+1:02d}"
        child_sha=scenario(mr["scenario"],mass,child,parent=base_sha)
        require(er["scenario"]["revision_id"]==child_sha,
                "event/native child physical model SHA differs")
        mt=mr["trajectory"]
        require(mt["termination"]=="duration","M28a child stopped unexpectedly")
        m1=require_state(mt["samples"],mass,"M28a child "+child)
        m1.update(mass=mass,samples=mt["samples"])
        e1=native_event(er["event_run"],mass,child_sha,"M28b child "+child)
        e1.update(mass=mass)
        paired+=assert_pairs(mr,m0,m1,"M28a "+child)
        paired+=assert_pairs(er,e0,e1,"M28b "+child,event=True)
        root_shifts+=1
        states+=m1["saved_states"]+e1["saved_states"]
        world_count+=2
    require(motion["time_interpolation"] is False and
            motion["extrapolation"] is False and
            events["time_interpolation"] is False and
            events["postguard_extrapolation"] is False,
            "invented interpolated or later afterguard physical evidence")
    return dict(physical_native_cases=world_count,saved_three_component_samples=states,
                literal_shared_comparison_rows=paired,
                independently_verified_event_shifts=root_shifts,
                verdict="PASS")


def adversarial(motion,events):
    attacks=[
        ("complete-without-model",lambda a,b: b["entries"][0].__setitem__("scenario",None)),
        ("complete-without-native",lambda a,b: b["entries"][0].__setitem__("event_run",None)),
        ("complete-without-motion",lambda a,b: a["entries"][0].__setitem__("trajectory",None)),
        ("Boolean-zero-clock",lambda a,b: a["baseline_trajectory"]["samples"][0].__setitem__("time_s",False)),
        ("Boolean-event-clock",lambda a,b: b["entries"][0]["event_run"]["trajectory"]["samples"][0].__setitem__("time_s",False)),
        ("wrong-root",lambda a,b: b["entries"][0]["event_run"]["events"][0]["sample"].__setitem__("time_s",.49)),
        ("wrong-root-bracket",lambda a,b: b["entries"][0]["event_run"]["events"][0].__setitem__("time_bracket_s",[.49,.5])),
        ("fake-source-id",lambda a,b: a["entries"][0]["scenario"].__setitem__("scenario_id","FAKE")),
        ("wrong-signed-shift",lambda a,b: b["entries"][0]["event_time_comparisons"][0].__setitem__("estimated_numerical_shift_s",4.)),
        ("fake-total-force",lambda a,b: a["entries"][0]["trajectory"]["samples"][1].__setitem__("force_n",[0,0,0])),
    ]
    for name,attack in attacks:
        a,b=copy.deepcopy(motion),copy.deepcopy(events)
        attack(a,b)
        try: verify(a,b)
        except (AssertionError,ValueError,KeyError,TypeError,IndexError):
            continue
        raise AssertionError("Failed to detect hostile changed native evidence: "+name)
    return len(attacks)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--motion",required=True,type=Path)
    p.add_argument("--events",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    rawm=args.motion.read_bytes();rawe=args.events.read_bytes()
    motion=json.loads(rawm);events=json.loads(rawe)
    report=verify(motion,events)
    report["hostile_controls_rejected"]=adversarial(motion,events)
    report["motion_sha256"]=sha256(rawm).hexdigest()
    report["event_sha256"]=sha256(rawe).hexdigest()
    report["production_conductor_sim_imports"]=0
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,allow_nan=False,indent=2)+"\n",
                           encoding="utf-8")
    print(json.dumps(report,sort_keys=True))


if __name__=="__main__":
    main()
