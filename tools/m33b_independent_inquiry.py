"""Independent-form M33b SI physics checker (NO conductor_sim imports).

Separately evaluates every saved 3D constant-acceleration state, force and
signed detector root of FOUR actual newly reintegrated M24a world observations,
pairwise saved-time comparisons, in-process graph-provenance hash and source
identity ancestry. A finite regression oracle, not a general ODE error theorem
and not a separate-context scientific reviewer verdict.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from math import dist, fsum, isclose, isfinite, sqrt
from pathlib import Path

EXPECTED = {
    "baseline": dict(e=(100.,50.,0.),mass=1e-5,
        ids=("field:E0","field:Ey"), path=("baseline",), parent=None),
    "alpha": dict(e=(140.,50.,0.),mass=1e-5,
        ids=("field:E0","field:Ey","field:alternate"),
        path=("baseline","alpha"), parent="baseline"),
    "beta": dict(e=(220.,50.,0.),mass=1e-5,
        ids=("field:E0","field:Ey","field:alternate"),
        path=("baseline","beta"), parent="baseline"),
    "heavier": dict(e=(140.,50.,0.),mass=2e-5,
        ids=("field:E0","field:Ey","field:alternate"),
        path=("baseline","alpha","heavier"), parent="alpha"),
}
ORDER=("baseline","alpha","beta","heavier")
PAIRS=(("siblings","alpha","beta"),
       ("cousins","beta","heavier"),
       ("baseline","baseline","beta"))


def require(check, msg):
    if not check:
        raise AssertionError(msg)


def number(value):
    require(type(value) in (float,int) and isfinite(value),
            "numeric datum missing/Boolean/non-finite")
    return float(value)


def vector(value):
    require(type(value) in (list,tuple) and len(value)==3,
            "not a physical SI 3-vector")
    return tuple(number(x) for x in value)


def eq(actual,expected,tolerance,msg):
    require(isclose(number(actual),expected,rel_tol=0,abs_tol=tolerance),
            msg+f": {actual!r} vs {expected!r}")


def veq(actual,expected,tolerance,msg):
    for k,(a,b) in enumerate(zip(vector(actual), expected)):
        eq(a,b,tolerance,msg+f" / axis {k}")


def digest(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(",",":"),
        ensure_ascii=True,allow_nan=False).encode("ascii")).hexdigest()


def cert(data,model,name):
    exp=EXPECTED[name]
    require(data["schema"]=="conductor_sim.source_world_ancestry.v1",
            name+": no M32b ancestry")
    require(data["world_id"]==name and
            data["parent_world_id"]==exp["parent"],
            name+": wrong ancestry immediate parent world")
    require(data["model_revision_id"]==model["revision_id"],
            name+": wrong source-model revision")
    require(tuple(data["active_source_ids"])==exp["ids"] and
            tuple(data["seen_source_ids"])==exp["ids"] and
            tuple(data["world_path"])==exp["path"] and
            not data["retired_source_ids"],
            name+": altered branch-local source history")
    require(data["gravity_present"] is False,
            name+": unexpected gravity component")
    if name=="heavier":
        require(data["introduced_source_ids"]==[],
                "physical-property-only change fabricated source introduction")
    elif name!="baseline":
        require(data["introduced_source_ids"]==["field:alternate"],
                name+": independent sibling new ID lost")


def physics(model,name):
    spec=model["spec"]
    exp=EXPECTED[name]
    require(spec.get("gravity") is None,"uniform reference has no gravity")
    require(all(source["kind"]=="UniformElectricField"
                for source in spec["field"]["sources"]),
            name+": unsupported field in reference")
    actual_e=tuple(fsum(vector(source["field_v_m"])[k]
              for source in spec["field"]["sources"]) for k in range(3))
    veq(actual_e,exp["e"],1e-12,name+": held E")
    eq(spec["particle"]["charge_c"],1e-9,1e-18,name+": q")
    eq(spec["mass_kg"],exp["mass"],1e-16,name+": mass")
    veq(spec["particle"]["position_m"],(0.,0.,0.),1e-15,name+": r0")
    veq(spec["initial_velocity_m_s"],(.1,0.,0.),1e-15,name+": v0")
    eq(spec["duration_s"],.1,1e-12,name+": duration")
    require(tuple(row[0] for row in model["source_annotations"])==exp["ids"],
            name+": model annotations not expected stable sources")
    q=number(spec["particle"]["charge_c"])
    mass=number(spec["mass_kg"])
    return dict(q=q,mass=mass,e=actual_e,
        a=tuple(q*actual_e[k]/mass for k in range(3)),
        r0=vector(spec["particle"]["position_m"]),
        v0=vector(spec["initial_velocity_m_s"]))


def expected_state(p,t):
    r=tuple(p["r0"][k]+p["v0"][k]*t+.5*p["a"][k]*t*t for k in range(3))
    v=tuple(p["v0"][k]+p["a"][k]*t for k in range(3))
    force=tuple(p["q"]*p["e"][k] for k in range(3))
    return r,v,force


def root(p):
    distance=.006-p["r0"][0]
    speed=p["v0"][0]
    acceleration=p["a"][0]
    return 2*distance/(speed+sqrt(speed*speed+2*acceleration*distance))


def check_native(row,source_model,name):
    require(row["world_id"]==name and row["status"]=="complete",
            name+": missing complete NEW inquiry")
    require(row["model_revision_id"]==source_model["revision_id"],
            name+": wrong source-model revision in inquiry")
    require(row["original_graph_observation_status"]=="complete",
            name+": prior ordinary native graph status altered")
    cert(row["source_ancestry"],source_model,name)
    require(tuple(s["id"] for s in row["source_inventory"])==EXPECTED[name]["ids"],
            name+": source inventory IDs differ")
    require(tuple(s["label"] for s in row["source_inventory"])==
            tuple(label for _id,label in source_model["source_annotations"]),
            name+": authored source labels altered")
    require(all(item["kind"]=="UniformElectricField" for item in row["source_inventory"]),
            name+": source kind changed")
    native=row["native_event_run"]
    require(native["schema"]=="conductor_sim.event_run.v1",
            name+": missing genuine M24a event run")
    require(native["study"]["model_revision_id"]==source_model["revision_id"] and
            native["study"]["model"]==source_model,
            name+": new native event study is not from unchanged source world")
    require(tuple(q["event_id"] for q in native["study"]["questions"])==
            ("x:followup",),
            name+": old or invented event question used")
    require(native["trajectory"]["termination"]=="duration",
            name+": event trajectory not duration-complete")
    require(native["outcomes"]["x:followup"]=="located" and
            row["native_event_outcomes"]==native["outcomes"],
            name+": event outcome altered")
    p=physics(source_model,name)
    samples=native["trajectory"]["samples"]
    require(len(samples)>=3,name+": too few actual native SI samples")
    times=[]
    max_pos=max_vel=max_force=0.
    for sample in samples:
        t=number(sample["time_s"])
        require(type(sample["time_s"]) is float and 0<=t<=.1,
                name+": saved native time is Boolean, int or outside horizon")
        times.append(t)
        true_r,true_v,true_f=expected_state(p,t)
        actual_r=vector(sample["position_m"])
        actual_v=vector(sample["velocity_m_s"])
        actual_f=vector(sample["force_n"])
        max_pos=max(max_pos,*(abs(actual_r[k]-true_r[k]) for k in range(3)))
        max_vel=max(max_vel,*(abs(actual_v[k]-true_v[k]) for k in range(3)))
        max_force=max(max_force,*(abs(actual_f[k]-true_f[k]) for k in range(3)))
    require(times[0]==0. and abs(times[-1]-.1)<1e-10 and
            all(b>a for a,b in zip(times,times[1:])),
            name+": actual event-study SI clock invalid")
    require(max_pos<3e-8 and max_vel<3e-8 and max_force<2e-13,
            name+": new M24a native saved states disagree with independent Newton")
    roots=[event for event in native["events"] if event["event_id"]=="x:followup"]
    require(len(roots)==1 and roots[0]["direction"]==1,
            name+": genuine signed event root missing")
    true_t=root(p)
    actual_t=number(roots[0]["sample"]["time_s"])
    bracket=tuple(number(t) for t in roots[0]["time_bracket_s"])
    require(type(roots[0]["sample"]["time_s"]) is float and
            bracket[0]<=actual_t<=bracket[1] and
            abs(actual_t-true_t)<2e-7,
            name+": native event root not independently consistent")
    eq(roots[0]["sample"]["position_m"][0],.006,2e-7,name+": event plane")
    return dict(p=p,times=times,root=actual_t,true_root=true_t,bracket=bracket,
                max_pos=max_pos,max_vel=max_vel,max_force=max_force)


def check_pair(pair,byid):
    expected = PAIRS[("siblings","cousins","baseline").index(pair["comparison_id"])]
    ident,left,right=expected
    require((pair["comparison_id"],pair["left_world_id"],pair["right_world_id"])==
            expected and pair["status"]=="compared" and
            pair["numerical_orientation"]=="right minus left",
            "wrong inquiry pair identity/orientation")
    lhs,rhs=byid[left],byid[right]
    require(pair["left_revision_id"]==lhs["model"]["revision_id"] and
            pair["right_revision_id"]==rhs["model"]["revision_id"],
            "pair no longer binds true source model revisions")
    for side,name in (("left",left),("right",right)):
        cert(pair[side+"_source_ancestry"],byid[name]["model"],name)
        require(tuple(v["id"] for v in pair[side+"_source_inventory"])==
                EXPECTED[name]["ids"],"pair source inventory on wrong side")
    expected_times=[t for t in rhs["native"]["times"]
                    if t in set(lhs["native"]["times"])]
    rows=pair["exact_time_trajectory_comparisons"]
    require(len(rows)==len(expected_times) and rows,
            ident+": not literally equal actual saved SI times")
    max_delta_error=0.
    for row,t in zip(rows,expected_times):
        require(type(row["time_s"]) is float and row["time_s"]==t,
                ident+": Boolean, interpolated or missing native physical clock")
        lp,lv,_=expected_state(lhs["native"]["p"],t)
        rp,rv,_=expected_state(rhs["native"]["p"],t)
        dp=tuple(rp[k]-lp[k] for k in range(3))
        dv=tuple(rv[k]-lv[k] for k in range(3))
        veq(row["delta_position_m"],dp,4e-8,ident+": position difference")
        veq(row["delta_velocity_m_s"],dv,4e-8,ident+": velocity difference")
        eq(row["position_distance_m"],dist(lp,rp),4e-8,ident+": displacement norm")
        eq(row["velocity_distance_m_s"],dist(lv,rv),4e-8,ident+": velocity norm")
        max_delta_error=max(max_delta_error,*(
            abs(vector(row["delta_position_m"])[k]-dp[k]) for k in range(3)))
    events=pair["event_time_comparisons"]
    require(len(events)==1 and events[0]["status"]=="one-resolved-pair" and
            events[0]["direction"]==1 and events[0]["event_id"]=="x:followup",
            ident+": wrong unique directional event pairing")
    event=events[0]
    true_shift=rhs["native"]["true_root"]-lhs["native"]["true_root"]
    numerical_shift=rhs["native"]["root"]-lhs["native"]["root"]
    eq(event["estimated_numerical_shift_s"],true_shift,4e-7,
       ident+": independent analytic event shift")
    eq(event["estimated_numerical_shift_s"],numerical_shift,1e-12,
       ident+": actual native event time difference")
    bounds=(rhs["native"]["bracket"][0]-lhs["native"]["bracket"][1],
            rhs["native"]["bracket"][1]-lhs["native"]["bracket"][0])
    for a,b in zip(event["numerical_bracket_shift_interval_s"],bounds):
        eq(a,b,1e-12,ident+": real numerical root bracket difference")
    return dict(id=ident,saved_matches=len(rows),
                numerical_event_shift_s=numerical_shift,
                analytic_event_shift_s=true_shift,
                max_position_delta_error_m=max_delta_error)


def verify(source,inquiry):
    require(source["schema"]=="conductor_sim.unified_observation_graph.v2",
            "new inquiry expects released, retained-ancestry v2 graph")
    require(source["plan"]["schema"]=="conductor_sim.unified_observation_plan.v2",
            "inquiry source was not a retained-source plan")
    require(source["plan_id"]=="m33b:original-worlds" and
            not source["plan"]["questions"] and
            source["original_observation_mode"]=="trajectory",
            "new measurement must be distinct from old trajectory-only graph")
    require(inquiry["schema"]=="conductor_sim.graph_event_inquiry.v1",
            "not a genuine M33b inquiry record")
    plan=inquiry["plan"]
    require(plan["schema"]=="conductor_sim.graph_event_inquiry_plan.v1" and
            plan["source_graph_schema"]==source["schema"] and
            plan["source_graph_plan_id"]==source["plan_id"] and
            plan["source_graph_sha256"]==digest(source),
            "new inquiry lost full originally issued source-graph hash")
    require(plan["inquiry_id"]=="m33b:new-x-question" and
            tuple(plan["world_ids"])==ORDER,
            "wrong inquiry ID or world selection")
    require([q["event_id"] for q in plan["questions"]]==["x:followup"] and
            plan["questions"][0]["direction"]==1 and
            vector(plan["questions"][0]["point_m"])==(.006,0.,0.),
            "not the genuinely newly asked x-detector question")
    require(tuple((x["comparison_id"],x["left_world_id"],x["right_world_id"])
                  for x in plan["comparisons"])==PAIRS,
            "new typed pair requests reordered/replaced")
    require(len(source["nodes"])==3 and len(inquiry["nodes"])==4 and
            len(inquiry["cross_world_comparisons"])==3,
            "not expected genuine authored graph/inquiry bounds")
    models={"baseline":source["plan"]["original"]}
    source_certs={"baseline":source["original_source_ancestry"]}
    for idx,name in enumerate(ORDER[1:]):
        node=source["nodes"][idx]
        require(node["node"]["node_id"]==name and
                node["status"]=="complete",
                name+": source graph never admitted a genuine model")
        require(node["node"]["observation"]=="trajectory" and
                node["native_observation"] is not None and
                node["native_event_outcomes"] is None,
                name+": new M24a study was not a fresh measurement")
        model=node["admitted_scenario"]
        require(node["parent_revision_id"]==
                models[EXPECTED[name]["parent"]]["revision_id"],
                name+": source graph parent physical SHA false")
        models[name]=model
        source_certs[name]=node["source_ancestry"]
    for name in ORDER:
        cert(source_certs[name],models[name],name)
    byid={}
    for name,row in zip(ORDER,inquiry["nodes"]):
        require(row["world_id"]==name,name+": reordered selected world")
        require(row["source_ancestry"]==source_certs[name],
                name+": native certified source ancestry changed")
        native=check_native(row,models[name],name)
        byid[name]=dict(model=models[name],native=native)
    pairs=[check_pair(row,byid) for row in inquiry["cross_world_comparisons"]]
    require(inquiry["time_interpolation"] is False and
            inquiry["postguard_extrapolation"] is False,
            "no new fictional interpolation allowed")
    return dict(native_by_world={
        name:dict(samples=len(byid[name]["native"]["times"]),
            max_pos_error_m=byid[name]["native"]["max_pos"],
            max_vel_error_m_s=byid[name]["native"]["max_vel"],
            max_force_error_n=byid[name]["native"]["max_force"])
        for name in ORDER},pairs=pairs,source_callbacks_imported=0,
        new_event_studies=4)


def negative_controls(source,inquiry):
    mutations=(
        ("false-zero-time",lambda i: i["cross_world_comparisons"][0]
            ["exact_time_trajectory_comparisons"][0].__setitem__("time_s",False)),
        ("fake-shift",lambda i: i["cross_world_comparisons"][0]
            ["event_time_comparisons"][0].__setitem__("estimated_numerical_shift_s",.9)),
        ("fake-native-position",lambda i: i["nodes"][1]["native_event_run"]
            ["trajectory"]["samples"][3]["position_m"].__setitem__(1,.5)),
        ("fake-native-root",lambda i: i["nodes"][2]["native_event_run"]
            ["events"][0]["sample"].__setitem__("time_s",.08)),
        ("fake-source-hash",lambda i: i["plan"].__setitem__(
            "source_graph_sha256","fake-sha256")),
        ("fake-ancestry",lambda i: i["nodes"][3]["source_ancestry"].__setitem__(
            "world_path",["baseline","beta","heavier"])),
    )
    for label,edit in mutations:
        forged=copy.deepcopy(inquiry)
        edit(forged)
        try:
            verify(source,forged)
        except (ValueError,AssertionError,KeyError,TypeError):
            continue
        raise AssertionError("injected fake evidence wrongly passed: "+label)
    return len(mutations)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-graph",type=Path,required=True)
    parser.add_argument("--inquiry",type=Path,required=True)
    parser.add_argument("--output",type=Path,
                        default=Path("output/m33b-independent.json"))
    args=parser.parse_args()
    graph_data=json.loads(args.source_graph.read_text(encoding="utf-8"))
    inquiry_data=json.loads(args.inquiry.read_text(encoding="utf-8"))
    result=verify(graph_data,inquiry_data)
    result["rejected_tamper_controls"]=negative_controls(graph_data,inquiry_data)
    result["source_graph_file_sha256"]=hashlib.sha256(
        args.source_graph.read_bytes()).hexdigest()
    result["inquiry_file_sha256"]=hashlib.sha256(
        args.inquiry.read_bytes()).hexdigest()
    result["verdict"]="PASS"
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,allow_nan=False,indent=2)+"\n",
                           encoding="utf-8")
    print(json.dumps({"verdict":"PASS",
        "native_worlds":len(result["native_by_world"]),
        "new_event_studies":result["new_event_studies"],
        "paired_cross_world_inquiries":len(result["pairs"]),
        "tamper_controls":result["rejected_tamper_controls"],
        "output":str(args.output)},sort_keys=True))


if __name__=="__main__":
    main()
