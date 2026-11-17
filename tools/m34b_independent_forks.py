"""M34b separately expressed Newtonian SI and branch source-identity oracle.

Imports NO conductor_sim source code. Reads genuine source graph and new
incremental native fork as detached evidence, independently derives qΣE/m
saved 3D states and real M24a positive detector roots, checks all literal
shared native SI timestamps, source model hashes/ancestry, and adversarial
mutation controls. A finite numerical reference, not external signing,
source-model authentication or a general ODE truncation-error theorem.
"""
from __future__ import annotations

import argparse
import copy
from hashlib import sha256
import json
from math import dist, fsum, isclose, isfinite, sqrt
from pathlib import Path


EXPECTED = {
    "baseline": ((100., 50., 0.), 1e-5,
                 ("field:E0", "field:Ey"), ("field:E0", "field:Ey"),
                 ("baseline",), None),
    "old-a": ((140., 50., 0.), 1e-5,
              ("field:E0", "field:Ey", "field:alt"),
              ("field:E0", "field:Ey", "field:alt"),
              ("baseline", "old-a"), "baseline"),
    "old-b": ((220., 50., 0.), 1e-5,
              ("field:E0", "field:Ey", "field:alt"),
              ("field:E0", "field:Ey", "field:alt"),
              ("baseline", "old-b"), "baseline"),
    "mass-a": ((140., 50., 0.), 2e-5,
               ("field:E0", "field:Ey", "field:alt"),
               ("field:E0", "field:Ey", "field:alt"),
               ("baseline", "old-a", "mass-a"), "old-a"),
    "replace-a": ((190., 50., 0.), 2e-5,
                  ("field:E0", "field:Ey", "field:replacement"),
                  ("field:E0", "field:Ey", "field:alt", "field:replacement"),
                  ("baseline", "old-a", "mass-a", "replace-a"), "mass-a"),
    "remove-a": ((100., 50., 0.), 2e-5,
                 ("field:E0", "field:Ey"),
                 ("field:E0", "field:Ey", "field:alt", "field:replacement"),
                 ("baseline", "old-a", "mass-a", "replace-a", "remove-a"), "replace-a"),
    "mass-b": ((220., 50., 0.), 3e-5,
               ("field:E0", "field:Ey", "field:alt"),
               ("field:E0", "field:Ey", "field:alt"),
               ("baseline", "old-b", "mass-b"), "old-b"),
}
SOURCE_IDS = ("baseline", "old-a", "old-b")
ADMITTED_FORK_IDS = ("mass-a", "replace-a", "remove-a", "mass-b")
ALL_IDS = SOURCE_IDS + ADMITTED_FORK_IDS
PAIRS = (
    ("old-to-new", "old-a", "mass-a", "compared"),
    ("cross-branches", "old-b", "replace-a", "compared"),
    ("new-branches", "remove-a", "mass-b", "compared"),
    ("refused", "resurrect", "mass-b", "unpaired"),
)


def assert_true(condition, reason):
    if not condition:
        raise AssertionError(reason)


def num(value):
    assert_true(type(value) in (float, int) and isfinite(value),
                "non-native or nonfinite physical scalar, including False == 0.0")
    return float(value)


def triple(value):
    assert_true(type(value) in (tuple, list) and len(value) == 3,
                "missing actual three-component SI vector")
    return tuple(num(x) for x in value)


def close(actual, target, tolerance, reason):
    assert_true(isclose(num(actual), target, rel_tol=0., abs_tol=tolerance),
                reason + f": {actual!r} vs {target!r}")


def vector_close(actual, target, tolerance, reason):
    for i, (measured, expected) in enumerate(zip(triple(actual), target)):
        close(measured, expected, tolerance, reason + " axis " + str(i))


def canonical_sha(record):
    return sha256(json.dumps(record, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False
                             ).encode("ascii")).hexdigest()


def check_cert(cert, model, world):
    expected_field, mass, active, seen, path, parent = EXPECTED[world]
    assert_true(cert["schema"] == "conductor_sim.source_world_ancestry.v1",
                world + " missing native source-world certificate")
    assert_true(cert["world_id"] == world and cert["parent_world_id"] == parent,
                world + " source cert parent-world mismatch")
    assert_true(cert["model_revision_id"] == model["revision_id"],
                world + " source cert model hash mismatch")
    assert_true(tuple(cert["world_path"]) == path and
                tuple(cert["active_source_ids"]) == active and
                tuple(cert["seen_source_ids"]) == seen,
                world + " branch-local world/active/seen source paths forged")
    assert_true(tuple(cert["retired_source_ids"]) == tuple(
        identity for identity in seen if identity not in active),
                world + " retired source IDs false")
    assert_true(cert["gravity_present"] is False,
                world + " invented uniform gravity")
    if world == "mass-a" or world == "mass-b":
        assert_true(cert["introduced_source_ids"] == [] and
                    cert["retired_in_this_step"] == [],
                    world + " physical mass change forged source membership")
    elif world == "replace-a":
        assert_true(cert["introduced_source_ids"] == ["field:replacement"] and
                    cert["retired_in_this_step"] == ["field:alt"],
                    world + " genuine replacement and retirement missing")
    elif world == "remove-a":
        assert_true(cert["introduced_source_ids"] == [] and
                    cert["retired_in_this_step"] == ["field:replacement"],
                    world + " genuine removal did not retire replacement")


def model_physics(model,world):
    electric,mass,active,seen,path,parent = EXPECTED[world]
    spec = model["spec"]
    assert_true(spec.get("gravity") is None,
                world + " imported gravity in electric-only teacher example")
    assert_true(all(src["kind"] == "UniformElectricField"
                    for src in spec["field"]["sources"]),
                world + " non-uniform field invalidates independent formula")
    field = tuple(fsum(triple(src["field_v_m"])[k]
                       for src in spec["field"]["sources"]) for k in range(3))
    vector_close(field,electric,1e-12,world+" authored E")
    assert_true(tuple(i for i,label in model["source_annotations"]) == active,
                world+" stable source IDs no longer match model")
    close(spec["particle"]["charge_c"],1e-9,1e-20,world+" q")
    close(spec["mass_kg"],mass,1e-16,world+" mass")
    vector_close(spec["particle"]["position_m"],(0.,0.,0.),1e-15,world+" launch")
    vector_close(spec["initial_velocity_m_s"],(.1,0.,0.),1e-15,world+" initial velocity")
    close(spec["duration_s"],.1,1e-15,world+" duration")
    vector_close(spec["plane_normal"],(0.,0.,1.),1e-15,world+" qualified plane")
    charge = num(spec["particle"]["charge_c"])
    mass = num(spec["mass_kg"])
    return dict(
        e=field, q=charge, mass=mass,
        a=tuple(charge*field[k]/mass for k in range(3)),
        v0=(.1,0.,0.), r0=(0.,0.,0.), f=tuple(charge*x for x in field),
        revision=model["revision_id"],
        source_ids=active,
    )


def exact_state(phys,t):
    p = tuple(phys["r0"][k] + phys["v0"][k]*t + .5*phys["a"][k]*t*t
              for k in range(3))
    v = tuple(phys["v0"][k] + phys["a"][k]*t for k in range(3))
    return p,v,phys["f"]


def independent_root(phys):
    dx=.006 - phys["r0"][0]
    v=phys["v0"][0]
    ax=phys["a"][0]
    return 2*dx/(v+sqrt(v*v+2*ax*dx))


def verify_native(native, phys,world):
    assert_true(native["schema"] == "conductor_sim.event_run.v1",
                world+" missing genuine M24a event result")
    assert_true(native["study"]["model_revision_id"] == phys["revision"],
                world+" actual native event source revision changed")
    assert_true(native["study"]["questions"][0]["event_id"] == "x:detector" and
                native["study"]["questions"][0]["direction"] == 1 and
                len(native["study"]["questions"]) == 1,
                world+" event questions are not the common original physics study")
    assert_true(native["trajectory"]["termination"] == "duration",
                world+" native observation did not complete actual horizon")
    assert_true(native["outcomes"]["x:detector"] == "located",
                world+" no genuine resolved detector")
    samples=native["trajectory"]["samples"]
    assert_true(len(samples)>=3,world+" no real saved M24a trajectory states")
    times=[]
    max_p=max_v=max_f=0.
    for sample in samples:
        assert_true(type(sample["time_s"]) is float,
                    world+" nonnative Boolean/int SI timestamp")
        t=num(sample["time_s"])
        assert_true(0. <= t <= .1,world+" native time out of physical horizon")
        times.append(t)
        p,v,f=exact_state(phys,t)
        measured_p=triple(sample["position_m"])
        measured_v=triple(sample["velocity_m_s"])
        measured_f=triple(sample["force_n"])
        max_p=max(max_p,*(abs(measured_p[k]-p[k]) for k in range(3)))
        max_v=max(max_v,*(abs(measured_v[k]-v[k]) for k in range(3)))
        max_f=max(max_f,*(abs(measured_f[k]-f[k]) for k in range(3)))
    assert_true(times[0] == 0. and abs(times[-1]-.1)<1e-10 and
                all(b>a for a,b in zip(times,times[1:])),
                world+" invalid non-monotonic/incomplete source-physical time sequence")
    assert_true(max_p < 5e-8 and max_v < 5e-8 and max_f < 1e-13,
                world+" measured physical samples disagree with independent Newtonian SI")
    roots=[e for e in native["events"] if e["event_id"]=="x:detector"]
    assert_true(len(roots)==1 and roots[0]["direction"]==1,
                world+" expected one true positive detector crossing")
    root=roots[0]
    numeric=num(root["sample"]["time_s"])
    bounds=tuple(num(t) for t in root["time_bracket_s"])
    assert_true(type(root["sample"]["time_s"]) is float and
                bounds[0] <= numeric <= bounds[1],
                world+" fake clock or numerical bracket")
    true_t=independent_root(phys)
    close(numeric,true_t,2e-7,world+" independent signed x detector time")
    close(root["sample"]["position_m"][0],.006,2e-7,world+" root position on actual detector")
    return dict(times=times,native_root=numeric,expected_root=true_t,
                bracket=bounds,max_position_error=max_p,max_velocity_error=max_v,
                max_force_error=max_f)


def check_pairs(actual, records):
    assert_true(len(records)==len(PAIRS),"number of independent requested pairs differs")
    summary=[]
    for rec,expected in zip(records,PAIRS):
        name,lid,rid,status=expected
        assert_true((rec["comparison_id"],rec["left_world_id"],
                     rec["right_world_id"],rec["status"])==expected,
                    name+" identity/status/orientation no longer matches request")
        assert_true(rec["numerical_orientation"]=="right minus left",
                    name+" signed physical difference orientation changed")
        if status=="unpaired":
            assert_true(rec["left_revision_id"] is None and
                        rec["left_source_ancestry"] is None and
                        not rec["exact_time_trajectory_comparisons"] and
                        all(e["status"] == "unpaired" for e in rec["event_time_comparisons"]),
                        name+" physically refused world fabricated source or event data")
            continue
        l=actual[lid]
        r=actual[rid]
        assert_true(rec["left_revision_id"]==l["phys"]["revision"] and
                    rec["right_revision_id"]==r["phys"]["revision"],
                    name+" source model hashes mismatched side identities")
        check_cert(rec["left_source_ancestry"],l["model"],lid)
        check_cert(rec["right_source_ancestry"],r["model"],rid)
        assert_true(tuple(x["id"] for x in rec["left_source_inventory"]) ==
                    EXPECTED[lid][2] and
                    tuple(x["id"] for x in rec["right_source_inventory"]) ==
                    EXPECTED[rid][2],
                    name+" side-specific held source inventory reordered")
        assert_true(rec["event_study_compatible"] is True,
                    name+" expected same true original/new native M24a policy")
        ltimes=set(l["native"]["times"])
        matched=[t for t in r["native"]["times"] if t in ltimes]
        rows=rec["exact_time_trajectory_comparisons"]
        assert_true(len(rows)==len(matched) and rows,
                    name+" missing actual literally equal saved SI timestamps")
        largest=0.
        for row,t in zip(rows,matched):
            assert_true(type(row["time_s"]) is float and row["time_s"]==t,
                        name+" fabricated/rounded/Boolean native physical time")
            lp,lv,_=exact_state(l["phys"],t)
            rp,rv,_=exact_state(r["phys"],t)
            dp=tuple(rp[k]-lp[k] for k in range(3))
            dv=tuple(rv[k]-lv[k] for k in range(3))
            vector_close(row["delta_position_m"],dp,5e-8,name+" position difference")
            vector_close(row["delta_velocity_m_s"],dv,5e-8,name+" velocity difference")
            close(row["position_distance_m"],dist(lp,rp),5e-8,name+" 3D position norm")
            close(row["velocity_distance_m_s"],dist(lv,rv),5e-8,name+" 3D velocity norm")
            largest=max(largest,*(abs(triple(row["delta_position_m"])[k]-dp[k])
                                for k in range(3)))
        shifts=rec["event_time_comparisons"]
        assert_true(len(shifts)==1 and shifts[0]["event_id"]=="x:detector" and
                    shifts[0]["status"]=="one-resolved-pair" and
                    shifts[0]["direction"]==1,
                    name+" missing genuinely unique same-direction whole-duration M24a roots")
        s=shifts[0]
        true=r["native"]["expected_root"]-l["native"]["expected_root"]
        estimated=r["native"]["native_root"]-l["native"]["native_root"]
        close(s["estimated_numerical_shift_s"],true,4e-7,name+" independent event difference")
        close(s["estimated_numerical_shift_s"],estimated,1e-12,name+" actual M24a clock difference")
        interval=(r["native"]["bracket"][0]-l["native"]["bracket"][1],
                  r["native"]["bracket"][1]-l["native"]["bracket"][0])
        for a,b in zip(s["numerical_bracket_shift_interval_s"],interval):
            close(a,b,1e-12,name+" true numerical root interval arithmetic")
        summary.append(dict(id=name,shared_saved_times=len(rows),
                            max_position_delta_error_m=largest,
                            signed_numerical_shift_s=estimated,
                            independent_event_shift_s=true))
    return summary


def verify(old, fork):
    assert_true(old["schema"]=="conductor_sim.unified_observation_graph.v2" and
                old["plan"]["schema"]=="conductor_sim.unified_observation_plan.v2",
                "original native graph not true retained-source-certified v2")
    assert_true(old["plan_id"]=="m34b:original" and
                old["original_observation_mode"]=="events",
                "wrong saved original source graph")
    assert_true(fork["schema"]=="conductor_sim.graph_world_fork.v1" and
                fork["plan"]["schema"]=="conductor_sim.graph_world_fork_plan.v1",
                "not bounded new certified incremental fork")
    assert_true(fork["plan"]["source_graph_sha256"]==canonical_sha(old) and
                fork["source_graph_sha256"]==canonical_sha(old),
                "new incremental output not bound to complete original source proof")
    assert_true(fork["plan"]["fork_id"]=="m34b:incremental" and
                fork["plan"]["source_graph_plan_id"]=="m34b:original" and
                fork["plan"]["source_graph_schema"]==old["schema"],
                "new fork teacher and actual original source roots changed")
    assert_true(len(old["nodes"])==2 and len(fork["nodes"])==5,
                "not two original siblings plus five genuinely new alternatives")
    source_records=[("baseline",old["plan"]["original"],
                     old["original_native_observation"],
                     old["original_source_ancestry"])]
    for name,rec in zip(SOURCE_IDS[1:],old["nodes"]):
        assert_true(rec["node"]["node_id"]==name and rec["status"]=="complete" and
                    rec["native_observation"] is not None,
                    name+" originally admitted source world missing")
        source_records.append((name,rec["admitted_scenario"],
                               rec["native_observation"],rec["source_ancestry"]))
    actual={}
    for name,model,native,cert in source_records:
        check_cert(cert,model,name)
        phys=model_physics(model,name)
        native_result=verify_native(native,phys,name)
        actual[name]=dict(model=model,phys=phys,native=native_result)
    ordered=("mass-a","replace-a","remove-a","resurrect","mass-b")
    for index,(name,row) in enumerate(zip(ordered,fork["nodes"]),1):
        assert_true(row["index"]==index and row["node"]["node_id"]==name,
                    name+" requested fork physical order was changed")
        parent=row["node"]["parent_id"]
        expected_parent=(None if name=="resurrect" else EXPECTED[name][5])
        if name=="resurrect":
            assert_true(parent=="remove-a" and
                        row["status"]=="rejected-structural-model" and
                        row["admitted_scenario"] is None and
                        row["native_observation"] is None and
                        row["source_ancestry"] is None,
                        "forbidden retired source ID was resurrected or observed")
            continue
        assert_true(parent==expected_parent and row["status"]=="complete",
                    name+" newly authored native parent/refusal changed")
        model=row["admitted_scenario"]
        native=row["native_observation"]
        assert_true(model is not None and native is not None and
                    row["parent_revision_id"] == actual[parent]["phys"]["revision"],
                    name+" fictitious or wrong actual immediate source model")
        check_cert(row["source_ancestry"],model,name)
        if name in ("mass-a","mass-b"):
            assert_true(model["parent_revision_id"]==
                        actual[parent]["phys"]["revision"] and
                        row["structural_assembly"] is None,
                        name+" physical mass revision does not bind native parent")
        else:
            assert_true(row["structural_assembly"]["parent_revision_id"]==
                        actual[parent]["phys"]["revision"],
                        name+" M22a native source reassembly lost true parent")
        phys=model_physics(model,name)
        native_result=verify_native(native,phys,name)
        actual[name]=dict(model=model,phys=phys,native=native_result)
    assert_true(fork["time_interpolation"] is False and
                fork["postguard_extrapolation"] is False,
                "source-fork fabricated physical time or afterguard motion")
    pairs=check_pairs(actual,fork["cross_world_comparisons"])
    return dict(verified_source_worlds=len(SOURCE_IDS),
                verified_new_admitted_worlds=len(ADMITTED_FORK_IDS),
                refused_retired_resurrection=True,
                comparisons=pairs,
                per_world={
                    name: {k: v for k,v in actual[name]["native"].items()
                           if k not in ("times","bracket")}
                    for name in ALL_IDS},
                imported_conductor_sim_modules=0)


def negative_controls(old, fork):
    edits=(
        ("false-source-digest", lambda d: d.__setitem__("source_graph_sha256","FAKE")),
        ("falsified-parent",lambda d: d["nodes"][1].__setitem__("parent_revision_id","FAKE")),
        ("resurrected-model",lambda d: d["nodes"][3].__setitem__("status","complete")),
        ("false-clock",lambda d: d["nodes"][0]["native_observation"]["trajectory"]
            ["samples"][0].__setitem__("time_s",False)),
        ("forged-native-event",lambda d: d["nodes"][0]["native_observation"]
            ["events"][0]["sample"].__setitem__("time_s",.09)),
        ("wrong-signed-delta",lambda d: d["cross_world_comparisons"][1]
            ["exact_time_trajectory_comparisons"][1]
            ["delta_position_m"].__setitem__(0,.3)),
        ("fake-source-retirement",lambda d: d["nodes"][1]["source_ancestry"]
            .__setitem__("retired_source_ids",[])),
    )
    for name,attack in edits:
        forged=copy.deepcopy(fork)
        attack(forged)
        try:
            verify(old,forged)
        except (AssertionError,TypeError,ValueError,KeyError):
            continue
        raise AssertionError("detached physical evidence alteration missed: "+name)
    return len(edits)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--fork",type=Path,required=True)
    p.add_argument("--output",type=Path,
                   default=Path("output/m34b-independent-reference.json"))
    args=p.parse_args()
    raw_old=args.original.read_bytes()
    raw_fork=args.fork.read_bytes()
    source=json.loads(raw_old)
    fork=json.loads(raw_fork)
    result=verify(source,fork)
    result["tamper_controls_rejected"]=negative_controls(source,fork)
    result["original_file_sha256"]=sha256(raw_old).hexdigest()
    result["fork_file_sha256"]=sha256(raw_fork).hexdigest()
    result["verdict"]="PASS"
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,allow_nan=False,indent=2)+"\n",
                           encoding="utf-8")
    print(json.dumps(dict(
        verdict="PASS",source_worlds=result["verified_source_worlds"],
        new_admitted_worlds=result["verified_new_admitted_worlds"],
        comparisons=len(result["comparisons"]),
        refused_tamper_controls=result["tamper_controls_rejected"],
        output=str(args.output)),sort_keys=True))


if __name__=="__main__":
    main()
