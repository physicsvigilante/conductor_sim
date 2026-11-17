"""M34a independent-form SI oracle for genuinely reassembled oblique worlds.

NO conductor_sim imports or production force callbacks. Computes q ΣE + mg,
p(t), v(t), numerical detector root and full lineage from actual detached
native event results. Finite bounded reference, not exact-ODE error bounds.
"""
from __future__ import annotations
import argparse
import copy
from hashlib import sha256
import json
from math import dist, fsum, hypot, isclose, isfinite, sqrt, ulp
from pathlib import Path

ORDER=("baseline","add-a","mass","replace-a","sibling")
PARENTS=(None,"baseline","add-a","mass","baseline")
EXPECTED={
 "baseline":dict(E=(320.,-80.,160.),mass=1.6e-6,active=("field:base",),
                 seen=("field:base",),path=("baseline",)),
 "add-a":dict(E=(410.,-45.,320.),mass=1.6e-6,
              active=("field:base","field:aux"),
              seen=("field:base","field:aux"),
              path=("baseline","add-a")),
 "mass":dict(E=(410.,-45.,320.),mass=2.2e-6,
             active=("field:base","field:aux"),
             seen=("field:base","field:aux"),
             path=("baseline","add-a","mass")),
 "replace-a":dict(E=(450.,-90.,270.),mass=2.2e-6,
                  active=("field:base","field:final"),
                  seen=("field:base","field:aux","field:final"),
                  path=("baseline","add-a","mass","replace-a")),
 "sibling":dict(E=(330.,-65.,200.),mass=1.6e-6,
                active=("field:base","field:aux"),
                seen=("field:base","field:aux"),
                path=("baseline","sibling")),
}


def require(check,message):
    if not check:
        raise AssertionError(message)


def val(value):
    require(type(value) in (float,int) and isfinite(value),"nonfinite/Boolean SI value")
    return float(value)


def vec(v):
    require(type(v) in (list,tuple) and len(v)==3,"missing physical SI 3-vector")
    return tuple(val(x) for x in v)


def near(value,expected,threshold,message):
    require(isclose(val(value),expected,rel_tol=0.,abs_tol=threshold),
            message+f": {value!r} instead of {expected!r}")


def vecnear(a,b,tol,message):
    for k,(value,expect) in enumerate(zip(vec(a),b)):
        near(value,expect,tol,message+f" component {k}")


def native_model(model,name):
    exp=EXPECTED[name]
    spec=model["spec"]
    require(type(spec["plane_normal"]) in (list,tuple),"plane normal absent")
    normal=vec(spec["plane_normal"])
    original=vec([x/hypot(1.,2.,-1.) for x in (1.,2.,-1.)])
    require(normal==original,"already qualified exact plane-normal bits changed")
    require(hypot(*normal)==1.+ulp(1.),"normal should display original one-ULP drift")
    q=val(spec["particle"]["charge_c"])
    mass=val(spec["mass_kg"])
    near(q,2e-9,1e-20,name+" physical charge")
    near(mass,exp["mass"],1e-18,name+" physical mass")
    vecnear(spec["particle"]["position_m"],(.003,.012,.06),1e-15,name+" original position")
    vecnear(spec["initial_velocity_m_s"],(.25,-.125,0.),1e-15,name+" true initial velocity")
    vecnear(spec["gravity"]["acceleration_m_s2"],(.16,-.04,.08),1e-15,name+" gravity")
    near(spec["duration_s"],.6,1e-15,name+" actual duration")
    require(all(s["kind"]=="UniformElectricField" for s in spec["field"]["sources"]),
            name+" unexpected source kind")
    e=tuple(fsum(vec(s["field_v_m"])[k] for s in spec["field"]["sources"])
            for k in range(3))
    vecnear(e,exp["E"],1e-12,name+" authored source total E")
    require(tuple(row[0] for row in model["source_annotations"])==exp["active"],
            name+" source ID membership altered")
    a=tuple(q*e[k]/mass+(.16,-.04,.08)[k] for k in range(3))
    f=tuple(q*e[k]+mass*(.16,-.04,.08)[k] for k in range(3))
    require(abs(a[0]+2*a[1]-a[2])<1e-13,
            name+" independently computed force no longer in certified plane")
    return dict(a=a,f=f,q=q,mass=mass,e=e,r0=(.003,.012,.06),v0=(.25,-.125,0.))


def check_source_cert(cert,model,name,parent_model=None):
    exp=EXPECTED[name]
    require(cert["schema"]=="conductor_sim.source_world_ancestry.v1",
            name+" missing genuine retained source ancestry")
    require(cert["world_id"]==name and
            cert["parent_world_id"]==PARENTS[ORDER.index(name)],
            name+" wrong immediate parent world")
    require(cert["model_revision_id"]==model["revision_id"],
            name+" model revision SHA missing from certificate")
    require(tuple(cert["active_source_ids"])==exp["active"] and
            tuple(cert["seen_source_ids"])==exp["seen"] and
            tuple(cert["world_path"])==exp["path"],
            name+" inactive/active ID history mismatch")
    require(tuple(cert["retired_source_ids"])==
            tuple(x for x in exp["seen"] if x not in exp["active"]),
            name+" retired ID fabricated or hidden")
    require(cert["gravity_present"] is True,name+" gravity membership lost")
    if parent_model is not None:
        require(cert["parent_revision_id"]==parent_model["revision_id"],
                name+" source parent SHA disagrees with immediate real model")
    if name=="replace-a":
        require(cert["introduced_source_ids"]==["field:final"] and
                cert["retired_in_this_step"]==["field:aux"],
                name+" real replacement ledger altered")
    if name=="mass":
        require(not cert["introduced_source_ids"] and
                not cert["retired_in_this_step"],
                name+" plain mass edit fabricated source membership")


def state(physics,t):
    a=physics["a"]
    p=tuple(physics["r0"][k]+physics["v0"][k]*t+.5*a[k]*t*t
            for k in range(3))
    v=tuple(physics["v0"][k]+a[k]*t for k in range(3))
    return p,v,physics["f"]


def root_t(phys):
    x0=phys["r0"][0]
    speed=phys["v0"][0]
    ax=phys["a"][0]
    dx=.075-x0
    return 2*dx/(speed+sqrt(speed*speed+2*ax*dx))


def check_native(native,model,name,phys):
    require(native["schema"]=="conductor_sim.event_run.v1",
            name+" missing genuine M24a EventRun")
    require(native["study"]["model_revision_id"]==model["revision_id"],
            name+" native study not tied to actual qualified source model")
    require(native["trajectory"]["termination"]=="duration",
            name+" trajectory did not complete requested horizon")
    require(native["outcomes"]["x:detector"]=="located",
            name+" genuine signed detector unlocated")
    samples=native["trajectory"]["samples"]
    require(len(samples)>=10,name+" too few genuine SI saved samples")
    max_p=max_v=max_f=0.
    times=[]
    for item in samples:
        require(type(item["time_s"]) is float,
                name+" Boolean or fake native clock")
        t=val(item["time_s"])
        require(0<=t<=.6,name+" saved clock outside physical horizon")
        times.append(t)
        p,v,f=state(phys,t)
        measured_p=vec(item["position_m"])
        measured_v=vec(item["velocity_m_s"])
        measured_f=vec(item["force_n"])
        max_p=max(max_p,*(abs(measured_p[k]-p[k]) for k in range(3)))
        max_v=max(max_v,*(abs(measured_v[k]-v[k]) for k in range(3)))
        max_f=max(max_f,*(abs(measured_f[k]-f[k]) for k in range(3)))
    require(times[0]==0.0 and abs(times[-1]-.6)<1e-10 and
            all(b>a for a,b in zip(times,times[1:])),
            name+" SI saved physical clock is not complete and increasing")
    require(max_p<5e-8 and max_v<5e-8 and max_f<3e-12,
            name+" independent Newtonian position/velocity/force comparison failed")
    roots=[e for e in native["events"] if e["event_id"]=="x:detector"]
    require(len(roots)==1 and roots[0]["direction"]==1,
            name+" genuine unique positive root missing")
    numerical=val(roots[0]["sample"]["time_s"])
    bracket=tuple(map(val,roots[0]["time_bracket_s"]))
    true=root_t(phys)
    require(type(roots[0]["sample"]["time_s"]) is float and
            bracket[0]<=numerical<=bracket[1] and
            abs(numerical-true)<5e-7,
            name+" numerical detector root diverges from analytic x(t)")
    near(roots[0]["sample"]["position_m"][0],.075,5e-7,
         name+" native detector point not on authored x plane")
    return dict(samples=len(samples),max_position_error_m=max_p,
                max_velocity_error_m_s=max_v,max_force_error_n=max_f,
                root=numerical,independent_root=true,times=times,bracket=bracket)


def verify(graph):
    require(graph["schema"]=="conductor_sim.unified_observation_graph.v2",
            "M34a expected retained-ancestry native graph v2")
    require(graph["plan_id"]=="m34a:oblique" and
            graph["plan"]["retain_source_ancestry"] is True and
            graph["time_interpolation"] is False and
            graph["postguard_extrapolation"] is False,
            "changed input plan/source physics or interpolation")
    require(len(graph["nodes"])==4,"M34a must have four genuine altered source worlds")
    models={"baseline":graph["plan"]["original"]}
    native={"baseline":graph["original_native_observation"]}
    check_source_cert(graph["original_source_ancestry"],models["baseline"],
                      "baseline")
    for index,name in enumerate(ORDER[1:],0):
        row=graph["nodes"][index]
        parent=PARENTS[index+1]
        require(row["node"]["node_id"]==name and
                row["parent_id"]==parent and row["status"]=="complete",
                name+" expected actual admitted physical world")
        model=row["admitted_scenario"]
        require(model is not None and row["native_observation"] is not None,
                name+" fabricated source/observations")
        require(row["parent_revision_id"]==models[parent]["revision_id"],
                name+" world immediate parent SHA wrong")
        if name=="mass":
            require(model["parent_revision_id"]==models[parent]["revision_id"],
                    name+" physical M20b mass reassembly parent mismatched")
        else:
            assembly=row["structural_assembly"]
            require(assembly is not None and
                    assembly["parent_revision_id"]==models[parent]["revision_id"],
                    name+" structural M22a parent proof missing")
        check_source_cert(row["source_ancestry"],model,name,models[parent])
        models[name]=model
        native[name]=row["native_observation"]
    results={}
    for name in ORDER:
        phys=native_model(models[name],name)
        results[name]=check_native(native[name],models[name],name,phys)
    for index,name in enumerate(ORDER[1:]):
        row=graph["nodes"][index]
        parent=PARENTS[index+1]
        pairs=row["parent_event_time_comparisons"]
        require(len(pairs)==1 and pairs[0]["status"]=="one-resolved-pair",
                name+" parent event not paired")
        expect=results[name]["root"]-results[parent]["root"]
        near(pairs[0]["estimated_numerical_shift_s"],expect,1e-12,
             name+" native saved numerical root shift")
    return dict(
        scientific_reference="separate 3-component SI Newton qΣE/m+g, complete saved trajectories and signed x roots",
        source_callbacks_imported=0,accepted_native_worlds=len(ORDER),
        true_qualified_parent_and_ancestry=True,
        per_world={name:{key:value for key,value in results[name].items()
                         if key not in ("times","bracket")}
                   for name in ORDER})


def adversarial_controls(native):
    attacks=(
        ("normal-drift",lambda d: d["nodes"][2]["admitted_scenario"]["spec"]
            ["plane_normal"].__setitem__(0,.40)),
        ("false-clock",lambda d: d["nodes"][0]["native_observation"]
            ["trajectory"]["samples"][0].__setitem__("time_s",False)),
        ("false-event-root",lambda d: d["nodes"][1]["native_observation"]
            ["events"][0]["sample"].__setitem__("time_s",.55)),
        ("fake-ancestry",lambda d: d["nodes"][2]["source_ancestry"]
            .__setitem__("retired_source_ids",[])),
        ("fake-source-parent",lambda d: d["nodes"][2]
            .__setitem__("parent_revision_id","FAKE")),
        ("fake-field-force",lambda d: d["nodes"][3]["native_observation"]
            ["trajectory"]["samples"][3]["force_n"].__setitem__(2,10.)),
    )
    for label,attack in attacks:
        fake=copy.deepcopy(native)
        attack(fake)
        try:
            verify(fake)
        except (AssertionError,KeyError,ValueError,TypeError):
            continue
        raise AssertionError("independent oracle did not reject tampering: "+label)
    return len(attacks)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--native",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=Path("output/m34a-independent.json"))
    args=parser.parse_args()
    raw=args.native.read_bytes()
    data=json.loads(raw)
    evidence=verify(data)
    evidence["tamper_controls_rejected"]=adversarial_controls(data)
    evidence["native_json_sha256"]=sha256(raw).hexdigest()
    evidence["verdict"]="PASS"
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(evidence,allow_nan=False,indent=2)+"\n",
                           encoding="utf-8")
    print(json.dumps(dict(
        verdict=evidence["verdict"],worlds=evidence["accepted_native_worlds"],
        refused_tampers=evidence["tamper_controls_rejected"],
        output=str(args.output)),sort_keys=True))


if __name__=="__main__":
    main()
