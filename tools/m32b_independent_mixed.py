"""Independent M32b v2 mixed-order source ancestry and SI Newtonian reference.

NO conductor_sim imports. No production Coulomb/field callback, RK or native
event locator is reused. This bounded reference analytically recomputes all
genuine saved 3D state/force samples and signed detector crossings of a
teacher-authored uniform-field structural→physical→structural→physical graph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from math import fsum, sqrt, isfinite
from pathlib import Path


WORLDS = ("baseline","add-E","source-mass","replace-after","final")
PARENTS = (None,"baseline","add-E","source-mass","replace-after")
ACTIVE = {
    "baseline": ("field:E0","field:Ey"),
    "add-E": ("field:E0","field:Ey","field:Enew"),
    "source-mass": ("field:E0","field:Ey","field:Enew"),
    "replace-after": ("field:E0","field:Ey","field:Eafter"),
    "final": ("field:E0","field:Ey","field:Eafter"),
}
SEEN = {
    "baseline": ("field:E0","field:Ey"),
    "add-E": ("field:E0","field:Ey","field:Enew"),
    "source-mass": ("field:E0","field:Ey","field:Enew"),
    "replace-after": ("field:E0","field:Ey","field:Enew","field:Eafter"),
    "final": ("field:E0","field:Ey","field:Enew","field:Eafter"),
}
EXPECTED = {
    "baseline": (100.,50.,1e-5,.1),
    "add-E": (200.,50.,1e-5,.1),
    "source-mass": (250.,50.,2e-5,.1),
    "replace-after": (230.,50.,2e-5,.1),
    "final": (280.,50.,2e-5,.16),
}


def _native_model(model,label):
    spec=model["spec"]
    if spec.get("gravity") is not None:
        raise AssertionError(label+" analytic uniform transfer assumes absent gravity")
    if any(s["kind"]!="UniformElectricField" for s in spec["field"]["sources"]):
        raise AssertionError(label+" uses a source not in qualified uniform reference")
    e=tuple(fsum(s["field_v_m"][k] for s in spec["field"]["sources"])
            for k in range(3))
    mass=spec["mass_kg"]
    q=spec["particle"]["charge_c"]
    pos=tuple(spec["particle"]["position_m"])
    vel=tuple(spec["initial_velocity_m_s"])
    if q != 1e-9 or mass<=0 or not all(isfinite(x) for x in (*e,*pos,*vel,mass,q)):
        raise AssertionError(label+" invalid SI numerical source/probe")
    x,y,m,vx=EXPECTED[label]
    if (abs(e[0]-x)>1e-12 or abs(e[1]-y)>1e-12 or
            abs(mass-m)>1e-15 or abs(vel[0]-vx)>1e-12):
        raise AssertionError(label+" actual field/mass/launch inconsistent with source plan")
    if tuple(x[0] for x in model["source_annotations"]) != ACTIVE[label]:
        raise AssertionError(label+" native source identities don't match physical model")
    return dict(
        mass=mass,charge=q,e=e,r0=pos,v0=vel,
        accel=tuple(q*component/mass for component in e),
        duration=spec["duration_s"],
    )


def _states_and_roots(model,native,label):
    p=_native_model(model,label)
    if native["trajectory"]["termination"]!="duration":
        raise AssertionError(label+" genuine event trajectory must complete duration")
    samples=native["trajectory"]["samples"]
    if len(samples)<3:
        raise AssertionError(label+" not enough native saved SI states")
    max_r=max_v=max_f=0.
    times=[]
    for row in samples:
        t=row["time_s"]
        if type(t) is not float or not isfinite(t) or not 0<=t<=p["duration"]:
            raise AssertionError(label+" forged SI time/Boolean sample")
        times.append(t)
        exact_r=tuple(p["r0"][k]+p["v0"][k]*t+.5*p["accel"][k]*t*t
                      for k in range(3))
        exact_v=tuple(p["v0"][k]+p["accel"][k]*t for k in range(3))
        exact_f=tuple(p["charge"]*p["e"][k] for k in range(3))
        max_r=max(max_r,*(abs(row["position_m"][k]-exact_r[k]) for k in range(3)))
        max_v=max(max_v,*(abs(row["velocity_m_s"][k]-exact_v[k]) for k in range(3)))
        max_f=max(max_f,*(abs(row["force_n"][k]-exact_f[k]) for k in range(3)))
    if any(b<=a for a,b in zip(times,times[1:])):
        raise AssertionError(label+" saved native physical times are non-monotonic")
    if times[0]!=0.0 or abs(times[-1]-p["duration"])>1e-11:
        raise AssertionError(label+" no original full duration")
    if max_r>2e-8 or max_v>2e-8 or max_f>1e-13:
        raise AssertionError(label+" native states failed separately expressed Newtonian physics")
    roots=[x for x in native["events"] if x["event_id"]=="x:detector"]
    if len(roots)!=1 or roots[0]["direction"] != 1:
        raise AssertionError(label+" failed one unique positive real x-plane root")
    root=roots[0]
    dx=.006-p["r0"][0]
    ax=p["accel"][0]
    vx=p["v0"][0]
    true_t=2*dx/(vx+sqrt(vx*vx+2*ax*dx))
    actual=root["sample"]["time_s"]
    lower,upper=root["time_bracket_s"]
    if type(actual) is not float or not (lower<=actual<=upper):
        raise AssertionError(label+" invalid native numerical event root bracket")
    if abs(actual-true_t)>2e-7 or not lower-2e-7<=true_t<=upper+2e-7:
        raise AssertionError(label+" observed root diverges from independent SI analytic detector")
    if abs(root["sample"]["position_m"][0]-.006)>1e-7:
        raise AssertionError(label+" genuine saved root state is not on x detector")
    return dict(
        samples=len(samples),max_position_error_m=max_r,
        max_velocity_error_m_s=max_v,max_force_error_n=max_f,
        native_root_s=actual,independent_root_s=true_t,
        native_minus_reference_s=actual-true_t,
        numerical_bracket_s=[lower,upper],
    )


def _check_source_ancestry(data,parent,label,model,ids,known):
    if data is None or data.get("schema")!="conductor_sim.source_world_ancestry.v1":
        raise AssertionError(label+" lacks real native M32b source ancestry certificate")
    if data["world_id"]!=label or data["parent_world_id"]!=parent:
        raise AssertionError(label+" wrong immediate parent in provenance certificate")
    if tuple(data["active_source_ids"])!=ids or tuple(data["seen_source_ids"])!=known:
        raise AssertionError(label+" loses previously active/retired source IDs")
    if tuple(data["retired_source_ids"])!=tuple(x for x in known if x not in ids):
        raise AssertionError(label+" retired source history incorrect")
    if data["model_revision_id"]!=model["revision_id"]:
        raise AssertionError(label+" certificate not bound to real model revision")
    if tuple(data["world_path"])[-1]!=label or tuple(data["revision_path"])[-1]!=model["revision_id"]:
        raise AssertionError(label+" mixed-order world or revision paths corrupted")
    if data["gravity_present"] is not False:
        raise AssertionError(label+" phantom gravity component")
    if "supplied graph root" not in data["history_scope"]:
        raise AssertionError(label+" hides bounded source-history trust boundary")


def verify(graph):
    if graph.get("schema")!="conductor_sim.unified_observation_graph.v2":
        raise AssertionError("must be M32b v2 record with retained graph ancestry")
    if graph["plan"]["schema"]!="conductor_sim.unified_observation_plan.v2":
        raise AssertionError("native M32b v2 plan missing")
    if graph["plan"].get("retain_source_ancestry") is not True:
        raise AssertionError("mixed-order world must explicitly opt in")
    if graph["time_interpolation"] is not False or graph["postguard_extrapolation"] is not False:
        raise AssertionError("native data cannot be authored interpolations")
    if len(graph["nodes"])!=4 or graph["plan_id"]!="m32b:mix":
        raise AssertionError("unexpected authored graph specification")

    baseline=graph["plan"]["original"]
    models={"baseline":baseline}
    _check_source_ancestry(graph["original_source_ancestry"],None,
                            "baseline",baseline,ACTIVE["baseline"],SEEN["baseline"])
    out={"baseline":_states_and_roots(baseline,graph["original_native_observation"],"baseline")}

    for idx,row in enumerate(graph["nodes"],1):
        label=WORLDS[idx]
        parent=PARENTS[idx]
        if row["node"]["node_id"]!=label or row["parent_id"]!=parent or row["status"]!="complete":
            raise AssertionError(label+" unexpected node identity/status")
        model=row["admitted_scenario"]
        if model is None or row["native_observation"] is None:
            raise AssertionError(label+" fabricated physical model or observation")
        if row["parent_revision_id"]!=models[parent]["revision_id"]:
            raise AssertionError(label+" real source-world parent hash wrong")
        if row["node"]["operation"]=="structural":
            assembly=row["structural_assembly"]
            if assembly is None or assembly["parent_revision_id"]!=models[parent]["revision_id"]:
                raise AssertionError(label+" source reassembly no longer bound to true parent")
        elif (model["parent_revision_id"]!=models[parent]["revision_id"] or
              row["structural_assembly"] is not None):
            raise AssertionError(label+" M20b physical revision lost immediate source parent")
        inv=row["revised_source_inventory"]
        ids=tuple(x["id"] for x in inv)
        if ids!=ACTIVE[label]:
            raise AssertionError(label+" source inventory improperly reconstructed")
        if tuple(x["kind"] for x in inv)!=tuple("UniformElectricField" for _ in ids):
            raise AssertionError(label+" source kind invented")
        _check_source_ancestry(row["parent_source_ancestry"],PARENTS[idx-1] if idx>1 else None,
                               parent,models[parent],ACTIVE[parent],SEEN[parent])
        _check_source_ancestry(row["source_ancestry"],parent,
                               label,model,ACTIVE[label],SEEN[label])
        assert_row=row["parent_event_time_comparisons"]
        if len(assert_row)!=1 or assert_row[0]["status"]!="one-resolved-pair":
            raise AssertionError(label+" genuine complete parent event was not paired")
        out[label]=_states_and_roots(model,row["native_observation"],label)
        models[label]=model
    if "field:Enew" in graph["nodes"][2]["source_ancestry"]["active_source_ids"]:
        raise AssertionError("replaced source should remain RETIRED")
    if graph["nodes"][2]["source_ancestry"]["introduced_source_ids"]!=["field:Eafter"]:
        raise AssertionError("real newly introduced replacement source was lost")
    return dict(reference="separate direct SI Newtonian field/force/state and signed detector formula",
                source_callbacks_imported=0, source_ancestry_verified=True,
                physically_authored_alternatives=True,native_by_world=out)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--native",type=Path,required=True)
    parser.add_argument("--output",type=Path,
                        default=Path("output/m32b-independent-reference.json"))
    args=parser.parse_args()
    raw=args.native.read_bytes()
    data=json.loads(raw)
    evaluated=verify(data)
    evaluated["source_json_sha256"]=hashlib.sha256(raw).hexdigest()
    evaluated["verdict"]="PASS"
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(evaluated,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"verdict":"PASS","native_sha256":evaluated["source_json_sha256"],
        "native_minus_analytic_event_s":{
            k:v["native_minus_reference_s"] for k,v in evaluated["native_by_world"].items()},
        "source_ancestry_checked":True,"output":str(args.output)},sort_keys=True))


if __name__=="__main__":
    main()
