"""Independent SI analytic validation of M32a's exported uniform-source graph.

This program imports NO conductor_sim modules and executes NO source callbacks.
It recomputes forces, full native saved 3D paths and true detector crossing
times from SI Newtonian constant-acceleration equations, using the explicitly
authored model fields and mass recorded in the detached JSON. It validates one
finite teacher-authored transfer case, not universal correctness or PDF input.

Run:
    python examples/m32a_unified_observations.py --case uniform-events --output output/m32a-graph.json
    python tools/m32a_independent_uniform.py --native output/m32a-graph.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from math import fsum, sqrt, isfinite
from pathlib import Path


EXPECTED_NODES = ("add-E", "remove-Ey", "source-mass", "launch")
EXPECTED_PARENTS = ("baseline", "baseline", "add-E", "source-mass")
EXPECTED_SOURCE_IDS = {
    "baseline": ("field:E0", "field:Ey"),
    "add-E": ("field:E0", "field:Ey", "field:Enew"),
    "remove-Ey": ("field:E0",),
    "source-mass": ("field:E0", "field:Ey", "field:Enew"),
    "launch": ("field:E0", "field:Ey", "field:Enew"),
}
EXPECTED_E_AND_MASS_AND_VX = {
    "baseline": (100., 50., 1e-5, .1),
    "add-E": (200., 50., 1e-5, .1),
    "remove-Ey": (100., 0., 1e-5, .1),
    "source-mass": (250., 50., 2e-5, .1),
    "launch": (250., 50., 2e-5, .16),
}


def _true_constant_acceleration_model(model):
    spec = model["spec"]
    sources = spec["field"]["sources"]
    if not sources or any(s["kind"] != "UniformElectricField" for s in sources):
        raise AssertionError("independent reference expects only authentic uniform sources")
    e = tuple(fsum(s["field_v_m"][k] for s in sources) for k in range(3))
    if any(not isfinite(x) for x in e):
        raise AssertionError("nonfinite authored SI electric field")
    if "gravity" in spec:
        raise AssertionError("this independent constant-E transfer has no gravity")
    q = spec["particle"]["charge_c"]
    mass = spec["mass_kg"]
    if abs(q-1e-9)>1e-20 or mass <= 0:
        raise AssertionError("unexpected probe charge or mass")
    p0 = tuple(spec["particle"]["position_m"])
    v0 = tuple(spec["initial_velocity_m_s"])
    a = tuple(q*x/mass for x in e)
    return dict(e=e, q=q, mass=mass, p0=p0, v0=v0, a=a,
                duration=spec["duration_s"])


def _check_saved_samples(native, model, label):
    params = _true_constant_acceleration_model(model)
    rows = native["trajectory"]["samples"] if native.get("schema") == (
        "conductor_sim.event_run.v1") else native["samples"]
    if not rows:
        raise AssertionError(label + " no original numerical saved states")
    position_error, velocity_error, force_error = 0., 0., 0.
    saved_times = []
    for row in rows:
        t = row["time_s"]
        if type(t) is not float or not isfinite(t) or not 0 <= t <= params["duration"]:
            raise AssertionError(label + " invented or unphysical saved SI time")
        saved_times.append(t)
        p = [params["p0"][k]+params["v0"][k]*t+.5*params["a"][k]*t*t
             for k in range(3)]
        v = [params["v0"][k]+params["a"][k]*t for k in range(3)]
        f = [params["q"]*params["e"][k] for k in range(3)]
        position_error = max(position_error, *(abs(row["position_m"][k]-p[k])
                                               for k in range(3)))
        velocity_error = max(velocity_error, *(abs(row["velocity_m_s"][k]-v[k])
                                               for k in range(3)))
        force_error = max(force_error, *(abs(row["force_n"][k]-f[k])
                                         for k in range(3)))
    if any(x>=y for x,y in zip(saved_times,saved_times[1:])):
        raise AssertionError(label + " native saved physical clock is not increasing")
    if abs(saved_times[0])>1e-14 or abs(saved_times[-1]-params["duration"])>1e-12:
        raise AssertionError(label + " incomplete native physical observation horizon")
    if position_error>1e-8 or velocity_error>1e-8 or force_error>1e-14:
        raise AssertionError(label + " actual native states disagree with direct Newton reference")
    return dict(samples=len(rows), max_position_error_m=position_error,
                max_velocity_error_m_s=velocity_error, max_force_error_n=force_error)


def _check_signed_detector(native, model, label):
    params = _true_constant_acceleration_model(model)
    if native["trajectory"]["termination"] != "duration":
        raise AssertionError(label + " was not duration-complete")
    roots = [r for r in native["events"] if r["event_id"] == "x:detector"]
    if len(roots) != 1 or roots[0]["direction"] != 1:
        raise AssertionError(label + " does not have one genuine positive-x root")
    root = roots[0]
    target = .006-params["p0"][0]
    vx, ax = params["v0"][0], params["a"][0]
    if target <= 0 or vx <= 0 or ax <= 0:
        raise AssertionError(label + " analytical root conditions not met")
    independent_t = 2*target/(vx+sqrt(vx*vx+2*ax*target))
    native_t = root["sample"]["time_s"]
    lower,upper = root["time_bracket_s"]
    if not (0<lower<=native_t<=upper<=params["duration"]):
        raise AssertionError(label + " invalid native numerical root bracket")
    err = native_t-independent_t
    if abs(err) > 2e-7 or not lower-2e-7<=independent_t<=upper+2e-7:
        raise AssertionError(label + " native root does not match independent SI quadratic")
    if abs(root["sample"]["position_m"][0]-.006)>2e-7:
        raise AssertionError(label + " native root does not actually cross detector")
    return dict(native_time_s=native_t, independent_time_s=independent_t,
                native_minus_analytic_s=err,
                native_bracket_s=[lower,upper],direction=1)


def independently_check(export):
    if export.get("schema") != "conductor_sim.unified_observation_graph.v1":
        raise ValueError("not one authentic M32a graph-schema evidence record")
    if export.get("time_interpolation") is not False or export.get("postguard_extrapolation") is not False:
        raise AssertionError("native evidence falsely claimed interpolated/extrapolated observations")
    if export["plan"]["plan_id"] != "m32a:teacher":
        raise AssertionError("unexpected teacher graph plan")
    original = export["plan"]["original"]
    if original["origin"] not in ("source-given", "teacher-hypothetical"):
        raise AssertionError("improper original source provenance")
    nodes = export["nodes"]
    if tuple(row["node"]["node_id"] for row in nodes) != EXPECTED_NODES:
        raise AssertionError("real graph node identity/order mismatch")
    if tuple(row["node"]["parent_id"] for row in nodes) != EXPECTED_PARENTS:
        raise AssertionError("real graph parent links mismatch")
    root = export["original_revision_id"]
    if root != original["revision_id"]:
        raise AssertionError("baseline source-model revision lost")
    refs = {"baseline":(original,root)}
    output = {}
    baseline_native = export["original_native_observation"]
    if export["original_observation_status"] != "complete":
        raise AssertionError("baseline event native observation incomplete")
    output["baseline"] = dict(
        **_check_saved_samples(baseline_native,original,"baseline"),
        **_check_signed_detector(baseline_native,original,"baseline"))
    for row in nodes:
        name = row["node"]["node_id"]
        parent = row["node"]["parent_id"]
        model = row["admitted_scenario"]
        if model is None or row["status"] != "complete":
            raise AssertionError(name+" did not admit and observe physical world")
        if row["parent_revision_id"] != refs[parent][1]:
            raise AssertionError(name+" immediate parent SHA wrong")
        if name in ("source-mass","launch"):
            if model["parent_revision_id"] != refs[parent][1]:
                raise AssertionError(name+" true M20b revision parent wrong")
        else:
            assembly=row["structural_assembly"]
            if assembly is None or assembly["parent_revision_id"]!=refs[parent][1]:
                raise AssertionError(name+" missing true M22a structural parent")
        ids=tuple(item["id"] for item in row["revised_source_inventory"])
        if ids != EXPECTED_SOURCE_IDS[name]:
            raise AssertionError(name+" source inventory lost original native IDs")
        params = _true_constant_acceleration_model(model)
        ex,ey,mass,vx=EXPECTED_E_AND_MASS_AND_VX[name]
        if any(abs(actual-expected)>1e-12 for actual,expected in
               ((params["e"][0],ex),(params["e"][1],ey),
                (params["mass"],mass),(params["v0"][0],vx))):
            raise AssertionError(name+" genuine authored SI model differs from stated experiment")
        native = row["native_observation"]
        if native is None:
            raise AssertionError(name+" has no native physical observation")
        outcome = _check_saved_samples(native,model,name)
        if row["node"]["observation"] == "events":
            outcome.update(_check_signed_detector(native,model,name))
        else:
            if native["termination"]!="duration":
                raise AssertionError("trajectory did not complete")
            if row["parent_event_time_comparisons"]:
                raise AssertionError("ordinary trajectory invented event comparisons")
        if name in ("add-E","source-mass","launch"):
            pairs = row["parent_event_time_comparisons"]
            if len(pairs)!=1 or pairs[0]["status"] != "one-resolved-pair":
                raise AssertionError(name+" did not preserve one genuine same-sign pair")
        refs[name]=(model,model["revision_id"])
        output[name]=outcome
    return dict(reference="separately implemented SI constant acceleration and signed quadratic root",
                calls_to_production_fields=0,source_identities_checked=True,
                source_parent_hashes_checked=True,sample_and_event_checks=output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native",type=Path,required=True)
    p.add_argument("--output",type=Path,default=Path("output/m32a-analytic-evidence.json"))
    args=p.parse_args()
    raw=args.native.read_bytes()
    record=json.loads(raw)
    accepted=independently_check(record)
    accepted["native_sha256"]=hashlib.sha256(raw).hexdigest()
    accepted["source_json"]=str(args.native)
    accepted["verdict"]="PASS"
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(accepted,indent=2,allow_nan=False)+"\n",
                           encoding="utf-8")
    print(json.dumps(dict(
        verdict="PASS",native_sha256=accepted["native_sha256"],
        results={k:v.get("native_minus_analytic_s",None)
                 for k,v in accepted["sample_and_event_checks"].items()},
        output=str(args.output),
    ),sort_keys=True))


if __name__ == "__main__":
    main()
