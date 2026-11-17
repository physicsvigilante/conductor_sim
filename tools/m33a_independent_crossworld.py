"""M33a independently expressed SI numerical reference: NO conductor_sim imports.

Recomputes every saved 3D Newtonian state and force in separately authored
fixed-uniform-field worlds, positive detector roots, literal shared SI times,
right-minus-left paired differences, event time shifts and branch provenance.
This is a finite transfer check, NOT an exact numerical-error theorem or a
separate-context independent scientific reviewer report.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from math import dist, fsum, isclose, isfinite, sqrt
from pathlib import Path

WORLDS = ("baseline", "field-low", "field-high", "low-mass")
PARENTS = (None, "baseline", "baseline", "field-low")
EXPECTED = {
    "baseline": ((100., 50., 0.), 1e-5, ("field:E0", "field:Ey"),
                 ("baseline",)),
    "field-low": ((140., 50., 0.), 1e-5,
                  ("field:E0", "field:Ey", "field:added"),
                  ("baseline", "field-low")),
    "field-high": ((220., 50., 0.), 1e-5,
                   ("field:E0", "field:Ey", "field:added"),
                   ("baseline", "field-high")),
    "low-mass": ((140., 50., 0.), 2e-5,
                 ("field:E0", "field:Ey", "field:added"),
                 ("baseline", "field-low", "low-mass")),
}
PAIRS = (
    ("siblings", "field-low", "field-high"),
    ("cousins", "field-high", "low-mass"),
    ("original", "baseline", "field-high"),
)


def require(condition, why):
    if not condition:
        raise AssertionError(why)


def real(value):
    require(type(value) in (float, int) and isfinite(value),
            "non-finite or Boolean number in native evidence")
    return float(value)


def vector(value):
    require(type(value) in (list, tuple) and len(value) == 3,
            "missing genuine three-component SI vector")
    return tuple(real(v) for v in value)


def near(actual, expected, tolerance, why):
    require(isclose(real(actual), expected, rel_tol=0., abs_tol=tolerance),
            why + ": " + repr(actual) + " vs " + repr(expected))


def vec_near(actual, expected, tolerance, why):
    v = vector(actual)
    for k in range(3):
        near(v[k], expected[k], tolerance, why + " axis " + str(k))


def check_ancestry(cert, model, label):
    expected = EXPECTED[label]
    ids = expected[2]
    world_path = expected[3]
    require(cert["schema"] == "conductor_sim.source_world_ancestry.v1",
            label + " missing native source certificate")
    require(cert["world_id"] == label and
            cert["parent_world_id"] == PARENTS[WORLDS.index(label)],
            label + " wrong source-certificate graph edge")
    require(cert["model_revision_id"] == model["revision_id"],
            label + " source-certificate revision mismatch")
    require(tuple(cert["world_path"]) == world_path,
            label + " source ancestry path changed")
    require(tuple(cert["active_source_ids"]) == ids,
            label + " active source IDs were lost")
    require(tuple(cert["seen_source_ids"]) == ids and
            not cert["retired_source_ids"],
            label + " branch-local ID history changed")
    require(cert["gravity_present"] is False,
            label + " added nonexistent gravity")
    if label == "baseline":
        require(cert["parent_revision_id"] is None and
                not cert["introduced_source_ids"],
                "baseline fabricated a predecessor")
    elif label == "low-mass":
        require(not cert["introduced_source_ids"],
                "physical revision invented a new source ID")
    else:
        require(cert["introduced_source_ids"] == ["field:added"],
                label + " did not introduce one fresh per-branch source ID")


def check_model(model, label):
    e_expected, m_expected, ids, _path = EXPECTED[label]
    spec = model["spec"]
    require(spec["gravity"] is None,
            label + " uniform reference excludes gravity")
    require(all(source["kind"] == "UniformElectricField"
                for source in spec["field"]["sources"]),
            label + " unsupported field kind in Newtonian reference")
    e = tuple(fsum(vector(s["field_v_m"])[k]
                   for s in spec["field"]["sources"]) for k in range(3))
    vec_near(e, e_expected, 1e-12, label + " authored field")
    near(spec["mass_kg"], m_expected, 1e-15, label + " mass")
    near(spec["particle"]["charge_c"], 1e-9, 1e-19, label + " charge")
    vec_near(spec["particle"]["position_m"], (0., 0., 0.), 1e-15,
             label + " launch point")
    vec_near(spec["initial_velocity_m_s"], (.1, 0., 0.), 1e-14,
             label + " initial velocity")
    near(spec["duration_s"], .1, 1e-14, label + " duration")
    require(tuple(x[0] for x in model["source_annotations"]) == ids,
            label + " model source-annotation IDs altered")
    charge = real(spec["particle"]["charge_c"])
    mass = real(spec["mass_kg"])
    return dict(e=e, mass=mass, q=charge, r0=(0., 0., 0.),
                v0=(.1, 0., 0.),
                a=tuple(charge * e[k] / mass for k in range(3)),
                duration=.1)


def position_velocity(world, t):
    a, r0, v0 = world["a"], world["r0"], world["v0"]
    p = tuple(r0[k] + v0[k]*t + .5*a[k]*t*t for k in range(3))
    v = tuple(v0[k] + a[k]*t for k in range(3))
    return p, v


def positive_root(world):
    dx = .006 - world["r0"][0]
    v = world["v0"][0]
    a = world["a"][0]
    return 2*dx / (v + sqrt(v*v + 2*a*dx))


def check_native(model, native, label):
    world = check_model(model, label)
    traj = native["trajectory"]
    require(traj["termination"] == "duration",
            label + " did not finish physical observation horizon")
    samples = traj["samples"]
    require(len(samples) >= 3, label + " too few native states")
    times = []
    max_pos = max_vel = max_force = 0.
    for sample in samples:
        t = real(sample["time_s"])
        require(0 <= t <= world["duration"],
                label + " outside genuine physical SI time horizon")
        times.append(t)
        pos, vel = position_velocity(world, t)
        f = tuple(world["q"]*world["e"][k] for k in range(3))
        native_pos = vector(sample["position_m"])
        native_vel = vector(sample["velocity_m_s"])
        native_force = vector(sample["force_n"])
        max_pos = max(max_pos, *(abs(native_pos[k]-pos[k])
                                 for k in range(3)))
        max_vel = max(max_vel, *(abs(native_vel[k]-vel[k])
                                 for k in range(3)))
        max_force = max(max_force, *(abs(native_force[k]-f[k])
                                     for k in range(3)))
    require(times[0] == 0. and abs(times[-1]-.1) < 1e-10 and
            all(b>a for a,b in zip(times, times[1:])),
            label + " invalid genuine SI saved timestamp sequence")
    require(max_pos < 2e-8 and max_vel < 2e-8 and max_force < 1e-13,
            label + " trajectory/force disagrees with independent Newtonian reference")
    roots = [e for e in native["events"] if e["event_id"] == "x:detector"]
    require(len(roots) == 1 and roots[0]["direction"] == 1,
            label + " expected one located positive crossing")
    require(native["outcomes"]["x:detector"] == "located",
            label + " root outcome not genuine located")
    root = roots[0]
    numeric = real(root["sample"]["time_s"])
    bracket = tuple(real(v) for v in root["time_bracket_s"])
    exact = positive_root(world)
    require(bracket[0] <= numeric <= bracket[1] and
            abs(numeric-exact) < 2e-7,
            label + " detector event mismatches independent root")
    near(root["sample"]["position_m"][0], .006, 1e-7,
         label + " event on detector plane")
    return dict(model=model, world=world, native=native,
                times=times, root=numeric, bracket=bracket,
                max_pos=max_pos, max_vel=max_vel, max_force=max_force,
                analytic_root=exact)


def compare_pair(record, parts, pair):
    pair_id, lid, rid = pair
    require(record["comparison_id"] == pair_id and
            record["left_world_id"] == lid and
            record["right_world_id"] == rid,
            "comparison swapped requested physical identities")
    left, right = parts[lid], parts[rid]
    require(record["left_revision_id"] == left["model"]["revision_id"] and
            record["right_revision_id"] == right["model"]["revision_id"],
            pair_id + " wrong real physical revision on side")
    for side, label in (("left", lid), ("right", rid)):
        check_ancestry(record[side + "_source_ancestry"],
                       parts[label]["model"], label)
        require(tuple(v["id"] for v in record[side + "_source_inventory"]) ==
                EXPECTED[label][2],
                pair_id + " source inventory moved to wrong side")
        require(all(v["kind"] == "UniformElectricField"
                    for v in record[side + "_source_inventory"]),
                pair_id + " kind of a source silently altered")
    require(record["numerical_orientation"] == "right minus left" and
            record["status"] == "compared" and
            record["left_observation_mode"] == record["right_observation_mode"] == "events",
            pair_id + " not an explicitly oriented native event comparison")
    ltimes, rtimes = set(left["times"]), right["times"]
    expected_times = [t for t in right["times"] if t in ltimes]
    rows = record["exact_time_trajectory_comparisons"]
    require(len(rows) == len(expected_times) and bool(rows),
            pair_id + " dropped native shared-time samples")
    max_delta_error = 0.
    for row, t in zip(rows, expected_times):
        require(type(row["time_s"]) is float and row["time_s"] == t,
                pair_id + " fabricated, skipped or interpolated SI time")
        lp, lv = position_velocity(left["world"], t)
        rp, rv = position_velocity(right["world"], t)
        dp = tuple(rp[k] - lp[k] for k in range(3))
        dv = tuple(rv[k] - lv[k] for k in range(3))
        vec_near(row["delta_position_m"], dp, 3e-8,
                 pair_id + " independently expressed position difference")
        vec_near(row["delta_velocity_m_s"], dv, 3e-8,
                 pair_id + " independently expressed velocity difference")
        near(row["position_distance_m"], dist(lp, rp), 3e-8,
             pair_id + " independent position norm")
        near(row["velocity_distance_m_s"], dist(lv, rv), 3e-8,
             pair_id + " independent velocity norm")
        max_delta_error = max(max_delta_error,
            *(abs(vector(row["delta_position_m"])[k]-dp[k])
              for k in range(3)))
    events = record["event_time_comparisons"]
    require(len(events)==1 and events[0]["event_id"]=="x:detector" and
            events[0]["status"]=="one-resolved-pair" and
            events[0]["direction"]==1,
            pair_id + " lacks genuine duration-complete unique directional event")
    event = events[0]
    numerical_shift = right["root"]-left["root"]
    independent_shift = right["analytic_root"]-left["analytic_root"]
    near(event["estimated_numerical_shift_s"], numerical_shift, 1e-12,
         pair_id + " event shift native origins")
    near(event["estimated_numerical_shift_s"], independent_shift, 4e-7,
         pair_id + " independently expressed analytic event shift")
    vec_pair = (right["bracket"][0]-left["bracket"][1],
                right["bracket"][1]-left["bracket"][0])
    interval = event["numerical_bracket_shift_interval_s"]
    require(len(interval) == 2, pair_id + " missing bounded numerical bracket")
    for actual, expected in zip(interval, vec_pair):
        near(actual, expected, 1e-12, pair_id + " event bracket arithmetic")
    require(real(interval[0]) <= numerical_shift <= real(interval[1]),
            pair_id + " shifted event falls outside numerical bracket")
    return dict(id=pair_id, count=len(rows), max_position_delta_error_m=max_delta_error,
                numerical_event_shift_s=numerical_shift,
                analytic_event_shift_s=independent_shift)


def verify(graph):
    require(graph["schema"] == "conductor_sim.unified_observation_graph.v3" and
            graph["plan"]["schema"] == "conductor_sim.unified_observation_plan.v3",
            "reference expects M33a opted-in v3 native graph")
    require(graph["plan_id"] == "m33a:demo" and
            graph["plan"]["retain_source_ancestry"] is True and
            not graph["time_interpolation"] and
            not graph["postguard_extrapolation"],
            "not correct native teacher plan or interpolation contract")
    require(tuple((r["comparison_id"],r["left_world_id"],r["right_world_id"])
                  for r in graph["plan"]["comparisons"]) == PAIRS,
            "teacher request not recorded verbatim")
    require(len(graph["nodes"]) == 3 and
            len(graph["cross_world_comparisons"]) == 3,
            "wrong number of real native worlds or requests")
    baseline = graph["plan"]["original"]
    check_ancestry(graph["original_source_ancestry"], baseline, "baseline")
    parts = {"baseline":check_native(baseline,graph["original_native_observation"],
                                     "baseline")}
    for i, row in enumerate(graph["nodes"],1):
        label = WORLDS[i]
        parent = PARENTS[i]
        require(row["node"]["node_id"] == label and
                row["parent_id"] == parent and row["status"] == "complete",
                label + " is not independently admitted and observed")
        model = row["admitted_scenario"]
        require(model is not None and row["native_observation"] is not None,
                label + " missing genuine native model or event run")
        require(row["parent_revision_id"] == parts[parent]["model"]["revision_id"],
                label + " parent model ancestry changed")
        if label == "low-mass":
            require(row["node"]["operation"] == "physical" and
                    model["parent_revision_id"] == parts[parent]["model"]["revision_id"],
                    "low-mass is not an actual atomic physical edit")
        else:
            require(row["node"]["operation"] == "structural" and
                    row["structural_assembly"]["parent_revision_id"] ==
                    parts[parent]["model"]["revision_id"],
                    label + " structural reassembly lost native parent")
        check_ancestry(row["source_ancestry"], model, label)
        parts[label] = check_native(model,row["native_observation"],label)
    findings = [compare_pair(record,parts,expected)
                for record,expected in zip(graph["cross_world_comparisons"],PAIRS)]
    return dict(reference="independent 3D Newtonian SI acceleration, actual saved states, "
                          "force and same-direction detector quadratic root",
                native_worlds={label:dict(samples=len(parts[label]["times"]),
                                          max_position_error_m=parts[label]["max_pos"],
                                          max_velocity_error_m_s=parts[label]["max_vel"],
                                          max_force_error_n=parts[label]["max_force"])
                               for label in WORLDS},
                pairs=findings,source_callbacks_imported=0)


def negative_controls(source):
    for name, mutator in (
        ("revised-hash", lambda g: g["cross_world_comparisons"][0].__setitem__(
            "left_revision_id", "FAKE-REVISION")),
        ("falsified-delta", lambda g: g["cross_world_comparisons"][0]
            ["exact_time_trajectory_comparisons"][1]["delta_position_m"].__setitem__(1, .5)),
        ("event-time-tamper", lambda g: g["cross_world_comparisons"][1]
            ["event_time_comparisons"][0].__setitem__("estimated_numerical_shift_s", .5)),
        ("wrong-branch-history", lambda g: g["cross_world_comparisons"][0]
            ["right_source_ancestry"].__setitem__("world_path", ["baseline","field-low"])),
    ):
        tampered = copy.deepcopy(source)
        mutator(tampered)
        try:
            verify(tampered)
        except (AssertionError, ValueError, KeyError, TypeError):
            continue
        raise AssertionError("independent verifier missed tampering: " + name)
    return 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("output/m33a-independent-reference.json"))
    args = parser.parse_args()
    payload = args.native.read_bytes()
    data = json.loads(payload)
    result = verify(data)
    result["negative_controls_rejected"] = negative_controls(data)
    result["source_json_sha256"] = hashlib.sha256(payload).hexdigest()
    result["verdict"] = "PASS"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"verdict":result["verdict"],
                      "reference_worlds":len(result["native_worlds"]),
                      "compared_pairs":len(result["pairs"]),
                      "negative_controls":result["negative_controls_rejected"],
                      "output":str(args.output)},sort_keys=True))


if __name__ == "__main__":
    main()
