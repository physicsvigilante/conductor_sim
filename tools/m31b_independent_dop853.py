"""Independent numerical validation for the M31b rod/point/mass event.

This script NEVER imports conductor_sim. It uses Gauss–Legendre quadrature
for the charged rod, direct point Coulomb sums, and SciPy's DOP853 solver to
recompute real SI particle states and signed detector crossings. The reference
cannot reuse production field callbacks, event brackets, or RK output.

Optional scientific-only dependencies (not production engine dependencies):
    python -m pip install numpy scipy
Run:
    python examples/m31b_rod_reference.py --output output/m31b-rod-native.json
    python tools/m31b_independent_dop853.py --native output/m31b-rod-native.json
"""
from __future__ import annotations
import argparse
import json
from math import pi
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from numpy.polynomial.legendre import leggauss


# Separately declared physical inputs, in SI. No imports from conductor_sim.
EPSILON_0_SI = 8.8541878128e-12
COULOMB = 1/(4*pi*EPSILON_0_SI)
ROD_Q_C = 3e-9
ROD_START_M = np.array([.11, -.1, 0.])
ROD_END_M = np.array([.13, .12, 0.])
HELD_POINTS = (
    (2e-9, np.array([.18, .15, 0.])),
    (-1e-9, np.array([-.16, .10, 0.])),
)
ADDED_POINT_M = np.array([.21, -.09, 0.])
PROBE_Q_C = 1e-9
GRAVITY_M_S2 = np.array([0., -.1, 0.])
INITIAL = np.array([0., 0., 0., .05, .02, 0.])
DURATION_S = .7
DETECTOR_X_M = -.006


def evaluate(added_charge_c: float, mass_kg: float, *, quadrature_order: int = 96):
    nodes, weights = leggauss(quadrature_order)
    locations = ROD_START_M[None, :] + .5*(nodes[:,None]+1)*(
        ROD_END_M-ROD_START_M)[None,:]
    fractions = weights / 2

    def field_at(r):
        diffs = r[None,:] - locations
        lengths = np.linalg.norm(diffs, axis=1)
        rod_field = COULOMB*ROD_Q_C*np.sum(
            fractions[:,None]*diffs/lengths[:,None]**3,axis=0)
        total = rod_field
        for charge,site in (*HELD_POINTS,(added_charge_c,ADDED_POINT_M)):
            delta = r-site
            total = total + COULOMB*charge*delta/np.linalg.norm(delta)**3
        return total

    def rhs(_t, state):
        return np.r_[
            state[3:],
            PROBE_Q_C/mass_kg*field_at(state[:3]) + GRAVITY_M_S2,
        ]

    def detector(_t, state):
        return float(state[0]-DETECTOR_X_M)

    detector.direction = -1
    detector.terminal = False
    solved = solve_ivp(
        rhs, (0., DURATION_S), INITIAL.copy(), method="DOP853",
        rtol=1e-11, atol=1e-13, max_step=.002, events=detector,
    )
    if not solved.success or solved.t[-1] < DURATION_S:
        raise RuntimeError("independent DOP853 could not complete requested physical horizon")
    if len(solved.t_events[0]) != 1:
        raise RuntimeError("independent DOP853 did not locate one negative x crossing")
    return dict(
        source_charge_c=added_charge_c, probe_mass_kg=mass_kg,
        event_time_s=float(solved.t_events[0][0]),
        initial_force_n=(PROBE_Q_C*field_at(INITIAL[:3])+
                         mass_kg*GRAVITY_M_S2).tolist(),
        final_position_m=solved.y[:3,-1].tolist(),
        evaluations=int(solved.nfev),
    )


def assert_native_matches(native, reference, *, time_tolerance_s=5e-7,
                          force_tolerance_n=2e-12):
    if native.get("schema") != "conductor_sim.coupled_event_family.v1":
        raise ValueError("not M31b native detached event evidence")
    parent_section = native["parent_events"][0]
    child_section = native["coupled_events"][0]
    if child_section["parent_revision_id"] != parent_section["parent_revision_id"]:
        raise ValueError("child event lost actual structural parent revision")
    for record in (parent_section, child_section):
        ids = record.get("accepted_parent_inventory", record.get("accepted_source_inventory"))
        if not any(item.get("id") == "point:added" for item in ids or []):
            raise ValueError("real newly introduced held point identity was not retained")
    compared = []
    for label, section in (("parent",parent_section),("child",child_section)):
        event = section.get("event_run")
        if event is None or event["trajectory"]["termination"] != "duration":
            raise ValueError(f"{label} lacks a duration-complete real native event run")
        roots = [row for row in event["events"] if row.get("event_id")=="x:left"]
        if len(roots) != 1 or roots[0]["direction"] != -1:
            raise ValueError(f"{label} did not preserve its unique signed native root")
        saved = roots[0]["sample"]["time_s"]
        if type(saved) is not float:
            raise ValueError(f"{label} event time must be numeric SI float")
        diff = saved - reference[label]["event_time_s"]
        if abs(diff) > time_tolerance_s:
            raise AssertionError(f"{label} native root differs from independent DOP853 by {diff:.8g} s")
        low, high = roots[0]["time_bracket_s"]
        if not low - time_tolerance_s <= reference[label]["event_time_s"] <= high + time_tolerance_s:
            raise AssertionError(f"{label} numerical bracket inconsistent with independent event root")
        compared.append(dict(
            role=label, native_event_time_s=saved,
            reference_event_time_s=reference[label]["event_time_s"],
            native_minus_reference_s=diff,
            native_bracket_s=[low,high],
        ))
    first_force = np.array(child_section["event_run"]["trajectory"]["samples"][0]["force_n"])
    expected_force = np.array(reference["child"]["initial_force_n"])
    if float(np.max(np.abs(first_force-expected_force))) > force_tolerance_n:
        raise AssertionError("native child initial SI force disagrees with independently integrated Coulomb law")
    return compared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native",type=Path,
                        help="M31b genuine native detached JSON to independently challenge")
    parser.add_argument("--output",type=Path,default=Path("output/m31b-dop853-reference.json"))
    args = parser.parse_args()

    results = dict(
        parent=evaluate(1.7e-9,1e-5),
        child=evaluate(2e-9,9e-6),
    )
    result = dict(
        reference="independently authored 96-node Coulomb rod quadrature + SciPy DOP853",
        source_callback_imported=False,
        floating_point_SI=True,
        detector_plane_x_m=DETECTOR_X_M,
        direction=-1,
        duration_s=DURATION_S,
        scipy=scipy.__version__, numpy=np.__version__,
        rtol=1e-11, atol=1e-13, max_step_s=.002,
        reference_results=results,
    )
    if args.native is not None:
        record=json.loads(args.native.read_text(encoding="utf-8"))
        result["native_comparisons"]=assert_native_matches(record,results)
        result["native_reference_verdict"]="PASS"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,allow_nan=False,indent=2)+"\n",
                           encoding="utf-8")
    print(json.dumps({
        "reference_parent_s": results["parent"]["event_time_s"],
        "reference_child_s": results["child"]["event_time_s"],
        "native_reference_verdict": result.get("native_reference_verdict","not-run"),
        "output":str(args.output),
    },sort_keys=True))


if __name__ == "__main__":
    main()
