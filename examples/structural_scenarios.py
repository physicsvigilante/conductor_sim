"""M22a preselected JEE-relevant authoring cards; unchanged production physics.

These are curated teacher hypotheticals, not official exam question replicas.
Build new arrangements through the single public structural-reassembly API.
"""
from __future__ import annotations

import argparse
import json
from math import fsum, hypot, pi, sqrt
from pathlib import Path

from conductor_sim import (
    EPSILON_0, FixedPointField, PointCharge, PrescribedField,
    TestChargeMotionSpec, UniformElectricField, UniformRingAxisField,
    UniformRodField, UniformSphereField, UniformGravity,
    SolidSphere, FixedNetCharge, integrate_test_charge,
)
from conductor_sim.scenarios import PrescribedScenario, sample_scenario_viewer
from conductor_sim.structural_scenarios import (
    NamedSource, SourceReplacement, reassemble_scenario,
)
from conductor_sim.scenario_comparison import prepare_scenario_comparison
from conductor_sim.solver import UnsupportedProblem
from conductor_sim.viewer import render_html

POSITIVE = ("P1", "P2", "P3", "U1", "U2", "U3", "R1", "R2", "A1")
NEGATIVE = ("N1", "N2", "N3", "N4", "N5", "N6", "N7")
TITLES = {
    "P1": "Add a second held point charge",
    "P2": "Compose independent electric and gravitational forces",
    "P3": "Add a coaxial charged ring",
    "U1": "Three signed point charges in a plane",
    "U2": "Finite rod, point charge and gravity",
    "U3": "Three coaxial rings, applied E and gravity",
    "R1": "Replace a point source with a volume sphere",
    "R2": "Remove a point source and gravity, retain the rod",
    "A1": "Add a mirrored off-slice source pair atomically",
}


def _base(case, bindings=(), *, p=(0., .18, 0.), v=(.04, .02, 0.),
          normal=(0., 0., 1.), q=1e-9, mass=1e-6):
    spec = TestChargeMotionSpec(
        PrescribedField(tuple(b.source for b in bindings)),
        PointCharge(q, p), mass, v, .12, .005,
        plane_normal=normal, length_scale_m=.5,
    )
    return PrescribedScenario(
        "m22a-"+case+"-base", spec,
        tuple((b.source_id, b.label) for b in bindings),
        "Teacher hypothetical: M22a "+case+" initial model.",
    )


def build_card(case):
    """No case-specific callback, field law, solver or numerical policy."""
    B = NamedSource
    options = {}
    if case == "P1":
        base = _base(case, (B("point:A", "Held point A",
                             FixedPointField(2e-9, (-.15, 0., 0.))),))
        options["add_sources"] = (B("point:B", "Held point B",
                                   FixedPointField(-.8e-9, (.18, 0., 0.))),)
    elif case == "P2":
        base = _base(case, p=(0., 0., 0.), v=(.4, 1.2, 0.), q=1e-6, mass=2e-6)
        options = dict(add_sources=(B("field:E", "Applied E",
                                     UniformElectricField((6., 0., 0.))),),
                       gravity=UniformGravity((0., -10., 0.)))
    elif case in ("P3", "U3"):
        base = _base(case, (B("ring:A", "Ring A", UniformRingAxisField(
            1e-9, .12, axis=(1., 0., 0.))),),
            p=(.15, 0., 0.), v=(.03, 0., 0.))
        options["add_sources"] = (
            B("ring:B", "Ring B", UniformRingAxisField(
                -.6e-9, .12, center_m=(.35, 0., 0.), axis=(1., 0., 0.))),)
        if case == "U3":
            options["add_sources"] += (
                B("ring:C", "Ring C", UniformRingAxisField(
                    1.2e-9, .09, center_m=(-.18, 0., 0.), axis=(1., 0., 0.))),
                B("field:axial", "Axial applied E", UniformElectricField((3., 0., 0.))),
            )
            options["gravity"] = UniformGravity((-.08, 0., 0.))
    elif case == "U1":
        base = _base(case, (B("point:A", "Held positive A",
                             FixedPointField(1e-9, (-.2, -.1, 0.))),),
                     p=(0., .12, 0.), v=(.02, .1, 0.))
        options["add_sources"] = (
            B("point:B", "Held negative B", FixedPointField(-1.2e-9, (.2, -.1, 0.))),
            B("point:C", "Held positive C", FixedPointField(.7e-9, (.04, .3, 0.))),
        )
    elif case == "U2":
        base = _base(case, (B("rod:A", "Fixed finite rod",
                             UniformRodField(1e-9, (-.15, -.1, 0.), (.15, -.1, 0.))),),
                     v=(.12, .04, 0.))
        options = dict(
            add_sources=(B("point:B", "Held point B", FixedPointField(-1.2e-9, (.32, .12, 0.))),),
            gravity=UniformGravity((0., -1., 0.)),
        )
    elif case == "R1":
        base = _base(case, (B("point:A", "Original fixed point", FixedPointField(1e-9)),),
                     p=(.08, .06, 0.), v=(.01, .02, 0.))
        options["replace_sources"] = (SourceReplacement(
            "point:A", B("sphere:new", "New fixed volume sphere", UniformSphereField(1e-9, .2))),)
    elif case == "R2":
        base = build_card("U2")
        options = dict(remove_sources=("point:B",), gravity=None)
    elif case == "A1":
        base = _base(case, p=(0., 0., 0.), v=(.02, 0., .05), normal=(0., 1., 0.))
        options["add_sources"] = (
            B("point:mirror-up", "Upper held point", FixedPointField(1e-9, (.2, .12, 0.))),
            B("point:mirror-down", "Lower held point", FixedPointField(1e-9, (.2, -.12, 0.))),
        )
    else:
        raise ValueError("unknown preselected authoring card")
    return reassemble_scenario(
        base, scenario_id="m22a-"+case+"-assembled",
        note="M22a: "+TITLES[case]+".", **options)


def _source_reference(source, point, k):
    """Independent closed forms / midpoint quadrature, never provider methods."""
    if type(source) is UniformElectricField:
        potential = source.reference_potential_v - sum(
            e*(p-o) for e, p, o in zip(source.field_v_m, point, source.reference_point_m))
        return potential, source.field_v_m
    if type(source) is UniformRodField:
        n = 1600
        potential, ef = 0., [0., 0., 0.]
        for j in range(n):
            location = tuple(a+(b-a)*(j+.5)/n for a, b in zip(source.start_m, source.end_m))
            delta = tuple(p-s for p, s in zip(point, location))
            r = hypot(*delta)
            potential += k*source.charge_c/(n*r)
            for i in range(3):
                ef[i] += k*source.charge_c*delta[i]/(n*r**3)
        return potential, tuple(ef)
    centre = source.position_m if type(source) is FixedPointField else source.center_m
    delta = tuple(p-c for p, c in zip(point, centre))
    r = hypot(*delta)
    if type(source) is FixedPointField:
        return k*source.charge_c/r, tuple(k*source.charge_c*x/r**3 for x in delta)
    if type(source) is UniformRingAxisField:
        z = sum(x*a for x, a in zip(delta, source.axis))
        d = sqrt(z*z + source.radius_m**2)
        return k*source.charge_c/d, tuple(k*source.charge_c*z*a/d**3 for a in source.axis)
    if type(source) is UniformSphereField:
        R = source.radius_m
        if r <= R:
            return k*source.charge_c*(3*R*R-r*r)/(2*R**3), tuple(
                k*source.charge_c*x/R**3 for x in delta)
        return k*source.charge_c/r, tuple(k*source.charge_c*x/r**3 for x in delta)
    raise AssertionError("missing independent reference for predeclared source")


def reference(spec, point):
    k = 1/(4*pi*spec.field.permittivity_f_m)
    terms = [_source_reference(s, point, k) for s in spec.field.sources]
    v = fsum(t[0] for t in terms)
    E = tuple(fsum(t[1][i] for t in terms) for i in range(3))
    g = spec.gravity.acceleration_m_s2 if spec.gravity else (0., 0., 0.)
    ug = (-spec.mass_kg*sum(a*(p-o) for a, p, o in
            zip(g, point, spec.gravity.reference_position_m)) if spec.gravity else 0.)
    return v, E, tuple(spec.particle.charge_c*e+spec.mass_kg*a for e, a in zip(E, g)), ug


def check_physics(assembly, result):
    spec = assembly.scenario.spec
    assert result.spec == spec and result.termination == "duration"
    rows = result.samples
    max_v = max_e = max_energy = 0.
    first = rows[0]
    h0 = first.kinetic_energy_j + spec.particle.charge_c*reference(spec, first.position_m)[0] + reference(spec, first.position_m)[3]
    scale = max(abs(h0), first.kinetic_energy_j, abs(spec.particle.charge_c*reference(spec, first.position_m)[0]), 1e-30)
    for r in rows:
        v, E, F, ug = reference(spec, r.position_m)
        ev, ee = abs(v-r.potential_v), hypot(*(a-b for a, b in zip(E, r.electric_field_v_m)))
        max_v, max_e = max(max_v, ev), max(max_e, ee)
        assert ev <= 4e-6*max(abs(v), 1.), (ev, v)
        assert ee <= 4e-6*max(hypot(*E), 1.), (ee, E)
        assert hypot(*(a-b for a, b in zip(F, r.force_n))) <= 4e-6*max(hypot(*F), 1e-12)
        h = .5*spec.mass_kg*sum(x*x for x in r.velocity_m_s) + spec.particle.charge_c*v + ug
        max_energy = max(max_energy, abs(h-h0)/scale)
    assert max_energy < 5e-5, max_energy
    if assembly.scenario.scenario_id == "m22a-P2-assembled":
        acceleration = (3., -10., 0.)
        for r in rows:
            expected = tuple(p+v*r.time_s+.5*a*r.time_s**2 for p, v, a in
                             zip(spec.particle.position_m, spec.initial_velocity_m_s, acceleration))
            assert hypot(*(x-y for x, y in zip(expected, r.position_m))) < 1e-11
    return dict(reference="independent Coulomb/ring/sphere formulas; 1600-panel rod quadrature where applicable",
                max_potential_error_v=max_v, max_field_error_v_m=max_e,
                max_relative_energy_change=max_energy, checked_samples=len(rows))


def make_bundle(assembly, result=None):
    from conductor_sim.prescribed_viewer import sample_test_charge_viewer
    spec = assembly.scenario.spec
    bundle = sample_test_charge_viewer(
        spec, result=result, source_annotations=assembly.scenario.source_annotations,
        visual_frames=13, resolution=13, profile_samples=101, extent_m=.6,
        axis_orientation="horizontal" if spec.axis is not None else "vertical")
    for frame in bundle["frames"]:
        frame["notes"].append(assembly.scenario.viewer_provenance)
        frame["notes"].append(
            "Structural reassembly from "+(
                assembly.parent.scenario.scenario_id if hasattr(assembly.parent, "scenario")
                else assembly.parent.scenario_id)+
            "; teacher hypothetical. Parent/child inventories and change lineage are in the adjacent JSON.")
        for change in assembly.changes:
            before, after = change.get("before"), change.get("after")
            if change["operation"].endswith("source"):
                before_id = before["id"] if before else "none"
                after_id = after["id"] if after else "none"
                frame["notes"].append("Structural edit: "+change["operation"]+
                                      " ["+before_id+" -> "+after_id+"].")
            else:
                frame["notes"].append("Structural edit: "+change["operation"]+" [gravity].")
    return bundle


def export_case(card, output):
    assembly = build_card(card)
    run = integrate_test_charge(assembly.scenario.spec)
    physics = check_physics(assembly, run)
    bundle = make_bundle(assembly, run)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(bundle, title="M22a "+card+" · "+TITLES[card]), encoding="utf-8")
    audit = dict(assembly=assembly.as_dict(), numerical_result=run.as_dict(),
                 independent_physics=physics, viewer=bundle)
    output.with_suffix(".json").write_text(json.dumps(audit, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return audit


def negative_card(card):
    """Expected refusals keep qualification distinct from observation refusal."""
    B = NamedSource
    if card == "N1":
        parent = build_card("P3").scenario
        call = lambda: reassemble_scenario(parent, scenario_id="bad-axis", note="Transverse E.",
            add_sources=(B("field:bad", "Transverse E", UniformElectricField((0., 1., 0.))),))
        expected, token, stage = UnsupportedProblem, "axis/plane", "physical-qualification"
    elif card == "N2":
        parent = build_card("A1")
        call = lambda: reassemble_scenario(parent, scenario_id="bad-mirror", note="Remove one partner.",
                                           remove_sources=("point:mirror-up",))
        expected, token, stage = UnsupportedProblem, "reflection", "physical-qualification"
    elif card == "N3":
        parent = _base("negative")
        call = lambda: reassemble_scenario(parent, scenario_id="bad-conductor", note="Conductor is not prescribed.",
            add_sources=(B("metal:A", "Metal sphere", SolidSphere(.1, FixedNetCharge(1e-9))),))
        expected, token, stage = TypeError, "built-in fixed source", "component-type"
    elif card == "N4":
        parent = build_card("P1").scenario
        call = lambda: reassemble_scenario(parent, scenario_id="bad-identity", note="Duplicate source identity.",
            add_sources=(B("point:A", "New point with stolen ID", FixedPointField(1e-9, (.4, 0., 0.))),))
        expected, token, stage = ValueError, "identities", "identity"
    elif card == "N5":
        parent = _base("guard")
        call = lambda: reassemble_scenario(parent, scenario_id="bad-guard", note="Source at actual initial position.",
            add_sources=(B("point:guard", "Point on probe", FixedPointField(1e-9, parent.spec.particle.position_m)),))
        expected, token, stage = ValueError, "guard", "numerical-domain"
    elif card == "N6":
        assembly = build_card("P1")
        parent = assembly.parent
        left = sample_scenario_viewer(parent, visual_frames=13, resolution=13, profile_samples=101, extent_m=.6)
        right = sample_scenario_viewer(assembly.scenario, visual_frames=13, resolution=13, profile_samples=101, extent_m=.6)
        call = lambda: prepare_scenario_comparison(left, right)
        expected, token, stage = ValueError, "identities differ", "requested-observation"
    elif card == "N7":
        parent = build_card("P1").scenario
        call = lambda: reassemble_scenario(parent, scenario_id="bad-atomic", note="Unknown removal.",
                                           remove_sources=("point:missing",), gravity=UniformGravity((0., -1., 0.)))
        expected, token, stage = ValueError, "unknown", "identity"
    else:
        raise ValueError("unknown negative card")
    before = parent.as_dict()
    try:
        call()
    except expected as exc:
        assert token in str(exc), str(exc)
        assert parent.as_dict() == before
        return dict(card=card, outcome="expected-refusal", refusal_stage=stage,
                    reason=str(exc), parent_unchanged=True,
                    authoring="pass" if card=="N6" else "attempted",
                    qualification="pass" if card=="N6" else (
                        "refused" if stage in ("physical-qualification","numerical-domain") else "not_evaluated"),
                    numerical_completion="two-duration-histories" if card=="N6" else "not_attempted",
                    observation_data="refused" if card=="N6" else "not_attempted",
                    interpretation_fidelity={"status":"pass","evidence":["request and rejection stage recorded"]},
                    physical_correctness={"status":"not_evaluated","evidence":["no independent numerical correctness credit for a refusal"]},
                    simulation_coverage={"status":"partial" if card=="N6" else "not_applicable",
                        "represented":["two valid runs"] if card=="N6" else [],
                        "excluded":["requested comparison"] if card=="N6" else ["invalid requested assembly"]},
                    educational_usefulness={"status":"not_evaluated","evidence":["no browser acceptance in a negative card"]},
                    textbook_answer_completeness={"status":"not_attempted","required_for_simulator_acceptance":False})
    raise AssertionError("out-of-contract request was not refused")


def grade_trial():
    """All predeclared cards, never hide unexpected failures or collapse layers."""
    cases = []
    for card in POSITIVE:
        entry = dict(card=card, provenance="curated teacher hypothetical", production_case_branch=False,
                     authoring="not_evaluated", qualification="not_evaluated",
                     numerical_completion="not_attempted", observation_data="not_attempted")
        stage = "model-authoring"
        try:
            assembly = build_card(card)
            old = assembly.parent.as_dict()
            entry.update(authoring="pass", qualification="pass",
                         parent_revision_id=assembly.as_dict()["parent_revision_id"],
                         assembly_revision_id=assembly.revision_id,
                         structural_operations=[c["operation"] for c in assembly.changes])
            stage = "numerical-solution"
            result = integrate_test_charge(assembly.scenario.spec)
            entry["numerical_completion"] = result.termination
            stage = "independent-physics"
            entry["independent_physics"] = check_physics(assembly, result)
            stage = "requested-observation"
            bundle = make_bundle(assembly, result)
            assert assembly.parent.as_dict() == old
            entry.update(outcome="verified-authored", observation_data="pass",
                         physical_samples=len(result.samples), parent_unchanged=True,
                         interpretation_fidelity={"status":"pass","evidence":["explicit qualified teacher hypothetical, actual change ledger"]},
                         physical_correctness={"status":"pass","evidence":["independent source formulas/quadrature, force and energy checks"]},
                         simulation_coverage={"status":"pass","represented":["declared fixed sources and qualified trajectory"],"excluded":["source response","unqualified 3D motion"]},
                         educational_usefulness={"status":"not_evaluated","evidence":["browser review recorded separately; not inferred from data"]},
                         textbook_answer_completeness={"status":"not_attempted","required_for_simulator_acceptance":False})
        except Exception as exc:
            entry.update(outcome="unexpected-failure", stage=stage,
                         exception=type(exc).__name__, reason=str(exc))
        cases.append(entry)
    for card in NEGATIVE:
        try:
            cases.append(negative_card(card))
        except Exception as exc:
            cases.append(dict(card=card, outcome="unexpected-failure",
                              exception=type(exc).__name__, reason=str(exc)))
    return dict(schema="conductor_sim.m22a_composability.v1",
                evaluation_contract="teaching-accompaniment.v1",
                charter="docs/m22a_charter.md",
                design="predeclared curated developer trial; not blind or representative exam coverage",
                case_results=cases,
                denominators=dict(preselected_positive=9, preselected_negative=7,
                                  unfamiliar_positive=3, browser_review="separate"),
                counts=dict(verified_authored=sum(c["outcome"]=="verified-authored" for c in cases),
                            expected_refusals=sum(c["outcome"]=="expected-refusal" for c in cases)),
                accepted=[c["outcome"] for c in cases] == ["verified-authored"]*9+["expected-refusal"]*7)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=POSITIVE)
    parser.add_argument("--grade", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.grade:
        record = grade_trial()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print(record["counts"], "accepted:", record["accepted"])
        if not record["accepted"]:
            raise SystemExit(1)
    elif args.case:
        export_case(args.case, args.output)
        print("Offline authored scenario and structural JSON:", args.output)
    else:
        parser.error("select --case or --grade")


if __name__ == "__main__":
    main()
