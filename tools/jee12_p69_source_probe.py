"""Developer source-fidelity check: notes p69 Illustration 7, q shifts in one cavity.

The original source asks for distributions on BOTH faces of a spherical shell
with net 3q. The source illustrates central q versus displaced +q, not a
physical time evolution. Geometry and SI numerical values below are TEACHER
instantiations of the symbolic source; no original numeric answer is claimed.

Two separate native M40a static solutions are checked against independently
computed Gauss totals and real-plus-Kelvin-image boundary fields.
"""
from __future__ import annotations

from hashlib import sha256
from math import hypot, isclose, pi, sqrt
from pathlib import Path
import json

from conductor_sim.cavity_gauss import CavityCharge, GaussCavityAssembly, SealedCavity
from conductor_sim.model import EPSILON_0, FixedNetCharge
from conductor_sim.spherical_multicavity import (
    NamedSourcePosition, SphericalMetalGeometry, SphericalSigmaModel,
    SphericalVoid, solve_spherical_sigma,
)

PDF_SHA = "8d0d9b5cf821219058cfc4b7685a39b777e460944df44c3f9cffcb68157a60e5"
DIRECTIONS = ((1., 0., 0.), (-1., 0., 0.), (0., 1., 0.),
              (0., -1., 0.), (0., 0., 1.), (0., 0., -1.),
              (1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3)))


def _model(q: float, metal_q: float, source_position: tuple[float, float, float]):
    # Exactly ONE spherical cavity, one real point source and one connected
    # metal with the source's 3q conductor charge (no invented exterior source).
    ledger = GaussCavityAssembly(
        "Illustration7SphericalShell",
        (SealedCavity("sealed_cavity", (CavityCharge("original_q", q),)),),
        FixedNetCharge(metal_q), EPSILON_0,
    )
    geom = SphericalMetalGeometry(
        (0., 0., 0.), 1.2,
        (SphericalVoid("sealed_cavity", (0., 0., 0.), .8,
                       (NamedSourcePosition("original_q", source_position),)),),
    )
    return SphericalSigmaModel(ledger, geom)


def _independent_image_field_sigma(q: float, R: float, source, direction) -> float:
    """Compute conductor INNER sigma using real and Kelvin-image E·normal.

    Distinct from production's closed-form spherical Poisson kernel: construct
    an EXTERIOR Kelvin image of q, evaluate both Coulomb fields at the actual
    original sphere ray, then use sigma = -epsilon*E.radial. No image at the
    origin case is needed: centered source gives exactly uniform sigma.
    """
    d = hypot(*source)
    if d == 0:
        return -q / (4 * pi * R * R)
    image_q = -q * R / d
    image = tuple(R * R * x / (d * d) for x in source)
    point = tuple(R * x for x in direction)
    normal_field = 0.0
    for charge, position in ((q, source), (image_q, image)):
        relative = tuple(point[k] - position[k] for k in range(3))
        distance = hypot(*relative)
        normal_field += charge * sum(relative[k] * direction[k] for k in range(3)) / distance**3
    return -normal_field / (4 * pi)


def _one_teacher_case(q: float, shifted: tuple[float, float, float]):
    baseline = _model(q, 3*q, (0., 0., 0.))
    revised = _model(q, 3*q, shifted)
    before = solve_spherical_sigma(baseline)
    after = solve_spherical_sigma(revised)
    for sol in (before, after):
        record = sol.as_dict([("sealed_cavity", DIRECTIONS[0]), ("outer", DIRECTIONS[1])])
        inner = record["gauss"]["cavities"][0]["induced_inner_face_total_charge_c"]
        outer = record["gauss"]["outer_face_total_charge_c"]
        if not isclose(inner, -q, rel_tol=1e-13, abs_tol=1e-20):
            raise AssertionError("inner Gauss total must be -q")
        if not isclose(outer, 4*q, rel_tol=1e-13, abs_tol=1e-20):
            raise AssertionError("outer Gauss total must be +4q")
    samples = []
    central = []
    shifted_values = []
    for direction in DIRECTIONS:
        a = before.inner_surface_charge_density("sealed_cavity", direction)
        b = after.inner_surface_charge_density("sealed_cavity", direction)
        aref = _independent_image_field_sigma(q, .8, (0.,0.,0.), direction)
        bref = _independent_image_field_sigma(q, .8, shifted, direction)
        for label, native, expected in (("central", a, aref), ("shifted", b, bref)):
            if not isclose(native, expected, rel_tol=3e-13, abs_tol=3e-23):
                raise AssertionError(f"{label} signed sigma {native!r} != real+Kelvin {expected!r}")
        central.append(a)
        shifted_values.append(b)
        samples.append(dict(input_original_direction=list(direction),
                            sigma_centered_c_per_m2=a,
                            sigma_shifted_c_per_m2=b,
                            independent_centered_c_per_m2=aref,
                            independent_shifted_c_per_m2=bref))
    if max(central) - min(central) > abs(q) / (4*pi*.8**2) * 1e-13:
        raise AssertionError("source-central cavity should have UNIFORM negative-q density")
    if max(shifted_values) - min(shifted_values) < abs(q) / (4*pi*.8**2) * 1e-2:
        raise AssertionError("shifted point source must give real NONuniform inner density")
    outer_before = before.outer_surface_charge_density(DIRECTIONS[0])
    outer_after = after.outer_surface_charge_density(DIRECTIONS[4])
    expected_outer = 4*q / (4*pi*1.2**2)
    if not (isclose(outer_before, expected_outer, rel_tol=1e-13, abs_tol=3e-23)
            and isclose(outer_after, expected_outer, rel_tol=1e-13, abs_tol=3e-23)):
        raise AssertionError("original unchanged external +4q uniform density lost")
    if baseline.geometry.voids[0].positions[0].position_m != (0.,0.,0.):
        raise AssertionError("baseline mutated by static teacher position hypothesis")
    return dict(teacher_q_c=q, teacher_conductor_net_charge_c=3*q,
                teacher_inner_radius_m=.8, teacher_outer_radius_m=1.2,
                teacher_centered_source_position_m=[0.,0.,0.],
                teacher_shifted_source_position_m=list(shifted),
                native_inner_face_charge_c=-q, native_outer_face_charge_c=4*q,
                outer_signed_uniform_density_c_per_m2=outer_before,
                inner_centered_uniform=True,
                inner_shifted_nonuniform=True, discrete_signed_samples=samples)


def check(root: Path) -> dict:
    source_pdf = root / "POTENTIAL_CONDUCTORS1.pdf"
    if sha256(source_pdf.read_bytes()).hexdigest() != PDF_SHA:
        raise AssertionError("original notes p69 frozen SHA256 mismatch")
    historical=json.loads((root / "docs/m36a_jee30_coverage.json").read_text(encoding="utf-8"))
    case=next(x for x in historical["cases"] if x["id"] == "p69-cavity-central-vs-shifted")
    if (case["pdf_page"] != 69 or case["printed_anchor"] != "Illustration 7"
            or case["source"] != source_pdf.name):
        raise AssertionError("historical page/question attribution changed")
    return dict(
        schema="conductor_sim.jee12_p69_developer_source_probe.v1",
        status="DEVELOPER ORIGINAL-SOURCE GEOMETRY and KELVIN TEST, NOT INDEPENDENT REVIEW",
        source=dict(pdf=source_pdf.name, sha256=PDF_SHA,
                    page=69, printed_anchor="Illustration 7",
                    provenance="notes illustration; not automatically official JEE exam",
                    source_given_metal_charge="3q", source_given_point_charge="q",
                    asked="charge distribution on each surface in central versus displaced configurations",
                    original_qualitative_results=["inner -q uniform if centered",
                                                  "inner -q nonuniform if displaced",
                                                  "outer +4q uniform in both"]),
        teacher_signed_static_cases=[
            _one_teacher_case(+1e-9, (.24,.07,.12)),
            _one_teacher_case(-.6e-9, (.12,-.15,.23)),
        ],
        original_numeric_values_claimed=False,
        no_physical_elapsed_time_or_current_claimed=True,
        newly_independently_browser_verified=False,
        original_source_problem_newly_independently_reviewed=False,
        raw_density_values_are_discrete_samples_not_integrals=True,
    )


if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    result=check(Path(__file__).resolve().parents[1])
    encoded=json.dumps(result,indent=2,allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(encoded,encoding="utf-8")
    else:
        print(encoded)
