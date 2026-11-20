"""Two strictly source-anchored, offline JEE teaching walkthroughs.

This is a small display adapter over released, authentic static physics. It
does not author a new solver or certify arbitrary original JEE questions.
"""
from __future__ import annotations

import argparse
import json
from hashlib import sha256
from html import escape
from math import pi, sqrt
from pathlib import Path

from conductor_sim import (
    ConnectedShellBackend, ConnectionPlan, ElectricalLink,
    ElectrostaticProblem, FixedNetCharge, PrescribedField,
    SolidSphere, SphericalShell, UniformElectricField,
)
from conductor_sim.jee_potential_explorer import endpoint_field_work
from examples.solid_sphere_connected import scenario

PDF = "02_conductors_half_size.pdf"
PDF_SHA = "9dff3c02a9f1848d2c3d99cb1a6e7a44dc7f5a51f7a3a2b8cc92d10ce1fbca5d"
ELECTRON_C = 1.602176634e-19
EPSILON0 = 8.8541878128e-12
SCHEMA = "conductor_sim.jee12_original_walkthrough.v1"


def source_check(root: Path) -> None:
    if sha256((root / PDF).read_bytes()).hexdigest() != PDF_SHA:
        raise AssertionError("original workbook PDF does not match pinned bytes")
    records = json.loads((root / "docs" / "jee12_evaluation_cohort.json").read_text(encoding="utf-8"))
    cases = {x["id"]: x for x in records["cases"]}
    for ident, page, anchor in (
        ("p01-single-08", 1, "Single Correct Q8"),
        ("p03-single-27", 3, "Single Correct Q27"),
    ):
        row = cases[ident]
        if (row["source"], row["pdf_page"], row["printed_anchor"]) != (PDF, page, anchor):
            raise AssertionError("source anchor changed: " + ident)


def _source(case: str, page: int, anchor: str) -> dict:
    return {
        "pdf": PDF, "sha256": PDF_SHA, "pdf_page_index": page,
        "anchor": anchor, "case_id": case,
        "provenance": "original printed workbook exercise; not automatically official JEE exam",
    }


def work_record() -> dict:
    dx = sqrt(5.0 ** 2 - 3.0 ** 2)
    start, end = (0.0, 0.0, 0.0), (dx, 3.0, 0.0)
    def compute(e_x: float) -> dict:
        field = PrescribedField((UniformElectricField((e_x, 0.0, 0.0)),))
        return endpoint_field_work(
            field, source_annotations=(("E_source", "uniform electric field +x"),),
            probe_charge_c=-ELECTRON_C, start_m=start, end_m=end,
        )
    original = compute(10.0)
    hypothetical = compute(20.0)
    return {
        "schema": SCHEMA, "source": _source("p01-single-08", 1, "Single Correct Q8"),
        "mode": "static endpoint work; no particle trajectory or elapsed time",
        "source_given": {
            "uniform_field_x_v_m": 10.0, "electron_charge_c": -ELECTRON_C,
            "ab_m": 5.0, "transverse_bc_m": 3.0,
        },
        "derived": {
            "horizontal_displacement_m": dx, "start_m": list(start), "end_m": list(end),
        },
        "native": {
            "start_potential_v": original["start_potential_v"],
            "end_potential_v": original["end_potential_v"],
            "field_work_j": original["field_work_j"],
            "external_work_j": original["external_quasistatic_work_j"],
        },
        "original_asked": "work done by us on an electron from A to B",
        "printed_answer": {"option": 2, "external_work_eV": 40.0},
        "teacher_hypothetical": {
            "changed_field_x_v_m": 20.0,
            "native_field_work_j": hypothetical["field_work_j"],
            "native_external_work_j": hypothetical["external_quasistatic_work_j"],
            "role": "TEACHER HYPOTHETICAL, not printed givens",
        },
    }


def _equilibrium(result) -> dict:
    b, a = result.before, result.after
    def state(solution):
        solid = solution.conductor_by_name("solid")
        shell = solution.conductor_by_name("shell")
        return {
            "solid_net_charge_c": solid.net_charge_c,
            "solid_outer_face_c": solid.outer_surface_charge_c,
            "shell_net_charge_c": shell.net_charge_c,
            "shell_inner_face_c": shell.inner_surface_charge_c,
            "shell_outer_face_c": shell.outer_surface_charge_c,
            "solid_potential_v": solid.potential_v,
            "shell_potential_v": shell.potential_v,
        }
    return {
        "before": state(b), "after": state(a),
        "energy_before_j": result.radial_field_energy_before_j,
        "energy_after_j": result.radial_field_energy_after_j,
        "energy_released_j": result.released_field_energy_j,
    }


def energy_record() -> dict:
    original = scenario("pdf-p03-q27")
    solid, shell = original.initial.conductors
    if not isinstance(solid, SolidSphere) or not isinstance(shell, SphericalShell):
        raise AssertionError("original apparatus must contain an actual solid sphere and shell")
    if original.initial.charges:
        raise AssertionError("original source has no internal point-charge inventory")
    if (solid.outer_radius_m, shell.inner_radius_m, shell.outer_radius_m) != (0.1, 0.199999, 0.2):
        raise AssertionError("source-authored radial geometry changed")
    if not (len(original.links) == 1 and
            tuple(original.links[0].members) == ("solid", "shell") and
            original.links[0].mode == "floating"):
        raise AssertionError("source requires one charge-conserving link")
    native = ConnectedShellBackend().solve(original)
    teacher = ConnectionPlan(
        ElectrostaticProblem(conductors=(
            SolidSphere(0.1, FixedNetCharge(10e-6), name="solid"),
            SphericalShell(0.199999, 0.2, FixedNetCharge(0.0), name="shell"),
        )), (ElectricalLink(("solid", "shell")),),
    )
    teacher_native = ConnectedShellBackend().solve(teacher)
    k = 1.0 / (4.0 * pi * EPSILON0)
    ideal = 0.5 * k * (20e-6) ** 2 * (1.0 / 0.1 - 1.0 / 0.2)
    return {
        "schema": SCHEMA, "source": _source("p03-single-27", 3, "Single Correct Q27"),
        "mode": "two static equilibria; not current, heat-rate or time evolution",
        "source_given": {
            "solid_radius_m": 0.1, "shell_radius_m": 0.2,
            "initial_solid_charge_c": 20e-6, "initial_shell_net_charge_c": 0.0,
            "wire_connection": "floating charge-conserving",
        },
        "teacher_model_qualification": {
            "finite_shell_inner_radius_m": shell.inner_radius_m,
            "finite_shell_outer_radius_m": shell.outer_radius_m,
            "metal_thickness_m": shell.outer_radius_m - shell.inner_radius_m,
            "actual_inner_body": "SOLID conducting sphere, no interior cavity",
            "line_width_on_svg_is_not_physical_thickness": True,
        },
        "native": _equilibrium(native),
        "original_asked": "heat generated on joining the solid sphere and thin shell, inferred from loss of field energy",
        "printed_answer": {"option": 2, "heat_j_rounded": 9.0, "ideal_thin_limit_j": ideal},
        "teacher_hypothetical": {
            "changed_initial_solid_charge_c": 10e-6,
            "native_energy_released_j": teacher_native.released_field_energy_j,
            "role": "TEACHER HYPOTHETICAL, not printed givens",
        },
    }


def _value(value: float, unit: str, ident: str) -> str:
    if not isinstance(value, (int, float)):
        raise TypeError("numeric native observation required")
    return ('<output id="' + escape(ident) + '" data-si="' + format(value, ".17g") +
            '">' + format(value, ".10g") + ' ' + escape(unit) + '</output>')


CSS = """
:root{color-scheme:light;font-family:system-ui,-apple-system,sans-serif;color:#172539;background:#f5f7fa}
*{box-sizing:border-box}body{margin:0}main{max-width:1000px;margin:auto;padding:18px}
header{border-bottom:3px solid #173d69;padding-bottom:12px}h1{font-size:clamp(1.4rem,4vw,2rem);line-height:1.2}
h2{font-size:1.2rem;margin-top:0}h3{font-size:1.05rem}p{line-height:1.5}
section{background:white;border:1px solid #d9e2ee;border-radius:12px;padding:18px;margin:16px 0}
.diagram{display:flex;flex-wrap:wrap;gap:16px}.diagram>figure{flex:1 1 280px;min-width:0}
svg{display:block;width:100%;max-width:600px;height:auto;border:1px solid #e1e6ec;border-radius:10px;background:#fbfdff}
figcaption{font-size:.88rem;line-height:1.4;color:#3c4b5d;margin-top:7px}
dl{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,240px),1fr));gap:12px}
dl>div{padding:10px;border-radius:8px;background:#f1f5f9;min-width:0}dt{font-weight:650;color:#415269;margin-bottom:5px}
dd{margin:0;font-weight:650;overflow-wrap:anywhere}output{font-variant-numeric:tabular-nums}
.answer{border-left:5px solid #296a4c;background:#f4faf6}.hypothetical{border-left:5px solid #7d598f}
small,.muted{color:#586878}code{overflow-wrap:anywhere}
@media(max-width:420px){main{padding:10px}section{padding:13px}dl{grid-template-columns:minmax(0,1fr)}}
"""


def _page(title: str, body: str) -> str:
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>' + escape(title) + '</title><style>' + CSS + '</style></head>'
            '<body><main><header><h1>' + escape(title) + '</h1>'
            '<p>Original printed workbook exercise &middot; source facts and teacher '
            'hypotheticals are explicitly separate. All physics is precomputed '
            'by the released native engine.</p></header>' + body +
            '<footer><p class="muted">Static evidence only. No free particle path '
            'is inferred from prescribed displacement; no wire current or time '
            'trace is inferred from two equilibria. Standalone HTML edits are '
            'not cryptographically authenticated by the separate JSON.</p>'
            '</footer></main></body></html>')


WORK_TRIANGLE = """
<figure><svg viewBox="0 0 560 375" role="img" aria-label="True scale 5-3-4 right triangle with A at lower left, B at upper right, 4 m horizontal displacement and 3 m vertical displacement. Uniform E points right.">
<defs><marker id="field-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10z" fill="#1c62a1"/></marker></defs>
<line x1="100" y1="310" x2="420" y2="310" stroke="#59758d" stroke-width="2"/>
<line x1="420" y1="310" x2="420" y2="70" stroke="#59758d" stroke-width="2"/>
<line x1="100" y1="310" x2="420" y2="70" stroke="#bd5c32" stroke-width="4"/>
<path d="M398 310v-22h22" fill="none" stroke="#59758d" stroke-width="2"/>
<circle cx="100" cy="310" r="6" fill="#213e60"/><circle cx="420" cy="70" r="6" fill="#213e60"/>
<text x="79" y="332">A</text><text x="427" y="60">B</text><text x="427" y="329">C</text>
<text x="236" y="336">4 m (+x)</text><text x="428" y="195">3 m</text><text x="223" y="180">5 m</text>
<path d="M150 46h200" stroke="#1c62a1" stroke-width="5" marker-end="url(#field-arrow)"/>
<text x="168" y="33" fill="#1c62a1">E = +10 V/m</text></svg>
<figcaption>One metre corresponds to 80 diagram units in both axes. A→B is a specified endpoint displacement, not a time-resolved freely moving electron.</figcaption></figure>
"""


def work_html(r: dict) -> str:
    n, t = r["native"], r["teacher_hypothetical"]
    body = (
        '<section><h2>Source given — PDF p1 · Single Correct Q8</h2>'
        '<p>Uniform electric field +10 V/m along x; electron is moved <strong>by us</strong> '
        'from A to B along a 5 m displacement with 3 m transverse leg. '
        'Asked: work done by us. Printed choice (2): 40 eV.</p>'
        '<div class="diagram">' + WORK_TRIANGLE + '</div></section>'
        '<section class="answer"><h2>Original requested answer</h2><dl>'
        '<div><dt>Work by external agent (+)</dt><dd>' +
        _value(n["external_work_j"] / ELECTRON_C, "eV", "p1-external-ev") + '</dd></div>'
        '<div><dt>Work by electric field (−)</dt><dd>' +
        _value(n["field_work_j"] / ELECTRON_C, "eV", "p1-field-ev") + '</dd></div>'
        '</dl><p>Independent SI equation: W<sub>field</sub> = qEΔx = '
        '(−e)(10 V/m)(4 m) = −40 eV; '
        'W<sub>us</sub> = −W<sub>field</sub> = +40 eV.</p>'
        '<p>This is a conservative <strong>static endpoint-work</strong> result. '
        'It does not specify a free electron trajectory, travel time or mechanical guide.</p></section>'
        '<section class="hypothetical"><h2>TEACHER HYPOTHETICAL — not source given</h2>'
        '<p>Only the electric field is changed from +10 to +20 V/m; endpoints and '
        'electron are unchanged. Genuine separately recomputed native external work: ' +
        _value(t["native_external_work_j"] / ELECTRON_C, "eV", "p1-teacher-external-ev") +
        '. This is not an additional original exercise.</p></section>'
    )
    return _page("Workbook p1 Q8 — external work vs field work", body)


def _sphere_svg(stage: str, solid_q: float, shell_net: float,
                shell_inner: float, shell_outer: float) -> str:
    # Both panels use exactly 550 SVG units per metre: 0.10m -> 55, 0.20m -> 110.
    # Metal's 1um width is explicitly not drawn to physical stroke thickness.
    return ('''<figure><svg viewBox="0 0 320 320" role="img" aria-label="''' +
        escape(stage) + ''' static equilibrium; filled solid sphere radius 0.10 metre and concentric metal shell radius 0.20 metre.">
        <circle cx="155" cy="155" r="110" fill="none" stroke="#496d93" stroke-width="3"/>
        <circle cx="155" cy="155" r="55" fill="#98c3df" stroke="#225980" stroke-width="2"/>
        <circle cx="155" cy="155" r="2" fill="#225980"/>
        <text x="155" y="143" text-anchor="middle" font-size="12">SOLID</text>
        <text x="155" y="161" text-anchor="middle" font-size="12">R=0.10 m</text>
        <text x="155" y="180" text-anchor="middle" font-size="11">''' +
        escape(format(solid_q * 1e6, ".7g")) + ''' µC</text>
        <text x="155" y="30" text-anchor="middle" font-size="12">Shell R=0.20 m</text>
        <text x="155" y="285" text-anchor="middle" font-size="12">Shell net ''' +
        escape(format(shell_net * 1e6, ".7g")) + ''' µC</text>
        <path d="M155 155h110" stroke="#7b8c9c" stroke-dasharray="3 4" stroke-width="1"/>
        <text x="160" y="226" font-size="11">gap</text>
        </svg><figcaption><strong>''' + escape(stage) + '''</strong>. Native shell faces: inner ''' +
        escape(format(shell_inner * 1e6, ".7g")) +
        ''' µC; outer ''' + escape(format(shell_outer * 1e6, ".7g")) +
        ''' µC. Same physical length scale in both panels; visible shell line width is not the 1 µm material thickness.</figcaption></figure>''')


def energy_html(r: dict) -> str:
    n, t = r["native"], r["teacher_hypothetical"]
    before, after = n["before"], n["after"]
    body = (
        '<section><h2>Source given — PDF p3 · Single Correct Q27</h2>'
        '<p>An initially isolated <strong>solid conducting sphere</strong> of radius '
        '0.10 m carries +20 µC. A concentric thin metal shell of radius 0.20 m '
        'is initially uncharged. An ideal conducting wire joins them. '
        'Asked: heat released; printed choice (2), 9 J.</p>'
        '<div class="diagram">' +
        _sphere_svg("BEFORE connecting", before["solid_net_charge_c"],
                    before["shell_net_charge_c"], before["shell_inner_face_c"],
                    before["shell_outer_face_c"]) +
        _sphere_svg("AFTER connecting", after["solid_net_charge_c"],
                    after["shell_net_charge_c"], after["shell_inner_face_c"],
                    after["shell_outer_face_c"]) + '</div>'
        '<p class="muted">Qualified native model: finite shell spans '
        '0.199999–0.200000 m (teacher-chosen thickness 1 µm). '
        'The original printed ideal thin-shell limit has no finite thickness. '
        'The inner conductor is a <strong>filled solid</strong>, not a hollow proxy.</p></section>'
        '<section class="answer"><h2>Original requested answer — field-energy decrease</h2><dl>'
        '<div><dt>Initial static field energy</dt><dd>' +
        _value(n["energy_before_j"], "J", "p3-before-j") + '</dd></div>'
        '<div><dt>Final static field energy</dt><dd>' +
        _value(n["energy_after_j"], "J", "p3-after-j") + '</dd></div>'
        '<div><dt>Released field energy (native finite shell)</dt><dd>' +
        _value(n["energy_released_j"], "J", "p3-released-j") + '</dd></div>'
        '<div><dt>Exact ideal thin-shell model limit</dt><dd>' +
        _value(r["printed_answer"]["ideal_thin_limit_j"], "J", "p3-ideal-j") + '</dd></div>'
        '</dl><p>Independent SI equation for ideal conductor surfaces: '
        'ΔU = kQ²(1/R<sub>solid</sub> − 1/R<sub>shell</sub>)/2 '
        '= 8.98755 J ≈ <strong>9 J, option (2)</strong>. '
        'Native field energy includes the separately disclosed 1 µm shell thickness. '
        'Charge is conserved; the field in the gap disappears on connection.</p>'
        '<p>These are <strong>two separately solved static equilibria</strong>. '
        'The energy decrease can represent ideal dissipated heat, but no physical '
        'wire-current, connection-time, radiation or heat-rate model is present.</p></section>'
        '<section class="hypothetical"><h2>TEACHER HYPOTHETICAL — not source given</h2>'
        '<p>With initial solid charge changed to +10 µC, the same native model '
        'recomputes the released field energy as ' +
        _value(t["native_energy_released_j"], "J", "p3-teacher-released-j") +
        '. This static variation is not the printed Q27 apparatus.</p></section>'
    )
    return _page("Workbook p3 Q27 — solid sphere, shell and static energy", body)


def generate(root: Path, out: Path) -> dict:
    source_check(root)
    out.mkdir(parents=True, exist_ok=True)
    records = (
        ("p1-external-work", work_record(), work_html),
        ("p3-solid-shell-energy", energy_record(), energy_html),
    )
    manifest = {"schema": "conductor_sim.jee12_walkthrough_index.v1",
                "original_complete_answers_certified_in_frozen_jee12_cohort": 0,
                "pages": []}
    for name, record, renderer in records:
        (out / (name + ".json")).write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (out / (name + ".html")).write_text(renderer(record), encoding="utf-8")
        manifest["pages"].append({"case_id": record["source"]["case_id"],
                                  "json": name + ".json", "html": name + ".html"})
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/jee12-walkthroughs"))
    args = parser.parse_args()
    manifest = generate(Path(__file__).resolve().parents[1], args.output_dir)
    print("Native two-case JEE-12 static teaching walkthroughs:", manifest["pages"])
    print("The frozen original-answer certification ledger is unchanged.")


if __name__ == "__main__":
    main()
