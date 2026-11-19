"""M39b: deterministic, completely offline topology-only teacher Gauss viewer.

Compartment positions are NOT geometrical coordinates or actual cavity shapes.
Only the independently qualified complete-face Gauss totals from M39a appear.
"""
from __future__ import annotations

from html import escape
from math import isfinite
from typing import Any

from .cavity_gauss import GaussCavityLedger, validate_cavity_gauss_ledger

SCHEMA = "conductor_sim.m39b_topology_view.v1"
TEACHER = "teacher_hypothetical_topology_gauss_only"
WARNING = "TOPOLOGY ONLY — NOT TO SCALE — CARD ORDER IS NOT PHYSICAL POSITION"
PROVENANCE = (
    "Teacher-authored source membership. No physical shape, coordinates or "
    "geometric charge containment was verified."
)
REFUSAL = (
    "Whole-face Gauss-law totals ONLY. No local E(x), V(x), sigma(x), "
    "forces, ground-wire current or real 3D field map."
)

_CSS = """
<style>
:root{color-scheme:light;font:16px/1.52 system-ui,-apple-system,"Segoe UI",sans-serif;
  color:#193149;background:#f4f7fb}
*{box-sizing:border-box}
body{margin:0;min-width:280px}
header{background:#112f49;color:white;padding:2rem max(1rem,calc((100vw - 1120px)/2));}
h1{font-size:clamp(1.6rem,3.2vw,2.4rem);line-height:1.2;margin:.4rem 0}
h2{font-size:1.3rem;margin:.25rem 0 .7rem}
h3{font-size:1.05rem;margin:.2rem 0}
p{margin:.45rem 0 1rem}
main,footer{max-width:1120px;padding:0 1rem;margin:1.4rem auto}
small,.muted{color:#40556b}
header .muted{color:#e2eaf2}
.kicker{font-weight:800;letter-spacing:.09em;text-transform:uppercase;font-size:.8rem}
.notice{border-left:5px solid #dca63a;padding:.75rem 1rem;background:#fff7e4;color:#4f3412}
section{margin:1.1rem 0 2rem}
.metal{border:3px dashed #477897;border-radius:16px;padding:1rem;background:#e9f3f9}
.frame-label{font-weight:650;margin-bottom:.8rem}
.cavity-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,220px),1fr));gap:1rem}
.cavity{border:2px solid #9db4c5;border-radius:12px;background:#fff;padding:1rem;min-width:0;overflow-wrap:anywhere}
.cavity h3{color:#123a59}
.cavity .tag{display:inline-block;font-size:.73rem;font-weight:700;background:#e8eef7;
  border-radius:12px;padding:.1rem .5rem;margin:.3rem 0}
.cavity .neutral{background:#f9eac8}
.cavity ul{padding-left:1.1rem;margin:.45rem 0 .9rem}
.cavity li{margin:.25rem 0}
code{font-size:.92em;overflow-wrap:anywhere}
.charge{font-variant-numeric:tabular-nums;white-space:normal;font-weight:700}
.derived{border-top:1px solid #c7d6e0;padding-top:.6rem}
.derived p{margin:.2rem 0}
.table-scroll{overflow-x:auto;border:1px solid #d0dce6;border-radius:12px;background:white}
table{border-collapse:collapse;width:100%;min-width:600px;font-variant-numeric:tabular-nums}
th,td{padding:.75rem;text-align:left;border-bottom:1px solid #dde6ed;vertical-align:top}
th{background:#edf3f8}
tbody tr:last-child th,tbody tr:last-child td{border-bottom:0}
thead th{background:#1d4764;color:white}
.truth{font-weight:700;color:#125735}
.nontruth{color:#5b4224}
footer{border-top:1px solid #cdd8e2;padding-top:1rem;padding-bottom:2rem}
@media(max-width:620px){header{padding:1.4rem 1rem}main{padding:0 .75rem}
  .metal{padding:.7rem}.cavity{padding:.8rem}body{font-size:15px}}
</style>
"""


def _si(value: Any, unit: str) -> str:
    """Round-trip-safe SI float, with genuine mathematical zero kept distinct."""
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError("topology viewer refuses nonfinite or Boolean SI values")
    number = float(value)
    if number == 0:
        return "0 " + unit
    return ("+" if number > 0 else "−") + repr(abs(number)) + " " + unit


def _record(ledger: GaussCavityLedger) -> dict:
    if type(ledger) is not GaussCavityLedger:
        raise TypeError("topology viewer requires real M39a GaussCavityLedger")
    validate_cavity_gauss_ledger(ledger)
    return ledger.as_dict()


def topology_view_manifest(isolated: GaussCavityLedger,
                           grounded: GaussCavityLedger) -> dict:
    """Return a versioned teacher viewer envelope from two authentic ledgers.

    Both states MUST have identical authored source/cavity inventory, order
    and permittivity, but separate correct isolated/grounded constraints.
    """
    a, b = _record(isolated), _record(grounded)
    if (a["conductor"]["constraint"] != "isolated_fixed_net_charge"
            or b["conductor"]["constraint"] != "grounded_zero_potential"):
        raise ValueError("viewer requires separate genuine isolated and grounded static states")
    if (a["conductor"]["conductor_id"] != b["conductor"]["conductor_id"]
            or a["permittivity_f_per_m"] != b["permittivity_f_per_m"]):
        raise ValueError("viewer paired states must share metal identity and permittivity")
    identity = lambda row: (
        row["cavity_id"], row["sealed"],
        tuple((s["source_id"], s["charge_c"]) for s in row["authored_sources"]),
    )
    if [identity(x) for x in a["cavities"]] != [identity(x) for x in b["cavities"]]:
        raise ValueError("viewer paired states require identical authored cavity/source inventory")
    if any(x["source_charge_sum_c"] != y["source_charge_sum_c"]
           or x["induced_inner_face_total_charge_c"] != y["induced_inner_face_total_charge_c"]
           for x, y in zip(a["cavities"], b["cavities"])):
        raise ValueError("paired Gauss face totals cannot differ for identical source inventory")
    if (a["source_role"] != TEACHER or b["source_role"] != TEACHER
            or a["original_workbook_items_newly_solved"] != 0
            or b["original_workbook_items_newly_solved"] != 0):
        raise ValueError("viewer must retain M39a teacher-only original-source attribution")
    return {
        "schema": SCHEMA,
        "source_role": TEACHER,
        "original_workbook_items_newly_solved": 0,
        "warning": WARNING,
        "membership_provenance": PROVENANCE,
        "unsupported": REFUSAL,
        "geometry_verified": False,
        "diagram_is_spatial_geometry": False,
        "time_evolution_defined": False,
        "metal_id": a["conductor"]["conductor_id"],
        "permittivity_f_per_m": a["permittivity_f_per_m"],
        "cavities": [
            {"cavity_id": row["cavity_id"],
             "authored_sources": row["authored_sources"],
             "source_charge_sum_c": row["source_charge_sum_c"],
             "induced_inner_face_total_charge_c": row["induced_inner_face_total_charge_c"]}
            for row in a["cavities"]
        ],
        "states": {"isolated": a, "grounded": b},
    }


def render_cavity_topology(isolated: GaussCavityLedger,
                           grounded: GaussCavityLedger) -> str:
    """Pure canonical HTML. NO script, remote asset, physical map or movement."""
    m = topology_view_manifest(isolated, grounded)
    e = lambda text: escape(str(text), quote=True)
    g = m["states"]["grounded"]
    i = m["states"]["isolated"]
    rows = []
    for index, cavity in enumerate(m["cavities"]):
        sources = cavity["authored_sources"]
        neutral = cavity["source_charge_sum_c"] == 0
        tag = ("empty cavity" if not sources else
               "signed cancellation (local sigma unknown)" if neutral
               else "charged cavity")
        entries = "".join(
            '<li id="m39b-source-{0}-{1}"><code>{2}</code>: '
            '<span class="charge">{3}</span></li>'.format(
                index, j, e(s["source_id"]), e(_si(s["charge_c"], "C"))
            ) for j, s in enumerate(sources)
        ) if sources else '<li id="m39b-source-empty-{0}">No sources assigned</li>'.format(index)
        rows.append(
            '<article class="cavity" id="m39b-card-{0}" data-cavity-id="{1}">'
            '<h3>Cavity <code id="m39b-cavity-{0}">{2}</code></h3>'
            '<span class="tag {3}">{4}</span>'
            '<p>Declared enclosed sources</p><ul>{5}</ul>'
            '<div class="derived"><p>Σ enclosed q: '
            '<strong class="charge" id="m39b-q-{0}">{6}</strong></p>'
            '<p>Entire inner face Q: '
            '<strong class="charge" id="m39b-inner-{0}">{7}</strong></p></div></article>'
            .format(index,e(cavity["cavity_id"]),e(cavity["cavity_id"]),
                    "neutral" if neutral else "",e(tag),entries,
                    e(_si(cavity["source_charge_sum_c"],"C")),
                    e(_si(cavity["induced_inner_face_total_charge_c"],"C")))
        )
    def value(key, state, item, unit):
        return '<td id="m39b-{0}-{1}" class="charge">{2}</td>'.format(
            key, state, e(_si(item, unit))
        )
    def tr(label, key, ivalue, gvalue, unit):
        return '<tr><th scope="row">{}</th>{}{}</tr>'.format(
            e(label),value(key,"isolated",ivalue,unit),
            value(key,"grounded",gvalue,unit)
        )
    table = "".join((
        tr("Whole metal net charge","metal",
           i["conductor"]["equilibrium_net_metal_charge_c"],
           g["conductor"]["equilibrium_net_metal_charge_c"],"C"),
        tr("Entire outer-face charge","outer",i["outer_face_total_charge_c"],
           g["outer_face_total_charge_c"],"C"),
        tr("Exterior Gauss electric flux","flux",i["exterior_electric_flux_v_m"],
           g["exterior_electric_flux_v_m"],"V·m"),
    ))
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" '
        'content="default-src &#39;none&#39;; style-src &#39;unsafe-inline&#39;; '
        'script-src &#39;none&#39;; connect-src &#39;none&#39;; '
        'font-src &#39;none&#39;; img-src &#39;none&#39;">'
        '<title>Teacher Gauss-law cavity topology only</title>' + _CSS +
        '</head><body><header>'
        '<div class="kicker" id="m39b-provenance">TEACHER HYPOTHETICAL · NO VERIFIED ORIGINAL JEE ANSWER</div>'
        '<h1 id="m39b-title">Gauss law · multiple sealed cavities</h1>'
        '<p class="muted">Exact complete-face charge totals for a single connected ideal conductor.</p>'
        '</header><main><p class="notice" id="m39b-warning">' + e(WARNING) + '</p>'
        '<p id="m39b-membership">' + e(PROVENANCE) + '</p>'
        '<section aria-labelledby="m39b-topology-heading">'
        '<h2 id="m39b-topology-heading">Authored cavity membership (not a physical cross-section)</h2>'
        '<div class="metal" id="m39b-metal-frame" role="group" '
        'aria-label="Abstract one-metal membership grouping, not geometry">'
        '<p class="frame-label">One connected metal: <code id="m39b-metal">' +
        e(m["metal_id"]) + '</code> · abstract enclosing frame only</p>'
        '<div class="cavity-grid" id="m39b-cavities">' + "".join(rows) +
        '</div></div></section>'
        '<section aria-labelledby="m39b-static-heading"><h2 id="m39b-static-heading">'
        'Two separate static hypothetical states — NOT time evolution</h2>'
        '<p id="m39b-pair">Same source IDs, charges, cavity membership and permittivity (' +
        e(_si(m["permittivity_f_per_m"],"F/m")) +
        '). No ground-wire current or transfer history is inferred.</p>'
        '<div class="table-scroll"><table id="m39b-summary">'
        '<caption>Verified whole-face Gauss totals (not local surface density)</caption>'
        '<thead><tr><th scope="col">Qualified property</th>'
        '<th scope="col">Isolated metal · fixed Q</th>'
        '<th scope="col">Grounded metal · zero V</th></tr></thead>'
        '<tbody>' + tr("Total enclosed source charge","sources",
                         i["total_interior_source_charge_c"],
                         g["total_interior_source_charge_c"],"C") +
        tr("Authored isolated metal charge","fixed",
           i["conductor"]["authored_isolated_metal_charge_c"],0.0,"C").replace(
               'id="m39b-fixed-grounded" class="charge">0 C',
               'id="m39b-fixed-grounded">Not a fixed-charge input'
           ) + table +
        '<tr><th scope="row">Exterior E=0 by grounded uniqueness</th>'
        '<td id="m39b-theorem-isolated" class="nontruth">Not generally implied for isolated metal</td>'
        '<td id="m39b-theorem-grounded" class="truth">Yes, under declared no-exterior-source boundary</td></tr>'
        '</tbody></table></div></section>'
        '<section aria-labelledby="m39b-limit-heading">'
        '<h2 id="m39b-limit-heading">What this diagram does not solve</h2>'
        '<p id="m39b-refusal">' + e(REFUSAL) + '</p>'
        '<p id="m39b-zero-warning">A zero entire-face net charge does NOT imply zero '
        'local surface charge density throughout that face.</p>'
        '<p id="m39b-original-count">Original JEE source problems newly solved: '
        '<strong>0</strong>. Workbook p11 and irregular three-cavity notes p70 '
        'remain unsupported for local 3D field/density.</p></section></main>'
        '<footer>Verified theorem values come from the immutable M39a Gauss ledger. '
        'Card positions, sizes and ordering are presentation only, never cavity geometry.</footer>'
        '</body></html>\n'
    )
