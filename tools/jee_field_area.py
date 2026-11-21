"""Export field area and potential walkthroughs for workbook Q39, Q40 and Q55."""
import argparse
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from math import sqrt
from pathlib import Path

from conductor_sim.affine_fields import AffineFieldAssembly
from conductor_sim.axial_field_graph import AxialFieldGraph
from conductor_sim.prescribed_fields import UniformElectricField
from tools.jee12_walkthroughs import PDF, PDF_SHA


@dataclass(frozen=True)
class Case:
    number: int
    title: str
    givens: str
    graph: AxialFieldGraph
    target_m: int
    difference_requested: bool


def cases():
    source = UniformElectricField((50, 50*sqrt(3), 0), reference_point_m=(-2, 0, 0))
    uniform = AxialFieldGraph.from_affine(AffineFieldAssembly((source,)), -2, 0)
    return (
        Case(39, "An oblique uniform field", "E = 100 V/m makes a 60° angle with +x. A is 2 m to the left "
             "of O. Only Eₓ = 50 V/m contributes along A → O. The workbook asks for V(O) − V(A). "
             "For the potential plot we choose V(A) = 0 V.", uniform, 0, True),
        Case(40, "A field that changes slope", "The other field components are zero. The workbook gives V(0) = 10 V and asks for V at 2 m. "
             "Its graph passes through −20 V/m at 2 m, changes sign at 3 m and reaches +20 V/m at 4 m.",
             AxialFieldGraph(((0, 0), (2, -20), (4, 20), (5, 20)), 10), 2, False),
    )


def _plot(number, graph, target_m, potential, *, shade_right=False):
    start, end = graph.knots[0][0], graph.knots[-1][0]
    # Sample each authored segment and include the exact marker position.
    grid = [a + (b-a)*Fraction(j, 20) for (a, _), (b, _) in zip(graph.knots, graph.knots[1:])
            for j in range(20)] + [end]
    grid = sorted(set(grid) | {target_m})
    observe = graph.potential_v if potential else graph.axial_field_v_m
    values = [observe(s) for s in grid]
    low, high = min(0, *values), max(0, *values)
    if high == low:
        low, high = low-1, high+1
    def x(s):
        return 62 + 330*float((s-start)/(end-start))
    def y(value):
        return 28 + 100*(high-value)/(high-low)
    curve = " ".join(f"{x(s):.2f},{y(v):.2f}" for s, v in zip(grid, values))
    fill = ""
    if not potential:
        def shade(left, right, color):
            area = " ".join(f"{x(s):.2f},{y(v):.2f}" for s, v in zip(grid, values) if left <= s <= right)
            return f'<polygon points="{x(left):.2f},{y(0):.2f} {area} {x(right):.2f},{y(0):.2f}" fill="{color}"/>'
        fill = shade(start, target_m, "#c6e4df")
        if shade_right:
            fill += shade(target_m, end, "#f4cba2")
    label = "Potential (V)" if potential else "Axial field (V/m)"
    ticks = "".join(f'<text x="56" y="{y(v)+4:.1f}" text-anchor="end">{v:g}</text>' for v in sorted({low, high, 0}))
    positions = range(int(start), int(end)+1)
    labels = "".join(f'<text x="{x(s):.2f}" y="151" text-anchor="middle">{float(s):g}</text>' for s in positions)
    return f'''<svg viewBox="0 0 416 175" role="img" aria-label="Q{number} {label}">
    <text x="5" y="17">{label}</text>{fill}{ticks}{labels}
    <path d="M62 26 V130 M62 {y(0):.2f} H392" stroke="#839494" fill="none"/>
    <polyline points="{curve}" fill="none" stroke="#145e68" stroke-width="3"/>
    <circle cx="{x(target_m):.2f}" cy="{y(observe(target_m)):.2f}" r="4" fill="#963815"/>
    <text x="227" y="172" text-anchor="middle">x (m)</text></svg>'''


def _panel(case):
    start = case.graph.knots[0][0]
    # Field work per unit positive charge equals the signed area in volts.
    area = case.graph.field_work_j(1, start, case.target_m)
    change = -area
    target = case.graph.potential_v(case.target_m)
    answer = change if case.difference_requested else target
    requested = "V(O) − V(A)" if case.difference_requested else "V(2 m)"
    return f'''<section id="q{case.number}"><h2>Q{case.number} · {case.title}</h2><p>{case.givens}</p>
    <div class="plots">{_plot(case.number, case.graph, case.target_m, False)}{_plot(case.number, case.graph, case.target_m, True)}</div>
    <p>Shading covers the field integral to the requested position. The dot marks that position on both graphs.</p>
    <dl><dt>Signed field area</dt><dd>{area:+g} V</dd><dt>Potential change</dt><dd>{change:+g} V</dd>
    <dt>Requested {requested}</dt><dd><strong>{answer:+g} V</strong> · option 4</dd></dl></section>'''


def _equal_drop_panel():
    start, end = 1, 3
    field = AxialFieldGraph(((start, 2), (end, 18)), quadratic_v_m3=(2,))
    lo, hi = field.equal_drop_bracket_m(start, end, tolerance_m=Fraction(1, 10**12))
    middle = (lo+hi)/2
    first = field.potential_v(middle) - field.potential_v(start)
    second = field.potential_v(end) - field.potential_v(middle)
    return f'''<section id="q55"><h2>Q55 · Equal drops between equipotential planes</h2>
    <p>Workbook page 5 gives Eₓ = 2x² on the positive x axis. The two potential drops are equal.
    A stronger field needs less distance for the same drop, so <strong>x₁ &gt; x₂ · option 2</strong>.</p>
    <h3>Teacher example</h3><p>Choose outer planes at 1 m and 3 m and set V(1 m) = 0 V.
    The dot locates the middle plane. The two shaded regions have equal field integrals.</p>
    <div class="plots">{_plot(55, field, middle, False, shade_right=True)}{_plot(55, field, middle, True)}</div>
    <dl><dt>Middle plane</dt><dd>{float(middle):.3f} m</dd>
    <dt>First spacing</dt><dd>{float(middle-start):.3f} m</dd>
    <dt>Second spacing</dt><dd>{float(end-middle):.3f} m</dd>
    <dt>First potential change</dt><dd>{first:+.3f} V</dd>
    <dt>Second potential change</dt><dd>{second:+.3f} V</dd></dl></section>'''


def generate(output):
    if sha256((Path(__file__).resolve().parents[1] / PDF).read_bytes()).hexdigest() != PDF_SHA:
        raise ValueError("workbook differs from verified source")
    markup = "".join(_panel(case) for case in cases()) + _equal_drop_panel()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>From field area to potential</title><style>
    *{{box-sizing:border-box}}body{{font:1rem/1.5 system-ui;color:#182d31;max-width:56rem;padding:1rem;margin:auto}}
    h1,h2{{line-height:1.2}}section{{border-top:1px solid #a2b3b4;margin:1.4rem 0}}
    .plots{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(17rem,100%),1fr));gap:.75rem}}
    svg{{width:100%;height:auto}}svg text{{font:14px system-ui;fill:#182d31}}
    dl{{display:grid;grid-template-columns:max-content 1fr;gap:.4rem 1rem}}dd{{margin:0}}
    </style><main><h1>From field area to potential</h1><p>Workbook page 4, Q39 and Q40, and page 5, Q55.
    Along x, the potential change is the negative signed area under the axial field curve.</p>{markup}</main></html>''', encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    generate(parser.parse_args().output)
