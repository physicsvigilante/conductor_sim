"""Export an offline teaching comparison for workbook page 3, Single Correct Q35."""
import argparse
from fractions import Fraction
from hashlib import sha256
from math import pi, sqrt
from pathlib import Path

from conductor_sim.model import EPSILON_0
from conductor_sim.path_barriers import straight_path_energy_profile
from conductor_sim.prescribed_fields import PrescribedField, UniformElectricField, UniformSphereField
from tools.jee12_walkthroughs import PDF, PDF_SHA


def _profiles():
    radius, density = 2., 1e-9
    length = sqrt(3) * radius / 2
    ends = ((-length, radius/2, 0), (length, radius/2, 0))
    sphere = UniformSphereField(4*pi*radius**3*density/3, radius)
    bias = density * length / (6*EPSILON_0)
    fields = (PrescribedField((sphere,)), PrescribedField((sphere, UniformElectricField((bias, 0, 0)))))
    return bias, [straight_path_energy_profile(field, 1, *ends) for field in fields]


def render():
    bias, profiles = _profiles()
    colors = ("#11606e", "#963d18")
    names = ("Sphere alone", "Sphere with axial bias")
    fractions = [Fraction(i, 80) for i in range(81)]
    peaks = [profile.barrier() for profile in profiles]
    high = max(p.rise_j for p in peaks)
    low = min(profile.energy_change_j(1) for profile in profiles)
    def y(value):
        return 36 + 146 * (high-value)/(high-low)
    def x(s):
        return 76 + 376*float(s)
    curves, rows = [], []
    for name, color, profile, peak in zip(names, colors, profiles, peaks):
        points = " ".join(f"{x(s):.2f},{y(profile.energy_change_j(s)):.2f}" for s in fractions)
        dashed = ' stroke-dasharray="7 4"' if name == names[1] else ""
        curves.append(f'<polyline fill="none" stroke="{color}"{dashed} stroke-width="3" points="{points}"/>'
                      f'<circle fill="{color}" cx="{x(peak.peak_fraction):.2f}" cy="{y(peak.rise_j):.2f}" r="4"/>')
        rows.append(f'<tr><th scope="row" style="color:{color}">{name}</th>'
                    f'<td>{float(peak.peak_fraction):.0%}</td><td>{peak.rise_j:.2f} J</td>'
                    f'<td>{profile.energy_change_j(1):.2f} J</td></tr>')
    ticks = "".join(f'<text x="68" y="{y(v)+4:.1f}" text-anchor="end">{v:.1f}</text>' for v in (high, 0, low))
    return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Energy inside the tunnel</title>
<style>
*{{box-sizing:border-box}}body{{max-width:44rem;margin:2rem auto;padding:0 1rem;color:#172c30;font:1rem/1.5 system-ui}}
h1,h2{{line-height:1.2}}svg{{width:100%;height:auto}}svg text{{font:14px system-ui;fill:#172c30}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{text-align:left;padding:.45rem;border-bottom:1px solid #afbdbf}}
@media(max-width:420px){{th,td{{padding:.25rem;font-size:.875rem}}}}
</style><main><h1>Energy inside the tunnel</h1>
<p>Workbook page 3, Single Correct Q35. A unit positive charge enters a straight tunnel offset by R/2
from the centre of a uniformly charged insulating sphere. Assume a narrow, frictionless guide and positive density ρ.</p>
<h2>Why equal endpoint potentials still require launch energy</h2>
<p>The field along the chord is Eₓ = ρx/(3ε₀), with x measured from the midpoint.
The charge climbs to a maximum energy rise of ρR²/(8ε₀) there.
The threshold speed is √(ρR²/(4mε₀)), option 1. At that exact speed the charge approaches
the midpoint asymptotically. Finite-time passage requires a higher speed.</p>
<h2>Change the field along the guide</h2>
<p>Teacher-selected values: R = 2 m and ρ = 1 nC/m³. The second curve adds a uniform field of
{bias:.2f} V/m directed from entry to exit. The solid curve shows the sphere alone. The dashed curve includes the bias. Both measure energy relative to entry.</p>
<svg viewBox="0 0 490 235" role="img" aria-label="Potential energy change along the guide, with maxima marked for both fields">
<text x="10" y="18">ΔU (J)</text><path d="M76 31 V184 M76 {y(0):.2f} H452" stroke="#819497" fill="none"/>
{ticks}{''.join(curves)}<text x="76" y="207" text-anchor="middle">0%</text>
<text x="264" y="207" text-anchor="middle">50%</text><text x="452" y="207" text-anchor="middle">100%</text>
<text x="264" y="230" text-anchor="middle">Position from entry to exit</text></svg>
<table><caption>Native energy observations</caption><thead><tr><th>Field</th><th>Peak position</th>
<th>Peak rise</th><th>Exit change</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p>The axial bias moves the barrier toward the entry. The negative exit change means the charge finishes at
a lower potential energy than it started with.</p></main></html>"""


def generate(output):
    workbook = Path(__file__).resolve().parents[1] / PDF
    if sha256(workbook.read_bytes()).hexdigest() != PDF_SHA:
        raise ValueError("workbook differs from the verified source")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    generate(parser.parse_args().output)
