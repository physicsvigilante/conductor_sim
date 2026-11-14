"""M21b JEE source-anchored static E/V/work previews; no invented motion.

From the repository root:
  python examples/jee_potential_explorer.py --case 2021-equipotential --output output/2021-circle.html
  python examples/jee_potential_explorer.py --case 2012-shell --output output/2012-shell.html
  python examples/jee_potential_explorer.py --case ring-work --output output/ring-work.html

The reference questions have symbolic charge/radius where noted. All numbers
not given in the source are explicitly illustrative and preserve geometry.
"""
from __future__ import annotations

import argparse
from math import sqrt
from pathlib import Path

from conductor_sim import (
    FixedPointField, PrescribedField, UniformElectricField,
    UniformRingAxisField, UniformShellField,
)
from conductor_sim.visualization import SliceSpec
from conductor_sim.viewer import render_html
from conductor_sim.jee_potential_explorer import (
    sample_fixed_potential_map, sample_static_profile, endpoint_field_work,
    render_static_profile, write_observation,
)

CASES = ("2021-equipotential", "2012-shell", "ring-work")


def make_2021():
    """Official JEE Advanced 2021 Paper 1 Q9–10: V=0 equipotential circle."""
    Q = 1e-9  # Illustrative; source only specifies symbolic magnitude Q.
    sources = (
        FixedPointField(-Q, (0., 0., 0.)),
        FixedPointField(+Q/sqrt(3), (2., 0., 0.)),
    )
    field = PrescribedField(sources)
    plane = SliceSpec(
        origin_m=(2.5, 0., 0.), u=(1., 0., 0.), v=(0., 1., 0.),
        x_range_m=(-3., 3.), y_range_m=(-3., 3.), resolution=101)
    bundle = sample_fixed_potential_map(
        field, plane, source_annotations=(
            ("charge:negative", "Negative fixed charge -Q"),
            ("charge:positive", "Positive fixed charge +Q/sqrt(3)")),
        source_mask_m=.11, highlight_potential_v=0., line_steps=220)
    bundle["frames"][0]["notes"].extend([
        "Official JEE Advanced 2021 Paper 1 Q9–10: original -Q at (0,0) and +Q/sqrt(3) at (2,0) m.",
        "Numerical Q=1 nC is teacher illustrative; V=0 geometry is independent of Q.",
        "Analytical reference: circle centre (3,0) m, radius sqrt(3) m. Pink is the sampled zero-volt contour.",
        "No time evolution or test-particle trajectory."
    ])
    bundle["jee_anchor"] = dict(
        source="JEE Advanced 2021 Paper 1, official linked Physics Q9–10",
        url="https://jeeadv.ac.in/past_qps/2021_1_English.pdf",
        pdf_page=4, case="2021-equipotential",
        given_charge_relation="-Q at (0,0); +Q/sqrt(3) at (2,0)",
        illustrative_Q_c=Q,
        independent_circle_center_m=[3.,0.],
        independent_radius_m=sqrt(3.),
        source_fidelity="original source geometry, illustrative charge magnitude")
    return bundle


def make_2012():
    """2012 paper: thin fixed uniformly charged shell, true jump in |E|."""
    field = PrescribedField((UniformShellField(+1e-9, .1),))
    profile=sample_static_profile(
        field, source_annotations=(("surface:shell", "Uniform positive thin shell"),),
        origin_m=(0.,0.,0.), axis=(1.,0.,0.),
        start_m=0.,end_m=.24, sample_count=241, coordinate="radial")
    anchor=dict(
        source="IIT-JEE 2012 archived shell E(r)/V(r) graph, workbook physical PDF p15",
        case="2012-shell", pdf_file="02_conductors_half_size.pdf", pdf_page=15,
        original_given="Uniform positive surface charge density, ideal thin sphere of radius R",
        illustrative_Q_c=1e-9, illustrative_R_m=.1,
        source_fidelity="actual radial field/potential laws; numeric Q and R illustrative")
    html=render_static_profile(
        profile, title="IIT-JEE 2012 · Thin charged shell: E(r), V(r)",
        source_note=("Original question asks which E(r)/V(r) graph is physical. "
                     "Illustrative Q=+1 nC, R=0.10 m. This is a prescribed nonconducting surface charge, not a conducting metal solution."),
        surface_coordinate=.1)
    return html,dict(schema="conductor_sim.m21b_example.v1",anchor=anchor,profile=profile)


def _ring_schematic():
    """Schematic orientation only, not sampled off-axis field or forced path."""
    return '''<svg viewBox="0 0 840 232" role="img" aria-label="Rings A and B have a common horizontal world x axis, separation 40 cm; uniform E points vertically up" style="max-width:100%;height:auto" xmlns="http://www.w3.org/2000/svg">
      <rect width="840" height="232" fill="#fbfdff"/>
      <line x1="90" y1="121" x2="750" y2="121" stroke="#526777" stroke-dasharray="6 5" stroke-width="2"/>
      <path d="M744,116 L755,121 L744,126" stroke="#526777" fill="none" stroke-width="2"/>
      <text x="756" y="112" fill="#354c62" font-size="17">+x</text>
      <ellipse cx="185" cy="121" rx="27" ry="69" fill="#d7e9f5" fill-opacity=".36" stroke="#277cb7" stroke-width="3"/>
      <ellipse cx="563" cy="121" rx="27" ry="69" fill="#fbdddd" fill-opacity=".36" stroke="#bb4a35" stroke-width="3"/>
      <circle cx="185" cy="121" r="4" fill="#205078"/>
      <circle cx="563" cy="121" r="4" fill="#205078"/>
      <text x="165" y="37" font-size="19" fill="#235e8d">Ring A · +10 µC</text>
      <text x="550" y="37" font-size="19" fill="#a34235">Ring B · −20 µC</text>
      <text x="133" y="213" font-size="15">x=0 m (A)</text>
      <text x="511" y="213" font-size="15">x=0.40 m (B)</text>
      <line x1="382" y1="116" x2="382" y2="50" stroke="#118b69" stroke-width="4"/>
      <path d="M374,64 L382,47 L390,64" stroke="#118b69" stroke-width="3" fill="none"/>
      </svg>
      <p class="muted">E = 10⁵ N/C vertically upward (+y); A → B is endpoint displacement only. No physical guide or axial free path asserted.</p>'''


def make_ring():
    """Workbook p12 linked Q40–41: endpoint field work, never axial motion."""
    field=PrescribedField((
        UniformRingAxisField(+10e-6,.3,center_m=(0.,0.,0.),axis=(1.,0.,0.)),
        UniformRingAxisField(-20e-6,.3,center_m=(.4,0.,0.),axis=(1.,0.,0.)),
        UniformElectricField((0.,1e5,0.)),
    ))
    labels=(("ring:A","Ring A"),("ring:B","Ring B"),
            ("field:vertical","Upward applied E"))
    profile=sample_static_profile(
        field, source_annotations=labels,
        origin_m=(0.,0.,0.),axis=(1.,0.,0.),
        start_m=0.,end_m=.4,sample_count=201,coordinate="axial")
    work=endpoint_field_work(
        field,source_annotations=labels, probe_charge_c=+10e-6,
        start_m=(0.,0.,0.), end_m=(.4,0.,0.))
    anchor=dict(
        source="Workbook linked Q40–41 (physical PDF page 12)",
        case="ring-work",pdf_file="02_conductors_half_size.pdf",pdf_page=12,
        source_fidelity="all charge values, radii, separation, endpoints and transverse external field source-given",
        original_mass="symbolic m; no numerical value or 1D motion is authored",
        source_problem="Q40: electric work for hypothetical displacement A to B",
        original_external_field_v_m=[0.,1e5,0.])
    html=render_static_profile(
        profile,title="Workbook Q40 · Two charged rings and field work",
        source_note=("Source-given ring radii 0.30 m, separation 0.40 m; +10 µC and −20 µC; "
                     "probe +10 µC; applied E=10⁵ N/C up (+y). The plotted x-axis is the qualified ring symmetry axis. "
                     "The external field has a genuine transverse component, so no axial free trajectory is claimed."),
        work=work,geometry_svg=_ring_schematic())
    return html,dict(schema="conductor_sim.m21b_example.v1",anchor=anchor,
                     profile=profile, endpoint_work=work)


def make_case(case):
    if case=="2021-equipotential":
        bundle=make_2021()
        return render_html(bundle,title="JEE Advanced 2021 Q9–10 · V=0 equipotential circle"),bundle
    if case=="2012-shell":
        return make_2012()
    if case=="ring-work":
        return make_ring()
    raise ValueError("unknown source-anchored JEE example")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case",choices=CASES,default=CASES[0])
    ap.add_argument("--output",type=Path,default=Path("output/m21b-jee-potential.html"))
    args=ap.parse_args()
    html,audit=make_case(args.case)
    output,sidecar=write_observation(args.output,html,audit)
    print(args.case,":",output,"and",sidecar)
    print("Static source-based observations only; no browser field solve or invented dynamics.")


if __name__=="__main__":
    main()
