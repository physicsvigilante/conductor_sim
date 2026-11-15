"""M27a: source-faithful STATIC off-axis ring E/V explorer (no motion).

Original workbook p12 linked Q40–41 supplies both ring charges/radii,
separation and +world-y uniform E. The 2D observation window and display
exclusion width below are clearly teacher-chosen presentation parameters.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from conductor_sim import PrescribedField, UniformRingField, UniformElectricField
from conductor_sim.ring_field_observer import (
    sample_ring_field_slice, write_ring_field_slice,
)
from conductor_sim.visualization import SliceSpec

CASES=("workbook-two-rings", "single-ring")


def make_case(case):
    if case == "workbook-two-rings":
        # Workbook 02_conductors_half_size.pdf physical p12 Q40–41.
        # x is horizontal world axis, y is upwards. Transverse external
        # E_y must NOT be thrown out to pretend motion stays on the x axis.
        field = PrescribedField((
            UniformRingField(+10e-6,.3,center_m=(0.,0.,0.),axis=(1.,0.,0.)),
            UniformRingField(-20e-6,.3,center_m=(.4,0.,0.),axis=(1.,0.,0.)),
            UniformElectricField((0.,1e5,0.)),
        ))
        names=(("ring:A","Ring A"),("ring:B","Ring B"),
               ("field:upward","Upward prescribed field"))
        plane=SliceSpec(
            origin_m=(.2,0.,0.), u=(1.,0.,0.), v=(0.,0.,1.),
            x_range_m=(-.45,.45), y_range_m=(-.45,.45), resolution=37)
        mask=.012
    elif case == "single-ring":
        field=PrescribedField((UniformRingField(1e-9,.12),))
        names=(("ring:single","Single charged ring"),)
        plane=SliceSpec(
            origin_m=(0.,0.,0.), u=(1.,0.,0.), v=(0.,0.,1.),
            x_range_m=(-.22,.22), y_range_m=(-.22,.22), resolution=33)
        mask=.006
    else:
        raise ValueError("unknown M27a static source configuration")
    return sample_ring_field_slice(
        field,plane,source_annotations=names,exclusion_width_m=mask)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case",choices=CASES,default=CASES[0])
    ap.add_argument("--output",type=Path,default=Path("output/m27a-ring-field.html"))
    args=ap.parse_args()
    observation=make_case(args.case)
    write_ring_field_slice(args.output,observation,
         title=("Workbook Q40–41 · Off-axis E/V of Ring A and Ring B"
                if args.case=="workbook-two-rings" else
                "One charged ring · True off-axis static field"))
    print("M27a static E/V:",args.output)
    print("Saved JSON:",args.output.with_suffix(".json"))
    print("No particle path, guide force, conductor induction or browser physics.")


if __name__=="__main__":
    main()
