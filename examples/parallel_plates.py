"""M12a parallel infinite-plate analytic visualization (no FEMM required)."""
from __future__ import annotations

import argparse
from pathlib import Path

from conductor_sim import (
    ConductingPlane, ElectrostaticProblem, FixedPotential, Grounded,
    ParallelPlateBackend, Femm5ParallelPlateBackend, PointCharge,
)
from conductor_sim.visualization import SliceSpec, sample_sweep, write_visualization


def plates(lower, upper, charges=()):
    return ElectrostaticProblem(
        charges=tuple(charges),
        conductors=(
            ConductingPlane(constraint=lower, name="lower"),
            ConductingPlane(point_m=(0.,0.,.20), normal=(0.,0.,-1.),
                            constraint=upper, name="upper"),
        ),
    )


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='output/parallel_plates.html')
    parser.add_argument('--backend',choices=('analytical','femm'),default='analytical')
    parser.add_argument('--resolution',type=int,default=31)
    parser.add_argument('--image-order',type=int,default=48)
    args=parser.parse_args()
    scenarios=[
        ('20 V capacitor, no point charges',plates(Grounded(),FixedPotential(20.))),
        ('Grounded plates with one +1 nC source',plates(Grounded(),Grounded(),
               (PointCharge(1e-9,(0.,0.,.065)),))),
        ('Grounded plates with two signed sources',plates(Grounded(),Grounded(),
               (PointCharge(1e-9,(0.,0.,.065)),PointCharge(-.5e-9,(0.,0.,.14))))),
    ]
    reference=(ParallelPlateBackend(image_order=args.image_order) if args.backend=='analytical'
               else Femm5ParallelPlateBackend())
    plane=SliceSpec.meridian(scenarios[0][1],extent_m=.29,resolution=args.resolution)
    # Explicit middle-gap seeds show capacitor field traces even with zero
    # point charges. These are visualization seeds, not fictitious sources.
    seeds=[((x,.10),direction) for x in (-.19,-.095,0.,.095,.19)
           for direction in (-1,1)]
    bundle=sample_sweep(scenarios,reference,plane=plane,seeds=seeds,
                        line_steps=(180 if args.backend=='femm' else 360))
    output=write_visualization(bundle,Path(args.output))
    print(f"M12 {args.backend} parallel-plate preview: {output}")
    print("Source-free capacitor field: 100 V/m, uniform between the plates")
    for label,problem in scenarios:
        solution=reference.solve(problem)
        try:
            totals=[(r.name,r.net_charge_c) for r in solution.conductor_results]
            print(f"{label}: electrode totals (C; None means undefined): {totals}")
            if args.backend=='femm':
                print("  finite radius (m):",solution.outer_radius_m)
                print("  independently measured FEMM sources (C):",
                      [(s.name,s.measured_charge_c) for s in solution.numerical_sources])
        finally:
            close=getattr(solution,'close',None)
            if callable(close):
                close()
    if args.backend=='analytical':
        print("Charged states use finite-order images, not a live FEMM solve.")
    else:
        print("Real FEMM charges are measured on finite electrode disks with insulated side.")


if __name__=='__main__':
    main()
