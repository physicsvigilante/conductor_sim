"""M12a parallel infinite-plate analytic visualization (no FEMM required)."""
from __future__ import annotations

import argparse
from pathlib import Path

from conductor_sim import (
    ConductingPlane, ElectrostaticProblem, FixedPotential, Grounded,
    ParallelPlateBackend, PointCharge,
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
    reference=ParallelPlateBackend(image_order=args.image_order)
    plane=SliceSpec.meridian(scenarios[0][1],extent_m=.29,resolution=args.resolution)
    # Explicit middle-gap seeds show capacitor field traces even with zero
    # point charges. These are visualization seeds, not fictitious sources.
    seeds=[((x,.10),direction) for x in (-.19,-.095,0.,.095,.19)
           for direction in (-1,1)]
    bundle=sample_sweep(scenarios,reference,plane=plane,seeds=seeds,line_steps=360)
    output=write_visualization(bundle,Path(args.output))
    print(f"Analytical M12a preview: {output}")
    print("Source-free capacitor field: 100 V/m, uniform between the plates")
    for label,problem in scenarios:
        solution=reference.solve(problem)
        totals=[(r.name,r.net_charge_c) for r in solution.conductor_results]
        print(f"{label}: electrode totals (C; None means undefined): {totals}")
    print("Charged states use finite-order images, not a live FEMM solve.")


if __name__=='__main__':
    main()
