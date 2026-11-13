"""M13a: one observation interface for planes, spheres and parallel plates.

No FEMM required. Exports SI-valued observations, not a moving-particle model.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from conductor_sim import (
    ConductingPlane, ElectrostaticProblem, ExternalSphereBackend, FixedNetCharge,
    FixedPotential, Grounded, GroundedPlaneBackend, ParallelPlateBackend,
    PointCharge, SolidSphere, force_energy,
)


def capacitor(height: float, *, upper_voltage: float = 0.) -> ElectrostaticProblem:
    return ElectrostaticProblem((PointCharge(1e-9,(0.,0.,height)),), (
        ConductingPlane(name='lower'),
        ConductingPlane((0.,0.,.20),(0.,0.,-1.),FixedPotential(upper_voltage),'upper')))


def observations(problem, backend):
    result = force_energy(backend.solve(problem))
    record = asdict(result)
    record['energy']['total_j'] = result.energy.total_j
    for stored, charge in zip(record['charges'],result.charges):
        stored['total_force_n'] = charge.total_force_n
        stored['conductor_force_n'] = charge.conductor_force_n
    return record


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('output/force_energy.json'))
    parser.add_argument('--image-order',type=int,default=128)
    args=parser.parse_args()
    plates=ParallelPlateBackend(image_order=args.image_order)
    cases=[
        ('Charge above a grounded plane',
         ElectrostaticProblem((PointCharge(1e-9,(0.,0.,.10)),),(ConductingPlane(),)),
         GroundedPlaneBackend()),
        ('Charge outside a grounded sphere',
         ElectrostaticProblem((PointCharge(1e-9,(0.,0.,.25)),),(SolidSphere(.10,Grounded()),)),
         ExternalSphereBackend()),
        ('Charge outside an isolated +1 nC sphere',
         ElectrostaticProblem((PointCharge(1e-9,(0.,0.,.25)),),(SolidSphere(.10,FixedNetCharge(1e-9)),)),
         ExternalSphereBackend()),
        *[(f'Grounded plates: charge height {h:.3f} m',capacitor(h),plates)
          for h in (.05,.10,.15)],
        ('20 V plate bias plus induction',capacitor(.065,upper_voltage=20.),plates),
    ]
    records=[]
    for label, problem, backend in cases:
        record=observations(problem,backend)
        records.append(dict(label=label,**record))
        print(label)
        print('  F (N):',record['charges'][0]['total_force_n'])
        print('  U (J):',record['energy']['total_j'])
    # Same charge, conductors, voltages and gauge; only its position changes.
    start=force_energy(plates.solve(capacitor(.06,upper_voltage=20.)))
    end=force_energy(plates.solve(capacitor(.08,upper_voltage=20.)))
    delta=end.energy.total_j-start.energy.total_j
    work={'from_height_m':.06,'to_height_m':.08,'upper_potential_v':20.,
          'external_quasistatic_work_j':delta,'work_by_electric_force_j':-delta,
          'conditions':'Fixed charges, fixed conductor geometry and voltages; no kinetic-energy change.'}
    report={'schema':'conductor_sim.force_energy_example.v1','cases':records,'work_example':work,
            'scope':'Analytical mechanical interaction energies. Point self energies and empty-conductor '
                    'constants omitted. No total infinite-capacitor field energy or FEMM force claim.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('External quasistatic work (J):',delta)
    print('Report:',args.output)


if __name__=='__main__':
    main()
