"""Reusable offline analytical force/energy explorer for JEE conductor problems.

Every frame is a static analytical equilibrium; the position slider does not
run an electrostatic solver or represent elapsed time / particle motion.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from conductor_sim import (ConductingPlane, ElectrostaticProblem, SphericalShell,
                           MultiImageShellBackend, NestedShellBackend,
                           ExternalSphereBackend, FixedNetCharge,
                           FixedPotential, Grounded, GroundedPlaneBackend,
                           ParallelPlateBackend, PointCharge, SolidSphere)
from conductor_sim.position_sweep import PositionSweepSpec, sample_position_sweep
from conductor_sim.visualization import write_visualization


def scenario(name: str, image_order: int):
    q = 1e-9
    if name == 'plane':
        problem = ElectrostaticProblem((PointCharge(q,(0.,0.,.06)),),
                                       (ConductingPlane(name='grounded-plane'),))
        return PositionSweepSpec(problem,0,(0.,0.,.18)),GroundedPlaneBackend()
    if name in ('sphere-grounded','sphere-isolated'):
        constraint = (Grounded() if name=='sphere-grounded'
                      else FixedNetCharge(2e-9))
        problem = ElectrostaticProblem((PointCharge(q,(0.,0.,.17)),),
                                       (SolidSphere(.10,constraint,name='sphere'),))
        return PositionSweepSpec(problem,0,(0.,0.,.33)),ExternalSphereBackend()
    if name in ('plates-grounded','plates-biased'):
        upper = Grounded() if name=='plates-grounded' else FixedPotential(20.)
        problem = ElectrostaticProblem((PointCharge(q,(0.,0.,.045)),),(
            ConductingPlane(name='lower'),
            ConductingPlane((0.,0.,.20),(0.,0.,-1.),upper,'upper')))
        return PositionSweepSpec(problem,0,(0.,0.,.155)),ParallelPlateBackend(image_order=image_order)
    if name in ('cavity-grounded', 'cavity-isolated', 'cavity-centered',
                'cavity-nested', 'cavity-multi'):
        constraint = (FixedNetCharge(-.7e-9) if name=='cavity-isolated'
                      else FixedNetCharge(.4e-9) if name=='cavity-nested'
                      else Grounded())
        shell = SphericalShell(.10, .12, constraint, name='cavity-shell')
        conductors = (shell,)
        if name=='cavity-nested':
            conductors += (SphericalShell(.20,.22,Grounded(),name='outer-shell'),)
        moving = ((0.,0.,0.) if name=='cavity-centered' else
                  (.018,0.,-.030) if name=='cavity-multi' else (0.,0.,.025))
        other_sources = ((PointCharge(-.4e-9,(-.037,0.,.012)),)
                         if name=='cavity-multi' else ())
        problem = ElectrostaticProblem((PointCharge(q,moving),)+other_sources,conductors)
        endpoint = ((.044,0.,.066) if name=='cavity-multi' else (0.,0.,.080))
        backend = NestedShellBackend() if name=='cavity-nested' else MultiImageShellBackend()
        return PositionSweepSpec(problem,0,endpoint),backend
    raise ValueError('unknown named force-energy scenario')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('plane','sphere-grounded','sphere-isolated',
                                           'plates-grounded','plates-biased',
                                           'cavity-grounded','cavity-isolated','cavity-centered',
                                           'cavity-nested','cavity-multi'),
                        default='plates-grounded')
    parser.add_argument('--output',type=Path,default=Path('output/force_energy_explorer.html'))
    parser.add_argument('--steps',type=int,default=11)
    parser.add_argument('--resolution',type=int,default=25)
    parser.add_argument('--image-order',type=int,default=48)
    parser.add_argument('--line-steps',type=int,default=110)
    args=parser.parse_args()
    original,backend=scenario(args.case,args.image_order)
    # The selected source is an authored component index; not a sorted source
    # index, so it remains identifiable as it moves.
    spec=PositionSweepSpec(original.problem,original.charge_index,original.end_m,
                           steps=args.steps)
    bundle=sample_position_sweep(spec,backend,resolution=args.resolution,
                                 line_steps=args.line_steps)
    output=write_visualization(bundle,args.output,title='Force and energy explorer · '+args.case)
    print('HTML:',output)
    print('States:',len(bundle['frames']),'backend: analytical')
    for r in bundle['position_sweep']['samples']:
        print(f"s={r['coordinate_m']:.4f} m  F_parallel={r['force_parallel_n']:.5e} N "
              f"U={r['energy_j']:.5e} J W_ext={r['external_quasistatic_work_j']:.5e} J")
    print('Static equilibria, not a moving-particle simulation.')


if __name__ == '__main__':
    main()
