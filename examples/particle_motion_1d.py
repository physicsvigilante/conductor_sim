"""M16a elapsed-time electrostatic dynamics (JSON; no HTML time viewer yet).

These use illustrative test particles with user-specified inertial mass. This
is NOT a kinetic simulation of conductor currents or a finite-radius particle.
The M16a solver stops before the ideal point-charge/surface singularity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from conductor_sim import (ConductingPlane, ElectrostaticProblem, FixedPotential,
                           Grounded, GroundedPlaneBackend, MultiImageShellBackend,
                           PointCharge, SphericalShell, SolidSphere,
                           ExternalSphereBackend, ParallelPlateBackend,
                           Motion1DSpec, integrate_motion_1d)


def scenario(name: str, q: float = 1e-9):
    if name=='plane':
        return (ElectrostaticProblem((PointCharge(q,(0,0,.10)),),
                                     (ConductingPlane(name='ground'),)),
                GroundedPlaneBackend(), (0,0,1), 0.)
    if name=='cavity':
        return (ElectrostaticProblem((PointCharge(q,(0,0,.025)),),
                                     (SphericalShell(.1,.12,Grounded(),name='cavity'),)),
                MultiImageShellBackend(), (0,0,1), 0.)
    if name=='plates':
        return (ElectrostaticProblem((PointCharge(q,(0,0,.10)),),
                    (ConductingPlane(name='lower'),
                     ConductingPlane((0,0,.2),(0,0,-1),FixedPotential(20.),'upper'))),
                ParallelPlateBackend(image_order=64), (0,0,1), 0.)
    if name=='sphere-exterior':
        return (ElectrostaticProblem((PointCharge(q,(0,0,.22)),),
                    (SolidSphere(.10,Grounded(),name='sphere'),)),
                ExternalSphereBackend(), (0,0,1), 0.)
    if name=='turning':
        return (ElectrostaticProblem((PointCharge(q,(0,0,.10)),),
                                     (ConductingPlane(name='ground'),)),
                GroundedPlaneBackend(), (0,0,1), .10)
    raise ValueError('unknown M16a example')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('plane','cavity','plates','sphere-exterior','turning'),default='plane')
    parser.add_argument('--mass-kg',type=float,default=1e-6)
    parser.add_argument('--charge-c',type=float,default=1e-9)
    parser.add_argument('--duration-s',type=float,default=.15)
    parser.add_argument('--sample-interval-s',type=float,default=.01)
    parser.add_argument('--initial-velocity-m-s',type=float,default=None)
    parser.add_argument('--surface-guard-m',type=float,default=None)
    parser.add_argument('--output',type=Path,default=Path('output/m16a-motion.json'))
    args=parser.parse_args()
    problem,backend,axis,default_velocity=scenario(args.case,args.charge_c)
    spec=Motion1DSpec(problem,0,args.mass_kg,axis,
                      default_velocity if args.initial_velocity_m_s is None else args.initial_velocity_m_s,
                      args.duration_s,args.sample_interval_s,
                      surface_guard_m=args.surface_guard_m)
    result=integrate_motion_1d(spec,backend)
    payload=result.as_dict()
    payload['source']={'example':args.case,
                       'model':'Ideal point charge; conductor geometry and constraints fixed',
                       'boundary':'Guarded surface approach is not an exact point-particle collision',
                       'scope':'Analytical time dynamics; no FEMM force or HTML time animation'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    last=result.samples[-1]
    print('M16a elapsed-time dynamics:',args.case)
    print(f'Samples: {len(result.samples)}, termination: {result.termination}, events: {[e.kind for e in result.events]}')
    print(f'Final: t={last.time_s:.6g} s; r={last.position_m} m; v_axis={last.velocity_m_s:.6g} m/s')
    print(f'Max absolute energy drift: {result.max_abs_energy_drift_j:.6g} J')
    print('Written:',args.output)


if __name__=='__main__':
    main()
