"""M14b radial conductor energy: solid sphere and concentric shells.

Static before/after states, no time-resolved current. Illustrative thickness
of the PDF's ideal thin shell is deliberately small and documented.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from math import pi

from conductor_sim import (ConnectedShellBackend, ConnectionPlan, ElectricalLink,
                           ElectrostaticProblem, FixedNetCharge, Grounded,
                           SolidSphere, SphericalShell)
from conductor_sim.visualization import write_visualization


def scenario(name):
    if name == 'pdf-p03-q27':
        # Given in PDF: solid R=.10m with +20uC; thin outer shell R=.20m,
        # initially uncharged. Inner radius .199999m approximates thin shell.
        problem = ElectrostaticProblem(conductors=(
            SolidSphere(.10, FixedNetCharge(20e-6), name='solid'),
            SphericalShell(.199999, .20, FixedNetCharge(0.), name='shell')))
        return ConnectionPlan(problem, (ElectricalLink(('solid','shell')),))
    if name == 'pdf-p10-q13-q14':
        # Two thin, isolated shells with equal and opposite charge from Emax.
        # E just outside the inner 10cm shell is 1.8e6 N/C.
        radius=.1
        epsilon=8.8541878128e-12
        charge=4*pi*epsilon*radius**2*1.8e6
        problem = ElectrostaticProblem(conductors=(
            SphericalShell(.099999,.10,FixedNetCharge(charge),name='inner'),
            SphericalShell(.199999,.20,FixedNetCharge(-charge),name='outer')))
        return ConnectionPlan(problem,(ElectricalLink(('inner','outer')),))
    if name == 'grounded-outer':
        problem=ElectrostaticProblem(conductors=(
            SolidSphere(.10, FixedNetCharge(2e-9), name='solid'),
            SphericalShell(.19,.20,Grounded(),name='outer')))
        return ConnectionPlan(problem,(ElectricalLink(('solid','outer'),'grounded'),))
    if name == 'nonadjacent':
        problem=ElectrostaticProblem(conductors=(
            SolidSphere(.10,FixedNetCharge(2e-9),name='solid'),
            SphericalShell(.19,.20,FixedNetCharge(-.3e-9),name='middle'),
            SphericalShell(.29,.30,FixedNetCharge(.5e-9),name='outer')))
        return ConnectionPlan(problem,(ElectricalLink(('solid','outer')),))
    raise ValueError('unknown M14b scenario')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case',choices=('pdf-p03-q27','pdf-p10-q13-q14',
                                       'grounded-outer','nonadjacent'),
                    default='pdf-p03-q27')
    ap.add_argument('--resolution',type=int,default=31)
    ap.add_argument('--line-steps',type=int,default=140)
    ap.add_argument('--output',type=Path,default=Path('output/solid_sphere_connected.html'))
    args=ap.parse_args()
    result=ConnectedShellBackend().solve(scenario(args.case))
    bundle=result.visualization(resolution=args.resolution,line_steps=args.line_steps)
    bundle['connection']['example_case']=args.case
    if args.case=='pdf-p03-q27':
        bundle['connection']['source_anchor']={
            'pdf':'02_conductors_half_size.pdf', 'page':3,
            'printed_question':'Single Correct Q27',
            'given':'R_solid=0.10m, R_outer=0.20m, Q_solid=20uC, Q_outer=0',
            'approximation':'Model shell has inner radius 0.199999m rather than ideal zero thickness.',
        }
    elif args.case=='pdf-p10-q13-q14':
        bundle['connection']['source_anchor']={
            'pdf':'02_conductors_half_size.pdf','page':10,
            'printed_question':'Linked Comprehension Q13-Q14',
            'given':'concentric shells 10/20cm, Emax=1.8e6N/C, equal-and-opposite initial charges',
            'approximation':'Model shells have small finite thickness.'}
    output=write_visualization(bundle,args.output,title='Concentric connected conductors · '+args.case)
    args.output.with_suffix('.json').write_text(json.dumps(bundle,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('Analytical static two-state explorer:',output)
    print('Before:',result.before.conductor_results)
    print('After:',result.after.conductor_results)
    print('Initial field energy (J):',result.radial_field_energy_before_j)
    print('Final field energy (J):',result.radial_field_energy_after_j)
    print('Released field energy (J):',result.released_field_energy_j)
    print('No moving charges or time-resolved currents are claimed.')


if __name__=='__main__':
    main()
