"""M14a connected conducting shells: one compiler and the existing viewer.

The PDF page 14 Numerical Q7 version is source-anchored in physical setup,
with illustrative R = .10 m and q = 1 nC. The answer ΔV_outer=0 does not
require assigning numerical values to R or q. Shell thickness in this model
is finite but field and V of the pictured ideal thin shells are unaffected.
The slider switches static equilibria; it is not a current-time animation.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from math import pi
import json

from conductor_sim import (ConnectionPlan, ConnectedShellBackend, ElectricalLink,
                           ElectrostaticProblem, FixedNetCharge, Grounded,
                           PointCharge, SphericalShell)
from conductor_sim.visualization import write_visualization


def scenario(name: str):
    if name=='pdf-p14-q7':
        # The document's Q7: two thin concentric shells of radii R, 2R,
        # initially carrying q and 2q. Connecting their surfaces leaves
        # the exterior total charge and the outer voltage unchanged.
        q=1e-9
        p=ElectrostaticProblem(conductors=(
            SphericalShell(.095,.10,FixedNetCharge(q),name='inner'),
            SphericalShell(.195,.20,FixedNetCharge(2*q),name='outer'),
        ))
        return ConnectionPlan(p,(ElectricalLink(('inner','outer')),))
    if name=='ground-link':
        p=ElectrostaticProblem(conductors=(
            SphericalShell(.08,.10,FixedNetCharge(1e-9),name='inner'),
            SphericalShell(.20,.22,Grounded(),name='outer'),
        ))
        return ConnectionPlan(p,(ElectricalLink(('inner','outer'),'grounded'),))
    if name=='cavity-charge':
        p=ElectrostaticProblem(
            charges=(PointCharge(1e-9,(0,0,.025)),),
            conductors=(
                SphericalShell(.08,.10,FixedNetCharge(.4e-9),name='inner'),
                SphericalShell(.20,.22,FixedNetCharge(-.1e-9),name='outer'),
            ),
        )
        return ConnectionPlan(p,(ElectricalLink(('inner','outer')),))
    if name=='three-shell':
        p=ElectrostaticProblem(conductors=(
            SphericalShell(.08,.10,FixedNetCharge(1.2e-9),name='inner'),
            SphericalShell(.20,.22,FixedNetCharge(.4e-9),name='middle'),
            SphericalShell(.33,.35,FixedNetCharge(-.2e-9),name='outer'),
        ))
        return ConnectionPlan(p,(ElectricalLink(('inner','outer')),))
    raise ValueError('unknown M14a scenario')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case',choices=('pdf-p14-q7','ground-link','cavity-charge','three-shell'),default='pdf-p14-q7')
    ap.add_argument('--resolution',type=int,default=31)
    ap.add_argument('--line-steps',type=int,default=140)
    ap.add_argument('--output',type=Path,default=Path('output/connected_shells.html'))
    args=ap.parse_args()
    result=ConnectedShellBackend().solve(scenario(args.case))
    bundle=result.visualization(resolution=args.resolution,line_steps=args.line_steps)
    bundle['connection']['example_case']=args.case
    if args.case=='pdf-p14-q7':
        bundle['connection']['source_anchor']={
            'pdf':'02_conductors_half_size.pdf','page':14,'printed_question':'Numerical Value Type Q7',
            'illustrative_values':'R=0.10 m, q=1 nC; PDF answer ΔV_outer=0 V is independent of these choices.',
        }
    path=write_visualization(bundle,args.output,title='Connected conducting shells · '+args.case)
    path.with_suffix('.json').write_text(json.dumps(bundle,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('Analytical two-state viewer:',path)
    print('Connected groups:',[(link.mode,link.members) for link in result.plan.links])
    for record in result.transfers:
        print(record.name,'before C=',record.before_c,'after C=',record.after_c,'received C=',record.received_c)
    print('Gap/exterior energy released (J):',result.released_field_energy_j)
    if args.case=='pdf-p14-q7':
        delta=result.after.conductor_by_name('outer').potential_v-result.before.conductor_by_name('outer').potential_v
        print('PDF p14 Q7 outer shell voltage change (V):',delta)
    print('These are two static configurations, not a wire-current or discharge-time simulation.')


if __name__=='__main__':main()
