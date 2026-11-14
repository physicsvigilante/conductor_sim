"""JEE-backed far-separated spherical conductor connections; SI, static only."""
from __future__ import annotations
import argparse
from hashlib import sha256
from math import pi
from pathlib import Path

from conductor_sim import (ConnectionPlan, ElectricalLink, ElectrostaticProblem,
                           FixedNetCharge, SolidSphere, SphericalShell)
from conductor_sim.distant_spheres import APPROXIMATION, DistantSphereConnectionBackend
from conductor_sim.distant_sphere_viewer import write_distant_spheres

CASES = ('jee2011','notes-sharing','equal-density','surface-ratio','signed-groups')
PINS = {
    '02_conductors_half_size.pdf':'9dff3c02a9f1848d2c3d99cb1a6e7a44dc7f5a51f7a3a2b8cc92d10ce1fbca5d',
    'POTENTIAL_CONDUCTORS1.pdf':'8d0d9b5cf821219058cfc4b7685a39b777e460944df44c3f9cffcb68157a60e5',
}


def make_case(case):
    if case not in CASES:
        raise ValueError('unknown source example')
    def sphere(name, r, q, x, shell=False):
        if shell:
            return SphericalShell(r*.99,r,FixedNetCharge(q),(x,0.,0.),name)
        return SolidSphere(r,FixedNetCharge(q),(x,0.,0.),name)
    if case=='jee2011':
        cs=(sphere('A',.2,1e-9,0.,True),sphere('B',.1,1e-9,20.))
        source='Official IIT-JEE 2011 Paper 1 Q31, physical PDF page 13; workbook p16 Multiple Q1'
        source_file='02_conductors_half_size.pdf'
        given='Larger empty metal shell A; smaller solid sphere B; initially equal +Q; far apart.'
        choices='Illustrative RA=0.20 m, RB=0.10 m, Q=1 nC, centres 20 m apart; shell wall 1% of radius.'
        question='After connection: QA>QB, sigmaA/sigmaB=RB/RA, EA<EB; empty cavity and metal E=0.'
    elif case=='notes-sharing':
        cs=(sphere('small',.1,.5e-3,0.,True),sphere('large',.2,.7e-3,20.,True))
        source='POTENTIAL_CONDUCTORS1.pdf p68 Illustration 4'
        source_file='POTENTIAL_CONDUCTORS1.pdf'
        given='Radii 0.10/0.20 m; initial charges +0.5/+0.7 mC; far-separated metal shells.'
        choices='Only centre spacing 20 m and 1% wall thickness are illustrative. Ideal electrostatics: no air breakdown/material model.'
        question='Final charges should be +0.4 mC and +0.8 mC. Energy release is an additional teaching observation.'
    elif case=='equal-density':
        sigma=1e-8
        cs=(sphere('small',.1,4*pi*.1**2*sigma,0.),sphere('large',.2,4*pi*.2**2*sigma,20.))
        source='POTENTIAL_CONDUCTORS1.pdf p68 Illustration 5'
        source_file='POTENTIAL_CONDUCTORS1.pdf'
        given='Far-separated radii R and 2R, same initial positive surface density.'
        choices='Illustrative R=0.10 m, sigma=1e-8 C/m², centre separation 20 m.'
        question='Larger final surface density = 5/6 of its initial density; smaller = 5/3.'
    elif case=='surface-ratio':
        cs=(sphere('R1',.1,2e-9,0.),sphere('R2',.2,4e-9,20.))
        source='02_conductors_half_size.pdf p2 Single Correct Q15'
        source_file='02_conductors_half_size.pdf'
        given='Distant wired spherical conductors; asks E1/E2 at the outer surfaces.'
        choices='Illustrative R1=0.10 m, R2=0.20 m; initial +2/+4 nC; separation 20 m.'
        question='E1/E2=R2/R1=2. These initial potentials already match: no redistribution required.'
    else:
        cs=(sphere('plus',.1,3e-9,0.),sphere('minus',.2,-3e-9,20.),sphere('unlinked',.15,2e-9,40.))
        source='Teacher hypothetical, not an exam transcription'
        source_file=None
        given='Only plus and minus are connected. Third conductor remains isolated.'
        choices='All geometry and charges are teacher-authored; leading approximation only.'
        question='Linked total charge cancels; third charge is unchanged. Neglected mutual induction can dominate relative errors near zero.'
    links=(ElectricalLink((cs[0].name,cs[1].name)),)
    plan=ConnectionPlan(ElectrostaticProblem(conductors=cs),links)
    result=DistantSphereConnectionBackend().solve(plan,approximation=APPROXIMATION)
    provenance=dict(source=source,source_given=given,additional_choices=choices,
                    teaching_question=question,origin='teacher-hypothetical' if not source_file else 'source-anchored with disclosed numerical choices')
    if source_file:
        path=Path(__file__).resolve().parents[1]/source_file
        if sha256(path.read_bytes()).hexdigest()!=PINS[source_file]:
            raise ValueError('source PDF changed; recheck source geometry before exporting')
        provenance.update(source_sha256=PINS[source_file])
    if case=='jee2011':
        provenance['official_reference']='https://jeeadv.ac.in/past_qps/2011_1.pdf (p13, Q31; ABCD)'
    return result,provenance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=CASES,default='jee2011')
    parser.add_argument('--output',type=Path,default=Path('output/m21d-jee2011.html'))
    args=parser.parse_args()
    result,source=make_case(args.case)
    write_distant_spheres(result,args.output,title='M21d · '+args.case,provenance=source)
    print(args.output)
    print('Two static leading-order equilibria; energy released (J):',result.released_field_energy_j)

if __name__=='__main__':
    main()
