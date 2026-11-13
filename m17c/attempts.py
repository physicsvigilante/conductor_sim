"""One open assistant-authored submission using production public APIs only.

No grading key, reference force, independent integrator or special viewer is
imported here. Each supported problem is composed from existing components.
"""
from dataclasses import dataclass
from conductor_sim import (ConductingPlane, ElectrostaticProblem, FixedNetCharge,
    FixedPotential, Grounded, Motion2DSpec, PointCharge, SolidSphere, SphericalShell)

@dataclass(frozen=True)
class Submission:
    backend: str | None
    motion: Motion2DSpec | None = None
    reason: str = ''


def compose(backend, charge, position, velocity, conductors, duration, interval, *, mass=1e-6):
    # One milligram is 1e-6 kg. Prompts deliberately use a non-SI mass unit.
    problem=ElectrostaticProblem((PointCharge(charge,position),),tuple(conductors))
    return Submission(backend,Motion2DSpec(problem,0,mass,velocity,duration,interval))

SUBMISSIONS = {
    'tilted-plane': compose('plane',-.8e-9,(.43,-.128,.396),(.13,.009,.012),
        [ConductingPlane((.4,-.2,.3),(0,.6,.8),Grounded())],.16,.008),
    'electron-plates': compose('plates',-1.602176634e-19,(0,0,.1),(8e5,2e5,-1e5),
        [ConductingPlane(name='lower'),
         ConductingPlane((0,0,.2),(0,0,-1),FixedPotential(20.),'upper')],5e-8,2.5e-9,
        mass=9.1093837139e-31),
    'grounded-sphere': compose('sphere',-1e-9,(.38,-.26,.1),(-.02,.18,.06),
        [SolidSphere(.07,Grounded(),(.2,-.3,.1))],.12,.006),
    'isolated-sphere': compose('sphere',.7e-9,(.15,0,0),(-.04,.11,0),
        [SolidSphere(.06,FixedNetCharge(1.6e-9))],.08,.004),
    'cavity-oblique': compose('cavity',-1.1e-9,(.025,-.01,.005),(.02,.11,-.04),
        [SphericalShell(.12,.15,Grounded())],.1,.005),
    'nested-cavity': compose('nested',.6e-9,(-.48,.26,.07),(.06,-.04,.03),
        [SphericalShell(.12,.15,FixedNetCharge(-.2e-9),(-.5,.25,.1),'inner'),
         SphericalShell(.24,.28,FixedNetCharge(1e-9),(-.5,.25,.1),'outer')],.08,.004),
    'rotated-radial': compose('sphere',1e-9,(.1,.2,.3),(-.01,-.02,-.03),
        [SolidSphere(.01,Grounded())],.06,.003),
    'centre-crossing': compose('cavity',.2e-9,(.02,0,0),(-.25,0,0),
        [SphericalShell(.12,.16,FixedNetCharge(.3e-9))],.12,.006),
    'two-moving': Submission(None,reason='Only one moving charge is supported; coupled multi-particle motion is not implemented.'),
    'numerical-femm': Submission(None,reason='FEMM does not provide the numerical force interface; analytical substitution is not numerical FEMM dynamics.'),
    'finite-plate-edge': Submission(None,reason='Infinite plates have no finite edges or fringing field; replacing finite geometry would change the problem.'),
    'varying-voltage': Submission(None,reason='The current motion model assumes fixed voltage and autonomous electrostatic force, not time-dependent voltage during flight.'),
}
