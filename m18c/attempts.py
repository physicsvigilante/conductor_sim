"""One open assistant-authored submission using only existing public model APIs.

No grading-key, reference-physics, or private engine imports. New problems are
compositions of existing components, not new force laws or integrators.
"""
from dataclasses import dataclass
from conductor_sim import (ConductingPlane, ElectrostaticProblem, FixedNetCharge,
    FixedPotential, Grounded, Motion2DSpec, PointCharge, SolidSphere, SphericalShell)

@dataclass(frozen=True)
class Submission:
    backend: str | None
    motion: Motion2DSpec | None = None
    reason: str = ''

def compose(backend, charges, moving, velocity, conductors, duration, mass_mg=1.):
    # 1 mg = 1e-6 kg; point-charge values here have already been converted to C.
    p=ElectrostaticProblem(tuple(PointCharge(q,xyz) for q,xyz in charges),tuple(conductors))
    return Submission(backend,Motion2DSpec(p,moving,mass_mg*1e-6,velocity,duration,duration/20))

SUBMISSIONS = {
    'plane-three':compose('plane',[(.4e-9,(-.055,0,.14)),(-.7e-9,(.01,0,.10)),
        (-.2e-9,(.075,0,.16))],1,(.11,0,-.015),[ConductingPlane()],.08),
    'tilted-plane':compose('plane',[(-.3e-9,(.35,-.11,.42)),(.5e-9,(.42,-.14,.38))],
        1,(.09,-.006,-.008),[ConductingPlane((.4,-.2,.3),(0,.6,.8),Grounded())],.10,2.),
    'grounded-sphere':compose('sphere',[(.6e-9,(.37,-.1,.3)),(-.3e-9,(.43,-.02,.3))],
        0,(.015,.12,0),[SolidSphere(.06,Grounded(),(.2,-.1,.3))],.08),
    'isolated-sphere':compose('sphere',[(.3e-9,(.22,-.05,0)),(-.5e-9,(.16,0,0)),
        (-.2e-9,(-.19,.07,0))],1,(.02,.10,0),[SolidSphere(.055,FixedNetCharge(1.2e-9))],.06),
    'cavity-three':compose('cavity',[(.2e-9,(-.045,.02,0)),(-.15e-9,(.04,-.03,0)),
        (.4e-9,(.01,.035,0))],2,(.08,-.02,0),[SphericalShell(.12,.15,Grounded())],.08,1.5),
    'nested-cavity':compose('nested',[(-.2e-9,(-.34,.22,.1)),(.5e-9,(-.28,.21,.1))],
        1,(.04,.07,0),[SphericalShell(.12,.15,FixedNetCharge(.4e-9),(-.3,.2,.1),'inner'),
        SphericalShell(.23,.27,Grounded(),(-.3,.2,.1),'outer')],.10),
    'biased-plates':compose('plates',[(.08e-9,(-.04,0,.13)),(-.12e-9,(.01,0,.08))],
        1,(.11,0,.02),[ConductingPlane(constraint=FixedPotential(3.),name='lower'),
        ConductingPlane((0,0,.2),(0,0,-1),FixedPotential(23.),'upper')],.12),
    'centre-release':compose('cavity',[(.06e-9,(0,.04,0)),(-.05e-9,(0,0,.05)),
        (.10e-9,(0,0,0))],2,(0,0,0),[SphericalShell(.12,.16,Grounded())],.20),
    'two-moving':Submission(None,reason='Only one moving charge is supported; coupled motion cannot be replaced by externally held charges.'),
    'nonplanar':Submission(None,reason='The held source breaks the invariant plane. General 3D motion is unsupported; projecting the force would change the physics.'),
    'numerical-femm':Submission(None,reason='FEMM lacks the numerical force interface required by this motion engine; analytical substitution is not numerical FEMM dynamics.'),
    'varying-voltage':Submission(None,reason='This API assumes fixed voltage and autonomous electrostatics; time-dependent voltage requires additional work and a different model.'),
}
