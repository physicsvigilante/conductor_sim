"""Assistant's first M13c submitted compositions; NO import from reference keys.

Uses only existing public problem/constraint/sweep objects and analytical
backends. Independent expected outputs are not hardcoded here.
"""
from conductor_sim import (ConductingPlane, ElectrostaticProblem, FixedNetCharge,
                           FixedPotential, Grounded, PointCharge, SolidSphere,
                           PositionSweepSpec)
from .protocol import Submission


def charge(q, xyz):
    return PointCharge(q, xyz)


def plane(point=(0., 0., 0.), normal=(0., 0., 1.), name='plane'):
    return ConductingPlane(point, normal, Grounded(), name)


def slab(bottom=Grounded(), top=Grounded(), gap=.2):
    return (ConductingPlane((0.,0.,0.),(0.,0.,1.),bottom,'lower'),
            ConductingPlane((0.,0.,gap),(0.,0.,-1.),top,'upper'))


def authored(kind, charges, conductors, endpoint, steps, *, moving=0):
    model=ElectrostaticProblem(tuple(charges), tuple(conductors))
    return Submission('analytical-only',kind,PositionSweepSpec(model,moving,endpoint,steps))


def blocked(reason):
    return Submission('unsupported',reason=reason)


SUBMISSIONS = {
    'plane-approach': authored('plane',[charge(1e-9,(0.,0.,.18))],
        [plane(name='earth')],(0.,0.,.06),7),
    'plane-two-signed': authored('plane',
        [charge(.8e-9,(0.,0.,.06)),charge(-.35e-9,(.045,0.,.08))],
        [plane(name='ground')],(0.,0.,.13),7),
    'plane-rotated': authored('plane',[charge(-.6e-9,(.30,.02,0.))],
        [plane((.20,0.,0.),(1.,0.,0.),'side')],(.40,.02,0.),5),
    'sphere-grounded': authored('sphere',[charge(1e-9,(0.,0.,.17))],
        [SolidSphere(.11,Grounded(),name='sphere')],(0.,0.,.30),7),
    'sphere-isolated': authored('sphere',[charge(.7e-9,(0.,0.,.22))],
        [SolidSphere(.12,FixedNetCharge(1.8e-9),name='sphere')],(0.,0.,.35),7),
    'sphere-two-sources': authored('sphere',
        [charge(.8e-9,(0.,0.,.19)),charge(-.2e-9,(.24,.10,.03))],
        [SolidSphere(.10,Grounded(),name='sphere')],(0.,0.,.30),7),
    'plates-grounded': authored('plates',[charge(1e-9,(0.,0.,.05))],
        slab(),(0.,0.,.15),9),
    'plates-biased-negative': authored('plates',[charge(-.8e-9,(0.,0.,.06))],
        slab(top=FixedPotential(20.),gap=.24),(0.,0.,.18),7),
    'plates-oblique-mixed': authored('plates',
        [charge(.65e-9,(-.04,0.,.05)),charge(-.23e-9,(-.02,.06,.11))],
        slab(),(.045,0.,.14),7),
    'plates-dual-bias': authored('plates',[charge(.9e-9,(0.,0.,.06))],
        slab(bottom=FixedPotential(-4.),top=FixedPotential(8.)),(0.,0.,.14),5),
    'dipole-torque': blocked('Rigid-dipole rotation requires torque, moment of inertia and time integration; a translating point-charge sweep does not provide angular dynamics.'),
    'ring-field': blocked('The engine lacks a uniformly charged circular ring field/force primitive; replacing the ring by a conducting sphere changes the physics.'),
    'insulating-volume': blocked('The existing conductor components do not model a charged insulating volume or its charge-density field; a conducting solid sphere is physically different.'),
    'voltage-change-during-motion': blocked('The position-sweep contract holds electrode voltages fixed; voltage-source electrical work with a changing bias is not provided by mechanical ΔU.'),
    'connected-spheres': blocked('Two separated, electrically connected conducting spheres require a coupled equal-potential constraint and geometry not supported by the sphere backend.'),
    'time-trajectory': blocked('There is no time-dynamics integrator, particle mass, velocity or collision-time observable; stored equilibria are not a physical trajectory.'),
    'offaxis-femm-force': blocked('FEMM 5 axisymmetry cannot represent two off-axis point sources; even aligned FEMM solutions do not supply numerical force/energy observations.'),
    'charge-in-cavity-motion': blocked('Cavity and nested-shell potential solvers exist, but force/interaction-energy observations for moving cavity charges are not yet supported.'),
}
