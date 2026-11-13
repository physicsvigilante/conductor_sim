"""One open assistant-authored attempt using existing public components only.

This is deliberately separate from independent reference keys and formulas.
"""
from dataclasses import dataclass
from conductor_sim import (
    ConnectedShellBackend, ConnectionPlan, ElectricalLink, ElectrostaticProblem,
    FixedNetCharge, Grounded, PointCharge, PositionSweepSpec, SphericalShell,
)

@dataclass(frozen=True)
class Submission:
    backend: str | None
    sweep: PositionSweepSpec | None = None
    reason: str = ''

def compose(sources, shells, end, *, center=(0.,0.,0.), backend='single'):
    world=lambda v:tuple(center[i]+v[i] for i in range(3))
    model=ElectrostaticProblem(tuple(PointCharge(q,world(x)) for q,x in sources),
        tuple(SphericalShell(a,b,c,center,f's{i}') for i,(a,b,c) in enumerate(shells)))
    return Submission(backend,PositionSweepSpec(model,0,world(end),5))

def connected():
    initial=ElectrostaticProblem((PointCharge(.6e-9,(0,0,.02)),),(
        SphericalShell(.1,.13,FixedNetCharge(.3e-9),name='s0'),
        SphericalShell(.22,.25,FixedNetCharge(-.1e-9),name='s1')))
    after=ConnectedShellBackend().solve(ConnectionPlan(initial,(
        ElectricalLink(('s0','s1')),))).resolved_problem
    return Submission('nested',PositionSweepSpec(after,0,(0,0,.07),5))

SUBMISSIONS = {
    'centre-outward':compose([(1.2e-9,(0,0,0))],[(.12,.15,Grounded())],(0,0,.07)),
    'negative-inward':compose([(-.7e-9,(.08,0,0))],[(.11,.14,Grounded())],(.02,0,0)),
    'translated-isolated':compose([(.9e-9,(.02,-.01,.01))],
        [(.12,.16,FixedNetCharge(1.5e-9))],(.06,.02,.04),center=(.8,-.4,.2)),
    'signed-reordering':compose([(.8e-9,(-.03,0,-.03)),(-.45e-9,(.01,.03,.015))],
        [(.1,.13,Grounded())],(.04,0,.04)),
    'nested-ground':compose([(1e-9,(0,0,-.04))],
        [(.12,.15,FixedNetCharge(.4e-9)),(.45,.50,Grounded())],(0,0,.08),backend='nested'),
    'inner-ground-shield':compose([(.6e-9,(0,0,.02))],
        [(.1,.12,Grounded()),(.2,.22,FixedNetCharge(-.3e-9)),(.3,.32,FixedNetCharge(.8e-9))],
        (0,0,.06),backend='nested'),
    'connect-disconnect':connected(),
    'zero-net-sources':compose([(1e-9,(0,0,-.05)),(-1e-9,(.03,0,0))],
        [(.12,.15,FixedNetCharge(.2e-9))],(0,0,.05)),
    'cross-metal':Submission(None,reason='The path crosses conducting metal; a vacuum cavity sweep must stay strictly inside the inner wall.'),
    'source-collision':Submission(None,reason='The path intersects another point source: collision makes the direct interaction singular and prevents tracking separate finite-energy particles.'),
    'cavity-time-motion':Submission(None,reason='No time integrator, particle mass or velocity evolution is provided; static positions are not a trajectory or impact-time prediction.'),
    'numerical-cavity-force':Submission(None,reason='FEMM solutions do not provide numerical force/energy observations; analytical cavity images cannot be substituted for a measured numerical force.'),
}
