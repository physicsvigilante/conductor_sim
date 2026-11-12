"""Independent grading keys; not imported by the authored submissions.

Expected fields follow Gauss's law / superposition, never numerical values
copied from a solver. Model fingerprints detect incomplete or fabricated input.
"""
from dataclasses import dataclass
from math import pi

from conductor_sim.model import EPSILON_0
from .protocol import Probe

K = 1/(4*pi*EPSILON_0)


def shell(name,a,b,q=None,center=(0.,0.,0.)):
    # q=None means grounded. All values SI.
    return (name, a, b, 'grounded' if q is None else 'fixed', q, tuple(center))


def charge(q,at=(0.,0.,0.)):
    return (q,tuple(at))


def model(shells,charges=()):
    return (tuple(shells),tuple(charges))


def expected(state,kind,value,point=None,conductor=None):
    return (Probe(state,'independent expected observation',kind,point,conductor),float(value))


@dataclass(frozen=True,slots=True)
class Key:
    capability: str
    models: tuple=()
    observations: tuple=()
    required_reason_words: tuple[str,...]=()


KEYS = {
    'two-shell-negative-net':Key('both',(
        model([shell('inner',.08,.10,1e-9),shell('outer',.19,.21,-2e-9)]),),(
        expected(0,'shell_net_charge',-2e-9,conductor='outer'),
        expected(0,'field_x',K*1e-9/.15**2,(.15,0,0)),
        expected(0,'potential',K*(-1e-9)/.30,(.30,0,0)),
    )),
    'grounded-outer-shield':Key('both',(
        model([shell('inner',.06,.08,.6e-9),shell('outer',.17,.20)]),),(
        expected(0,'shell_net_charge',-.6e-9,conductor='outer'),
        expected(0,'shell_voltage',K*.6e-9*(1/.08-1/.17),conductor='inner'),
        expected(0,'field_x',0,(.30,0,0)),
    )),
    'offcentre-negative-grounded':Key('both',(
        model([shell('body',.10,.13)], [charge(-1.4e-9,(0,0,.028))]),),(
        expected(0,'shell_voltage',0,conductor='body'),
        expected(0,'shell_net_charge',1.4e-9,conductor='body'),
        expected(0,'field_x',0,(.25,0,0)),
    )),
    'central-source-and-charged-shell':Key('both',(
        model([shell('shell',.09,.115,.3e-9)], [charge(1.2e-9)]),),(
        expected(0,'inner_total',-1.2e-9,conductor='shell'),
        expected(0,'outer_total',1.5e-9,conductor='shell'),
        expected(0,'potential',K*1.5e-9/.23,(.23,0,0)),
    )),
    'noncollinear-math-only':Key('analytical-only',(
        model([shell('shell',.12,.15,-.5e-9)],
              [charge(1e-9,(.020,.013,0)),charge(-.4e-9,(-.016,0,.024))]),),(
        expected(0,'outer_total',.1e-9,conductor='shell'),
        expected(0,'potential',K*.1e-9/.30,(.30,0,0)),
    ),('non-collinear','axisymmetry')),
    'reordered-three-shells':Key('both',(
        model([shell('outer',.19,.21,.2e-9),shell('middle',.115,.13,-.9e-9),
               shell('inner',.05,.06,.4e-9)]),),(
        expected(0,'outer_total',-.3e-9,conductor='outer'),
        expected(0,'field_x',K*(-.5e-9)/.16**2,(.16,0,0)),
        expected(0,'shell_net_charge',.2e-9,conductor='outer'),
    )),
    'cancelled-central-charges':Key('both',(
        model([shell('shell',.10,.12)], [charge(1.2e-9),charge(-1.2e-9)]),),(
        expected(0,'shell_voltage',0,conductor='shell'),
        expected(0,'shell_net_charge',0,conductor='shell'),
        expected(0,'field_x',0,(.055,0,0)),
    )),
    'translated-rotated-cavity-charge':Key('both',(
        model([shell('body',.10,.12,1e-9,(.4,-.2,.1))],
              [charge(-.4e-9,(.43,-.16,.1))]),),(
        expected(0,'outer_total',.6e-9,conductor='body'),
        expected(0,'potential',K*.6e-9/.25,(.65,-.2,.1)),
    )),
    'two-positive-separated-sources':Key('both',(
        model([shell('shell',.10,.12,0)], [charge(.7e-9,(0,0,.027)),charge(.3e-9,(0,0,-.021))]),),(
        expected(0,'inner_total',-1e-9,conductor='shell'),
        expected(0,'outer_total',1e-9,conductor='shell'),
        expected(0,'potential',K*1e-9/.24,(.24,0,0)),
    )),
    'grounded-two-state-sweep':Key('both',(
        model([shell('inner',.09,.11,1e-9),shell('outer',.18,.20)]),
        model([shell('inner',.09,.11,3e-9),shell('outer',.18,.20)]),
    ),(
        expected(0,'shell_voltage',K*1e-9*(1/.11-1/.18),conductor='inner'),
        expected(1,'shell_voltage',K*3e-9*(1/.11-1/.18),conductor='inner'),
        expected(0,'field_x',0,(.30,0,0)),
        expected(1,'field_x',0,(.30,0,0)),
    )),
    'exterior-charge-grounded-sphere':Key('unsupported',required_reason_words=('external','outside')),
    'charged-ring-on-axis':Key('unsupported',required_reason_words=('ring',)),
    'charge-between-conductors':Key('unsupported',required_reason_words=('gap',)),
    'connected-separated-spheres':Key('unsupported',required_reason_words=('separated','connected')),
    'moving-charge-trajectory':Key('unsupported',required_reason_words=('moving','dynamics')),
}
