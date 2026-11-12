"""One assistant-authored, first completed M6b attempt.

Allowed APIs: existing typed physical components, no solver implementations,
no pasted field/voltage formulas and no viewer-code modifications. Reference
numerical answers and grader keys are NOT imported into this module.
"""
from conductor_sim import (
    ElectrostaticProblem, FixedNetCharge, Grounded, PointCharge, SphericalShell,
)
from .protocol import Probe, Submission


def state(shells, charges=(), label='Static equilibrium'):
    return ((label, ElectrostaticProblem(tuple(charges), tuple(shells))),)


def p(kind, label, *, state=0, point=None, conductor=None):
    return Probe(state, label, kind, point, conductor)


def author_two_shell_negative_net():
    return Submission('both',state([
        SphericalShell(.08,.10,FixedNetCharge(1e-9),name='inner'),
        SphericalShell(.19,.21,FixedNetCharge(-2e-9),name='outer')]),(
        p('shell_net_charge','outer conductor net charge',conductor='outer'),
        p('field_x','field in dielectric gap',point=(.15,0,0)),
        p('potential','potential outside shells',point=(.30,0,0)),
    ))


def author_grounded_outer_shield():
    return Submission('both',state([
        SphericalShell(.06,.08,FixedNetCharge(.6e-9),name='inner'),
        SphericalShell(.17,.20,Grounded(),name='outer')]),(
        p('shell_net_charge','grounded outer charge',conductor='outer'),
        p('shell_voltage','inner voltage',conductor='inner'),
        p('field_x','shielded exterior',point=(.30,0,0)),
    ))


def author_offcentre_negative_grounded():
    return Submission('both',state([
        SphericalShell(.10,.13,Grounded(),name='body')],
        [PointCharge(-1.4e-9,(0,0,.028))]),(
        p('shell_voltage','grounded conductor potential',conductor='body'),
        p('shell_net_charge','total induced charge',conductor='body'),
        p('field_x','field outside grounded shell',point=(.25,0,0)),
    ))


def author_central_source_and_charged_shell():
    return Submission('both',state([
        SphericalShell(.09,.115,FixedNetCharge(.3e-9),name='shell')],
        [PointCharge(1.2e-9)]),(
        p('inner_total','inner induced charge',conductor='shell'),
        p('outer_total','outer induced charge',conductor='shell'),
        p('potential','exterior potential',point=(.23,0,0)),
    ))


def author_noncollinear_math_only():
    return Submission('analytical-only',state([
        SphericalShell(.12,.15,FixedNetCharge(-.5e-9),name='shell')],
        [PointCharge(1e-9,(.020,.013,0)),PointCharge(-.4e-9,(-.016,0,.024))]),(
        p('outer_total','outer induced charge',conductor='shell'),
        p('potential','exterior voltage',point=(.30,0,0)),
    ),reason='The two distinct charge sites are not collinear through the shell centre; FEMM axisymmetry is unavailable, but Kelvin-image superposition is exact.')


def author_reordered_three_shells():
    # Deliberately reverse authored order, relying on the solver's radial sort.
    return Submission('both',state([
        SphericalShell(.19,.21,FixedNetCharge(.2e-9),name='outer'),
        SphericalShell(.115,.13,FixedNetCharge(-.9e-9),name='middle'),
        SphericalShell(.05,.06,FixedNetCharge(.4e-9),name='inner')]),(
        p('outer_total','exterior face charge',conductor='outer'),
        p('field_x','field outside middle shell',point=(.16,0,0)),
        p('shell_net_charge','outer net charge',conductor='outer'),
    ))


def author_cancelled_central_charges():
    return Submission('both',state([
        SphericalShell(.10,.12,Grounded(),name='shell')],
        [PointCharge(1.2e-9),PointCharge(-1.2e-9)]),(
        p('shell_voltage','zero conductor voltage',conductor='shell'),
        p('shell_net_charge','net induced conductor charge',conductor='shell'),
        p('field_x','no central field',point=(.055,0,0)),
    ))


def author_translated_rotated_cavity_charge():
    center=(.4,-.2,.1)
    return Submission('both',state([
        SphericalShell(.10,.12,FixedNetCharge(1e-9),center_m=center,name='body')],
        [PointCharge(-.4e-9,(.43,-.16,.1))]),(
        p('outer_total','outside face charge',conductor='body'),
        p('potential','potential at offset probe',point=(.65,-.2,.1)),
    ))


def author_two_positive_separated_sources():
    return Submission('both',state([
        SphericalShell(.10,.12,FixedNetCharge(0),name='shell')],
        [PointCharge(.7e-9,(0,0,.027)),PointCharge(.3e-9,(0,0,-.021))]),(
        p('inner_total','inner induced charge',conductor='shell'),
        p('outer_total','outer induced charge',conductor='shell'),
        p('potential','outer potential',point=(.24,0,0)),
    ))


def author_grounded_two_state_sweep():
    states=[]
    for q in (1e-9,3e-9):
        shells=(SphericalShell(.09,.11,FixedNetCharge(q),name='inner'),
                SphericalShell(.18,.20,Grounded(),name='outer'))
        states.append((f'Inner {q*1e9:g} nC',ElectrostaticProblem((),shells)))
    return Submission('both',tuple(states),(
        p('shell_voltage','initial inner voltage',state=0,conductor='inner'),
        p('shell_voltage','final inner voltage',state=1,conductor='inner'),
        p('field_x','initial grounded exterior',state=0,point=(.30,0,0)),
        p('field_x','final grounded exterior',state=1,point=(.30,0,0)),
    ))


SUBMISSIONS = {
    'two-shell-negative-net':author_two_shell_negative_net(),
    'grounded-outer-shield':author_grounded_outer_shield(),
    'offcentre-negative-grounded':author_offcentre_negative_grounded(),
    'central-source-and-charged-shell':author_central_source_and_charged_shell(),
    'noncollinear-math-only':author_noncollinear_math_only(),
    'reordered-three-shells':author_reordered_three_shells(),
    'cancelled-central-charges':author_cancelled_central_charges(),
    'translated-rotated-cavity-charge':author_translated_rotated_cavity_charge(),
    'two-positive-separated-sources':author_two_positive_separated_sources(),
    'grounded-two-state-sweep':author_grounded_two_state_sweep(),
    'exterior-charge-grounded-sphere':Submission('unsupported',reason='External charge lies outside the spherical cavity; current engine cannot compile induced fields from outside a grounded sphere.'),
    'charged-ring-on-axis':Submission('unsupported',reason='A thin uniformly charged ring requires a ring source component; replacing it with a spherical conductor changes the physics.'),
    'charge-between-conductors':Submission('unsupported',reason='Charge is located in the inter-shell dielectric gap, beyond current solver support requiring charges in the innermost cavity.'),
    'connected-separated-spheres':Submission('unsupported',reason='Two separated electrically connected spheres are not concentric nested shells and require geometry plus shared-potential conductor constraints.'),
    'moving-charge-trajectory':Submission('unsupported',reason='A moving proton requires charged-particle dynamics and force/time integration, not static electrostatic equilibrium states.'),
}
