"""Traceable, deterministic JEE authorability benchmark from POTENTIAL_CONDUCTORS1.pdf.

This module is NOT a natural-language parser: every runnable case is authored
with existing, solver-independent physics components. The benchmark checks
whether a new JEE arrangement requires a *new authored problem* rather than a
new solver implementation. Unsupported source exercises never get fabricated
solutions or silent geometry substitutions.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from math import cos, isclose, pi, sin
from pathlib import Path
from typing import Callable, Literal

from .model import EPSILON_0, ElectrostaticProblem, FixedNetCharge, Grounded, PointCharge, SphericalShell
from .nested_shells import NestedShellBackend
from .femm5 import Femm5Backend
from .visualization import SliceSpec, sample_sweep, write_visualization
from .solver import UnsupportedProblem

PDF_NAME = "POTENTIAL_CONDUCTORS1.pdf"
PDF_SHA256 = "8d0d9b5cf821219058cfc4b7685a39b777e460944df44c3f9cffcb68157a60e5"
PDF_PAGES = 71
K = 1 / (4*pi*EPSILON_0)
Q = 1e-9
A = .1
B = .2
C = .3


@dataclass(frozen=True)
class Check:
    """Expected observation, independently specified in SI units."""
    state: int
    label: str
    kind: Literal["potential", "field_x", "shell_voltage", "shell_net_charge",
                  "inner_total", "outer_total", "shell_voltage_difference", "inner_sigma"]
    expected: float
    point_m: tuple[float, float, float] | None = None
    conductor: str | None = None
    other_conductor: str | None = None
    theta_rad: float | None = None


@dataclass(frozen=True)
class AuthoredExample:
    states: tuple[tuple[str, ElectrostaticProblem], ...]
    checks: tuple[Check, ...]


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    page: int
    source_anchor: str
    summary: str
    classification: Literal["runnable", "new-component", "new-constraint",
                             "new-solver-domain", "new-dynamics"]
    blocker: str
    note: str
    author: Callable[[], AuthoredExample] | None = None

    @property
    def runnable(self) -> bool:
        return self.author is not None

    def metadata(self) -> dict:
        return {"id":self.id, "pdf":PDF_NAME, "pdf_page":self.page,
                "source_anchor":self.source_anchor, "summary":self.summary,
                "classification":self.classification,
                "runnable":self.runnable,"blocker":self.blocker,"note":self.note}


def shell(outer: float, charge: float | None, *, name: str, width: float = .01) -> SphericalShell:
    """Finite-thickness conductor whose OUTER radius matches a textbook thin shell."""
    return SphericalShell(outer-width, outer,
                          Grounded() if charge is None else FixedNetCharge(charge), name=name)


def one_shell():
    return (shell(A,Q,name="sphere"),)


def two_shells(inner_q: float = Q, outer_q: float = 2*Q):
    return (shell(A,inner_q,name="inner"), shell(B,outer_q,name="outer"))


def three_shells():
    return (shell(A,Q,name="inner"), shell(B,2*Q,name="middle"), shell(C,3*Q,name="outer"))


def problem(charges=(), shells=()) -> ElectrostaticProblem:
    return ElectrostaticProblem(charges=tuple(charges),conductors=tuple(shells))


def check(state, label, kind, expected, point=None, conductor=None, other=None, theta=None) -> Check:
    return Check(state,label,kind,float(expected),point,conductor,other,theta)


def sphere_radial() -> AuthoredExample:
    p=problem(shells=one_shell())
    return AuthoredExample((('Charged conductor (Q=1 nC, R=10 cm)',p),), (
        check(0,'potential at R/2','potential',K*Q/A,(A/2,0,0)),
        check(0,'potential at 3R/2','potential',K*Q/(1.5*A),(1.5*A,0,0)),
        check(0,'electric field in hollow interior','field_x',0,(A/2,0,0)),
    ))


def sphere_voltage_difference() -> AuthoredExample:
    p=problem(shells=one_shell())
    return AuthoredExample((('Centre and observation at 3R',p),),(
        check(0,'potential difference via two samples, equivalent to 6ER','potential',K*Q/A,(0,0,0)),
        check(0,'potential at 3R','potential',K*Q/(3*A),(3*A,0,0)),
        check(0,'E at 3R','field_x',K*Q/(9*A*A),(3*A,0,0)),
    ))


def two_shells_with_cavity_charge() -> AuthoredExample:
    p=problem(charges=(PointCharge(Q,(0,0,.02)),),shells=two_shells(2*Q,3*Q))
    return AuthoredExample((('Central-cavity charge and two shells',p),),(
        check(0,'inner shell net charge','shell_net_charge',2*Q,conductor='inner'),
        check(0,'outer shell net charge','shell_net_charge',3*Q,conductor='outer'),
        check(0,'inner surface induced charge','inner_total',-Q,conductor='inner'),
        check(0,'first shell potential','shell_voltage',K*((Q+2*Q)*(1/A-1/(B-.01))+(Q+2*Q+3*Q)/B),conductor='inner'),
    ))


def two_shells_equipotential() -> AuthoredExample:
    p=problem(shells=two_shells())
    expected_inner = K*(3*Q/B + Q*(1/A-1/(B-.01)))
    return AuthoredExample((('Two charged concentric shells',p),),(
        check(0,'centre potential','potential',expected_inner,(0,0,0)),
        check(0,'inner conductor potential','shell_voltage',expected_inner,conductor='inner'),
        check(0,'exterior potential at 2B','potential',K*3*Q/(2*B),(2*B,0,0)),
    ))


def two_shells_twice_charge() -> AuthoredExample:
    # Keep the original exercise's radii a and 3a. Metal is assigned a
    # nonzero thickness; shell potential inside the gap must account for
    # the induced charge on the inner face of the outer conductor.
    outer=3*A
    inner_boundary=outer-.01
    p=problem(shells=(shell(A,Q,name='inner'),shell(outer,2*Q,name='outer')))
    inside=K*(3*Q/outer + Q*(1/A-1/inner_boundary))
    gap=K*(3*Q/outer + Q*(1/.15-1/inner_boundary))
    return AuthoredExample((('Inner Q, outer 2Q at radii a and 3a',p),),(
        check(0,'potential in innermost cavity','potential',inside,(0,0,0)),
        check(0,'inter-shell gap potential','potential',gap,(.15,0,0)),
        check(0,'outer conductor total','shell_net_charge',2*Q,conductor='outer'),
    ))


def three_shell_superposition() -> AuthoredExample:
    p=problem(shells=three_shells())
    # Finite conductor widths alter conductor voltages relative to the
    # textbook's zero-thickness mathematical surfaces. These are the
    # independent Gauss-law potential drops for the authored geometry.
    outer=K*6*Q/C
    mid=outer+K*3*Q*(1/B-1/(C-.01))
    inner=mid+K*Q*(1/A-1/(B-.01))
    return AuthoredExample((('Three independently charged nested shells',p),),(
        check(0,'centre potential','potential',inner,(0,0,0)),
        check(0,'between first and second','potential',mid+K*Q*(1/.15-1/(B-.01)),(.15,0,0)),
        check(0,'between second and third','potential',outer+K*3*Q*(1/.25-1/(C-.01)),(.25,0,0)),
        check(0,'outside all shells','potential',K*6*Q/.4,(.4,0,0)),
    ))


def voltage_difference() -> AuthoredExample:
    p=problem(shells=two_shells(Q,2*Q))
    return AuthoredExample((('Potential difference of two shells',p),),(
        check(0,'inner minus outer conductor voltage','shell_voltage_difference',K*Q*(1/A-1/(B-.01)),conductor='inner',other='outer'),
    ))


def outer_charge_invariance() -> AuthoredExample:
    a=problem(shells=two_shells(Q,0))
    b=problem(shells=two_shells(Q,4*Q))
    delta=K*Q*(1/A-1/(B-.01))
    return AuthoredExample((('Before adding 4Q to outer shell',a),('After adding 4Q to outer shell',b)),(
        check(0,'before: potential difference','shell_voltage_difference',delta,conductor='inner',other='outer'),
        check(1,'after: potential difference remains unchanged','shell_voltage_difference',delta,conductor='inner',other='outer'),
        check(1,'outer shell final net charge','shell_net_charge',4*Q,conductor='outer'),
    ))


def inner_charge_scales_voltage() -> AuthoredExample:
    a=problem(shells=two_shells(Q,0))
    b=problem(shells=two_shells(4*Q,0))
    delta=K*Q*(1/A-1/(B-.01))
    return AuthoredExample((('Inner shell Q',a),('Inner shell 4Q',b)),(
        check(0,'initial voltage difference','shell_voltage_difference',delta,conductor='inner',other='outer'),
        check(1,'quadrupled difference','shell_voltage_difference',4*delta,conductor='inner',other='outer'),
    ))


def outer_grounded() -> AuthoredExample:
    p=problem(shells=(shell(A,Q,name='inner'),shell(B,None,name='outer')))
    return AuthoredExample((('Outer concentric shell grounded',p),),(
        check(0,'grounded outer potential','shell_voltage',0,conductor='outer'),
        check(0,'charge acquired by earth-connected outer','shell_net_charge',-Q,conductor='outer'),
        check(0,'field outside grounded outer shell','field_x',0,(.3,0,0)),
        check(0,'field between shells','field_x',K*Q/(.15*.15),(.15,0,0)),
    ))


def inner_grounded() -> AuthoredExample:
    p=problem(shells=(shell(A,None,name='inner'),shell(B,Q,name='outer')))
    return AuthoredExample((('Inner shell grounded and outer charged',p),),(
        check(0,'grounded inner potential','shell_voltage',0,conductor='inner'),
        check(0,'inner shell induced net charge','shell_net_charge',-Q/(B*(1/A-1/(B-.01)+1/B)),conductor='inner'),
        check(0,'outer shell remains at fixed Q','shell_net_charge',Q,conductor='outer'),
    ))


def cavity_charge_shift() -> AuthoredExample:
    shells=(shell(A,3*Q,name='body',width=.02),)
    centered=problem(charges=(PointCharge(Q),),shells=shells)
    shifted=problem(charges=(PointCharge(Q,(0,0,.04)),),shells=shells)
    return AuthoredExample((('Charge at centre',centered),('Charge displaced 4 cm',shifted)),(
        check(0,'central charge induces -q inner','inner_total',-Q,conductor='body'),
        check(0,'outer charge is 4q','outer_total',4*Q,conductor='body'),
        check(1,'off-center inner total remains -q','inner_total',-Q,conductor='body'),
        check(1,'outer total remains 4q','outer_total',4*Q,conductor='body'),
        check(0,'centered inner sigma uniform','inner_sigma',-Q/(4*pi*(A-.02)**2),conductor='body',theta=0),
        check(1,'near-side enhanced inner sigma','inner_sigma',-Q*((A-.02)**2-.04**2)/(4*pi*(A-.02)*((A-.02)-.04)**3),conductor='body',theta=0),
        check(1,'far-side weaker inner sigma','inner_sigma',-Q*((A-.02)**2-.04**2)/(4*pi*(A-.02)*((A-.02)+.04)**3),conductor='body',theta=pi),
    ))


def cavity_offcenter_neutral() -> AuthoredExample:
    body=shell(A,0,name='body',width=.02)
    p=problem(charges=(PointCharge(Q,(0,0,.035)),),shells=(body,))
    return AuthoredExample((('One off-centre charge in neutral spherical shell',p),),(
        check(0,'inner induced total','inner_total',-Q,conductor='body'),
        check(0,'outer induced total','outer_total',Q,conductor='body'),
        check(0,'outer shell potential','shell_voltage',K*Q/A,conductor='body'),
    ))


CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase('p26-conductor-potential',26,'Illustration 1','Potential at R/2 and 3R/2 for a charged conducting sphere.',
                  'runnable','','Exact spherical conductor field/potential; thin surface represented by a finite metal shell.',sphere_radial),
    BenchmarkCase('p26-centre-to-three-r',26,'Illustration 3','Potential difference between sphere centre and observation point at 3R.',
                  'runnable','','Rewrites the textbook E-at-3R condition using Q and R.',sphere_voltage_difference),
    BenchmarkCase('p27-cavity-two-shells',27,'Illustration 4','Potential with a point charge in a cavity and two concentric charged shells.',
                  'runnable','','A representative off-centre source is used; sampling and conductor quantities are checked.',two_shells_with_cavity_charge),
    BenchmarkCase('p27-two-shell-potentials',27,'Illustration 5','Potentials in a two-shell concentric arrangement.',
                  'runnable','','Finite shell thickness changes the induced potential relative to an ideal mathematical zero-thickness sheet.',two_shells_equipotential),
    BenchmarkCase('p28-two-shell-q-2q',28,'Illustration 6','Potentials for nested thin shells with charges Q and 2Q.',
                  'runnable','','Radii a, 3a match PDF; finite shell metal thickness changes internal potential drops.',two_shells_twice_charge),
    BenchmarkCase('p28-three-shells',28,'Illustration 7','Potential at four radial locations with three concentric shells.',
                  'runnable','','Radial probes and charges chosen as numeric SI representatives.',three_shell_superposition),
    BenchmarkCase('p29-shell-potential-difference',29,'Illustration 10','Potential difference between two concentric shells.',
                  'runnable','','Checks conductor potentials directly, not a renderer readout.',voltage_difference),
    BenchmarkCase('p29-outer-charge-change',29,'Illustration 11','Does adding charge to the outer shell alter the shell-to-shell potential difference?',
                  'runnable','','Two independently solved stored states; models changes as separate static equilibria.',outer_charge_invariance),
    BenchmarkCase('p30-inner-charge-change',30,'Illustration 12','What happens to shell potential difference when inner charge becomes 4Q?',
                  'runnable','','Checks 4x difference across two static equilibria.',inner_charge_scales_voltage),
    BenchmarkCase('p62-ground-outer',62,'Illustration 2','Grounded outer spherical shell surrounding a charged inner conductor.',
                  'runnable','','Net induced grounded charge and exterior shielding are checked.',outer_grounded),
    BenchmarkCase('p63-ground-inner',63,'Illustration 4','Inner concentric shell grounded; determine its resulting charge.',
                  'runnable','','The PDF ideal-thin-shell -Qa/b limit is adapted to finite metal thickness in the authored model.',inner_grounded),
    BenchmarkCase('p69-cavity-central-vs-shifted',69,'Illustration 7','Charge at centre versus displaced inside a spherical cavity: surface charge totals and symmetry.',
                  'runnable','','3q shell net charge and q central source; finite metal width chosen for validation.',cavity_charge_shift),
    BenchmarkCase('p70-cavity-offset',70,'Illustration 8(i)','Off-centre charge in a spherical cavity of a neutral conducting body.',
                  'runnable','','Spherical case only (part ii has an irregular outer surface).',cavity_offcenter_neutral),
    BenchmarkCase('p20-ring-axis',20,'Ring potential derivation','Potential on axis of a uniformly charged ring.',
                  'new-component','Charged ring field/potential primitive is not in the physical model.','No spherical-shell substitution.'),
    BenchmarkCase('p25-insulating-sphere',25,'Nonconducting sphere','Potential inside a uniformly charged insulating solid sphere.',
                  'new-component','Volumetric fixed charge density and dielectric sphere component are missing.','A conducting shell is not equivalent.'),
    BenchmarkCase('p13-projected-charge',13,'Illustration 7','Moving negative charge projected between two fixed point charges; turning point and initial velocity.',
                  'new-dynamics','No particle trajectories, energy/mass state or event integration.','Field/potential alone would not simulate motion.'),
    BenchmarkCase('p61-external-grounded-sphere',61,'Illustration 1','External point charge near an earthed conducting sphere.',
                  'new-solver-domain','Historical M6a baseline: external-source sphere physics was unavailable then.','M7 now implements this as SolidSphere + ExternalGroundedSphereBackend/Femm5ExternalSphereBackend; this original baseline remains frozen.'),
    BenchmarkCase('p68-connected-separated-spheres',68,'Illustration 4','Two widely separated charged spheres connected by a wire; equilibrium charge redistribution.',
                  'new-constraint','Separated connected conductors and charge-sharing constraints are not modeled.','No approximation using nested concentric shells.'),
    BenchmarkCase('p70-three-cavities',70,'Illustration 9','Multiple distinct cavities with charges inside one irregular conductor.',
                  'new-solver-domain','Requires a multi-cavity conductor geometry and 3D numerical backend.','Single concentric cavity cannot reproduce topology.'),
)

_BY_ID = {case.id:case for case in CASES}
assert len(_BY_ID)==len(CASES)


def load_case(case_id: str) -> BenchmarkCase:
    try:
        return _BY_ID[case_id]
    except KeyError as exc:
        raise ValueError(f"unknown benchmark {case_id!r}") from exc


def measure(solution, spec: Check, problem_: ElectrostaticProblem) -> float:
    if spec.kind=='potential':
        return solution.potential_at(spec.point_m)
    if spec.kind=='field_x':
        return solution.electric_field_at(spec.point_m)[0]
    if spec.kind=='inner_sigma':
        shell_ = next(s for s in problem_.conductors if s.name == spec.conductor)
        t=spec.theta_rad
        point=(shell_.center_m[0]+shell_.inner_radius_m*sin(t),
               shell_.center_m[1],
               shell_.center_m[2]+shell_.inner_radius_m*cos(t))
        return solution.surface_charge_density_at(spec.conductor,'inner',point)
    conductor = next(c for c in solution.conductor_results if c.name==spec.conductor)
    if spec.kind == 'shell_voltage':
        return conductor.potential_v
    if spec.kind == 'shell_net_charge':
        return conductor.net_charge_c
    if spec.kind == 'inner_total':
        return conductor.inner_surface_charge_c
    if spec.kind == 'outer_total':
        return conductor.outer_surface_charge_c
    if spec.kind == 'shell_voltage_difference':
        other = next(c for c in solution.conductor_results if c.name==spec.other_conductor)
        return conductor.potential_v-other.potential_v
    raise AssertionError(f'unknown observation kind {spec.kind}')


def run_checks(case: BenchmarkCase) -> dict:
    """Run independent analytic reference checks; *not* a FEMM test."""
    if not case.runnable:
        return {"id":case.id,"result":"unsupported", "reason":case.blocker}
    example=case.author()
    backend=NestedShellBackend()
    observations=[]
    for i,(label,p) in enumerate(example.states):
        if not backend.supports(p):
            return {"id":case.id,"result":"error", "reason":f"analytical backend rejected state {i}: {label}"}
    for spec in example.checks:
        label,p = example.states[spec.state]
        actual=measure(backend.solve(p),spec,p)
        # Independent closed-form expectations; numeric tolerance scales
        # with the expected magnitude, plus a tiny charge floor.
        atol=2e-19 if spec.kind in ('shell_net_charge','inner_total','outer_total','inner_sigma') else 1e-8
        ok=isclose(actual,spec.expected,rel_tol=2e-8,abs_tol=atol)
        observations.append({"label":spec.label,"state":label,"kind":spec.kind,
                             "expected":spec.expected,"actual":actual,"passed":ok})
    return {"id":case.id,"result":"passed" if all(x['passed'] for x in observations) else "failed",
            "states":len(example.states), "checks":observations}


def report() -> dict:
    results=[run_checks(case) for case in CASES]
    return {"pdf":PDF_NAME,"sha256":PDF_SHA256,"pages":PDF_PAGES,
            "cases":len(results),"runnable":sum(c.runnable for c in CASES),
            "blocked":sum(not c.runnable for c in CASES),
            "failures":sum(r['result'] in ('failed','error') for r in results),
            "results":results}


def export_case(case_id: str,output: Path,*,backend_name: str='analytical',resolution:int=31) -> dict:
    case=load_case(case_id)
    if not case.runnable:
        raise UnsupportedProblem(f"PDF p.{case.page}: {case.blocker}")
    authored=case.author()
    backend=NestedShellBackend() if backend_name=='analytical' else Femm5Backend()
    for label,p in authored.states:
        if not backend.supports(p):
            raise UnsupportedProblem(f"{case_id} state {label!r} unsupported by {backend_name} backend")
    largest_shell = max(shell_.outer_radius_m for _,state in authored.states
                        for shell_ in state.conductors)
    sample_extent = max((max(abs(x) for x in c.point_m)
                         for c in authored.checks if c.point_m is not None), default=0.0)
    extent = max(1.65*largest_shell,1.12*sample_extent)
    plane=SliceSpec.meridian(authored.states[0][1],extent_m=extent,resolution=resolution)
    data=sample_sweep(authored.states,backend,plane=plane)
    # Avoid using a simulated reference in place of a failed real FEMM run.
    data['benchmark_source']={"pdf":PDF_NAME,"pdf_sha256":PDF_SHA256,"page":case.page,
                              "source_anchor":case.source_anchor,"id":case.id,
                              "authoring_note":case.note}
    write_visualization(data,output,title=f"PDF p.{case.page}: {case.summary}")
    metadata=output.with_suffix('.json')
    metadata.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    return {"case":case.id,"backend":backend_name,"states":len(authored.states),
            "html":str(output),"json":str(metadata)}


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description='Curated M6 JEE conductor PDF benchmark')
    actions=parser.add_subparsers(dest='action',required=True)
    listing=actions.add_parser('list',help='List every source-anchored PDF question')
    listing.add_argument('--json',action='store_true')
    checking=actions.add_parser('check',help='Verify independent analytical expectations')
    checking.add_argument('--output',type=Path)
    rendering=actions.add_parser('render',help='Render one supported benchmark to offline HTML and JSON')
    rendering.add_argument('case_id')
    rendering.add_argument('--backend',choices=['analytical','femm'],default='analytical')
    rendering.add_argument('--resolution',type=int,default=31)
    rendering.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    if args.action=='list':
        if args.json: print(json.dumps([case.metadata() for case in CASES],ensure_ascii=False,indent=2))
        else:
            for case in CASES:
                status='RUN' if case.runnable else 'BLOCK'
                print(f"{case.id:39} p.{case.page:02} {status:5} {case.classification}: {case.summary}")
        return 0
    if args.action=='check':
        checked=report()
        txt=json.dumps(checked,ensure_ascii=False,indent=2,allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(txt,encoding='utf-8')
        print(f"PDF benchmark: {checked['runnable']} runnable, {checked['blocked']} blocked, {checked['failures']} failed")
        return 1 if checked['failures'] else 0
    if args.action=='render':
        output=args.output or Path('output/m6')/f'{args.case_id}.html'
        output.parent.mkdir(parents=True,exist_ok=True)
        try:
            item=export_case(args.case_id,output,backend_name=args.backend,resolution=args.resolution)
        except (UnsupportedProblem,ValueError) as exc:
            parser.exit(2,f"Cannot render: {exc}\n")
        print(json.dumps(item,indent=2))
        return 0
    raise AssertionError(args.action)


if __name__=='__main__':
    raise SystemExit(main())
