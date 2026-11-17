"""Reproducible bounded repository/numerical audit, not an independent review.

The audit deliberately reports unresolved legacy-export findings separately
from new-pipeline correction checks. A green regression suite is not a proof
of every supported physics system or all data-integrity surfaces.
"""
from __future__ import annotations
import argparse
import ast
from dataclasses import replace
import json
from math import cos, sin, sqrt, pi, hypot, fsum, ulp
from pathlib import Path
from inspect import getfile

import conductor_sim
from sys import version

from conductor_sim import (
    PointCharge, PrescribedField, TestChargeMotionSpec, UniformElectricField,
    FixedPointField, UniformRodField, UniformRingField, UniformSphereField,
    UniformShellField, CoupledPropertyEdit, PhysicalWorldNode, WorldComparison,
    GraphWorldForkPlan, run_graph_world_fork,
)
from conductor_sim.prescribed_fields import _unit
from conductor_sim.prescribed_motion import integrate_test_charge
from conductor_sim.event_sweep import run_event_sweep
from conductor_sim.trajectory_sweep import run_trajectory_sweep
from examples.m28b_event_sweep import make_plan as sweep_plan
from examples.m34b_incremental_forks import build_source_graph

EPS=8.8541878128e-12  # explicit authored permittivity, not an imported oracle constant
K=1/(4*pi*EPS)


def inventory(root):
    modules=sorted((root/'src/conductor_sim').glob('*.py'))
    private=[]
    for p in modules:
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node,ast.ImportFrom) and node.level:
                private.extend((p.name,node.module,n.name) for n in node.names
                               if n.name.startswith('_'))
    workflows=list((root/'.github/workflows').glob('*.yml'))
    return dict(production_modules=len(modules),
        production_lines=sum(len(p.read_text().splitlines()) for p in modules),
        test_modules=len(list((root/'tests').glob('test_*.py'))),
        workflow_files=len(workflows),private_imports=len(private),
        workflows_without_explicit_permissions=[p.name for p in workflows
            if 'permissions:' not in p.read_text()],
        workflows_without_concurrency=[p.name for p in workflows
            if 'concurrency:' not in p.read_text()])


def corrections():
    source=build_source_graph()
    node=PhysicalWorldNode('new','old-a','Mass',(
        CoupledPropertyEdit('test-particle','mass_kg',2e-5),),'events')
    pair=WorldComparison('pair','old-a','new')
    object.__setattr__(pair,'left_world_id','new')
    try:
        plan=GraphWorldForkPlan(source,'audit:self',(node,),source.plan.questions,
                                source.plan.policy,(pair,))
        outcome=run_graph_world_fork(plan).as_dict()['cross_world_comparisons'][0]['status']
        self_pair=dict(rejected=False,exported_status=outcome)
    except (ValueError,TypeError) as error:
        self_pair=dict(rejected=True,error=str(error))
    object.__setattr__(node,'observation','typo-event')
    try:
        # Supply no event policy: the malformed mode must reach the mode guard,
        # not be rejected earlier for a policy on a non-event request.
        plan=GraphWorldForkPlan(source,'audit:mode',(node,))
        malformed=dict(rejected=False,status=run_graph_world_fork(plan).entries[0].status)
    except (ValueError,TypeError) as error:
        malformed=dict(rejected=True,error=str(error))
    unit=_unit((ulp(0.),ulp(0.),0.),'subnormal direction')
    return dict(mutated_self_pair=self_pair,malformed_observation=malformed,
                subnormal_normal=dict(value=unit,length=hypot(*unit),
                                      unit_within_ulp=abs(hypot(*unit)-1.)<=ulp(1.)))


def legacy_export_findings():
    plan=sweep_plan('uniform-mass')
    events=run_event_sweep(plan)
    bad=replace(events.entries[0],scenario=None,run=None,status='complete',reason='forged')
    try:
        rec=replace(events,entries=(bad,)+events.entries[1:]).as_dict()['entries'][0]
        event=dict(accepts_complete_without_native=rec['status']=='complete' and rec['event_run'] is None)
    except (ValueError,TypeError):
        event=dict(accepts_complete_without_native=False)
    motion=run_trajectory_sweep(plan.sweep)
    bad=replace(motion.entries[0],scenario=None,trajectory=None,status='complete',explanation='forged')
    try:
        rec=replace(motion,entries=(bad,)+motion.entries[1:]).as_dict()['entries'][0]
        trajectory=dict(accepts_complete_without_native=rec['status']=='complete' and rec['trajectory'] is None)
    except (ValueError,TypeError):
        trajectory=dict(accepts_complete_without_native=False)
    return dict(event_sweep=event,trajectory_sweep=trajectory,
        disposition='Separate legacy-result validation hardening; these types are not accepted as M35a source histories.')


def observation(source,point):
    return source.potential_at(point,permittivity_f_m=EPS),source.electric_field_at(point,permittivity_f_m=EPS)


def physics_transfers():
    point=(.23,.17,-.08)
    sources=[FixedPointField(2e-9,(-.03,.01,.05)),
        UniformRodField(2e-9,(-.2,-.1,.05),(.1,.2,-.1)),
        UniformRingField(2e-9,.08,(-.02,.01,.03),(1.,2.,-1.)),
        UniformSphereField(2e-9,.15,(0.,0.,0.)),
        UniformShellField(2e-9,.15,(0.,0.,0.))]
    gradients=[]
    for source in sources:
        actual=observation(source,point)[1]
        step=1e-5
        predicted=[]
        for k in range(3):
            def value(n):
                p=list(point);p[k]+=n*step
                return observation(source,tuple(p))[0]
            predicted.append((value(2)-8*value(1)+8*value(-1)-value(-2))/(12*step))
        relative=max(abs(a-b) for a,b in zip(actual,predicted))/max(hypot(*actual),1e-30)
        assert relative<2e-7,(type(source).__name__,relative)
        gradients.append(dict(source=type(source).__name__,relative_gradient_error=relative))
    # Numerical Coulomb quadrature entirely independent of rod/ring field formulas.
    quadratures=[]
    for source in sources[1:3]:
        count=16000
        sites=[]
        if type(source) is UniformRodField:
            for i in range(count):
                fraction=(i+.5)/count
                sites.append(tuple(source.start_m[k]+fraction*(source.end_m[k]-source.start_m[k]) for k in range(3)))
        else:
            normal=source.axis
            length=hypot(*normal);normal=tuple(x/length for x in normal)
            helper=(1.,0.,0.) if abs(normal[0])<.8 else (0.,1.,0.)
            def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
            e1=cross(normal,helper);n=hypot(*e1);e1=tuple(x/n for x in e1)
            e2=cross(normal,e1)
            for i in range(count):
                angle=2*pi*(i+.5)/count
                sites.append(tuple(source.center_m[k]+source.radius_m*(cos(angle)*e1[k]+sin(angle)*e2[k]) for k in range(3)))
        distances=[tuple(point[k]-site[k] for k in range(3)) for site in sites]
        q=source.charge_c/count
        potential=K*q*fsum(1/hypot(*d) for d in distances)
        electric=tuple(K*q*fsum(d[k]/hypot(*d)**3 for d in distances) for k in range(3))
        V,E=observation(source,point)
        errV=abs(V-potential)/abs(potential)
        errE=hypot(*(E[k]-electric[k] for k in range(3)))/hypot(*electric)
        assert errV<2e-8 and errE<2e-8,(type(source).__name__,errV,errE)
        quadratures.append(dict(source=type(source).__name__,quadrature_elements=count,
                                relative_potential_error=errV,relative_field_error=errE))
    # Nonuniform inverse-square trajectory with independently known circular orbit.
    Q=1e-9;q=-1e-9;mass=1e-6;radius=.1
    speed=sqrt(K*abs(Q*q)/(mass*radius));omega=speed/radius;period=2*pi/omega
    spec=TestChargeMotionSpec(PrescribedField((FixedPointField(Q,(0.,0.,0.)),),EPS),
        PointCharge(q,(radius,0.,0.)),mass,(0.,speed,0.),period,period/64,
        plane_normal=(0.,0.,1.),relative_tolerance=1e-10,length_scale_m=radius,
        source_guard_m=1e-4)
    run=integrate_test_charge(spec)
    assert run.termination=='duration'
    position_error=velocity_error=0.
    for sample in run.samples:
        t=sample.time_s
        expected_p=(radius*cos(omega*t),radius*sin(omega*t),0.)
        expected_v=(-speed*sin(omega*t),speed*cos(omega*t),0.)
        position_error=max(position_error,hypot(*(sample.position_m[k]-expected_p[k] for k in range(3))))
        velocity_error=max(velocity_error,hypot(*(sample.velocity_m_s[k]-expected_v[k] for k in range(3))))
    assert position_error<1e-7 and velocity_error<5e-7,(position_error,velocity_error)
    return dict(equals_negative_potential_gradient=gradients,independent_coulomb_quadrature=quadratures,
        nonuniform_circular_orbit=dict(actual_saved_samples=len(run.samples),
        position_error_m=position_error,velocity_error_m_s=velocity_error))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('output/m35a/audit.json'))
    parser.add_argument('--expect',choices=('released','candidate'),default='candidate')
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1],
                        help='Repository whose source inventory is being audited')
    args=parser.parse_args()
    source_root=args.source_root.resolve()
    package_root=Path(conductor_sim.__file__).resolve().parent
    example_path=Path(getfile(build_source_graph)).resolve()
    assert package_root == source_root/'src/conductor_sim', (package_root,source_root)
    assert example_path.is_relative_to(source_root/'examples'), example_path
    findings=corrections()
    expected_fixed=args.expect=='candidate'
    # Every red-before/green-after probe must hit its intended boundary.
    # A single conjunction could hide one falsely green released-source probe.
    for probe,flag in (('mutated_self_pair','rejected'),
                       ('malformed_observation','rejected'),
                       ('subnormal_normal','unit_within_ulp')):
        assert findings[probe][flag] is expected_fixed, (probe,findings[probe])
    report=dict(interpreter=version,expected_source_behavior=args.expect,
        imported_production_root=str(package_root),
        imported_example_path=str(example_path),
        inventory=inventory(source_root),
        reproduced_corrections=findings,physics=physics_transfers(),
        open_legacy_findings=legacy_export_findings(),
        audit_scope='Complete source inventory and targeted native numerical/validation probes; not exhaustive proof or separate-context review.',
        result='AUDIT_COMPLETE_WITH_DOCUMENTED_OPEN_FINDINGS')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(report,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
