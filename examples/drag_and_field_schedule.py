"""M24b: one reusable driven model, not a separate solver per demonstration.

The oil source supplies equilibrium data, not a unique drag coefficient or
switching history. Its transient inputs below are explicitly hypothetical.
"""
from dataclasses import replace
import argparse
import json
from math import pi
from pathlib import Path

from conductor_sim import (PrescribedField, UniformElectricField, UniformGravity,
    FixedPointField, PointCharge, TestChargeMotionSpec)
from conductor_sim.scenarios import PrescribedScenario
from conductor_sim.driven_motion import (LinearDrag, FieldPhase, UniformFieldSchedule,
    DrivenModel, DrivenStudy, EventPolicy, PlaneEvent, VelocityReversal,
    investigate_driven_motion, revise_driven_model)

CASES=('constant-drag','switched-oblique','oil-hypothetical','revised-drag','source-guard')


def make_case(case):
    metadata={};guard=1e-6;work_atol=1e-24
    if case=='constant-drag':
        sources=(UniformElectricField((.4,0,-.2)),);g=(0,0,-.3)
        q,m,b=1e-9,1e-9,2e-9;r=(.1,0,.3);v=(.3,0,.2);duration=1.;interval=.1
        phases=(FieldPhase(0,(0,0,0)),)
        questions=(VelocityReversal('apex',(0,0,1),'Vertical apex, not full rest',-1),)
        pol=EventPolicy(.025,1e-9,1e-10,1e-9)
        note='Teacher hypothetical P1: independent fixed E, gravity and stationary-medium linear drag. Compare with the vector exponential response; dissipation is not numerical drift.'
    elif case in ('switched-oblique','revised-drag'):
        sources=();g=(0,0,-.2);q,m,b=1e-9,1e-9,1e-9
        r=(0,0,.4);v=(.4,0,.15);duration=1.5;interval=.1
        phases=(FieldPhase(0,(.2,0,0)),FieldPhase(.37,(-1.2,0,.5)))
        questions=(VelocityReversal('turn:x',(1,0,0),'Post-switch x reversal'),
            PlaneEvent('upper',(0,0,.5),(0,0,1),'Upper passive detector',1,True))
        pol=EventPolicy(.02,1e-9,1e-10,1e-9)
        note='Teacher hypothetical P2: oblique motion; E switches at exactly .37 s between output times. One model answers reversal, detector and energy questions. No velocity reset or impulse.'
    elif case=='oil-hypothetical':
        # Original source inputs. The continuous effective q reproduces exact
        # classical force balance; it is NOT asserted to be an integer charge.
        rho=900.;radius=8e-7;spacing=.01;voltage=200.;g=(0,0,-10.)
        mass=(4/3)*pi*radius**3*rho;E=voltage/spacing
        q=-mass*10/E;m=mass;tau=.002;b=m/tau
        sources=();r=(0,0,.008);v=(0,0,-10*tau);duration=.035;interval=.002
        phases=(FieldPhase(0,(0,0,0)),FieldPhase(.013,(0,0,-E)))
        questions=(VelocityReversal('rest-test',(0,0,1),'Test for true vertical reversal'),
            PlaneEvent('lower',(0,0,0),(0,0,1),'Ideal lower limit',-1,True,'ideal-stop-surface'),
            PlaneEvent('upper',(0,0,spacing),(0,0,1),'Ideal upper limit',1,True,'ideal-stop-surface'))
        pol=EventPolicy(.0005,1e-10,1e-11,1e-10);work_atol=1e-28
        note=('JEE Advanced 2020 Paper 2 Q3 motivates equilibrium ONLY. Teacher hypothetical: stationary-medium linear drag, tau=.002 s, already-terminal downward launch at z=.008 m, switch at .013 s. '
              'Continuous effective q=-mg/E is used for exact classical balance (about 6.032 electron charges; source integer answer rounds to 6). This is not a unique original-exam trajectory or exact quantized equilibrium. '
              'After switching, fall speed decays exponentially and never resets to zero. Uniform-field approximation; no buoyancy, fringe fields, conductor response or switching electromagnetics.')
        metadata=dict(original_source=dict(title='JEE Advanced 2020 Paper 2 Q3',
            url='https://jeeadv.ac.in/past_qps/2020_2_English.pdf',workbook_pdf_page=16,
            density_kg_m3=rho,radius_m=radius,spacing_m=spacing,voltage_v=voltage,
            gravity_m_s2=10.,electron_charge_magnitude_c=1.6e-19,neglect_buoyancy=True),
            derived_equilibrium=dict(mass_kg=m,field_v_m=E,continuous_charge_c=q,
                electron_charge_count=abs(q)/1.6e-19,rounded_integer_count=round(abs(q)/1.6e-19)),
            teacher_additions=dict(drag_coefficient_kg_s=b,relaxation_time_s=tau,
                initial_position_m=r,initial_velocity_m_s=v,switch_time_s=.013,
                continuous_balancing_charge_not_exact_integer=True))
    elif case=='source-guard':
        sources=(FixedPointField(-1e-9,(0,0,0)),);g=(0,0,0)
        q,m,b=1e-9,1e-6,1e-7;r=(.1,0,0);v=(-1,0,0);duration=.5;interval=.01;guard=.01
        phases=(FieldPhase(0,(0,0,0)),FieldPhase(.3,(.1,0,0)))
        questions=(PlaneEvent('inside-guard',(.005,0,0),(1,0,0),'Unqualified inner detector',-1,True),)
        pol=EventPolicy(.005,1e-9,1e-9,1e-8)
        note='Teacher hypothetical: attractive fixed point plus drag. A source guard truncates observation before the requested inner detector and before the scheduled change; neither is fabricated.'
    else:raise ValueError('unknown driven example')
    spec=TestChargeMotionSpec(PrescribedField(sources),PointCharge(q,r),m,v,duration,interval,
        plane_normal=(0,1,0),relative_tolerance=1e-10,length_scale_m=.01 if case=='oil-hypothetical' else 1.,
        source_guard_m=guard,gravity=UniformGravity(g))
    base_note=(note if len(note)<=400 else 'JEE Advanced 2020 Q3 equilibrium source; transient teacher hypothetical. Full source givens, continuous-charge idealization, added drag/launch/switching parameters and limitations are retained in the DrivenModel provenance and example record.')
    base=PrescribedScenario('m24b:'+case,spec,tuple((f'fixed:{i}',type(s).__name__) for i,s in enumerate(sources)),base_note)
    model=DrivenModel(base,LinearDrag(b),UniformFieldSchedule(phases),'driven:'+case,note)
    if case=='revised-drag':
        original,_=make_case('switched-oblique')
        model,revision=revise_driven_model(original.model,model_id='driven:doubled-drag',
            note='Teacher variation: double drag, retain all source, launch and schedule data.',
            drag=replace(original.model.drag,coefficient_kg_s=2*original.model.drag.coefficient_kg_s))
        metadata['physical_revision']=revision
    return DrivenStudy(model,questions,pol,work_absolute_tolerance_j=work_atol),metadata


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--case',choices=CASES,default=CASES[1]);ap.add_argument('--output',type=Path,default=Path('output/m24b-driven.html'));a=ap.parse_args()
    from conductor_sim.driven_viewer import render_driven_run
    study,meta=make_case(a.case);run=investigate_driven_motion(study)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(render_driven_run(run,title='M24b · '+a.case),encoding='utf-8')
    record=run.as_dict();record['example_provenance']=meta
    a.output.with_suffix('.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(a.case,run.termination,'switches',len(run.switches),'events',len(run.events),'states',len(run.samples),'work residual J',run.max_abs_work_balance_error_j)


if __name__=='__main__':main()
