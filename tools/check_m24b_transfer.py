"""Reserved R: execute only after candidate freeze; retain first outcome.

Recipe from the preimplementation charter, not an implementation-tuned example.
Independent phasewise affine damped-harmonic solution and work quadrature.
"""
from math import cos, exp, sin, sqrt, pi, hypot
from pathlib import Path
import argparse
import json
from conductor_sim import (EPSILON_0, UniformSphereField, PrescribedField,
    PointCharge, UniformGravity, TestChargeMotionSpec)
from conductor_sim.scenarios import PrescribedScenario
from conductor_sim.driven_motion import (DrivenModel, DrivenStudy, LinearDrag,
    UniformFieldSchedule, FieldPhase, EventPolicy, PlaneEvent, VelocityReversal,
    investigate_driven_motion)
from conductor_sim.driven_viewer import render_driven_run


def run(output):
    q=-1e-9;mass=1e-9;b=.7e-9;Q=1e-9;R=1.;g=(0,0,-.1)
    p0=(.2,0,.1);v0=(.1,0,-.03);switch=.31;duration=.8
    fields=((.3,0,-.2),(-.2,0,.15))
    base=PrescribedScenario('m24b:reserved',TestChargeMotionSpec(
        PrescribedField((UniformSphereField(Q,R),)),PointCharge(q,p0),mass,v0,duration,.1,
        gravity=UniformGravity(g),plane_normal=(0,1,0),relative_tolerance=1e-10),
        (('sphere:fixed','Fixed uniform volume sphere'),),
        'Reserved preselected M24b transfer; teacher hypothetical, author-exposed not blind. Sphere, scheduled E, gravity and drag.')
    model=DrivenModel(base,LinearDrag(b),UniformFieldSchedule((FieldPhase(0,fields[0]),FieldPhase(switch,fields[1]))),
        'driven:reserved','Frozen M24b transfer: two event questions on unchanged mixed-source driven model. This is not a blinded LLM trial.')
    study=DrivenStudy(model,(VelocityReversal('turn:x',(1,0,0),'x reversal'),
        PlaneEvent('down',(0,0,.02),(0,0,1),'Downward detector',-1,True)),
        EventPolicy(.01,1e-9,1e-10,1e-9))
    original=model.as_dict()
    result=investigate_driven_motion(study)
    K=1/(4*pi*EPSILON_0);omega2=-q*K*Q/(mass*R**3);alpha=b/(2*mass);w=sqrt(omega2-alpha*alpha)
    def phase(r0,v0,E,t):
        a=tuple(q*e/mass+gv for e,gv in zip(E,g));eq=tuple(x/omega2 for x in a)
        A=tuple(x-y for x,y in zip(r0,eq));B=tuple((vv+alpha*x)/w for vv,x in zip(v0,A))
        decay=exp(-alpha*t);c=cos(w*t);s=sin(w*t)
        r=tuple(eq[i]+decay*(A[i]*c+B[i]*s) for i in range(3))
        v=tuple(decay*(-alpha*(A[i]*c+B[i]*s)+w*(-A[i]*s+B[i]*c)) for i in range(3))
        return r,v
    rp,vp=phase(p0,v0,fields[0],switch)
    def ref(t):return phase(p0,v0,fields[0],t) if t<=switch else phase(rp,vp,fields[1],t-switch)
    assert max(hypot(*ref(duration*i/1000)[0]) for i in range(1001))<R
    def first_root(fun,direction):
        lo=0.;fl=fun(lo)
        for j in range(1,2001):
            hi=duration*j/2000;fh=fun(hi)
            if fl*fh<0 and (direction==0 or (1 if fh>fl else -1)==direction):
                for _ in range(70):
                    mid=(lo+hi)/2;fm=fun(mid)
                    if fl*fm>0:lo=mid;fl=fm
                    else:hi=mid
                return (lo+hi)/2
            lo=hi;fl=fh
        raise AssertionError('reserved reference did not cross')
    expected={'turn:x':first_root(lambda t:ref(t)[1][0],0),
              'down':first_root(lambda t:ref(t)[0][2]-.02,-1)}
    actual={e.event_id:e.sample.time_s for e in result.events}
    assert set(actual)==set(expected)
    max_time=max(abs(actual[k]-v) for k,v in expected.items());assert max_time<5e-8
    max_p=max_v=0.
    for row in (*result.samples,*(e.sample for e in result.events)):
        r,v=ref(row.time_s);max_p=max(max_p,hypot(*(x-y for x,y in zip(r,row.position_m))))
        max_v=max(max_v,hypot(*(x-y for x,y in zip(v,row.velocity_m_s))))
    assert max_p<2e-8 and max_v<5e-8
    end=result.samples[-1].time_s;rend,vend=ref(end)
    def simpson(lo,hi):
        n=2000;h=(hi-lo)/n
        return h/3*sum((1 if i in (0,n) else 4 if i%2 else 2)*b*hypot(*ref(lo+i*h)[1])**2 for i in range(n+1))
    D=simpson(0,min(end,switch))+(simpson(switch,end) if end>switch else 0)
    Wfixed=q*K*Q/(2*R**3)*(hypot(*rend)**2-hypot(*p0)**2)
    Wg=mass*sum(gg*(rr-r0) for gg,rr,r0 in zip(g,rend,p0))
    def work(E,a,z):return q*sum(e*(zz-aa) for e,aa,zz in zip(E,a,z))
    Wapplied=work(fields[0],p0,ref(min(end,switch))[0])+(work(fields[1],rp,rend) if end>switch else 0)
    last=result.samples[-1]
    for observed,reference in ((last.dissipated_energy_j,D),(last.fixed_work_j,Wfixed),
                                (last.gravitational_work_j,Wg),(last.applied_work_j,Wapplied)):
        assert abs(observed-reference)<3e-17,(observed,reference)
    assert result.termination=='terminal-event' and model.as_dict()==original
    assert len(result.switches)==1 and result.switches[0].time_s==switch
    report=dict(case='R',reference='independent affine damped-harmonic solution + Simpson dissipation quadrature',
        expected_event_times_s=expected,actual_event_times_s=actual,max_time_error_s=max_time,
        max_position_error_m=max_p,max_velocity_error_m_s=max_v,reference_dissipation_j=D,
        computed_dissipation_j=last.dissipated_energy_j,work_balance_residual_j=last.work_balance_error_j,
        model_retained_unchanged=True,exposure='preselected developer-exposed, first execution must be retained; not blind')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(render_driven_run(result,title='M24b · Reserved mixed-source transfer'),encoding='utf-8')
    record=result.as_dict();record['independent_reference']=report
    output.with_suffix('.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path('output/m24b-transfer.html'))
    a=ap.parse_args();print(json.dumps(run(a.output),indent=2))
