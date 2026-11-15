"""First execution of preselected author-exposed R, only after candidate freeze.

Independent reference: displaced harmonic oscillator inside a fixed uniformly
charged volume sphere. This is not an independent/blinded authoring trial.
"""
from math import pi,sqrt,atan2,sin,cos,hypot
from pathlib import Path
import argparse,json
from conductor_sim import EPSILON_0
from conductor_sim.motion_events import investigate_motion
from conductor_sim.motion_event_viewer import render_event_run
from examples.motion_event_studies import make_case


def check(output):
    study=make_case('reserved-transfer')
    frozen=study.scenario.as_dict()
    result=investigate_motion(study)
    assert study.scenario.as_dict()==frozen
    # Reference constants are from the charter, not derived from production E.
    omega2=1e-9/(4*pi*EPSILON_0)
    omega=sqrt(omega2)
    equilibrium=(-.3/omega2,0.,.1/omega2)
    initial=(.2,0.,.1);velocity=(.1,0.,-.03)
    def position(t):return tuple(e+(p-e)*cos(omega*t)+v/omega*sin(omega*t)
                             for e,p,v in zip(equilibrium,initial,velocity))
    def speed(t):return tuple(-omega*(p-e)*sin(omega*t)+v*cos(omega*t)
                             for e,p,v in zip(equilibrium,initial,velocity))
    turn=atan2(.1,omega*(.2-equilibrium[0]))/omega
    lo,hi=0.,.8
    for _ in range(70):
        mid=(lo+hi)/2
        if position(mid)[2]>.02:lo=mid
        else:hi=mid
    crossing=(lo+hi)/2
    assert result.trajectory.termination=='terminal-event'
    assert [e.event_id for e in result.events]==['turn:x','downward']
    times=[e.sample.time_s for e in result.events]
    assert abs(times[0]-turn)<2e-7,(times[0],turn)
    assert abs(times[1]-crossing)<2e-7,(times[1],crossing)
    position_error=velocity_error=0.
    for row in (*result.trajectory.samples,*(e.sample for e in result.events)):
        position_error=max(position_error,hypot(*(a-b for a,b in zip(row.position_m,position(row.time_s)))))
        velocity_error=max(velocity_error,hypot(*(a-b for a,b in zip(row.velocity_m_s,speed(row.time_s)))))
        assert hypot(*row.position_m)<1.
    assert position_error<2e-8 and velocity_error<2e-7,(position_error,velocity_error)
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(render_event_run(result,title='M24a · frozen mixed-source event transfer'),encoding='utf-8')
    path.with_suffix('.json').write_text(json.dumps(result.as_dict(),indent=2,allow_nan=False)+'\n',encoding='utf-8')
    report=dict(case='R',exposure='preselected same-author, first execution after candidate freeze',
        analytic_event_times_s=[turn,crossing],observed_event_times_s=times,
        max_position_error_m=position_error,max_velocity_error_m_s=velocity_error,
        unchanged_model=True,termination=result.trajectory.termination)
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();print(json.dumps(check(a.output),indent=2))
