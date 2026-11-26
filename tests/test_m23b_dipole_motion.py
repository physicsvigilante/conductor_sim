"""M23b preselected physics, numerical, guard and public authoring regressions."""
from dataclasses import replace
from math import cos, sin, sqrt, pi, hypot, fsum
import json
import pytest

from conductor_sim import EPSILON_0
from conductor_sim.dipoles import RigidDipole, DipoleModel, orient_dipole, observe_dipole
from conductor_sim.structural_scenarios import NamedSource
from conductor_sim.prescribed_fields import (FixedPointField, UniformElectricField,
    UniformRodField, UniformShellField, UniformRingAxisField, UniformDiskAxisField)
from conductor_sim.dipole_motion import (DipoleMotionSpec, integrate_dipole_motion,
    revise_dipole_motion, validate_dipole_motion, _swept_safe)
from conductor_sim.dipole_motion_viewer import render_dipole_motion, write_dipole_motion
from conductor_sim.solver import UnsupportedProblem
from conductor_sim.motion_1d import MotionIntegrationError
from examples.dipole_rotation import CASES, make_case

K = 1/(4*pi*EPSILON_0)


@pytest.fixture(scope='module')
def runs():
    return {case: integrate_dipole_motion(make_case(case)) for case in CASES}


def test_case_inventory_and_qualified_source_identity(runs):
    assert len(runs)==6
    for case,r in runs.items():
        assert validate_dipole_motion(r)
        assert r.spec.model.model_id=='m23b:'+case
        assert 'Teacher hypothetical' in r.spec.model.provenance
        d=r.as_dict()
        assert d['schema']=='conductor_sim.dipole_motion.v1'
        assert d['spec']['model_digest']==r.spec.model.digest
        assert d['termination']==('source-guard' if case=='source-guard' else 'duration')
        assert all(s['observation']['torque_n_m'] is not None for s in d['samples'])
        assert r.spec.model.dipole.endpoints_m==make_case(case).model.dipole.endpoints_m


def test_uniform_pendulum_torque_energy_rigidity_and_phase(runs):
    s=runs['small-oscillation'].spec
    pE=1e-9*.04*120
    for row in runs['small-oscillation'].samples:
        assert row.axis_torque_n_m==pytest.approx(-pE*sin(row.angle_rad),rel=2e-13,abs=1e-24)
        assert row.angular_acceleration_rad_s2==pytest.approx(-100*sin(row.angle_rad),rel=2e-13,abs=1e-15)
        assert row.interaction_energy_j==pytest.approx(-pE*cos(row.angle_rad),rel=2e-13)
        assert row.kinetic_energy_j==pytest.approx(.5*4.8e-11*row.angular_velocity_rad_s**2,rel=2e-13)
        obs=observe_dipole(orient_dipole(s.model,row.angle_rad))
        a,b=obs['positive_position_m'],obs['negative_position_m']
        assert hypot(*(x-y for x,y in zip(a,b)))==pytest.approx(.04,abs=2e-16)
        assert obs['net_force_n']==[0.,0.,0.]
    assert runs['small-oscillation'].as_dict()['max_abs_energy_drift_j']<2e-17


def test_small_angle_frequency_and_independent_nonlinear_period(runs):
    # Energy-derived pendulum period via independent Simpson quadrature:
    # T=4/w0*integral_0^(pi/2) (1-sin^2(theta0/2)*sin^2(x))^(-1/2) dx.
    a=.1;w0=10.;n=2000;h=(pi/2)/n
    vals=[1/sqrt(1-sin(a/2)**2*sin(i*h)**2) for i in range(n+1)]
    integral=h/3*(vals[0]+vals[-1]+4*sum(vals[1:-1:2])+2*sum(vals[2:-1:2]))
    period=4*integral/w0
    assert abs(period/(2*pi/w0)-1)<.001
    events=runs['small-oscillation'].as_dict()['turning_brackets']
    assert events[0]['start_s'] <= period/2 <= events[0]['end_s']
    assert events[1]['start_s'] <= period <= events[1]['end_s']
    # Independent small-angle solution, within its O(theta0^3) approximation.
    for row in runs['small-oscillation'].samples:
        assert abs(row.angle_rad-a*cos(w0*row.time_s))<.001


def test_energy_threshold_separates_libration_and_rotation(runs):
    critical=2*sqrt((1e-9*.04*120)/4.8e-11)
    assert critical==pytest.approx(20.)
    bound=runs['libration'];rot=runs['rotation']
    assert 0<max(r.angle_rad for r in bound.samples)<pi
    assert any(r.angular_velocity_rad_s<0 for r in bound.samples)
    assert min(r.angular_velocity_rad_s for r in rot.samples)>0
    assert rot.samples[-1].angle_rad>4*2*pi
    for result in (bound,rot):
        assert max(abs(r.energy_drift_j) for r in result.samples)<1e-15
    assert bound.as_dict()['turning_brackets']
    assert rot.as_dict()['turning_brackets']==[]


def point_reference(theta):
    # Independently assembled endpoints and Coulomb forces about actual pivot.
    pivot=(-.015,0.);center=(0.,0.);Q=2e-9;q=1e-9
    torque=0.;energy=0.
    for sign,dx in ((1,.02),(-1,-.02)):
        vx=center[0]+dx-pivot[0];vy=center[1]-pivot[1]
        rx=cos(theta)*vx-sin(theta)*vy;ry=sin(theta)*vx+cos(theta)*vy
        px=rx+pivot[0];py=ry+pivot[1]
        ex=px-.12;ey=py-.06;r=hypot(ex,ey)
        fx=sign*q*K*Q*ex/r**3;fy=sign*q*K*Q*ey/r**3
        torque+=rx*fy-ry*fx;energy+=sign*q*K*Q/r
    return torque,energy


def verlet(acceleration, angle, omega, duration, steps):
    h=duration/steps;a=acceleration(angle)
    for _ in range(steps):
        new=angle+h*omega+.5*h*h*a
        na=acceleration(new)
        omega+=.5*h*(a+na);angle,a=new,na
    return angle,omega


def test_point_pivot_independent_coulomb_and_alternate_time_integration(runs):
    result=runs['point-pivot']
    for row in result.samples:
        tau,u=point_reference(row.angle_rad)
        assert row.axis_torque_n_m==pytest.approx(tau,rel=1e-10,abs=1e-22)
        assert row.interaction_energy_j==pytest.approx(u,rel=1e-10,abs=1e-22)
    coarse=verlet(lambda a:point_reference(a)[0]/4.8e-11,.4,1.,.5,10000)
    fine=verlet(lambda a:point_reference(a)[0]/4.8e-11,.4,1.,.5,20000)
    observed=(result.samples[-1].angle_rad,result.samples[-1].angular_velocity_rad_s)
    assert observed==pytest.approx(fine,rel=1e-6,abs=1e-5)
    assert abs(fine[0]-observed[0])<abs(coarse[0]-observed[0])
    o=result.as_dict()['samples'][-1]['observation']
    assert hypot(*o['dipole']['center_m'])>.001  # centre really orbits off-centre pivot


def test_mixed_unfamiliar_arrangement_independent_line_quadrature(runs):
    result=runs['mixed-sources']
    for row in result.samples[::25]:
        theta=row.angle_rad;torque=energy=0.
        for sign in (1.,-1.):
            px=sign*.02*cos(theta);py=sign*.02*sin(theta)
            e=[90.,25.,0.];v=-90*px-25*py
            # Uniform rod numerical integral, independent of production rod formula.
            n=2048
            for j in range(n):
                u=(j+.5)/n;x=-.09+.16*u;y=.12+.02*u
                dx=px-x;dy=py-y;r=hypot(dx,dy)
                v+=K*.3e-9/n/r
                e[0]+=K*.3e-9/n*dx/r**3;e[1]+=K*.3e-9/n*dy/r**3
            dx=px-.15;dy=py+.08;r=hypot(dx,dy)
            assert r>.03
            v+=K*.4e-9/r;e[0]+=K*.4e-9*dx/r**3;e[1]+=K*.4e-9*dy/r**3
            torque+=sign*1e-9*(px*e[1]-py*e[0]);energy+=sign*1e-9*v
        assert row.axis_torque_n_m==pytest.approx(torque,rel=2e-6,abs=1e-15)
        assert row.interaction_energy_j==pytest.approx(energy,rel=2e-6,abs=1e-14)
    assert len(result.spec.model.sources)==3
    assert max(abs(r.energy_drift_j) for r in result.samples)<1e-15


def test_minus_energy_derivative_equals_axis_torque():
    s=make_case('mixed-sources');h=1e-5
    for theta in (.2,.7,1.6,3.4):
        plus=observe_dipole(orient_dipole(s.model,theta+h))['interaction_energy_j']
        minus=observe_dipole(orient_dipole(s.model,theta-h))['interaction_energy_j']
        tau=observe_dipole(orient_dipole(s.model,theta))['torque_n_m'][2]
        assert tau==pytest.approx(-(plus-minus)/(2*h),rel=2e-7,abs=1e-17)


@pytest.mark.parametrize('omega',(0.,3.,-5.))
def test_zero_field_free_spin_and_no_spurious_damping(omega):
    s=make_case('small-oscillation')
    s=replace(s,model=replace(s.model,sources=()),initial_angular_velocity_rad_s=omega)
    r=integrate_dipole_motion(s)
    for row in r.samples:
        assert row.angle_rad==pytest.approx(.1+omega*row.time_s,abs=1e-12)
        assert row.angular_velocity_rad_s==omega
        assert row.axis_torque_n_m==0
        assert row.energy_drift_j==0


@pytest.mark.parametrize('theta',(0.,pi))
def test_exact_uniform_rotational_equilibria_have_no_invented_perturbation(theta):
    s=replace(make_case('small-oscillation'),initial_angle_rad=theta,duration_s=.1)
    r=integrate_dipole_motion(s)
    assert max(abs(x.angle_rad-theta) for x in r.samples)<1e-13
    assert max(abs(x.angular_velocity_rad_s) for x in r.samples)<1e-12


def test_swept_arc_guard_is_not_an_endpoint_or_chord_only_check(runs):
    s=make_case('source-guard')
    assert not _swept_safe(s,-.3,.3)
    assert not _swept_safe(s,-pi,pi)  # endpoints identical; full circle hits the point
    assert _swept_safe(s,-.3,-.2)
    r=runs['source-guard']
    assert r.termination=='source-guard' and r.samples[-1].time_s<s.duration_s
    assert s.source_guard_m<r.samples[-1].source_clearance_m<1.021*s.source_guard_m
    assert all(x.source_clearance_m>s.source_guard_m for x in r.samples)
    assert r.samples[-1].time_s != round(r.samples[-1].time_s/.005)*.005


@pytest.mark.parametrize('kind',('rod','shell'))
def test_whole_arc_also_refuses_line_singularity_and_shell_jump(kind):
    s=make_case('source-guard')
    source=(UniformRodField(-1e-13,(.019,0.,0.),(.021,0.,0.)) if kind=='rod'
            else UniformShellField(1e-12,.019,(.02,0.,0.)))
    # Shell at offset has a surface crossing during the sweep, not endpoint overlap.
    s=replace(s,model=replace(s.model,sources=(NamedSource('source:guard','Guard source',source),)),
              initial_angle_rad=-2.,source_guard_m=.0001)
    assert not _swept_safe(s,-2.,0.)


@pytest.mark.parametrize('field,value',[
    ('inertia_kg_m2',0.),('inertia_kg_m2',-1.),('inertia_kg_m2',True),
    ('inertia_kg_m2',float('nan')),('initial_angle_rad',True),
    ('initial_angular_velocity_rad_s',float('inf')),('rotation_axis',(0,0,0)),
    ('duration_s',0),('sample_interval_s',2.),('source_guard_m',0.),
    ('relative_tolerance',0.1),('max_internal_steps',True),
])
def test_invalid_physical_or_numerical_inputs_refused(field,value):
    with pytest.raises((TypeError,ValueError)):
        replace(make_case('small-oscillation'),**{field:value})


def test_missing_inertia_pivot_and_free_body_not_assumed():
    with pytest.raises(TypeError):
        DipoleMotionSpec(model=make_case('small-oscillation').model)
    s=make_case('small-oscillation')
    with pytest.raises(UnsupportedProblem,match='pivot'):
        replace(s,model=replace(s.model,dipole=replace(s.model.dipole,pivot_m=None)))
    with pytest.raises(UnsupportedProblem,match='endpoints fixed'):
        replace(s,rotation_axis=(1,0,0))
    with pytest.raises(UnsupportedProblem,match='0.1c'):
        replace(s,initial_angular_velocity_rad_s=2e9)


@pytest.mark.parametrize('source',(
    UniformRingAxisField(1e-9,.1),UniformDiskAxisField(1e-9,.1)))
def test_axis_only_sources_stay_unsupported(source):
    s=make_case('small-oscillation')
    with pytest.raises(UnsupportedProblem,match='axis-only'):
        replace(s,model=replace(s.model,sources=(NamedSource('source:axis','Axis source',source),)))


def test_initial_singularity_and_step_exhaustion_are_explicit():
    s=make_case('source-guard')
    with pytest.raises(ValueError,match='guard'):
        replace(s,initial_angle_rad=0.)
    with pytest.raises(MotionIntegrationError,match='budget'):
        integrate_dipole_motion(replace(make_case('rotation'),max_internal_steps=1))


def test_fixed_axis_may_react_transverse_torque_not_a_free_body_projection():
    s=make_case('small-oscillation')
    src=UniformElectricField((120.,0.,70.))
    variant=replace(s,model=replace(s.model,sources=(NamedSource('e','E',src),)))
    r=integrate_dipole_motion(variant)
    assert abs(r.as_dict()['samples'][0]['observation']['torque_n_m'][1])>0
    base=integrate_dipole_motion(s)
    assert [x.angle_rad for x in r.samples]==pytest.approx([x.angle_rad for x in base.samples],abs=1e-12)


def test_reverse_axis_and_world_rotation_translation_covariance(runs):
    s=make_case('small-oscillation')
    reversed_axis=replace(s,rotation_axis=(0,0,-1),initial_angle_rad=-.1)
    a=integrate_dipole_motion(reversed_axis)
    assert [r.angle_rad for r in a.samples]==pytest.approx([-r.angle_rad for r in runs['small-oscillation'].samples],abs=1e-12)
    # Rotate world x,y,z to y,z,x and translate; no special z-axis physics.
    rot=lambda v:(v[2],v[0],v[1])
    t=(.2,-.3,.4)
    mov=lambda v:tuple(x+y for x,y in zip(rot(v),t))
    d=s.model.dipole
    model=replace(s.model,dipole=replace(d,center_m=mov(d.center_m),pivot_m=mov(d.pivot_m),direction=rot(d.direction)),
                  sources=(NamedSource('e','Rotated E',UniformElectricField(rot((120.,0.,0.)))),))
    b=integrate_dipole_motion(replace(s,model=model,rotation_axis=rot(s.rotation_axis)))
    assert [r.angle_rad for r in b.samples]==pytest.approx([r.angle_rad for r in runs['small-oscillation'].samples],abs=2e-10)


def test_tighter_error_policy_agrees_and_revision_preserves_parent(runs):
    s=make_case('point-pivot');old=s.as_dict()
    tighter=integrate_dipole_motion(replace(s,relative_tolerance=1e-11))
    assert tighter.samples[-1].angle_rad==pytest.approx(runs['point-pivot'].samples[-1].angle_rad,abs=2e-6)
    child,record=revise_dipole_motion(s,note='Teacher doubles I',inertia_kg_m2=2*s.inertia_kg_m2)
    assert child.model is s.model and child.inertia_kg_m2==2*s.inertia_kg_m2
    assert record['actual_changes'][0]['property']=='inertia_kg_m2'
    assert s.as_dict()==old
    assert integrate_dipole_motion(child).samples[-1].angle_rad!=tighter.samples[-1].angle_rad
    with pytest.raises(ValueError):
        revise_dipole_motion(s,note='Do not disguise numerical changes',source_guard_m=0)


@pytest.mark.parametrize('attribute',('angle_rad','angular_velocity_rad_s','angular_acceleration_rad_s2',
                                      'axis_torque_n_m','total_energy_j','kinetic_energy_j','energy_drift_j'))
def test_corrupt_saved_observations_rejected(runs,attribute):
    original=runs['libration'];rows=list(original.samples);v=getattr(rows[3],attribute)
    rows[3]=replace(rows[3],**{attribute:v+1})
    bad=replace(original,samples=tuple(rows))
    with pytest.raises(ValueError):
        validate_dipole_motion(bad)


def test_corrupt_time_and_guard_metadata_rejected(runs):
    r=runs['libration']
    with pytest.raises(ValueError):
        validate_dipole_motion(replace(r,samples=r.samples[:-1]))
    with pytest.raises(ValueError):
        validate_dipole_motion(replace(r,termination='source-guard'))
    with pytest.raises(ValueError):
        validate_dipole_motion(replace(r,internal_steps=True))


@pytest.mark.parametrize('case',CASES)
def test_self_contained_saved_time_viewer_not_static_angle_interpolation(tmp_path,runs,case):
    r=runs[case]
    p=write_dipole_motion(r,tmp_path/(case+'.html'))
    text=p.read_text(encoding="utf-8")
    assert 'id="play"' in text and 'id="time"' in text
    assert 'physical seconds / wall second' in text
    assert 'orthographic projection' in text and 'ideal frictionless hinge' in text.lower()
    assert 'No browser physics' in text
    assert 'fetch(' not in text and 'src="http' not in text
    audit=json.loads(p.with_suffix('.json').read_text(encoding="utf-8"))
    assert audit['spec']['model_digest']==r.spec.model.digest
    assert len(audit['samples'])==len(r.samples)
    if case=='source-guard':
        assert 'SOURCE GUARD' in text


def test_unwrapped_revolutions_do_not_loosen_angle_error_control():
    s=make_case('rotation')
    shifted=replace(s,initial_angle_rad=20*pi)
    a,b=integrate_dipole_motion(s),integrate_dipole_motion(shifted)
    for x,y in zip(a.samples,b.samples):
        assert y.angle_rad-20*pi==pytest.approx(x.angle_rad,abs=2e-8)
        assert y.angular_velocity_rad_s==pytest.approx(x.angular_velocity_rad_s,abs=2e-7)


@pytest.mark.parametrize('field',('time_s','angle_rad','angular_velocity_rad_s','kinetic_energy_j'))
def test_boolean_saved_physical_quantities_refused(runs,field):
    r=runs['small-oscillation'];rows=list(r.samples)
    rows[0]=replace(rows[0],**{field:False})
    with pytest.raises(ValueError):
        validate_dipole_motion(replace(r,samples=tuple(rows)))
