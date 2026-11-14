"""Independent fixed-source electrostatics and prescribed test-particle checks."""
from dataclasses import FrozenInstanceError, replace
import json
from math import cos, fsum, hypot, pi, sqrt

import pytest

from conductor_sim import (
    EPSILON_0, PointCharge, SolidSphere, Grounded, PrescribedField,
    UniformElectricField, FixedPointField, UniformRingAxisField,
    UniformDiskAxisField, UniformRodField, UniformSphereField, UniformShellField,
    TestChargeMotionSpec as MotionSpec, integrate_test_charge,
)
from conductor_sim.solver import SingularField, UnsupportedProblem

K = 1/(4*pi*EPSILON_0)


def spec(sources, p=(0.,0.,.2), v=(0.,0.,-.2), **kwargs):
    options = dict(field=PrescribedField(tuple(sources)),particle=PointCharge(1e-9,p),
                   mass_kg=1e-6,initial_velocity_m_s=v,duration_s=.2,sample_interval_s=.01,
                   length_scale_m=.1)
    return MotionSpec(**(options|kwargs))


def test_ring_values_center_far_field_and_both_sides():
    ring = UniformRingAxisField(2e-9,.12)
    for z in (0.,.04,-.04,1e5):
        assert ring.potential_at((0,0,z)) == pytest.approx(K*2e-9/sqrt(.12**2+z*z))
        assert ring.electric_field_at((0,0,z)) == pytest.approx((0,0,K*2e-9*z/(.12**2+z*z)**1.5))
    with pytest.raises(UnsupportedProblem,match='axis'):
        ring.potential_at((.01,0,.2))


def test_disc_values_potential_continuity_and_field_jump():
    disc = UniformDiskAxisField(-2e-9,.12)
    sigma = -2e-9/(pi*.12**2)
    for z in (.04,-.04):
        assert disc.potential_at((0,0,z)) == pytest.approx(2*pi*K*sigma*(sqrt(.12**2+z*z)-abs(z)))
        expected = 2*pi*K*sigma*(1-abs(z)/sqrt(.12**2+z*z))*(1 if z>0 else -1)
        assert disc.electric_field_at((0,0,z))[2] == pytest.approx(expected)
    assert disc.potential_at((0,0,0)) == pytest.approx(2*K*(-2e-9)/.12)
    with pytest.raises(SingularField,match='discontinuous'):
        disc.electric_field_at((0,0,0))
    assert disc.potential_at((0,0,1e7)) == pytest.approx(K*(-2e-9)/1e7,rel=1e-12)
    with pytest.raises(UnsupportedProblem):
        disc.electric_field_at((.02,0,.1))


@pytest.mark.parametrize('source,p',[
    (FixedPointField(1e-9,(.02,.01,-.03)),(.1,.04,.2)),
    (UniformRodField(-1e-9,(-.08,0,0),(.08,0,0)),(.02,.07,.11)),
    (UniformRodField(1e-9,(-.08,0,0),(.08,0,0)),(.2,.07,.11)),
    (UniformRodField(1e-9,(-.08,0,0),(.08,0,0)),(-.2,.07,.11)),
    (UniformSphereField(1e-9,.1),(.02,.03,.04)),
    (UniformSphereField(1e-9,.1),(.2,.03,.04)),
    (UniformShellField(1e-9,.1),(.02,.03,.04)),
    (UniformShellField(1e-9,.1),(.2,.03,.04)),
    (UniformElectricField((2.,-3.,4.),reference_potential_v=5.),(.2,.03,.04)),
])
def test_field_is_negative_gradient_of_potential(source,p):
    step=1e-6
    expected=[]
    for i in range(3):
        a,b=list(p),list(p);a[i]+=step;b[i]-=step
        expected.append(-(source.potential_at(a)-source.potential_at(b))/(2*step))
    assert source.electric_field_at(p) == pytest.approx(expected,rel=2e-8,abs=2e-8)


@pytest.mark.parametrize('source',[UniformRingAxisField(1e-9,.1),UniformDiskAxisField(1e-9,.1)])
def test_axial_field_negative_derivative(source):
    h=1e-6
    expected=-(source.potential_at((0,0,.2+h))-source.potential_at((0,0,.2-h)))/(2*h)
    assert source.electric_field_at((0,0,.2))[2] == pytest.approx(expected,rel=1e-9)


def test_rod_matches_independent_coulomb_quadrature_and_end_reversal():
    source=UniformRodField(3e-9,(-.08,0,0),(.08,0,0))
    for p in ((.02,.03,.09),(.2,0,0),(-.2,0,0)):
        n=6000
        positions=[(-.08+.16*(j+.5)/n,0.,0.) for j in range(n)]
        distances=[hypot(*(p[i]-q[i] for i in range(3))) for q in positions]
        potential=K*3e-9/n*fsum(1/r for r in distances)
        field=[K*3e-9/n*fsum((p[i]-q[i])/r**3 for q,r in zip(positions,distances)) for i in range(3)]
        assert source.potential_at(p)==pytest.approx(potential,rel=1e-8)
        assert source.electric_field_at(p)==pytest.approx(field,rel=3e-8)
        reversed_source=UniformRodField(3e-9,source.end_m,source.start_m)
        assert reversed_source.electric_field_at(p)==pytest.approx(source.electric_field_at(p),rel=1e-12)
    with pytest.raises(SingularField):
        source.potential_at((0,0,0))


def test_spherical_limits_boundary_and_permittivity():
    sphere,shell=UniformSphereField(1e-9,.1),UniformShellField(1e-9,.1)
    assert sphere.electric_field_at((0,0,0))==(0.,0.,0.)
    assert sphere.potential_at((0,0,0))==pytest.approx(1.5*K*1e-9/.1)
    assert shell.electric_field_at((0,0,0))==(0.,0.,0.)
    assert shell.potential_at((0,0,0))==pytest.approx(K*1e-9/.1)
    assert sphere.electric_field_at((.1,0,0))[0]==pytest.approx(K*1e-9/.1**2)
    assert shell.potential_at((.1,0,0))==pytest.approx(K*1e-9/.1)
    with pytest.raises(SingularField):
        shell.electric_field_at((.1,0,0))
    for source in (sphere,shell):
        assert source.potential_at((.2,0,0))==pytest.approx(K*1e-9/.2)
        assert source.electric_field_at((.2,0,0),permittivity_f_m=2*EPSILON_0)[0]==pytest.approx(.5*K*1e-9/.2**2)


def test_composition_includes_all_sources_and_probe_never_changes_them():
    field=PrescribedField((UniformRingAxisField(1e-9,.1),UniformRingAxisField(-.4e-9,.07,(0,0,-.1)),
                           UniformElectricField((0,0,4.),reference_potential_v=7.)))
    p=(0,0,.2)
    assert field.potential_at(p)==pytest.approx(sum(s.potential_at(p) for s in field.sources))
    assert field.electric_field_at(p)[2]==pytest.approx(sum(s.electric_field_at(p)[2] for s in field.sources))
    original=field.as_dict()
    a=spec(field.sources,field=field)
    for charge in (1e-9,-2e-9):
        result=integrate_test_charge(replace(a,particle=PointCharge(charge,p)))
        row=result.samples[0]
        assert row.force_n==pytest.approx(tuple(charge*x for x in field.electric_field_at(p)))
        assert row.potential_energy_j==pytest.approx(charge*field.potential_at(p),rel=1e-13)
        assert row.potential_energy_j!=pytest.approx(.5*charge*field.potential_at(p),rel=1e-3,abs=1e-20)
        assert field.as_dict()==original
    with pytest.raises(FrozenInstanceError):
        field.sources=()


def test_uniform_field_exact_parabola_and_no_fabricated_plates():
    model=spec((UniformElectricField((0,0,-100)),),p=(0,0,0),v=(.2,0,.03),duration_s=.4)
    result=integrate_test_charge(model)
    assert result.termination=='duration'
    for row in result.samples:
        t=row.time_s
        assert row.position_m==pytest.approx((.2*t,0,.03*t-.05*t*t),abs=1e-13)
        assert row.velocity_m_s==pytest.approx((.2,0,.03-.1*t),abs=1e-13)
    assert result.max_abs_energy_drift_j<1e-20
    data=json.loads(json.dumps(result.as_dict(),allow_nan=False))
    assert data['schema']=='conductor_sim.test_charge_motion.v1'
    assert 'conductors' not in data and 'motion_2d' not in data


def test_ring_turning_point_from_independent_energy_equation():
    model=spec((UniformRingAxisField(1e-9,.1),),duration_s=1.4,sample_interval_s=.002)
    result=integrate_test_charge(model)
    energy=.5*model.mass_kg*.2**2+K*1e-18/sqrt(.1**2+.2**2)
    turning=sqrt((K*1e-18/energy)**2-.1**2)
    minimum=min(s.position_m[2] for s in result.samples)
    assert minimum==pytest.approx(turning,abs=2e-6)
    assert result.samples[-1].velocity_m_s[2]>0
    assert result.max_abs_energy_drift_j<1e-13


def test_uniform_sphere_internal_shm_exact_solution_and_centre_passage():
    model=spec((UniformSphereField(1e-9,.1),),p=(.02,0,0),v=(0,0,0),
               particle=PointCharge(-1e-9,(.02,0,0)),duration_s=1.,sample_interval_s=.01)
    result=integrate_test_charge(model)
    omega=sqrt(K*1e-18/model.mass_kg/.1**3)
    for row in result.samples:
        assert row.position_m[0]==pytest.approx(.02*cos(omega*row.time_s),abs=1e-9)
    assert result.samples[-1].position_m[0]<0
    assert result.max_abs_energy_drift_j<1e-14


def test_volume_boundary_is_continuous_and_can_be_crossed():
    model=spec((UniformSphereField(1e-9,.1),),p=(-.15,0,0),v=(.3,0,0),
               particle=PointCharge(-1e-9,(-.15,0,0)),duration_s=1.4)
    result=integrate_test_charge(model)
    assert result.termination=='duration' and result.samples[-1].position_m[0]>.1
    assert any(abs(s.position_m[0])<.1 for s in result.samples)
    assert result.max_abs_energy_drift_j<2e-12


@pytest.mark.parametrize('source,p,v',[
    (UniformShellField(1e-9,.1),(0,0,0),(.5,0,0)),
    (UniformDiskAxisField(1e-9,.1),(0,0,.02),(0,0,-.5)),
    (FixedPointField(1e-15),(0,0,.02),(0,0,-.5)),
    (UniformRodField(1e-15,(-.1,0,0),(.1,0,0)),(0,0,.02),(0,0,-.5)),
])
def test_source_guards_stop_before_singularities_or_jumps(source,p,v):
    model=spec((source,),p=p,v=v,particle=PointCharge(-1e-9,p),duration_s=.5,source_guard_m=1e-4)
    result=integrate_test_charge(model)
    assert result.termination=='source-guard'
    assert result.events[-1].time_s==result.samples[-1].time_s
    assert result.events[-1].time_s<model.duration_s
    assert all(source.clearance_at(s.position_m)>model.source_guard_m for s in result.samples)
    assert 'collision' in result.events[-1].description


@pytest.mark.parametrize('sources,p,v',[
    ((UniformRingAxisField(1e-9,.1),),(0,0,.2),(.1,0,0)),
    ((UniformRingAxisField(1e-9,.1),UniformElectricField((1,0,0))),(0,0,.2),(0,0,0)),
    ((UniformRingAxisField(1e-9,.1),FixedPointField(1e-9,(.1,0,0))),(0,0,.2),(0,0,0)),
    ((UniformElectricField((0,1,0)),),(0,0,.2),(0,0,0)),
    ((FixedPointField(1e-9,(0,.1,0)),),(0,0,.2),(0,0,0)),
    ((UniformRingAxisField(1e-9,.1),UniformRingAxisField(1e-9,.1,axis=(1,0,0))),(0,0,0),(0,0,0)),
])
def test_refuses_unproved_plane_or_axis_instead_of_projecting(sources,p,v):
    with pytest.raises(UnsupportedProblem):
        spec(sources,p,v)


def test_rotated_translated_ring_matches_axial_motion():
    axis=tuple(x/sqrt(14) for x in (1,2,3));centre=(2.,-1.,.4)
    first=integrate_test_charge(spec((UniformRingAxisField(1e-9,.1),)))
    start=tuple(centre[i]+.2*axis[i] for i in range(3))
    other=integrate_test_charge(spec((UniformRingAxisField(1e-9,.1,centre,axis),),p=start,v=tuple(-.2*x for x in axis)))
    for a,b in zip(first.samples,other.samples):
        assert b.position_m==pytest.approx(tuple(centre[i]+a.position_m[2]*axis[i] for i in range(3)),abs=1e-10)


def test_gauge_shift_changes_qv_but_not_motion():
    first=spec((UniformElectricField((0,0,-100)),),v=(.2,0,.03))
    second=replace(first,field=PrescribedField((UniformElectricField((0,0,-100),reference_potential_v=17),)))
    a,b=integrate_test_charge(first),integrate_test_charge(second)
    for x,y in zip(a.samples,b.samples):
        assert x.position_m==y.position_m and x.velocity_m_s==y.velocity_m_s
        assert y.potential_energy_j-x.potential_energy_j==pytest.approx(first.particle.charge_c*17,abs=1e-22)


@pytest.mark.parametrize('override',[
    dict(mass_kg=True),dict(source_guard_m=0),dict(length_scale_m=float('nan')),
    dict(relative_tolerance=.1),dict(max_internal_steps=True),
    dict(initial_velocity_m_s=(False,0,0)),dict(sample_interval_s=.3),
    dict(plane_normal=(0,0,0)),dict(duration_s=1000),
])
def test_invalid_spec_rejected(override):
    with pytest.raises((ValueError,TypeError)):
        spec((),v=(0,0,0),**override)


def test_speed_limit_and_no_conductor_source_substitution():
    with pytest.raises(UnsupportedProblem,match='0.1c'):
        spec((),v=(3e7,0,0))
    with pytest.raises(UnsupportedProblem,match='conductor'):
        PrescribedField((SolidSphere(.1,Grounded()),))
    with pytest.raises(UnsupportedProblem,match='0.1c'):
        integrate_test_charge(spec((UniformElectricField((1e20,0,0)),),v=(0,0,0)))


def test_shared_controller_used_without_a_conductor_solve(monkeypatch):
    import conductor_sim.prescribed_motion as module
    import conductor_sim.motion_2d as old
    assert module._integrate_vector_motion is old._integrate_vector_motion
    def forbidden(*args,**kwargs):
        raise AssertionError('conductor force must not be used by prescribed-field dynamics')
    monkeypatch.setattr(old,'force_energy',forbidden)
    result=integrate_test_charge(spec((),v=(.2,0,0),duration_s=.07))
    assert len(result.samples)==8
    assert result.samples[-1].time_s==.07
    assert all(a.time_s<b.time_s for a,b in zip(result.samples,result.samples[1:]))


@pytest.mark.parametrize('factory',[
    lambda: UniformRingAxisField(True,.1),
    lambda: UniformDiskAxisField(1e-9,-.1),
    lambda: UniformSphereField(1e-9,0),
    lambda: UniformShellField(0,.1),
    lambda: FixedPointField(float('inf')),
    lambda: UniformRodField(1e-9,(0,0,0),(0,0,0)),
    lambda: UniformElectricField((1,False,0)),
    lambda: UniformRingAxisField(1e-9,.1,axis=(0,0,0)),
    lambda: PrescribedField((),permittivity_f_m=0),
    lambda: PrescribedField((UniformElectricField((0,0,0)),)*65),
])
def test_invalid_source_inputs_fail_closed(factory):
    with pytest.raises((ValueError,TypeError)):
        factory()


def test_rod_far_field_and_reflection_symmetric_normal_rod():
    rod=UniformRodField(1e-9,(-.08,0,0),(.08,0,0))
    p=(1e6,1e6,1e6);r=hypot(*p)
    assert rod.potential_at(p)==pytest.approx(K*1e-9/r,rel=1e-12)
    assert rod.electric_field_at(p)==pytest.approx(tuple(K*1e-9*x/r**3 for x in p),rel=1e-12,abs=1e-30)
    normal_rod=UniformRodField(1e-9,(0,-.08,0),(0,.08,0))
    result=integrate_test_charge(spec((normal_rod,),v=(.1,0,0)))
    assert all(row.position_m[1]==0 for row in result.samples)


def test_axis_source_can_compose_with_bisected_perpendicular_rod():
    ring=UniformRingAxisField(1e-9,.1)
    rod=UniformRodField(1e-9,(-.08,0,0),(.08,0,0))
    result=integrate_test_charge(spec((ring,rod),v=(0,0,.1)))
    assert result.termination=='duration'
    assert all(s.velocity_m_s[:2]==(0.,0.) for s in result.samples)


def test_source_chord_guards_cannot_jump_over_an_excluded_region():
    from conductor_sim.prescribed_motion import _FixedDomain
    for source in (FixedPointField(1e-9),UniformShellField(1e-9,.05),
                   UniformRodField(1e-9,(-.1,0,0),(.1,0,0)),UniformDiskAxisField(1e-9,.1)):
        model=spec((source,),p=(0,0,.2),v=(0,0,0))
        domain=_FixedDomain(model)
        assert domain.clearance((0,0,.2))>0 and domain.clearance((0,0,-.2))>0
        assert domain.charge_segment_clearance((0,0,.2),(0,0,-.2))<0


def test_rotated_rod_on_source_roundoff_does_not_create_finite_field():
    rod=UniformRodField(1e-9,(.1,.2,.3),(.4,.8,.7))
    midpoint=tuple((a+b)/2 for a,b in zip(rod.start_m,rod.end_m))
    with pytest.raises(SingularField):
        rod.electric_field_at(midpoint)


def test_rotated_shell_surface_refuses_ambiguous_side():
    shell=UniformShellField(1e-9,.1,(.2,-.1,.3))
    p=tuple(c+.1/sqrt(3) for c in shell.center_m)
    assert shell.potential_at(p)==pytest.approx(K*1e-9/.1)
    with pytest.raises(SingularField):
        shell.electric_field_at(p)


@pytest.mark.parametrize('case',['ring-turning','ring-crossing','two-rings','uniform-deflection',
                                 'sphere-transit','shell-interior','disc-approach','rod-deflection'])
def test_public_cli_json_is_finite_and_has_no_conductor_inventory(case,tmp_path):
    import subprocess
    import sys
    path=tmp_path/(case+'.json')
    process=subprocess.run([sys.executable,'examples/test_charge_motion.py','--case',case,
                            '--output',str(path)],capture_output=True,text=True,timeout=30)
    assert process.returncode==0,process.stdout+process.stderr
    data=json.loads(path.read_text())
    assert data['example']==case and data['schema']=='conductor_sim.test_charge_motion.v1'
    assert 'conductors' not in data and 'frames' not in data
    assert data['samples'][0]['time_s']==0
    assert data['samples'][-1]['time_s']<=data['duration_s']
    json.dumps(data,allow_nan=False)
