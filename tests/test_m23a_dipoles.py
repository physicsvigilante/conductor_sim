"""Preselected dipole laws, source fidelity, transfer and refusal checks."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from math import cos, hypot, pi, sin, sqrt
import json

import pytest

from conductor_sim.dipoles import (RigidDipole, DipoleModel, observe_dipole,
    orient_dipole, dipole_work, sample_orientations, uniform_rotational_equilibria,
    revise_dipole_model)
from conductor_sim.dipole_viewer import render_dipole_explorer
from conductor_sim.model import EPSILON_0, SolidSphere, Grounded
from conductor_sim.prescribed_fields import (FixedPointField, UniformElectricField,
    UniformRingAxisField, UniformDiskAxisField, UniformShellField, UniformRodField)
from conductor_sim.solver import SingularField, UnsupportedProblem
from conductor_sim.structural_scenarios import NamedSource
from examples.rigid_dipoles import make_model

K = 1/(4*pi*EPSILON_0)


def cross(a,b):
    return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])


def dot(a,b):
    return sum(x*y for x,y in zip(a,b))


def point_reference(dipole, source):
    pp,pm = dipole.endpoints_m
    out=[]
    for p,q in ((pp,dipole.charge_c),(pm,-dipole.charge_c)):
        delta=tuple(p[i]-source.position_m[i] for i in range(3));r=hypot(*delta)
        out.append(tuple(K*q*source.charge_c*x/r**3 for x in delta))
    force=tuple(out[0][i]+out[1][i] for i in range(3))
    arms=[tuple(p[i]-dipole.torque_origin_m[i] for i in range(3)) for p in (pp,pm)]
    ts=[cross(r,f) for r,f in zip(arms,out)]
    torque=tuple(ts[0][i]+ts[1][i] for i in range(3))
    energy=K*dipole.charge_c*source.charge_c*(
        1/hypot(*(pp[i]-source.position_m[i] for i in range(3)))-
        1/hypot(*(pm[i]-source.position_m[i] for i in range(3))))
    return force,torque,energy


@pytest.mark.parametrize('angle',[0,.13,pi/3,pi/2,2.2,pi,4.1,2*pi])
def test_uniform_exact_finite_pair_torque_energy(angle):
    m,_=make_model('workbook-q47'); m=orient_dipole(m,angle)
    o=observe_dipole(m);p=m.dipole.charge_c*m.dipole.separation_m
    assert o['net_force_n']==pytest.approx([0,0,0],abs=1e-22)
    assert o['torque_n_m']==pytest.approx([0,0,-p*120*sin(angle)],abs=1e-21)
    assert o['interaction_energy_j']==pytest.approx(-p*120*cos(angle),abs=1e-21)
    assert hypot(*o['dipole_moment_c_m'])==pytest.approx(p)


def test_workbook_only_qualitative_angles_and_positive_delta_u_order():
    m,_=make_model('workbook-q47')
    assert 'NO numerical angles' in m.provenance
    changes=[dipole_work(m,0,a*pi/180)['external_quasistatic_work_j'] for a in (40,90,115,180)]
    assert 0<changes[0]<changes[1]<changes[2]<changes[3]
    assert changes[-1]==pytest.approx(2*1e-9*.04*120)
    assert dipole_work(m,pi,0)['electric_work_j']==pytest.approx(changes[-1])
    assert uniform_rotational_equilibria(m)['orientations'][0]['direction']==[1,0,0]


@pytest.mark.parametrize('angle',[0,.31,pi/2,2.2,pi,5.3])
def test_nonuniform_point_full_endpoint_reference_and_pivot(angle):
    m,_=make_model('point-field'); pose=orient_dipole(m,angle)
    o=observe_dipole(pose);f,t,u=point_reference(pose.dipole,pose.sources[0].source)
    assert o['net_force_n']==pytest.approx(f,rel=2e-12,abs=1e-23)
    assert o['torque_n_m']==pytest.approx(t,rel=2e-12,abs=1e-23)
    assert o['interaction_energy_j']==pytest.approx(u,rel=2e-12,abs=1e-23)
    assert pose.dipole.pivot_m==m.dipole.pivot_m
    assert hypot(*(pose.dipole.center_m[i]-pose.dipole.pivot_m[i] for i in range(3)))==pytest.approx(
        hypot(*(m.dipole.center_m[i]-m.dipole.pivot_m[i] for i in range(3))))
    if angle: assert pose.dipole.center_m!=m.dipole.center_m
    centred=replace(pose,dipole=replace(pose.dipole,pivot_m=None))
    shift=tuple(pose.dipole.center_m[i]-pose.dipole.pivot_m[i] for i in range(3))
    oc=observe_dipole(centred);shift_t=cross(shift,f)
    assert t==pytest.approx([oc['torque_n_m'][i]+shift_t[i] for i in range(3)],abs=1e-23)


@pytest.mark.parametrize('case',['workbook-q47','point-field','mixed-sources'])
@pytest.mark.parametrize('angle',[.27,1.14,2.61])
def test_torque_is_negative_angular_energy_derivative(case,angle):
    m,_=make_model(case);h=1e-5
    a=observe_dipole(orient_dipole(m,angle-h));b=observe_dipole(orient_dipole(m,angle+h))
    torque=observe_dipole(orient_dipole(m,angle))['torque_n_m'][2]
    assert torque==pytest.approx(-(b['interaction_energy_j']-a['interaction_energy_j'])/(2*h),rel=2e-7,abs=1e-16)


def test_mixed_transfer_independent_rod_quadrature_sphere_uniform():
    m,_=make_model('mixed-sources');m=orient_dipole(m,.73);obs=observe_dipole(m)
    # Independent composite Simpson integration; no production rod formula.
    count=2048
    xs=[-1+2*i/count for i in range(count+1)]
    ws=[(1 if i in (0,count) else 4 if i%2 else 2)*2/(3*count) for i in range(count+1)]
    rod,sphere,uniform=[s.source for s in m.sources]
    ref_f=[];ref_u=0.
    for p,q in zip(m.dipole.endpoints_m,(m.dipole.charge_c,-m.dipole.charge_c)):
        v=0.;e=[0.,0.,0.]
        for x,w in zip(xs,ws):
            site=[rod.start_m[i]+(x+1)/2*(rod.end_m[i]-rod.start_m[i]) for i in range(3)]
            d=[p[i]-site[i] for i in range(3)];r=hypot(*d)
            v+=K*rod.charge_c*w/2/r
            for i in range(3):e[i]+=K*rod.charge_c*w/2*d[i]/r**3
        d=[p[i]-sphere.center_m[i] for i in range(3)];r=hypot(*d)
        assert r>sphere.radius_m
        v+=K*sphere.charge_c/r-dot(uniform.field_v_m,p)
        for i in range(3):e[i]+=K*sphere.charge_c*d[i]/r**3+uniform.field_v_m[i]
        ref_f.append([q*x for x in e]);ref_u+=q*v
    assert obs['interaction_energy_j']==pytest.approx(ref_u,rel=3e-12)
    assert obs['positive_force_n']==pytest.approx(ref_f[0],rel=3e-12)
    assert obs['negative_force_n']==pytest.approx(ref_f[1],rel=3e-12)
    parts=[]
    for s in m.sources:parts.append(observe_dipole(replace(m,sources=(s,))))
    assert obs['interaction_energy_j']==pytest.approx(sum(x['interaction_energy_j'] for x in parts))
    for key in ('net_force_n','torque_n_m'):
        assert obs[key]==pytest.approx([sum(x[key][i] for x in parts) for i in range(3)])


def test_uniform_gauge_cancellation_at_large_reference():
    m,_=make_model('workbook-q47');old=observe_dipole(m)
    source=replace(m.sources[0],source=UniformElectricField((120,0,0),reference_potential_v=1e25))
    new=observe_dipole(replace(m,sources=(source,)))
    for key in ('interaction_energy_j','net_force_n','torque_n_m'):assert new[key]==old[key]


def test_covariance_arbitrary_3d_rotation_and_translation():
    m,_=make_model('point-field')
    def R(p):return (p[2],p[0],p[1])
    shift=(.32,-.51,.8)
    def T(p):return tuple(x+y for x,y in zip(R(p),shift))
    d=m.dipole;s=m.sources[0]
    n=replace(m,dipole=replace(d,center_m=T(d.center_m),direction=R(d.direction),pivot_m=T(d.pivot_m)),
              sources=(replace(s,source=replace(s.source,position_m=T(s.source.position_m))),))
    a,b=observe_dipole(m),observe_dipole(n)
    assert b['interaction_energy_j']==pytest.approx(a['interaction_energy_j'],rel=1e-12)
    for key in ('net_force_n','torque_n_m'):assert b[key]==pytest.approx(R(a[key]),rel=1e-12,abs=1e-22)


def test_neutral_equilibria_and_nonuniform_refusal():
    m,_=make_model('workbook-q47')
    assert uniform_rotational_equilibria(replace(m,sources=()))['classification']=='neutral: all orientations'
    n,_=make_model('point-field')
    with pytest.raises(UnsupportedProblem,match='nonuniform'):uniform_rotational_equilibria(n)


def test_immutable_revision_preserves_id_and_requalifies_observations():
    m,_=make_model('workbook-q47');snapshot=m.as_dict();digest=m.digest
    n,a=revise_dipole_model(m,model_id='child',note='Field doubled',source_changes={'field:uniform':{'field_v_m':(240,0,0)}})
    assert m.as_dict()==snapshot and m.digest==digest
    assert n.sources[0].source_id==m.sources[0].source_id
    assert a['actual_changes']==[dict(target='field:uniform',property='field_v_m',before=[120.,0.,0.],after=[240.,0.,0.])]
    assert observe_dipole(n)['interaction_energy_j']==2*observe_dipole(m)['interaction_energy_j']
    with pytest.raises(FrozenInstanceError):m.dipole.charge_c=4
    audit=m.as_dict();audit['dipole']['center_m'][0]=9
    assert m.dipole.center_m[0]==0
    bad=replace(m,sources=(NamedSource('point','Point',FixedPointField(1e-9,(.02,0,0))),))
    with pytest.raises(SingularField):observe_dipole(bad)


@pytest.mark.parametrize('kwargs',[{'charge_c':0},{'charge_c':-1},{'charge_c':True},
 {'separation_m':0},{'separation_m':float('nan')},{'direction':(0,0,0)},
 {'center_m':(0,0)},{'pivot_m':(0,False,0)},{'source_id':'bad id'}])
def test_invalid_dipoles(kwargs):
    d=dict(charge_c=1e-9,separation_m=.04);d.update(kwargs)
    with pytest.raises((ValueError,TypeError)):RigidDipole(**d)


def test_resolution_source_identity_and_conductor_refusals():
    with pytest.raises(UnsupportedProblem):RigidDipole(1e-9,.001,center_m=(1e15,0,0))
    m,_=make_model('workbook-q47')
    with pytest.raises(ValueError):replace(m,sources=m.sources*2)
    with pytest.raises(ValueError):replace(m,sources=(NamedSource('dipole','Self',UniformElectricField((1,0,0))),))
    with pytest.raises(TypeError):NamedSource('metal','Conductor',SolidSphere(.1,Grounded()))
    with pytest.raises(ValueError):revise_dipole_model(m,model_id='bad',note='bad',dipole_changes={'moment':1})
    with pytest.raises(ValueError):revise_dipole_model(m,model_id='bad',note='bad',source_changes={'unknown':{}})


def test_axis_only_ring_accepted_at_axis_endpoints_but_rotation_refused():
    m=DipoleModel(RigidDipole(1e-9,.04,direction=(0,0,1)),
         (NamedSource('ring','Ring',UniformRingAxisField(1e-9,.1)),),'axial','teacher')
    assert observe_dipole(m)['net_force_n'][2]>0
    with pytest.raises(UnsupportedProblem,match='axis'):
        sample_orientations(m,(0.,.5),rotation_axis=(0,1,0))


def test_undefined_surface_field_preserves_potential_and_work():
    m=DipoleModel(RigidDipole(1e-9,.04,center_m=(.08,0,0)),
        (NamedSource('shell','Fixed shell',UniformShellField(1e-9,.1)),),'surface','teacher')
    obs=observe_dipole(m)
    assert obs['interaction_energy_j']==0
    assert obs['positive_force_n'] is None and obs['negative_force_n']==[0,0,0]
    assert obs['torque_n_m'] is None and obs['net_force_n'] is None
    assert obs['source_contributions'][0]['endpoint_field_unavailable'][0]
    assert dipole_work(m,0,pi/2)['external_quasistatic_work_j']==pytest.approx(0)


def test_singular_samples_are_gaps_not_zero():
    m=DipoleModel(RigidDipole(1e-9,.04),
        (NamedSource('point','Held point',FixedPointField(1e-9,(.02,0,0))),),'singular','teacher')
    rows=sample_orientations(m,(0.,.1,.2))['rows']
    assert rows[0]['observation'] is None and rows[0]['status']=='singular-potential'
    assert rows[1]['status']=='observed'
    with pytest.raises(SingularField):dipole_work(m,0,.2)


@pytest.mark.parametrize('case',['workbook-q47','point-field','mixed-sources','revised-field'])
def test_export_keeps_every_static_sample_and_escapes_text(case):
    m,_=make_model(case)
    html,record=render_dipole_explorer(m,(0.,.2,.5),title='<script>bad</script>')
    assert '&lt;script&gt;bad&lt;/script&gt;' in html
    assert 'static-orientation' in html and 'not time playback' in html
    assert 'World x–y projection' in html and 'Depth is not shown' in html
    assert len(record['rows'])==3 and record['model_digest']==m.digest
    assert 'fetch(' not in html and 'requestAnimationFrame' not in html
    assert 'polyline' not in html  # no path is invented through unsampled angles
    for row in record['rows']:
        assert row['observation']==observe_dipole(orient_dipole(m,row['angle_rad']))


@pytest.mark.parametrize('angles',[(),(0,),[0,0],[1,0],[0,True],[0,float('inf')]])
def test_invalid_angle_investigations(angles):
    m,_=make_model('workbook-q47')
    with pytest.raises(ValueError):sample_orientations(m,angles)


def test_axis_only_observation_does_not_authorize_a_projected_apparatus():
    m=DipoleModel(RigidDipole(1e-9,.04,direction=(0,0,1)),
         (NamedSource('ring','Ring',UniformRingAxisField(1e-9,.1)),),'axial','teacher')
    assert observe_dipole(m)['net_force_n'] is not None
    with pytest.raises(UnsupportedProblem,match='projected apparatus'):
        render_dipole_explorer(m,(0.,.2))


def test_revision_q_length_direction_and_pivot_retains_parent():
    m,_=make_model('point-field');before=m.as_dict()
    n,a=revise_dipole_model(m,model_id='new-pose',note='Change q and separation',
        dipole_changes={'charge_c':2e-9,'separation_m':.05,'direction':(0,1,0),'pivot_m':None})
    assert m.as_dict()==before
    assert n.dipole.moment_c_m==pytest.approx((0.,1e-10,0.),rel=1e-12,abs=1e-24)
    assert len(a['actual_changes'])==4
    assert observe_dipole(n)['interaction_energy_j']!=observe_dipole(m)['interaction_energy_j']


def test_surface_gap_visible_and_full_z_components_not_discarded():
    m=DipoleModel(RigidDipole(1e-9,.04,center_m=(.08,0,0)),
        (NamedSource('shell','Shell',UniformShellField(1e-9,.1)),),'surface','teacher')
    html,data=render_dipole_explorer(m,(0.,.2,.5))
    assert data['rows'][0]['observation']['net_force_n'] is None
    assert 'endpoint_field_unavailable' in html
    z=replace(m,sources=(NamedSource('E','Depth field',UniformElectricField((0,0,12))),))
    _,data=render_dipole_explorer(z,(0.,.2,.5))
    row=data['rows'][0]['observation']
    assert row['positive_force_n'][2]==pytest.approx(12e-9)
    assert row['torque_n_m'][1]!=0
