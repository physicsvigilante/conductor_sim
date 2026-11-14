"""M21e: no problem-specific force or trajectory code enters the engine."""
from copy import deepcopy
from dataclasses import asdict, FrozenInstanceError, replace
from math import atan2, degrees, fsum, hypot, pi, sqrt
import json

import pytest

from conductor_sim import (EPSILON_0, FixedPointField, PointCharge, PrescribedField,
    TestChargeMotionSpec as MotionSpec, UniformElectricField, UniformRingAxisField,
    integrate_test_charge)
from conductor_sim.conservative_forces import (UniformGravity, PrescribedElectricForce,
    ConservativeForceSystem, COMPOSED_ENERGY, CONTRIBUTION_KEYS, restore_gravity)
from conductor_sim.prescribed_viewer import sample_test_charge_viewer
from conductor_sim.prescribed_viewer_validation import validate_prescribed_viewer
from conductor_sim.scenarios import PrescribedScenario, revise_scenario, sample_scenario_viewer
from conductor_sim.scenario_comparison import prepare_scenario_comparison, render_scenario_comparison
from conductor_sim.solver import UnsupportedProblem
from conductor_sim.viewer import render_html
from examples.electric_gravity_motion import build_case, make_bundle, export_case, CASES, MP, E_CHARGE, G


def uniform(*, E=(3.,0.,0.), g=(0.,-10.,0.), q=1e-6, m=2e-6, v=(.2,.4,0.)):
    return MotionSpec(PrescribedField((UniformElectricField(E),)),PointCharge(q),m,v,
        .12,.01,plane_normal=(0.,0.,1.),length_scale_m=.2,
        gravity=UniformGravity(g) if g is not None else None)


@pytest.fixture(scope='module')
def proton():
    return make_bundle('jee2012-proton')


def test_jee_source_voltage_acceleration_and_preplate_domain(proton):
    scene,bundle,anchor=proton
    assert anchor['pdf_page']==15 and anchor['plate_gap_m']==.01
    assert anchor['voltage_v']==pytest.approx(MP*G*.01/E_CHARGE,rel=1e-14)
    assert anchor['voltage_v']==pytest.approx(1.04375e-9,rel=1e-14)
    rows=bundle['prescribed_motion']['samples']
    assert rows[0]['velocity_m_s']==[0.,0.,0.]
    ax,ay,az=rows[0]['acceleration_m_s2']
    assert degrees(atan2(abs(ax),abs(ay)))==pytest.approx(45.,abs=1e-12)
    assert az==0.
    for r in rows:
        t=r['time_s']
        assert r['position_m']==pytest.approx([.5*G*t*t,-.5*G*t*t,0.],abs=1e-15)
        assert r['velocity_m_s']==pytest.approx([G*t,-G*t,0.],abs=1e-14)
        assert abs(r['position_m'][0])<.005
        assert r['gravitational_force_n']==pytest.approx([0.,-MP*G,0.],rel=1e-14,abs=1e-50)
        assert r['electric_force_n']==pytest.approx([MP*G,0.,0.],rel=1e-14,abs=1e-50)
        expected_K=MP*G**2*t*t
        assert r['kinetic_energy_j']==pytest.approx(expected_K,rel=1e-13,abs=1e-48)
        assert abs(r['total_energy_j'])<1e-42
    plane=bundle['frames'][0]['plane']
    assert plane['u']==[1.,0.,0.] and plane['v']==[0.,1.,0.]
    assert bundle['prescribed_motion']['energy_convention']==COMPOSED_ENERGY


@pytest.mark.parametrize('case',CASES[:-1])
def test_different_arrangements_use_one_motion_and_view_pipeline(case):
    scenario,bundle,anchor=make_bundle(case)
    assert type(scenario.spec) is MotionSpec
    assert type(scenario.spec.gravity) is UniformGravity
    assert bundle['prescribed_motion']['schema']=='conductor_sim.test_charge_viewer.v2'
    assert validate_prescribed_viewer(bundle) is True
    assert bundle['prescribed_motion']['termination']=='duration'
    assert len(bundle['prescribed_motion']['samples'])>=20
    assert bundle['frames'][0]['geometry']==[]  # No pretend plate-conductor solve.
    assert all(k in bundle['prescribed_motion']['samples'][0] for k in CONTRIBUTION_KEYS)


@pytest.mark.parametrize('E,g,v',[
    ((4.,2.,0.),(0.,-10.,0.),(.1,.5,0.)),
    ((0.,0.,0.),(0.,-10.,0.),(1.,1.,0.)),
    ((0.,20.,0.),(0.,-10.,0.),(0.,0.,0.)),
    ((-2.,5.,0.),(3.,-4.,0.),(-.4,.3,0.)),
    ((3.,0.,0.),(0.,0.,0.),(.2,.4,0.)),
])
def test_closed_form_constant_acceleration_and_separate_energies(E,g,v):
    spec=uniform(E=E,g=g,v=v)
    result=integrate_test_charge(spec)
    acceleration=tuple(spec.particle.charge_c*E[i]/spec.mass_kg+g[i] for i in range(3))
    for r in result.samples:
        t=r.time_s
        expected=tuple(v[i]*t+.5*acceleration[i]*t*t for i in range(3))
        assert r.position_m==pytest.approx(expected,abs=2e-14)
        assert r.velocity_m_s==pytest.approx([v[i]+acceleration[i]*t for i in range(3)],abs=2e-13)
        assert r.electric_potential_energy_j==pytest.approx(-spec.particle.charge_c*sum(E[i]*r.position_m[i] for i in range(3)),abs=1e-24)
        assert r.gravitational_potential_energy_j==pytest.approx(-spec.mass_kg*sum(g[i]*r.position_m[i] for i in range(3)),abs=1e-24)
        assert r.potential_energy_j==fsum((r.electric_potential_energy_j,r.gravitational_potential_energy_j))
    assert result.max_abs_energy_drift_j<2e-20


def test_support_equilibrium_not_a_hidden_mechanical_constraint():
    spec=build_case('electric-support')[0].spec
    result=integrate_test_charge(spec)
    assert all(r.position_m==(0.,0.,0.) and r.velocity_m_s==(0.,0.,0.) for r in result.samples)
    assert all(r.force_n==(0.,0.,0.) for r in result.samples)
    assert result.samples[0].electric_force_n[1]>0
    assert result.samples[0].gravitational_force_n[1]<0
    changed=integrate_test_charge(replace(spec,mass_kg=2*spec.mass_kg))
    assert changed.samples[-1].position_m[1]<0


def test_mass_revision_preserves_gravitational_acceleration_not_electric_acceleration():
    scene=build_case('oblique-launch')[0];before=scene.as_dict()
    child=revise_scenario(scene,scenario_id='mass-double',note='Double mass only',motion_changes={'mass_kg':2*scene.spec.mass_kg})
    a,b=(integrate_test_charge(s.spec).samples[-1] for s in (scene,child))
    assert a.electric_force_n==b.electric_force_n
    assert b.gravitational_force_n==tuple(2*x for x in a.gravitational_force_n)
    assert a.acceleration_m_s2[1]==b.acceleration_m_s2[1]==-G
    assert b.acceleration_m_s2[0]==a.acceleration_m_s2[0]/2
    assert scene.as_dict()==before and child.parent_revision_id==scene.revision_id


def test_charge_reversal_changes_only_electric_force():
    s=uniform();a=integrate_test_charge(s).samples[-1]
    b=integrate_test_charge(replace(s,particle=replace(s.particle,charge_c=-s.particle.charge_c))).samples[-1]
    assert b.electric_force_n==tuple(-x for x in a.electric_force_n)
    assert a.gravitational_force_n==b.gravitational_force_n
    assert a.acceleration_m_s2[1]==b.acceleration_m_s2[1]


def test_gravity_reference_is_gauge_not_physical_launch_change():
    s=uniform();shift=replace(s,gravity=replace(s.gravity,reference_position_m=(3.,7.,0.)))
    a,b=integrate_test_charge(s),integrate_test_charge(shift)
    for x,y in zip(a.samples,b.samples):
        assert x.position_m==y.position_m and x.velocity_m_s==y.velocity_m_s
        assert x.electric_field_v_m==y.electric_field_v_m and x.force_n==y.force_n
        assert y.gravitational_potential_energy_j-x.gravitational_potential_energy_j==pytest.approx(-70*s.mass_kg,rel=1e-14)
        assert x.electric_potential_energy_j==y.electric_potential_energy_j
    sample_test_charge_viewer(shift,result=b,visual_frames=3,resolution=9)


def test_voltage_gauge_keeps_gravity_and_motion_independent():
    s=uniform();field=PrescribedField((replace(s.field.sources[0],reference_potential_v=12.),))
    b=integrate_test_charge(replace(s,field=field));a=integrate_test_charge(s)
    for x,y in zip(a.samples,b.samples):
        assert x.position_m==y.position_m and x.velocity_m_s==y.velocity_m_s
        assert x.gravitational_potential_energy_j==y.gravitational_potential_energy_j
        assert y.electric_potential_energy_j-x.electric_potential_energy_j==pytest.approx(12*s.particle.charge_c,rel=1e-14)


def test_nonuniform_point_plus_g_matches_independent_verlet_reference():
    s=build_case('point-plus-gravity')[0].spec
    actual=integrate_test_charge(s).samples[-1]
    # Independent Coulomb formula plus constant gravity, separate integrator.
    k=1/(4*pi*EPSILON_0);source=s.field.sources[0];q=s.particle.charge_c
    def acc(p):
        d=[p[i]-source.position_m[i] for i in range(3)];r=hypot(*d)
        return [k*q*source.charge_c*d[i]/(s.mass_kg*r**3)+s.gravity.acceleration_m_s2[i] for i in range(3)]
    p=list(s.particle.position_m);v=list(s.initial_velocity_m_s);dt=s.duration_s/2000
    for _ in range(2000):
        a=acc(p);p=[p[i]+v[i]*dt+.5*a[i]*dt*dt for i in range(3)]
        b=acc(p);v=[v[i]+.5*(a[i]+b[i])*dt for i in range(3)]
    assert actual.position_m==pytest.approx(p,abs=2e-9)
    assert actual.velocity_m_s==pytest.approx(v,abs=2e-9)
    for i in range(3):
        plus=list(p);minus=list(p);h=1e-6;plus[i]+=h;minus[i]-=h
        def U(r):
            return k*q*source.charge_c/hypot(*(r[j]-source.position_m[j] for j in range(3)))-s.mass_kg*sum(s.gravity.acceleration_m_s2[j]*r[j] for j in range(3))
        gradient=-(U(plus)-U(minus))/(2*h)
        assert actual.force_n[i]==pytest.approx(gradient,rel=2e-7,abs=1e-16)


def test_ring_axis_compatibility_and_force_decomposition():
    s=build_case('ring-plus-gravity')[0].spec
    for row in integrate_test_charge(s).samples:
        y=row.position_m[1];source=s.field.sources[0]
        expected=(1/(4*pi*EPSILON_0))*s.particle.charge_c*source.charge_c*y/(y*y+source.radius_m**2)**1.5
        assert row.electric_force_n[1]==pytest.approx(expected,rel=1e-13)
        assert row.force_n[1]==pytest.approx(expected-.1*s.mass_kg,rel=1e-13)
        assert row.position_m[0]==row.position_m[2]==0.


@pytest.mark.parametrize('g',[(1.,-.1,0.),(0.,-.1,1.)])
def test_offaxis_gravity_refused_for_axial_ring(g):
    s=build_case('ring-plus-gravity')[0].spec
    with pytest.raises(UnsupportedProblem,match='gravity'):
        replace(s,gravity=UniformGravity(g))


def test_out_of_plane_terms_refused_even_if_resultant_cancels_at_all_points():
    # Deliberately conservative policy: no cross-domain symmetry proof is claimed.
    with pytest.raises(UnsupportedProblem,match='gravity'):
        uniform(E=(0.,0.,20.),g=(0.,0.,-10.))
    with pytest.raises(UnsupportedProblem):
        uniform(E=(0.,0.,1.),g=(0.,-10.,0.))


def test_rotation_translation_covariance():
    s=uniform()
    def rot(v):return (v[2],v[0],v[1])
    offset=(.5,-.2,.3)
    moved=replace(s,field=PrescribedField((UniformElectricField(rot((3.,0.,0.)),reference_point_m=offset),)),
        particle=PointCharge(s.particle.charge_c,offset), initial_velocity_m_s=rot(s.initial_velocity_m_s),
        plane_normal=rot(s.plane_normal),gravity=UniformGravity(rot(s.gravity.acceleration_m_s2),offset))
    a,b=integrate_test_charge(s),integrate_test_charge(moved)
    for x,y in zip(a.samples,b.samples):
        assert y.position_m==pytest.approx([rot(x.position_m)[i]+offset[i] for i in range(3)],abs=3e-14)
        assert y.force_n==pytest.approx(rot(x.force_n),rel=1e-14,abs=1e-25)
        assert y.potential_energy_j==pytest.approx(x.potential_energy_j,rel=1e-12,abs=2e-20)


def test_disabled_gravity_reproduces_legacy_physics_and_v1_serialization():
    s=uniform(g=None);zero=replace(s,gravity=UniformGravity((0.,0.,0.)))
    a,b=integrate_test_charge(s),integrate_test_charge(zero)
    for x,y in zip(a.samples,b.samples):
        assert x.position_m==y.position_m and x.velocity_m_s==y.velocity_m_s
        assert x.force_n==y.force_n and x.potential_energy_j==y.potential_energy_j
    assert 'gravity' not in a.as_dict()
    assert not (set(CONTRIBUTION_KEYS)&set(a.as_dict()['samples'][0]))
    assert a.as_dict()['schema'].endswith('.v1')
    assert b.as_dict()['schema'].endswith('.v2')
    bundle=sample_test_charge_viewer(s,result=a,visual_frames=3,resolution=9)
    assert 'gravity-presentation' not in render_html(bundle)


def test_reference_revision_and_mass_comparison_ledger_include_gravity():
    s=build_case('oblique-launch')[0]
    t=revise_scenario(s,scenario_id='half-g',note='Half gravity and doubled mass',
        gravity_changes={'acceleration_m_s2':(0.,-5.,0.)},motion_changes={'mass_kg':4e-6})
    assert t.required_action=='reassemble-forces-and-reintegrate'
    assert t.spec.gravity is not s.spec.gravity
    assert {c.component_id for c in t.changes}=={'test-particle','gravity'}
    kw=dict(visual_frames=3,resolution=9,extent_m=.5)
    a,b=sample_scenario_viewer(s,**kw),sample_scenario_viewer(t,**kw)
    manifest=prepare_scenario_comparison(a,b)
    assert manifest['potential_energy_label']=='U_e + U_g'
    assert manifest['revisions']==json.loads(json.dumps([asdict(c) for c in t.changes]))
    html=render_scenario_comparison(a,b)
    assert 'gravity_inventories' in html and 'U_e + U_g' in html


def test_broken_gravity_revision_rejected_without_mutating_baseline():
    s=build_case('ring-plus-gravity')[0];before=s.as_dict()
    with pytest.raises(UnsupportedProblem,match='gravity'):
        revise_scenario(s,scenario_id='bad-g',note='Unsupported transverse gravity',gravity_changes={'acceleration_m_s2':(1.,0.,0.)})
    assert s.as_dict()==before
    with pytest.raises(ValueError,match='unsupported property'):
        revise_scenario(s,scenario_id='bad-prop',note='Invalid force property',gravity_changes={'drag':1.})


def test_component_membership_and_identity_are_not_silently_changed():
    s=PrescribedScenario('legacy-g-id',uniform(g=None),(('gravity','An electric source'),),'Legacy electric source ID')
    assert s.as_dict()['spec'].get('gravity') is None
    with pytest.raises(ValueError,match='authored gravity component'):
        revise_scenario(s,scenario_id='add-g',note='Not a parameter revision',gravity_changes={'acceleration_m_s2':(0,-10,0)})
    with pytest.raises(ValueError,match='reserved'):
        replace(s,spec=replace(s.spec,gravity=UniformGravity((0,-10,0))))


@pytest.mark.parametrize('key,delta',[
    ('electric_force_n',1e-30),('gravitational_force_n',1e-30),('force_n',1e-30),
    ('acceleration_m_s2',.001),('electric_potential_energy_j',1e-33),
    ('gravitational_potential_energy_j',1e-33),('potential_energy_j',1e-33),
    ('total_energy_j',1e-33),('energy_drift_j',1e-33),('interaction_energy_j',1e-33),
])
def test_proton_scale_tampering_is_not_hidden_by_macroscopic_tolerance(proton,key,delta):
    bad=deepcopy(proton[1]);r=bad['prescribed_motion']['samples'][6]
    if isinstance(r[key],list):r[key][0]+=delta
    else:r[key]+=delta
    with pytest.raises(ValueError):render_html(bad)


@pytest.mark.parametrize('mutation',[
    lambda m:m.pop('gravity'),
    lambda m:m['gravity'].update(extra=1),
    lambda m:m['gravity'].update(acceleration_m_s2=[0.,-11.,0.]),
    lambda m:m.update(schema='conductor_sim.test_charge_viewer.v1'),
    lambda m:m.update(max_abs_energy_drift_j=1e-32),
    lambda m:m['samples'][2].pop('electric_force_n'),
    lambda m:m['samples'][2].update(gravitational_force_n=[0.,0.]),
])
def test_missing_misdeclared_or_inconsistent_composition_rejected(proton,mutation):
    bad=deepcopy(proton[1]);mutation(bad['prescribed_motion'])
    with pytest.raises((ValueError,TypeError)):render_html(bad)


def test_equal_opposite_force_tampering_does_not_pass_a_zero_resultant_check():
    scene,bundle,_=make_bundle('electric-support')
    bad=deepcopy(bundle);r=bad['prescribed_motion']['samples'][2]
    r['electric_force_n'][1]*=2;r['gravitational_force_n'][1]*=2
    assert r['force_n']==[0.,0.,0.]
    with pytest.raises(ValueError,match='component'):render_html(bad)


def test_original_result_contribution_tampering_rejected_before_sampling(proton):
    s=proton[0].spec;r=integrate_test_charge(s)
    rows=list(r.samples);rows[3]=replace(rows[3],gravitational_potential_energy_j=1e-33)
    with pytest.raises(ValueError):sample_test_charge_viewer(s,result=replace(r,samples=tuple(rows)))


@pytest.mark.parametrize('g',[(float('nan'),0,0),(0,float('inf'),0),(0,False,0),(1,2)])
def test_invalid_gravity_components(g):
    with pytest.raises((ValueError,TypeError)):UniformGravity(g)


def test_immutable_typed_force_components_no_plugins_or_equivalent_E():
    g=UniformGravity([0.,-10.,0.]);assert g.acceleration_m_s2==(0.,-10.,0.)
    with pytest.raises(FrozenInstanceError):g.acceleration_m_s2=(0.,0.,0.)
    with pytest.raises(TypeError):ConservativeForceSystem(PrescribedElectricForce(uniform().field),lambda r:r)
    with pytest.raises(TypeError):replace(uniform(),gravity={'g':10})
    assert restore_gravity(json.loads(json.dumps(g.as_dict())))==g
    system=ConservativeForceSystem(PrescribedElectricForce(uniform().field),g)
    a=system.observe((0.,0.,0.),charge_c=0.,mass_kg=1.)
    assert a.electric_force_n==(0.,0.,0.) and a.gravitational_force_n==(0.,-10.,0.)


def test_source_guard_is_still_real_and_not_a_gravity_collision_solver():
    s=MotionSpec(PrescribedField((FixedPointField(-1e-9),)),PointCharge(1e-9,(.1,0.,0.)),
        1.,(-1.,0.,0.),.2,.02,plane_normal=(0.,0.,1.),length_scale_m=.1,
        source_guard_m=.01,gravity=UniformGravity((0.,0.,0.)))
    r=integrate_test_charge(s)
    assert r.termination=='source-guard' and r.samples[-1].time_s<s.duration_s
    assert sample_test_charge_viewer(s,result=r,visual_frames=3,resolution=9)['prescribed_motion']['events'][0]['kind']=='source-guard'


@pytest.mark.parametrize('case',CASES)
def test_reproducible_offline_export_and_detached_audit(tmp_path,case):
    out=tmp_path/(case+'.html');audit=export_case(case,out)
    assert out.is_file() and out.with_suffix('.json').is_file()
    text=out.read_text(encoding='utf-8')
    assert 'gravity-presentation' in text and 'fetch(' not in text
    assert 'F_total = qE + mg' in text
    assert audit==json.loads(out.with_suffix('.json').read_text())

@pytest.mark.parametrize('g,m',[(1e308,1e308),(1e-300,1e-100)])
def test_gravity_overflow_underflow_fail_explicitly(g,m):
    with pytest.raises(ValueError):
        uniform(g=(0.,-g,0.),m=m)


def test_no_new_unqualified_field_subclass_in_composed_motion():
    class Unqualified(PrescribedField):
        pass
    with pytest.raises(TypeError):
        replace(uniform(),field=Unqualified((UniformElectricField((1.,0.,0.)),)))


def test_classical_speed_limit_unchanged_with_gravity():
    with pytest.raises(UnsupportedProblem,match='0.1c'):
        uniform(v=(4e7,0.,0.))


def test_gravity_zero_control_and_disabled_electric_field_remain_distinct():
    s=uniform(g=(0.,0.,0.));r=integrate_test_charge(s)
    assert all(x.gravitational_force_n==(0.,0.,0.) for x in r.samples)
    assert r.samples[-1].acceleration_m_s2[0]>0
    s=uniform(E=(0.,0.,0.));r=integrate_test_charge(s)
    assert all(x.electric_field_v_m==(0.,0.,0.) and x.potential_v==0. for x in r.samples)
    assert r.samples[-1].acceleration_m_s2==(0.,-10.,0.)


def test_composed_gravity_reuses_joint_reflection_qualification():
    field=PrescribedField((FixedPointField(1e-9,(0.,.1,0.)),
                           FixedPointField(1e-9,(0.,-.1,0.))))
    s=MotionSpec(field,PointCharge(-1e-9),1e-6,(.1,0.,.03),.1,.01,
        plane_normal=(0.,1.,0.),gravity=UniformGravity((0.,0.,-.3)))
    for row in integrate_test_charge(s).samples:
        assert row.position_m[1]==0. and row.force_n[1]==0.
        assert row.gravitational_force_n[2]==pytest.approx(-3e-7)
