"""M22c: whole finite-rod reflection, negative contracts and independent physics.

The reserved transfer card is deliberately NOT imported or executed here.
Two motivating cases were frozen in docs/m22c_charter.md before code changes.
"""
from dataclasses import FrozenInstanceError, asdict, replace
from itertools import permutations
from math import cos, fsum, hypot, isclose, nextafter, pi, sin
import copy

import pytest

from conductor_sim import (EPSILON_0, FixedPointField, PointCharge,
    PrescribedField, TestChargeMotionSpec as Spec, UniformElectricField,
    UniformGravity, UniformRodField, UniformRingAxisField, UnsupportedProblem,
    integrate_test_charge, sample_test_charge_viewer)
from conductor_sim.prescribed_viewer_validation import validate_prescribed_viewer
from conductor_sim.source_symmetry import certify_rod_reflection
from conductor_sim.scenarios import revise_scenario
from conductor_sim.structural_scenarios import NamedSource, reassemble_scenario
from examples.mirrored_rods import make_case, create_case

K = 1/(4*pi*EPSILON_0)


def rods():
    return (UniformRodField(1.5e-9,(-.05,.07,-.02),(.05,.07,-.02)),
            UniformRodField(1.5e-9,(-.05,-.07,-.02),(.05,-.07,-.02)))


def certificate(src, *, origin=(0.,0.,.1), normal=(0.,1.,0.)):
    return certify_rod_reflection(src, origin_m=origin, normal=normal)


def independent_line_integral(sources, point, *, panels=4096):
    """Independent midpoint discretization of the Coulomb *line integral*."""
    e=[0.,0.,0.];v=0.
    for rod in sources:
        a,b=rod.start_m,rod.end_m
        charge_slice=rod.charge_c/panels
        for j in range(panels):
            f=(j+.5)/panels
            p=tuple(a[i]+(b[i]-a[i])*f for i in range(3))
            d=tuple(point[i]-p[i] for i in range(3))
            r=hypot(*d)
            w=K*charge_slice/r
            v+=w
            for i in range(3):e[i]+=w*d[i]/(r*r)
    return v,tuple(e)


def axis_spec(src, *, p=(0.,0.,.1),v=(.03,0.,.01),gravity=None):
    return Spec(PrescribedField(tuple(src)),PointCharge(-4e-10,p),3e-6,v,
                .14,.01,plane_normal=(0.,1.,0.),length_scale_m=.4,
                source_guard_m=.002,gravity=gravity)


def test_m22b_frozen_geometry_and_certificate():
    s=rods(); original=tuple(x.as_dict() for x in s)
    c=certificate(s)
    assert c.pairs==((0,1),) and c.invariant_indices==()
    assert c.max_pair_residual_m<=c.tolerance_m<1e-12
    assert tuple(x.as_dict() for x in s)==original
    with pytest.raises(FrozenInstanceError):
        c.pairs=()
    assert make_case('P1').spec.field.sources==s


def test_tilted_reversed_endpoint_order_is_same_full_segment():
    case=make_case('P2')
    c=certificate(case.spec.field.sources[:2],origin=case.spec.particle.position_m)
    assert c.pairs==((0,1),)
    s=case.spec.field.sources[1]
    reversed_rod=replace(s,start_m=s.end_m,end_m=s.start_m)
    changed=replace(case.spec,field=PrescribedField((case.spec.field.sources[0],
                         reversed_rod,*case.spec.field.sources[2:])))
    assert integrate_test_charge(changed).samples[-1].position_m==pytest.approx(
        integrate_test_charge(case.spec).samples[-1].position_m,abs=1e-11)


@pytest.mark.parametrize('order', list(permutations(range(4))))
def test_multiple_mirror_pairs_permutation_and_full_multiplicity(order):
    src=rods()+(
       UniformRodField(-1e-9,(-.1,.09,.1),(.06,.11,.04)),
       UniformRodField(-1e-9,(.06,-.11,.04),(-.1,-.09,.1)))
    src=tuple(src[i] for i in order)
    c=certificate(src)
    assert len(c.pairs)==2
    assert sorted(i for pair in c.pairs for i in pair)==[0,1,2,3]
    assert all(src[i].charge_c==src[j].charge_c for i,j in c.pairs)


def test_duplicate_identical_pairs_are_matched_without_source_loss():
    src=rods()*2
    c=certificate(src)
    assert len(c.pairs)==2
    assert sorted(i for pair in c.pairs for i in pair)==[0,1,2,3]
    with pytest.raises(UnsupportedProblem,match='rod.*reflection'):
        certificate((src[0],src[0],src[1]))


def test_existing_inplane_and_exact_normal_rod_remain_admitted():
    in_plane=UniformRodField(+1e-9,(-.1,0.,-.04),(.1,0.,-.04))
    crossing=UniformRodField(-1e-9,(.15,-.1,.03),(.15,.1,.03))
    c=certificate((in_plane,crossing))
    assert c.invariant_indices==(0,1) and c.pairs==()
    spec=axis_spec((in_plane,crossing))
    assert integrate_test_charge(spec).termination=='duration'
    # M19 pre-existing per-source path works without any newly emitted omit key.
    b=sample_test_charge_viewer(spec,visual_frames=5,resolution=9,extent_m=.3)
    assert 'mirrored_rod_source_indices' not in b['prescribed_motion']


@pytest.mark.parametrize('case',('P1','P2'))
def test_invariant_force_true_3d_and_independent_line_integral(case):
    spec=make_case(case).spec
    rods_only=tuple(s for s in spec.field.sources if type(s) is UniformRodField)
    for point in ((.02,0.,.11),(-.03,0.,.16),(.04,0.,.04),(.12,0.,.12)):
        approx_v,approx_e=independent_line_integral(rods_only,point)
        exact_v=fsum(s.potential_at(point,permittivity_f_m=spec.field.permittivity_f_m) for s in rods_only)
        exact_e=tuple(fsum(s.electric_field_at(point,permittivity_f_m=spec.field.permittivity_f_m)[i]
                           for s in rods_only) for i in range(3))
        assert exact_v==pytest.approx(approx_v,rel=8e-7,abs=1e-6)
        assert exact_e==pytest.approx(approx_e,rel=2e-6,abs=2e-4)
        assert abs(exact_e[1])<2e-10
        field_e=spec.field.electric_field_at(point)
        assert field_e[1]==pytest.approx(0.,abs=2e-10)
        h=1e-5
        for j in (0,2):
            plus=list(point);minus=list(point);plus[j]+=h;minus[j]-=h
            ref=-(spec.field.potential_at(plus)-spec.field.potential_at(minus))/(2*h)
            assert field_e[j]==pytest.approx(ref,rel=2e-6,abs=2e-3)


def test_p1_and_p2_saved_full_vector_forces_energy_and_trajectories():
    for case in ('P1','P2'):
        model=make_case(case)
        result=integrate_test_charge(model.spec)
        assert result.termination=='duration' and len(result.samples)==15
        assert result.samples[-1].time_s==pytest.approx(.14)
        assert result.max_abs_energy_drift_j<1e-12
        for row in result.samples:
            assert abs(row.position_m[1])<1e-10
            assert abs(row.velocity_m_s[1])<1e-10
            assert abs(row.force_n[1])<1e-13
            assert abs(row.electric_field_v_m[1])<2e-10
            assert min(s.clearance_at(row.position_m) for s in model.spec.field.sources)>.002
        html,audit=create_case(case)
        assert audit['full_motion']['samples'][-1]['time_s']==.14
        assert validate_prescribed_viewer(audit['viewer']) is True
        assert html.startswith('<!doctype html>')
        m=audit['viewer']['prescribed_motion']
        assert m['mirrored_rod_source_indices']==[0,1]
        for frame in audit['viewer']['frames']:
            notes='\n'.join(frame['notes'])
            assert 'rod:upper' in notes if case=='P1' else 'rod:tilted-1' in notes
            assert 'full physical 3D field is included' in notes
            assert 'Not drawn here' in notes


def test_independent_velocity_verlet_dynamic_reference():
    for case in ('P1','P2'):
        s=make_case(case).spec
        actual=integrate_test_charge(s)
        h=2e-5
        steps=round(s.duration_s/h)
        p=list(s.particle.position_m);v=list(s.initial_velocity_m_s)
        q=s.particle.charge_c; m=s.mass_kg
        def a(x):
            e=s.field.electric_field_at(x)
            g=s.gravity.acceleration_m_s2 if s.gravity is not None else (0.,0.,0.)
            return [q*e[i]/m+g[i] for i in range(3)]
        accel=a(p)
        for _ in range(steps):
            np=[p[i]+h*v[i]+.5*h*h*accel[i] for i in range(3)]
            na=a(np)
            v=[v[i]+.5*h*(accel[i]+na[i]) for i in range(3)]
            p,accel=np,na
        assert actual.samples[-1].position_m==pytest.approx(p,abs=1.5e-7)
        assert actual.samples[-1].velocity_m_s==pytest.approx(v,abs=2e-6)


def test_rotation_translation_conjugate_geometry_and_trajectory():
    base=make_case('P1').spec
    angle=.63;co,si=cos(angle),sin(angle)
    def rotate(p):return (co*p[0]-si*p[1],si*p[0]+co*p[1],p[2])
    offset=(2.,-1.,.8)
    def transform(p):return tuple(a+b for a,b in zip(rotate(p),offset))
    rs=tuple(UniformRodField(s.charge_c,transform(s.start_m),transform(s.end_m))
             for s in base.field.sources)
    variant=replace(base,field=PrescribedField(rs),
                    particle=PointCharge(base.particle.charge_c,transform(base.particle.position_m)),
                    initial_velocity_m_s=rotate(base.initial_velocity_m_s),
                    plane_normal=rotate(base.plane_normal))
    c=certificate(rs,origin=variant.particle.position_m,normal=variant.plane_normal)
    assert c.pairs==((0,1),)
    first,second=integrate_test_charge(base),integrate_test_charge(variant)
    for old,new in zip(first.samples,second.samples):
        assert new.position_m==pytest.approx(transform(old.position_m),abs=2e-9)
        assert new.velocity_m_s==pytest.approx(rotate(old.velocity_m_s),abs=2e-8)


@pytest.mark.parametrize('bad',[
    rods()[:1],
    (rods()[0],replace(rods()[1],charge_c=-rods()[1].charge_c)),
    (rods()[0],replace(rods()[1],charge_c=nextafter(rods()[1].charge_c,2e-9))),
    (rods()[0],replace(rods()[1],start_m=(-.050000001,-.07,-.02))),
    (rods()[0],replace(rods()[1],end_m=(.05,-.07,-.01999))),
    rods()+(rods()[0],),
])
def test_asymmetric_geometry_charge_and_multiplicity_are_refused(bad):
    with pytest.raises(UnsupportedProblem,match='rod.*reflection'):
        certificate(bad)
    with pytest.raises(UnsupportedProblem):
        axis_spec(bad)


def test_pointwise_cancellation_never_replaces_reflection_proof():
    # Adjust charge and y offset for E_y=0 only at x=z=0 by symmetry at one point.
    a=UniformRodField(1e-9,(-.05,.08,-.02),(.05,.08,-.02))
    b=UniformRodField(4e-9,(-.05,-.16,-.02),(.05,-.16,-.02))
    with pytest.raises(UnsupportedProblem,match='rod.*reflection'):
        certificate((a,b))
    with pytest.raises(UnsupportedProblem):
        axis_spec((a,b))


def test_large_translation_cannot_hide_asymmetric_roundoff():
    large=1e12
    translated=tuple(UniformRodField(s.charge_c,
        tuple(x+large if k==0 else x for k,x in enumerate(s.start_m)),
        tuple(x+large if k==0 else x for k,x in enumerate(s.end_m))) for s in rods())
    with pytest.raises(UnsupportedProblem,match='poorly resolved'):
        certificate(translated,origin=(large,0,.1))


@pytest.mark.parametrize('bad', [(), 'rod', [1], [UniformElectricField((10,0,0))], rods()*33])
def test_malformed_or_oversized_certificates_are_refused(bad):
    with pytest.raises((TypeError,ValueError)):
        certificate(bad)


def test_axial_ring_disc_and_transverse_fields_still_rejected():
    ring=UniformRingAxisField(+1e-9,.1,axis=(1.,0.,0.))
    with pytest.raises(UnsupportedProblem):
        axis_spec(rods()+(ring,))
    with pytest.raises(UnsupportedProblem,match='gravity'):
        axis_spec(rods(),gravity=UniformGravity((0.,-.2,0.)))
    with pytest.raises(UnsupportedProblem,match='fixed sources'):
        axis_spec(rods()+(UniformElectricField((0.,3.,0.)),))
    with pytest.raises(UnsupportedProblem,match='velocity'):
        axis_spec(rods(),v=(.03,.02,.01))


def test_actual_source_guard_is_not_waived_by_global_symmetry():
    with pytest.raises(ValueError,match='guard'):
        # Very close to an extra in-plane rod, while mirrored pair remains valid.
        axis_spec(rods()+(UniformRodField(.1e-9,(-.02,0.,.1),(.02,0.,.1)),),
                  p=(0.,0.,.1001),v=(0.,0.,0.))


def test_revisions_and_atomic_mirror_pair_requalification():
    original=make_case('P1')
    snapshot=original.as_dict()
    with pytest.raises(UnsupportedProblem,match='rod.*reflection'):
        revise_scenario(original,scenario_id='broken',note='Change only one held rod.',
            source_changes={'rod:upper':{'charge_c':+1.6e-9}})
    valid=revise_scenario(original,scenario_id='equal-edit',note='Change both held rods.',
        source_changes={'rod:upper':{'charge_c':+1.6e-9},
                        'rod:lower':{'charge_c':+1.6e-9}})
    assert valid.parent_revision_id==original.revision_id
    assert valid.spec.field.sources[0].charge_c==valid.spec.field.sources[1].charge_c==1.6e-9
    assert integrate_test_charge(valid.spec).termination=='duration'
    parent=make_case('P2')
    # Strictly different named line-charge components are inserted atomically.
    fresh=(NamedSource('added:upper','Added upper fixed rod',
                UniformRodField(.2e-9,(-.01,.2,.05),(.08,.2,.05))),
           NamedSource('added:lower','Added lower fixed rod',
                UniformRodField(.2e-9,(-.01,-.2,.05),(.08,-.2,.05))))
    with pytest.raises(UnsupportedProblem):
        reassemble_scenario(parent,scenario_id='one-rod-added',
             note='Broken half assembly',add_sources=fresh[:1])
    child=reassemble_scenario(parent,scenario_id='two-rods-added',
         note='Teacher hypothetical, add both full mirrored rods.',add_sources=fresh)
    assert [x['operation'] for x in child.changes]==['add-source','add-source']
    assert integrate_test_charge(child.scenario.spec).termination=='duration'
    assert original.as_dict()==snapshot


def test_viewer_omission_metadata_tampering_rejected():
    _,audit=create_case('P1')
    model=audit['viewer']
    for changed in ([],[0],[0,1,2],[1,0],['0',1]):
        bad=copy.deepcopy(model)
        bad['prescribed_motion']['mirrored_rod_source_indices']=changed
        with pytest.raises(ValueError,match='off-plane rod'):
            validate_prescribed_viewer(bad)
    bad=copy.deepcopy(model)
    del bad['prescribed_motion']['mirrored_rod_source_indices']
    with pytest.raises(ValueError,match='off-plane rod'):
        validate_prescribed_viewer(bad)
