"""Geometric proof, full-vector dynamics, JEE references and honest boundaries."""
from dataclasses import FrozenInstanceError, asdict, replace
from itertools import permutations
import json
from math import cos, fsum, hypot, nextafter, pi, sin, sqrt

import pytest

from conductor_sim import (EPSILON_0, FixedPointField, PointCharge, PrescribedField,
    TestChargeMotionSpec as Spec, UniformElectricField, UniformRingAxisField,
    UniformRodField, UnsupportedProblem, integrate_test_charge, sample_test_charge_viewer)
from conductor_sim.source_symmetry import certify_point_reflection
from conductor_sim.scenarios import PrescribedScenario, revise_scenario
from conductor_sim.prescribed_viewer_validation import validate_prescribed_viewer
from examples.symmetric_charge_motion import (
    A, Q, Q_PROBE, MASS, K, CASES, ANNOTATIONS, make_scenario, build)


def points():
    return (FixedPointField(Q, (0., A, 0.)), FixedPointField(Q, (0., -A, 0.)))


def certify(sources, origin=(0.,0.,0.), normal=(0.,1.,0.)):
    return certify_point_reflection(sources, origin_m=origin, normal=normal)


def test_equal_signed_reflected_sources_and_original_index_binding():
    src = points(); original = tuple(s.as_dict() for s in src)
    c = certify(src)
    assert c.pairs == ((0,1),) and c.in_plane_indices == ()
    assert c.max_pair_residual_m == 0.
    assert 0 < c.tolerance_m < 1e-14
    assert tuple(s.as_dict() for s in src) == original
    with pytest.raises(FrozenInstanceError):
        c.pairs = ()


@pytest.mark.parametrize('sign', (1,-1))
def test_pairing_multiple_pairs_onplane_singletons_and_multiplicity(sign):
    src = (FixedPointField(sign*Q, (0,A,0)), FixedPointField(sign*Q, (0,-A,0)),
           FixedPointField(-2*Q, (.2,A,.03)), FixedPointField(-2*Q, (.2,-A,.03)),
           FixedPointField(3*Q, (.4,0,.05)),
           FixedPointField(sign*Q, (0,A,0)), FixedPointField(sign*Q, (0,-A,0)))
    c = certify(src)
    assert c.in_plane_indices == (4,)
    assert len(c.pairs) == 3
    all_indices = c.in_plane_indices + tuple(i for pair in c.pairs for i in pair)
    assert sorted(all_indices) == list(range(7))
    for i,j in c.pairs:
        assert src[i].charge_c == src[j].charge_c
        assert src[i].position_m[1] == -src[j].position_m[1]


@pytest.mark.parametrize('order', list(permutations(range(4))))
def test_source_order_does_not_decide_symmetry(order):
    src = points() + (FixedPointField(-2*Q,(.2,.3,0)),FixedPointField(-2*Q,(.2,-.3,0)))
    src = tuple(src[i] for i in order)
    c = certify(src)
    assert len(c.pairs) == 2
    assert all(src[i].charge_c == src[j].charge_c for i,j in c.pairs)


@pytest.mark.parametrize('bad', [
    points()[:1],
    (points()[0], FixedPointField(-Q,(0,-A,0))),
    (points()[0], FixedPointField(nextafter(Q,2*Q),(0,-A,0))),
    (points()[0], FixedPointField(Q,(1e-10,-A,0))),
    (points()[0], FixedPointField(Q,(0,-A-1e-10,0))),
    points() + (points()[0],),
])
def test_broken_charge_position_or_multiplicity_refused(bad):
    with pytest.raises(UnsupportedProblem, match='reflection'):
        certify(bad)


def test_pointwise_force_cancellation_is_not_a_geometric_certificate():
    field = PrescribedField((FixedPointField(Q,(0,A,0)),FixedPointField(4*Q,(0,-2*A,0))))
    assert field.electric_field_at((0,0,0)) == pytest.approx((0,0,0),abs=1e-12)
    assert abs(field.electric_field_at((.04,0,0))[1]) > 1.
    with pytest.raises(UnsupportedProblem, match='reflection'):
        Spec(field,PointCharge(-Q),MASS,(.1,0,0),.1,.01,
             plane_normal=(0,1,0),length_scale_m=A)


def test_nearby_unmatched_source_not_hidden_by_matching_a_partner_twice():
    src = (points()[0], points()[0], points()[1])
    with pytest.raises(UnsupportedProblem):
        certify(src)


@pytest.mark.parametrize('bad', [(), 'sources', [1], [UniformElectricField((1,0,0))]])
def test_nonpoint_and_malformed_certificates_refused(bad):
    with pytest.raises((TypeError,ValueError)):
        certify(bad)


def test_large_translation_cannot_hide_physical_asymmetry():
    # World resolution is inadequate relative to the small authored separation.
    src = (FixedPointField(Q,(1e12,A,0)),FixedPointField(Q,(1e12,-A,0)))
    with pytest.raises(UnsupportedProblem, match='poorly resolved'):
        certify(src,origin=(1e12,0,0))


def test_more_than_sixtyfour_sources_and_unsupported_custom_laws_refused():
    with pytest.raises(ValueError):
        certify(points()*33)
    class CustomPoint(FixedPointField):
        pass
    with pytest.raises(TypeError):
        certify((CustomPoint(Q,(0,A,0)),points()[1]))


@pytest.mark.parametrize('angle', [.31, .81, 1.27])
def test_rotated_translated_pair_and_physical_trajectory_covariance(angle):
    co,si=cos(angle),sin(angle)
    def rotate(p): return (co*p[0]-si*p[1],si*p[0]+co*p[1],p[2])
    origin=(2.,-3.,.7)
    def transform(p): return tuple(a+b for a,b in zip(rotate(p),origin))
    src=points()
    transformed=tuple(FixedPointField(s.charge_c,transform(s.position_m)) for s in src)
    normal=rotate((0,1,0))
    c=certify(transformed,origin=origin,normal=normal)
    assert c.pairs==((0,1),) and c.max_pair_residual_m<=c.tolerance_m
    p0=(.04,0,.02);v0=(.03,0,.06)
    base=Spec(PrescribedField(src),PointCharge(-Q,p0),MASS,v0,.04,.005,
              plane_normal=(0,1,0),length_scale_m=A)
    variant=Spec(PrescribedField(transformed),PointCharge(-Q,transform(p0)),MASS,rotate(v0),.04,.005,
                 plane_normal=normal,length_scale_m=A)
    first,last=integrate_test_charge(base),integrate_test_charge(variant)
    for a,b in zip(first.samples,last.samples):
        assert b.position_m==pytest.approx(transform(a.position_m),abs=2e-11)
        assert b.velocity_m_s==pytest.approx(rotate(a.velocity_m_s),abs=2e-10)


def test_unmodified_source_visible_plane_and_new_mirrored_plane_agree():
    base=make_scenario().spec
    other=replace(base,plane_normal=(0.,1.,0.))
    first,second=integrate_test_charge(base),integrate_test_charge(other)
    # Both integrate the same full 3-vector physical law. No r/v projection.
    assert first.samples==second.samples
    assert first.spec.axis is None and second.spec.axis is None
    assert all(r.position_m[1:]==(0.,0.) for r in second.samples)


def independent_turning_time(panels=8192):
    amplitude=sqrt(3)*A
    # x=X*(1-u²) removes the integrable singularity at the endpoint. This is
    # the independent energy-time quadrature, not the numerical RK trajectory.
    terms=[]
    for j in range(panels):
        u=(j+.5)/panels; x=amplitude*(1-u*u); h=hypot(A,x)
        speed2=4*K*Q*abs(Q_PROBE)/MASS*(amplitude-x)*(amplitude+x)/(2*A*h*(2*A+h))
        terms.append(2*amplitude*u/sqrt(speed2)/panels)
    return fsum(terms)


def test_source_illustration_turning_point_and_independent_turning_time():
    scenario,result,bundle=build('turning')
    ref=bundle['m21c_teaching']['reference']
    assert scenario.spec.initial_velocity_m_s[0]==pytest.approx(sqrt(2*K*Q*abs(Q_PROBE)/(MASS*A)))
    assert ref['largest_saved_x_m']==pytest.approx(sqrt(3)*A,abs=1e-6)
    t=independent_turning_time()
    assert ref['turning_velocity_bracket_s'][0]<=t<=ref['turning_velocity_bracket_s'][1]
    assert result.samples[-1].velocity_m_s[0]<0
    assert ref['max_energy_change_j']<1e-13
    assert ref['max_reference_E_V_error']<1e-9
    assert result.termination=='duration'
    assert 'not asserted' in bundle['m21c_teaching']['provenance']


@pytest.mark.parametrize('case', ['escape-threshold','above-escape'])
def test_escape_energy_relation_and_only_finite_time_claim(case):
    scenario,result,bundle=build(case)
    initial=result.samples[0]
    if case=='escape-threshold':
        assert abs(initial.total_energy_j)<1e-20
    else:
        assert initial.total_energy_j>0
    for row in result.samples:
        expected_v2=2/MASS*(initial.total_energy_j+2*K*Q*abs(Q_PROBE)/hypot(A,row.position_m[0]))
        assert row.velocity_m_s[0]**2==pytest.approx(expected_v2,abs=3e-10)
        assert row.velocity_m_s[0]>0
    assert result.samples[-1].position_m[0]<1.
    assert result.samples[-1].time_s==2.
    assert 'Finite physical times' in bundle['m21c_teaching']['reference']['time_scope']
    assert scenario.parent_revision_id==make_scenario().revision_id


def test_genuine_oblique_plane_case_no_inplane_replacement_sources():
    scenario,result,bundle=build('mirror-plane')
    assert scenario.spec.plane_normal==(0.,1.,0.)
    assert scenario.spec.axis is None
    assert scenario.spec.field.sources==points()
    assert all(row.position_m[1]==0 and row.force_n[1]==0 for row in result.samples)
    assert result.samples[-1].position_m[0]!=result.samples[0].position_m[0]
    assert result.samples[-1].position_m[2]!=result.samples[0].position_m[2]
    assert abs(result.samples[-1].force_n[0])>0 and abs(result.samples[-1].force_n[2])>0
    assert validate_prescribed_viewer(bundle) is True
    for f in bundle['frames']:
        assert sum('signed normal offset' in n for n in f['notes'])==2
        assert any('charge:upper' in n for n in f['notes'])
        assert any('charge:lower' in n for n in f['notes'])
    assert 'Teacher hypothetical' in bundle['m21c_teaching']['source_fidelity']


def test_axis_only_and_other_source_qualification_remains_strict():
    with pytest.raises(UnsupportedProblem):
        Spec(PrescribedField(points()+(UniformRingAxisField(Q,.2,axis=(1,0,0)),)),
             PointCharge(-Q),MASS,(.1,0,0),.1,.01,plane_normal=(0,1,0))
    with pytest.raises(UnsupportedProblem):
        Spec(PrescribedField(points()+(UniformElectricField((0,1,0)),)),
             PointCharge(-Q),MASS,(.1,0,0),.1,.01,plane_normal=(0,1,0))
    with pytest.raises(UnsupportedProblem):
        Spec(PrescribedField(points()+(UniformRodField(Q,(0,.2,0),(.1,.2,0)),)),
             PointCharge(-Q),MASS,(.1,0,0),.1,.01,plane_normal=(0,1,0))


def test_runtime_transverse_force_check_not_relaxed_or_projected():
    class BadField(PrescribedField):
        def electric_field_at(self,p):
            e=super().electric_field_at(p)
            return (e[0],e[1]+1.,e[2])
    spec=Spec(BadField(points()),PointCharge(-Q,(.04,0,.02)),MASS,(.03,0,.06),.1,.01,
              plane_normal=(0,1,0),length_scale_m=A)
    with pytest.raises(UnsupportedProblem,match='transverse force'):
        integrate_test_charge(spec)


def test_asymmetric_revision_invalidates_parent_certificate_without_mutation():
    base=make_scenario('mirror-plane');before=base.as_dict()
    with pytest.raises(UnsupportedProblem,match='reflection'):
        revise_scenario(base,scenario_id='asymmetric',note='Change only upper source.',
                        source_changes={'charge:upper':{'charge_c':2*Q}})
    assert base.as_dict()==before
    new=revise_scenario(base,scenario_id='symmetric-revision',note='Double both sources.',
        source_changes={'charge:upper':{'charge_c':2*Q},'charge:lower':{'charge_c':2*Q}})
    assert new.parent_revision_id==base.revision_id
    assert new.source_annotations==ANNOTATIONS
    assert integrate_test_charge(new.spec).samples[-1].position_m!=integrate_test_charge(base.spec).samples[-1].position_m
    assert base.as_dict()==before


def test_nonaxial_launch_breaks_axis_but_valid_planar_motion_is_not_suppressed():
    base=make_scenario().spec
    # In the visible xy plane this remains a valid 2D motion, not forced onto x.
    tilted=replace(base,initial_velocity_m_s=(.3,.02,0.),duration_s=.1)
    result=integrate_test_charge(tilted)
    assert abs(result.samples[-1].position_m[1])>1e-5
    # In the xz reflection plane that same launch has a real normal component.
    with pytest.raises(UnsupportedProblem,match='initial velocity'):
        replace(tilted,plane_normal=(0,1,0))
