"""M8 exact exterior isolated-conductor physics and FEMM command contracts.

These tests use independent physical expectations and an instrumented fake client.
Fake results do not count as real FEMM numerical validation.
"""
from math import cos, pi, sin

import pytest

from conductor_sim import (
    EPSILON_0, ElectrostaticProblem, ExternalGroundedSphereBackend,
    ExternalIsolatedSphereBackend, ExternalIsolatedSphereSolution,
    ExternalSphereBackend, Femm5ExternalSphereBackend, FixedNetCharge,
    Grounded, PointCharge, SolidSphere,
)
from conductor_sim.femm5_external import _SOURCE_NAME
from conductor_sim.solver import SingularField, UnsupportedProblem
from conductor_sim.visualization import SliceSpec, sample_sweep
from test_femm5_adapter import FakeClient, FakeSession, calls

A = .12
D = .27
Q = 1e-9
K = 1/(4*pi*EPSILON_0)


def problem(sphere_charge=0., source_charge=Q, center=(0.,0.,0.), direction=(0.,0.,1.)):
    source_position = tuple(center[i]+D*direction[i] for i in range(3))
    return ElectrostaticProblem(
        charges=(PointCharge(source_charge,source_position),),
        conductors=(SolidSphere(A,FixedNetCharge(sphere_charge),center_m=center,name='metal'),)
    )


def grounded():
    return ElectrostaticProblem((PointCharge(Q,(0.,0.,D)),),
                                (SolidSphere(A,Grounded(),name='metal'),))


def test_neutral_isolated_uses_two_image_charges_and_constant_nonzero_potential():
    s = ExternalIsolatedSphereBackend().solve(problem())
    assert isinstance(s, ExternalIsolatedSphereSolution)
    assert s.image_charge_c == pytest.approx(-Q*A/D)
    assert s.central_image_charge_c == pytest.approx(+Q*A/D)
    assert s.image_charge_c+s.central_image_charge_c == pytest.approx(0.,abs=1e-25)
    c=s.conductor_results[0]
    assert c.net_charge_c == 0
    assert c.inner_surface_charge_c == 0
    assert c.outer_surface_charge_c == 0
    assert c.potential_v == pytest.approx(K*Q/D)
    for theta in (0.,.15,.7,pi/2,2.4,pi):
        surface=(A*sin(theta),0.,A*cos(theta))
        assert s.potential_at(surface) == pytest.approx(c.potential_v,abs=2e-12)
    assert s.potential_at((0.,0.,0.)) == pytest.approx(c.potential_v)
    assert s.electric_field_at((0.,0.,0.)) == (0.,0.,0.)
    with pytest.raises(SingularField):
        s.electric_field_at((0.,0.,A))
    with pytest.raises(SingularField):
        s.potential_at((0.,0.,D))


@pytest.mark.parametrize('net', (-1.2e-9,-.6e-9,0.,.4e-9,2e-9))
def test_fixed_net_charge_surface_integral_and_far_field(net):
    s=ExternalIsolatedSphereBackend().solve(problem(sphere_charge=net))
    assert s.conductor_results[0].net_charge_c == pytest.approx(net,abs=1e-28)
    assert s.central_image_charge_c + s.image_charge_c == pytest.approx(net,rel=1e-13,abs=1e-26)
    assert s.conductor_results[0].potential_v == pytest.approx(K*(net+Q*A/D)/A)
    h=pi/2000
    area_integral=2*pi*A*A*h*sum(
        s.surface_charge_density('outer',(i+.5)*h)*sin((i+.5)*h)
        for i in range(2000)
    )
    assert area_integral == pytest.approx(net,rel=1e-4,abs=1.5e-13)
    point=(0.,0.,2000.)
    # The exterior system consists of the external q plus the metal net Q.
    asymptotic=K*(Q+net)/point[2]
    assert s.potential_at(point) == pytest.approx(asymptotic,rel=5e-4,abs=1e-5)


def test_superposition_difference_is_monopole_field_plus_uniform_sigma():
    base=ExternalIsolatedSphereBackend().solve(problem(sphere_charge=0))
    delta_net=1.8e-9
    charged=ExternalIsolatedSphereBackend().solve(problem(sphere_charge=delta_net))
    for pt in ((.15,0,.05),(.3,.08,-.1),(-.24,.01,.18)):
        r=sum(v*v for v in pt)**.5
        assert charged.potential_at(pt)-base.potential_at(pt) == pytest.approx(K*delta_net/r,rel=1e-13)
        diff=tuple(a-b for a,b in zip(charged.electric_field_at(pt),base.electric_field_at(pt)))
        assert diff == pytest.approx(tuple(K*delta_net*v/(r**3) for v in pt),rel=2e-13)
    for theta in (.1,.9,2.6,pi):
        assert (charged.surface_charge_density('outer',theta)-base.surface_charge_density('outer',theta)) == pytest.approx(
            delta_net/(4*pi*A*A),rel=1e-13
        )
    assert charged.image_charge_c == base.image_charge_c
    assert charged.image_position_m == base.image_position_m


def test_neutral_is_attractive_but_strong_positive_net_charge_can_repulse():
    neutral=ExternalIsolatedSphereBackend().solve(problem())
    charged=ExternalIsolatedSphereBackend().solve(problem(sphere_charge=2e-9))
    assert neutral.induced_force_on_charge_n[2] < 0
    assert charged.induced_force_on_charge_n[2] > 0
    assert neutral.surface_charge_density('outer',0.) < 0
    assert neutral.surface_charge_density('outer',pi) > 0
    assert charged.conductor_results[0].net_charge_c == 2e-9
    # A positive source with an isolated neutral sphere still polarizes and attracts.
    d=D
    expected_neutral=-K*Q*Q*A*d/(d*d-A*A)**2+K*Q*Q*A/d**3
    assert neutral.induced_force_on_charge_n == pytest.approx((0.,0.,expected_neutral))


@pytest.mark.parametrize('net', (-.7e-9,0.,2e-9))
def test_force_is_minus_distance_derivative_of_interaction_energy(net):
    def e(distance):
        p=ElectrostaticProblem(
            (PointCharge(Q,(0,0,distance)),),
            (SolidSphere(A,FixedNetCharge(net)),)
        )
        return ExternalIsolatedSphereBackend().solve(p).interaction_energy_j
    s=ExternalIsolatedSphereBackend().solve(problem(sphere_charge=net))
    step=3e-6
    force=-(e(D+step)-e(D-step))/(2*step)
    assert s.induced_force_on_charge_n[2] == pytest.approx(force,rel=3e-8)
    # Energy excludes the sphere's distance-independent electrostatic self energy.
    if net==0:
        assert s.interaction_energy_j < 0


def test_reference_field_matches_negative_potential_gradient_and_rotation():
    s=ExternalIsolatedSphereBackend().solve(problem(sphere_charge=-.5e-9))
    for pt in ((.15,0,.03),(.18,.04,-.2),(.29,-.08,.09)):
        step=1e-6
        gradient=[]
        for i in range(3):
            plus=tuple(pt[k]+(step if k==i else 0) for k in range(3))
            minus=tuple(pt[k]-(step if k==i else 0) for k in range(3))
            gradient.append((s.potential_at(plus)-s.potential_at(minus))/(2*step))
        assert s.electric_field_at(pt)==pytest.approx(tuple(-x for x in gradient),rel=2e-5,abs=1e-3)
    moved=ExternalIsolatedSphereBackend().solve(problem(-.5e-9,center=(1,-2,.5),direction=(1,0,0)))
    old_pt=(.16,.04,.2)
    new_pt=(1+old_pt[2],-2+old_pt[1],.5+old_pt[0])
    assert moved.potential_at(new_pt)==pytest.approx(s.potential_at(old_pt))
    old_field=s.electric_field_at(old_pt)
    assert moved.electric_field_at(new_pt)==pytest.approx((old_field[2],old_field[1],old_field[0]))


def test_grounded_backend_remains_strict_and_common_analytical_dispatches():
    g=ExternalSphereBackend().solve(grounded())
    i=ExternalSphereBackend().solve(problem())
    assert g.conductor_results[0].potential_v == 0
    assert i.conductor_results[0].net_charge_c == 0
    assert g.conductor_results[0].net_charge_c == pytest.approx(-Q*A/D)
    assert not ExternalGroundedSphereBackend().supports(problem())
    assert not ExternalIsolatedSphereBackend().supports(grounded())
    for p in (
        ElectrostaticProblem((PointCharge(Q,(0,0,.05)),),(SolidSphere(A,FixedNetCharge(0)),)),
        ElectrostaticProblem((PointCharge(Q,(0,0,A)),),(SolidSphere(A,FixedNetCharge(0)),)),
        ElectrostaticProblem((PointCharge(Q,(0,0,D)),PointCharge(-Q,(0,0,-D))),
                             (SolidSphere(A,FixedNetCharge(0)),)),
    ):
        for b in (ExternalIsolatedSphereBackend(),ExternalSphereBackend(),Femm5ExternalSphereBackend()):
            assert not b.supports(p)
            with pytest.raises(UnsupportedProblem):b.solve(p)


@pytest.mark.parametrize('net', (0.,-2e-9,1.25e-9))
def test_femm_fixed_charge_conductor_property_and_numerical_provenance(net):
    fake=FakeClient()
    fake.eo_getconductorproperties=lambda name: (15.,Q) if name==_SOURCE_NAME else (123.,-4e-9)
    session=FakeSession(fake)
    backend=Femm5ExternalSphereBackend(session_factory=lambda:session)
    p=problem(net)
    with backend.solve(p) as numeric:
        assert numeric.conductor_results[0].potential_v == 123.
        assert numeric.conductor_results[0].net_charge_c == -4e-9
        assert numeric.numerical_source_charge_c == Q
        assert numeric.potential_at((0,0,-.3)) == 12
        assert numeric.surface_charge_density('outer',.5) != 0
        assert numeric.sphere.constraint.charge_c == net
    assert calls(fake,'ei_addconductorprop') == [
        ('metal',0,net,0),(_SOURCE_NAME,0,Q,0)
    ]
    assert calls(fake,'ei_addpointprop') == []
    assert session.opens == session.closes == 1


def test_viewer_composes_grounded_and_isolated_states_without_new_renderer():
    cases=[('grounded',grounded()),('neutral',problem(0)),('positive',problem(1e-9))]
    bundle=sample_sweep(cases,ExternalSphereBackend(),
                        plane=SliceSpec.meridian(cases[0][1],extent_m=.48,resolution=19),line_steps=110)
    assert len(bundle['frames'])==3
    for (label,p),frame in zip(cases,bundle['frames']):
        assert frame['label']==label
        assert len(frame['profiles'])==1
        assert frame['profiles'][0]['surface']=='outer'
        assert frame['conductor_results'][0]['name']=='metal'
        assert frame['density_method'].startswith('exact Kelvin')
    assert bundle['frames'][0]['conductor_results'][0]['net_charge_c'] < 0
    assert bundle['frames'][1]['conductor_results'][0]['net_charge_c'] == 0
    assert bundle['frames'][2]['conductor_results'][0]['net_charge_c'] == Q


def test_source_sign_reversal_neutral_induction_and_force_symmetry():
    positive=ExternalIsolatedSphereBackend().solve(problem(0.,Q))
    negative=ExternalIsolatedSphereBackend().solve(problem(0.,-Q))
    for pt in ((.13,0,.02),(.23,.05,-.06)):
        assert negative.potential_at(pt)==pytest.approx(-positive.potential_at(pt))
        assert negative.electric_field_at(pt)==pytest.approx(
            tuple(-a for a in positive.electric_field_at(pt)))
    for theta in (.12,.8,2.9):
        assert negative.surface_charge_density('outer',theta)==pytest.approx(
            -positive.surface_charge_density('outer',theta))
    assert negative.force_on_charge_n == pytest.approx(positive.force_on_charge_n)
    assert negative.interaction_energy_j == pytest.approx(positive.interaction_energy_j)
