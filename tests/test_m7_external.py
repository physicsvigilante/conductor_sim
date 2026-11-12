"""M7 exact exterior sphere physics and FEMM compiler/adapter contract tests.

These tests do not claim FEMM numerical accuracy; see test_m7_live.py.
"""
from math import cos, pi, sin, sqrt

import pytest

from conductor_sim import (
    ElectrostaticProblem, ExternalGroundedSphereBackend,
    Femm5ExternalSphereBackend, Femm5Settings, FixedNetCharge,
    Grounded, PointCharge, SolidSphere, SphericalShell,
)
from conductor_sim.model import EPSILON_0
from conductor_sim.femm5_external import _SOURCE_NAME
from conductor_sim.solver import SingularField, UnsupportedProblem
from conductor_sim.visualization import SliceSpec, sample_sweep

from test_femm5_adapter import FakeClient, FakeSession, calls


A = .12
Q = 1e-9
D = .27


def p(charge=Q, distance=D, sphere_center=(0.,0.,0.), direction=(0.,0.,1.)):
    center=sphere_center
    return ElectrostaticProblem(
        charges=(PointCharge(charge,tuple(center[i]+distance*direction[i] for i in range(3))),),
        conductors=(SolidSphere(A,Grounded(),center_m=center,name='metal'),),
    )


def test_new_solid_sphere_component_validation_and_legacy_backend_rejection():
    from conductor_sim import ConcentricShellBackend, Femm5Backend, NestedShellBackend
    sphere=SolidSphere(A,Grounded())
    assert sphere.radius_m==sphere.outer_radius_m==A
    assert sphere.inner_radius_m==0
    assert ElectrostaticProblem(conductors=(sphere,)).conductors==(sphere,)
    for radius in (0.,-1.,float('nan')):
        with pytest.raises(ValueError):SolidSphere(radius,Grounded())
    with pytest.raises(ValueError):SolidSphere(.12,Grounded(),name='bad"name')
    with pytest.raises(ValueError):SolidSphere(.12,Grounded(),name='\n')
    with pytest.raises(TypeError):SolidSphere(.12,object())
    with pytest.raises(ValueError):SolidSphere(.12,Grounded(),center_m=(float('inf'),0,0))
    for backend in (ConcentricShellBackend(), Femm5Backend(), NestedShellBackend()):
        assert not backend.supports(p())
        with pytest.raises(UnsupportedProblem):backend.solve(p())


def test_exact_grounded_surface_image_charge_and_net_charge():
    s=ExternalGroundedSphereBackend().solve(p())
    expected=-Q*A/D
    assert s.image_charge_c == pytest.approx(expected)
    assert s.image_position_m==(0,0,A*A/D)
    assert s.conductor_results[0].potential_v==0
    assert s.conductor_results[0].net_charge_c==pytest.approx(expected)
    assert s.conductor_results[0].inner_surface_charge_c==0
    assert s.conductor_results[0].outer_surface_charge_c==pytest.approx(expected)
    for theta in (0.,.25,.6,1.2,pi/2,2.5,pi):
        x=(A*sin(theta),0,A*cos(theta))
        assert s.potential_at(x)==pytest.approx(0,abs=2e-12)
        assert s.surface_charge_density_at('outer',x)==pytest.approx(s.surface_charge_density('outer',theta))
    assert s.potential_at((0,0,0))==0
    assert s.electric_field_at((0,0,0))==(0,0,0)
    with pytest.raises(SingularField):s.electric_field_at((A,0,0))
    with pytest.raises(SingularField):s.potential_at((0,0,D))
    with pytest.raises(SingularField):s.electric_field_at((0,0,D))


def test_exact_density_integrates_to_image_charge_and_is_nearer_more_negative():
    s=ExternalGroundedSphereBackend().solve(p())
    near=s.surface_charge_density('outer',0)
    far=s.surface_charge_density('outer',pi)
    assert near<far<0
    assert abs(near/far)>10
    n=1800
    h=pi/n
    surface_total=2*pi*A*A*h*sum(
        s.surface_charge_density('outer',(i+.5)*h)*sin((i+.5)*h) for i in range(n))
    assert surface_total==pytest.approx(s.conductor_results[0].net_charge_c,rel=2e-6)
    with pytest.raises(ValueError):s.surface_charge_density('inner',0)
    with pytest.raises(ValueError):s.surface_charge_density('outer',-1)
    with pytest.raises(ValueError):s.surface_charge_density('outer',float('nan'))
    with pytest.raises(ValueError):s.surface_charge_density_at('outer',(0,0,A+.01))


def test_exact_e_negative_gradient_and_image_force_negative_energy_derivative():
    s=ExternalGroundedSphereBackend().solve(p())
    for x in ((.15,0,.04),(.22,.07,-.08),(.3,0,.1)):
        step=2e-6
        field=s.electric_field_at(x)
        gradient=tuple((s.potential_at(tuple(x[k]+(step if k==i else 0) for k in range(3)))-
                        s.potential_at(tuple(x[k]-(step if k==i else 0) for k in range(3))))/
                       (2*step) for i in range(3))
        assert field==pytest.approx(tuple(-a for a in gradient),rel=2e-4,abs=1e-3)
    k=1/(4*pi*EPSILON_0)
    exact_force=-k*Q*Q*A*D/(D*D-A*A)**2
    assert s.induced_force_on_charge_n==(0,0,pytest.approx(exact_force))
    dh=1e-5
    u_plus=ExternalGroundedSphereBackend().solve(p(distance=D+dh)).interaction_energy_j
    u_minus=ExternalGroundedSphereBackend().solve(p(distance=D-dh)).interaction_energy_j
    assert -(u_plus-u_minus)/(2*dh)==pytest.approx(exact_force,rel=3e-8)
    assert s.interaction_energy_j<0


def test_sign_flip_and_translation_rotation_covariance():
    up=ExternalGroundedSphereBackend().solve(p())
    neg=ExternalGroundedSphereBackend().solve(p(charge=-Q))
    assert neg.conductor_results[0].net_charge_c==-up.conductor_results[0].net_charge_c
    for point in ((.13,0,.04),(.25,.09,-.13)):
        assert neg.potential_at(point)==pytest.approx(-up.potential_at(point))
        assert neg.electric_field_at(point)==pytest.approx(tuple(-v for v in up.electric_field_at(point)))
    assert neg.induced_force_on_charge_n==pytest.approx(up.induced_force_on_charge_n)
    moved=ExternalGroundedSphereBackend().solve(p(sphere_center=(2.,-1.,.5),direction=(1,0,0)))
    pt=(.2,.06,.24)
    mapped=(2.+pt[2],-1.+pt[1],.5+pt[0])
    assert moved.potential_at(mapped)==pytest.approx(up.potential_at(pt))
    old=up.electric_field_at(pt)
    assert moved.electric_field_at(mapped)==pytest.approx((old[2],old[1],old[0]))


def test_strict_unsupported_physics_never_silently_fallback():
    backends=(ExternalGroundedSphereBackend(),Femm5ExternalSphereBackend())
    bad=(
        ElectrostaticProblem(),
        ElectrostaticProblem((PointCharge(Q,(0,0,A)),),(SolidSphere(A,Grounded()),)),
        ElectrostaticProblem((PointCharge(Q,(0,0,A*.5)),),(SolidSphere(A,Grounded()),)),
        ElectrostaticProblem((PointCharge(Q,(0,0,D)),),(SphericalShell(.10,A,Grounded()),)),
        ElectrostaticProblem((PointCharge(Q,(0,0,D)),PointCharge(-Q,(0,0,-D))),
                             (SolidSphere(A,Grounded()),)),
        ElectrostaticProblem((PointCharge(0,(0,0,D)),),(SolidSphere(A,Grounded()),)),
    )
    for backend in backends:
        for problem in bad:
            assert not backend.supports(problem)
            with pytest.raises(UnsupportedProblem):backend.solve(problem)
    # M7 remains grounded-only analytically; M8 extends the *numerical*
    # compiler to accept a genuinely isolated prescribed-net-charge sphere.
    isolated=ElectrostaticProblem((PointCharge(Q,(0,0,D)),),
                                  (SolidSphere(A,FixedNetCharge(0)),))
    assert not ExternalGroundedSphereBackend().supports(isolated)
    assert Femm5ExternalSphereBackend().supports(isolated)
    reserved=ElectrostaticProblem((PointCharge(Q,(0,0,D)),),
                 (SolidSphere(A,Grounded(),name=_SOURCE_NAME),))
    assert ExternalGroundedSphereBackend().supports(reserved)
    assert not Femm5ExternalSphereBackend().supports(reserved)
    with pytest.raises(UnsupportedProblem):Femm5ExternalSphereBackend().solve(reserved)


def test_femm_command_sequence_physical_identity_and_resource_cleanup():
    fake=FakeClient()
    fake.eo_getconductorproperties=lambda name: (35.,Q) if name==_SOURCE_NAME else (12.,2e-9)
    session=FakeSession(fake)
    # The fake reports arbitrary 12V,2nC on the grounded conductor;
    # the adapter reads this rather than substituting image-charge values.
    b=Femm5ExternalSphereBackend(session_factory=lambda:session)
    with b.solve(p()) as sol:
        assert sol.sphere.name=='metal'
        assert sol.conductor_results[0].net_charge_c==2e-9
        assert sol.conductor_results[0].potential_v==12
        assert sol.numerical_source_charge_c==1e-9
        assert sol.source_radius_m>0
        assert sol.symmetry_axis==(0,0,1)
        assert sol.potential_at((0,0,.18))==12
        assert sol.electric_field_at((.2,0,0))==(4,0,6)
        with pytest.raises(UnsupportedProblem):sol.potential_at((0,0,D))
        with pytest.raises(SingularField):sol.electric_field_at((0,0,A))
        assert sol.surface_charge_density('outer',0)==pytest.approx(6*EPSILON_0)
        with pytest.raises(ValueError):sol.surface_charge_density('inner',.1)
        assert len(sol.numerical_sources)==1
    assert session.opens==session.closes==1
    assert calls(fake,'ei_addconductorprop')==[('metal',0,0,1),(_SOURCE_NAME,0,Q,0)]
    assert calls(fake,'ei_addpointprop')==[]
    assert len(calls(fake,'ei_addarc'))==6
    assert len(calls(fake,'ei_setarcsegmentprop'))==6
    assert len(calls(fake,'ei_addblocklabel'))==3
    assert calls(fake,'newdocument')==[(1,)]
    assert calls(fake,'ei_probdef')==[('meters','axi',1e-9)]
    assert calls(fake,'ei_analyze')==[(1,)]
    assert calls(fake,'ei_loadsolution')==[()]
    assert calls(fake,'ei_saveas')[0][0].endswith('external_sphere.fee')
    with pytest.raises(RuntimeError):sol.potential_at((.3,0,0))


def test_femm_cleanup_on_command_failure():
    from conductor_sim.femm5 import Femm5ExecutionError
    fake=FakeClient(fail_on='ei_analyze');session=FakeSession(fake)
    with pytest.raises(Femm5ExecutionError,match='synthetic FEMM fault'):
        Femm5ExternalSphereBackend(session_factory=lambda:session).solve(p())
    assert session.closes==1


def test_viewer_uses_solid_geometry_with_only_outer_profile():
    problem=p()
    exact=ExternalGroundedSphereBackend()
    frame=sample_sweep([('external source',problem)],exact,
                       plane=SliceSpec.meridian(problem,extent_m=.6,resolution=21),
                       line_steps=140)['frames'][0]
    assert len(frame['profiles'])==1
    assert frame['profiles'][0]['surface']=='outer'
    assert frame['geometry'][0]['inner_radius_m']==0
    assert frame['geometry'][0]['outer_radius_m']==A
    assert frame['conductor_results'][0]['net_charge_c']==pytest.approx(-Q*A/D)
    assert frame['density_method'].startswith('exact Kelvin')
    assert all(region in ('metal','vacuum','source-mask','surface')
               for row in frame['grid']['region'] for region in row)


def test_pdf_external_grounded_sphere_blocker_is_frozen_historical_status():
    """M7 closes the physics gap, but earlier M6a benchmark scores are historical."""
    from conductor_sim.pdf_benchmark import load_case
    old=load_case('p61-external-grounded-sphere')
    assert old.page==61 and not old.runnable
    assert 'Historical M6a baseline' in old.blocker
    assert 'M7 now implements' in old.note
    assert ExternalGroundedSphereBackend().supports(p())
    assert Femm5ExternalSphereBackend().supports(p())
