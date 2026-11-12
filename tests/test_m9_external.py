"""M9 exact multi-image physics, capability boundaries, and fake FEMM API tests."""
from math import acos, cos, pi, sin

import pytest

from conductor_sim import (ElectrostaticProblem, ExternalSphereBackend,
                           ExternalMultiSphereBackend, ExternalGroundedSphereBackend,
                           ExternalIsolatedSphereBackend, Femm5ExternalSphereBackend,
                           FixedNetCharge, Grounded, PointCharge, SolidSphere)
from conductor_sim.external_multi import active_external_sites
from conductor_sim.femm5_external import _SOURCE_NAME
from conductor_sim.model import EPSILON_0
from conductor_sim.solver import SingularField, UnsupportedProblem
from conductor_sim.visualization import SliceSpec, sample_sweep
from test_femm5_adapter import FakeClient, FakeSession, calls

A=.12
q1=1e-9
q2=-.5e-9

def problem(constraint=Grounded(), sites=((q1,(0.,0.,.27)),(q2,(0.,0.,-.32))),
            center=(0.,0.,0.), name='sphere'):
    return ElectrostaticProblem(
        charges=tuple(PointCharge(q, tuple(center[i]+pos[i] for i in range(3))) for q,pos in sites),
        conductors=(SolidSphere(A,constraint,center_m=center,name=name),))


def test_exact_two_sites_grounded_kelvin_images_additive_superposition():
    p=problem()
    b=ExternalMultiSphereBackend()
    assert b.supports(p)
    s=b.solve(p)
    assert isinstance(ExternalSphereBackend().solve(p),type(s))
    assert len(s.image_sources)==2
    image_total=sum(img.charge_c for img in s.image_sources)
    assert image_total==pytest.approx(-q1*A/.27-q2*A/.32)
    assert s.conductor_results[0].net_charge_c==pytest.approx(image_total)
    assert s.conductor_results[0].potential_v==0
    for point in ((.22,0,.08),(.25,0,-.1),(.18,.2,.03),(.45,.09,.1)):
        total=sum(ExternalSphereBackend().solve(problem(
            Grounded(),((q,pos),))).potential_at(point) for q,pos in ((q1,(0,0,.27)),(q2,(0,0,-.32))))
        assert s.potential_at(point)==pytest.approx(total,abs=1e-9)
    for theta in (0.,.3,.7,pi/2,2.7,pi):
        point=(A*sin(theta),0.,A*cos(theta))
        assert s.potential_at(point)==pytest.approx(0,abs=1e-9)
        # Polar theta is referenced to the farthest source (at -z here).
        oriented=(A*sin(theta),0.,-A*cos(theta))
        assert s.surface_charge_density_at('outer',oriented)==pytest.approx(
            s.surface_charge_density('outer',theta),rel=1e-12)
    assert s.potential_at((0,0,.03))==0
    assert s.electric_field_at((0,0,.03))==(0,0,0)
    with pytest.raises(SingularField):s.electric_field_at((0,0,A))
    with pytest.raises(SingularField):s.potential_at((0,0,.27))
    with pytest.raises(ValueError):s.surface_charge_density_at('outer',(.4,0,0))


def test_exact_isolated_neutral_and_charged_metal_are_equipotential_and_net_Q():
    for Q in (0.,1.3e-9,-.8e-9):
        s=ExternalMultiSphereBackend().solve(problem(FixedNetCharge(Q)))
        qimage=sum(p.charge_c for p in s.image_sources)
        qcenter=Q-qimage
        k=1/(4*pi*EPSILON_0)
        expected=k*qcenter/A
        assert s.conductor_results[0].net_charge_c==Q
        assert s.conductor_results[0].potential_v==pytest.approx(expected)
        for theta in (0.,.23,1.25,2.42,pi):
            pt=(A*sin(theta),0.,A*cos(theta))
            assert s.potential_at(pt)==pytest.approx(expected,abs=2e-10)
        assert s.potential_at((0,0,0))==pytest.approx(expected)
        assert s.electric_field_at((0,0,0))==(0,0,0)
        # The exact sigma integrates to the PRESCRIBED Q, not the grounded result.
        n=1800;h=pi/n
        total=2*pi*A*A*h*sum(s.surface_charge_density('outer',(i+.5)*h)*sin((i+.5)*h) for i in range(n))
        assert total==pytest.approx(Q,abs=5e-15)


def test_noncollinear_arbitrary_3d_exact_but_numerical_FEMM_refuses():
    p=problem(FixedNetCharge(0.),sites=((1e-9,(0.,0.,.27)),(-.4e-9,(.24,0,0)),(.3e-9,(0.,-.3,0))))
    analytic=ExternalSphereBackend()
    assert analytic.supports(p)
    assert ExternalMultiSphereBackend().supports(p)
    assert not Femm5ExternalSphereBackend().supports(p)
    with pytest.raises(UnsupportedProblem,match='non-collinear'):
        Femm5ExternalSphereBackend().solve(p)
    result=analytic.solve(p)
    assert result.conductor_results[0].net_charge_c==0
    assert result.potential_at((0.,0.,0.))==pytest.approx(result.conductor_potential_v)
    for theta,phi in ((.2,.3),(.9,1.7),(2.2,3.6)):
        pt=(A*sin(theta)*cos(phi),A*sin(theta)*sin(phi),A*cos(theta))
        assert result.potential_at(pt)==pytest.approx(result.conductor_potential_v,abs=1e-9)
        assert result.surface_charge_density_at('outer',pt)!=0
    with pytest.raises(UnsupportedProblem,match='non-collinear'):
        result.surface_charge_density('outer',.5)
    # General 3D surface-area quadrature verifies net Q independent of axis.
    nt,np=90,180;dtheta=pi/nt;dphi=2*pi/np
    charge=0.
    for it in range(nt):
        theta=(it+.5)*dtheta
        ring=0.
        for ip in range(np):
            phi=(ip+.5)*dphi
            point=(A*sin(theta)*cos(phi), A*sin(theta)*sin(phi),A*cos(theta))
            ring+=result.surface_charge_density_at('outer',point)
        charge+=ring*sin(theta)
    charge*=A*A*dtheta*dphi
    assert charge==pytest.approx(0.,abs=1.3e-13)


def test_signed_sources_force_energy_derivative_includes_real_pair_force():
    for constraint in (Grounded(),FixedNetCharge(0.),FixedNetCharge(1.3e-9)):
        sol=ExternalSphereBackend().solve(problem(constraint))
        assert sol.force_on_charge_n(0)[2]!=sol.conductor_force_on_charge_n(0)[2]
        h=1e-5
        for i,site in enumerate(sol.sites):
            forces=sol.force_on_charge_n(i)
            def energy(displacement):
                charges=list(sol.sites)
                q=charges[i]
                charges[i]=PointCharge(q.charge_c,(q.position_m[0],q.position_m[1],q.position_m[2]+displacement))
                return ExternalSphereBackend().solve(ElectrostaticProblem(tuple(charges),problem(constraint).conductors)).interaction_energy_j
            fd=-(energy(h)-energy(-h))/(2*h)
            assert forces[2]==pytest.approx(fd,rel=4e-7,abs=2e-13)
        pt=(.25,.04,.08)
        e=sol.electric_field_at(pt)
        h=1e-6
        for idx in range(3):
            v1=sol.potential_at(tuple(pt[j]+(h if j==idx else 0) for j in range(3)))
            v0=sol.potential_at(tuple(pt[j]-(h if j==idx else 0) for j in range(3)))
            assert e[idx]==pytest.approx(-(v1-v0)/(2*h),rel=1e-4,abs=.01)


def test_combining_coincident_sites_and_charge_cancellation():
    p=problem(Grounded(),sites=((.6e-9,(0,0,.27)),(.4e-9,(0,0,.27)),(-.5e-9,(0,0,-.3))))
    assert len(active_external_sites(p))==2
    s=ExternalMultiSphereBackend().solve(p)
    assert len(s.sites)==2 and sorted(x.charge_c for x in s.sites)==pytest.approx(sorted((1e-9,-.5e-9)))
    gone=problem(Grounded(),sites=((1e-9,(0,0,.27)),(-1e-9,(0,0,.27))))
    assert not ExternalMultiSphereBackend().supports(gone)
    assert not Femm5ExternalSphereBackend().supports(gone)
    # No geometry can be licensed by cancelling an invalid input inside metal.
    illegal=problem(Grounded(),sites=((1e-9,(0,0,.05)),(-1e-9,(0,0,.05)),(1e-9,(0,0,.27))))
    assert not ExternalMultiSphereBackend().supports(illegal)


def test_femm_compiler_creates_independent_source_cores_and_measures_all_charges():
    fake=FakeClient()
    responses={f'{_SOURCE_NAME}_1':(-.4e-9),f'{_SOURCE_NAME}_2':(+1e-9)}
    def conductor_properties(name):
        return (23., responses[name]) if name in responses else (14.,.7e-9)
    fake.eo_getconductorproperties=conductor_properties
    session=FakeSession(fake)
    p=problem(FixedNetCharge(.7e-9))
    backend=Femm5ExternalSphereBackend(session_factory=lambda:session)
    assert backend.supports(p)
    with backend.solve(p) as sol:
        assert sol.conductor_results[0].potential_v==14.
        assert sol.conductor_results[0].net_charge_c==.7e-9
        assert len(sol.numerical_sources)==2
        assert tuple(round(s.axial_offset_m,9) for s in sol.numerical_sources)==(-.27,.32)
        assert sol.numerical_source_charge_c==pytest.approx(.6e-9)
        assert sol.source_radius_m is None
        for core in sol.numerical_sources:
            assert core.radius_m>0
            with pytest.raises(UnsupportedProblem):sol.potential_at(core.position_m)
        assert sol.potential_at((.2,0,0))==12
        assert sol.surface_charge_density('outer',.5)==pytest.approx(EPSILON_0*4*sin(.5)+EPSILON_0*6*cos(.5))
    assert session.opens==session.closes==1
    conductor_commands=calls(fake,'ei_addconductorprop')
    assert conductor_commands[0]==('sphere',0,.7e-9,0)
    assert len(conductor_commands)==3
    assert len([x for x in conductor_commands if x[0].startswith(_SOURCE_NAME)])==2
    assert len(calls(fake,'ei_addarc'))==8  # 2 per physical sphere, each core, outer boundary
    assert len(calls(fake,'ei_addblocklabel'))==4
    assert len(calls(fake,'ei_addpointprop'))==0
    assert calls(fake,'ei_saveas')[0][0].endswith('external_sphere.fee')


def test_femm_world_axis_can_have_sources_on_opposite_sides():
    fake=FakeClient()
    fake.eo_getconductorproperties=lambda name:(0.,-.1e-9) if name=='sphere' else (4.,-1e-9 if name.endswith('_1') else 1e-9)
    sess=FakeSession(fake)
    p=problem(Grounded(),sites=((1e-9,(.27,0,0)),(-1e-9,(-.27,0,0))),center=(2.,-1.,.3))
    with Femm5ExternalSphereBackend(session_factory=lambda:sess).solve(p) as sol:
        assert len(sol.numerical_sources)==2
        assert sol.symmetry_axis==(1.,0.,0.) or sol.symmetry_axis==(-1.,0.,0.)
        assert sorted(round(abs(c.axial_offset_m),6) for c in sol.numerical_sources)==[.27,.27]
        assert sol.numerical_source_charge_c==pytest.approx(0,abs=1e-20)
        assert sol.potential_at((2.,-.78,.3))==12
    assert sess.opens==sess.closes==1


def test_noncollinear_and_invalid_source_placement_do_not_open_FEMM():
    fake=FakeClient();session=FakeSession(fake)
    b=Femm5ExternalSphereBackend(session_factory=lambda:session)
    bad=(
       problem(sites=((1e-9,(0,0,.27)),(-.5e-9,(.27,0,0)))),
       problem(sites=((1e-9,(0,0,.27)),(-.5e-9,(0,0,.1)))),
       problem(sites=((1e-9,(0,0,.27)),(-.5e-9,(0,0,.12)))),
       ElectrostaticProblem((PointCharge(1e-9,(0,0,.27)),),(SolidSphere(A,FixedNetCharge(0),name=_SOURCE_NAME+'_1'),)),
    )
    for p in bad:
        assert not b.supports(p)
        with pytest.raises(UnsupportedProblem):b.solve(p)
    assert session.opens==0


def test_visualizer_sample_sweep_analytical_three_states_no_numerical_fallback():
    backend=ExternalSphereBackend()
    cases=[('grounded',problem(Grounded())),('neutral',problem(FixedNetCharge(0.))),('charged',problem(FixedNetCharge(1e-9)))]
    bundle=sample_sweep(cases,backend,plane=SliceSpec.meridian(cases[0][1],extent_m=.44,resolution=21),line_steps=180)
    assert len(bundle['frames'])==3
    for frame in bundle['frames']:
        assert len(frame['charges'])==2
        assert frame['conductor_results'][0]['name']=='sphere'
        assert len(frame['profiles'])==1
        assert frame['profiles'][0]['method'].startswith('exact Kelvin multi-image')
        assert frame['density_method'].startswith('exact Kelvin multi-image')
        assert frame['conductor_results_method'].startswith('Analytical')
    assert bundle['frames'][0]['conductor_results'][0]['potential_v']==0
    assert bundle['frames'][1]['conductor_results'][0]['net_charge_c']==0
