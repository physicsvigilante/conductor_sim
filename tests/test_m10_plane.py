"""Exact plane images, shape-aware sampling, and mock FEMM command contracts."""
from math import pi, hypot, sqrt

import pytest

from conductor_sim import (ConductingPlane, ElectrostaticProblem, Grounded,
                           GroundedPlaneBackend, Femm5PlaneBackend,
                           Femm5Settings, FixedNetCharge, PointCharge, SolidSphere)
from conductor_sim.geometry import PlaneBoundary, contains_metal, physical_boundaries
from conductor_sim.model import EPSILON_0
from conductor_sim.solver import SingularField, UnsupportedProblem
from conductor_sim.visualization import (SliceSpec, sample_sweep, trace_field_line)
from conductor_sim.visual_adapters import context_for_solution
from conductor_sim.femm5_plane import _SOURCE_NAME
from test_femm5_adapter import FakeClient,FakeSession,calls

Q=1e-9
H=.15

def p(q=Q,height=H,normal=(0.,0.,1.),origin=(0.,0.,0.),offset=(0.,0.,0.)):
    n=hypot(*normal)
    unit=tuple(x/n for x in normal)
    position=tuple(origin[i]+height*unit[i]+offset[i] for i in range(3))
    return ElectrostaticProblem((PointCharge(q,position),),
             (ConductingPlane(point_m=origin,normal=normal,constraint=Grounded(),name='ground'),))


def test_plane_component_normalization_and_strict_constraint():
    c=ConductingPlane(normal=(0,0,3))
    assert c.normal==(0.,0.,1.)
    assert c.signed_distance_m((0,0,.5))==pytest.approx(.5)
    assert c.signed_distance_m((0,0,-.5))==pytest.approx(-.5)
    assert contains_metal(c,(1.,1.,-.1))
    assert not contains_metal(c,(1.,1.,.1))
    assert isinstance(physical_boundaries(c)[0],PlaneBoundary)
    for value in [(0.,0.,0.),(float('nan'),0.,1.),(float('inf'),0.,1.)]:
        with pytest.raises(ValueError):ConductingPlane(normal=value)
    with pytest.raises(TypeError):ConductingPlane(constraint=FixedNetCharge(0))
    with pytest.raises(ValueError):ConductingPlane(name='<None>')
    with pytest.raises(ValueError):ConductingPlane(point_m=(0.,0.))
    with pytest.raises(ValueError):ConductingPlane(name='bad"name')


def test_grounded_plane_exact_image_condition_field_and_force():
    s=GroundedPlaneBackend().solve(p())
    k=1/(4*pi*EPSILON_0)
    assert s.conductor_results[0].net_charge_c==-Q
    for x,y in ((0,0),(.12,0),(.1,-.2)):
        assert s.potential_at((x,y,0.))==0
        assert s.potential_at((x,y,-.2))==0
        assert s.electric_field_at((x,y,-.2))==(0.,0.,0.)
        with pytest.raises(SingularField):s.electric_field_at((x,y,0.))
    x=(.06,0.,.07)
    exact=k*Q*(1/hypot(.06,.07-H)-1/hypot(.06,.07+H))
    assert s.potential_at(x)==pytest.approx(exact)
    step=1e-6
    f=s.electric_field_at(x)
    for i in range(3):
        plus=tuple(x[j]+(step if i==j else 0) for j in range(3))
        minus=tuple(x[j]-(step if i==j else 0) for j in range(3))
        assert f[i]==pytest.approx(-(s.potential_at(plus)-s.potential_at(minus))/(2*step),rel=1e-7,abs=1e-4)
    expected_force=-k*Q*Q/(4*H*H)
    assert s.induced_force_on_charge_n==pytest.approx((0.,0.,expected_force))
    assert s.interaction_energy_j==pytest.approx(-k*Q*Q/(4*H))
    dh=1e-5
    up=GroundedPlaneBackend().solve(p(height=H+dh)).interaction_energy_j
    down=GroundedPlaneBackend().solve(p(height=H-dh)).interaction_energy_j
    assert -(up-down)/(2*dh)==pytest.approx(expected_force,rel=1e-8)
    with pytest.raises(SingularField):s.potential_at((0.,0.,H))


def test_exact_surface_density_and_charge_flux_integral():
    s=GroundedPlaneBackend().solve(p())
    assert s.surface_charge_density('outer',0)==pytest.approx(-Q/(2*pi*H*H))
    for radius in (0.,.01,.1,.8):
        assert s.surface_charge_density('outer',radius)==pytest.approx(
            s.surface_charge_density_at('outer',(radius,0.,0.)))
    n=2000;max_r=30*H;delta=max_r/n
    total=sum(2*pi*(i+.5)*delta*s.surface_charge_density('outer',(i+.5)*delta)*delta
              for i in range(n))
    expected=-Q*(1-H/sqrt(H*H+max_r*max_r))
    assert total==pytest.approx(expected,rel=2e-5)
    assert expected==pytest.approx(-Q,rel=.034)
    with pytest.raises(ValueError):s.surface_charge_density_at('inner',(0,0,0))
    with pytest.raises(ValueError):s.surface_charge_density_at('outer',(0,0,.001))
    with pytest.raises(ValueError):s.surface_charge_density('outer',-1)


def test_multiple_noncollinear_3d_sources_are_analytically_superposed():
    plane=ConductingPlane()
    q1=PointCharge(Q,(.05,.03,.14));q2=PointCharge(-Q/2,(-.04,.07,.21))
    prob=ElectrostaticProblem((q1,q2),(plane,))
    backend=GroundedPlaneBackend();s=backend.solve(prob)
    assert backend.supports(prob)
    assert s.conductor_results[0].net_charge_c==pytest.approx(-Q/2)
    assert all(s.potential_at((x,y,0.))==0 for x,y in ((0.,0.),(.15,0.),(0.,-.18)))
    for i,source in enumerate(s.sources):
        step=1e-5
        # Moving one source also moves its image. Energy derivative must give
        # direct inter-source Coulomb force plus conductor-induced force.
        for j in range(3):
            displaced=[]
            for sign in (1,-1):
                charges=list(s.sources)
                pt=list(source.position_m);pt[j]+=sign*step
                charges[i]=PointCharge(source.charge_c,tuple(pt))
                displaced.append(backend.solve(ElectrostaticProblem(tuple(charges),(plane,))).interaction_energy_j)
            force=s.force_on_charge_n(i)
            assert force[j]==pytest.approx(-(displaced[0]-displaced[1])/(2*step),rel=2e-7,abs=1e-10)
    assert abs(s.surface_charge_density_at('outer',(0,0,0)))>0
    with pytest.raises(UnsupportedProblem):s.surface_charge_density('outer',.1)


def test_arbitrary_orientation_and_translated_plane():
    up=GroundedPlaneBackend().solve(p())
    rotated=GroundedPlaneBackend().solve(p(normal=(1,0,0),origin=(2.,-1.,.5)))
    x=(.05,.01,.07)
    tx=(2.+x[2],-1.+x[1],.5+x[0])
    assert rotated.potential_at(tx)==pytest.approx(up.potential_at(x))
    f=up.electric_field_at(x)
    assert rotated.electric_field_at(tx)==pytest.approx((f[2],f[1],f[0]))
    assert rotated.induced_force_on_charge_n==pytest.approx((up.induced_force_on_charge_n[2],0.,0.))
    # A plane's authored point need not coincide with the projection of q.
    shifted=GroundedPlaneBackend().solve(p(offset=(.02,-.03,0.)))
    assert shifted.surface_charge_density('outer',.04)==pytest.approx(up.surface_charge_density('outer',.04))


def test_capability_checks_do_not_fallback():
    analytic=GroundedPlaneBackend();numeric=Femm5PlaneBackend()
    good=p()
    assert analytic.supports(good) and numeric.supports(good)
    multi=ElectrostaticProblem((PointCharge(Q,(0,0,H)),PointCharge(Q/2,(.01,.02,H))),
                               (ConductingPlane(),))
    assert analytic.supports(multi) and not numeric.supports(multi)
    bad=[ElectrostaticProblem(),
         ElectrostaticProblem((PointCharge(Q,(0,0,0)),),(ConductingPlane(),)),
         ElectrostaticProblem((PointCharge(Q,(0,0,-.1)),),(ConductingPlane(),)),
         ElectrostaticProblem((PointCharge(Q,(0,0,H)),),(SolidSphere(.1,Grounded()),)),
         multi]
    for problem in bad[:4]:
        assert not analytic.supports(problem)
        with pytest.raises(UnsupportedProblem):analytic.solve(problem)
    for problem in bad:
        assert not numeric.supports(problem)
        with pytest.raises(UnsupportedProblem):numeric.solve(problem)


def test_plane_exact_segment_crossing_and_visual_trace_termination():
    plane=ConductingPlane();boundary=physical_boundaries(plane)[0]
    assert boundary.first_hit((0,0,.5),(0,0,-.5))==pytest.approx(.5)
    assert boundary.first_hit((0,0,.5),(0,0,.4)) is None
    assert boundary.first_hit((0,0,-.5),(0,0,.5))==pytest.approx(.5)
    prob=p();solution=GroundedPlaneBackend().solve(prob)
    slice_spec=SliceSpec.meridian(prob,resolution=21)
    assert slice_spec.v==plane.normal
    context=context_for_solution(prob,solution)
    assert context.region((0,0,-.1))=='metal'
    assert context.region((0,0,.1))=='vacuum'
    trace=trace_field_line(context,slice_spec,(0.,.055),step_m=.01,max_steps=200)
    assert trace['termination']=='surface'
    assert trace['points_m'][-1][1]>0
    assert trace['points_m'][-1][1]<.01


def test_planar_renderer_sweep_with_stored_signed_distance_profiles():
    b=GroundedPlaneBackend()
    problems=[('far',p(height=.20)),('near',p(height=.10))]
    bundle=sample_sweep(problems,b,resolution=21,line_steps=160)
    assert len(bundle['frames'])==2
    for f in bundle['frames']:
        assert f['geometry'][0]['kind']=='plane'
        assert f['geometry'][0]['visible']
        assert f['geometry'][0]['normal_xy'][1]>0
        assert any('surface'==r for row in f['grid']['region'] for r in row)
        assert any('metal'==r for row in f['grid']['region'] for r in row)
        assert len(f['profiles'])==1
        assert len(f['profiles'][0]['coordinate_m'])==91
        assert all(v < 0 for v in f['profiles'][0]['sigma_c_per_m2'])
        assert len(f['field_lines'])>1
    from conductor_sim.viewer import render_html
    html=render_html(bundle)
    assert 's.kind===\'plane\'' in html
    assert 'series.coordinate_m' in html


def test_femm_plane_compiler_command_sequence_and_measured_results():
    client=FakeClient()
    client.eo_getconductorproperties=lambda name: ((0.,-1.2e-9)
                            if name=='ground' else (8.,9.8e-10))
    session=FakeSession(client)
    problem=p()
    backend=Femm5PlaneBackend(session_factory=lambda:session)
    with backend.solve(problem) as sol:
        assert sol.conductor_results[0].potential_v==0
        assert sol.conductor_results[0].net_charge_c==-1.2e-9
        assert sol.numerical_source_charge_c==9.8e-10
        assert sol._origin_m==(0.,0.,0.)
        assert sol.symmetry_axis==(0.,0.,1.)
        assert sol.potential_at((.02,0,.12))==12
        assert sol.potential_at((0.,0.,-.01))==0
        assert sol.electric_field_at((0.,0.,-.01))==(0,0,0)
        assert sol.electric_field_at((.02,0,.12))==pytest.approx((4.,0.,6.))
        assert sol.surface_charge_density('outer',.02)==pytest.approx(6*EPSILON_0)
        with pytest.raises(SingularField):sol.electric_field_at((.01,0,0.))
        with pytest.raises(UnsupportedProblem):sol.potential_at((0.,0.,H))
        with pytest.raises(UnsupportedProblem):sol.potential_at((2.,0.,0.))
    assert session.opens==session.closes==1
    assert calls(client,'ei_addconductorprop')==[('ground',0,0,1),(_SOURCE_NAME,0,Q,0)]
    assert len(calls(client,'ei_addsegment'))==6
    assert len(calls(client,'ei_setsegmentprop'))==3
    assert all(args[-1]=='ground' for args in calls(client,'ei_setsegmentprop'))
    assert len(calls(client,'ei_addarc'))==3
    assert len(calls(client,'ei_setarcsegmentprop'))==3
    assert calls(client,'ei_addboundprop')[0][0]=='open_dipole'
    assert calls(client,'ei_saveas')[0][0].endswith('grounded_plane.fee')
    with pytest.raises(RuntimeError):sol.potential_at((0,0,.03))


def test_femm_plane_translated_source_projection_and_cleanup_on_failure():
    # The point above the plane need not be aligned with the author's
    # arbitrary plane point. FEMM recenters the coordinate frame exactly.
    problem=p(normal=(1,0,0),origin=(2.,-1.,.5),offset=(0.,.04,-.03))
    client=FakeClient()
    client.eo_getconductorproperties=lambda name: (0.,Q)
    sess=FakeSession(client)
    with Femm5PlaneBackend(session_factory=lambda:sess).solve(problem) as sol:
        assert sol._origin_m==pytest.approx((2.,-.96,.47))
        assert sol.symmetry_axis==(1.,0.,0.)
    assert sess.closes==1
    bad=FakeClient(fail_on='ei_analyze')
    s=FakeSession(bad)
    from conductor_sim.femm5 import Femm5ExecutionError
    with pytest.raises(Femm5ExecutionError,match='synthetic FEMM fault'):
        Femm5PlaneBackend(session_factory=lambda:s).solve(p())
    assert s.closes==1


def test_femm_planar_metadata_never_labels_charge_as_shell_gauss_inference():
    client=FakeClient()
    client.eo_getconductorproperties=lambda name: (0.,-8.75e-10) if name=='ground' else (4.,Q)
    session=FakeSession(client)
    problem=p()
    with Femm5PlaneBackend(session_factory=lambda:session).solve(problem) as sol:
        context=context_for_solution(problem,sol)
        assert any('finite modeled disk' in note for note in context.notes)
        assert not any('Shell inner/outer' in note for note in context.notes)
        assert 'truncated conductor segment' in context.conductor_results_method
