"""M11: multi-source grounded-plane analytical/numerical interface contracts.

Fake FEMM clients validate geometry commands and independent measurement; real
numerical accuracy is tested separately in test_m11_live.py.
"""
from math import hypot, sqrt

import pytest

from conductor_sim import (ConductingPlane, ElectrostaticProblem, Femm5PlaneBackend,
                           Femm5Settings, GroundedPlaneBackend, PointCharge)
from conductor_sim.femm5_plane import _SOURCE_NAME, _layout
from conductor_sim.solver import UnsupportedProblem
from conductor_sim.visual_adapters import context_for_solution
from conductor_sim.visualization import SliceSpec, sample_sweep
from test_femm5_adapter import FakeClient, FakeSession, calls

Q=1e-9


def problem(heights=(.12,.26),charges=(Q,-.5*Q),origin=(0.,0.,0.),normal=(0.,0.,1.),transverse=(0.,0.,0.)):
    sources=tuple(PointCharge(q, tuple(origin[i]+h*normal[i]+transverse[i] for i in range(3)))
                  for h,q in zip(heights,charges))
    return ElectrostaticProblem(sources,(ConductingPlane(point_m=origin,normal=normal,
                                                        name='ground'),))


def test_two_axial_sources_reuse_existing_models_and_exact_solver():
    p=problem()
    ref=GroundedPlaneBackend().solve(p)
    backend=Femm5PlaneBackend()
    assert backend.supports(p)
    assert ref.conductor_results[0].net_charge_c==pytest.approx(-.5e-9)
    assert ref.potential_at((0,0,0))==0
    R,foot,axis,cores=_layout(p,backend.settings)
    assert foot==(0,0,0) and axis==(0.,0.,1.)
    assert len(cores)==2 and {c.prescribed_charge_c for c in cores}=={Q,-.5*Q}
    assert [c.name for c in cores]==[_SOURCE_NAME,_SOURCE_NAME+'_1']
    assert all(c.radius_m>0 for c in cores)
    assert cores[0].axial_offset_m+cores[0].radius_m < cores[1].axial_offset_m-cores[1].radius_m
    assert R>cores[1].axial_offset_m+cores[1].radius_m


def test_femm_command_contract_preserves_each_signed_source():
    client=FakeClient()
    answers={'ground':(0.,-4.3e-10),_SOURCE_NAME:(18.,.999e-9),
             _SOURCE_NAME+'_1':(-9.,-.499e-9)}
    client.eo_getconductorproperties=lambda name: answers[name]
    session=FakeSession(client)
    p=problem()
    with Femm5PlaneBackend(session_factory=lambda:session).solve(p) as sol:
        assert sol.conductor_results[0].net_charge_c==-4.3e-10
        assert sol.numerical_source_charge_c is None  # not ambiguous multi-core scalar
        assert sol.source_radius_m is None
        assert sol.source_position_m is None
        assert [s.measured_charge_c for s in sol.numerical_sources]==pytest.approx([.999e-9,-.499e-9])
        assert [s.axial_offset_m for s in sol.numerical_sources]==pytest.approx([.12,.26])
        for s in sol.numerical_sources:
            with pytest.raises(UnsupportedProblem):sol.potential_at(s.position_m)
        assert sol.potential_at((0,0,-.03))==0
        assert sol.surface_charge_density('outer',0)==pytest.approx(6*8.8541878128e-12)
        ctx=context_for_solution(p,sol)
        assert len(ctx.masks)==2
        assert 'truncated conductor segment' in ctx.conductor_results_method
        assert len(calls(client,'ei_setarcsegmentprop'))==5
        assert len(calls(client,'ei_addarc'))==5
    assert session.opens==session.closes==1
    assert calls(client,'ei_addconductorprop')==[
        ('ground',0,0,1),(_SOURCE_NAME,0,Q,0),(_SOURCE_NAME+'_1',0,-.5*Q,0)]
    assert all(x[-1]=='ground' for x in calls(client,'ei_setsegmentprop'))
    assert len(calls(client,'ei_addblocklabel'))==4  # 2 cores + 2 vacuum regions


def test_m11_single_core_backward_compatible_and_grouped_sources():
    p=problem(heights=(.12,.12,.26),charges=(Q,-Q,.4e-9))
    assert Femm5PlaneBackend().supports(p)
    client=FakeClient()
    client.eo_getconductorproperties=lambda name: (0.,-.39e-9) if name=='ground' else (5.,.4e-9)
    with Femm5PlaneBackend(session_factory=lambda:FakeSession(client)).solve(p) as sol:
        assert len(sol.numerical_sources)==1
        assert sol.numerical_source_charge_c==pytest.approx(.4e-9)
        assert sol.source_position_m==(0.,0.,.26)
        assert calls(client,'ei_addconductorprop')==[
            ('ground',0,0,1),(_SOURCE_NAME,0,.4e-9,0)]


def test_multi_source_plane_viewer_reuses_single_shared_geometry():
    p=problem()
    states=[('mixed',p),('both positive',problem(charges=(Q,.5*Q)))]
    bundle=sample_sweep(states,GroundedPlaneBackend(),resolution=19,line_steps=120)
    assert len(bundle['frames'])==2
    assert len(bundle['frames'][0]['profiles'])==1
    assert bundle['frames'][0]['geometry'][0]['kind']=='plane'
    assert bundle['frames'][0]['conductor_results'][0]['net_charge_c']==pytest.approx(-.5e-9)
    assert bundle['frames'][1]['conductor_results'][0]['net_charge_c']==pytest.approx(-1.5e-9)
    for frame in bundle['frames']:
        assert len(frame['charges'])==2
        assert 'coordinate_m' in frame['profiles'][0]


def test_axis_constraints_reject_noncollinear_sites_before_starting_session():
    p=problem()
    src=list(p.charges)
    src[1]=PointCharge(-.5e-9,(.001,0.,.26))
    wrong=ElectrostaticProblem(tuple(src),p.conductors)
    assert GroundedPlaneBackend().supports(wrong)
    started=[]
    num=Femm5PlaneBackend(session_factory=lambda:started.append(1))
    assert not num.supports(wrong)
    with pytest.raises(UnsupportedProblem,match='normal-line'):
        num.solve(wrong)
    assert not started
    # All points may be collinear in 3D on a *tilted* line yet are not
    # rotationally symmetric when the plane is perpendicular to z.
    tilted=ElectrostaticProblem((PointCharge(Q,(0.,0.,.12)),
                                PointCharge(-Q,(.05,0.,.26))),p.conductors)
    assert not num.supports(tilted)
    with pytest.raises(UnsupportedProblem):num.solve(tilted)
    assert not started


def test_translated_rotated_plane_and_footpoint_alignment():
    p=problem(origin=(2.,-1.,.5),normal=(1.,0.,0.),transverse=(0.,.04,-.03))
    R,foot,axis,cores=_layout(p,Femm5Settings())
    assert axis==(1.,0.,0.)
    assert foot==pytest.approx((2.,-.96,.47))
    assert [c.axial_offset_m for c in cores]==pytest.approx([.12,.26])
    # The authored plane point is NOT required to be the source footpoint.
    assert Femm5PlaneBackend().supports(p)


def test_site_limit_and_inadequate_clearance_fail_before_femm():
    plane=ConductingPlane(name='ground')
    many=ElectrostaticProblem(tuple(PointCharge(Q,(0.,0.,.1+.02*i)) for i in range(17)),(plane,))
    assert GroundedPlaneBackend().supports(many)
    assert not Femm5PlaneBackend().supports(many)
    started=[]
    num=Femm5PlaneBackend(session_factory=lambda:started.append(1))
    with pytest.raises(UnsupportedProblem,match='1–16'):
        num.solve(many)
    assert not started
    near=problem(heights=(.2,.2+1e-320)) # rounded to the SAME site; not a problem
    assert Femm5PlaneBackend().supports(near)


def test_distinct_fixed_core_sizes_do_not_overlap_and_invalid_sites_rejected():
    p=problem(heights=(.12,.1200000001),charges=(Q,-.5*Q))
    R,foot,axis,cores=_layout(p,Femm5Settings())
    assert cores[0].radius_m+cores[1].radius_m < cores[1].axial_offset_m-cores[0].axial_offset_m
    bad=problem(heights=(.12,0.),charges=(Q,-Q))
    assert not GroundedPlaneBackend().supports(bad)
    assert not Femm5PlaneBackend().supports(bad)
    big=problem(heights=(.12,1e308),charges=(Q,-.5*Q))
    assert not Femm5PlaneBackend().supports(big)
