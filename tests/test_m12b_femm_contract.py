"""M12b deterministic compiler/geometry contracts (fake client; not real FEMM)."""
from math import hypot, pi

import pytest

from conductor_sim import (
    ConductingPlane, ElectrostaticProblem, FixedPotential, Grounded,
    Femm5ParallelPlateBackend, Femm5Settings, ParallelPlateBackend,
    PointCharge,
)
from conductor_sim.femm5_parallel import _layout, _SOURCE_PREFIX
from conductor_sim.geometry import CylinderSideBoundary
from conductor_sim.visualization import (
    CylindricalDomain, SamplingContext, SliceSpec, SourceMask,
    sample_sweep, trace_field_line,
)
from conductor_sim.visual_adapters import context_for_solution
from conductor_sim.femm5 import Femm5ExecutionError
from conductor_sim.solver import UnsupportedProblem, SingularField
from test_femm5_adapter import FakeClient, FakeSession, calls

D=.2
Q=1e-9


def problem(charges=(), low=Grounded(),high=Grounded(),*,shift=(0.,0.,0.),axis=(0.,0.,1.)):
    base=tuple(shift)
    top=tuple(base[i]+D*axis[i] for i in range(3))
    planes=(
        ConductingPlane(point_m=base,normal=axis,constraint=low,name="lower"),
        ConductingPlane(point_m=top,normal=tuple(-x for x in axis),constraint=high,name="upper"),
    )
    return ElectrostaticProblem(tuple(charges),planes)


def source(q,z,*,shift=(0.,0.,0.),axis=(0.,0.,1.),perp=(0.,0.,0.)):
    return PointCharge(q,tuple(shift[i]+z*axis[i]+perp[i] for i in range(3)))


def test_charge_free_fixed_bias_compiles_two_femm_electrodes_and_insulated_side():
    p=problem(low=FixedPotential(12.),high=FixedPotential(-6.))
    b=Femm5ParallelPlateBackend()
    assert b.supports(p)
    R,foot,axis,d,cores=_layout(p,Femm5Settings())
    assert R==pytest.approx(1.6) and d==D and axis==(0.,0.,1.)
    assert foot==(0.,0.,0.) and cores==()
    fake=FakeClient()
    fake.eo_getconductorproperties=lambda name: (12.,4e-9) if name=="lower" else (-6.,-4e-9)
    session=FakeSession(fake)
    with Femm5ParallelPlateBackend(session_factory=lambda:session).solve(p) as sol:
        assert len(sol.conductor_results)==2
        assert sol.conductor_results[0].net_charge_c==4e-9
        assert sol.conductor_results[1].net_charge_c==-4e-9
        assert not sol.numerical_sources
        assert sol.potential_at((0.,0.,-.1))==12.
        assert sol.potential_at((0.,0.,.3))==-6.
        assert sol.potential_at((.04,0.,.09))==12 # fake FEMM field
        assert sol.electric_field_at((.03,.04,.1))==pytest.approx((2.4,3.2,6.))
        assert sol.surface_charge_density('outer',0.,conductor_name="lower")==pytest.approx(6*p.permittivity_f_per_m)
        assert sol.surface_charge_density('outer',0.,conductor_name="upper")==pytest.approx(-6*p.permittivity_f_per_m)
        with pytest.raises(SingularField):
            sol.electric_field_at((0,0,0))
        with pytest.raises(UnsupportedProblem):
            sol.potential_at((R,0,D/2))
    assert session.opens==session.closes==1
    assert calls(fake,'ei_addconductorprop')==[
        ('lower',12.,0,1),('upper',-6.,0,1)]
    assert len(calls(fake,'ei_setsegmentprop'))==10
    assert all(s[-1] in ('lower','upper') for s in calls(fake,'ei_setsegmentprop'))
    # A separate vertical segment r=R has NO conductor or prescribed V.
    assert (R,0,R,D) in calls(fake,'ei_addsegment')
    assert calls(fake,'ei_addboundprop')==[]
    assert len(calls(fake,'ei_addblocklabel'))==1
    assert calls(fake,'ei_saveas')[0][0].endswith('parallel_plates.fee')
    with pytest.raises(RuntimeError):
        sol.potential_at((0.,0.,D/2))


def test_two_sources_keep_signed_charge_and_finite_core_identities():
    p=problem((source(Q,.06),source(-.5*Q,.14)))
    R,foot,axis,d,cores=_layout(p,Femm5Settings())
    assert len(cores)==2
    assert [s.name for s in cores]==[f'{_SOURCE_PREFIX}_0',f'{_SOURCE_PREFIX}_1']
    assert [s.prescribed_charge_c for s in cores]==[Q,-.5*Q]
    assert all(c.radius_m < min(c.axial_offset_m,d-c.axial_offset_m) for c in cores)
    fake=FakeClient()
    data={"lower":(0.,-.55e-9),"upper":(0.,.05e-9),
          cores[0].name:(21.,1.005e-9),cores[1].name:(-11.,-.498e-9)}
    fake.eo_getconductorproperties=lambda name:data[name]
    session=FakeSession(fake)
    with Femm5ParallelPlateBackend(session_factory=lambda:session).solve(p) as sol:
        assert [r.name for r in sol.conductor_results]==['lower','upper']
        assert [q.measured_charge_c for q in sol.numerical_sources]==pytest.approx([1.005e-9,-.498e-9])
        assert len(sol.numerical_sources)==2
        for c in sol.numerical_sources:
            with pytest.raises(UnsupportedProblem):
                sol.potential_at(c.position_m)
        context=context_for_solution(p,sol)
        assert context.domain_radius_m is None
        assert context.domain_cylinder.radius_m==pytest.approx(R)
        assert len(context.masks)==2
        assert 'finite disks' in context.conductor_results_method
        assert context.region((R+.01,0,D/2))=='outside-domain'
        assert context.region((0,0,-.01))=='metal'
        assert context.region((0,0,.04))=='vacuum'
        assert len(calls(fake,'ei_addarc'))==4
        assert len(calls(fake,'ei_addblocklabel'))==3
        frame=sample_sweep([('mixed',p)],Femm5ParallelPlateBackend(
                           session_factory=lambda:FakeSession(FakeClient())),
                           plane=SliceSpec.meridian(p,extent_m=.28,resolution=11),
                           seeds=[],line_steps=30)['frames'][0]
        assert len(frame['geometry'])==2
        assert len(frame['profiles'])==2
        assert frame['domain_cylinder']['radius_m']==pytest.approx(R)
    assert session.opens==session.closes==1
    assert calls(fake,'ei_addconductorprop')==[
        ('lower',0.,0,1),('upper',0.,0,1),
        (cores[0].name,0,Q,0),(cores[1].name,0,-.5*Q,0)]


def test_grouped_cancellation_and_rotated_translated_normal_line():
    shift=(2.,-1.,.5);axis=(1.,0.,0.);perp=(0.,.04,-.03)
    p=problem((source(Q,.04,shift=shift,axis=axis,perp=perp),
               source(-Q,.04,shift=shift,axis=axis,perp=perp),
               source(-.2*Q,.15,shift=shift,axis=axis,perp=perp)),
              shift=shift,axis=axis)
    assert Femm5ParallelPlateBackend().supports(p)
    R,foot,n,d,cores=_layout(p,Femm5Settings())
    assert n==axis and foot==pytest.approx((2.,-.96,.47))
    assert len(cores)==1 and cores[0].prescribed_charge_c==pytest.approx(-.2e-9)
    fake=FakeClient()
    fake.eo_getconductorproperties=lambda name:(0.,0.)
    with Femm5ParallelPlateBackend(session_factory=lambda:FakeSession(fake)).solve(p) as sol:
        assert sol._origin_m==pytest.approx(foot)
        assert sol.electric_field_at((2.+.08,-.96+.03,.47+.04))==pytest.approx((6.,2.4,3.2))
    assert [s[2] for s in calls(fake,'ei_addconductorprop')]==pytest.approx([0,0,-.2e-9],abs=1e-24)


def test_refuse_off_axis_before_open_and_keep_analytic_three_dimensional():
    misaligned=problem((source(Q,.06),source(-Q,.14,perp=(.01,0.,0.))))
    started=[]
    b=Femm5ParallelPlateBackend(session_factory=lambda:started.append('opened'))
    assert ParallelPlateBackend().supports(misaligned)
    assert not b.supports(misaligned)
    with pytest.raises(UnsupportedProblem,match="normal-line"):
        b.solve(misaligned)
    badnames=problem()
    a,bp=badnames.conductors
    reserved=ElectrostaticProblem((),(
        ConductingPlane(a.point_m,a.normal,a.constraint,'__conductor_sim_foo'),bp))
    assert not b.supports(reserved)
    with pytest.raises(UnsupportedProblem):
        b.solve(reserved)
    many=problem(tuple(source(Q,.01+.01*i) for i in range(17)))
    assert ParallelPlateBackend().supports(many)
    assert not b.supports(many)
    with pytest.raises(UnsupportedProblem,match='16'):
        b.solve(many)
    assert not started


def test_numerical_cylinder_radial_exit_is_not_mistaken_for_a_metal_surface():
    p=problem()
    class Radial:
        conductor_results=()
        def potential_at(self,point):return -point[0]
        def electric_field_at(self,point):return (1.,0.,0.)
    domain=CylindricalDomain((0.,0.,0.),(0.,0.,1.),.12,0.,D)
    ctx=SamplingContext(p,Radial(),'fake',domain_cylinder=domain)
    spec=SliceSpec(x_range_m=(-.2,.2),y_range_m=(-.05,.25),resolution=13)
    ray=trace_field_line(ctx,spec,(0.,.1),step_m=.3,max_steps=5)
    assert ray['termination']=='outside-domain'
    assert ray['points_m'][-1][0]==pytest.approx(.12,rel=1e-6)
    wall=CylinderSideBoundary((0,0,0),(0,0,1),.12,0,D)
    assert wall.first_hit((0.,0.,.1),(.2,0.,.1))==pytest.approx(.6)
    assert wall.first_hit((0.,0.,.3),(.2,0.,.3)) is None
    with pytest.raises(ValueError):
        CylindricalDomain((0,0,0),(0,0,1),-1,0,D)


def test_femm_errors_cleanup_and_validation():
    p=problem()
    bad=FakeClient(fail_on="ei_analyze")
    session=FakeSession(bad)
    with pytest.raises(Femm5ExecutionError,match='synthetic FEMM fault'):
        Femm5ParallelPlateBackend(session_factory=lambda:session).solve(p)
    assert session.closes==1
    broken=FakeClient()
    broken.eo_getconductorproperties=lambda n:('NaN',0.)
    failed=FakeSession(broken)
    with pytest.raises(Femm5ExecutionError,match='nonfinite'):
        Femm5ParallelPlateBackend(session_factory=lambda:failed).solve(p)
    assert failed.closes==1
