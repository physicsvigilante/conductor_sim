"""M4a pure-Python composability and analytic invariants; NO FEMM process."""
from math import pi, sin
import pytest

from conductor_sim import ElectrostaticProblem, FixedNetCharge, Grounded, PointCharge, SphericalShell
from conductor_sim.femm5 import Femm5Backend, Femm5Settings, _SOURCE_CORE
from conductor_sim.image_charge import ImageChargeShellBackend
from conductor_sim.multi_image import MultiImageShellBackend
from conductor_sim.solver import SingularField, UnsupportedProblem
from test_femm5_adapter import FakeClient, FakeSession, calls


def make_problem(constraint=FixedNetCharge(0), q2=.45e-9, orientation='z'):
    def oriented(z):
        return (z,0.,0.) if orientation == 'x' else (0.,0.,z)
    return ElectrostaticProblem(
        charges=(PointCharge(1e-9, oriented(.035)), PointCharge(q2, oriented(-.025))),
        conductors=(SphericalShell(.1,.12,constraint),),
    )


def fake_solve(p, settings=None):
    client=FakeClient()
    session=FakeSession(client)
    b=Femm5Backend(settings=settings, session_factory=lambda:session)
    return b,client,session


def test_collinear_sources_have_independent_core_conductors():
    p=make_problem()
    b,client,sess=fake_solve(p)
    assert b.supports(p)
    with b.solve(p) as sol:
        assert len(sol.conductor_results) == 1
        assert len(sol.numerical_sources) == 2
        assert [s.axial_offset_m for s in sol.numerical_sources] == pytest.approx((-.025,.035))
        assert [s.radius_m for s in sol.numerical_sources] == pytest.approx((.01,.01))
        assert sol.source_radius_m is None  # ambiguous for multi-site source
        assert sol.numerical_source_charge_c == pytest.approx(2e-9) # two fake FEMM source measurements
        with pytest.raises(UnsupportedProblem, match='source core'):
            sol.potential_at((0,0,.035))
        with pytest.raises(UnsupportedProblem, match='source core'):
            sol.potential_at((0,0,-.025))
        assert sol.potential_at((.05,0,0)) == 12.
    assert sess.opens == sess.closes == 1
    props=calls(client,'ei_addconductorprop')
    assert [p[0] for p in props] == ['shell',f'{_SOURCE_CORE}_1',f'{_SOURCE_CORE}_2']
    assert props[1][2] == pytest.approx(.45e-9)
    assert props[2][2] == pytest.approx(1e-9)
    assert len(calls(client,'ei_addarc')) == 10
    assert len(calls(client,'ei_addsegment')) == 9
    assert len(calls(client,'ei_addblocklabel')) == 5
    assert calls(client,'ei_addpointprop') == []


def test_source_order_does_not_change_geometry():
    p=make_problem()
    b,c,_=fake_solve(p)
    with b.solve(p):
        pass
    rev=ElectrostaticProblem(tuple(reversed(p.charges)),p.conductors)
    b2,c2,_=fake_solve(rev)
    with b2.solve(rev):
        pass
    for op in ('ei_addconductorprop','ei_addnode','ei_addarc','ei_addsegment','ei_setarcsegmentprop'):
        assert calls(c,op) == calls(c2,op)


def test_axes_rotate_to_world_and_preserve_queries():
    p=make_problem(orientation='x')
    b,c,_=fake_solve(p)
    with b.solve(p) as sol:
        assert sol.symmetry_axis == pytest.approx((1,0,0))
        assert sol.potential_at((.065,.03,0)) == 12
        assert calls(c,'eo_getpointvalues')[-1] == pytest.approx((.03,.065))
        assert sol.electric_field_at((.065,.03,0)) == pytest.approx((6,4,0))


def test_two_sources_on_general_world_axis():
    a=(.8,.6,0.)
    p=ElectrostaticProblem(
        charges=(PointCharge(1e-9,tuple(.035*x for x in a)),
                 PointCharge(-.4e-9,tuple(-.025*x for x in a))),
        conductors=(SphericalShell(.1,.12,Grounded()),))
    b,_,_=fake_solve(p)
    assert b.supports(p)
    with b.solve(p) as sol:
        assert sol.symmetry_axis == pytest.approx(a)
        assert [s.axial_offset_m for s in sol.numerical_sources] == pytest.approx((-.025,.035))


def test_noncollinear_sources_rejected_before_solver_start():
    p=ElectrostaticProblem(charges=(PointCharge(1e-9,(0,0,.035)),
                 PointCharge(.5e-9,(.02,0,0))),
                 conductors=(SphericalShell(.1,.12,Grounded()),))
    b,_,sess=fake_solve(p)
    assert not b.supports(p)
    with pytest.raises(UnsupportedProblem,match='non-collinear'):
        b.solve(p)
    assert sess.opens==0
    assert MultiImageShellBackend().supports(p)  # analytic reference permits 3D


def test_core_radius_limited_by_nearest_neighbor():
    p=ElectrostaticProblem(charges=(PointCharge(1e-9,(0,0,.01)),
                 PointCharge(.5e-9,(0,0,.011))),
                 conductors=(SphericalShell(.1,.12,Grounded()),))
    b,_,_=fake_solve(p)
    with b.solve(p) as sol:
        assert [s.radius_m for s in sol.numerical_sources] == pytest.approx((.00025,.00025))


def test_cancelled_site_needs_no_core_or_symmetry_constraint():
    p=ElectrostaticProblem(charges=(PointCharge(.4e-9,(0,0,.03)),
        PointCharge(1e-9,(.02,0,0)),PointCharge(-1e-9,(.02,0,0))),
        conductors=(SphericalShell(.1,.12,FixedNetCharge(0)),))
    b,_,_=fake_solve(p)
    assert b.supports(p)
    with b.solve(p) as sol:
        assert len(sol.numerical_sources)==1
        assert sol.numerical_sources[0].name==_SOURCE_CORE


def test_cap_at_16_active_sites():
    p=ElectrostaticProblem(
        charges=tuple(PointCharge(1e-12,(0,0,-.08 + i*.01)) for i in range(17)),
        conductors=(SphericalShell(.1,.12,Grounded()),))
    assert not Femm5Backend().supports(p)


def test_exact_multisource_matches_single_source_superposition():
    p=make_problem()
    both=MultiImageShellBackend().solve(p)
    individual=[ImageChargeShellBackend().solve(ElectrostaticProblem((source,),p.conductors))
                for source in p.charges]
    for pt in ((.055,0,0),(0,0,.075),(0,0,-.07),(.2,0,.1)):
        assert both.potential_at(pt) == pytest.approx(sum(s.potential_at(pt) for s in individual))
        assert both.electric_field_at(pt) == pytest.approx(tuple(
            sum(s.electric_field_at(pt)[i] for s in individual) for i in range(3)))
    for theta in (.2,.65,1.4,2.3,pi-.2):
        pos=(.1*sin(theta),0,.1*__import__('math').cos(theta))
        expected=(individual[0].surface_charge_density('inner',theta) +
                  individual[1].surface_charge_density('inner',pi-theta))
        assert both.surface_charge_density_at('inner',pos) == pytest.approx(expected)


def test_exact_multisource_grounding_and_shell_charge():
    p=make_problem(Grounded())
    sol=MultiImageShellBackend().solve(p)
    assert sol.conductor_results[0].net_charge_c == pytest.approx(-1.45e-9)
    assert sol.conductor_results[0].potential_v == 0
    assert sol.conductor_results[0].inner_surface_charge_c == pytest.approx(-1.45e-9)
    for pt in ((.18,0,0),(.23,0,.05),(.105,0,0)):
        if abs(sum(x*x for x in pt)**.5-.105) < 1e-9:
            pass
        assert abs(sol.potential_at(pt)) < 1e-10


def test_exact_multisource_integrated_inner_sigma():
    p=make_problem()
    sol=MultiImageShellBackend().solve(p)
    n=800
    from math import cos
    integral=sum(sol.surface_charge_density_at('inner',(.1*sin((k+.5)*pi/n),0,
                      .1*cos((k+.5)*pi/n)))*sin((k+.5)*pi/n)
                 for k in range(n))*2*pi*.1**2*pi/n
    assert integral == pytest.approx(-1.45e-9,rel=.002)


def test_analytical_allows_noncollinear_sources_and_enforces_singularities():
    p=ElectrostaticProblem(
        charges=(PointCharge(1e-9,(.03,0,0)),PointCharge(2e-9,(0,.025,0))),
        conductors=(SphericalShell(.1,.12,FixedNetCharge(0)),))
    b=MultiImageShellBackend()
    assert b.supports(p)
    s=b.solve(p)
    with pytest.raises(SingularField):
        s.potential_at((.03,0,0))
    with pytest.raises(SingularField):
        s.electric_field_at((0,.025,0))
    assert s.conductor_results[0].outer_surface_charge_c == pytest.approx(3e-9)
    assert s.surface_charge_density_at('inner',(.1,0,0)) < 0
    with pytest.raises(ValueError):
        s.surface_charge_density_at('inner',(.05,0,0))
    assert not b.supports(ElectrostaticProblem(
        charges=(PointCharge(1e-9,(.15,0,0)),),conductors=p.conductors))
