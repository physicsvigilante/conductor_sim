"""Real FEMM 5 validation of nested concentric conducting spherical shells."""
from math import pi, sin, cos
from os import environ
import pytest

from conductor_sim import ElectrostaticProblem, FixedNetCharge, Grounded, PointCharge, SphericalShell
from conductor_sim.femm5 import Femm5Backend, Femm5Settings
from conductor_sim.nested_shells import NestedShellBackend

pytestmark=[pytest.mark.femm5,
 pytest.mark.skipif(not environ.get('FEMM5_BINDING'),reason='real FEMM 5 not configured')]


def p(first=FixedNetCharge(0),second=Grounded(),charges=None,reverse=False):
    if charges is None:
        charges=(PointCharge(1e-9,(0,0,.035)),PointCharge(.45e-9,(0,0,-.025)))
    shells=(SphericalShell(.1,.12,first,name='inner'),
            SphericalShell(.2,.22,second,name='outer'))
    return ElectrostaticProblem(tuple(charges),tuple(reversed(shells)) if reverse else shells)


@pytest.mark.parametrize('first,second',[
    (FixedNetCharge(0),Grounded()),
    (FixedNetCharge(0),FixedNetCharge(0)),
    (Grounded(),FixedNetCharge(0)),
    (Grounded(),Grounded()),
])
def test_real_m4b_mixed_grounding_and_nested_charge_induction(first,second):
    problem=p(first,second)
    reference=NestedShellBackend().solve(problem)
    with Femm5Backend().solve(problem) as sol:
        assert len(sol.conductor_results)==2
        assert len(sol.numerical_sources)==2
        assert sol.numerical_source_charge_c==pytest.approx(1.45e-9,rel=.04)
        for shell in problem.conductors:
            actual=sol.conductor_by_name(shell.name)
            exact=reference.conductor_by_name(shell.name)
            assert actual.net_charge_c==pytest.approx(exact.net_charge_c,rel=.07,abs=4e-11)
            assert actual.potential_v==pytest.approx(exact.potential_v,rel=.12,abs=1)
            assert actual.inner_surface_charge_c==pytest.approx(exact.inner_surface_charge_c,rel=.08,abs=4e-11)
            assert actual.outer_surface_charge_c==pytest.approx(exact.outer_surface_charge_c,rel=.08,abs=4e-11)
        points=((.055,0,0),(0,0,.075),(.11,0,0),(.16,0,0),(.21,0,0),(.3,0,0))
        for pt in points:
            assert sol.potential_at(pt)==pytest.approx(reference.potential_at(pt),rel=.20,abs=1.5)
            assert sol.electric_field_at(pt)==pytest.approx(reference.electric_field_at(pt),rel=.45,abs=180)


def test_real_m4b_induced_charge_profiles_each_shell():
    problem=p()
    ref=NestedShellBackend().solve(problem)
    with Femm5Backend(settings=Femm5Settings(mesh_scale=.7,arc_degrees=3)).solve(problem) as sol:
        for theta in (.2,.75,1.4,2.7):
            inner_pt=(.1*sin(theta),0,.1*cos(theta))
            inner_expect=ref.surface_charge_density_at('inner','inner',inner_pt)
            inner_got=sol.surface_charge_density('inner',theta,conductor_name='inner')
            assert inner_got==pytest.approx(inner_expect,rel=.6)
            outer_gap_pt=(.2*sin(theta),0,.2*cos(theta))
            outer_expect=ref.surface_charge_density_at('outer','inner',outer_gap_pt)
            outer_got=sol.surface_charge_density('inner',theta,conductor_name='outer')
            assert outer_got==pytest.approx(outer_expect,rel=.5,abs=1e-10)
        n=60
        total_inner=sum(sol.surface_charge_density('inner',(i+.5)*pi/n,conductor_name='inner')*
                        sin((i+.5)*pi/n) for i in range(n))*2*pi*.1**2*pi/n
        assert total_inner==pytest.approx(-1.45e-9,rel=.4)
        assert abs(sol.surface_charge_density('outer',.6,conductor_name='outer')) < 5e-10


def test_real_m4b_reversed_shell_order_and_empty_gap_field():
    a=p(charges=(PointCharge(1e-9),),reverse=False)
    b=p(charges=(PointCharge(1e-9),),reverse=True)
    with Femm5Backend().solve(a) as s1, Femm5Backend().solve(b) as s2:
        assert [x.name for x in s2.conductor_results]==['outer','inner']
        for point in ((.05,0,0),(.16,0,0),(.3,0,0)):
            assert s1.potential_at(point)==pytest.approx(s2.potential_at(point),rel=.04,abs=.3)


def test_real_m4b_empty_inner_shell_and_grounded_outer():
    problem=p(FixedNetCharge(1e-9),Grounded(),charges=())
    ref=NestedShellBackend().solve(problem)
    with Femm5Backend().solve(problem) as numeric:
        assert not numeric.numerical_sources
        for name in ('inner','outer'):
            assert numeric.conductor_by_name(name).net_charge_c==pytest.approx(
                ref.conductor_by_name(name).net_charge_c,rel=.05,abs=1e-11)
        for r in (.05,.11,.16,.21,.3):
            assert numeric.potential_at((r,0,0))==pytest.approx(
                ref.potential_at((r,0,0)),rel=.12,abs=.5)


def test_real_m4b_three_nested_shells_mixed_constraints():
    shells=(SphericalShell(.1,.12,FixedNetCharge(0),name='i'),
            SphericalShell(.18,.2,Grounded(),name='m'),
            SphericalShell(.27,.29,FixedNetCharge(1e-9),name='o'))
    problem=ElectrostaticProblem((PointCharge(1e-9,(0,0,.03)),),shells)
    ref=NestedShellBackend().solve(problem)
    with Femm5Backend().solve(problem) as numeric:
        assert len(numeric.conductor_results)==3
        for name in ('i','m','o'):
            a=numeric.conductor_by_name(name);b=ref.conductor_by_name(name)
            assert a.potential_v==pytest.approx(b.potential_v,rel=.15,abs=1)
            assert a.net_charge_c==pytest.approx(b.net_charge_c,rel=.1,abs=3e-11)
        for r in (.05,.14,.19,.235,.28,.35):
            assert numeric.potential_at((r,0,0))==pytest.approx(
                ref.potential_at((r,0,0)),rel=.20,abs=1.5)


def test_real_m4b_domain_and_mesh_sensitivity():
    problem=p(FixedNetCharge(0),Grounded(),charges=(PointCharge(1e-9,(0,0,.025)),))
    ref=NestedShellBackend().solve(problem)
    expect=ref.potential_at((.16,0,0))
    vals=[]
    for settings in (Femm5Settings(outer_radius_factor=5,mesh_scale=1),
                     Femm5Settings(outer_radius_factor=8,mesh_scale=.65)):
        with Femm5Backend(settings=settings).solve(problem) as numerical:
            vals.append(numerical.potential_at((.16,0,0)))
    assert all(v==pytest.approx(expect,rel=.15,abs=.5) for v in vals)
    assert abs(vals[0]-vals[1]) < .12*abs(expect)+.2
