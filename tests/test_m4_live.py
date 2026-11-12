"""REAL FEMM 5 M4a multi-site induction and preservation checks."""
from math import cos, pi, sin
from os import environ
import pytest
from conductor_sim import ElectrostaticProblem, FixedNetCharge, Grounded, PointCharge, SphericalShell
from conductor_sim.femm5 import Femm5Backend, Femm5Settings
from conductor_sim.multi_image import MultiImageShellBackend

pytestmark = [pytest.mark.femm5,
              pytest.mark.skipif(not environ.get('FEMM5_BINDING'),reason='FEMM 5 not configured')]


def problem(constraint=FixedNetCharge(0), axis='z'):
    def pos(z):
        return (z,0,0) if axis == 'x' else (0,0,z)
    return ElectrostaticProblem(charges=(PointCharge(1e-9,pos(.035)),
        PointCharge(.45e-9,pos(-.025))),
        conductors=(SphericalShell(.1,.12,constraint),))


@pytest.mark.parametrize('constraint',[FixedNetCharge(0),Grounded(),FixedNetCharge(1e-9)])
def test_real_m4a_two_distinct_sources_potential_field_and_net_charge(constraint):
    p=problem(constraint)
    exact=MultiImageShellBackend().solve(p)
    with Femm5Backend().solve(p) as numeric:
        assert len(numeric.numerical_sources)==2
        assert sum(s.measured_charge_c for s in numeric.numerical_sources)==pytest.approx(1.45e-9,rel=.05)
        assert numeric.numerical_source_charge_c==pytest.approx(1.45e-9,rel=.05)
        got=numeric.conductor_results[0]; reference=exact.conductor_results[0]
        assert got.potential_v==pytest.approx(reference.potential_v,rel=.12,abs=1.)
        assert got.net_charge_c==pytest.approx(reference.net_charge_c,rel=.08,abs=3e-11)
        assert got.inner_surface_charge_c==pytest.approx(-1.45e-9,rel=.05)
        for pt in ((.055,0,0),(0,0,.077),(0,0,-.071),(.075,0,.015),
                   (.11,0,0),(.23,0,.035)):
            assert numeric.potential_at(pt)==pytest.approx(exact.potential_at(pt),rel=.18,abs=1.5)
            assert numeric.electric_field_at(pt)==pytest.approx(exact.electric_field_at(pt),rel=.42,abs=160)


def test_real_m4a_nonuniform_charge_density_and_integral():
    p=problem()
    exact=MultiImageShellBackend().solve(p)
    with Femm5Backend(settings=Femm5Settings(mesh_scale=.7,arc_degrees=3)).solve(p) as numerical:
        for theta in (.16,.6,1.3,2.,2.8):
            surface_point=(.1*sin(theta),0,.1*cos(theta))
            expected=exact.surface_charge_density_at('inner',surface_point)
            observed=numerical.surface_charge_density('inner',theta)
            assert expected < 0 and observed < 0
            assert observed == pytest.approx(expected,rel=.55)
        n=60
        integrated=sum(numerical.surface_charge_density('inner',(k+.5)*pi/n)*
                      sin((k+.5)*pi/n) for k in range(n)) * 2*pi*.1**2*pi/n
        assert integrated == pytest.approx(-1.45e-9,rel=.35)


def test_real_m4a_world_axis_rotation():
    px=problem(axis='x')
    pz=problem(axis='z')
    with Femm5Backend().solve(px) as a, Femm5Backend().solve(pz) as b:
        p1=(.07,.025,0);p2=(0,.025,.07)
        assert a.potential_at(p1)==pytest.approx(b.potential_at(p2),rel=.05)
        ea=a.electric_field_at(p1);eb=b.electric_field_at(p2)
        assert ea == pytest.approx((eb[2],eb[1],eb[0]),rel=.12,abs=20)


def test_real_m4a_core_radius_sensitivity():
    p=problem()
    exact=MultiImageShellBackend().solve(p)
    pt=(.055,0,0)
    expect=exact.potential_at(pt)
    values=[]
    for f in (.06,.1):
        with Femm5Backend(settings=Femm5Settings(source_radius_fraction=f,mesh_scale=.8)).solve(p) as sol:
            values.append(sol.potential_at(pt))
            assert len(sol.numerical_sources)==2
    assert all(v==pytest.approx(expect,rel=.17) for v in values)
    assert abs(values[0]-values[1]) < .14*abs(expect)
