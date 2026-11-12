"""Run with a real FEMM 5 installed; NEVER replace numerical values by references."""
from math import pi, sin
from os import environ
import pytest
from conductor_sim import (ElectrostaticProblem,PointCharge,SolidSphere,Grounded,
                           ExternalGroundedSphereBackend,Femm5ExternalSphereBackend,Femm5Settings)

pytestmark=[pytest.mark.femm5,pytest.mark.skipif(not environ.get('FEMM5_BINDING'),
                         reason='real FEMM 5 binding not configured')]

A=.12
Q=1e-9
D=.27

def problem(charge=Q,distance=D,center=(0.,0.,0.),direction=(0.,0.,1.)):
    return ElectrostaticProblem(
        charges=(PointCharge(charge,tuple(center[i]+distance*direction[i] for i in range(3))),),
        conductors=(SolidSphere(A,Grounded(),center_m=center,name='metal'),))


@pytest.mark.parametrize('q', [1e-9,-1e-9])
def test_actual_femm_grounded_sphere_charge_potential_field_and_induction(q):
    p=problem(charge=q)
    exact=ExternalGroundedSphereBackend().solve(p)
    with Femm5ExternalSphereBackend().solve(p) as numeric:
        got=numeric.conductor_results[0];exp=exact.conductor_results[0]
        assert got.net_charge_c==pytest.approx(exp.net_charge_c,rel=.14,abs=1e-11)
        assert got.potential_v==pytest.approx(0,abs=.1)
        assert numeric.numerical_source_charge_c==pytest.approx(q,rel=.04)
        for pt in ((0,0,.16),(.1,0,.2),(.17,0,-.07),(.22,0,.12),(.31,0,-.10)):
            assert numeric.potential_at(pt)==pytest.approx(exact.potential_at(pt),rel=.22,abs=3)
            assert numeric.electric_field_at(pt)==pytest.approx(exact.electric_field_at(pt),rel=.38,abs=100)
        assert abs(numeric.potential_at((0,0,0)))<1
        assert max(abs(x) for x in numeric.electric_field_at((0,0,.04)))<5
        sig=[numeric.surface_charge_density('outer',theta) for theta in (.17,.65,1.4,2.6,2.97)]
        assert all(x*q<0 for x in sig)
        assert abs(sig[0])>3*abs(sig[-1])
        for theta,val in zip((.17,.65,1.4,2.6,2.97),sig):
            assert val==pytest.approx(exact.surface_charge_density('outer',theta),rel=.65,abs=1e-10)


def test_actual_femm_near_surface_sigma_integrates_to_induced_charge():
    p=problem()
    ref=ExternalGroundedSphereBackend().solve(p)
    with Femm5ExternalSphereBackend(settings=Femm5Settings(mesh_scale=.7,arc_degrees=3)).solve(p) as sol:
        n=60;step=pi/n
        total=2*pi*A*A*step*sum(sol.surface_charge_density('outer',(i+.5)*step)*
                                 sin((i+.5)*step) for i in range(n))
        assert total==pytest.approx(ref.conductor_results[0].net_charge_c,rel=.40)
        assert total*Q<0


def test_actual_femm_domain_and_source_size_sensitivity():
    p=problem()
    exact=ExternalGroundedSphereBackend().solve(p)
    at=(0,0,.16)
    expected=exact.potential_at(at)
    values=[]
    for setting in (Femm5Settings(source_radius_fraction=.04,outer_radius_factor=6),
                    Femm5Settings(source_radius_fraction=.08,outer_radius_factor=8),
                    Femm5Settings(source_radius_fraction=.08,outer_radius_factor=10,mesh_scale=.7)):
        with Femm5ExternalSphereBackend(settings=setting).solve(p) as sol:
            values.append(sol.potential_at(at))
            assert sol.conductor_results[0].net_charge_c==pytest.approx(
                exact.conductor_results[0].net_charge_c,rel=.2)
    assert all(x==pytest.approx(expected,rel=.25,abs=2) for x in values)
    assert max(values)-min(values)<.20*max(abs(expected),2.)


def test_actual_femm_world_orientation_and_translation():
    pz=problem()
    px=problem(center=(1.,-2.,.5),direction=(1,0,0))
    with Femm5ExternalSphereBackend().solve(pz) as z, Femm5ExternalSphereBackend().solve(px) as x:
        pt=(.12,.08,.22)
        mapped=(1.+pt[2],-2.+pt[1],.5+pt[0])
        assert x.potential_at(mapped)==pytest.approx(z.potential_at(pt),rel=.07,abs=1)
        fz=z.electric_field_at(pt)
        fx=x.electric_field_at(mapped)
        assert fx==pytest.approx((fz[2],fz[1],fz[0]),rel=.13,abs=30)
