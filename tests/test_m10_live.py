"""Real FEMM 5 numerical acceptance tests; skipped without a FEMM 5 install."""
from math import pi, sin, sqrt
from os import environ

import pytest

from conductor_sim import (ConductingPlane, ElectrostaticProblem, Femm5PlaneBackend,
                           Femm5Settings, Grounded, GroundedPlaneBackend,
                           PointCharge)
from conductor_sim.solver import UnsupportedProblem

pytestmark=[pytest.mark.femm5,
            pytest.mark.skipif(not environ.get('FEMM5_BINDING'),
                reason='actual FEMM 5 binding not configured')]


def p(q=1e-9,h=.15,n=(0,0,1),center=(0.,0.,0.)):
    return ElectrostaticProblem((PointCharge(q,tuple(center[i]+h*n[i] for i in range(3))),),
                                (ConductingPlane(point_m=center,normal=n,
                                                 constraint=Grounded(),name='metal'),))


@pytest.mark.parametrize('q',[1e-9,-1e-9])
def test_real_femm_grounded_plane_potential_field_and_induced_charge(q):
    problem=p(q=q)
    exact=GroundedPlaneBackend().solve(problem)
    with Femm5PlaneBackend().solve(problem) as got:
        assert got.numerical_source_charge_c==pytest.approx(q,rel=.05)
        conductor=got.conductor_results[0]
        assert conductor.potential_v==pytest.approx(0,abs=.3)
        # FEMM measures only the finite plane segment from rho=0 to R.
        # The exact image-charge density integrates over that disk to
        # -q*(1-h/sqrt(h*h+R*R)), NOT the infinite-plane total -q.
        R=got.outer_radius_m
        exact_finite=-q*(1-.15/sqrt(.15**2+R**2))
        assert conductor.net_charge_c==pytest.approx(exact_finite,rel=.06,abs=2e-11)
        for pt in ((0,0,.035),(.04,0,.06),(.07,0,.11),(.04,0,.24),(.11,0,.2)):
            assert got.potential_at(pt)==pytest.approx(exact.potential_at(pt),rel=.24,abs=2.5)
            assert got.electric_field_at(pt)==pytest.approx(exact.electric_field_at(pt),
                                                            rel=.44,abs=200)
        assert got.potential_at((0,0,0))==0
        assert got.electric_field_at((0,0,-.07))==(0.,0.,0.)
        for rho in (0.,.04,.1,.2):
            actual=got.surface_charge_density('outer',rho)
            expected=exact.surface_charge_density('outer',rho)
            assert actual*q < 0
            assert actual==pytest.approx(expected,rel=.65,abs=4e-10)


def test_real_femm_plane_sigma_integrates_to_near_source_flux():
    problem=p()
    exact=GroundedPlaneBackend().solve(problem)
    with Femm5PlaneBackend(settings=Femm5Settings(arc_degrees=3,mesh_scale=.7)).solve(problem) as sol:
        radius=.42
        n=90
        dx=radius/n
        area_flux=sum(2*pi*(i+.5)*dx*sol.surface_charge_density('outer',(i+.5)*dx)*dx
                      for i in range(n))
        exact_truncated=-1e-9*(1-.15/sqrt(.15**2+radius**2))
        assert area_flux<0
        assert area_flux==pytest.approx(exact_truncated,rel=.38)
        assert sol.conductor_results[0].net_charge_c<0
        assert sol.conductor_results[0].net_charge_c==pytest.approx(-1e-9,rel=.3)


def test_real_femm_plane_outer_boundary_mesh_and_source_size():
    problem=p();exact=GroundedPlaneBackend().solve(problem)
    point=(0.07,0.,.075)
    readings=[]
    for setting in (Femm5Settings(outer_radius_factor=6,source_radius_fraction=.05),
                    Femm5Settings(outer_radius_factor=8,source_radius_fraction=.08,mesh_scale=.75),
                    Femm5Settings(outer_radius_factor=10,source_radius_fraction=.1,arc_degrees=3)):
        with Femm5PlaneBackend(settings=setting).solve(problem) as sol:
            readings.append(sol.potential_at(point))
            R=sol.outer_radius_m
            exact_finite=-1e-9*(1-.15/sqrt(.15**2+R*R))
            assert sol.conductor_results[0].net_charge_c==pytest.approx(exact_finite,rel=.09)
    expected=exact.potential_at(point)
    assert all(x==pytest.approx(expected,rel=.26,abs=2.5) for x in readings)
    assert max(readings)-min(readings) <= .22*max(abs(expected),2.0)


def test_real_femm_plane_world_rotation_translation():
    original=p();moved=p(n=(1,0,0),center=(2.,-1.,.5))
    with Femm5PlaneBackend().solve(original) as z, Femm5PlaneBackend().solve(moved) as x:
        point=(.05,.02,.11)
        mapped=(2.+point[2],-1.+point[1],.5+point[0])
        assert z.potential_at(point)==pytest.approx(x.potential_at(mapped),rel=.1,abs=1.0)
        f=z.electric_field_at(point)
        assert x.electric_field_at(mapped)==pytest.approx((f[2],f[1],f[0]),rel=.15,abs=80)
        assert x.conductor_results[0].net_charge_c==pytest.approx(z.conductor_results[0].net_charge_c,rel=.12)


def test_real_femm_non_axisymmetric_multi_source_unsupported_before_session():
    plane=ConductingPlane()
    problem=ElectrostaticProblem((PointCharge(1e-9,(0.,0.,.15)),
                                 PointCharge(-.5e-9,(.04,0.,.2))),(plane,))
    assert GroundedPlaneBackend().supports(problem)
    num=Femm5PlaneBackend()
    assert not num.supports(problem)
    with pytest.raises(UnsupportedProblem):num.solve(problem)
