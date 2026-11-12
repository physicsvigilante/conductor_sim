"""Pinned *actual* FEMM 5 validation for multi-source grounded-plane models.

Every numerical charge, field and potential here comes from FEMM postprocessing;
image charges are an independent reference, never a numerical fallback.
"""
from math import pi, sqrt
from os import environ

import pytest

from conductor_sim import (ConductingPlane, ElectrostaticProblem, Femm5PlaneBackend,
                           Femm5Settings, GroundedPlaneBackend, PointCharge)
from conductor_sim.solver import UnsupportedProblem

pytestmark = [pytest.mark.femm5,
              pytest.mark.skipif(not environ.get('FEMM5_BINDING'),
                                 reason='real FEMM 5 binding unavailable')]

Q = 1e-9


def prob(charges=(Q,-.5*Q),heights=(.12,.26),normal=(0.,0.,1.),origin=(0.,0.,0.)):
    return ElectrostaticProblem(
        tuple(PointCharge(q,tuple(origin[i]+h*normal[i] for i in range(3)))
              for q,h in zip(charges,heights)),
        (ConductingPlane(point_m=origin,normal=normal,name='ground'),))


def exact_truncated_charge(problem, radius):
    plane = problem.conductors[0]
    return -sum(s.charge_c*(1-h/sqrt(h*h+radius*radius))
                for s in problem.charges for h in [plane.signed_distance_m(s.position_m)])


@pytest.mark.parametrize('charges',[(Q,-.5*Q),(-Q,.5*Q),(Q,.5*Q)])
def test_real_femm_distinct_axial_sources_fields_and_truncated_plane_charge(charges):
    p=prob(charges=charges)
    ref=GroundedPlaneBackend().solve(p)
    with Femm5PlaneBackend().solve(p) as sol:
        assert len(sol.numerical_sources)==2
        assert sol.source_radius_m is None
        assert sol.numerical_source_charge_c is None
        for measured,q in zip(sol.numerical_sources,charges):
            assert measured.measured_charge_c==pytest.approx(q,rel=.035,abs=3e-12)
            assert measured.radius_m>0
        expected_disk=exact_truncated_charge(p,sol.outer_radius_m)
        # The physical FEMM plane is a finite disk; its Q is not -sum(q).
        assert sol.conductor_results[0].net_charge_c==pytest.approx(expected_disk,rel=.14,abs=3e-11)
        assert sol.conductor_results[0].potential_v==pytest.approx(0,abs=.3)
        for pt in ((0.,0.,.035),(.06,0.,.065),(.07,0.,.17),(.15,0.,.38),(.12,0.,.08)):
            assert sol.potential_at(pt)==pytest.approx(ref.potential_at(pt),rel=.24,abs=2.5)
            assert sol.electric_field_at(pt)==pytest.approx(ref.electric_field_at(pt),rel=.45,abs=180.)


def test_real_femm_zero_total_charge_retains_both_signed_cores():
    p=prob(charges=(Q,-Q))
    ref=GroundedPlaneBackend().solve(p)
    assert ref.conductor_results[0].net_charge_c==pytest.approx(0,abs=1e-24)
    with Femm5PlaneBackend(settings=Femm5Settings(arc_degrees=3,mesh_scale=.8)).solve(p) as sol:
        assert len(sol.numerical_sources)==2
        assert sol.numerical_sources[0].measured_charge_c==pytest.approx(Q,rel=.04)
        assert sol.numerical_sources[1].measured_charge_c==pytest.approx(-Q,rel=.04)
        assert sol.potential_at((0.06,0,.09))==pytest.approx(ref.potential_at((0.06,0,.09)),rel=.3,abs=3)
        disk=exact_truncated_charge(p,sol.outer_radius_m)
        assert abs(disk)>1e-12  # finite disk does not generally have zero net Q
        assert sol.conductor_results[0].net_charge_c==pytest.approx(disk,rel=.50,abs=2e-11)


def test_real_femm_multi_plane_sigma_profile_and_near_disk_integral():
    p=prob(charges=(Q,.5*Q),heights=(.12,.26))
    ref=GroundedPlaneBackend().solve(p)
    with Femm5PlaneBackend(settings=Femm5Settings(mesh_scale=.7,arc_degrees=3)).solve(p) as sol:
        for r in (0.,.04,.10,.24,.40):
            measured=sol.surface_charge_density('outer',r)
            exact=ref.surface_charge_density_at('outer',(r,0,0))
            assert measured<0
            assert measured==pytest.approx(exact,rel=.70,abs=6e-10)
        radius=.45
        n=75
        dr=radius/n
        integrated=sum(2*pi*(i+.5)*dr*sol.surface_charge_density('outer',(i+.5)*dr)*dr
                       for i in range(n))
        exact_disk=exact_truncated_charge(p,radius)
        assert integrated==pytest.approx(exact_disk,rel=.45)


def test_real_femm_world_transform_and_another_plane_anchor():
    pz=prob()
    px=prob(normal=(1.,0.,0.),origin=(2.,-1.,.5))
    with Femm5PlaneBackend().solve(pz) as z, Femm5PlaneBackend().solve(px) as x:
        pt=(.07,.02,.07)
        mapped=(2.+pt[2],-1.+pt[1],.5+pt[0])
        assert x.potential_at(mapped)==pytest.approx(z.potential_at(pt),rel=.09,abs=1)
        f=z.electric_field_at(pt)
        assert x.electric_field_at(mapped)==pytest.approx((f[2],f[1],f[0]),rel=.15,abs=50)
        assert x.conductor_results[0].net_charge_c==pytest.approx(z.conductor_results[0].net_charge_c,rel=.10)


def test_real_femm_mesh_core_domain_sensitivity_with_multiple_sources():
    p=prob()
    reference=GroundedPlaneBackend().solve(p)
    sample=(.08,0,.095)
    vals=[]
    for settings in (Femm5Settings(outer_radius_factor=6,source_radius_fraction=.06),
                     Femm5Settings(outer_radius_factor=8,source_radius_fraction=.10,mesh_scale=.75),
                     Femm5Settings(outer_radius_factor=10,source_radius_fraction=.08,arc_degrees=3)):
        with Femm5PlaneBackend(settings=settings).solve(p) as sol:
            vals.append(sol.potential_at(sample))
            expected=exact_truncated_charge(p,sol.outer_radius_m)
            assert sol.conductor_results[0].net_charge_c==pytest.approx(expected,rel=.16,abs=3e-11)
    exact=reference.potential_at(sample)
    assert all(v==pytest.approx(exact,rel=.3,abs=3.) for v in vals)
    assert max(vals)-min(vals)<.22*max(3.,abs(exact))


def test_real_femm_nonaxisymmetric_multi_plane_refused_prelaunch():
    p=ElectrostaticProblem(
        (PointCharge(Q,(0,0,.12)),PointCharge(-.5*Q,(.01,0,.26))),
        (ConductingPlane(),))
    b=Femm5PlaneBackend()
    assert GroundedPlaneBackend().supports(p)
    assert not b.supports(p)
    with pytest.raises(UnsupportedProblem,match='normal-line'):
        b.solve(p)
