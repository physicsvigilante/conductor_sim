"""M12b: genuine pinned FEMM 5 numerical plate-pair acceptance tests.

This file is only run by the dedicated REAL FEMM 5 GitHub Actions workflow.
No analytical solution is substituted for actual finite-disk measurements.
"""
from math import pi
from os import environ

import pytest

from conductor_sim import (
    ConductingPlane, ElectrostaticProblem, FixedPotential, Grounded,
    PointCharge, ParallelPlateBackend, Femm5ParallelPlateBackend, Femm5Settings,
)
from conductor_sim.model import EPSILON_0
from conductor_sim.femm5_parallel import _layout
from conductor_sim.solver import UnsupportedProblem

pytestmark = [pytest.mark.femm5,
              pytest.mark.skipif(not environ.get('FEMM5_BINDING'),
                                 reason='real FEMM 5 binding unavailable')]

Q=1e-9
D=.2


def prob(charges=(),v0=0.,v1=0.,*,origin=(0.,0.,0.),axis=(0.,0.,1.),
         transverse=(0.,0.,0.)):
    top=tuple(origin[i]+axis[i]*D for i in range(3))
    sources=tuple(PointCharge(q,tuple(origin[i]+axis[i]*height+transverse[i]
                                       for i in range(3))) for q,height in charges)
    return ElectrostaticProblem(sources,(
        ConductingPlane(point_m=origin,normal=axis,
                        constraint=FixedPotential(v0),name="lower"),
        ConductingPlane(point_m=top,normal=tuple(-x for x in axis),
                        constraint=FixedPotential(v1),name="upper"),
    ))


def expected_disk_charges(p,R):
    lower,upper=p.conductors
    gap=lower.signed_distance_m(upper.point_m)
    c=p.permittivity_f_per_m*pi*R*R/gap
    v0,v1=lower.constraint.potential_v,upper.constraint.potential_v
    q0=c*(v0-v1)-sum(q.charge_c*(1-lower.signed_distance_m(q.position_m)/gap)
                    for q in p.charges)
    q1=c*(v1-v0)-sum(q.charge_c*lower.signed_distance_m(q.position_m)/gap
                    for q in p.charges)
    return q0,q1


def test_real_femm_charge_free_bias_field_and_finite_disk_capacitance():
    p=prob(v0=12.,v1=-6.)
    a=ParallelPlateBackend().solve(p)
    with Femm5ParallelPlateBackend(settings=Femm5Settings(mesh_scale=.8)).solve(p) as sol:
        R=sol.outer_radius_m
        assert len(sol.numerical_sources)==0
        assert [c.potential_v for c in sol.conductor_results]==pytest.approx([12.,-6.],abs=.15)
        assert [c.net_charge_c for c in sol.conductor_results]==pytest.approx(
            expected_disk_charges(p,R),rel=.08,abs=8e-11)
        for point in ((0.,0.,.05),(.06,0.,.11),(.23,0.,.15),(.32,0.,.08)):
            assert sol.potential_at(point)==pytest.approx(a.potential_at(point),rel=.01,abs=.65)
            assert sol.electric_field_at(point)==pytest.approx(
                a.electric_field_at(point),rel=.18,abs=20.)
        for name,z in (('lower',0.),('upper',D)):
            for r in (0.,.08,.25):
                sigma=sol.surface_charge_density('outer',r,conductor_name=name)
                exact=a.surface_charge_density_at('outer',(r,0,z),conductor_name=name)
                assert sigma==pytest.approx(exact,rel=.28,abs=2.5e-10)


def test_real_femm_one_source_induces_distinct_two_electrode_charges():
    p=prob(((Q,.065),))
    a=ParallelPlateBackend(image_order=96).solve(p)
    with Femm5ParallelPlateBackend(settings=Femm5Settings(mesh_scale=.8)).solve(p) as sol:
        assert len(sol.numerical_sources)==1
        assert sol.numerical_sources[0].measured_charge_c==pytest.approx(Q,rel=.04,abs=3e-11)
        assert [c.net_charge_c for c in sol.conductor_results]==pytest.approx(
            expected_disk_charges(p,sol.outer_radius_m),rel=.20,abs=9e-11)
        assert sum(c.net_charge_c for c in sol.conductor_results)==pytest.approx(-Q,rel=.11)
        for x in ((.055,0.,.09),(.09,0.,.145),(.17,0.,.12)):
            assert sol.potential_at(x)==pytest.approx(a.potential_at(x),rel=.30,abs=2.5)
            assert sol.electric_field_at(x)==pytest.approx(a.electric_field_at(x),rel=.45,abs=140.)
        for name,z in [('lower',0.),('upper',D)]:
            for r in (.04,.12):
                sigma=sol.surface_charge_density('outer',r,conductor_name=name)
                exact=a.surface_charge_density_at('outer',(r,0,z),conductor_name=name)
                assert sigma<0
                assert sigma==pytest.approx(exact,rel=.8,abs=8e-10)


def test_real_femm_mixed_source_signs_and_bias_not_spurious_total():
    p=prob(((Q,.06),(-.5*Q,.14)),v0=0.,v1=20.)
    a=ParallelPlateBackend(image_order=96).solve(p)
    with Femm5ParallelPlateBackend(settings=Femm5Settings(mesh_scale=.75)).solve(p) as sol:
        assert len(sol.numerical_sources)==2
        assert [s.measured_charge_c for s in sol.numerical_sources]==pytest.approx(
            [Q,-.5*Q],rel=.04,abs=3e-11)
        expected=expected_disk_charges(p,sol.outer_radius_m)
        assert [c.net_charge_c for c in sol.conductor_results]==pytest.approx(
            expected,rel=.07,abs=1.5e-10)
        assert sum(c.net_charge_c for c in sol.conductor_results)==pytest.approx(
            -.5*Q,rel=.20,abs=1.1e-10)
        for point in ((.05,0,.08),(.11,0,.10),(.15,0,.17)):
            assert sol.potential_at(point)==pytest.approx(a.potential_at(point),rel=.35,abs=3.5)
            assert sol.electric_field_at(point)==pytest.approx(a.electric_field_at(point),
                                                               rel=.50,abs=180.)


def test_real_femm_zero_sum_retains_both_source_cores_and_plate_dipole():
    p=prob(((Q,.06),(-Q,.14)))
    with Femm5ParallelPlateBackend().solve(p) as sol:
        assert len(sol.numerical_sources)==2
        assert [c.measured_charge_c for c in sol.numerical_sources]==pytest.approx(
            [Q,-Q],rel=.05)
        qa,qb=expected_disk_charges(p,sol.outer_radius_m)
        assert qa<0<qb
        assert [c.net_charge_c for c in sol.conductor_results]==pytest.approx(
            [qa,qb],rel=.20,abs=7e-11)
        assert sum(c.net_charge_c for c in sol.conductor_results)==pytest.approx(
            0.,abs=9e-11)


def test_real_femm_transform_matches_independent_image_reference():
    a=prob(((Q,.075),),v0=0.,v1=9.)
    b=prob(((Q,.075),),v0=0.,v1=9.,
           origin=(2.,-1.,.5),axis=(1.,0.,0.),transverse=(0.,.04,-.03))
    exact=ParallelPlateBackend(image_order=80)
    settings=Femm5Settings(mesh_scale=.75,arc_degrees=3.)
    with Femm5ParallelPlateBackend(settings=settings).solve(a) as sa,\
         Femm5ParallelPlateBackend(settings=settings).solve(b) as sb:
        pt=(.08,.02,.11)
        mapped=(2.+pt[2],-1.+.04+pt[1],.5-.03+pt[0])
        va=exact.solve(a).potential_at(pt)
        vb=exact.solve(b).potential_at(mapped)
        assert vb==pytest.approx(va,rel=1e-9,abs=1e-6)
        assert sa.potential_at(pt)==pytest.approx(va,rel=.35,abs=3.)
        assert sb.potential_at(mapped)==pytest.approx(vb,rel=.35,abs=3.)
        f=exact.solve(a).electric_field_at(pt)
        fa=sa.electric_field_at(pt)
        fb=sb.electric_field_at(mapped)
        assert fa==pytest.approx(f,rel=.50,abs=150.)
        assert fb==pytest.approx((f[2],f[1],f[0]),rel=.50,abs=150.)
        assert [c.net_charge_c for c in sb.conductor_results]==pytest.approx(
            [c.net_charge_c for c in sa.conductor_results],rel=.10,abs=1e-10)


def test_real_femm_radial_domain_mesh_and_core_sensitivity():
    p=prob(((Q,.065),(-.35*Q,.145)))
    exact=ParallelPlateBackend(image_order=96).solve(p)
    point=(.08,0.,.10)
    potentials=[]
    for settings in (Femm5Settings(outer_radius_factor=4,mesh_scale=1.),
                     Femm5Settings(outer_radius_factor=8,mesh_scale=.8),
                     Femm5Settings(outer_radius_factor=10,source_radius_fraction=.06,arc_degrees=3.)):
        with Femm5ParallelPlateBackend(settings=settings).solve(p) as sol:
            potentials.append(sol.potential_at(point))
            assert [c.net_charge_c for c in sol.conductor_results]==pytest.approx(
                expected_disk_charges(p,sol.outer_radius_m),rel=.22,abs=1e-10)
    val=exact.potential_at(point)
    assert all(v==pytest.approx(val,rel=.4,abs=3.) for v in potentials)
    assert max(potentials)-min(potentials)<max(3.,abs(val)*.35)


def test_real_femm_invalid_offaxis_rejected_without_simulating_rings():
    p=ElectrostaticProblem(
        (PointCharge(Q,(0,0,.06)),PointCharge(-Q,(.01,0,.14))),
        prob().conductors)
    assert ParallelPlateBackend().supports(p)
    backend=Femm5ParallelPlateBackend()
    assert not backend.supports(p)
    with pytest.raises(UnsupportedProblem,match='normal-line'):
        backend.solve(p)


def test_real_femm_does_not_silently_supply_analytical_force_energy():
    from conductor_sim import force_energy
    p=prob(((Q,.065),),v1=20.)
    with Femm5ParallelPlateBackend().solve(p) as numerical:
        before=numerical.potential_at((.05,0.,.10))
        with pytest.raises(UnsupportedProblem,match='does not provide'):
            force_energy(numerical)
        # Observation refusal must not close or mutate the caller-owned solve.
        assert numerical.potential_at((.05,0.,.10))==before
        assert len(numerical.numerical_sources)==1
