"""M4b exact nested-shell charge/potential/field benchmarks without FEMM."""
from math import pi, sin, cos
import pytest

from conductor_sim import ElectrostaticProblem, FixedNetCharge, Grounded, PointCharge, SphericalShell
from conductor_sim.nested_shells import NestedShellBackend
from conductor_sim.solver import SingularField, UnsupportedProblem

K=1/(4*pi*8.8541878128e-12)


def shells(first=FixedNetCharge(0),second=Grounded()):
    return (SphericalShell(.1,.12,first,name='inner'),
            SphericalShell(.2,.22,second,name='outer'))


def make_p(first=FixedNetCharge(0), second=Grounded(),charges=(PointCharge(1e-9),)):
    return ElectrostaticProblem(tuple(charges),shells(first,second))


def by_name(sol,name):
    return sol.conductor_by_name(name)


def test_outer_grounded_inner_isolated_gauss_and_potential():
    sol=NestedShellBackend().solve(make_p())
    a,b=by_name(sol,'inner'),by_name(sol,'outer')
    assert a.net_charge_c == 0
    assert a.inner_surface_charge_c == pytest.approx(-1e-9)
    assert a.outer_surface_charge_c == pytest.approx(1e-9)
    assert b.net_charge_c == pytest.approx(-1e-9)
    assert b.inner_surface_charge_c == pytest.approx(-1e-9)
    assert b.outer_surface_charge_c == pytest.approx(0)
    assert a.potential_v == pytest.approx(K*1e-9*(1/.12-1/.2))
    assert b.potential_v == pytest.approx(0)
    assert sol.potential_at((.16,0,0)) == pytest.approx(K*1e-9*(1/.16-1/.2))
    assert sol.electric_field_at((.16,0,0)) == pytest.approx((K*1e-9/.16**2,0,0))
    assert sol.potential_at((.26,0,0)) == 0
    assert sol.electric_field_at((.26,0,0)) == (0,0,0)


def test_both_isolated_neutral_spherical_gaps():
    sol=NestedShellBackend().solve(make_p(FixedNetCharge(0), FixedNetCharge(0)))
    a,b=by_name(sol,'inner'),by_name(sol,'outer')
    assert b.potential_v == pytest.approx(K*1e-9/.22)
    assert a.potential_v == pytest.approx(K*1e-9*(1/.22+1/.12-1/.2))
    assert sol.potential_at((.15,0,0)) == pytest.approx(K*1e-9*(1/.22+1/.15-1/.2))
    assert sol.potential_at((.28,0,0)) == pytest.approx(K*1e-9/.28)
    assert sol.electric_field_at((.28,0,0)) == pytest.approx((K*1e-9/.28**2,0,0))


def test_inner_grounded_outer_fixed_neutral_shields_all_exterior():
    sol=NestedShellBackend().solve(make_p(Grounded(),FixedNetCharge(0)))
    assert by_name(sol,'inner').net_charge_c == pytest.approx(-1e-9)
    assert by_name(sol,'outer').net_charge_c == 0
    for x in (.11,.15,.21,.28):
        assert sol.potential_at((x,0,0)) == pytest.approx(0,abs=1e-10)
        assert sol.electric_field_at((x,0,0)) == pytest.approx((0,0,0),abs=1e-9)


def test_two_grounded_shells_and_charged_gap():
    sol=NestedShellBackend().solve(make_p(Grounded(),Grounded()))
    assert by_name(sol,'inner').net_charge_c == pytest.approx(-1e-9)
    assert by_name(sol,'outer').net_charge_c == pytest.approx(0)
    assert all(abs(item.potential_v)<1e-10 for item in sol.conductor_results)


def test_reverse_shell_input_order_preserves_labels_results():
    p=make_p(FixedNetCharge(1e-9),Grounded())
    rev=ElectrostaticProblem(p.charges,tuple(reversed(p.conductors)))
    assert NestedShellBackend().supports(rev)
    left=NestedShellBackend().solve(p);right=NestedShellBackend().solve(rev)
    assert [x.name for x in right.conductor_results]==['outer','inner']
    assert left.potential_at((.14,0,0))==pytest.approx(right.potential_at((.14,0,0)))
    for name in ('inner','outer'):
        assert left.conductor_by_name(name)==right.conductor_by_name(name)


def test_three_shells_mixed_grounding_and_net_charge():
    conductors=(SphericalShell(.1,.12,FixedNetCharge(2e-9),name='i'),
                SphericalShell(.18,.2,Grounded(),name='m'),
                SphericalShell(.26,.28,FixedNetCharge(-1e-9),name='o'))
    p=ElectrostaticProblem((PointCharge(1e-9,(0,0,.035)),),conductors)
    sol=NestedShellBackend().solve(p)
    assert sol.conductor_by_name('m').potential_v == pytest.approx(0,abs=1e-10)
    assert sol.conductor_by_name('i').net_charge_c == pytest.approx(2e-9)
    assert sol.conductor_by_name('o').net_charge_c == pytest.approx(-1e-9)
    for i,s in enumerate(sol.radial_shells):
        q=sol.conductor_by_name(s.name)
        assert q.inner_surface_charge_c + q.outer_surface_charge_c==pytest.approx(q.net_charge_c)
    for r in (.15,.23,.33):
        dr=.00001
        e=sol.electric_field_at((r,0,0))[0]
        finite_diff=-(sol.potential_at((r+dr,0,0))-sol.potential_at((r-dr,0,0)))/(2*dr)
        assert e==pytest.approx(finite_diff,rel=1e-6)


def test_offcentre_multi_charge_cavity_shift_versus_grounded_single_shell():
    p=make_p(FixedNetCharge(0),Grounded(),
             (PointCharge(1e-9,(0,0,.035)),PointCharge(-.3e-9,(.02,0,0))))
    s=NestedShellBackend().solve(p)
    for point in ((.06,0,0),(0,.05,0),(0,0,-.07)):
        assert s.potential_at(point)==pytest.approx(s.conductor_by_name('inner').potential_v+
                                                    s.inner_reference.potential_at(point))
        assert s.electric_field_at(point)==pytest.approx(s.inner_reference.electric_field_at(point))
    at=(0,0,.1)
    assert s.surface_charge_density_at('inner','inner',at)==pytest.approx(
        s.inner_reference.surface_charge_density_at('inner',at))
    outer=s.conductor_by_name('outer')
    at_b=(0,0,.22)
    assert s.surface_charge_density_at('outer','outer',at_b)==pytest.approx(
        outer.outer_surface_charge_c/(4*pi*.22**2))


def test_touching_or_eccentric_shells_and_source_in_gap_rejected():
    b=NestedShellBackend()
    bad=(
        ElectrostaticProblem(conductors=(SphericalShell(.1,.2,Grounded(),name='a'),SphericalShell(.2,.3,Grounded(),name='b'))),
        ElectrostaticProblem(conductors=(SphericalShell(.1,.12,Grounded(),name='a'),SphericalShell(.18,.2,Grounded(),center_m=(0,0,.01),name='b'))),
        ElectrostaticProblem(charges=(PointCharge(1e-9,(0,0,.15)),),conductors=shells()),
    )
    for p in bad:
        assert not b.supports(p)
        with pytest.raises(UnsupportedProblem):
            b.solve(p)


def test_field_jumps_at_each_interface():
    s=NestedShellBackend().solve(make_p())
    for radius in (.1,.12,.2,.22):
        with pytest.raises(SingularField):
            s.electric_field_at((radius,0,0))
