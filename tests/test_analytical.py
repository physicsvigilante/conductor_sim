from math import pi
import pytest
from conductor_sim import (
    EPSILON_0, ConcentricShellBackend, ElectrostaticProblem, FixedNetCharge,
    Grounded, PointCharge, SingularField, SphericalShell, UnsupportedProblem,
)
K = 1/(4*pi*EPSILON_0)
A, B, Q = 0.1, 0.2, 2e-9

def case(constraint, charges=(PointCharge(Q),), center=(0.,0.,0.)):
    return ElectrostaticProblem(charges, (SphericalShell(A, B, constraint, center_m=center),))

def test_grounded():
    s = ConcentricShellBackend().solve(case(Grounded()))
    c = s.conductor_results[0]
    assert c.inner_surface_charge_c == pytest.approx(-Q)
    assert c.outer_surface_charge_c == pytest.approx(0)
    assert c.net_charge_c == pytest.approx(-Q)
    assert c.potential_v == pytest.approx(0)
    assert s.potential_at((A/2,0,0)) == pytest.approx(K*Q/A)
    assert s.potential_at((2*B,0,0)) == pytest.approx(0)
    assert s.electric_field_at((A/2,0,0)) == pytest.approx((K*Q/(A/2)**2,0,0))

def test_isolated_neutral():
    s = ConcentricShellBackend().solve(case(FixedNetCharge(0)))
    c = s.conductor_results[0]
    assert c.inner_surface_charge_c == pytest.approx(-Q)
    assert c.outer_surface_charge_c == pytest.approx(Q)
    assert c.net_charge_c == 0
    assert c.potential_v == pytest.approx(K*Q/B)
    assert s.potential_at((A/2,0,0)) == pytest.approx(K*Q*(1/A+1/B))
    assert s.potential_at((2*B,0,0)) == pytest.approx(K*Q/(2*B))
    assert s.electric_field_at((2*B,0,0)) == pytest.approx((K*Q/(2*B)**2,0,0))

def test_fixed_nonzero_charge():
    s = ConcentricShellBackend().solve(case(FixedNetCharge(3*Q), (PointCharge(-Q),)))
    c = s.conductor_results[0]
    assert c.inner_surface_charge_c == pytest.approx(Q)
    assert c.outer_surface_charge_c == pytest.approx(2*Q)
    assert s.electric_field_at((A/2,0,0))[0] < 0

def test_centered_charges_combine():
    s = ConcentricShellBackend().solve(case(FixedNetCharge(0), (PointCharge(Q),PointCharge(-Q))))
    assert s.potential_at((0,0,0)) == 0
    assert s.electric_field_at((0,0,0)) == (0,0,0)

def test_translated_shell():
    c = (2.,-3.,.7)
    s = ConcentricShellBackend().solve(case(FixedNetCharge(0), (PointCharge(Q,c),), c))
    assert s.potential_at((c[0]+A/2,c[1],c[2])) == pytest.approx(K*Q*(1/A+1/B))

def test_surface_jump_and_source_singularity():
    s = ConcentricShellBackend().solve(case(FixedNetCharge(0)))
    for r in (A,B):
        with pytest.raises(SingularField):
            s.electric_field_at((r,0,0))
    with pytest.raises(SingularField):
        s.potential_at((0,0,0))

def test_offcenter_is_not_supported():
    backend = ConcentricShellBackend()
    p = case(Grounded(), (PointCharge(Q,(0,0,A/2)),))
    assert not backend.supports(p)
    with pytest.raises(UnsupportedProblem):
        backend.solve(p)

def test_permittivity_scaling():
    p = case(FixedNetCharge(0))
    scaled = ElectrostaticProblem(p.charges,p.conductors,4*EPSILON_0)
    backend = ConcentricShellBackend()
    assert backend.solve(scaled).potential_at((2*B,0,0)) == pytest.approx(backend.solve(p).potential_at((2*B,0,0))/4)
