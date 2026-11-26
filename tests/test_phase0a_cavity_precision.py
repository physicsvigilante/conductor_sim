"""Phase 0a: cavity charge, potential and field cancellation (M01 C3/D1, C4/D2).

Oracles are exact rational Kelvin constructions on the symmetry axis, written
from the closed form (image -q*a/|y| at a**2*y/|y|**2), not from the code under
test. A small authored residual must survive cancellation of large terms.
"""
from fractions import Fraction as F
from math import isclose, pi

import pytest

from conductor_sim import (
    EPSILON_0, ElectrostaticProblem, FixedNetCharge, Grounded, MultiImageShellBackend,
    NestedShellBackend, PointCharge, SphericalShell,
)

K = 1/(4*pi*EPSILON_0)
BIG, SMALL = F(1, 2**30), F(1, 2**90)
BACKENDS = (MultiImageShellBackend, NestedShellBackend)


def _axis_terms(x0, q, a, probe):
    """Exact (V/k, E_x/k) at an axial probe from an axial source and its image."""
    xi, qi = a*a/x0, -q*a/abs(x0)
    v = q/abs(probe-x0) + qi/abs(probe-xi)
    e = q*(probe-x0)/abs(probe-x0)**3 + qi*(probe-xi)/abs(probe-xi)**3
    return v, e


def _problem(sources, constraint, a=1., b=2.):
    charges = tuple(PointCharge(float(q), (float(x), 0., 0.)) for x, q in sources)
    return ElectrostaticProblem(charges, (SphericalShell(a, b, constraint),))


@pytest.mark.parametrize("backend", BACKENDS)
def test_fixed_net_charge_keeps_small_outer_residual(backend):
    # W-a1 (D1): neutralising metal charge cancels only the large source.
    problem = _problem(((F(-1, 2), BIG), (F(1, 4), SMALL)), FixedNetCharge(-float(BIG)))
    solution = backend().solve(problem)
    result = solution.conductor_results[0]
    assert result.outer_surface_charge_c == float(SMALL)
    assert result.inner_surface_charge_c == -float(BIG + SMALL)
    assert isclose(solution.potential_at((4., 0., 0.)), K*float(SMALL)/4, rel_tol=1e-15)
    assert isclose(result.potential_v, K*float(SMALL)/2, rel_tol=1e-15)


@pytest.mark.parametrize("sign", (1, -1))
@pytest.mark.parametrize("backend", BACKENDS)
def test_grounded_cavity_potential_and_field_survive_cancellation(backend, sign):
    # W-a2/W-a3 (D2): mirror-symmetric large pair cancels exactly at the centre.
    sources = ((F(-1, 2), sign*BIG), (F(1, 4), sign*SMALL), (F(1, 2), -sign*BIG))
    v_exact = sum(_axis_terms(x, q, F(1), F(0))[0] for x, q in sources)
    assert v_exact == 3*sign*SMALL
    solution = backend().solve(_problem(sources, Grounded()))
    assert isclose(solution.potential_at((0., 0., 0.)), K*float(v_exact), rel_tol=1e-12)

    field_sources = ((F(-1, 2), sign*BIG), (F(1, 4), sign*SMALL), (F(1, 2), sign*BIG))
    e_exact = sum(_axis_terms(x, q, F(1), F(0))[1] for x, q in field_sources)
    assert e_exact == -F(63, 4)*sign*SMALL
    field = backend().solve(_problem(field_sources, Grounded())).electric_field_at((0., 0., 0.))
    assert isclose(field[0], K*float(e_exact), rel_tol=1e-12)
    assert field[1:] == (0., 0.)


@pytest.mark.parametrize("backend", BACKENDS)
def test_grounded_controls_remain_exact_zero(backend):
    # W-a4: grounding gives zero outer flux, so exterior V and E are exactly 0.
    sources = ((F(-1, 2), BIG), (F(1, 4), SMALL), (F(1, 2), -BIG))
    solution = backend().solve(_problem(sources, Grounded()))
    result = solution.conductor_results[0]
    assert result.outer_surface_charge_c == 0.0
    assert result.potential_v == 0.0
    assert solution.potential_at((5., 1., 0.)) == 0.0
    assert solution.electric_field_at((5., 1., 0.)) == (0., 0., 0.)
    # A mirror-symmetric pair produces an exact zero field at the centre.
    pair = backend().solve(_problem(((F(-1, 2), BIG), (F(1, 2), BIG)), Grounded()))
    assert pair.electric_field_at((0., 0., 0.)) == (0., 0., 0.)


def test_nested_fixed_chain_keeps_residual_in_every_flux():
    shells = (SphericalShell(1., 2., FixedNetCharge(-float(BIG)), name="inner"),
              SphericalShell(3., 4., FixedNetCharge(0.), name="outer"))
    charges = (PointCharge(float(BIG), (-.5, 0., 0.)), PointCharge(float(SMALL), (.25, 0., 0.)))
    solution = NestedShellBackend().solve(ElectrostaticProblem(charges, shells))
    inner, outer = (solution.conductor_by_name(n) for n in ("inner", "outer"))
    assert inner.outer_surface_charge_c == float(SMALL)
    assert outer.inner_surface_charge_c == -float(SMALL)
    assert outer.outer_surface_charge_c == float(SMALL)
    assert isclose(solution.potential_at((8., 0., 0.)), K*float(SMALL)/8, rel_tol=1e-15)
    # Gap field between the shells is from the enclosed residual alone.
    assert isclose(solution.electric_field_at((2.5, 0., 0.))[0], K*float(SMALL)/6.25,
                   rel_tol=1e-15)


def test_nested_grounded_outer_shell_keeps_zero_outer_flux():
    shells = (SphericalShell(1., 2., FixedNetCharge(-float(BIG)), name="inner"),
              SphericalShell(3., 4., Grounded(), name="outer"))
    charges = (PointCharge(float(BIG), (-.5, 0., 0.)), PointCharge(float(SMALL), (.25, 0., 0.)))
    solution = NestedShellBackend().solve(ElectrostaticProblem(charges, shells))
    outer = solution.conductor_by_name("outer")
    assert outer.outer_surface_charge_c == 0.0
    assert outer.potential_v == 0.0
    assert solution.potential_at((8., 0., 0.)) == 0.0
    assert solution.conductor_by_name("inner").outer_surface_charge_c == float(SMALL)
    # The grounded charge balances the exact enclosed residual (Gauss).
    assert outer.inner_surface_charge_c == -float(SMALL)
    assert outer.net_charge_c == -float(SMALL)


@pytest.mark.parametrize("backend", BACKENDS)
def test_ordinary_cavity_values_match_exact_kelvin(backend):
    # Compatibility: an ordinary off-centre pair in an isolated charged shell.
    q1, q2, net = F(3, 10**9), F(-1, 10**9), F(2, 10**9)
    sources = ((F(-3, 10), q1), (F(2, 5), q2))
    solution = backend().solve(_problem(sources, FixedNetCharge(float(net))))
    b = F(2)
    wall = (q1 + q2 + net)/b
    for probe in (F(0), F(1, 10), F(-7, 10)):
        v, e = (sum(t) for t in zip(*(_axis_terms(x, q, F(1), probe) for x, q in sources)))
        assert isclose(solution.potential_at((float(probe), 0., 0.)), K*float(v + wall),
                       rel_tol=1e-14)
        assert isclose(solution.electric_field_at((float(probe), 0., 0.))[0], K*float(e),
                       rel_tol=1e-14)
