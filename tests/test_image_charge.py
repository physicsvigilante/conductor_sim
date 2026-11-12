"""Independent exact image-charge reference, no FEMM import/execution."""
from math import cos, pi, sin

import pytest

from conductor_sim import (
    ConcentricShellBackend, ElectrostaticProblem, FixedNetCharge, Grounded,
    PointCharge, SphericalShell, UnsupportedProblem,
)
from conductor_sim.image_charge import ImageChargeShellBackend
from conductor_sim.solver import SingularField

Q, A, B = 1e-9, 0.1, 0.12


def problem(constraint=FixedNetCharge(0), pos=(0, 0, .04), q=Q):
    return ElectrostaticProblem(
        charges=(PointCharge(q, pos),),
        conductors=(SphericalShell(A, B, constraint),),
    )


def test_exact_grounded_offcentre_boundary_value_and_charge():
    sol = ImageChargeShellBackend().solve(problem(Grounded()))
    c = sol.conductor_results[0]
    assert c.net_charge_c == pytest.approx(-Q)
    assert c.potential_v == pytest.approx(0)
    assert c.inner_surface_charge_c == pytest.approx(-Q)
    assert c.outer_surface_charge_c == pytest.approx(0)
    for theta in (0., .4, 1., 1.7, 2.8, pi):
        p = (A*sin(theta), 0., A*cos(theta))
        assert sol.potential_at(p) == pytest.approx(0, abs=1e-10)
    assert sol.potential_at((.2,0,0)) == pytest.approx(0)


def test_exact_isolated_shell_exterior_and_cavity_composition():
    neutral = ImageChargeShellBackend().solve(problem())
    charged = ImageChargeShellBackend().solve(problem(FixedNetCharge(2e-9)))
    for p in [(.03, 0, 0), (0, 0, -.05), (0, 0, .075)]:
        assert charged.potential_at(p) - neutral.potential_at(p) == pytest.approx(
            charged.conductor_results[0].potential_v - neutral.conductor_results[0].potential_v,
            abs=1e-10,
        )
        assert charged.electric_field_at(p) == pytest.approx(neutral.electric_field_at(p))
    assert neutral.electric_field_at((0,0,.11)) == (0,0,0)
    assert neutral.potential_at((.24,0,0)) == pytest.approx(
        charged.potential_at((.24,0,0))/3
    )


def test_inner_density_nonuniform_but_integrates_to_minus_q():
    sol = ImageChargeShellBackend().solve(problem())
    near = sol.surface_charge_density("inner", 0)
    far = sol.surface_charge_density("inner", pi)
    assert near < far < 0
    assert abs(near/far) > 10  # (1.4/.6)^3
    assert sol.surface_charge_density("outer", 0) == pytest.approx(
        sol.surface_charge_density("outer", pi)
    )
    n = 2000
    h = pi/n
    integral = 0.0
    for k in range(1, n):
        t = k*h
        integral += sol.surface_charge_density("inner", t)*sin(t)
    integral *= 2*pi*A*A*h
    assert integral == pytest.approx(-Q, rel=5e-5)


def test_exact_centered_limit_matches_original_reference():
    p = problem(pos=(0,0,0))
    new = ImageChargeShellBackend().solve(p)
    old = ConcentricShellBackend().solve(p)
    for pt in [(0.025,0,0), (0,0,.07), (.11,0,0), (.2,0,0)]:
        assert new.potential_at(pt) == pytest.approx(old.potential_at(pt))
        assert new.electric_field_at(pt) == pytest.approx(old.electric_field_at(pt))
    assert new.surface_charge_density("inner", 0) == pytest.approx(
        new.surface_charge_density("inner", pi)
    )


def test_reference_world_rotation_and_translation_covariant():
    a = ImageChargeShellBackend().solve(problem(pos=(.04,0,0)))
    b = ImageChargeShellBackend().solve(problem(pos=(0,0,.04)))
    for x, z in [(0.06,0.02),(-.025,.01),(.08,.01)]:
        p1=(x,0,z);p2=(z,0,x)
        assert a.potential_at(p1) == pytest.approx(b.potential_at(p2))
        ea=a.electric_field_at(p1);eb=b.electric_field_at(p2)
        assert ea == pytest.approx((eb[2], eb[1], eb[0]))


def test_reference_rejects_multiple_and_outside_charge():
    backend = ImageChargeShellBackend()
    bad = [
        problem(pos=(0,0,.1)),
        problem(pos=(0,0,.15)),
        ElectrostaticProblem(charges=(PointCharge(Q), PointCharge(-Q)), conductors=problem().conductors),
    ]
    for p in bad:
        assert not backend.supports(p)
        with pytest.raises(UnsupportedProblem):
            backend.solve(p)
    exact=backend.solve(problem())
    with pytest.raises(SingularField):
        exact.potential_at((0,0,.04))
    with pytest.raises(SingularField):
        exact.electric_field_at((0,0,.1))
    with pytest.raises(ValueError):
        exact.surface_charge_density("inner", -1)
    with pytest.raises(ValueError):
        exact.surface_charge_density("metal", .7)
