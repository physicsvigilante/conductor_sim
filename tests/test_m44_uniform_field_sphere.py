"""Independent multipole, Kelvin, surface flux, and numerical checks for M44."""

from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction as F
from math import inf, nextafter, pi, ulp

import pytest

from conductor_sim import (
    EPSILON_0, FixedNetCharge, FixedPotential, Grounded, SingularField,
    SphericalCavity, SphericalConductorProblem, SphericalPointSource, SphericalRegion,
    UniformBackgroundField, UnsupportedProblem, force_energy, solve_spherical_conductor,
)

ZERO = (0., 0., 0.)
FOUR_PI = F(4 * pi)


def solve(radius=5., constraint=Grounded(), *, E=(2., -3., 4.), ref=None, **geometry):
    return solve_spherical_conductor(SphericalConductorProblem(
        radius, constraint, background_field=UniformBackgroundField(E, ref), **geometry,
    ))


def test_grounded_shifted_sphere_gauge_and_exact_wall():
    center, ref, E = (8., -2., 7.), (4., 0., 2.), (2., -3., 4.)
    solution = solve(center_m=center, E=E, ref=ref)
    expected_q = FOUR_PI * F(EPSILON_0) * 5 * 34
    assert solution.outer_surface_charge_c == solution.net_charge_c == float(expected_q)
    assert solution.background_induced_dipole_c_m == tuple(
        float(FOUR_PI * F(EPSILON_0) * 5**3 * e) for e in E
    )
    assert solution.region_at((11., 2., 7.)) == SphericalRegion("outer_wall")
    assert solution.potential_at((11., 2., 7.)) == solution.potential_v == 0.
    assert solution.electric_field_at(center) == ZERO
    with pytest.raises(SingularField, match="one-sided"):
        solution.electric_field_at((11., 2., 7.))
    # The reference pins the background potential. The response is present there too.
    assert solution.potential_at(ref) != 0.
    centered = solve(center_m=center, E=E)
    assert centered.outer_surface_charge_c == 0.
    assert centered.outer_wall_force_n == ZERO


def test_fixed_potential_exact_rational_multipoles_and_pressure():
    solution = solve(constraint=FixedPotential(7.), permittivity_f_per_m=.25)
    r, radius, E = F(10), F(5), (F(2), F(-3), F(4))
    assert solution.potential_at((0., 0., 10.)) == float(7 * radius/r - E[2] * r * (1-(radius/r)**3))
    assert solution.electric_field_at((0., 0., 10.)) == tuple(map(float, (
        E[0] * (1-(radius/r)**3),
        E[1] * (1-(radius/r)**3),
        E[2] * (1+2*(radius/r)**3) + 7 * radius/r**2,
    )))
    wall = (3., 4., 0.)
    # n = (3/5, 4/5, 0). The normal field includes both the imposed voltage and polarization.
    normal = (F(3, 5), F(4, 5), F(0))
    en = F(7, 5) + 3 * (E[0]*normal[0] + E[1]*normal[1])
    assert solution.outer_wall_electric_field_at(wall) == tuple(float(en * n) for n in normal)
    assert solution.outer_surface_charge_density_at(wall) == float(en / 4)
    assert solution.outer_wall_pressure_at(wall) == float(en**2 / 8)
    assert solution.potential_at(wall) == 7.
    assert solution.outer_surface_charge_c == float(FOUR_PI * F(1, 4) * 5 * 7)


def test_cavity_shielding_and_outer_force_with_held_cavity_sources():
    cavities = (SphericalCavity("A", ZERO, 1.), SphericalCavity("B", (3., 0., 0.), .5))
    sources = (SphericalPointSource("inside", 4e-9, (.25, .25, 0.)),)
    args = dict(cavities=cavities, sources=sources, constraint=FixedNetCharge(-1e-9))
    field = solve(**args, E=(4., 2., -8.), ref=(2., 1., 0.))
    zero = solve(**args, E=ZERO)
    assert field.inner_surface_charge_c == zero.inner_surface_charge_c == {"A": -4e-9, "B": 0.}
    assert field.outer_surface_charge_c == zero.outer_surface_charge_c == float(3 * F(1e-9))
    assert field.net_charge_c == -1e-9
    for probe in ((0., .125, 0.), cavities[1].center_m):
        assert field.electric_field_at(probe) == zero.electric_field_at(probe)
        assert field.potential_at(probe) - zero.potential_at(probe) == pytest.approx(10.)
    assert field.potential_at((1., 0., 0.)) == field.potential_v
    with pytest.raises(SingularField):
        field.electric_field_at((1., 0., 0.))
    with pytest.raises(SingularField):
        field.potential_at(sources[0].position_m)
    assert field.outer_wall_force_n == pytest.approx((12e-9, 6e-9, -24e-9), rel=1e-15, abs=0)
    # The six-axis rule integrates the cubic traction integrand exactly.
    for k in range(3):
        plus = tuple(5. if i == k else 0. for i in range(3))
        minus = tuple(-x for x in plus)
        stress_force = (4*pi*25/6) * (field.outer_wall_pressure_at(plus) - field.outer_wall_pressure_at(minus))
        assert stress_force == pytest.approx(field.outer_wall_force_n[k], rel=1e-12, abs=0)


def test_exterior_kelvin_superposition_and_fixed_charge_balancing_monopole():
    E = (3., -2., 1.)
    source = SphericalPointSource("outside", 5e-9, (8., 8., 4.))
    Q = -2e-9
    solution = solve(4., FixedNetCharge(Q), E=E, sources=(source,))
    probe = (5., 7., 3.)
    with localcontext() as ctx:
        ctx.prec = 80
        R = Decimal(4)
        k = 1 / (Decimal(4*pi) * Decimal(EPSILON_0))
        x = tuple(map(Decimal, probe))
        s = tuple(map(Decimal, source.position_m))
        d = sum(t*t for t in s).sqrt()
        q = Decimal(source.charge_c)
        image = tuple(t * (R/d)**2 for t in s)
        r = sum(t*t for t in x).sqrt()
        p_dot_x = sum(Decimal(e)*t for e, t in zip(E, x))
        voltage = k * (Decimal(Q)/R + q/d)
        central_q = R*voltage/k
        v = -p_dot_x * (1-(R/r)**3)
        e = [Decimal(a)*(1-(R/r)**3) + 3*R**3*p_dot_x*t/r**5 for a, t in zip(E, x)]
        for charge, position in ((q, s), (-q*R/d, image), (central_q, (Decimal(0),)*3)):
            dx = tuple(a-b for a, b in zip(x, position))
            distance = sum(t*t for t in dx).sqrt()
            v += k*charge/distance
            for i in range(3):
                e[i] += k*charge*dx[i]/distance**3
    assert solution.potential_v == float(voltage)
    assert solution.outer_surface_charge_c == Q
    assert solution.potential_at(probe) == float(v)
    assert solution.electric_field_at(probe) == tuple(map(float, e))
    with pytest.raises(UnsupportedProblem, match="exterior point charges"):
        _ = solution.outer_wall_force_n
    with pytest.raises(UnsupportedProblem, match="force"):
        force_energy(solution)


@pytest.mark.parametrize("constraint", [FixedNetCharge(.5), FixedPotential(2.)])
def test_signed_rotation_and_translation_invariance(constraint):
    center, field, ref = (2., 3., 4.), (1., 2., -4.), (0., 0., -3.)
    source = SphericalPointSource("outside", .75, (10., 9., 4.))
    kwargs = dict(center_m=center, E=field, ref=ref, sources=(source,), permittivity_f_per_m=.5)
    solution = solve(5., constraint, **kwargs)

    def rotate(p):
        return (-p[2], p[0], -p[1])

    def transform(p):
        return tuple(x + a for x, a in zip(rotate(p), (8., -3., 2.)))

    if isinstance(constraint, FixedNetCharge):
        opposite = FixedNetCharge(-constraint.charge_c)
    else:
        opposite = FixedPotential(-constraint.potential_v)
    transformed = solve(5., opposite, center_m=transform(center), ref=transform(ref),
                        E=tuple(-x for x in rotate(field)), permittivity_f_per_m=.5,
                        sources=(replace(source, charge_c=-.75, position_m=transform(source.position_m)),))
    probe = (9., 6., 1.)
    wall = (5., 7., 4.)
    assert transformed.potential_at(transform(probe)) == -solution.potential_at(probe)
    assert transformed.electric_field_at(transform(probe)) == tuple(
        -x for x in rotate(solution.electric_field_at(probe))
    )
    assert transformed.outer_surface_charge_density_at(transform(wall)) == -solution.outer_surface_charge_density_at(wall)
    assert transformed.outer_wall_pressure_at(transform(wall)) == solution.outer_wall_pressure_at(wall)


def test_outer_sigma_flux_with_exterior_charge():
    # Sample the same dimensionless problem at integer points on exact spheres.
    # For cos(theta)=(1-t²)/(1+t²), the area weight in d(cos(theta)) is 4t/(1+t²)².
    n = 96
    mean_terms = []
    charge_per_r_sq = F(7, 8)
    for i in range(n+1):
        radius = n*n + i*i
        pz = n*n - i*i
        solution = solve(radius, FixedNetCharge(float(charge_per_r_sq * radius**2)), E=(0., 0., 1.),
                         permittivity_f_per_m=.125,
                         sources=(SphericalPointSource("ext", -2. * radius**2, (0., 0., 3.*radius)),))
        pair = sum(solution.outer_surface_charge_density_at((2.*n*i, 0., sign*pz)) for sign in (1, -1))
        t = i / n
        weight = 1 if i in (0, n) else (4 if i % 2 else 2)
        mean_terms.append(weight * pair * (4*t/(1+t*t)**2))
    assert 2*pi * sum(mean_terms)/(3*n) == pytest.approx(float(charge_per_r_sq), rel=1e-7)


def test_one_ulp_wall_limit_and_exact_fraction_witness():
    radius = F(1)
    x = F(nextafter(1., inf))
    solution = solve(1., E=(2., -4., 1.))
    assert solution.region_at((float(x), 0., 0.)).kind == "exterior"
    assert solution.potential_at((float(x), 0., 0.)) == float(-2*x*(1-(radius/x)**3))
    assert solution.electric_field_at((float(x), 0., 0.))[1] == float(-4*(1-(radius/x)**3))
    assert solution.outer_wall_electric_field_at((1., 0., 0.)) == (6., 0., 0.)
    assert solution.electric_field_at((nextafter(1., 0.), 0., 0.)) == ZERO
    # hypot(1, 2^-400) rounds to one. The tangential E deficit is still representable.
    probe = (1., 2.**-400, 0.)
    assert solution.region_at(probe).kind == "exterior"
    assert solution.electric_field_at(probe)[2] == 1.5 * 2.**-800


def test_near_wall_kelvin_pair_one_ulp_from_wall():
    d = F(nextafter(1., inf))
    solution = solve(1., E=ZERO, sources=(SphericalPointSource("q", 1e-9, (float(d), 0., 0.)),))
    C = F(1e-9) / (FOUR_PI * F(EPSILON_0))
    assert solution.potential_at((-2., 0., 0.)) == float(C*(1/(2+d) - 1/(2*d+1)))
    assert solution.outer_surface_charge_density_at((-1., 0., 0.)) == float(
        -F(1e-9)*(d*d-1)/(FOUR_PI*(1+d)**3)
    )
    near_source = (nextafter(float(d), inf), 0., 0.)
    assert solution.electric_field_at(near_source)[0] > 0.


def test_extreme_scales_keep_finite_observables_and_refuse_overflow_on_query():
    solution = solve(2.**-1025, E=(2.**1023, 0., 0.))
    point = (2.**-1024, 0., 0.)
    assert solution.potential_at(point) == -7/16
    assert solution.electric_field_at(point) == (1.25 * 2.**1023, 0., 0.)
    with pytest.raises(UnsupportedProblem, match="electric field component"):
        solution.outer_wall_electric_field_at((2.**-1025, 0., 0.))
    with pytest.raises(UnsupportedProblem, match="dipole"):
        _ = solution.background_induced_dipole_c_m
    pressure = solve(1., E=(1e50, 0., 0.), permittivity_f_per_m=ulp(0.))
    assert pressure.outer_wall_pressure_at((1., 0., 0.)) == float(F(9, 2) * F(ulp(0.)) * F(1e50)**2)
    # Each term of the center-reference dot product overflows on its own.
    center, ref = (0., 0., 0.), (1e308, 1e308, 0.)
    zero_gauge = solve(1., FixedNetCharge(0.), center_m=center, ref=ref, E=(1e308, -1e308, 0.))
    assert zero_gauge.potential_v == 0.
    with pytest.raises(UnsupportedProblem, match="local potential"):
        zero_gauge.potential_at((1e308, -1e308, 0.))
    far = solve(1., center_m=(-1e308, 0., 0.), E=(2.**-1023, 0., 0.))
    assert far.potential_at((1e308, 0., 0.)) == -2.**-1022 * 1e308


def test_gauge_and_charge_residuals_survive_separate_overflow_and_rounding():
    big = 2.**70
    # The center-reference offset E0*(c-r_ref) is below the ulp of each large product.
    solution = solve(2., E=(big, big, 1.), ref=(1., -1., -1.))
    assert solution.outer_surface_charge_c == float(2 * FOUR_PI * F(EPSILON_0))
    # The exterior potential keeps its small background variation even when the
    # metal-to-background voltage difference exceeds the binary64 range.
    R = 2.**-600
    solution = solve(R, FixedPotential(1e308), E=(1e308, 0., 0.), ref=(-1., 0., 0.))
    assert solution.outer_surface_charge_c == float(FOUR_PI * F(EPSILON_0) * F(R) * 2 * F(1e308))
    assert solution.potential_at((R, 0., 0.)) == 1e308
    assert solution.potential_at((2*R, 0., 0.)) == float(-F(1e308) * (2*F(R) - F(R)/4))


def test_equivalent_irrational_distances_cancel_from_the_charge_ledger():
    sources = (SphericalPointSource("a", 1., (2., 2., 0.)),
               SphericalPointSource("b", -2., (4., 4., 0.)))
    solution = solve(1., sources=sources)
    assert solution.net_charge_c == solution.outer_surface_charge_c == 0.
    assert solve(1., FixedNetCharge(0.), sources=sources).potential_v == 0.


@pytest.mark.parametrize("background", [(1., 2., 3.), object()])
def test_invalid_background_type(background):
    with pytest.raises(UnsupportedProblem, match="UniformBackgroundField"):
        SphericalConductorProblem(1., Grounded(), background_field=background)


@pytest.mark.parametrize("E,ref", [((1, 2), None), ((inf, 0, 0), None), (ZERO, (10**400, 0, 0))])
def test_invalid_background_vectors(E, ref):
    with pytest.raises(UnsupportedProblem, match="finite"):
        UniformBackgroundField(E, ref)


def test_geometry_surface_and_constraint_refusals():
    with pytest.raises(UnsupportedProblem, match="radius"):
        solve(inf)
    with pytest.raises(UnsupportedProblem, match="constraint|requires"):
        solve(constraint=object())
    with pytest.raises(UnsupportedProblem, match="touch or overlap"):
        solve(cavities=(SphericalCavity("A", ZERO, 1.), SphericalCavity("B", (2., 0., 0.), 1.)))
    with pytest.raises(UnsupportedProblem, match="wall or inside metal"):
        solve(sources=(SphericalPointSource("wall", 0., (5., 0., 0.)),))
    with pytest.raises(UnsupportedProblem, match="outer wall"):
        solve().outer_surface_charge_density_at((nextafter(5., inf), 0., 0.))
    with pytest.raises(UnsupportedProblem, match="sample point"):
        solve().potential_at((10**400, 0., 0.))
    with pytest.raises(UnsupportedProblem, match="charge outside"):
        solve(1e300, FixedPotential(1e300), permittivity_f_per_m=1e100)


def test_zero_field_is_explicit_opt_in_for_nonzero_fixed_potential():
    plain = SphericalConductorProblem(1., FixedPotential(2.))
    with pytest.raises(UnsupportedProblem, match="nonzero FixedPotential"):
        solve_spherical_conductor(plain)
    assert solve_spherical_conductor(replace(plain, background_field=UniformBackgroundField(ZERO))).potential_v == 2.


def test_review_regressions_vector_orthogonal_underflow_and_bool_rejection():
    # Far-field off-axis probe: E_x = 1.0 V/m while orthogonal dipole component E_y ~ 3e-400 underflows to 0.0
    s_far = solve(1., E=(1., 0., 0.))
    assert s_far.electric_field_at((1e100, 1., 0.)) == (1., 0., 0.)
    # Legacy M42a far-field probe retains finite radial component when orthogonal component underflows
    s_leg = solve_spherical_conductor(SphericalConductorProblem(1., FixedNetCharge(1e-9)))
    e_leg = s_leg.electric_field_at((1e100, 1e-150, 0.))
    assert e_leg[0] == pytest.approx(1e-9 / (4 * pi * EPSILON_0 * 1e200), rel=1e-14)
    assert e_leg[1] == 0. and e_leg[2] == 0.
    # Force and dipole keep dominant finite component when orthogonal component underflows
    s_force = solve(1., FixedNetCharge(1e-200), E=(1., 1e-200, 0.))
    assert s_force.outer_wall_force_n == (1e-200, 0., 0.)
    s_dip = solve(1e-100, E=(1., 1e-50, 0.))
    assert s_dip.background_induced_dipole_c_m[0] > 0. and s_dip.background_induced_dipole_c_m[1] == 0.
    # Bool coordinates in UniformBackgroundField or wall/sample probes are refused explicitly
    with pytest.raises(UnsupportedProblem, match="finite"):
        UniformBackgroundField((True, 0., 0.))
    with pytest.raises(UnsupportedProblem, match="finite"):
        UniformBackgroundField(ZERO, (False, 0., 0.))
    with pytest.raises(UnsupportedProblem, match="finite"):
        s_far.outer_surface_charge_density_at((True, 0., 0.))
