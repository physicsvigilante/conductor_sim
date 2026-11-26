"""Independent Kelvin, Gauss, cancellation, and legacy checks for M42a."""

from dataclasses import replace
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
from math import cos, dist, hypot, isclose, nextafter, pi, sin, sqrt, ulp

import pytest

from conductor_sim import (
    EPSILON_0, ConcentricShellBackend, ElectrostaticProblem,
    ExternalGroundedSphereBackend, ExternalIsolatedSphereBackend, ExternalMultiSphereBackend,
    FixedNetCharge, FixedPotential, Grounded, ImageChargeShellBackend, MultiImageShellBackend,
    PointCharge, SingularField, SolidSphere, SphericalCavity, SphericalConductorProblem,
    SphericalPointSource, SphericalRegion, SphericalShell, UnsupportedProblem, force_energy,
    solve_spherical_conductor,
)
from conductor_sim.cavity_gauss import CavityCharge, GaussCavityAssembly, SealedCavity
from conductor_sim.spherical_multicavity import (
    NamedSourcePosition, SphericalMetalGeometry, SphericalSigmaModel, SphericalVoid,
    solve_spherical_sigma,
)


ZERO = (0., 0., 0.)
K = 1 / (4 * pi * EPSILON_0)
NC = 1e-9


def _source(name, q, x, y=0., z=0.):
    return SphericalPointSource(name, q, (x, y, z))


def _scalar(actual, expected, tolerance=1e-9):
    assert isclose(actual, float(expected), rel_tol=tolerance), (actual, expected)


def _vector(actual, expected, tolerance=1e-9):
    expected = tuple(map(float, expected))
    assert dist(actual, expected) <= tolerance * hypot(*expected), (actual, expected)


def _kelvin(center, radius, sources, voltage, epsilon, point, *, exterior):
    """High-precision real and explicit inverted charges in world coordinates."""
    with localcontext() as ctx:
        ctx.prec = 65
        c, p = tuple(map(D, center)), tuple(map(D, point))
        radius = D(radius)
        k = 1 / (D(4) * D(pi) * D(epsilon))
        charges = []
        constant = D(0)
        for src in sources:
            q, s = D(src.charge_c), tuple(map(D, src.position_m))
            offset = tuple(x - y for x, y in zip(s, c))
            d2 = sum(x*x for x in offset)
            charges.append((q, s))
            if d2:
                image = tuple(x + radius**2 * v / d2 for x, v in zip(c, offset))
                charges.append((-q * radius / d2.sqrt(), image))
            else:
                constant -= k * q / radius
        if exterior:
            charges.append((D(voltage) * radius / k, c))
        else:
            constant += D(voltage)
        electric = [D(0)] * 3
        for q, s in charges:
            delta = tuple(x - y for x, y in zip(p, s))
            norm = sum(x*x for x in delta).sqrt()
            constant += k * q / norm
            for i, component in enumerate(delta):
                electric[i] += k * q * component / norm**3
        return constant, tuple(electric)


def _state_oracle(problem, cavity_sources, exterior_sources):
    with localcontext() as ctx:
        ctx.prec = 65
        wall = tuple(map(D, problem.center_m))
        radius = D(problem.outer_radius_m)
        inner = {}
        for cavity_id, sources in cavity_sources.items():
            exact = -sum((F(s.charge_c) for s in sources), F())
            inner[cavity_id] = D(exact.numerator) / D(exact.denominator)
        induced = -sum((
            radius * D(s.charge_c)
            / sum((D(x) - y)**2 for x, y in zip(s.position_m, wall)).sqrt()
            for s in exterior_sources
        ), D(0))
        if isinstance(problem.constraint, Grounded):
            return D(0), induced + sum(inner.values()), induced, inner
        metal = D(problem.constraint.charge_c)
        outer = metal - sum(inner.values())
        voltage = (outer - induced) / (D(4) * D(pi) * D(problem.permittivity_f_per_m) * radius)
        return voltage, metal, outer, inner


@pytest.mark.parametrize("constraint", (FixedNetCharge(0.), Grounded()), ids=("G1", "G1b"))
def test_hollow_shell_empty_cavity_and_exterior_source(constraint):
    charge = _source("outside", 3 * NC, 5.)
    problem = SphericalConductorProblem(
        2., constraint, cavities=(SphericalCavity("air", ZERO, 1.),), sources=(charge,),
    )
    result = solve_spherical_conductor(problem)
    expected_outer = -2 * charge.charge_c / 5 if isinstance(constraint, Grounded) else 0.
    voltage = 0. if isinstance(constraint, Grounded) else K * charge.charge_c / 5
    _scalar(result.potential_v, voltage)
    _scalar(result.outer_surface_charge_c, expected_outer)
    _scalar(result.net_charge_c, expected_outer)
    assert result.inner_surface_charge_c == {"air": 0.}
    assert result.source_regions == {"outside": SphericalRegion("exterior")}
    for point, kind, name in ((ZERO, "cavity", "air"), ((1.5, 0., 0.), "metal", None),
                              ((2., 0., 0.), "outer_wall", None), ((1., 0., 0.), "cavity_wall", "air")):
        assert result.region_at(point) == SphericalRegion(kind, name)
        assert result.potential_at(point) == result.potential_v
        if kind.endswith("wall"):
            with pytest.raises(SingularField, match="one-sided"):
                result.electric_field_at(point)
        else:
            assert result.electric_field_at(point) == ZERO


@pytest.mark.parametrize("constraint", (FixedNetCharge(2 * NC), Grounded()))
def test_g2_eccentric_charge_with_exterior_charge(constraint):
    cavity = SphericalCavity("air", (.25, -.375, .125), .75)
    inside = _source("inside", 1.5 * NC, .25, -.125, .25)
    outside = _source("outside", -4 * NC, 0., 0., 4.)
    problem = SphericalConductorProblem(2., constraint, cavities=(cavity,), sources=(inside, outside))
    result = solve_spherical_conductor(problem)
    before = solve_spherical_conductor(replace(problem, sources=(inside,)))
    voltage, net, outer, inner = _state_oracle(problem, {"air": (inside,)}, (outside,))
    _scalar(result.potential_v, voltage)
    _scalar(result.net_charge_c, net)
    _scalar(result.outer_surface_charge_c, outer)
    _scalar(result.inner_surface_charge_c["air"], inner["air"])
    delta_v = 0. if isinstance(constraint, Grounded) else K * outside.charge_c / 4
    _scalar(result.potential_v - before.potential_v, delta_v)
    for point in ((.5, -.5, .25), cavity.center_m):
        expected_v, expected_e = _kelvin(cavity.center_m, cavity.radius_m, (inside,), voltage,
                                         EPSILON_0, point, exterior=False)
        _scalar(result.potential_at(point), expected_v)
        _vector(result.electric_field_at(point), expected_e)
        _scalar(result.potential_at(point) - before.potential_at(point), delta_v)
        assert result.electric_field_at(point) == before.electric_field_at(point)
    point = (-3., .25, 2.)
    expected_v, expected_e = _kelvin(ZERO, 2., (outside,), voltage, EPSILON_0, point, exterior=True)
    _scalar(result.potential_at(point), expected_v)
    _vector(result.electric_field_at(point), expected_e)


@pytest.mark.parametrize("constraint", (FixedNetCharge(-2 * NC), Grounded()))
def test_static_relocation_between_disjoint_cavities(constraint):
    a = SphericalCavity("A", (-1., 0., 0.), .5)
    b = SphericalCavity("B", (1.25, .5, 0.), .625)
    moved_from = _source("moved", 3 * NC, -1.25, 0., .125)
    moved_to = replace(moved_from, position_m=(1.125, .5, .25))
    external = _source("held", 5 * NC, 4., 2.)
    problem = SphericalConductorProblem(3., constraint, cavities=(a, b), sources=(moved_from, external))
    for source, charged, empty in ((moved_from, a, b), (moved_to, b, a)):
        result = solve_spherical_conductor(replace(problem, sources=(source, external)))
        assert result.inner_surface_charge_c == {charged.cavity_id: -source.charge_c, empty.cavity_id: 0.}
        assert result.source_regions["moved"] == SphericalRegion("cavity", charged.cavity_id)
        assert result.electric_field_at(empty.center_m) == ZERO
        assert result.potential_at(empty.center_m) == result.potential_v
        expected_v, expected_e = _kelvin(charged.center_m, charged.radius_m, (source,),
                                         result.potential_v, EPSILON_0, charged.center_m, exterior=False)
        _scalar(result.potential_at(charged.center_m), expected_v)
        _vector(result.electric_field_at(charged.center_m), expected_e)
        if source is moved_from:
            before = result
    point = (4.5, -1., 3.)
    assert result.potential_v == before.potential_v
    assert result.net_charge_c == before.net_charge_c
    assert result.outer_surface_charge_c == before.outer_surface_charge_c
    assert result.potential_at(point) == before.potential_at(point)
    assert result.electric_field_at(point) == before.electric_field_at(point)


# Positive abscissas and weights of the 16-point Gauss-Legendre rule.
_LATITUDES = (
    (.09501250983763744, .18945061045506850), (.28160355077925891, .18260341504492359),
    (.45801677765722739, .16915651939500254), (.61787624440264375, .14959598881657673),
    (.75540440835500303, .12462897125553387), (.86563120238783174, .09515851168249278),
    (.94457502307323258, .06225352393864789), (.98940093499164993, .02715245941175409),
)


def _gauss_charge(solution, center, radius):
    weighted_flux = []
    for node, weight in _LATITUDES:
        for z in (-node, node):
            transverse = sqrt(1 - z*z)
            for j in range(32):
                azimuth = 2*pi*j/32
                n = (transverse*cos(azimuth), transverse*sin(azimuth), z)
                point = tuple(c + radius*u for c, u in zip(center, n))
                e = solution.electric_field_at(point)
                weighted_flux.append(weight * sum(u*v for u, v in zip(n, e)))
    return solution.problem.permittivity_f_per_m * radius**2 * (2*pi/32) * sum(weighted_flux)


def test_shifted_rotated_three_cavity_composition_via_public_api():
    # Exact orthogonal rotation with axis (1,1,1) and an independent translation.
    def rotate(p):
        x, y, z = p
        return ((2*x-y+2*z)/3, (2*x+2*y-z)/3, (-x+2*y+2*z)/3)

    shift = (7.25, -5.5, 3.125)

    def transform(p):
        return tuple(a + b for a, b in zip(rotate(p), shift))

    cavities = (
        SphericalCavity("empty", (-1.625, .375, .125), .625),
        SphericalCavity("dipole", (.5, 1.125, -.25), .875),
        SphericalCavity("net", (.875, -.875, .625), .75),
    )
    sources = (
        _source("plus", 1.5*NC, .5625, 1.0, -.1875),
        _source("minus", -1.5*NC, .4375, 1.25, -.375),
        _source("in", 2.5*NC, .875, -.8125, .75),
        _source("out1", 4.25*NC, 31., 14., -8.),
        _source("out2", -1.125*NC, 8., -20., 16.),
    )
    problem = SphericalConductorProblem(
        3.5, FixedNetCharge(-.625*NC), cavities=cavities, sources=sources,
        permittivity_f_per_m=2.5*EPSILON_0,
    )
    rotated = replace(problem, center_m=shift,
                      cavities=tuple(replace(c, center_m=transform(c.center_m)) for c in cavities),
                      sources=tuple(replace(s, position_m=transform(s.position_m)) for s in sources))
    original = solve_spherical_conductor(problem)
    solution = solve_spherical_conductor(rotated)
    region_sources = {"empty": (), "dipole": rotated.sources[:2], "net": rotated.sources[2:3]}
    external = rotated.sources[3:]
    voltage, net, outer, inner = _state_oracle(rotated, region_sources, external)
    _scalar(solution.potential_v, voltage)
    _scalar(solution.net_charge_c, net)
    _scalar(solution.outer_surface_charge_c, outer)
    for cavity_id, expected in inner.items():
        _scalar(solution.inner_surface_charge_c[cavity_id], expected)
    assert tuple(solution.problem.sources) == rotated.sources
    assert solution.source_regions == {
        "plus": SphericalRegion("cavity", "dipole"), "minus": SphericalRegion("cavity", "dipole"),
        "in": SphericalRegion("cavity", "net"),
        "out1": SphericalRegion("exterior"), "out2": SphericalRegion("exterior"),
    }
    _scalar(solution.potential_v, original.potential_v)
    _scalar(solution.outer_surface_charge_c, original.outer_surface_charge_c)
    probes = ((-1.75, .625, .125), (.4, 1.5, -.5), (1., -.625, .375), (4.25, 2.5, -1.))
    for probe, cavity in zip(probes, (*rotated.cavities, None)):
        p = transform(probe)
        if cavity is None:
            center, radius, own = shift, problem.outer_radius_m, external
            region = SphericalRegion("exterior")
        else:
            center, radius, own = cavity.center_m, cavity.radius_m, region_sources[cavity.cavity_id]
            region = SphericalRegion("cavity", cavity.cavity_id)
        assert solution.region_at(p) == region
        v, e = _kelvin(center, radius, own, voltage, problem.permittivity_f_per_m,
                       p, exterior=cavity is None)
        _scalar(solution.potential_at(p), v)
        _vector(solution.electric_field_at(p), e)
        _scalar(solution.potential_at(p), original.potential_at(probe))
        _vector(solution.electric_field_at(p), rotate(original.electric_field_at(probe)))
        if cavity is None:
            _scalar(solution.electric_field_at(p)[2], e[2])

    assert solution.region_at(shift) == SphericalRegion("metal")
    assert solution.potential_at(shift) == solution.potential_v
    assert solution.electric_field_at(shift) == ZERO
    walls = ((shift, problem.outer_radius_m, SphericalRegion("outer_wall")), *(
        (c.center_m, c.radius_m, SphericalRegion("cavity_wall", c.cavity_id)) for c in rotated.cavities
    ))
    for center, radius, region in walls:
        p = (center[0], center[1] + radius, center[2])
        assert solution.region_at(p) == region
        assert solution.potential_at(p) == solution.potential_v
        with pytest.raises(SingularField):
            solution.electric_field_at(p)

    # Integrate E on Gaussian spheres wholly within nonmetal regions and metal.
    # Positive flux in a cavity measures the authored sources. The enclosing
    # metal has zero flux and therefore its inner face cancels that charge.
    _scalar(_gauss_charge(solution, shift, 4.75), outer)
    for cavity in rotated.cavities:
        charge = _gauss_charge(solution, cavity.center_m, .8*cavity.radius_m)
        assert abs(charge + float(inner[cavity.cavity_id])) < 1e-9 * NC
        assert _gauss_charge(solution, cavity.center_m, 1.05*cavity.radius_m) == 0.
    _scalar(solution.outer_surface_charge_c + sum(solution.inner_surface_charge_c.values()), net)


@pytest.mark.parametrize(("backend", "constraint", "points"), (
    (ConcentricShellBackend, FixedNetCharge(1.5*NC), (PointCharge(2.0*NC),)),
    (ImageChargeShellBackend, Grounded(), (PointCharge(2.0*NC, (.25, .125, -.3)),)),
    (MultiImageShellBackend, FixedNetCharge(-1.5*NC),
     (PointCharge(2.0*NC, (.25, .125, -.3)), PointCharge(-.25*NC, (-.45, .25, .125)))),
))
def test_legacy_interior_routes(backend, constraint, points):
    sources = tuple(SphericalPointSource(f"source{i}", s.charge_c, s.position_m) for i, s in enumerate(points))
    current = solve_spherical_conductor(SphericalConductorProblem(
        2., constraint, cavities=(SphericalCavity("inner", ZERO, 1.),), sources=sources,
    ))
    old = backend().solve(ElectrostaticProblem(points, (SphericalShell(1., 2., constraint, name="sphere"),)))
    _legacy_check(current, old, ((.1, -.25, .3), (-.5, .125, -.2), (1.5, 0., 0.), (3., 1., -.5)))


@pytest.mark.parametrize(("backend", "constraint", "points"), (
    (ExternalGroundedSphereBackend, Grounded(), (PointCharge(2.5*NC, (2.5, .25, .5)),)),
    (ExternalIsolatedSphereBackend, FixedNetCharge(-.25*NC), (PointCharge(2.5*NC, (2.5, .25, .5)),)),
    (ExternalMultiSphereBackend, FixedNetCharge(-.25*NC),
     (PointCharge(2.5*NC, (2.5, .25, .5)), PointCharge(-.5*NC, (-2., -1., 1.25)))),
    (ExternalMultiSphereBackend, Grounded(),
     (PointCharge(2.5*NC, (2.5, .25, .5)), PointCharge(-.5*NC, (-2., -1., 1.25)))),
))
def test_legacy_exterior_routes(backend, constraint, points):
    center = (.125, .25, -.125)
    sources = tuple(SphericalPointSource(f"source{i}", s.charge_c, s.position_m) for i, s in enumerate(points))
    current = solve_spherical_conductor(SphericalConductorProblem(
        1., constraint, center_m=center, sources=sources,
    ))
    old = backend().solve(ElectrostaticProblem(points, (SolidSphere(1., constraint, center),)))
    _legacy_check(current, old, (center, (3., 1., -.5), (-1.25, -1., 0.), (-2., 1., 2.)))


def _legacy_check(current, old, points):
    new_state, old_state = current.conductor_results[0], old.conductor_results[0]
    for field in ("potential_v", "net_charge_c", "inner_surface_charge_c", "outer_surface_charge_c"):
        actual, expected = getattr(new_state, field), getattr(old_state, field)
        if expected is None:
            assert actual is None
        else:
            _scalar(actual, expected, 1e-12)
    for point in points:
        _scalar(current.potential_at(point), old.potential_at(point), 1e-12)
        _vector(current.electric_field_at(point), old.electric_field_at(point), 1e-12)


@pytest.mark.parametrize("constraint", (Grounded(), FixedNetCharge(.35*NC)))
def test_legacy_m40a_face_totals(constraint):
    a = SphericalCavity("A", (-1., 0., 0.), .5)
    b = SphericalCavity("B", (1., 0., 0.), .5)
    qa, qb = _source("pa", 2.0*NC, -.875), _source("pb", -1.25*NC, 1.25)
    m39 = GaussCavityAssembly("sphere", (
        SealedCavity("A", (CavityCharge(qa.source_id, qa.charge_c),)),
        SealedCavity("B", (CavityCharge(qb.source_id, qb.charge_c),)),
    ), constraint)
    geometry = SphericalMetalGeometry(ZERO, 2., tuple(
        SphericalVoid(c.cavity_id, c.center_m, c.radius_m, (NamedSourcePosition(q.source_id, q.position_m),))
        for c, q in ((a, qa), (b, qb))
    ))
    ledger = solve_spherical_sigma(SphericalSigmaModel(m39, geometry)).gauss_ledger
    current = solve_spherical_conductor(SphericalConductorProblem(
        2., constraint, cavities=(a, b), sources=(qa, qb),
    ))
    _scalar(current.outer_surface_charge_c, ledger.outer_face_charge_c, 1e-12)
    _scalar(current.net_charge_c, ledger.net_metal_charge_c, 1e-12)
    for face in ledger.faces:
        _scalar(current.inner_surface_charge_c[face.cavity_id], face.induced_inner_face_charge_c, 1e-12)


@pytest.mark.parametrize("cavities", (
    (SphericalCavity("touch", (1.5, 0., 0.), .5),),
    (SphericalCavity("protrude", (1.5, 0., 0.), .75),),
    (SphericalCavity("coincident", ZERO, .25), SphericalCavity("same", ZERO, .25)),
    (SphericalCavity("left", (-.5, 0., 0.), .5), SphericalCavity("right", (.5, 0., 0.), .5)),
    (SphericalCavity("left", (-.5, 0., 0.), .5), SphericalCavity("right", (.49, 0., 0.), .5)),
))
def test_refuses_invalid_cavity_geometry(cavities):
    with pytest.raises(UnsupportedProblem, match="cavit"):
        solve_spherical_conductor(SphericalConductorProblem(2., Grounded(), cavities=cavities))


@pytest.mark.parametrize("x", (1., 1.5, 2.))
@pytest.mark.parametrize("charges", ((NC,), (NC, -NC), (0.,)))
def test_illegal_sources_refused_before_coalescing(x, charges):
    sources = tuple(_source(f"charge{i}", q, x) for i, q in enumerate(charges))
    with pytest.raises(UnsupportedProblem, match="wall or inside metal"):
        solve_spherical_conductor(SphericalConductorProblem(
            2., Grounded(), cavities=(SphericalCavity("A", ZERO, 1.),), sources=sources,
        ))


def test_singular_unresolved_and_cancelled_sites():
    near_wall = (nextafter(1., 0.), 0., 0.)
    cavity = SphericalCavity("A", ZERO, 1.)
    problem = SphericalConductorProblem(2., Grounded(), cavities=(cavity,))
    with pytest.raises(UnsupportedProblem, match="unresolved near-wall"):
        solve_spherical_conductor(replace(problem, sources=(_source("n", NC, *near_wall),)))
    solution = solve_spherical_conductor(replace(problem, sources=(
        _source("a", NC, .25), _source("b", -NC, .25),
        _source("c", NC, 4.), _source("d", -NC, 4.),
        _source("e", 2*NC, .5), _source("f", NC, 5.),
    )))
    assert len(solution.source_regions) == 6
    for x in (.25, 4.):
        assert isinstance(solution.potential_at((x, 0., 0.)), float)
        assert len(solution.electric_field_at((x, 0., 0.))) == 3
    for x in (.5, 5.):
        for query in (solution.potential_at, solution.electric_field_at):
            with pytest.raises(SingularField, match="active point source"):
                query((x, 0., 0.))
            with pytest.raises(UnsupportedProblem, match="unresolved near-source"):
                query((nextafter(x, 1.), 0., 0.))
    for query in (solution.region_at, solution.potential_at, solution.electric_field_at):
        with pytest.raises(UnsupportedProblem, match="unresolved near-wall"):
            query(near_wall)
        with pytest.raises(UnsupportedProblem, match="unresolved near-wall"):
            # hypot rounds this to 1 despite an exactly nonzero squared gap.
            query((1., ulp(1.), 0.))
    empty = solve_spherical_conductor(replace(problem, sources=solution.problem.sources[:4]))
    for p in ((.25, 0., 0.), (4., 0., 0.)):
        assert empty.potential_at(p) == 0.
        assert empty.electric_field_at(p) == ZERO


def test_api_refuses_out_of_scope_inputs_and_distinguishes_zero_dirichlet():
    problem = SphericalConductorProblem(1., Grounded())
    with pytest.raises(UnsupportedProblem, match="one"):
        solve_spherical_conductor((problem, problem))
    with pytest.raises(UnsupportedProblem, match="nonzero FixedPotential"):
        solve_spherical_conductor(replace(problem, constraint=FixedPotential(1.)))
    with pytest.raises(UnsupportedProblem, match="force"):
        force_energy(solve_spherical_conductor(problem))
    grounded = SphericalConductorProblem(1., Grounded(), sources=(_source("q", NC, 2.),))
    assert solve_spherical_conductor(replace(grounded, constraint=FixedPotential(0.))).conductor_results == (
        solve_spherical_conductor(grounded).conductor_results
    )
    with pytest.raises(ValueError, match="cavity IDs"):
        cavity = SphericalCavity("air", ZERO, .5)
        replace(problem, cavities=(cavity, cavity))
    with pytest.raises(ValueError, match="source IDs"):
        replace(problem, sources=grounded.sources * 2)


BIG, SMALL = F(1, 2**30), F(1, 2**90)


def test_d1_fixed_charge_residual_survives_cavity_and_site_grouping():
    problem = SphericalConductorProblem(
        4., FixedNetCharge(-float(BIG)),
        cavities=(SphericalCavity("A", ZERO, 1.), SphericalCavity("B", (2., 0., 0.), .5)),
        sources=(_source("big", float(BIG), .25), _source("small", float(SMALL), .25),
                 _source("tiny", float(SMALL), 2.)),
    )
    result = solve_spherical_conductor(problem)
    assert result.inner_surface_charge_c == {"A": -float(BIG + SMALL), "B": -float(SMALL)}
    assert result.outer_surface_charge_c == float(2 * SMALL)
    _scalar(result.potential_v, K * float(2 * SMALL / 4), 1e-12)
    _scalar(result.potential_at((8., 0., 0.)), K * float(2 * SMALL / 8), 1e-12)


def test_grounded_net_charge_survives_opposite_cavity_and_image_subtotals():
    solution = solve_spherical_conductor(SphericalConductorProblem(
        2., Grounded(), cavities=(SphericalCavity("A", ZERO, 1.),),
        sources=(_source("big", -float(BIG), -.5), _source("small", float(SMALL), .25),
                 _source("out", float(2 * BIG), 4.)),
    ))
    assert solution.outer_surface_charge_c == -float(BIG)
    assert solution.inner_surface_charge_c["A"] == float(BIG - SMALL)
    assert solution.net_charge_c == -float(SMALL)


@pytest.mark.parametrize("sign", (1, -1))
def test_d2_signed_potential_and_field_cancellation(sign):
    problem = SphericalConductorProblem(
        2., Grounded(), cavities=(SphericalCavity("air", ZERO, 1.),),
        sources=(_source("left", float(sign * BIG), -.5), _source("witness", float(sign * SMALL), .25),
                 _source("right", float(-sign * BIG), .5)),
    )
    opposite = solve_spherical_conductor(problem)
    # At the center the image of a source at x is -q/|x| at 1/x.
    _scalar(opposite.potential_at(ZERO), K * float(3 * sign * SMALL), 1e-12)
    mirrored = replace(problem.sources[-1], charge_c=float(sign * BIG))
    same = solve_spherical_conductor(replace(problem, sources=(*problem.sources[:2], mirrored)))
    ex, ey, ez = same.electric_field_at(ZERO)
    _scalar(ex, K * float(-F(63, 4) * sign * SMALL), 1e-12)
    assert (ey, ez) == (0., 0.)
    assert opposite.potential_at((5., 1., 0.)) == 0.
    assert opposite.electric_field_at((5., 1., 0.)) == ZERO
    assert solve_spherical_conductor(replace(problem, sources=(problem.sources[0], mirrored))
                                     ).electric_field_at(ZERO) == ZERO
    assert solve_spherical_conductor(replace(problem, sources=problem.sources[::2])
                                     ).potential_at(ZERO) == 0.


@pytest.mark.parametrize("exterior", (False, True))
def test_near_wall_pairs_match_independent_kelvin_oracle(exterior):
    # A rotated direction with no axis-aligned shortcuts in the Kelvin pair.
    a, b = .6, .8
    scale = 1. + (2e-8 if exterior else -2e-8)
    source = _source("near", 4*NC, scale*a, scale*b)
    if exterior:
        problem = SphericalConductorProblem(1., Grounded(), sources=(source,))
        probes = ((1.2, .2, .1), (a*(1+4e-8), b*(1+4e-8), 5e-9))
    else:
        problem = SphericalConductorProblem(
            2., Grounded(), cavities=(SphericalCavity("air", ZERO, 1.),), sources=(source,),
        )
        probes = ((.2, -.4, .1), (a*(1-4e-8), b*(1-4e-8), 5e-9))
    solution = solve_spherical_conductor(problem)
    for probe in probes:
        v, e = _kelvin(ZERO, 1., (source,), 0., EPSILON_0, probe, exterior=exterior)
        _scalar(solution.potential_at(probe), v)
        _vector(solution.electric_field_at(probe), e)
        _scalar(solution.electric_field_at(probe)[2], e[2])


def test_exact_non_axial_wall():
    solution = solve_spherical_conductor(SphericalConductorProblem(5., FixedNetCharge(NC)))
    p = (3., 4., 0.)
    assert solution.region_at(p) == SphericalRegion("outer_wall")
    assert solution.potential_at(p) == solution.potential_v
    with pytest.raises(SingularField):
        solution.electric_field_at(p)
