"""Public stack API acceptance with independent Gauss-law and energy oracles."""

from dataclasses import replace
from fractions import Fraction as F
from itertools import permutations
from math import dist, fsum, prod

import pytest

from conductor_sim import (
    ConcentricConductor, ConductorGroup, ConductingPlane, ConnectedShellBackend,
    ConnectionPlan, ElectricalLink, ElectrostaticProblem, FixedNetCharge, FixedPotential,
    Floating, Grounded, MixedRadialBackend, NestedShellBackend, ParallelPlateBackend,
    PlanarStackProblem, PlanarVoltageReference, PointCharge, SingularField, SolidSphere,
    SphericalCavity, SphericalConductorProblem, SphericalPointSource, SphericalShell,
    SphericalStackProblem, StackPointCharge, UnsupportedProblem, solve_planar_stack,
    solve_spherical_conductor, solve_spherical_stack, switch_conductor_stack,
)
from conductor_sim.thin_shells import ConcentricThinShell, solve_thin_shell_charges

EPSILON = F("8.8541878128e-12")
K = 1 / (4 * F("3.141592653589793238462643383279502884197") * EPSILON)
NC = F("1e-9")
BOUNDARIES = dict(d_left_c_per_m2=0, d_right_c_per_m2=0)


def group(member, constraint):
    return ConductorGroup((member,), constraint)


def expected(actual, oracle):
    assert actual == pytest.approx(float(oracle), rel=1e-12, abs=1e-26)


def determinant(rows):
    return sum((-1)**sum(p[i] > p[j] for i in range(len(p)) for j in range(i + 1, len(p)))
               * prod(row[j] for row, j in zip(rows, p))
               for p in permutations(range(len(rows))))


def solve_by_charge(elastance, count, conditions, *, neutral=False, gauge=None):
    """Solve for each body's free charge using a Q -> V map and Cramer's rule."""
    def equations(unknowns):
        charges = unknowns[:count]
        potentials = elastance(unknowns)
        values = []
        for members, mode, value in conditions:
            first, *rest = members
            values.extend(potentials[i] - potentials[first] for i in rest)
            values.append((potentials[first] if mode == "v" else sum(charges[i] for i in members)) - value)
        if neutral:
            if gauge is None:
                values.append(sum(charges))
            else:
                first, value = gauge
                values.append(potentials[first] - value)
        return values

    n = count + neutral
    base = equations([F(0)] * n)
    columns = [equations([F(i == j) for i in range(n)]) for j in range(n)]
    matrix = [[columns[j][i] - base[i] for j in range(n)] for i in range(n)]
    det = determinant(matrix)
    unknowns = tuple(determinant([[(-base[i] if k == j else cell) for k, cell in enumerate(row)]
                                  for i, row in enumerate(matrix)]) / det for j in range(n))
    return unknowns[:count], elastance(unknowns)


def sphere_oracle(radii, conditions, *, cavity=F(0), annuli=None, exterior=()):
    """Gauss flux from cumulative free charge; integrate the angular mean V."""
    n = len(radii)
    if annuli is None:
        annuli = ((),) * (n - 1)

    def faces(charges):
        inner, outer = [-cavity], []
        enclosed = cavity
        for i, charge in enumerate(charges):
            enclosed += charge
            outer.append(enclosed)
            if i < n - 1:
                enclosed += sum(q for r, q in annuli[i])
                inner.append(-enclosed)
        return inner, outer

    def voltage(charges):
        _, outer = faces(charges)
        values = [K * (outer[-1] / radii[-1][1] + sum(q / r for r, q in exterior))]
        for i in reversed(range(n - 1)):
            a, b = F(radii[i][1]), F(radii[i + 1][0])
            drop = K * (outer[i] * (1 / a - 1 / b) + sum(q * (1 / r - 1 / b) for r, q in annuli[i]))
            values.append(values[-1] + drop)
        return tuple(reversed(values))

    charges, potentials = solve_by_charge(voltage, n, conditions)
    return charges, potentials, *faces(charges)


def planar_oracle(positions, area, conditions, *, gauge=None):
    def voltage(unknowns):
        values = [unknowns[-1]]
        for i in reversed(range(len(positions) - 1)):
            width = positions[i + 1] - positions[i]
            values.append(values[-1] + width * sum(unknowns[:i + 1]) / (EPSILON * area))
        return tuple(reversed(values))

    return solve_by_charge(voltage, len(positions), conditions, neutral=True, gauge=gauge)


def check_sphere(solution, oracle):
    for result, q, v, inner, outer in zip(solution.conductor_results, *oracle, strict=True):
        expected(result.net_charge_c, q)
        expected(result.potential_v, v)
        expected(result.inner_surface_charge_c, inner)
        expected(result.outer_surface_charge_c, outer)


def field_energy(q, v):
    return sum(charge * potential for charge, potential in zip(q, v, strict=True)) / 2


def check_switch(result, ids, before, after, terminal_conditions):
    q0, v0 = before[:2]
    q1, v1 = after[:2]
    delta = tuple(b - a for a, b in zip(q0, q1, strict=True))
    u0, u1 = field_energy(q0, v0), field_energy(q1, v1)
    ground = fixed = work = F(0)
    for terminal, (members, mode, voltage) in zip(result.terminal_transfers, terminal_conditions, strict=True):
        assert terminal.members == tuple(ids[i] for i in members)
        delivered = -sum(delta[i] for i in members)
        expected(terminal.charge_delivered_c, delivered)
        if mode == "ground":
            ground += delivered
        else:
            fixed += delivered
            work -= voltage * delivered
    for name, q in zip(ids, delta, strict=True):
        expected(result.delta_q_c[name], q)
    expected(result.charge_delivered_to_ground_c, ground)
    expected(result.charge_delivered_to_fixed_potential_supplies_c, fixed)
    assert sum(delta) + ground + fixed == 0
    expected(result.initial_field_energy_j, u0)
    expected(result.final_field_energy_j, u1)
    expected(result.delta_field_energy_j, u1 - u0)
    expected(result.supply_work_j, work)
    assert result.external_work_j == 0
    expected(result.energy_residual_j, work - (u1 - u0))
    assert result.energy_residual_j >= 0


@pytest.mark.parametrize("core_inner", [0, 1], ids=["G8-solid", "G7-hollow"])
@pytest.mark.parametrize("constraints,conditions", [
    ((Grounded(), Grounded()), (("v", 0), ("v", 0))),
    ((FixedNetCharge(2e-9), Grounded()), (("q", 2 * NC), ("v", 0))),
    ((FixedPotential(-4), FixedNetCharge(-3e-9)), (("v", -4), ("q", -3 * NC))),
])
def test_annulus_reciprocity(core_inner, constraints, conditions):
    ids = ("a", "b")
    radii = ((core_inner, 2), (6, 8))
    q = 9 * NC
    source = StackPointCharge("held", float(q), (0, 3, 0))
    problem = SphericalStackProblem(tuple(ConcentricConductor(n, *r) for n, r in zip(ids, radii)),
                                    tuple(map(group, ids, constraints)), sources=(source,))
    solution = solve_spherical_stack(problem)
    oracle = sphere_oracle(radii, [((i,), *c) for i, c in enumerate(conditions)],
                           annuli=(((F(3), q),),))
    check_sphere(solution, oracle)
    a, b = solution.conductor_results
    expected(a.outer_surface_charge_c + b.inner_surface_charge_c, -q)
    if constraints == (Grounded(), Grounded()):
        expected(a.outer_surface_charge_c, -q / 2)
        expected(b.inner_surface_charge_c, -q / 2)
    exterior_charge = oracle[3][-1]
    expected(solution.potential_at((16, 0, 0)), K * exterior_charge / 16)
    expected(solution.electric_field_at((16, 0, 0))[0], K * exterior_charge / 256)
    assert solution.source_regions["held"].gap_index == 0
    assert solution.potential_at((2, 0, 0)) == a.potential_v
    assert solution.potential_at((6, 0, 0)) == b.potential_v
    assert solution.potential_at((7, 0, 0)) == b.potential_v
    assert solution.electric_field_at((7, 0, 0)) == (0, 0, 0)
    assert solution.potential_at((0, 0, 0)) == a.potential_v
    assert solution.electric_field_at((0, 0, 0)) == (0, 0, 0)
    for sample in (solution.potential_at, solution.electric_field_at):
        with pytest.raises(UnsupportedProblem, match="charged annulus"):
            sample((4, 0, 0))
        with pytest.raises(SingularField, match="active point source"):
            sample(source.position_m)
    with pytest.raises(SingularField, match="wall"):
        solution.electric_field_at((8, 0, 0))
    assert solution.field_energy_j is None
    with pytest.raises(UnsupportedProblem, match="active point charges"):
        switch_conductor_stack(solution, problem.groups)


def test_shifted_composite_annular_and_exterior_sources_and_floating_short():
    center = (13, -7, 4)
    ids = ("core", "middle", "outer")
    radii = ((0, 1), (3, 4), (6, 7))
    source = StackPointCharge("inner-annulus", 7e-9, (13, -5, 4))
    outside = StackPointCharge("exterior", -4e-9, (13, -7, 18))
    groups = tuple(group(i, FixedNetCharge(q)) for i, q in zip(ids, (1e-9, 2e-9, -3e-9)))
    problem = SphericalStackProblem(tuple(ConcentricConductor(n, *r) for n, r in zip(ids, radii)),
                                    groups, center_m=center, sources=(source, outside))
    before = solve_spherical_stack(problem)
    after = solve_spherical_stack(replace(problem, groups=(group("middle", FixedPotential(-5)),
                                                          ConductorGroup(("outer", "core"), FixedNetCharge(-2e-9)))))
    sources = dict(annuli=(((F(2), 7 * NC),), ()), exterior=((F(14), -4 * NC),))
    check_sphere(before, sphere_oracle(radii, [((0,), "q", NC), ((1,), "q", 2 * NC),
                                              ((2,), "q", -3 * NC)], **sources))
    oracle = sphere_oracle(radii, [((1,), "v", -5), ((0, 2), "q", -2 * NC)], **sources)
    check_sphere(after, oracle)
    assert after.conductor_results[0].potential_v == after.conductor_results[2].potential_v
    assert after.source_regions["inner-annulus"].kind == "annulus"
    assert after.source_regions["exterior"].kind == "exterior"
    gap_point = (18, -7, 4)
    expected(after.potential_at(gap_point), oracle[1][2] + K * oracle[3][1] * (F(1, 5) - F(1, 6)))
    assert after.electric_field_at(gap_point) == pytest.approx((float(K * oracle[3][1] / 25), 0, 0), rel=1e-12)

    probe = (23, -7, 4)
    image = (13, -7, 7.5)
    image_q = 2 * NC
    exterior_background_q = 7 * oracle[1][2] / K
    charges = ((center, exterior_background_q), (outside.position_m, -4 * NC), (image, image_q))
    expected(after.potential_at(probe), sum(float(K * q) / dist(probe, p) for p, q in charges))
    field = tuple(sum(float(K * q) * (x - p[i]) / dist(probe, p)**3 for p, q in charges)
                  for i, x in enumerate(probe))
    assert after.electric_field_at(probe) == pytest.approx(field, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("terminal,voltage", [(Grounded(), 0), (FixedPotential(9), 9)])
def test_linked_spherical_terminal_with_sources_in_all_regions(terminal, voltage):
    ids = ("a", "b", "c")
    radii = ((1, 2), (6, 7), (11, 12))
    sources = (
        StackPointCharge("cavity", 2e-9, (0, 0, 0)),
        StackPointCharge("inner-gap", -3e-9, (0, 3, 4)),
        StackPointCharge("outer-gap", 4e-9, (0, -6, 8)),
        StackPointCharge("exterior", -5e-9, (0, -9, 12)),
    )
    problem = SphericalStackProblem(tuple(ConcentricConductor(n, *r) for n, r in zip(ids, radii)),
                                    (ConductorGroup(ids, terminal),), sources=sources)
    result = solve_spherical_stack(problem)
    oracle = sphere_oracle(radii, [((0, 1, 2), "v", voltage)], cavity=2 * NC,
                           annuli=(((F(5), -3 * NC),), ((F(10), 4 * NC),)), exterior=((F(15), -5 * NC),))
    check_sphere(result, oracle)
    assert result.source_regions["cavity"].conductor_id == "a"
    assert result.source_regions["outer-gap"].gap_index == 1
    expected(result.potential_at((0.5, 0, 0)), voltage + K * 2 * NC)
    expected(result.electric_field_at((0.5, 0, 0))[0], K * 8 * NC)
    for point in ((3, 0, 0), (8, 0, 0)):
        with pytest.raises(UnsupportedProblem, match="charged annulus"):
            result.electric_field_at(point)


@pytest.mark.parametrize("terminal", [Grounded(), FixedPotential(-5)])
def test_spherical_three_conductor_switch_from_independent_charges(terminal):
    ids = ("core", "middle", "outer")
    radii = ((0, 1), (2, 3), (4, 5))
    before = solve_spherical_stack(SphericalStackProblem(
        tuple(ConcentricConductor(n, *r) for n, r in zip(ids, radii)),
        tuple(group(n, FixedNetCharge(q)) for n, q in zip(ids, (2e-9, -3e-9, 4e-9))),
    ))
    final_groups = (group("middle", terminal), ConductorGroup(("core", "outer"), Floating()))
    result = switch_conductor_stack(before, final_groups)
    initial = sphere_oracle(radii, [((0,), "q", 2 * NC), ((1,), "q", -3 * NC), ((2,), "q", 4 * NC)])
    voltage = 0 if type(terminal) is Grounded else -5
    final = sphere_oracle(radii, [((1,), "v", voltage), ((0, 2), "q", 6 * NC)])
    check_sphere(result.after, final)
    kind = "ground" if type(terminal) is Grounded else "fixed"
    check_switch(result, ids, initial, final, [((1,), kind, voltage)])
    noop = switch_conductor_stack(result.after, final_groups)
    assert noop.energy_residual_j == noop.delta_field_energy_j == noop.supply_work_j == 0
    assert set(noop.delta_q_c.values()) == {0}


def test_sphere_with_simultaneous_prescribed_potential_ground_and_floating_link():
    ids = ("core", "middle", "guard", "outer")
    radii = ((0, 1), (2, 3), (4, 5), (6, 7))
    before = solve_spherical_stack(SphericalStackProblem(
        tuple(ConcentricConductor(n, *r) for n, r in zip(ids, radii)),
        tuple(group(n, FixedNetCharge(0)) for n in ids),
    ))
    groups = (ConductorGroup(("core", "guard"), Floating()), group("middle", FixedPotential(6)),
              group("outer", Grounded()))
    result = switch_conductor_stack(before, groups)
    initial = sphere_oracle(radii, [((i,), "q", 0) for i in range(4)])
    final = sphere_oracle(radii, [((0, 2), "q", 0), ((1,), "v", 6), ((3,), "v", 0)])
    check_sphere(result.after, final)
    check_switch(result, ids, initial, final, [((1,), "fixed", 6), ((3,), "ground", 0)])


def test_spherical_capacitance_and_two_conductor_switch():
    ids = ("a", "b")
    radii = ((0, 2), (5, 6))
    before = solve_spherical_stack(SphericalStackProblem(
        tuple(ConcentricConductor(n, *r) for n, r in zip(ids, radii)),
        (group("a", Grounded()), group("b", Grounded())),
    ))
    result = switch_conductor_stack(before, (group("a", FixedPotential(12)), group("b", Grounded())))
    capacitance = 1 / (K * (F(1, 2) - F(1, 5)))
    matrix = result.after.capacitance_matrix_f
    assert matrix[0][1] == matrix[1][0]
    expected(matrix[0][0], capacitance)
    expected(matrix[0][1], -capacitance)
    expected(matrix[1][1], capacitance + 6 / K)
    expected(result.after.field_energy_j, capacitance * 12**2 / 2)
    check_switch(result, ids, sphere_oracle(radii, [((0,), "v", 0), ((1,), "v", 0)]),
                 sphere_oracle(radii, [((0,), "v", 12), ((1,), "v", 0)]),
                 [((0,), "fixed", 12), ((1,), "ground", 0)])


@pytest.mark.parametrize("count", [2, 3])
def test_planar_capacitor_and_isolated_middle_plate(count):
    ids = tuple("abc"[:count])
    positions = (-2, 1, 3)[:count]
    area = 5
    before = solve_planar_stack(PlanarStackProblem(
        ids, positions, area, tuple(group(n, FixedNetCharge(0)) for n in ids),
        voltage_reference=PlanarVoltageReference(ids[-1]), **BOUNDARIES,
    ))
    middle = (group(ids[1], Floating()),) if count == 3 else ()
    final_groups = (group(ids[0], FixedPotential(9)), *middle, group(ids[-1], Grounded()))
    result = switch_conductor_stack(before, final_groups)
    after = result.after
    specs = [((0,), "v", 9), ((count - 1,), "v", 0)]
    if count == 3:
        specs.append(((1,), "q", 0))
    oracle = planar_oracle(positions, area, specs)
    check_switch(result, ids, ((0,) * count, (0,) * count), oracle,
                 [((0,), "fixed", 9), ((count - 1,), "ground", 0)])
    for i, (plate, q, v) in enumerate(zip(after.plate_results, *oracle, strict=True)):
        expected(plate.net_charge_c, q)
        expected(plate.potential_v, v)
        expected(plate.left_face_charge_c, -sum(oracle[0][:i]))
        expected(plate.right_face_charge_c, sum(oracle[0][:i + 1]))
        expected(plate.left_sigma_c_per_m2, -sum(oracle[0][:i]) / area)
        expected(plate.right_sigma_c_per_m2, sum(oracle[0][:i + 1]) / area)
        expected(after.potential_at(positions[i]), v)
    assert fsum(p.net_charge_c for p in after.plate_results) == 0
    assert after.problem.d_left_c_per_m2 == after.problem.d_right_c_per_m2 == 0
    assert after.electric_field_at(-3) == after.electric_field_at(4) == 0
    assert after.potential_at(-3) == 9 and after.potential_at(4) == 0
    for i in range(count - 1):
        left, right = positions[i:i + 2]
        x = (left + right) / 2
        field = (oracle[1][i] - oracle[1][i + 1]) / (right - left)
        expected(after.electric_field_at(x), field)
        expected(after.potential_at(x), (oracle[1][i] + oracle[1][i + 1]) / 2)
        expected(after.gap_fields_v_per_m[i], field)
        expected(after.gap_displacements_c_per_m2[i], EPSILON * field)
    matrix = after.capacitance_matrix_f
    for i in range(count):
        c = EPSILON * area / (positions[i + 1] - positions[i]) if i + 1 < count else 0
        previous = EPSILON * area / (positions[i] - positions[i - 1]) if i else 0
        expected(matrix[i][i], c + previous)
        for j in range(i + 1, count):
            assert matrix[i][j] == matrix[j][i]
            expected(matrix[i][j], -c if j == i + 1 else 0)
    expected(after.field_energy_j, (EPSILON * area / (positions[-1] - positions[0])) * 9**2 / 2)
    assert switch_conductor_stack(after, final_groups).energy_residual_j == 0
    with pytest.raises(SingularField, match="plate"):
        after.electric_field_at(positions[0])


def test_planar_wired_terminals_and_neutral_gauge_only_switch():
    ids = ("a", "b", "c")
    positions = (0, 3, 5)
    area = 5
    problem = PlanarStackProblem(
        ids, positions, area, tuple(group(n, FixedNetCharge(q)) for n, q in zip(ids, (1e-9, 2e-9, -3e-9))),
        voltage_reference=PlanarVoltageReference("b", 7), **BOUNDARIES,
    )
    before = solve_planar_stack(problem)
    initial = planar_oracle(positions, area, [((0,), "q", NC), ((1,), "q", 2 * NC), ((2,), "q", -3 * NC)],
                            gauge=(1, 7))
    for plate, q, v in zip(before.plate_results, *initial, strict=True):
        expected(plate.net_charge_c, q)
        expected(plate.potential_v, v)
    gauge_change = switch_conductor_stack(before, tuple(group(n, Floating()) for n in ids),
                                          voltage_reference=PlanarVoltageReference("b", -6))
    assert gauge_change.energy_residual_j == gauge_change.delta_field_energy_j == gauge_change.supply_work_j == 0
    assert not gauge_change.terminal_transfers
    assert tuple(gauge_change.delta_q_c.values()) == (0, 0, 0)
    assert gauge_change.after.gap_fields_v_per_m == before.gap_fields_v_per_m

    floating_short = switch_conductor_stack(before, (
        ConductorGroup(("a", "c"), Floating()), group("b", Floating()),
    ), voltage_reference=PlanarVoltageReference("c", -6))
    floating_final = planar_oracle(positions, area, [((0, 2), "q", -2 * NC), ((1,), "q", 2 * NC)],
                                   gauge=(2, -6))
    check_switch(floating_short, ids, initial, floating_final, [])

    for terminal in (Grounded(), FixedPotential(-2)):
        v = 0 if type(terminal) is Grounded else -2
        kind = "ground" if type(terminal) is Grounded else "fixed"
        change = switch_conductor_stack(before, (ConductorGroup(("a", "c"), terminal),
                                                 group("b", FixedPotential(11 + v))))
        final = planar_oracle(positions, area, [((0, 2), "v", v), ((1,), "v", 11 + v)])
        check_switch(change, ids, initial, final, [((0, 2), kind, v), ((1,), "fixed", 11 + v)])


def test_uniform_permittivity_scales_both_capacitance_matrices_and_energy():
    groups = (group("a", FixedPotential(4)), group("b", Grounded()))
    sphere = SphericalStackProblem((ConcentricConductor("a", 0, 1), ConcentricConductor("b", 2, 3)), groups)
    plane = PlanarStackProblem(("a", "b"), (0, 2), 3, groups, **BOUNDARIES)
    for solve, problem in ((solve_spherical_stack, sphere), (solve_planar_stack, plane)):
        base = solve(problem)
        dielectric = solve(replace(problem, permittivity_f_per_m=4 * problem.permittivity_f_per_m))
        assert dielectric.field_energy_j == 4 * base.field_energy_j
        assert dielectric.capacitance_matrix_f == tuple(tuple(4 * c for c in row) for row in base.capacitance_matrix_f)


def compare_legacy_sphere(stack, legacy, points):
    for result, reference in zip(stack.conductor_results, legacy.conductor_results, strict=True):
        assert result.name == reference.name
        for attr in ("potential_v", "net_charge_c", "inner_surface_charge_c", "outer_surface_charge_c"):
            expected(getattr(result, attr), getattr(reference, attr))
    for point in points:
        expected(stack.potential_at(point), legacy.potential_at(point))
        assert stack.electric_field_at(point) == pytest.approx(legacy.electric_field_at(point), rel=1e-12, abs=1e-14)


@pytest.mark.parametrize("solid", [False, True])
def test_legacy_nested_mixed_and_connected_parity(solid):
    ids = ("a", "b", "c")
    constraints = (FixedNetCharge(2e-9), FixedNetCharge(1e-9), FixedNetCharge(-4e-9))
    core = 0 if solid else 1
    new_shapes = (ConcentricConductor("a", core, 2), ConcentricConductor("b", 3, 4), ConcentricConductor("c", 5, 6))
    old_shapes = (SolidSphere(2, constraints[0], name="a") if solid else SphericalShell(1, 2, constraints[0], name="a"),
                  SphericalShell(3, 4, constraints[1], name="b"), SphericalShell(5, 6, constraints[2], name="c"))
    points = ((0.3, 0.2, 0), (1.5, 0, 0), (2.5, 0, 0), (4.5, 0, 0), (7, 0, 0))
    sources = () if solid else (StackPointCharge("cavity", 3e-9, (0, 0, 0.5)),)
    stack_problem = SphericalStackProblem(new_shapes, tuple(map(group, ids, constraints)), sources=sources)
    old = ElectrostaticProblem(tuple(PointCharge(s.charge_c, s.position_m) for s in sources), old_shapes)
    backend = MixedRadialBackend() if solid else NestedShellBackend()
    initial = solve_spherical_stack(stack_problem)
    compare_legacy_sphere(initial, backend.solve(old), points)

    links = (ElectricalLink(("a", "b")), ElectricalLink(("c",), "grounded"))
    legacy = ConnectedShellBackend().solve(ConnectionPlan(old, links))
    final = solve_spherical_stack(replace(stack_problem, groups=(ConductorGroup(("a", "b"), FixedNetCharge(3e-9)),
                                                               group("c", Grounded()))))
    compare_legacy_sphere(final, legacy.after, points)
    if solid:
        switch = switch_conductor_stack(initial, (ConductorGroup(("a", "b"), Floating()), group("c", Grounded())))
        expected(switch.initial_field_energy_j, legacy.radial_field_energy_before_j)
        expected(switch.final_field_energy_j, legacy.radial_field_energy_after_j)
        expected(switch.energy_residual_j, legacy.released_field_energy_j)


def test_legacy_thin_shells_and_two_independent_faces():
    ids = ("a", "b", "c")
    radii = (1, 2, 3)
    constraints = (FixedNetCharge(5e-9), Grounded(), FixedNetCharge(-1e-9))
    thin_shells = tuple(ConcentricThinShell(n, r, c) for n, r, c in zip(ids, radii, constraints))
    charges = solve_thin_shell_charges(thin_shells, (ElectricalLink(("a", "c")),))
    problem = SphericalStackProblem(tuple(ConcentricConductor(n, r, r) for n, r in zip(ids, radii)),
                                    (ConductorGroup(("a", "c"), FixedNetCharge(4e-9)), group("b", Grounded())))
    solution = solve_spherical_stack(problem)
    for plate in solution.conductor_results:
        expected(plate.net_charge_c, charges[plate.name])
    expected(solution.conductor_results[1].inner_surface_charge_c, -charges["a"])
    expected(solution.conductor_results[1].outer_surface_charge_c, charges["b"] + charges["a"])
    assert solution.region_at((2, 0, 0)).kind == "thin_wall"
    assert solution.potential_at((2, 0, 0)) == 0
    with pytest.raises(SingularField):
        solution.electric_field_at((2, 0, 0))


@pytest.mark.parametrize("constraint", [Grounded(), FixedNetCharge(-2e-9)])
def test_m42a_kelvin_parity_cavity_and_exterior(constraint):
    sources = (StackPointCharge("offset", 3e-9, (1, 0, 0)), StackPointCharge("center", 1e-9, (0, 0, 0)),
               StackPointCharge("outside", 4e-9, (0, 0, -7)), StackPointCharge("outside2", -2e-9, (0, -5, 0)))
    legacy = solve_spherical_conductor(SphericalConductorProblem(
        4, constraint, cavities=(SphericalCavity("cavity", (0, 0, 0), 2),), conductor_id="shell",
        sources=tuple(SphericalPointSource(s.source_id, s.charge_c, s.position_m) for s in sources),
    ))
    problem = SphericalStackProblem((ConcentricConductor("shell", 2, 4),), (group("shell", constraint),), sources=sources)
    result = solve_spherical_stack(problem)
    compare_legacy_sphere(result, legacy, ((0.5, 0.2, -0.5), (1.5, 0, 0), (3, 0, 0), (5, 1, 1), (-6, 0, 0)))
    for s in sources:
        with pytest.raises(SingularField):
            result.potential_at(s.position_m)
        with pytest.raises(SingularField):
            result.electric_field_at(s.position_m)


def test_legacy_parallel_plates_local_parity():
    constraints = (FixedPotential(3), FixedPotential(-4))
    legacy = ParallelPlateBackend().solve(ElectrostaticProblem(conductors=(
        ConductingPlane((-1, 0, 0), (1, 0, 0), constraints[0], "a"),
        ConductingPlane((3, 0, 0), (-1, 0, 0), constraints[1], "b"),
    )))
    solution = solve_planar_stack(PlanarStackProblem(("a", "b"), (-1, 3), 2,
                                                   tuple(map(group, ("a", "b"), constraints)), **BOUNDARIES))
    for x in (-0.5, 0.25, 1.5, 2.5):
        point = (x, 2, -7)
        expected(solution.potential_at(x), legacy.potential_at(point))
        expected(solution.electric_field_at(x), legacy.electric_field_at(point)[0])


def test_annulus_same_site_cancellation_removes_local_refusal_but_retains_ids():
    shapes = (ConcentricConductor("a", 0, 1), ConcentricConductor("b", 3, 4))
    groups = (group("a", FixedPotential(2)), group("b", Grounded()))
    canceling = tuple(StackPointCharge(n, q, (2, 0, 0)) for n, q in (("plus", 1e-9), ("minus", -1e-9)))
    solution = solve_spherical_stack(SphericalStackProblem(shapes, groups, sources=canceling))
    expected(solution.potential_at((2, 0, 0)), F(1, 2))
    expected(solution.electric_field_at((2, 0, 0))[0], F(3, 4))
    assert set(solution.source_regions) == {"plus", "minus"}
    assert not solution.has_active_sources
    assert switch_conductor_stack(solution, groups).energy_residual_j == 0
    # Equal and opposite charges at distinct sites remain active.
    distinct = replace(solution.problem, sources=(canceling[0], replace(canceling[1], position_m=(0, 2, 0))))
    active = solve_spherical_stack(distinct)
    assert active.has_active_sources
    with pytest.raises(UnsupportedProblem, match="charged annulus"):
        active.potential_at((0, 0, 2))


@pytest.mark.parametrize("r", [1, 2, 3, 3.5, 4])
def test_on_wall_or_metal_refused_before_cancellation(r):
    sources = tuple(StackPointCharge(n, q, (r, 0, 0)) for n, q in (("plus", 1), ("minus", -1)))
    problem = SphericalStackProblem((ConcentricConductor("a", 1, 2), ConcentricConductor("b", 3, 4)),
                                    (group("a", Grounded()), group("b", Grounded())), sources=sources)
    with pytest.raises(UnsupportedProblem, match="on a wall or inside metal"):
        solve_spherical_stack(problem)


@pytest.mark.parametrize("r", [1, 2])
def test_touching_or_overlapping_spheres_refused(r):
    with pytest.raises(UnsupportedProblem, match="strictly separated"):
        SphericalStackProblem((ConcentricConductor("a", 0, 2), ConcentricConductor("b", r, 3)), ())


def test_planar_consistency_and_invalid_wiring_refusals():
    problem = PlanarStackProblem(("a", "b"), (0, 1), 1,
                                 (group("a", FixedNetCharge(1e-9)), group("b", FixedNetCharge(-1e-9))), **BOUNDARIES)
    with pytest.raises(UnsupportedProblem, match="requires a voltage reference"):
        solve_planar_stack(problem)
    with pytest.raises(UnsupportedProblem, match="neutral total charge"):
        solve_planar_stack(replace(problem, groups=(group("a", FixedNetCharge(1e-9)), group("b", FixedNetCharge(0))),
                                   voltage_reference=PlanarVoltageReference("b")))
    with pytest.raises(UnsupportedProblem, match="zero exterior displacement"):
        replace(problem, d_left_c_per_m2=1e-9)
    with pytest.raises(UnsupportedProblem, match="overlapping groups"):
        solve_planar_stack(replace(problem, groups=(group("a", Grounded()), ConductorGroup(("a", "b"), Floating()))))
    with pytest.raises(UnsupportedProblem, match="unknown conductor ID"):
        solve_planar_stack(replace(problem, groups=(group("a", Grounded()), group("unknown", Grounded()))))
    with pytest.raises(UnsupportedProblem, match="must belong to one"):
        solve_planar_stack(replace(problem, groups=(group("a", Grounded()),)))
    with pytest.raises(UnsupportedProblem, match="static floating groups"):
        solve_planar_stack(replace(problem, groups=(group("a", Grounded()), group("b", Floating()))))
    before = solve_planar_stack(replace(problem, voltage_reference=PlanarVoltageReference("a")))
    with pytest.raises(UnsupportedProblem, match="conserve"):
        switch_conductor_stack(before, (group("a", Grounded()), group("b", FixedNetCharge(0))))


@pytest.mark.parametrize("invalid", [True, float("nan"), float("inf"), "1"])
def test_invalid_numeric_values_are_refused(invalid):
    with pytest.raises((TypeError, ValueError)):
        group("a", FixedPotential(invalid))
    with pytest.raises((TypeError, ValueError)):
        group("a", FixedNetCharge(invalid))
    with pytest.raises(ValueError):
        ConcentricConductor("a", invalid, 2)
    with pytest.raises(ValueError):
        StackPointCharge("q", invalid, (0, 0, 0))
    with pytest.raises(ValueError):
        StackPointCharge("q", 1, (invalid, 0, 0))
    with pytest.raises(ValueError):
        PlanarStackProblem(("a", "b"), (0, invalid), 1, (), **BOUNDARIES)
    with pytest.raises(ValueError):
        PlanarStackProblem(("a", "b"), (0, 1), 1, (), d_left_c_per_m2=invalid, d_right_c_per_m2=0)


def test_unsupported_topology_has_no_netlist_or_fringing_route():
    with pytest.raises(UnsupportedProblem, match="expected a PlanarStackProblem"):
        solve_planar_stack("R1 plate ground 10k")
    with pytest.raises(UnsupportedProblem, match="expected a SphericalStackProblem"):
        solve_spherical_stack(ElectrostaticProblem())
    with pytest.raises(UnsupportedProblem, match="requires a solved"):
        switch_conductor_stack(ElectrostaticProblem(), ())
