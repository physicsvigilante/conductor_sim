"""M43 acceptance suite: layered linear isotropic dielectrics on planar and spherical stacks."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction as F
from math import isclose, nextafter, pi

import pytest

from conductor_sim import (
    EPSILON_0,
    ConcentricConductor,
    ConductorGroup,
    DielectricLayer,
    FixedNetCharge,
    FixedPotential,
    Floating,
    Grounded,
    PlanarStackProblem,
    PlanarVoltageReference,
    SingularField,
    SphericalStackProblem,
    StackPointCharge,
    UnsupportedProblem,
    quasistatic_stack_transition,
    solve_planar_stack,
    solve_spherical_stack,
    switch_conductor_stack,
)


def _group(name: str, constraint) -> ConductorGroup:
    return ConductorGroup((name,), constraint)


def _approx(actual: float, expected: float, rel: float = 1e-10) -> None:
    assert isclose(actual, expected, rel_tol=rel, abs_tol=1e-24), (actual, expected)


# ---------------------------------------------------------------------------
# Frozen M01 Canonical Acceptance Cards: M43-P2, M43-P4, M43-U1, M43-U2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("vb", "expected_w_bat", "expected_du", "expected_res"),
    [
        (+100.0, 1.67812817192e-6, 6.17946549042e-7, 1.06018162288e-6),
        (-100.0, None, None, 3.60181622878e-7),
    ],
)
def test_m43_p2_planar_three_plate_dielectric_battery_switch(vb, expected_w_bat, expected_du, expected_res):
    area = 0.04
    layers = ((DielectricLayer(0.002, 1.0),), (DielectricLayer(0.002, 3.0),))
    before = solve_planar_stack(
        PlanarStackProblem(
            ("p0", "p1", "p2"),
            (0.0, 0.002, 0.004),
            area,
            (_group("p0", FixedNetCharge(4e-9)), _group("p1", FixedNetCharge(-2e-9)), _group("p2", FixedNetCharge(-2e-9))),
            d_left_c_per_m2=0.0,
            d_right_c_per_m2=0.0,
            voltage_reference=PlanarVoltageReference("p0", 0.0),
            gap_layers=layers,
        )
    )
    sw = switch_conductor_stack(
        before,
        (_group("p0", Grounded()), _group("p1", Floating()), _group("p2", FixedPotential(vb))),
    )
    # Exact two-terminal charge return
    assert isclose(sw.delta_q_c["p0"] + sw.delta_q_c["p2"], 0.0, abs_tol=1e-24)
    assert sw.delta_q_c["p1"] == 0.0
    assert sw.external_work_j == 0.0
    assert sw.energy_residual_j > 0.0
    _approx(sw.energy_residual_j, expected_res, rel=1e-9)
    if expected_du is not None:
        _approx(sw.supply_work_j, expected_w_bat, rel=1e-9)
        _approx(sw.delta_field_energy_j, expected_du, rel=1e-9)
    # Bound charge neutrality per gap
    for g in (0, 1):
        q_b = sum(item.total_bound_charge_c for item in sw.after.dielectric_interfaces if item.gap_index == g)
        assert abs(q_b) < 1e-23


def test_m43_p4_quasistatic_reversible_endpoint_ledgers():
    area = 0.02
    d = 0.001
    v_supply = 100.0
    c_vac = EPSILON_0 * area / d

    # (a) Fixed-V dielectric slab removal: kappa 4 -> 1
    p_before_v = PlanarStackProblem(
        ("p0", "p1"), (0.0, d), area,
        (_group("p0", FixedPotential(v_supply)), _group("p1", Grounded())),
        d_left_c_per_m2=0.0, d_right_c_per_m2=0.0,
        gap_layers=((DielectricLayer(d, 4.0),),),
    )
    p_after_v = replace(p_before_v, gap_layers=((DielectricLayer(d, 1.0),),))
    tr_v = quasistatic_stack_transition(solve_planar_stack(p_before_v), p_after_v)
    _approx(tr_v.delta_field_energy_j, -2.65625634384e-6, rel=1e-9)
    _approx(tr_v.external_work_j, +2.65625634384e-6, rel=1e-9)
    _approx(tr_v.electrostatic_work_on_agent_j, -2.65625634384e-6, rel=1e-9)
    assert tr_v.energy_residual_j == 0.0

    # (b) Fixed-Q isolated opposite plates: kappa 4 -> 1
    q0 = 4.0 * c_vac * v_supply
    p_before_q = PlanarStackProblem(
        ("p0", "p1"), (0.0, d), area,
        (_group("p0", FixedNetCharge(q0)), _group("p1", FixedNetCharge(-q0))),
        d_left_c_per_m2=0.0, d_right_c_per_m2=0.0,
        voltage_reference=PlanarVoltageReference("p1", 0.0),
        gap_layers=((DielectricLayer(d, 4.0),),),
    )
    p_after_q = replace(p_before_q, gap_layers=((DielectricLayer(d, 1.0),),))
    tr_q = quasistatic_stack_transition(solve_planar_stack(p_before_q), p_after_q)
    assert tr_q.supply_work_j == 0.0
    _approx(tr_q.delta_field_energy_j, +1.06250253754e-5, rel=1e-9)
    _approx(tr_q.external_work_j, +1.06250253754e-5, rel=1e-9)
    assert tr_q.energy_residual_j == 0.0
    # Opposite Delta_U sign vs fixed-V!
    assert tr_v.delta_field_energy_j < 0.0 < tr_q.delta_field_energy_j

    # (c) Fixed-V vacuum spacing halving: d -> d/2 (C -> 2C)
    p_vac_before = PlanarStackProblem(
        ("p0", "p1"), (0.0, d), area,
        (_group("p0", FixedPotential(v_supply)), _group("p1", Grounded())),
        d_left_c_per_m2=0.0, d_right_c_per_m2=0.0,
    )
    p_vac_after = replace(p_vac_before, plate_positions_m=(0.0, d / 2.0))
    tr_2c = quasistatic_stack_transition(solve_planar_stack(p_vac_before), p_vac_after)
    _approx(tr_2c.supply_work_j, +1.77083756256e-6, rel=1e-9)
    _approx(tr_2c.delta_field_energy_j, +8.8541878128e-7, rel=1e-9)
    _approx(tr_2c.electrostatic_work_on_agent_j, +8.8541878128e-7, rel=1e-9)
    _approx(tr_2c.external_work_j, -8.8541878128e-7, rel=1e-9)
    assert tr_2c.energy_residual_j == 0.0


def test_m43_u1_four_plate_multilayer_stack_and_wire_switch():
    area = 0.05
    layers = (
        (DielectricLayer(0.002, 2.0),),
        (DielectricLayer(0.001, 1.0), DielectricLayer(0.002, 4.0)),
        (DielectricLayer(0.001, 3.0),),
    )
    unlinked_prob = PlanarStackProblem(
        ("p0", "p1", "p2", "p3"),
        (0.0, 0.002, 0.005, 0.006),
        area,
        (
            _group("p0", Grounded()),
            _group("p1", FixedNetCharge(3e-9)),
            _group("p2", FixedNetCharge(2e-9)),
            _group("p3", FixedPotential(12.0)),
        ),
        d_left_c_per_m2=0.0,
        d_right_c_per_m2=0.0,
        gap_layers=layers,
    )
    unlinked = solve_planar_stack(unlinked_prob)
    # Gap 1 has two distinct kappas -> gap_fields_v_per_m[1] must be None (no fake average!)
    assert unlinked.gap_fields_v_per_m[0] is not None
    assert unlinked.gap_fields_v_per_m[1] is None
    assert unlinked.gap_fields_v_per_m[2] is not None

    # Check frozen Gap-1 internal dielectric interface bound charge density
    gap1_ifaces = [
        item for item in unlinked.dielectric_interfaces
        if item.gap_index == 1 and item.kind == "dielectric_interface"
    ]
    assert len(gap1_ifaces) == 1
    _approx(gap1_ifaces[0].sigma_bound_c_per_m2, +1.57721259936e-8, rel=1e-9)
    # Verify local field jump at x = 0.003 (interface between 1mm k=1 and 2mm k=4)
    e_left = unlinked.electric_field_at(0.0025)
    e_right = unlinked.electric_field_at(0.004)
    _approx(e_left, 4.0 * e_right, rel=1e-12)
    with pytest.raises(SingularField, match="dielectric interface"):
        unlinked.electric_field_at(0.003)
    # Potential is continuous at the dielectric interface
    v_if = unlinked.potential_at(0.003)
    _approx(v_if, unlinked.plate_results[1].potential_v - 0.001 * e_left, rel=1e-12)
    _approx(v_if, unlinked.plate_results[2].potential_v + 0.002 * e_right, rel=1e-12)

    # Close internal wire P1-P2
    sw = switch_conductor_stack(
        unlinked,
        (
            _group("p0", Grounded()),
            ConductorGroup(("p1", "p2"), Floating()),
            _group("p3", FixedPotential(12.0)),
        ),
    )
    _approx(sw.after.plate_results[1].potential_v, 11.8235226684, rel=1e-9)
    _approx(sw.after.plate_results[2].potential_v, 11.8235226684, rel=1e-9)
    assert sw.after.gap_displacements_c_per_m2[1] == 0.0
    linked_if = [
        item for item in sw.after.dielectric_interfaces
        if item.gap_index == 1 and item.kind == "dielectric_interface"
    ][0]
    assert linked_if.sigma_bound_c_per_m2 == 0.0
    _approx(sw.supply_work_j, 1.41949133942e-8, rel=1e-9)
    _approx(sw.delta_field_energy_j, 1.02147635662e-8, rel=1e-9)
    _approx(sw.energy_residual_j, 3.98014982806e-9, rel=1e-9)

    # Exact sign reversal at -12 V, -5 nC
    rev_prob = replace(
        unlinked_prob,
        groups=(
            _group("p0", Grounded()),
            ConductorGroup(("p1", "p2"), FixedNetCharge(-5e-9)),
            _group("p3", FixedPotential(-12.0)),
        ),
    )
    rev = solve_planar_stack(rev_prob)
    for pos_r, neg_r in zip(sw.after.plate_results, rev.plate_results):
        assert isclose(pos_r.potential_v + neg_r.potential_v, 0.0, abs_tol=1e-12)
        assert isclose(pos_r.net_charge_c + neg_r.net_charge_c, 0.0, abs_tol=1e-22)


def test_m43_u2_three_sphere_concentric_dielectric_switch():
    conductors = (
        ConcentricConductor("s0", 0.0, 0.10),
        ConcentricConductor("s1", 0.20, 0.22),
        ConcentricConductor("s2", 0.30, 0.32),
    )
    gap_layers = (
        (DielectricLayer(0.10, 3.0),),
        (DielectricLayer(0.08, 1.0),),
    )
    before = solve_spherical_stack(
        SphericalStackProblem(
            conductors,
            (_group("s0", FixedNetCharge(2e-9)), _group("s1", FixedNetCharge(-1e-9)), _group("s2", FixedNetCharge(3e-9))),
            gap_layers=gap_layers,
        )
    )
    sw = switch_conductor_stack(
        before,
        (ConductorGroup(("s0", "s2"), Floating()), _group("s1", FixedPotential(-5.0))),
    )
    v_after = tuple(r.potential_v for r in sw.after.conductor_results)
    _approx(v_after[0], 21.6692028432, rel=1e-9)
    _approx(v_after[1], -5.0, rel=1e-12)
    _approx(v_after[2], 21.6692028432, rel=1e-9)
    _approx(sw.energy_residual_j, 2.11492475579e-7, rel=1e-9)
    # Inner dielectric face sigma_b at r = 0.10 m and exact gap-0 bound-charge sum == 0
    gap0_ifaces = [item for item in sw.after.dielectric_interfaces if item.gap_index == 0]
    inner_face = [item for item in gap0_ifaces if item.kind == "conductor_outer"][0]
    _approx(inner_face.sigma_bound_c_per_m2, -9.44536523164e-9, rel=1e-9)
    assert abs(sum(item.total_bound_charge_c for item in gap0_ifaces)) < 1e-23


# ---------------------------------------------------------------------------
# Multi-layer Spherical Gaps, Exterior Dielectric Coating & Gauss E-Law
# ---------------------------------------------------------------------------


def test_spherical_multilayer_gap_and_exterior_dielectric_coating():
    # Single solid sphere R=0.10 coated with 2 concentric dielectric shells (0.10..0.15 k=2, 0.15..0.25 k=5) in vacuum
    q = 4e-9
    prob = SphericalStackProblem(
        (ConcentricConductor("core", 0.0, 0.10),),
        (_group("core", FixedNetCharge(q)),),
        exterior_layers=(DielectricLayer(0.05, 2.0), DielectricLayer(0.10, 5.0)),
    )
    sol = solve_spherical_stack(prob)
    k0 = 1.0 / (4.0 * pi * EPSILON_0)
    # Independent closed-form potential of coated sphere:
    v_expected = q * k0 * ((1.0 / 0.10 - 1.0 / 0.15) / 2.0 + (1.0 / 0.15 - 1.0 / 0.25) / 5.0 + 1.0 / 0.25)
    _approx(sol.conductor_results[0].potential_v, v_expected, rel=1e-12)
    _approx(sol.field_energy_j, 0.5 * q * v_expected, rel=1e-12)

    # Local potential & field inside layer 0 (r=0.12), layer 1 (r=0.20), and vacuum exterior (r=0.50)
    _approx(sol.potential_at((0.12, 0.0, 0.0)), q * k0 * ((1 / 0.12 - 1 / 0.15) / 2 + (1 / 0.15 - 1 / 0.25) / 5 + 1 / 0.25), rel=1e-12)
    _approx(sol.potential_at((0.20, 0.0, 0.0)), q * k0 * ((1 / 0.20 - 1 / 0.25) / 5 + 1 / 0.25), rel=1e-12)
    _approx(sol.potential_at((0.50, 0.0, 0.0)), q * k0 / 0.50, rel=1e-12)

    _approx(sol.electric_field_at((0.12, 0.0, 0.0))[0], q * k0 / (2.0 * 0.12**2), rel=1e-12)
    _approx(sol.electric_field_at((0.20, 0.0, 0.0))[0], q * k0 / (5.0 * 0.20**2), rel=1e-12)
    _approx(sol.electric_field_at((0.50, 0.0, 0.0))[0], q * k0 / (0.50**2), rel=1e-12)

    # Continuity of V and SingularField jump of E at dielectric interfaces r=0.15 and r=0.25
    _approx(sol.potential_at((0.15, 0.0, 0.0)), q * k0 * ((1 / 0.15 - 1 / 0.25) / 5 + 1 / 0.25), rel=1e-12)
    _approx(sol.potential_at((0.25, 0.0, 0.0)), q * k0 / 0.25, rel=1e-12)
    for r_if in (0.15, 0.25):
        with pytest.raises(SingularField, match="dielectric interface"):
            sol.electric_field_at((r_if, 0.0, 0.0))
    # One-sided limits 1 ULP inside/outside r=0.15 match layer 0 (k=2) and layer 1 (k=5)
    _approx(sol.electric_field_at((nextafter(0.15, 0.0), 0.0, 0.0))[0], q * k0 / (2.0 * 0.15**2), rel=1e-12)
    _approx(sol.electric_field_at((nextafter(0.15, 1.0), 0.0, 0.0))[0], q * k0 / (5.0 * 0.15**2), rel=1e-12)

    # Bound charge conservation across the entire coating: sum(Q_b) == 0 identically
    ext_ifaces = [item for item in sol.dielectric_interfaces if item.region == "exterior"]
    assert len(ext_ifaces) == 3
    _approx(ext_ifaces[0].total_bound_charge_c, -(1 - 1 / 2) * q, rel=1e-12)
    _approx(ext_ifaces[1].total_bound_charge_c, (1 / 5 - 1 / 2) * q, rel=1e-12)
    _approx(ext_ifaces[2].total_bound_charge_c, (1 - 1 / 5) * q, rel=1e-12)
    assert abs(sum(item.total_bound_charge_c for item in ext_ifaces)) < 1e-23

    # Quasi-static removal of exterior dielectric coating at fixed Q: W_ext == Delta_U > 0, E_res == 0
    bare_prob = replace(prob, exterior_layers=())
    tr = quasistatic_stack_transition(sol, bare_prob)
    assert tr.supply_work_j == 0.0
    assert tr.energy_residual_j == 0.0
    assert tr.delta_field_energy_j > 0.0
    _approx(tr.external_work_j, tr.delta_field_energy_j, rel=1e-12)


def test_uniform_permittivity_limit_matches_m42b_exactly():
    # Planar: 2-layer gap with identical kappa=3 must match homogeneous M42b permittivity=3*EPSILON_0
    groups_p = (_group("a", FixedPotential(15.0)), _group("b", Grounded()))
    m42b_p = solve_planar_stack(PlanarStackProblem(
        ("a", "b"), (0.0, 0.006), 0.1, groups_p,
        d_left_c_per_m2=0.0, d_right_c_per_m2=0.0, permittivity_f_per_m=3.0 * EPSILON_0,
    ))
    m43_p = solve_planar_stack(PlanarStackProblem(
        ("a", "b"), (0.0, 0.006), 0.1, groups_p,
        d_left_c_per_m2=0.0, d_right_c_per_m2=0.0,
        gap_layers=((DielectricLayer(0.002, 3.0), DielectricLayer(0.004, 3.0)),),
    ))
    _approx(m43_p.field_energy_j, m42b_p.field_energy_j, rel=1e-13)
    assert m43_p.gap_fields_v_per_m == m42b_p.gap_fields_v_per_m
    # Continuous E at artificial same-kappa split boundary x = 0.002
    _approx(m43_p.electric_field_at(0.002), m42b_p.electric_field_at(0.002), rel=1e-13)

    # Spherical: 2-layer gap with identical kappa=4 and exterior kappa=4 matches homogeneous M42b
    conds = (ConcentricConductor("a", 0.0, 0.1), ConcentricConductor("b", 0.3, 0.4))
    groups_s = (_group("a", FixedPotential(20.0)), _group("b", Grounded()))
    m42b_s = solve_spherical_stack(SphericalStackProblem(conds, groups_s, permittivity_f_per_m=4.0 * EPSILON_0))
    m43_s = solve_spherical_stack(SphericalStackProblem(
        conds, groups_s, permittivity_f_per_m=4.0 * EPSILON_0,
        gap_layers=((DielectricLayer(0.08, 4.0), DielectricLayer(0.12, 4.0)),),
    ))
    _approx(m43_s.field_energy_j, m42b_s.field_energy_j, rel=1e-13)
    for x in (0.14, 0.18, 0.25):
        _approx(m43_s.potential_at((x, 0.0, 0.0)), m42b_s.potential_at((x, 0.0, 0.0)), rel=1e-13)
        _approx(m43_s.electric_field_at((x, 0.0, 0.0))[0], m42b_s.electric_field_at((x, 0.0, 0.0))[0], rel=1e-13)


def test_unfamiliar_shifted_three_conductor_multilayer_composition_via_public_api():
    # Shifted 3-conductor stack (solid core + thin shell + thick shell) with multi-layer gaps AND exterior coating
    shift = (25.0, -15.0, 8.0)
    conds = (
        ConcentricConductor("core", 0.0, 0.10),
        ConcentricConductor("mid", 0.20, 0.20),
        ConcentricConductor("out", 0.35, 0.40),
    )
    gap_layers = (
        (DielectricLayer(0.04, 2.0), DielectricLayer(0.06, 5.0)),
        (DielectricLayer(0.05, 3.0), DielectricLayer(0.10, 1.5)),
    )
    ext_layers = (DielectricLayer(0.10, 4.0),)
    prob = SphericalStackProblem(
        conds,
        (_group("core", FixedNetCharge(3e-9)), _group("mid", FixedNetCharge(-2e-9)), _group("out", FixedNetCharge(1e-9))),
        center_m=shift,
        gap_layers=gap_layers,
        exterior_layers=ext_layers,
    )
    before = solve_spherical_stack(prob)
    sw = switch_conductor_stack(
        before,
        (ConductorGroup(("core", "out"), Floating()), _group("mid", FixedPotential(10.0))),
    )
    # Independent exact-rational elastances & energy checks
    inv_k0 = F(4) * F(pi) * F(EPSILON_0)
    a0 = (F(10) - F(100, 14)) / (inv_k0 * 2) + (F(100, 14) - F(5)) / (inv_k0 * 5)
    a1 = (F(5) - F(4)) / (inv_k0 * 3) + (F(4) - F(100, 35)) / (inv_k0 * F(3, 2))
    a_ext = (F(100, 40) - F(2)) / (inv_k0 * 4) + F(2) / inv_k0
    # Verify Maxwell capacitance matrix off-diagonals match 1/a0 and 1/a1
    _approx(sw.after.capacitance_matrix_f[0][1], -float(1 / a0), rel=1e-12)
    _approx(sw.after.capacitance_matrix_f[1][2], -float(1 / a1), rel=1e-12)
    _approx(sw.after.capacitance_matrix_f[2][2], float(1 / a1 + 1 / a_ext), rel=1e-12)
    assert sw.energy_residual_j > 0.0
    # Gauss's law for E inside layer 1 of gap 0 (r=0.17): eps0 * 4*pi*r^2 * E_r == Q_free + Q_bound_inner
    q_enc0 = sw.after.conductor_results[0].outer_surface_charge_c
    qb_in0 = sum(
        item.total_bound_charge_c for item in sw.after.dielectric_interfaces
        if item.gap_index == 0 and item.coordinate_m < 0.17
    )
    e_r = sw.after.electric_field_at((shift[0] + 0.17, shift[1], shift[2]))[0]
    _approx(EPSILON_0 * 4.0 * pi * (0.17**2) * e_r, q_enc0 + qb_in0, rel=1e-12)
    # Net bound charge is identically zero in every gap and in the exterior coating
    for g in (0, 1):
        assert abs(sum(item.total_bound_charge_c for item in sw.after.dielectric_interfaces if item.gap_index == g)) < 1e-23
    assert abs(sum(item.total_bound_charge_c for item in sw.after.dielectric_interfaces if item.region == "exterior")) < 1e-23


# ---------------------------------------------------------------------------
# Honest Refusals
# ---------------------------------------------------------------------------


def test_dielectric_and_quasistatic_refusals():
    with pytest.raises(UnsupportedProblem, match="relative_permittivity >= 1"):
        DielectricLayer(0.001, 0.8)
    with pytest.raises(ValueError):
        DielectricLayer(0.0, 2.0)
    with pytest.raises(ValueError):
        DielectricLayer(True, 2.0)

    # Mismatched layer thickness sum
    with pytest.raises(UnsupportedProblem, match="sum exactly to the gap width"):
        PlanarStackProblem(
            ("a", "b"), (0.0, 0.005), 0.1, (_group("a", Grounded()), _group("b", FixedPotential(5.0))),
            d_left_c_per_m2=0.0, d_right_c_per_m2=0.0,
            gap_layers=((DielectricLayer(0.002, 2.0), DielectricLayer(0.002, 3.0)),),
        )

    # Non-vacuum permittivity_f_per_m combined with planar gap_layers
    with pytest.raises(UnsupportedProblem, match="relative to vacuum EPSILON_0"):
        PlanarStackProblem(
            ("a", "b"), (0.0, 0.002), 0.1, (_group("a", Grounded()), _group("b", FixedPotential(5.0))),
            d_left_c_per_m2=0.0, d_right_c_per_m2=0.0,
            permittivity_f_per_m=2.0 * EPSILON_0,
            gap_layers=((DielectricLayer(0.002, 2.0),),),
        )

    # Active point charge inside multi-layer dielectric spherical gap
    with pytest.raises(UnsupportedProblem, match="multi-layer dielectric gap"):
        solve_spherical_stack(SphericalStackProblem(
            (ConcentricConductor("a", 0.0, 1.0), ConcentricConductor("b", 3.0, 4.0)),
            (_group("a", Grounded()), _group("b", Grounded())),
            sources=(StackPointCharge("q", 1e-9, (2.0, 0.0, 0.0)),),
            gap_layers=((DielectricLayer(1.0, 2.0), DielectricLayer(1.0, 4.0)),),
        ))

    # Active point charge outside dielectric-coated spherical stack
    with pytest.raises(UnsupportedProblem, match="dielectric-coated"):
        solve_spherical_stack(SphericalStackProblem(
            (ConcentricConductor("a", 0.0, 1.0),),
            (_group("a", Grounded()),),
            sources=(StackPointCharge("q", 1e-9, (5.0, 0.0, 0.0)),),
            exterior_layers=(DielectricLayer(0.5, 3.0),),
        ))

    # Quasistatic transition refuses terminal voltage change (irreversible switch!)
    p0 = PlanarStackProblem(
        ("a", "b"), (0.0, 0.002), 0.1, (_group("a", FixedPotential(10.0)), _group("b", Grounded())),
        d_left_c_per_m2=0.0, d_right_c_per_m2=0.0,
    )
    sol0 = solve_planar_stack(p0)
    with pytest.raises(UnsupportedProblem, match="unchanged electrical connections"):
        quasistatic_stack_transition(sol0, replace(p0, groups=(_group("a", FixedPotential(20.0)), _group("b", Grounded()))))
    with pytest.raises(UnsupportedProblem, match="unchanged electrical connections"):
        quasistatic_stack_transition(sol0, replace(p0, groups=(ConductorGroup(("a", "b"), Grounded()),)))
    # Malformed group object raises UnsupportedProblem (never raw AttributeError)
    with pytest.raises(UnsupportedProblem, match="ConductorGroup"):
        quasistatic_stack_transition(sol0, replace(p0, groups=(None,)))


def test_sol_review_regressions_boundary_snapping_precision_overflow_and_matching_media():
    g = (_group("a", FixedPotential(1.0)), _group("b", Grounded()))
    L = DielectricLayer

    # Repro A: snapped interior boundary must never create a negative-width final slab
    s_a = solve_planar_stack(PlanarStackProblem(
        ("a", "b"), (0.1, nextafter(0.8, 1.0)), 1.0, g,
        d_left_c_per_m2=0.0, d_right_c_per_m2=0.0,
        gap_layers=((L(0.7, 1e18), L(1.4e-16, 1e18), L(5.4e-17, 1.0)),),
    ))
    assert s_a.capacitance_matrix_f[0][0] > 0.0
    assert s_a.field_energy_j > 0.0
    assert all(lf.x_start_m <= lf.x_end_m for lf in s_a.gap_layer_fields)

    # Repro B1: 1-ULP spherical gap potential with single vacuum layer must match M42b exactly
    a = nextafter(1.0, 0.0)
    s_b1_m42b = solve_spherical_stack(SphericalStackProblem(
        (ConcentricConductor("a", 0.0, a), ConcentricConductor("b", 1.0, 2.0)), g,
    ))
    s_b1_m43 = solve_spherical_stack(SphericalStackProblem(
        (ConcentricConductor("a", 0.0, a), ConcentricConductor("b", 1.0, 2.0)), g,
        gap_layers=((L(1.0 - a, 1.0),),),
    ))
    _approx(s_b1_m43.potential_at((a, 1e-8, 0.0)), s_b1_m42b.potential_at((a, 1e-8, 0.0)), rel=1e-12)
    _approx(s_b1_m43.potential_at((a, 1e-8, 0.0)), 0.5496400373, rel=1e-9)

    # Repro B2: sub-ULP spherical dielectric interface must not round onto inner conductor
    b2 = nextafter(1.0, 2.0)
    s_b2 = solve_spherical_stack(SphericalStackProblem(
        (ConcentricConductor("a", 0.0, 1.0), ConcentricConductor("b", b2, 2.0)), g,
        gap_layers=((L(2**-53, 1.0), L(2**-53, 4.0)),),
    ))
    # Point (1, 1e-8, 0) has r^2 = 1 + 1e-16 < (1 + 2^-53)^2 = 1 + 2.22e-16, so it is in layer 0 (k=1)
    # Effective elastance ~ (2^-53/1 + 2^-53/4) / (4*pi*eps0) = (5/4)*2^-53 / (4*pi*eps0) -> E_layer0 = 1 / ((5/4)*2^-53)
    e_b2 = s_b2.electric_field_at((1.0, 1e-8, 0.0))[0]
    _approx(e_b2, 1.0 / (1.25 * 2**-53), rel=1e-10)

    # Repro C: extreme-scale homogeneous M42b planar problem must not overflow layer metadata
    s_c = solve_planar_stack(PlanarStackProblem(
        ("a", "b"), (0.0, 1.0), 1e-299, g,
        d_left_c_per_m2=0.0, d_right_c_per_m2=0.0, permittivity_f_per_m=1e299,
    ))
    _approx(s_c.capacitance_matrix_f[0][0], 1.0, rel=1e-12)
    _approx(s_c.field_energy_j, 0.5, rel=1e-12)
    assert s_c.gap_layer_fields == ()

    # Repro D3: coating with kappa=1.1 matching background 1.1*EPSILON_0 is continuous at outer radius
    s_d3 = solve_spherical_stack(SphericalStackProblem(
        (ConcentricConductor("a", 0.0, 1.0),),
        (_group("a", FixedNetCharge(1e-9)),),
        permittivity_f_per_m=1.1 * EPSILON_0,
        exterior_layers=(L(1.0, 1.1),),
    ))
    e_d3 = s_d3.electric_field_at((2.0, 0.0, 0.0))[0]
    _approx(e_d3, 1e-9 / (4.0 * pi * 1.1 * EPSILON_0 * 4.0), rel=1e-12)
    outer_if = [i for i in s_d3.dielectric_interfaces if i.region == "exterior"][-1]
    assert outer_if.total_bound_charge_c == 0.0

    # Repro D4: exterior coating radius overflow raises UnsupportedProblem (never raw OverflowError)
    with pytest.raises(UnsupportedProblem, match="binary64 range"):
        SphericalStackProblem(
            (ConcentricConductor("a", 0.0, 1e308),),
            (_group("a", FixedNetCharge(1e-9)),),
            exterior_layers=(L(1e308, 2.0),),
        )

    # Planar voltage-offset invariance for quasistatic_stack_transition
    p_base = PlanarStackProblem(
        ("a", "b"), (0.0, 0.002), 0.05,
        (_group("a", FixedPotential(0.0)), _group("b", FixedPotential(100.0))),
        d_left_c_per_m2=0.0, d_right_c_per_m2=0.0,
        gap_layers=((L(0.002, 4.0),),),
    )
    p_shift = replace(p_base, groups=(_group("a", FixedPotential(37.5)), _group("b", FixedPotential(137.5))))
    t_base = quasistatic_stack_transition(solve_planar_stack(p_base), replace(p_base, gap_layers=((L(0.002, 1.0),),)))
    t_shift = quasistatic_stack_transition(solve_planar_stack(p_shift), replace(p_shift, gap_layers=((L(0.002, 1.0),),)))
    _approx(t_shift.supply_work_j, t_base.supply_work_j, rel=1e-13)
    _approx(t_shift.external_work_j, t_base.external_work_j, rel=1e-13)
    _approx(t_shift.delta_field_energy_j, t_base.delta_field_energy_j, rel=1e-13)

    # Sol v2 Finding 1a: sub-ULP planar layers must NOT be altered by decimal snapping
    w1, w2 = 5.0 * 2**-55, 11.0 * 2**-55
    s_sub_p = solve_planar_stack(PlanarStackProblem(
        ("a", "b"), (1.0, 1.0 + 2.0 * (nextafter(1.0, 2.0) - 1.0)), 1.0, g,
        d_left_c_per_m2=0.0, d_right_c_per_m2=0.0,
        gap_layers=((L(w1, 1.0), L(w2, 1e18)),),
    ))
    c_expected = 1.0 / (w1 / EPSILON_0 + w2 / (EPSILON_0 * 1e18))
    _approx(s_sub_p.capacitance_matrix_f[0][0], c_expected, rel=1e-12)

    # Sol v2 Finding 1b: sub-ULP spherical coating must NOT move across probe (1, 2e-8, 0)
    # r = sqrt(1 + 4e-16) ~ 1 + 2e-16 > 1 + 5*2^-55 (~1 + 1.39e-16), so probe is OUTSIDE coating in k=1e18 background
    s_sub_s = solve_spherical_stack(SphericalStackProblem(
        (ConcentricConductor("a", 0.0, 1.0),),
        (_group("a", FixedNetCharge(1e-9)),),
        permittivity_f_per_m=1e18 * EPSILON_0,
        exterior_layers=(L(w1, 1.0),),
    ))
    e_sub_s = s_sub_s.electric_field_at((1.0, 2e-8, 0.0))[0]
    _approx(e_sub_s, 1e-9 / (4.0 * pi * 1e18 * EPSILON_0 * (1.0 + 4e-16)), rel=1e-10)

    # Sol v2 Finding 2: wide planar gap (-1e308, 1e308) with 1-ULP thickness shift must not raise raw OverflowError
    s_wide = solve_planar_stack(PlanarStackProblem(
        ("a", "b"), (-1e308, 1e308), 1e308, g,
        d_left_c_per_m2=0.0, d_right_c_per_m2=0.0,
        gap_layers=((L(1e308, 1e10), L(nextafter(1e308, 0.0), 1e10)),),
    ))
    assert s_wide.capacitance_matrix_f[0][0] > 0.0

    # Sol v2 non-blocking: subnormal-radius coated sphere sampling retains full binary64 precision
    r_sub = 2.0**-1025  # genuine subnormal (< DBL_MIN = 2^-1022)
    q_sub = 2.0**-1065
    s_subnorm = solve_spherical_stack(SphericalStackProblem(
        (ConcentricConductor("a", 0.0, r_sub),),
        (_group("a", FixedNetCharge(q_sub)),),
        exterior_layers=(L(r_sub, 2.0),),
    ))
    # Probe at (3*r_sub, 4*r_sub, 0) -> exact r = 5*r_sub (in vacuum outside coating of radius 2*r_sub)
    v_subnorm = s_subnorm.potential_at((3.0 * r_sub, 4.0 * r_sub, 0.0))
    e_subnorm = s_subnorm.electric_field_at((3.0 * r_sub, 4.0 * r_sub, 0.0))
    _approx(v_subnorm, float(F(q_sub) / (F(4 * pi) * F(EPSILON_0) * F(5) * F(r_sub))), rel=1e-14)
    e_mag = float(F(q_sub) / (F(4 * pi) * F(EPSILON_0) * F(25) * F(r_sub) ** 2))
    _approx(e_subnorm[0], e_mag * (3.0 / 5.0), rel=1e-14)
    _approx(e_subnorm[1], e_mag * (4.0 / 5.0), rel=1e-14)
