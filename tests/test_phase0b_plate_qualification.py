"""Phase 0b witness suite — parallel-plate qualification, certified-domain guard, and D3 fixes.

Independent oracle: self-contained Fourier-Bessel K0/K1 trapezoidal quadrature with 50-digit
Decimal exact range reduction on encoded IEEE-754 inputs, verified by step-halving h -> h/2
and carrying an explicit per-point oracle summation floor (zero production helpers).
"""
from __future__ import annotations

from decimal import Decimal, getcontext
from math import cos, cosh, exp, fsum, hypot, pi, sin
import pytest

from conductor_sim import (
    ConductingPlane,
    EPSILON_0,
    ElectrostaticProblem,
    Femm5ParallelPlateBackend,
    FixedPotential,
    Grounded,
    ParallelPlateBackend,
    PointCharge,
    force_energy,
)
from conductor_sim.parallel_plates import _tail_bounds_orientation
from conductor_sim.solver import OutsideDomain, UnsupportedProblem
from conductor_sim.visual_adapters import context_for_solution
from conductor_sim.visualization import SliceSpec, sample_frame

getcontext().prec = 50
_PI_D = Decimal("3.1415926535897932384626433832795028841971693993751")
_U_MACH = 2.0 ** -53


def _bessel_k0_k1(x: float, h: float = 0.025) -> tuple[float, float]:
    k0_t, k1_t = [], []
    n = 0
    while True:
        t = n * h
        ct = cosh(t)
        arg = x * ct
        if arg > 70.0:
            break
        e = exp(-arg)
        w = 0.5 if n == 0 else 1.0
        k0_t.append(w * e)
        k1_t.append(w * ct * e)
        n += 1
    return h * fsum(k0_t), h * fsum(k1_t)


def _sin_cos_pi_m(m: int, u_dec: Decimal) -> tuple[float, float]:
    mu = (Decimal(m) * u_dec) % Decimal(2)
    if mu > Decimal(1):
        mu -= Decimal(2)
    if mu > Decimal("0.5"):
        a = float((Decimal(1) - mu) * _PI_D)
        return sin(a), -cos(a)
    if mu < Decimal("-0.5"):
        a = float((Decimal(1) + mu) * _PI_D)
        return -sin(a), -cos(a)
    a = float(mu * _PI_D)
    return sin(a), cos(a)


def _exact_grounded_slab(
    q: float, d: float, z0: float, dx: float, dy: float, z: float, eps: float = EPSILON_0, *, h: float = 0.025
) -> tuple[tuple[float, tuple[float, float, float], float, float], tuple[float, float, float, float]]:
    """Exact (V, (Ex,Ey,Ez), sigma_lo, sigma_hi) and per-observable oracle summation error floor."""
    rho = hypot(dx, dy)
    assert rho > 0.0
    d_dec = Decimal.from_float(d)
    u0_dec = Decimal.from_float(z0) / d_dec
    u_dec = Decimal.from_float(z) / d_dec
    v_t, er_t, ez_t, slo_t, shi_t = [], [], [], [], []
    m = 1
    while True:
        arg = m * pi * rho / d
        if arg > 65.0:
            break
        k0, k1 = _bessel_k0_k1(arg, h=h)
        s0, _ = _sin_cos_pi_m(m, u0_dec)
        sz, cz = _sin_cos_pi_m(m, u_dec)
        v_t.append(sz * s0 * k0)
        er_t.append(m * sz * s0 * k1)
        ez_t.append(m * cz * s0 * k0)
        slo_t.append(m * s0 * k0)
        shi_t.append(m * ((-1.0) ** m) * s0 * k0)
        m += 1
        assert m <= 4000
    cv = q / (pi * eps * d)
    ce = q / (eps * d * d)
    cs = q / (d * d)
    v = cv * fsum(v_t)
    er = ce * fsum(er_t)
    ez = -ce * fsum(ez_t)
    ex, ey = er * (dx / rho), er * (dy / rho)
    sig_lo = -cs * fsum(slo_t)
    sig_hi = +cs * fsum(shi_t)
    # Oracle per-term float64 rounding envelope (truncation tail at arg>65 is <1e-28)
    eps_v = 8.0 * _U_MACH * abs(cv) * fsum(abs(x) for x in v_t)
    eps_er = 8.0 * _U_MACH * abs(ce) * fsum(abs(x) for x in er_t)
    eps_ez = 8.0 * _U_MACH * abs(ce) * fsum(abs(x) for x in ez_t)
    eps_slo = 8.0 * _U_MACH * abs(cs) * fsum(abs(x) for x in slo_t)
    return (v, (ex, ey, ez), sig_lo, sig_hi), (eps_v, eps_er, eps_ez, eps_slo)


def _plates(d: float, v_lo: float = 0.0, v_hi: float = 0.0):
    c_lo = Grounded() if v_lo == 0.0 else FixedPotential(v_lo)
    c_hi = Grounded() if v_hi == 0.0 else FixedPotential(v_hi)
    return (
        ConductingPlane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), c_lo, "Lo"),
        ConductingPlane((0.0, 0.0, d), (0.0, 0.0, -1.0), c_hi, "Hi"),
    )


def test_oracle_quadrature_halving_accuracy():
    for x in (0.1, 0.5, 1.5, 5.0, 15.0):
        a0, a1 = _bessel_k0_k1(x, 0.025)
        b0, b1 = _bessel_k0_k1(x, 0.0125)
        assert abs(a0 - b0) / abs(b0) < 1e-14
        assert abs(a1 - b1) / abs(b1) < 1e-14
    (v1, e1, s1, _), _ = _exact_grounded_slab(1e-9, 0.05, 0.9 * 0.05, 0.1 * 0.05, 0.0, 0.5 * 0.05, h=0.025)
    (v2, e2, s2, _), _ = _exact_grounded_slab(1e-9, 0.05, 0.9 * 0.05, 0.1 * 0.05, 0.0, 0.5 * 0.05, h=0.0125)
    assert abs(v1 - v2) / abs(v2) < 1e-13
    assert hypot(*(e1[i] - e2[i] for i in range(3))) / hypot(*e2) < 1e-13
    assert abs(s1 - s2) / abs(s2) < 1e-13


# --- W-b1: Grounded wrong-sign & far-field refusals (RED at base 953af40) ---

@pytest.mark.parametrize("z0_frac,rho_frac,order", [
    (0.50, 7.0, 48),
    (0.50, 10.0, 512),
    (0.25, 5.0, 48),
    (0.25, 10.0, 512),
])
def test_wb1_grounded_wrong_sign_sigma_refused(z0_frac, rho_frac, order):
    d, q = 0.05, 1e-9
    prob = ElectrostaticProblem((PointCharge(q, (0.0, 0.0, z0_frac * d)),), _plates(d))
    sol = ParallelPlateBackend(image_order=order).solve(prob)
    with pytest.raises(OutsideDomain):
        sol.surface_charge_density_at("outer", (rho_frac * d, 0.0, 0.0), conductor_name="Lo")


def test_wb1_grounded_uncertified_v_and_e_refused():
    d, q = 0.05, 1e-9
    s_mid = ParallelPlateBackend(image_order=48).solve(
        ElectrostaticProblem((PointCharge(q, (0.0, 0.0, 0.5 * d)),), _plates(d))
    )
    with pytest.raises(OutsideDomain):
        s_mid.potential_at((5.0 * d, 0.0, 0.5 * d))
    s_off = ParallelPlateBackend(image_order=48).solve(
        ElectrostaticProblem((PointCharge(q, (0.0, 0.0, 0.25 * d)),), _plates(d))
    )
    with pytest.raises(OutsideDomain):
        s_off.electric_field_at((5.0 * d, 0.0, 0.35 * d))


# --- W-b2: Sol's 3 biased-plate wrong-sign counterexamples (RED at base) + positive biased control ---

def test_wb2_sol_biased_wrong_sign_counterexamples_refused():
    d = 0.08
    be = ParallelPlateBackend(image_order=96)
    sol_v = be.solve(ElectrostaticProblem(
        (PointCharge(1.7e-9, (0.0, 0.0, 0.21 * d)), PointCharge(-0.65e-9, (0.17 * d, 0.0, 0.69 * d))),
        _plates(d, -1.49827614098, +0.90172385902),
    ))
    with pytest.raises(OutsideDomain):
        sol_v.potential_at((1.21 * d, 0.0, 0.39 * d))

    sol_s = be.solve(ElectrostaticProblem(
        (PointCharge(1.7e-9, (0.0, 0.0, 0.21 * d)), PointCharge(-0.65e-9, (0.17 * d, 0.0, 0.69 * d))),
        _plates(d, 0.0, -3.74598676810),
    ))
    with pytest.raises(OutsideDomain):
        sol_s.surface_charge_density_at("outer", (1.19 * d, 0.0, 0.0), conductor_name="Lo")

    sol_e = be.solve(ElectrostaticProblem(
        (PointCharge(1.2e-9, (-0.85 * d, 0.0, 0.23 * d)), PointCharge(1.2e-9, (+0.85 * d, 0.0, 0.23 * d))),
        _plates(d, 0.0, -24.48390777579),
    ))
    with pytest.raises(OutsideDomain):
        sol_e.electric_field_at((0.0, 0.0, 0.39 * d))


def test_wb2_sol_biased_positive_control_admitted_and_accurate():
    d = 0.08
    v_lo, v_hi = -2.3, +6.7
    charges = (PointCharge(1.7e-9, (0.0, 0.0, 0.21 * d)), PointCharge(-0.65e-9, (0.17 * d, 0.0, 0.69 * d)))
    sol = ParallelPlateBackend(image_order=96).solve(ElectrostaticProblem(charges, _plates(d, v_lo, v_hi)))
    pt = (1.62 * d, 0.0, 0.37 * d)
    v_ref = v_lo + (v_hi - v_lo) * 0.37
    ez_bg = -(v_hi - v_lo) / d
    e_ref = [0.0, 0.0, ez_bg]
    slo_ref = EPSILON_0 * ez_bg
    shi_ref = -EPSILON_0 * ez_bg
    for c in charges:
        (vc, ec, slc, shc), _ = _exact_grounded_slab(c.charge_c, d, c.position_m[2], pt[0] - c.position_m[0], 0.0, pt[2])
        v_ref += vc
        for i in range(3):
            e_ref[i] += ec[i]
        slo_ref += slc
        shi_ref += shc
    assert sol.potential_at(pt) == pytest.approx(v_ref, rel=0.01)
    e_got = sol.electric_field_at(pt)
    assert hypot(*(e_got[i] - e_ref[i] for i in range(3))) <= 0.01 * hypot(*e_ref)
    assert sol.surface_charge_density_at("outer", (pt[0], 0.0, 0.0), conductor_name="Lo") == pytest.approx(slo_ref, rel=0.01)
    assert sol.surface_charge_density_at("outer", (pt[0], 0.0, d), conductor_name="Hi") == pytest.approx(shi_ref, rel=0.01)


# --- W-b3: rho/d <= 2 admitted controls under default backend (N=96) with sign reversal ---

@pytest.mark.parametrize("z0_frac", [0.25, 0.50])
@pytest.mark.parametrize("rho_frac", [0.5, 2.0])
@pytest.mark.parametrize("sign", [+1.0, -1.0])
def test_wb3_default_backend_admits_rho_le_2_controls(z0_frac, rho_frac, sign):
    d, q = 0.05, sign * 1e-9
    prob = ElectrostaticProblem((PointCharge(q, (0.0, 0.0, z0_frac * d)),), _plates(d))
    sol = ParallelPlateBackend().solve(prob)
    assert sol.image_order == 96
    zp = 0.35 * d
    (v_ref, e_ref, slo_ref, _), _ = _exact_grounded_slab(q, d, z0_frac * d, rho_frac * d, 0.0, zp)
    assert sol.potential_at((rho_frac * d, 0.0, zp)) == pytest.approx(v_ref, rel=0.01)
    e_got = sol.electric_field_at((rho_frac * d, 0.0, zp))
    assert hypot(*(e_got[i] - e_ref[i] for i in range(3))) <= 0.01 * hypot(*e_ref)
    assert sol.surface_charge_density_at("outer", (rho_frac * d, 0.0, 0.0), conductor_name="Lo") == pytest.approx(slo_ref, rel=0.01)


# --- W-b4: D3 tilted-plane refusal, tiny-gap scaling, R-b4 consistency check & rescaled normals ---

def test_wb4_d3_tilted_planes_refused_by_analytic_and_femm_preflight():
    tilted = ElectrostaticProblem(
        (),
        (
            ConductingPlane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), Grounded(), "Lo"),
            ConductingPlane((0.0, 0.0, 1.0), (5e-13, 0.0, -1.0), FixedPotential(10.0), "Hi"),
        ),
    )
    assert not ParallelPlateBackend().supports(tilted)
    with pytest.raises(UnsupportedProblem):
        ParallelPlateBackend().solve(tilted)

    def never():
        raise AssertionError("FEMM session must never open on preflight refusal")

    femm = Femm5ParallelPlateBackend(session_factory=never)
    assert not femm.supports(tilted)
    with pytest.raises(UnsupportedProblem):
        femm.solve(tilted)


def test_wb4_d3_tiny_gap_midpoint_field_and_scaled_surface_tolerance():
    d = 1e-14
    prob = ElectrostaticProblem((), _plates(d, 0.0, 1e-3))
    sol = ParallelPlateBackend().solve(prob)
    mid = (0.0, 0.0, 0.5 * d)
    assert sol.potential_at(mid) == pytest.approx(0.5e-3, rel=1e-12)
    assert sol.electric_field_at(mid) == pytest.approx((0.0, 0.0, -1e11), rel=1e-12)
    assert sol.surface_charge_density_at("outer", (0.0, 0.0, 0.0), conductor_name="Lo") == pytest.approx(
        -EPSILON_0 * 1e11, rel=1e-12
    )
    with pytest.raises(ValueError):
        sol.surface_charge_density_at("outer", (0.0, 0.0, 1e-12), conductor_name="Lo")


def test_wb4_rb4_sub_threshold_tilt_refused_by_halfspace_consistency_and_rescaled_normals_admitted():
    # Sub-1e-15 tilt passes _layout normal check, but at x = 1e12 m the slab gap drifts by 5e-4 m >> 1e-12 m
    sub_tilt = ElectrostaticProblem(
        (),
        (
            ConductingPlane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), Grounded(), "Lo"),
            ConductingPlane((0.0, 0.0, 1.0), (5e-16, 0.0, -1.0), FixedPotential(10.0), "Hi"),
        ),
    )
    sol = ParallelPlateBackend().solve(sub_tilt)
    assert sol.potential_at((0.0, 0.0, 0.5)) == pytest.approx(5.0)
    with pytest.raises(OutsideDomain, match="geometric precision"):
        sol.potential_at((1e12, 0.0, 0.5))
    with pytest.raises(OutsideDomain, match="geometric precision"):
        sol.electric_field_at((1e12, 0.0, 0.5))

    # Rescaled legitimate opposite normals must stay admitted
    for n_lo, n_hi in (
        ((3.0, 4.0, 0.0), (-0.6, -0.8, 0.0)),
        ((1.0, 1.0, 1.0), (-3.0, -3.0, -3.0)),
    ):
        mag = hypot(*n_lo)
        unit_n = tuple(c / mag for c in n_lo)
        upper_pt = tuple(0.2 * c for c in unit_n)
        prob = ElectrostaticProblem(
            (),
            (
                ConductingPlane((0.0, 0.0, 0.0), n_lo, Grounded(), "Lo"),
                ConductingPlane(upper_pt, n_hi, FixedPotential(10.0), "Hi"),
            ),
        )
        assert ParallelPlateBackend().supports(prob)
        s_ok = ParallelPlateBackend().solve(prob)
        mid = tuple(0.1 * c for c in unit_n)
        assert s_ok.potential_at(mid) == pytest.approx(5.0, rel=1e-12)


# --- W-b5: Signed multi-source cancellation refusals & exact exemptions ---

def test_wb5_signed_cancellation_refused_and_exact_exemptions_preserved():
    d, q = 0.05, 1e-9
    cancel = ElectrostaticProblem(
        (PointCharge(+q, (-0.01, 0.0, 0.5 * d)), PointCharge(-q, (+0.01, 0.0, 0.5 * d))),
        _plates(d),
    )
    sol_c = ParallelPlateBackend(image_order=48).solve(cancel)
    with pytest.raises(OutsideDomain):
        sol_c.surface_charge_density_at("outer", (0.0, 2.0 * d, 0.0), conductor_name="Lo")

    empty = ParallelPlateBackend(image_order=4).solve(ElectrostaticProblem((), _plates(d, -5.0, +15.0)))
    assert empty.potential_at((100.0 * d, 0.0, 0.5 * d)) == pytest.approx(5.0)
    assert empty.electric_field_at((100.0 * d, 0.0, 0.5 * d)) == pytest.approx((0.0, 0.0, -20.0 / d))
    assert empty.surface_charge_density_at("outer", (100.0 * d, 0.0, 0.0), conductor_name="Lo") == pytest.approx(
        -EPSILON_0 * 20.0 / d
    )
    assert sol_c.potential_at((10.0 * d, 0.0, 0.0)) == 0.0
    assert sol_c.potential_at((10.0 * d, 0.0, d)) == 0.0


# --- W-b6: R-b3 low-N visualization masking witness & M13a force_energy regression ---

def test_wb6_rb3_low_order_visualization_masks_uncertified_cells_and_terminates_lines():
    d = 0.2
    prob = ElectrostaticProblem(
        (PointCharge(1e-9, (0.0, 0.0, 0.065)), PointCharge(-0.5e-9, (0.0, 0.0, 0.14))),
        _plates(d, 0.0, 0.0),
    )
    sol = ParallelPlateBackend(image_order=8).solve(prob)
    plane = SliceSpec.meridian(prob, extent_m=0.29, resolution=15)
    frame = sample_frame(context_for_solution(prob, sol), plane, seeds=None, line_steps=80, profile_samples=21)
    regions = frame["grid"]["region"]
    potentials = frame["grid"]["potential_v"]
    fields = frame["grid"]["electric_field_v_per_m"]
    assert any(r == "outside-domain" for row in regions for r in row)
    for r_row, v_row, e_row in zip(regions, potentials, fields):
        for r, v, e in zip(r_row, v_row, e_row):
            if r == "outside-domain":
                assert v is None or e is None
    assert frame["line_termination_counts"].get("outside-domain", 0) > 0
    assert any(v is None for prof in frame["profiles"] for v in prof["sigma_c_per_m2"])
    # Existing force_energy remains unaffected on explicit order
    fe = force_energy(sol)
    assert fe.image_order == 8 and len(fe.charges) == 2


# --- R-b1 (iii): Adversarial ratio grid & explicit plate-swap + 16u representational floor checks ---

def test_rb1_plate_swap_symmetry_and_16u_representational_floor_visible():
    d, q = 0.05, 1e-9
    sol_a = ParallelPlateBackend(image_order=48).solve(
        ElectrostaticProblem((PointCharge(q, (0.0, 0.0, 0.25 * d)),), _plates(d))
    )
    sol_b = ParallelPlateBackend(image_order=48).solve(
        ElectrostaticProblem((PointCharge(q, (0.0, 0.0, 0.75 * d)),), _plates(d))
    )
    # Swapping (u, u0) <-> (1-u, 1-u0) gives identical bounds via min(b1, b2)
    _, bv_a = sol_a._source_potential_and_bound((0.5 * d, 0.0, 0.35 * d), sol_a.sources[0])
    _, bv_b = sol_b._source_potential_and_bound((0.5 * d, 0.0, 0.65 * d), sol_b.sources[0])
    assert bv_a == pytest.approx(bv_b, rel=1e-14)
    _, ben_a, ber_a = sol_a._source_field_and_bounds((0.5 * d, 0.0, 0.35 * d), sol_a.sources[0])
    _, ben_b, ber_b = sol_b._source_field_and_bounds((0.5 * d, 0.0, 0.65 * d), sol_b.sources[0])
    assert ben_a == pytest.approx(ben_b, rel=1e-14)
    assert ber_a == pytest.approx(ber_b, rel=1e-14)

    # Confirm 16*u_mach representational floor strictly lifts bound above pure truncation tail
    tail_v = min(
        _tail_bounds_orientation(d, 48, 0.35, 0.25, 0.5 * d)[0],
        _tail_bounds_orientation(d, 48, 0.65, 0.75, 0.5 * d)[0],
    ) * sol_a._k * q
    assert bv_a > tail_v


def test_rb1_adversarial_remainder_bound_ratio_le_one():
    d, q = 0.05, 1e-9
    plates = _plates(d)
    Ns = (1, 2, 4, 8, 48, 96, 512)
    u0s = (0.01, 0.25, 0.50, 0.90)
    us = (0.01, 0.35, 0.50, 0.99)
    rhos = (0.1, 0.5, 2.0, 5.0, 10.0, 100.0, 1000.0)

    for N in Ns:
        be = ParallelPlateBackend(image_order=N)
        for u0 in u0s:
            sol = be.solve(ElectrostaticProblem((PointCharge(q, (0.0, 0.0, u0 * d)),), plates))
            src = sol.sources[0]
            for rf in rhos:
                rho = rf * d
                (_, _, sig_ref, _), (_, _, _, eps_sig) = _exact_grounded_slab(q, d, u0 * d, rho, 0.0, 0.5 * d)
                f_lo, ben_lo, _ = sol._field_inside_and_bounds((rho, 0.0, 0.0))
                esig = abs(EPSILON_0 * f_lo[2] - sig_ref)
                assert esig <= EPSILON_0 * ben_lo + eps_sig, f"sigma violation N={N} u0={u0} rho/d={rf}"

                for u in us:
                    (v_ref, e_ref, _, _), (eps_v, eps_er, eps_ez, _) = _exact_grounded_slab(q, d, u0 * d, rho, 0.0, u * d)
                    pt = (rho, 0.0, u * d)
                    v_hat, bv = sol._source_potential_and_bound(pt, src)
                    e_hat, ben, ber = sol._field_inside_and_bounds(pt)
                    assert abs(v_hat - v_ref) <= bv + eps_v, f"V violation N={N} u0={u0} u={u} rho/d={rf}"
                    assert abs(e_hat[2] - e_ref[2]) <= ben + eps_ez, f"Ez violation N={N} u0={u0} u={u} rho/d={rf}"
                    assert abs(e_hat[0] - e_ref[0]) <= ber + eps_er, f"Erho violation N={N} u0={u0} u={u} rho/d={rf}"
