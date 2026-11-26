"""M45 acceptance suite: UniformGravity coupled with responding-conductor dynamics & equilibria."""
from __future__ import annotations

from math import isclose, pi, sqrt

import pytest

from conductor_sim import (
    ConductingGravityEquilibrium,
    ConductingPlane,
    EPSILON_0,
    ElectrostaticProblem,
    ExternalGroundedSphereBackend,
    ExternalIsolatedSphereBackend,
    FixedNetCharge,
    FixedPotential,
    Grounded,
    GroundedPlaneBackend,
    Motion1DSpec,
    Motion2DSpec,
    MultiImageShellBackend,
    ParallelPlateBackend,
    PointCharge,
    SolidSphere,
    SphericalShell,
    UniformGravity,
    UnsupportedProblem,
    find_1d_equilibria,
    integrate_motion_1d,
    integrate_motion_2d,
    sample_motion_1d,
    sample_motion_2d,
)
from conductor_sim._motion_validation import validate_planar_motion
from conductor_sim.viewer import render_html


def test_overhead_grounded_plane_exact_unstable_equilibrium_and_divergence():
    # Overhead grounded conducting ceiling at z=0 with vacuum below (normal = (0, 0, -1)).
    # Image force pulls upward (+z); gravity g = (0, 0, -9.8) pulls downward (-z).
    q = 2e-6
    m = 1e-3
    g_mag = 9.8
    h_exact = abs(q) / (4.0 * sqrt(pi * EPSILON_0 * m * g_mag))
    k_exact = -(q * q) / (8.0 * pi * EPSILON_0 * h_exact**3)  # negative => unstable!

    plane = ConductingPlane(point_m=(0.0, 0.0, 0.0), normal=(0.0, 0.0, -1.0), constraint=Grounded())
    gravity = UniformGravity((0.0, 0.0, -g_mag), reference_position_m=(0.0, 0.0, 0.0))
    prob = ElectrostaticProblem((PointCharge(q, (0.0, 0.0, -h_exact)),), (plane,))
    spec = Motion1DSpec(prob, 0, m, (0.0, 0.0, -1.0), 0.0, 0.1, 0.01, gravity=gravity)

    eqs = find_1d_equilibria(
        spec, GroundedPlaneBackend(), displacement_bounds_m=(-0.5 * h_exact, 0.5 * h_exact),
    )
    assert len(eqs) == 1
    eq = eqs[0]
    assert isinstance(eq, ConductingGravityEquilibrium)
    assert isclose(-eq.position_m[2], h_exact, rel_tol=1e-12)
    assert abs(eq.displacement_m) < 1e-12
    assert abs(eq.net_force_n) < 1e-13 * m * g_mag
    assert eq.stability == "unstable"
    assert eq.small_oscillation_angular_frequency_rad_s is None
    assert isclose(eq.stiffness_n_per_m, k_exact, rel_tol=1e-6)
    assert isclose(eq.electric_force_n, -m * g_mag, rel_tol=1e-12)
    assert isclose(eq.gravitational_force_n, +m * g_mag, rel_tol=1e-12)

    # Nudge slightly closer to ceiling (h < h_exact) -> image force wins -> hits ceiling surface-guard!
    prob_up = ElectrostaticProblem((PointCharge(q, (0.0, 0.0, -0.95 * h_exact)),), (plane,))
    res_up = integrate_motion_1d(
        Motion1DSpec(
            prob_up, 0, m, (0.0, 0.0, -1.0), 0.0, 1.5, 0.01,
            relative_tolerance=1e-9, gravity=gravity, surface_guard_m=1e-2,
        ),
        GroundedPlaneBackend(),
    )
    assert res_up.termination == "surface-guard"
    assert res_up.samples[-1].position_m[2] > -0.95 * h_exact
    peak_energy = max(max(abs(s.kinetic_energy_j), abs(s.interaction_energy_j)) for s in res_up.samples)
    assert res_up.max_abs_energy_drift_j < 1e-7 * peak_energy

    # Nudge slightly farther from ceiling (h > h_exact) -> gravity wins -> accelerates downward (-z)!
    prob_down = ElectrostaticProblem((PointCharge(q, (0.0, 0.0, -1.05 * h_exact)),), (plane,))
    res_down = integrate_motion_1d(
        Motion1DSpec(prob_down, 0, m, (0.0, 0.0, -1.0), 0.0, 0.15, 0.005, gravity=gravity),
        GroundedPlaneBackend(),
    )
    assert res_down.termination == "duration"
    assert res_down.samples[-1].position_m[2] < -1.05 * h_exact
    assert res_down.samples[-1].velocity_m_s > 0.0  # moving along +axis = -z
    assert res_down.max_abs_energy_drift_j < 1e-11 * max(1e-12, abs(res_down.samples[0].total_energy_j))


def test_isolated_charged_sphere_two_equilibria_and_small_oscillation_period():
    # Solid isolated sphere R=0.10 m with Q=+5e-7 C; test particle q=+2e-7 C, m=5e-4 kg above it under g=-9.8 z.
    # Closer to sphere: attractive image force dominates -> inner UNSTABLE equilibrium.
    # Farther from sphere: repulsive monopole balances gravity -> outer STABLE equilibrium with small oscillations!
    R = 0.10
    Q = 5e-7
    q = 2e-7
    m = 5e-4
    g_mag = 9.8
    k_c = 1.0 / (4.0 * pi * EPSILON_0)

    def f_exact(r: float) -> float:
        # Exact force on q outside isolated sphere (radius R, net charge Q):
        # F_r = k_c * q * [ (Q + q*R/r)/r^2 - q*R*r/(r^2 - R^2)^2 ] - m*g
        return k_c * q * ((Q + q * R / r) / (r * r) - q * R * r / ((r * r - R * R) ** 2)) - m * g_mag

    def df_dr_exact(r: float) -> float:
        term1 = -(2.0 * Q + 3.0 * q * R / r) / (r**3)
        term2 = -q * R * ((r * r - R * R) - 4.0 * r * r) / ((r * r - R * R) ** 3)
        return k_c * q * (term1 + term2)

    sphere = SolidSphere(center_m=(0.0, 0.0, 0.0), radius_m=R, constraint=FixedNetCharge(Q))
    gravity = UniformGravity((0.0, 0.0, -g_mag), reference_position_m=(0.0, 0.0, 0.0))
    z0 = 0.30
    prob = ElectrostaticProblem((PointCharge(q, (0.0, 0.0, z0)),), (sphere,))
    spec = Motion1DSpec(prob, 0, m, (0.0, 0.0, 1.0), 0.0, 0.5, 0.01, gravity=gravity)

    eqs = find_1d_equilibria(
        spec, ExternalIsolatedSphereBackend(), displacement_bounds_m=(-0.195, 0.80), scan_samples=257,
    )
    assert len(eqs) == 2
    inner_eq, outer_eq = eqs
    assert inner_eq.stability == "unstable"
    assert inner_eq.small_oscillation_angular_frequency_rad_s is None
    assert outer_eq.stability == "stable"
    assert outer_eq.small_oscillation_angular_frequency_rad_s is not None

    r_in = inner_eq.position_m[2]
    r_out = outer_eq.position_m[2]
    assert abs(f_exact(r_in)) < 1e-12 * m * g_mag
    assert abs(f_exact(r_out)) < 1e-12 * m * g_mag
    assert isclose(inner_eq.stiffness_n_per_m, -df_dr_exact(r_in), rel_tol=1e-5)
    assert isclose(outer_eq.stiffness_n_per_m, -df_dr_exact(r_out), rel_tol=1e-5)

    # Small-amplitude 1D oscillation around the stable equilibrium matches T = 2*pi / omega!
    omega = outer_eq.small_oscillation_angular_frequency_rad_s
    period = 2.0 * pi / omega
    amp = 1e-4 * r_out
    osc_prob = ElectrostaticProblem((PointCharge(q, (0.0, 0.0, r_out + amp)),), (sphere,))
    osc_res = integrate_motion_1d(
        Motion1DSpec(
            osc_prob, 0, m, (0.0, 0.0, 1.0), 0.0, 1.2 * period, 0.01 * period,
            relative_tolerance=1e-9, gravity=gravity,
        ),
        ExternalIsolatedSphereBackend(),
    )
    assert osc_res.termination == "duration"
    turning_times = [ev.time_s for ev in osc_res.events if ev.kind == "turning-point"]
    assert len(turning_times) >= 2
    # First turning point is at T/2, second at T.
    assert isclose(turning_times[0], 0.5 * period, rel_tol=2e-3)
    assert isclose(turning_times[1], period, rel_tol=2e-3)
    assert osc_res.max_abs_energy_drift_j < 1e-12


def test_2d_grounded_plane_projectile_with_gravity_and_viewer_validation():
    # Charged particle launched horizontally above a grounded plane z=0 under downward gravity g=-9.8 z.
    # Both gravity and image force pull toward -z; horizontal velocity stays strictly constant!
    q = 3e-7
    m = 2e-3
    g_mag = 9.8
    vx0 = 2.5
    z0 = 0.50
    plane = ConductingPlane(point_m=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), constraint=Grounded())
    gravity = UniformGravity((0.0, 0.0, -g_mag), reference_position_m=(0.0, 0.0, 0.0))
    prob = ElectrostaticProblem((PointCharge(q, (0.0, 0.0, z0)),), (plane,))
    spec = Motion2DSpec(
        prob, 0, m, (vx0, 0.0, 0.0), 0.15, 0.015, relative_tolerance=1e-8, gravity=gravity,
    )
    res = integrate_motion_2d(spec, GroundedPlaneBackend())
    assert res.termination == "duration"
    assert res.max_abs_energy_drift_j < 1e-10
    for sample in res.samples:
        # No horizontal force -> exact vx == vx0 and exact x == vx0 * t!
        assert isclose(sample.velocity_m_s[0], vx0, rel_tol=1e-10)
        assert isclose(sample.position_m[0], vx0 * sample.time_s, rel_tol=1e-9, abs_tol=1e-11)
        assert sample.electric_force_n is not None
        assert sample.gravitational_force_n == (0.0, 0.0, -m * g_mag)
        assert isclose(sample.force_n[2], sample.electric_force_n[2] - m * g_mag, rel_tol=1e-12)
        assert isclose(
            sample.interaction_energy_j,
            sample.electric_interaction_energy_j + sample.gravitational_potential_energy_j,
            rel_tol=1e-12,
        )
    # Falls faster than pure free-fall z0 - 0.5*g*t^2 because attractive image force adds to gravity!
    t_end = res.samples[-1].time_s
    assert res.samples[-1].position_m[2] < z0 - 0.5 * g_mag * t_end * t_end

    # Viewer export & semantic validator accept gravity-coupled 2D trajectory cleanly!
    bundle = sample_motion_2d(spec, GroundedPlaneBackend(), result=res, visual_frames=5, resolution=11)
    assert "gravity" in bundle["motion_2d"]
    validate_planar_motion(bundle["motion_2d"], frames=bundle["frames"])

    # Forging gravitational force in exported JSON must be rejected by validate_planar_motion!
    forged = res.as_dict()
    forged["samples"][1]["gravitational_force_n"] = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="gravitational force"):
        validate_planar_motion(forged)


def test_2d_release_from_rest_with_tilted_gravity_selects_invariant_plane_and_parallel_plates_and_cavity():
    # 1. Grounded sphere + particle at (0.3, 0, 0) released from rest with gravity along -y selects xy-plane!
    sphere = SolidSphere(center_m=(0.0, 0.0, 0.0), radius_m=0.1, constraint=Grounded())
    g_y = UniformGravity((0.0, -9.8, 0.0))
    spec_sph = Motion2DSpec(
        ElectrostaticProblem((PointCharge(1e-7, (0.3, 0.0, 0.0)),), (sphere,)),
        0, 1e-3, (0.0, 0.0, 0.0), 0.05, 0.01, gravity=g_y,
    )
    res_sph = integrate_motion_2d(spec_sph, ExternalGroundedSphereBackend())
    assert abs(abs(res_sph.plane_normal[2]) - 1.0) < 1e-12
    assert res_sph.samples[-1].position_m[0] < 0.3  # pulled inward toward sphere
    assert res_sph.samples[-1].position_m[1] < 0.0  # pulled downward by gravity
    assert abs(res_sph.samples[-1].position_m[2]) < 1e-14
    assert res_sph.max_abs_energy_drift_j < 1e-10

    # 2. Parallel-plate capacitor + upward electric field vs downward gravity in 1D (with viewer check!)
    d = 0.04
    plates = (
        ConductingPlane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), Grounded(), name="bottom"),
        ConductingPlane((0.0, 0.0, d), (0.0, 0.0, -1.0), FixedPotential(-250000.0), name="top"),
    )
    prob_pp = ElectrostaticProblem((PointCharge(2e-9, (0.0, 0.0, 0.02)),), plates)
    spec_pp = Motion1DSpec(
        prob_pp, 0, 1e-3, (0.0, 0.0, 1.0), -0.05, 0.04, 0.004,
        gravity=UniformGravity((0.0, 0.0, -9.8)),
    )
    res_pp = integrate_motion_1d(spec_pp, ParallelPlateBackend())
    assert res_pp.termination == "duration"
    # Upward E = 62500 V/m -> qE = 0.0125 N overcomes mg (0.0098 N) + image attraction -> reverses downward launch!
    assert any(ev.kind == "turning-point" for ev in res_pp.events)
    assert res_pp.max_abs_energy_drift_j < 1e-9
    bundle_1d = sample_motion_1d(spec_pp, ParallelPlateBackend(), result=res_pp, visual_frames=4, resolution=11)
    assert "gravity" in bundle_1d["motion_1d"]

    # 3. Spherical cavity with held charge + moving charge + coplanar gravity in 2D
    shell = SphericalShell(inner_radius_m=0.20, outer_radius_m=0.25, constraint=Grounded(), center_m=(0.0, 0.0, 0.0))
    prob_cav = ElectrostaticProblem(
        (PointCharge(1e-7, (0.05, 0.0, 0.0)), PointCharge(1e-7, (-0.05, 0.0, 0.0))),
        (shell,),
    )
    spec_cav = Motion2DSpec(
        prob_cav, 0, 1e-3, (0.0, 0.2, 0.0), 0.03, 0.006,
        gravity=UniformGravity((0.0, -9.8, 0.0)),
    )
    res_cav = integrate_motion_2d(spec_cav, MultiImageShellBackend())
    assert res_cav.termination == "duration"
    assert res_cav.max_abs_energy_drift_j < 1e-9


def test_gauge_invariance_and_zero_gravity_backward_compatibility():
    plane = ConductingPlane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), Grounded())
    prob = ElectrostaticProblem((PointCharge(2e-7, (0.0, 0.0, 0.5)),), (plane,))

    # Large representable gravitational reference offset (1e6 m; |U_g| ~ 1e4 J vs ~5e-6 J fall)
    # must not degrade energy_drift_j while preserving binary64 U_g precision!
    g_ref0 = UniformGravity((0.0, 0.0, -9.8), reference_position_m=(0.0, 0.0, 0.0))
    g_ref_big = UniformGravity((0.0, 0.0, -9.8), reference_position_m=(0.0, 0.0, 1e6))
    r0 = integrate_motion_1d(Motion1DSpec(prob, 0, 1e-3, (0.0, 0.0, 1.0), 0.0, 0.05, 0.01, gravity=g_ref0), GroundedPlaneBackend())
    rb = integrate_motion_1d(Motion1DSpec(prob, 0, 1e-3, (0.0, 0.0, 1.0), 0.0, 0.05, 0.01, gravity=g_ref_big), GroundedPlaneBackend())
    assert r0.max_abs_energy_drift_j < 1e-12
    assert rb.max_abs_energy_drift_j < 1e-12
    assert isclose(r0.samples[-1].position_m[2], rb.samples[-1].position_m[2], rel_tol=1e-14)

    # Zero gravity matches gravity=None trajectory; gravity=None omits gravity keys in as_dict()
    r_none = integrate_motion_1d(Motion1DSpec(prob, 0, 1e-3, (0.0, 0.0, 1.0), 0.0, 0.05, 0.01), GroundedPlaneBackend())
    r_zero = integrate_motion_1d(
        Motion1DSpec(prob, 0, 1e-3, (0.0, 0.0, 1.0), 0.0, 0.05, 0.01, gravity=UniformGravity((0.0, 0.0, 0.0))),
        GroundedPlaneBackend(),
    )
    assert isclose(r_none.samples[-1].position_m[2], r_zero.samples[-1].position_m[2], rel_tol=1e-14)
    d_none = r_none.as_dict()
    assert "gravity" not in d_none
    assert "electric_force_n" not in d_none["samples"][0]
    d_zero = r_zero.as_dict()
    assert "gravity" in d_zero
    assert d_zero["samples"][0]["gravitational_force_n"] == 0.0


def test_symmetry_and_contract_refusals():
    plane = ConductingPlane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), Grounded())
    prob_p = ElectrostaticProblem((PointCharge(1e-7, (0.0, 0.0, 0.5)),), (plane,))

    # Non-UniformGravity type refused
    with pytest.raises(TypeError, match="UniformGravity"):
        Motion1DSpec(prob_p, 0, 1e-3, (0.0, 0.0, 1.0), 0.0, 0.1, 0.01, gravity=(0.0, 0.0, -9.8))
    with pytest.raises(TypeError, match="UniformGravity"):
        Motion2DSpec(prob_p, 0, 1e-3, (1.0, 0.0, 0.0), 0.1, 0.01, gravity=(0.0, 0.0, -9.8))

    # Transverse gravity breaks 1D symmetry -> refused
    with pytest.raises(UnsupportedProblem, match="collinear"):
        Motion1DSpec(prob_p, 0, 1e-3, (0.0, 0.0, 1.0), 0.0, 0.1, 0.01, gravity=UniformGravity((1.0, 0.0, -9.8)))

    # Out-of-plane gravity breaks 2D symmetry (plane + vx0 fixes xz plane; g_y is out of plane!) -> refused
    with pytest.raises(UnsupportedProblem, match="invariant plane"):
        Motion2DSpec(prob_p, 0, 1e-3, (1.0, 0.0, 0.0), 0.1, 0.01, gravity=UniformGravity((0.0, -9.8, 0.0)))

    # Out-of-plane gravity with sphere + vz0 (fixes xz plane; g_y is out of plane!) -> refused
    sphere = SolidSphere(radius_m=0.1, constraint=Grounded(), center_m=(0.0, 0.0, 0.0))
    prob_s = ElectrostaticProblem((PointCharge(1e-7, (0.3, 0.0, 0.0)),), (sphere,))
    with pytest.raises(UnsupportedProblem, match="invariant plane"):
        Motion2DSpec(prob_s, 0, 1e-3, (0.0, 0.0, 1.0), 0.1, 0.01, gravity=UniformGravity((0.0, -9.8, 0.0)))

    # Out-of-plane gravity with held cavity charge (xy plane; g_z is out of plane!) -> refused
    shell = SphericalShell(inner_radius_m=0.2, outer_radius_m=0.25, constraint=Grounded(), center_m=(0.0, 0.0, 0.0))
    prob_c = ElectrostaticProblem((PointCharge(1e-7, (0.05, 0.0, 0.0)), PointCharge(1e-7, (0.0, 0.05, 0.0))), (shell,))
    with pytest.raises(UnsupportedProblem, match="invariant plane"):
        Motion2DSpec(prob_c, 0, 1e-3, (0.0, 0.0, 0.0), 0.1, 0.01, gravity=UniformGravity((0.0, 0.0, -9.8)))

    # Unbounded 1D domain without explicit displacement_bounds_m -> refused
    spec_unbounded = Motion1DSpec(prob_p, 0, 1e-3, (0.0, 0.0, 1.0), 0.0, 0.1, 0.01, gravity=UniformGravity((0.0, 0.0, -9.8)))
    with pytest.raises(UnsupportedProblem, match="displacement_bounds_m"):
        find_1d_equilibria(spec_unbounded, GroundedPlaneBackend())

    # Bounds crossing surface guard -> refused
    with pytest.raises(UnsupportedProblem, match="safe vacuum domain"):
        find_1d_equilibria(spec_unbounded, GroundedPlaneBackend(), displacement_bounds_m=(-0.5, 0.5))

    # Overflowing int bound raises ValueError (never raw OverflowError)
    with pytest.raises(ValueError, match="finite"):
        find_1d_equilibria(spec_unbounded, GroundedPlaneBackend(), displacement_bounds_m=(0, 10**500))


def test_sol_m45_review_regressions():
    # 1. Exact 3-term total_energy_j agreement between integrator and validator (no HTML export refusal)
    plane = ConductingPlane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), Grounded())
    prob_e = ElectrostaticProblem((PointCharge(1e-7, (0.0, 0.0, 0.5)),), (plane,))
    u_elec0 = -(1e-7) ** 2 / (16.0 * pi * EPSILON_0 * 0.5)
    m_cancel = -2.0 * u_elec0
    spec_e = Motion2DSpec(
        prob_e, 0, m_cancel, (1.0, 0.0, 0.0), 0.01, 0.005,
        gravity=UniformGravity((1.0, 0.0, 0.0), reference_position_m=(1e-18, 0.0, 0.0)),
    )
    res_e = integrate_motion_2d(spec_e, GroundedPlaneBackend())
    bundle_e = sample_motion_2d(spec_e, GroundedPlaneBackend(), result=res_e, visual_frames=3, resolution=9)
    assert "gravity" in bundle_e["motion_2d"]

    # 2. Close pair of roots inside a single uniform grid cell is found by extremum refinement!
    sphere = SolidSphere(center_m=(0.0, 0.0, 0.0), radius_m=0.1, constraint=FixedNetCharge(5e-7))
    spec_close = Motion1DSpec(
        ElectrostaticProblem((PointCharge(2e-7, (0.0, 0.0, 0.3)),), (sphere,)),
        0, 5e-4, (0.0, 0.0, 1.0), 0.0, 0.1, 0.01,
        gravity=UniformGravity((0.0, 0.0, -42.646393506555945)),
    )
    eqs_close = find_1d_equilibria(spec_close, ExternalIsolatedSphereBackend(), displacement_bounds_m=(-0.19, 0.5))
    assert len(eqs_close) == 2
    assert isclose(eqs_close[0].position_m[2], 0.1716488813521, rel_tol=1e-7)
    assert isclose(eqs_close[1].position_m[2], 0.1716611949418, rel_tol=1e-7)
    assert eqs_close[0].stability == "unstable" and eqs_close[1].stability == "stable"

    # 3. Micro-scale stiffness accuracy (z = 2e-6 m, mg = 0.001 N -> k = -1000 N/m) + extreme omega scales
    z_micro = 2e-6
    mg_micro = 1e-3
    q_micro = 4.0 * z_micro * sqrt(pi * EPSILON_0 * mg_micro)
    ceil_plane = ConductingPlane((0.0, 0.0, 0.0), (0.0, 0.0, -1.0), Grounded())
    spec_micro = Motion1DSpec(
        ElectrostaticProblem((PointCharge(q_micro, (0.0, 0.0, -z_micro)),), (ceil_plane,)),
        0, 1e-4, (0.0, 0.0, -1.0), 0.0, 0.01, 0.005,
        surface_guard_m=1e-8, gravity=UniformGravity((0.0, 0.0, -10.0)),
    )
    eq_micro = find_1d_equilibria(
        spec_micro, GroundedPlaneBackend(), displacement_bounds_m=(-1e-6, 1e-6),
    )[0]
    assert isclose(eq_micro.stiffness_n_per_m, -1000.0, rel_tol=1e-6)

    # Extreme omega via scaled sqrt(k/m) without intermediate overflow/underflow
    from conductor_sim.conductor_gravity import _scaled_sqrt_ratio
    assert isclose(_scaled_sqrt_ratio(1e8, 1e-300), 1e154, rel_tol=1e-12)
    assert isclose(_scaled_sqrt_ratio(1e-300, 1e26), 1e-163, rel_tol=1e-12)

    # 4. Tiny out-of-plane gravity (0, 5e-13, 0) is rejected at Motion2DSpec construction!
    with pytest.raises(UnsupportedProblem, match="invariant plane"):
        Motion2DSpec(prob_e, 0, 1e-3, (1.0, 0.0, 0.0), 0.1, 0.01, gravity=UniformGravity((0.0, 5e-13, 0.0)))

    # 5. Returning 1D trajectory checks ALL rows (including graph-only rows) & checks backend frames
    spec_ret = Motion1DSpec(
        ElectrostaticProblem((PointCharge(1e-17, (0.0, 0.0, 1.0)),), (plane,)),
        0, 1.0, (0.0, 0.0, 1.0), -1.0, 1.0, 0.05,
        gravity=UniformGravity((0.0, 0.0, 2.0)),
    )
    res_ret = integrate_motion_1d(spec_ret, GroundedPlaneBackend())
    assert len(res_ret.samples) == 21
    from dataclasses import replace as dc_replace
    # Forged selected frame row with inconsistent sum -> caught by 1D gravity ledger validator
    forged_samples = (dc_replace(res_ret.samples[0], electric_force_n=99.0), *res_ret.samples[1:])
    with pytest.raises(ValueError, match="total force"):
        sample_motion_1d(spec_ret, GroundedPlaneBackend(), result=dc_replace(res_ret, samples=forged_samples), visual_frames=5, resolution=9)
    # Forged selected frame row with self-consistent sum -> caught by _CheckedBackend against real solver
    s0 = res_ret.samples[0]
    forged_consistent = (
        dc_replace(s0, electric_force_n=99.0, force_n=99.0 + s0.gravitational_force_n,
                   acceleration_m_s2=(99.0 + s0.gravitational_force_n) / spec_ret.mass_kg),
        *res_ret.samples[1:],
    )
    with pytest.raises(UnsupportedProblem, match="disagrees"):
        sample_motion_1d(spec_ret, GroundedPlaneBackend(), result=dc_replace(res_ret, samples=forged_consistent), visual_frames=5, resolution=9)
    # Forged graph-only row (index 1, skipped when visual_frames=5) -> caught by sample_motion_1d
    forged_graph = list(res_ret.samples)
    forged_graph[1] = dc_replace(forged_graph[1], gravitational_potential_energy_j=1234.0)
    with pytest.raises(ValueError, match="gravitational energy"):
        sample_motion_1d(spec_ret, GroundedPlaneBackend(), result=dc_replace(res_ret, samples=tuple(forged_graph)), visual_frames=5, resolution=9)
    # Post-sampling forged 1D row -> caught by render_html
    valid_bundle_1d = sample_motion_1d(spec_ret, GroundedPlaneBackend(), result=res_ret, visual_frames=5, resolution=9)
    assert "<canvas" in render_html(valid_bundle_1d, title="Valid 1D gravity")
    forged_bundle_1d = sample_motion_1d(spec_ret, GroundedPlaneBackend(), result=res_ret, visual_frames=5, resolution=9)
    forged_bundle_1d["motion_1d"]["samples"][1]["electric_force_n"] = 42.0
    with pytest.raises(ValueError, match="total force"):
        render_html(forged_bundle_1d, title="Forged 1D gravity")
    # Orphan gravity fields without gravity metadata -> caught by render_html
    orphan_bundle_1d = sample_motion_1d(spec_ret, GroundedPlaneBackend(), result=res_ret, visual_frames=5, resolution=9)
    del orphan_bundle_1d["motion_1d"]["gravity"]
    with pytest.raises(ValueError, match="gravity metadata"):
        render_html(orphan_bundle_1d, title="Orphan 1D gravity")

    # 2D validator rejects gravity contribution keys without gravity metadata and huge forged electric_force_n
    d_no_g = res_e.as_dict()
    del d_no_g["gravity"]
    with pytest.raises(ValueError, match="gravity metadata"):
        validate_planar_motion(d_no_g)
    d_huge = res_e.as_dict()
    d_huge["samples"][0]["electric_force_n"] = [1.5e308, 1.5e308, 1.5e308]
    d_huge["samples"][0]["gravitational_force_n"] = [999.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="gravitational force"):
        validate_planar_motion(d_huge)

    # 6. Closeable solutions are closed on EVERY find_1d_equilibria probe (0 leaks)
    counts = {"open": 0, "close": 0}
    base_backend = GroundedPlaneBackend()

    from conductor_sim import force_energy as _fe

    class _CountingBackend:
        def supports(self, p):
            return base_backend.supports(p)

        def solve(self, p):
            sol = base_backend.solve(p)
            counts["open"] += 1

            class _Wrap:
                def force_energy(self):
                    return _fe(sol)

                def close(self):
                    counts["close"] += 1

            return _Wrap()

    find_1d_equilibria(spec_micro, _CountingBackend(), displacement_bounds_m=(-1e-6, 1e-6), scan_samples=25)
    assert counts["open"] > 0 and counts["open"] == counts["close"]

    # 7. UniformGravity.potential_energy_at avoids intermediate underflow AND OverflowError
    g_ext = UniformGravity((1e-200, 0.0, 0.0), reference_position_m=(0.0, 0.0, 0.0))
    assert isclose(g_ext.potential_energy_at((1e-200, 0.0, 0.0), mass_kg=1e200), -1e-200, rel_tol=1e-14)
    g_ovf = UniformGravity((1.0, 1.0, 0.0), reference_position_m=(0.0, 0.0, 0.0))
    assert isclose(g_ovf.potential_energy_at((1e308, 1e308, 0.0), mass_kg=1e-308), -2.0, rel_tol=1e-14)


def test_m45_sol_v2_boundary_cell_false_root_tangent_and_subnormal():
    """Regression suite for Sol v2 findings: boundary-cell close roots, no false roots near
    singular surface, scan-invariant tangent stability, and subnormal out-of-plane gravity."""
    # 1. Exact first-cell close roots on [0.17164, 0.18] m outside an isolated charged sphere
    R = 0.10
    Q_net = 5.0e-7
    q = 2.0e-7
    g_mag = 9.81
    sphere = SolidSphere(center_m=(0.0, 0.0, 0.0), radius_m=R, constraint=FixedNetCharge(Q_net))
    backend = ExternalIsolatedSphereBackend()

    # Exact radial electric force for isolated sphere with net charge Q_net and external q at r:
    # q_img = -q * R / r at b = R^2 / r; q_center = Q_net - q_img = Q_net + q * R / r
    def f_elec(r: float) -> float:
        q_img = -q * R / r
        b = R * R / r
        q_c = Q_net - q_img
        return (q / (4.0 * pi * EPSILON_0)) * (q_c / (r * r) + q_img / ((r - b) ** 2))

    # Locate the exact peak of f_elec(r) on (0.12, 0.30)
    lo, hi = 0.12, 0.30
    inv_phi = (sqrt(5.0) - 1.0) / 2.0
    for _ in range(100):
        c = hi - inv_phi * (hi - lo)
        d = lo + inv_phi * (hi - lo)
        if f_elec(c) >= f_elec(d):
            hi = d
        else:
            lo = c
    r_peak = 0.5 * (lo + hi)
    f_peak = f_elec(r_peak)

    # (a) Sol's exact first-cell close roots on [0.17164, 0.18] m (both fall in [grid[0], grid[1]])
    spec_sol_cell = Motion1DSpec(
        ElectrostaticProblem((PointCharge(q, (0.0, 0.0, 0.3)),), (sphere,)),
        0, 5e-4, (0.0, 0.0, 1.0), 0.0, 0.1, 0.01,
        gravity=UniformGravity((0.0, 0.0, -42.646393506555945)),
    )
    eq_boundary = find_1d_equilibria(
        spec_sol_cell, backend, displacement_bounds_m=(0.17164 - 0.3, 0.18 - 0.3), scan_samples=129,
    )
    assert len(eq_boundary) == 2
    assert isclose(eq_boundary[0].position_m[2], 0.1716488813521, rel_tol=1e-7)
    assert isclose(eq_boundary[1].position_m[2], 0.1716611949418, rel_tol=1e-7)
    assert eq_boundary[0].stability == "unstable"
    assert eq_boundary[1].stability == "stable"
    # Also verify right-boundary cell [grid[-2], grid[-1]] catches both roots
    eq_right_boundary = find_1d_equilibria(
        spec_sol_cell, backend, displacement_bounds_m=(0.16 - 0.3, 0.17167 - 0.3), scan_samples=129,
    )
    assert len(eq_right_boundary) == 2
    assert eq_right_boundary[0].stability == "unstable"
    assert eq_right_boundary[1].stability == "stable"

    # (b) No false equilibrium when max net force is strictly negative (-1.0e-7 N) even near singular r=0.1000002
    m_none = (f_peak + 1.0e-7) / g_mag
    spec_none = Motion1DSpec(
        ElectrostaticProblem((PointCharge(q, (0.0, 0.0, 0.25)),), (sphere,)),
        0, m_none, (0.0, 0.0, 1.0), 0.0, 0.05, 0.01,
        surface_guard_m=1e-7,
        gravity=UniformGravity((0.0, 0.0, -g_mag)),
    )
    eq_none = find_1d_equilibria(
        spec_none, backend,
        displacement_bounds_m=(0.1000002 - 0.25, 0.7 - 0.25),
        scan_samples=9,
    )
    assert eq_none == ()

    # (c) Exact off-grid quadratic tangent (F_peak == mg) is ALWAYS indeterminate across scan_samples
    m_tan = f_peak / g_mag
    spec_tan = Motion1DSpec(
        ElectrostaticProblem((PointCharge(q, (0.0, 0.0, 0.25)),), (sphere,)),
        0, m_tan, (0.0, 0.0, 1.0), 0.0, 0.05, 0.01,
        gravity=UniformGravity((0.0, 0.0, -g_mag)),
    )
    for n_scan in (17, 33, 65, 129, 257):
        eq_tan = find_1d_equilibria(
            spec_tan, backend,
            displacement_bounds_m=(0.13 - 0.25, 0.35 - 0.25),
            scan_samples=n_scan,
        )
        assert len(eq_tan) == 1
        assert eq_tan[0].stability == "indeterminate"
        assert eq_tan[0].small_oscillation_angular_frequency_rad_s is None

    # 2. Subnormal out-of-plane gravity (5e-324, 0, 0) cannot bypass 2D plane checks!
    plane_xy = ConductingPlane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), Grounded())
    v_tilted = (0.916515138991168, -0.4, 0.0)  # plane normal = (0.4, 0.916515..., 0)
    prob_sub = ElectrostaticProblem((PointCharge(1e-9, (0.0, 0.0, 0.5)),), (plane_xy,))
    with pytest.raises(UnsupportedProblem, match="invariant plane"):
        Motion2DSpec(
            prob_sub, 0, 1.0, v_tilted, 0.05, 0.01,
            gravity=UniformGravity((5e-324, 0.0, 0.0)),
        )
    # Also verify validate_planar_motion rejects subnormal out-of-plane gravity in saved dicts
    spec_ok_2d = Motion2DSpec(
        prob_sub, 0, 1.0, v_tilted, 0.02, 0.01,
        gravity=UniformGravity((0.0, 0.0, -1.0)),
    )
    res_ok_2d = integrate_motion_2d(spec_ok_2d, GroundedPlaneBackend())
    d_sub = res_ok_2d.as_dict()
    d_sub["gravity"]["acceleration_m_s2"] = [5e-324, 0.0, 0.0]
    with pytest.raises(ValueError, match="outside the proven physical plane"):
        validate_planar_motion(d_sub)


def test_m45_sol_v3_exact_boundary_root_grid_aligned_tangent_opposing_overflow_and_inf_scale():
    """Regression suite for all 4 Sol v3 findings."""
    R = 0.10
    Q_net = 5.0e-7
    q = 2.0e-7
    sphere = SolidSphere(center_m=(0.0, 0.0, 0.0), radius_m=R, constraint=FixedNetCharge(Q_net))
    backend = ExternalIsolatedSphereBackend()

    # 1. Exact root on scan boundary z=0.17164 must NOT hide the second stable root inside [grid[0], grid[1]]
    spec_b = Motion1DSpec(
        ElectrostaticProblem((PointCharge(q, (0.0, 0.0, 0.17164)),), (sphere,)),
        0, 0.02132319569394952, (0.0, 0.0, 1.0), 0.0, 0.1, 0.01,
        gravity=UniformGravity((0.0, 0.0, -1.0)),
    )
    eq_b = find_1d_equilibria(
        spec_b, backend, displacement_bounds_m=(0.0, 0.18 - 0.17164), scan_samples=129,
    )
    assert len(eq_b) == 2
    assert isclose(eq_b[0].position_m[2], 0.17164, rel_tol=1e-9)
    assert isclose(eq_b[1].position_m[2], 0.17167008137887, rel_tol=1e-7)
    assert eq_b[0].stability == "unstable"
    assert eq_b[1].stability == "stable"

    # 2. Grid-aligned tangent (9 samples centered exactly on r_peak) returns 1 indeterminate root
    r_tan = 0.17165503777020558
    spec_tan_aligned = Motion1DSpec(
        ElectrostaticProblem((PointCharge(q, (0.0, 0.0, 0.25)),), (sphere,)),
        0, 0.002173618447146783, (0.0, 0.0, 1.0), 0.0, 0.05, 0.01,
        gravity=UniformGravity((0.0, 0.0, -9.81)),
    )
    eq_aligned = find_1d_equilibria(
        spec_tan_aligned, backend,
        displacement_bounds_m=(r_tan - 0.002 - 0.25, r_tan + 0.002 - 0.25),
        scan_samples=9,
    )
    assert len(eq_aligned) == 1
    assert eq_aligned[0].stability == "indeterminate"
    assert eq_aligned[0].small_oscillation_angular_frequency_rad_s is None

    # 3. Opposing overflow (+inf and -inf products) falls cleanly into exact Fraction fallback -> -3.0 J
    g_cancel_ovf = UniformGravity((2.0, 2.0, 1.0))
    assert isclose(
        g_cancel_ovf.potential_energy_at((1e308, -1e308, 3.0), mass_kg=1.0),
        -3.0,
        rel_tol=1e-14,
    )

    # 4. Off-axis graph-only position whose distance overflows to inf is rejected by render_html
    plane = ConductingPlane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), Grounded())
    spec_1d = Motion1DSpec(
        ElectrostaticProblem((PointCharge(1e-17, (0.0, 0.0, 1.0)),), (plane,)),
        0, 1.0, (0.0, 0.0, 1.0), 0.0, 0.1, 0.025,
        gravity=UniformGravity((0.0, 0.0, 2.0)),
    )
    bundle_1d = sample_motion_1d(spec_1d, GroundedPlaneBackend(), visual_frames=2, resolution=9)
    orig_z = bundle_1d["motion_1d"]["samples"][1]["position_m"][2]
    bundle_1d["motion_1d"]["samples"][1]["position_m"] = [1.3e308, 1.3e308, orig_z]
    with pytest.raises(ValueError, match="finite distance scale|disagrees"):
        render_html(bundle_1d, title="Forged overflowing 1D graph position")


def test_m45_sol_v4_exact_tangent_grid_zero_subnormal_bracket_translated_axis_and_cancellation():
    """Regression suite for all 4 Sol v4 findings."""
    from math import ulp

    # 1. Exact grid zero at the tangent peak must return 1 indeterminate root, never split into 3
    sphere = SolidSphere(center_m=(0.0, 0.0, 0.0), radius_m=0.10, constraint=FixedNetCharge(5.0e-7))
    spec_t0 = Motion1DSpec(
        ElectrostaticProblem((PointCharge(2.0e-7, (0.0, 0.0, 0.1716550374351809)),), (sphere,)),
        0, 0.021323196966509944, (0.0, 0.0, 1.0), 0.0, 0.05, 0.01,
        gravity=UniformGravity((0.0, 0.0, -1.0)),
    )
    eq_t0 = find_1d_equilibria(
        spec_t0, ExternalIsolatedSphereBackend(),
        displacement_bounds_m=(-0.005, 0.005), scan_samples=9,
    )
    assert len(eq_t0) == 1
    assert eq_t0[0].stability == "indeterminate"
    assert eq_t0[0].small_oscillation_angular_frequency_rad_s is None

    # 2. Subnormal forces (42*ulp(0.0)) cannot be resolved accurately in binary64:
    #    both (-0.2, 0.2) and the subnormal plateau (0.036, 0.038) must honestly refuse,
    #    while tiny normal-scale forces (~1e-305 N) still bracket cleanly.
    plane = ConductingPlane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), Grounded())
    spec_sub = Motion1DSpec(
        ElectrostaticProblem(
            (PointCharge(3e-161, (0.0, 0.0, 1.0)),),
            (plane,),
            permittivity_f_per_m=1.0 / (4.0 * pi),
        ),
        0, 1.0, (0.0, 0.0, 1.0), 0.0, 0.1, 0.05,
        gravity=UniformGravity((0.0, 0.0, 42.0 * ulp(0.0))),
    )
    with pytest.raises(UnsupportedProblem, match="subnormal"):
        find_1d_equilibria(spec_sub, GroundedPlaneBackend(), displacement_bounds_m=(-0.2, 0.2), scan_samples=9)
    with pytest.raises(UnsupportedProblem, match="subnormal"):
        find_1d_equilibria(spec_sub, GroundedPlaneBackend(), displacement_bounds_m=(0.036, 0.038), scan_samples=9)

    spec_tiny_normal = Motion1DSpec(
        ElectrostaticProblem(
            (PointCharge(2e-152, (0.0, 0.0, 1.0)),),
            (plane,),
            permittivity_f_per_m=1.0 / (4.0 * pi),
        ),
        0, 1.0, (0.0, 0.0, 1.0), 0.0, 0.1, 0.05,
        gravity=UniformGravity((0.0, 0.0, 1.0e-304)),
    )
    # Image force = -(2e-152)^2 / (4 z^2) = -1e-304 / z^2 -> exact equilibrium at z = 1.0 m!
    eq_tiny = find_1d_equilibria(
        spec_tiny_normal, GroundedPlaneBackend(), displacement_bounds_m=(-0.2, 0.2), scan_samples=9,
    )
    assert len(eq_tiny) == 1
    assert isclose(eq_tiny[0].position_m[2], 1.0, rel_tol=1e-12)

    # 3. Translated 1D bundle starting at (1e16, 0, 1) must reject BOTH y=1 AND x+=16/32 off-axis forgeries
    spec_tr = Motion1DSpec(
        ElectrostaticProblem((PointCharge(1e-17, (1e16, 0.0, 1.0)),), (plane,)),
        0, 1.0, (0.0, 0.0, 1.0), 0.0, 0.1, 0.025,
        gravity=UniformGravity((0.0, 0.0, 2.0)),
    )
    bundle_tr = sample_motion_1d(spec_tr, GroundedPlaneBackend(), visual_frames=2, resolution=9)
    assert "<canvas" in render_html(bundle_tr, title="Valid translated 1D gravity")
    orig_p = list(bundle_tr["motion_1d"]["samples"][1]["position_m"])
    bundle_tr["motion_1d"]["samples"][1]["position_m"] = [orig_p[0], 1.0, orig_p[2]]
    with pytest.raises(ValueError, match="axis displacement"):
        render_html(bundle_tr, title="Forged translated 1D off-axis y")
    bundle_tr["motion_1d"]["samples"][1]["position_m"] = [orig_p[0] + 16.0, orig_p[1], orig_p[2]]
    with pytest.raises(ValueError, match="axis displacement"):
        render_html(bundle_tr, title="Forged translated 1D off-axis x+16")
    bundle_tr["motion_1d"]["samples"][1]["position_m"] = [orig_p[0] + 32.0, orig_p[1], orig_p[2]]
    with pytest.raises(ValueError, match="axis displacement"):
        render_html(bundle_tr, title="Forged translated 1D off-axis x+32")

    # 4. Exact rational UniformGravity.potential_energy_at avoids catastrophic cancellation
    g_cancel = UniformGravity((1.0, 1.0, 1.0), reference_position_m=(1.0, 0.0, 0.0))
    assert isclose(g_cancel.potential_energy_at((1e16, -1e16, 0.5), mass_kg=1.0), 0.5, rel_tol=1e-14)


def test_m45_sol_v5_wide_scan_dedup_and_world_coordinate_precision():
    """Regression suite for Sol v5 findings: wide scan (1e6 m) preserving close roots,
    and refusing world coordinates along the active axis that lose gravitational energy precision."""
    # 1. Wide upper bound (1e6 m) must NOT merge two roots 1.23e-5 m apart
    sphere = SolidSphere(center_m=(0.0, 0.0, 0.0), radius_m=0.1, constraint=FixedNetCharge(5e-7))
    spec_close = Motion1DSpec(
        ElectrostaticProblem((PointCharge(2e-7, (0.0, 0.0, 0.3)),), (sphere,)),
        0, 5e-4, (0.0, 0.0, 1.0), 0.0, 0.1, 0.01,
        gravity=UniformGravity((0.0, 0.0, -42.646393506555945)),
    )
    eq_wide = find_1d_equilibria(
        spec_close, ExternalIsolatedSphereBackend(),
        displacement_bounds_m=(-0.19, 1e6), scan_samples=129,
    )
    assert len(eq_wide) == 2
    assert isclose(eq_wide[0].position_m[2], 0.1716488813521, rel_tol=1e-7)
    assert isclose(eq_wide[1].position_m[2], 0.1716611949418, rel_tol=1e-7)
    assert eq_wide[0].stability == "unstable"
    assert eq_wide[1].stability == "stable"

    # 2. World coordinates at z=1e16+100 m along +z lack binary64 precision for 1 m displacement:
    #    must never report zero drift while saved total energy jumps by 2 J!
    plane_far = ConductingPlane((0.0, 0.0, 1e16), (0.0, 0.0, 1.0), Grounded())
    p_far = (0.0, 0.0, 1e16 + 100.0)
    with pytest.raises(UnsupportedProblem, match="binary64 precision"):
        Motion1DSpec(
            ElectrostaticProblem((PointCharge(1e-17, p_far),), (plane_far,)),
            0, 1.0, (0.0, 0.0, 1.0), 0.0, 1.0, 0.5,
            gravity=UniformGravity((0.0, 0.0, 2.0), reference_position_m=p_far),
        )
    # 2D integration at z=1e16+100 m must also refuse when world position roundoff loses Ug resolution
    from conductor_sim.motion_1d import MotionIntegrationError
    spec_2d_far = Motion2DSpec(
        ElectrostaticProblem((PointCharge(1e-17, p_far),), (plane_far,)),
        0, 1.0, (0.0, 0.0, 0.0), 1.0, 0.5,
        gravity=UniformGravity((0.0, 0.0, 2.0), reference_position_m=p_far),
    )
    with pytest.raises(MotionIntegrationError, match="displacement resolution|gravitational potential energy"):
        integrate_motion_2d(spec_2d_far, GroundedPlaneBackend())


def test_m45_sol_v6_huge_bracket_stalled_position_remote_gauge_and_drift_validation():
    """Regression suite for all Sol v6 blocking and ledger findings."""
    from conductor_sim.motion_1d import MotionIntegrationError

    # 1a. Huge bracket (-0.5, 1e15) converges to exact z=2 m with ~0 residual (not 15.12 m!)
    plane0 = ConductingPlane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), Grounded())
    spec_big = Motion1DSpec(
        ElectrostaticProblem(
            (PointCharge(2.0, (0.0, 0.0, 1.0)),),
            (plane0,),
            permittivity_f_per_m=1.0 / (4.0 * pi),
        ),
        0, 1.0, (0.0, 0.0, 1.0), 0.0, 0.1, 0.05,
        gravity=UniformGravity((0.0, 0.0, 0.25)),
    )
    eq_big = find_1d_equilibria(
        spec_big, GroundedPlaneBackend(), displacement_bounds_m=(-0.5, 1e15), scan_samples=129,
    )
    assert len(eq_big) == 1
    assert isclose(eq_big[0].position_m[2], 2.0, rel_tol=1e-12)
    assert abs(eq_big[0].net_force_n) < 1e-12

    # 1b. Isolated-sphere pair STILL found when upper bound is widened all the way to 1e18 m
    sphere = SolidSphere(center_m=(0.0, 0.0, 0.0), radius_m=0.1, constraint=FixedNetCharge(5e-7))
    spec_close = Motion1DSpec(
        ElectrostaticProblem((PointCharge(2e-7, (0.0, 0.0, 0.3)),), (sphere,)),
        0, 5e-4, (0.0, 0.0, 1.0), 0.0, 0.1, 0.01,
        gravity=UniformGravity((0.0, 0.0, -42.646393506555945)),
    )
    eq_1e18 = find_1d_equilibria(
        spec_close, ExternalIsolatedSphereBackend(),
        displacement_bounds_m=(-0.19, 1e18), scan_samples=129,
    )
    assert len(eq_1e18) == 2
    assert isclose(eq_1e18[0].position_m[2], 0.1716488813521, rel_tol=1e-7)
    assert isclose(eq_1e18[1].position_m[2], 0.1716611949418, rel_tol=1e-7)

    # 2a. Grounded plane at z=0 with p0=(0,0,1e16+100) stalls world position -> refused in BOTH 1D & 2D
    p_far = (0.0, 0.0, 1e16 + 100.0)
    prob_stall = ElectrostaticProblem((PointCharge(1e-17, p_far),), (plane0,))
    spec_1d_stall = Motion1DSpec(
        prob_stall, 0, 1.0, (0.0, 0.0, 1.0), 0.0, 1.0, 0.5,
        gravity=UniformGravity((0.0, 0.0, 2.0), reference_position_m=p_far),
    )
    with pytest.raises(MotionIntegrationError, match="displacement resolution|gravitational potential energy"):
        integrate_motion_1d(spec_1d_stall, GroundedPlaneBackend())
    spec_2d_stall = Motion2DSpec(
        prob_stall, 0, 1.0, (0.0, 0.0, 0.0), 1.0, 0.5,
        gravity=UniformGravity((0.0, 0.0, 2.0), reference_position_m=p_far),
    )
    with pytest.raises(MotionIntegrationError, match="displacement resolution|gravitational potential energy"):
        integrate_motion_2d(spec_2d_stall, GroundedPlaneBackend())

    # 2b. Remote energy gauge (z_ref=5e15 m) causing a 2 J ULP jump in saved total energy -> refused in 1D & 2D
    prob_gauge = ElectrostaticProblem((PointCharge(1e-17, (0.0, 0.0, 1.25)),), (plane0,))
    g_remote = UniformGravity((0.0, 0.0, 2.0), reference_position_m=(0.0, 0.0, 5e15))
    with pytest.raises(MotionIntegrationError, match="gauge|total energy"):
        integrate_motion_1d(
            Motion1DSpec(prob_gauge, 0, 1.0, (0.0, 0.0, 1.0), 0.0, 0.5, 0.25, gravity=g_remote),
            GroundedPlaneBackend(),
        )
    with pytest.raises(MotionIntegrationError, match="gauge|total energy"):
        integrate_motion_2d(
            Motion2DSpec(prob_gauge, 0, 1.0, (0.0, 0.0, 0.0), 0.5, 0.25, gravity=g_remote),
            GroundedPlaneBackend(),
        )

    # 2c. 1D viewer rejects forged graph-only energy_drift_j ("invalid" or 1e100)
    spec_v = Motion1DSpec(
        ElectrostaticProblem((PointCharge(1e-17, (0.0, 0.0, 1.0)),), (plane0,)),
        0, 1.0, (0.0, 0.0, 1.0), 0.0, 0.1, 0.025,
        gravity=UniformGravity((0.0, 0.0, 2.0)),
    )
    b_bad_str = sample_motion_1d(spec_v, GroundedPlaneBackend(), visual_frames=2, resolution=9)
    b_bad_str["motion_1d"]["samples"][1]["energy_drift_j"] = "invalid"
    with pytest.raises(ValueError, match="kinematic and energy fields"):
        render_html(b_bad_str, title="Invalid string drift")
    b_bad_num = sample_motion_1d(spec_v, GroundedPlaneBackend(), visual_frames=2, resolution=9)
    b_bad_num["motion_1d"]["samples"][1]["energy_drift_j"] = 1e100
    with pytest.raises(ValueError, match="energy_drift_j|maximum energy drift"):
        render_html(b_bad_num, title="Forged huge drift")
    b_bad_both = sample_motion_1d(spec_v, GroundedPlaneBackend(), visual_frames=2, resolution=9)
    b_bad_both["motion_1d"]["samples"][1]["energy_drift_j"] = 1e100
    b_bad_both["motion_1d"]["max_abs_energy_drift_j"] = 1e100
    with pytest.raises(ValueError, match="energy_drift_j"):
        render_html(b_bad_both, title="Forged huge drift + max")


def test_m45_sol_v7_reversed_scan_fast_launch_static_gauge_drift_and_2d_drift():
    """Regression suite for all 4 Sol v7 findings."""
    from conductor_sim.motion_1d import MotionIntegrationError

    # 1. Reversed scan axis (0,0,-1) on (-1e18, 0.19) preserves both isolated-sphere roots
    sphere = SolidSphere(center_m=(0.0, 0.0, 0.0), radius_m=0.1, constraint=FixedNetCharge(5e-7))
    spec_rev = Motion1DSpec(
        ElectrostaticProblem((PointCharge(2e-7, (0.0, 0.0, 0.3)),), (sphere,)),
        0, 5e-4, (0.0, 0.0, -1.0), 0.0, 0.1, 0.01,
        gravity=UniformGravity((0.0, 0.0, -42.646393506555945)),
    )
    eq_rev = find_1d_equilibria(
        spec_rev, ExternalIsolatedSphereBackend(),
        displacement_bounds_m=(-1e18, 0.19), scan_samples=129,
    )
    assert len(eq_rev) == 2
    zs_rev = sorted(e.position_m[2] for e in eq_rev)
    assert isclose(zs_rev[0], 0.1716488813521, rel_tol=1e-7)
    assert isclose(zs_rev[1], 0.1716611949418, rel_tol=1e-7)

    # 2. High initial kinetic energy (+100 m/s) cannot mask stalled position or remote gauge in 1D & 2D
    plane0 = ConductingPlane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), Grounded())
    p_far = (0.0, 0.0, 1e16 + 100.0)
    prob_stall = ElectrostaticProblem((PointCharge(1e-17, p_far),), (plane0,))
    with pytest.raises(MotionIntegrationError, match="displacement resolution|gravitational potential energy"):
        integrate_motion_1d(
            Motion1DSpec(
                prob_stall, 0, 1.0, (0.0, 0.0, 1.0), 100.0, 1e-5, 1e-5,
                gravity=UniformGravity((0.0, 0.0, 2.0), reference_position_m=p_far),
            ),
            GroundedPlaneBackend(),
        )
    with pytest.raises(MotionIntegrationError, match="displacement resolution|gravitational potential energy"):
        integrate_motion_2d(
            Motion2DSpec(
                prob_stall, 0, 1.0, (0.0, 0.0, 100.0), 1e-5, 1e-5,
                gravity=UniformGravity((0.0, 0.0, 2.0), reference_position_m=p_far),
            ),
            GroundedPlaneBackend(),
        )
    prob_gauge = ElectrostaticProblem((PointCharge(1e-17, (0.0, 0.0, 1.25)),), (plane0,))
    g_remote = UniformGravity((0.0, 0.0, 2.0), reference_position_m=(0.0, 0.0, 5e15))
    with pytest.raises(MotionIntegrationError, match="gauge|total energy"):
        integrate_motion_1d(
            Motion1DSpec(prob_gauge, 0, 1.0, (0.0, 0.0, 1.0), 100.0, 1e-5, 1e-5, gravity=g_remote),
            GroundedPlaneBackend(),
        )
    with pytest.raises(MotionIntegrationError, match="gauge|total energy"):
        integrate_motion_2d(
            Motion2DSpec(prob_gauge, 0, 1.0, (0.0, 0.0, 100.0), 1e-5, 1e-5, gravity=g_remote),
            GroundedPlaneBackend(),
        )

    # 3. 1D viewer rejects forged 1000 J drift at static equilibrium even with z_ref = 1e18 m
    spec_eq_static = Motion1DSpec(
        ElectrostaticProblem(
            (PointCharge(2.0, (0.0, 0.0, 2.0)),),
            (plane0,),
            permittivity_f_per_m=1.0 / (4.0 * pi),
        ),
        0, 1.0, (0.0, 0.0, 1.0), 0.0, 0.1, 0.025,
        gravity=UniformGravity((0.0, 0.0, 0.25), reference_position_m=(0.0, 0.0, 1e18)),
    )
    b_eq = sample_motion_1d(spec_eq_static, GroundedPlaneBackend(), visual_frames=2, resolution=9)
    assert "<canvas" in render_html(b_eq, title="Valid static equilibrium with remote gauge")
    b_eq["motion_1d"]["samples"][1]["energy_drift_j"] = 1000.0
    b_eq["motion_1d"]["max_abs_energy_drift_j"] = 1000.0
    with pytest.raises(ValueError, match="energy_drift_j"):
        render_html(b_eq, title="Forged 1000 J static equilibrium drift")

    # 4. 2D viewer rejects forged 1e100 J drift (on row 1 or row 0) even when summary max matches
    spec_2d_v = Motion2DSpec(
        ElectrostaticProblem((PointCharge(1e-17, (0.0, 0.0, 1.0)),), (plane0,)),
        0, 1.0, (0.0, 0.0, 0.0), 0.1, 0.025,
        gravity=UniformGravity((0.0, 0.0, 2.0)),
    )
    b_2d = sample_motion_2d(spec_2d_v, GroundedPlaneBackend(), visual_frames=2, resolution=9)
    assert "<canvas" in render_html(b_2d, title="Valid 2D gravity bundle")
    b_2d["motion_2d"]["samples"][1]["energy_drift_j"] = 1e100
    b_2d["motion_2d"]["max_abs_energy_drift_j"] = 1e100
    with pytest.raises(ValueError, match="energy_drift_j"):
        render_html(b_2d, title="Forged 2D row 1 drift")
    b_2d_0 = sample_motion_2d(spec_2d_v, GroundedPlaneBackend(), visual_frames=2, resolution=9)
    b_2d_0["motion_2d"]["samples"][0]["energy_drift_j"] = 1e100
    b_2d_0["motion_2d"]["max_abs_energy_drift_j"] = 1e100
    with pytest.raises(ValueError, match="initial energy_drift_j"):
        render_html(b_2d_0, title="Forged 2D row 0 drift")


def test_m45_sol_v8_tied_reversed_scan_sub_1e24_loss_and_remote_gauge_energy_forgeries():
    """Regression suite for all 3 Sol v8 findings."""
    from conductor_sim.motion_1d import MotionIntegrationError

    # 1. Reversed scan (-z) with all-tied endpoint/interior forces shrinks toward conductor & finds both roots
    sphere = SolidSphere(center_m=(0.0, 0.0, 0.0), radius_m=0.125, constraint=FixedNetCharge(7.0))
    prob_tie = ElectrostaticProblem(
        (PointCharge(18.0, (0.0, 0.0, 0.5)),),
        (sphere,),
        permittivity_f_per_m=1.0 / (4.0 * pi),
    )
    g_tie = UniformGravity((0.0, 0.0, -100.0))
    spec_fwd = Motion1DSpec(prob_tie, 0, 1.0, (0.0, 0.0, 1.0), 0.0, 0.1, 0.05, gravity=g_tie)
    spec_rev = Motion1DSpec(prob_tie, 0, 1.0, (0.0, 0.0, -1.0), 0.0, 0.1, 0.05, gravity=g_tie)
    for n_scan in (9, 129):
        eq_f = find_1d_equilibria(spec_fwd, ExternalIsolatedSphereBackend(), displacement_bounds_m=(-0.25, 1e18), scan_samples=n_scan)
        eq_r = find_1d_equilibria(spec_rev, ExternalIsolatedSphereBackend(), displacement_bounds_m=(-1e18, 0.25), scan_samples=n_scan)
        assert len(eq_f) == 2 and len(eq_r) == 2
        zf = sorted(e.position_m[2] for e in eq_f)
        zr = sorted(e.position_m[2] for e in eq_r)
        assert isclose(zf[0], 0.253263038348, rel_tol=1e-7) and isclose(zr[0], 0.253263038348, rel_tol=1e-7)
        assert isclose(zf[1], 1.118382756045, rel_tol=1e-7) and isclose(zr[1], 1.118382756045, rel_tol=1e-7)

    # 2. Sub-1e-24 J gravitational exchange loss (2e-26 J at z_ref=5e15 m) is refused in BOTH 1D & 2D
    plane0 = ConductingPlane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), Grounded())
    prob_tiny_e = ElectrostaticProblem((PointCharge(1e-17, (0.0, 0.0, 1.25)),), (plane0,))
    g_far_ref = UniformGravity((0.0, 0.0, 2.0), reference_position_m=(0.0, 0.0, 5e15))
    with pytest.raises(MotionIntegrationError, match="gauge|total energy"):
        integrate_motion_1d(
            Motion1DSpec(prob_tiny_e, 0, 1e-18, (0.0, 0.0, 1.0), 0.0, 1e-4, 1e-4, gravity=g_far_ref),
            GroundedPlaneBackend(),
        )
    with pytest.raises(MotionIntegrationError, match="gauge|total energy"):
        integrate_motion_2d(
            Motion2DSpec(prob_tiny_e, 0, 1e-18, (0.0, 0.0, 0.0), 1e-4, 1e-4, gravity=g_far_ref),
            GroundedPlaneBackend(),
        )

    # 3. Both 1D and 2D viewers reject +1e8 J forgery to total_energy_j, interaction_energy_j, or U_g at z_ref=1e18 m
    prob_eq_g = ElectrostaticProblem(
        (PointCharge(2.0, (0.0, 0.0, 2.0)),),
        (plane0,),
        permittivity_f_per_m=1.0 / (4.0 * pi),
    )
    g_eq_1e18 = UniformGravity((0.0, 0.0, 0.25), reference_position_m=(0.0, 0.0, 1e18))
    b1 = sample_motion_1d(
        Motion1DSpec(prob_eq_g, 0, 1.0, (0.0, 0.0, 1.0), 0.0, 0.1, 0.025, gravity=g_eq_1e18),
        GroundedPlaneBackend(), visual_frames=2, resolution=9,
    )
    b2 = sample_motion_2d(
        Motion2DSpec(prob_eq_g, 0, 1.0, (0.0, 0.0, 0.0), 0.1, 0.025, gravity=g_eq_1e18),
        GroundedPlaneBackend(), visual_frames=2, resolution=9,
    )
    assert "<canvas" in render_html(b1, title="Valid 1D static equilibrium 1e18")
    assert "<canvas" in render_html(b2, title="Valid 2D static equilibrium 1e18")
    import copy
    for field in ("total_energy_j", "interaction_energy_j", "gravitational_potential_energy_j"):
        m1 = copy.deepcopy(b1)
        m1["motion_1d"]["samples"][1][field] += 1e8
        with pytest.raises(ValueError):
            render_html(m1, title=f"Forged 1D {field}")
        m2 = copy.deepcopy(b2)
        m2["motion_2d"]["samples"][1][field] += 1e8
        with pytest.raises(ValueError):
            render_html(m2, title=f"Forged 2D {field}")


def test_m45_sol_v9_transverse_gravity_large_k0_tiny_energy_forgeries_and_distant_launch():
    """Regression suite for all 4 Sol v9 findings."""
    import copy
    from conductor_sim.motion_1d import MotionIntegrationError

    plane0 = ConductingPlane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), Grounded())

    # 1. Transverse gravity g=(64, 0, 2^40) on +z axis must be rejected (never drop 64 m/s^2!)
    prob_tr = ElectrostaticProblem(
        (PointCharge(float(2**22), (0.0, 0.0, 2.0)),),
        (plane0,),
        permittivity_f_per_m=1.0 / (4.0 * pi),
    )
    with pytest.raises(UnsupportedProblem, match="collinear"):
        Motion1DSpec(
            prob_tr, 0, 1.0, (0.0, 0.0, 1.0), 0.0, 0.125, 0.025,
            gravity=UniformGravity((64.0, 0.0, float(2**40)), reference_position_m=(0.0, 0.0, 2.0)),
        )

    # 2. Large launch kinetic energy (v0 = 1e7 m/s) cannot swallow gravitational exchange in 1D or 2D
    prob_fast = ElectrostaticProblem((PointCharge(1e-17, (0.0, 0.0, 1.25)),), (plane0,))
    with pytest.raises(MotionIntegrationError, match="kinetic energy|total energy"):
        integrate_motion_1d(
            Motion1DSpec(
                prob_fast, 0, 1.0, (0.0, 0.0, 1.0), 1e7, 1e-4, 2.5e-5,
                gravity=UniformGravity((0.0, 0.0, 1e-10), reference_position_m=(0.0, 0.0, 1.25)),
            ),
            GroundedPlaneBackend(),
        )
    with pytest.raises(MotionIntegrationError, match="kinetic energy|total energy"):
        integrate_motion_2d(
            Motion2DSpec(
                prob_fast, 0, 1.0, (1e7, 0.0, 0.0), 1e-4, 2.5e-5,
                gravity=UniformGravity((0.0, 0.0, 2.0), reference_position_m=(0.0, 0.0, 1.25)),
            ),
            GroundedPlaneBackend(),
        )

    # 3. Sub-1e-30 J energy forgeries (+9e-31 J) and sub-1e-24 J drift forgeries (9e-25 J) are rejected in 1D & 2D
    q_tiny = float(2**-50)
    m_tiny = (q_tiny * q_tiny) / 4.0
    prob_sub30 = ElectrostaticProblem(
        (PointCharge(q_tiny, (0.0, 0.0, 2.0)),),
        (plane0,),
        permittivity_f_per_m=1.0 / (4.0 * pi),
    )
    g_sub30 = UniformGravity((0.0, 0.0, 0.25), reference_position_m=(0.0, 0.0, 3.0))
    b1 = sample_motion_1d(
        Motion1DSpec(prob_sub30, 0, m_tiny, (0.0, 0.0, 1.0), 0.0, 0.1, 0.025, gravity=g_sub30),
        GroundedPlaneBackend(), visual_frames=2, resolution=9,
    )
    b2 = sample_motion_2d(
        Motion2DSpec(prob_sub30, 0, m_tiny, (0.0, 0.0, 0.0), 0.1, 0.025, gravity=g_sub30),
        GroundedPlaneBackend(), visual_frames=2, resolution=9,
    )
    assert "<canvas" in render_html(b1, title="Valid tiny 1D equilibrium")
    assert "<canvas" in render_html(b2, title="Valid tiny 2D equilibrium")
    for field in ("total_energy_j", "interaction_energy_j", "gravitational_potential_energy_j"):
        m1 = copy.deepcopy(b1)
        m1["motion_1d"]["samples"][1][field] += 9e-31
        with pytest.raises(ValueError):
            render_html(m1, title=f"Forged tiny 1D {field}")
        m2 = copy.deepcopy(b2)
        m2["motion_2d"]["samples"][1][field] += 9e-31
        with pytest.raises(ValueError):
            render_html(m2, title=f"Forged tiny 2D {field}")
    d1 = copy.deepcopy(b1)
    d1["motion_1d"]["samples"][1]["energy_drift_j"] = 9e-25
    d1["motion_1d"]["max_abs_energy_drift_j"] = 9e-25
    with pytest.raises(ValueError, match="energy_drift_j"):
        render_html(d1, title="Forged tiny 1D drift")
    d2 = copy.deepcopy(b2)
    d2["motion_2d"]["samples"][1]["energy_drift_j"] = 9e-25
    d2["motion_2d"]["max_abs_energy_drift_j"] = 9e-25
    with pytest.raises(ValueError, match="energy_drift_j"):
        render_html(d2, title="Forged tiny 2D drift")

    # 4. Distant launch origin (z0 = 1e10 m) cannot resolve the root at z=0.25000003 m -> must refuse
    sphere = SolidSphere(center_m=(0.0, 0.0, 0.0), radius_m=0.125, constraint=FixedNetCharge(7.0))
    prob_near = ElectrostaticProblem(
        (PointCharge(18.0, (0.0, 0.0, 0.5)),),
        (sphere,),
        permittivity_f_per_m=1.0 / (4.0 * pi),
    )
    eq_near = find_1d_equilibria(
        Motion1DSpec(
            prob_near, 0, 1.0, (0.0, 0.0, 1.0), 0.0, 0.1, 0.05,
            surface_guard_m=0.1, gravity=UniformGravity((0.0, 0.0, -0.001)),
        ),
        ExternalIsolatedSphereBackend(),
        displacement_bounds_m=(0.23 - 0.5, 500.0 - 0.5),
        scan_samples=129,
    )
    assert len(eq_near) == 2
    assert isclose(eq_near[0].position_m[2], 0.2500000306372732, rel_tol=1e-9)
    prob_far_launch = ElectrostaticProblem(
        (PointCharge(18.0, (0.0, 0.0, 1e10)),),
        (sphere,),
        permittivity_f_per_m=1.0 / (4.0 * pi),
    )
    with pytest.raises(UnsupportedProblem, match="force balance|precision"):
        find_1d_equilibria(
            Motion1DSpec(
                prob_far_launch, 0, 1.0, (0.0, 0.0, 1.0), 0.0, 0.1, 0.05,
                surface_guard_m=0.1, gravity=UniformGravity((0.0, 0.0, -0.001)),
            ),
            ExternalIsolatedSphereBackend(),
            displacement_bounds_m=(0.23 - 1e10, 500.0 - 1e10),
            scan_samples=129,
        )


def test_m45_sol_v10_tilted_1d_axis_2d_azimuth_tiny_acc_ke_and_huge_rest_drift():
    """Regression suite for all 4 Sol v10 findings."""
    import copy

    plane0 = ConductingPlane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), Grounded())

    # 1. Tilted 1D axis (2^-35, 0, 1) with g=(32, 0, 2^40) must be rejected (never lose 32 N transverse load!)
    prob_tilt = ElectrostaticProblem(
        (PointCharge(float(2**22), (0.0, 0.0, 2.0)),),
        (plane0,),
        permittivity_f_per_m=1.0 / (4.0 * pi),
    )
    with pytest.raises(UnsupportedProblem, match="plane normal|transverse"):
        Motion1DSpec(
            prob_tilt, 0, 1.0, (float(2**-35), 0.0, 1.0), 0.0, 0.125, 0.025,
            gravity=UniformGravity((32.0, 0.0, float(2**40))),
        )

    # 2. 2D plane selection is azimuth-invariant for g=(0.125, 0, 2^40) vs g=(0, 0.125, 2^40)
    res_xz = integrate_motion_2d(
        Motion2DSpec(
            prob_tilt, 0, 1.0, (0.0, 0.0, 0.0), 0.125, 0.025,
            gravity=UniformGravity((0.125, 0.0, float(2**40)), reference_position_m=(0.0, 0.0, 2.0)),
        ),
        GroundedPlaneBackend(),
    )
    res_yz = integrate_motion_2d(
        Motion2DSpec(
            prob_tilt, 0, 1.0, (0.0, 0.0, 0.0), 0.125, 0.025,
            gravity=UniformGravity((0.0, 0.125, float(2**40)), reference_position_m=(0.0, 0.0, 2.0)),
        ),
        GroundedPlaneBackend(),
    )
    expected_disp = 0.5 * 0.125 * (0.125 ** 2)
    assert isclose(res_xz.samples[-1].position_m[0], expected_disp, rel_tol=1e-9)
    assert isclose(res_yz.samples[-1].position_m[1], expected_disp, rel_tol=1e-9)

    # 3. 2D viewer rejects sub-1e-30 acceleration and kinetic-energy forgeries when force/velocity are zero
    q_tiny = float(2**-50)
    m_tiny = (q_tiny * q_tiny) / 4.0
    prob_sub30 = ElectrostaticProblem(
        (PointCharge(q_tiny, (0.0, 0.0, 2.0)),),
        (plane0,),
        permittivity_f_per_m=1.0 / (4.0 * pi),
    )
    g_sub30 = UniformGravity((0.0, 0.0, 0.25), reference_position_m=(0.0, 0.0, 3.0))
    b2_tiny = sample_motion_2d(
        Motion2DSpec(prob_sub30, 0, m_tiny, (0.0, 0.0, 0.0), 0.1, 0.025, gravity=g_sub30),
        GroundedPlaneBackend(), visual_frames=2, resolution=9,
    )
    b2_acc = copy.deepcopy(b2_tiny)
    b2_acc["motion_2d"]["samples"][1]["acceleration_m_s2"] = [9e-31, 0.0, 0.0]
    with pytest.raises(ValueError, match="acceleration"):
        render_html(b2_acc, title="Forged tiny 2D acceleration")
    b2_ke = copy.deepcopy(b2_tiny)
    for row in b2_ke["motion_2d"]["samples"]:
        row["kinetic_energy_j"] += 9e-31
        row["total_energy_j"] += 9e-31
    with pytest.raises(ValueError, match="kinetic energy"):
        render_html(b2_ke, title="Forged tiny 2D kinetic energy")

    # 4. Both 1D and 2D viewers reject fabricated 1 J drift at exact rest even with U_elec = -2^45 J
    q_big = float(2**24)
    m_big = (q_big * q_big) / 16.0
    prob_rest_big = ElectrostaticProblem(
        (PointCharge(q_big, (0.0, 0.0, 2.0)),),
        (plane0,),
        permittivity_f_per_m=1.0 / (4.0 * pi),
    )
    g_rest_big = UniformGravity((0.0, 0.0, 1.0), reference_position_m=(0.0, 0.0, 2.0))
    b1_rest = sample_motion_1d(
        Motion1DSpec(prob_rest_big, 0, m_big, (0.0, 0.0, 1.0), 0.0, 0.1, 0.025, gravity=g_rest_big),
        GroundedPlaneBackend(), visual_frames=2, resolution=9,
    )
    b2_rest = sample_motion_2d(
        Motion2DSpec(prob_rest_big, 0, m_big, (0.0, 0.0, 0.0), 0.1, 0.025, gravity=g_rest_big),
        GroundedPlaneBackend(), visual_frames=2, resolution=9,
    )
    assert "<canvas" in render_html(b1_rest, title="Valid 1D huge rest")
    assert "<canvas" in render_html(b2_rest, title="Valid 2D huge rest")
    b1_rest["motion_1d"]["samples"][1]["energy_drift_j"] = 1.0
    b1_rest["motion_1d"]["max_abs_energy_drift_j"] = 1.0
    with pytest.raises(ValueError, match="energy_drift_j"):
        render_html(b1_rest, title="Forged 1D 1 J rest drift")
    b2_rest["motion_2d"]["samples"][1]["energy_drift_j"] = 1.0
    b2_rest["motion_2d"]["max_abs_energy_drift_j"] = 1.0
    with pytest.raises(ValueError, match="energy_drift_j"):
        render_html(b2_rest, title="Forged 2D 1 J rest drift")
