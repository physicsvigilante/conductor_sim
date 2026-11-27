"""M45 acceptance suite: UniformGravity coupled with responding-conductor dynamics & equilibria."""
from __future__ import annotations

from math import inf, isclose, nextafter, pi, sqrt

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

    # (c) Cancelling near-tangent fold (|F_peak - mg| <= 16 ulp) is honestly
    #     refused across all scan_samples rather than fabricating a root count.
    m_tan = f_peak / g_mag
    spec_tan = Motion1DSpec(
        ElectrostaticProblem((PointCharge(q, (0.0, 0.0, 0.25)),), (sphere,)),
        0, m_tan, (0.0, 0.0, 1.0), 0.0, 0.05, 0.01,
        gravity=UniformGravity((0.0, 0.0, -g_mag)),
    )
    for n_scan in (17, 33, 65, 129, 257):
        with pytest.raises(UnsupportedProblem, match="near-tangent"):
            find_1d_equilibria(
                spec_tan, backend,
                displacement_bounds_m=(0.13 - 0.25, 0.35 - 0.25),
                scan_samples=n_scan,
            )

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

    # 2. Grid-aligned cancelling near-tangent (within binary64 force rounding) is honestly refused
    r_tan = 0.17165503777020558
    spec_tan_aligned = Motion1DSpec(
        ElectrostaticProblem((PointCharge(q, (0.0, 0.0, 0.25)),), (sphere,)),
        0, 0.002173618447146783, (0.0, 0.0, 1.0), 0.0, 0.05, 0.01,
        gravity=UniformGravity((0.0, 0.0, -9.81)),
    )
    with pytest.raises(UnsupportedProblem, match="near-tangent"):
        find_1d_equilibria(
            spec_tan_aligned, backend,
            displacement_bounds_m=(r_tan - 0.002 - 0.25, r_tan + 0.002 - 0.25),
            scan_samples=9,
        )

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

    # 1. Exact grid zero at a binary64-cancelling fold peak is honestly refused, never split into 3
    sphere = SolidSphere(center_m=(0.0, 0.0, 0.0), radius_m=0.10, constraint=FixedNetCharge(5.0e-7))
    spec_t0 = Motion1DSpec(
        ElectrostaticProblem((PointCharge(2.0e-7, (0.0, 0.0, 0.1716550374351809)),), (sphere,)),
        0, 0.021323196966509944, (0.0, 0.0, 1.0), 0.0, 0.05, 0.01,
        gravity=UniformGravity((0.0, 0.0, -1.0)),
    )
    with pytest.raises(UnsupportedProblem, match="near-tangent"):
        find_1d_equilibria(
            spec_t0, ExternalIsolatedSphereBackend(),
            displacement_bounds_m=(-0.005, 0.005), scan_samples=9,
        )

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


def test_m45_sol_v11_tilted_normal_2d_exact_cross_cancel_acc_and_unchanged_pos_ulp():
    """Regression suite for all 4 Sol v11 findings."""
    import copy
    from math import nextafter

    # 1. Tilted plane normal (1+2^-45, 2, 2) vs axis (1, 2, 2) with g=(a,2a,2a), a=2^40/3 must refuse
    plane_t = ConductingPlane((0.0, 0.0, 0.0), (1.0 + float(2**-45), 2.0, 2.0), Grounded())
    a_big = float(2**40) / 3.0
    prob_t = ElectrostaticProblem(
        (PointCharge(float(2**22), (2.0 / 3.0, 4.0 / 3.0, 4.0 / 3.0)),),
        (plane_t,),
        permittivity_f_per_m=1.0 / (4.0 * pi),
    )
    with pytest.raises(UnsupportedProblem, match="plane normal|transverse"):
        Motion1DSpec(
            prob_t, 0, 1.0, (1.0, 2.0, 2.0), 0.0, 0.125, 0.025,
            gravity=UniformGravity((a_big, 2.0 * a_big, 2.0 * a_big)),
        )

    # 2. Valid tilted 2D plane with normal (3, 4, 0) and g = (3*2^40 + 0.125, 4*2^40, 0.125) succeeds
    plane_34 = ConductingPlane((0.0, 0.0, 0.0), (3.0, 4.0, 0.0), Grounded())
    g_34 = UniformGravity(
        (3.0 * float(2**40) + 0.125, 4.0 * float(2**40), 0.125),
        reference_position_m=(1.2, 1.6, 0.0),
    )
    prob_34_spec = ElectrostaticProblem(
        (PointCharge(float(2**22), (1.2, 1.6, 0.0)),),
        (plane_34,),
        permittivity_f_per_m=1.0 / (4.0 * pi),
    )
    # Sol's exact spec constructs cleanly without spurious out-of-plane refusal
    spec_34 = Motion2DSpec(prob_34_spec, 0, 1.0, (0.0, 0.0, 0.0), 0.125, 0.025, gravity=g_34)
    assert spec_34.gravity == g_34
    # Over a short nonrelativistic window (1e-8 s), full 2D integration stays cleanly inside the plane
    res_34 = integrate_motion_2d(
        Motion2DSpec(prob_34_spec, 0, 1.0, (0.0, 0.0, 0.0), 1e-8, 5e-9, gravity=g_34),
        GroundedPlaneBackend(),
    )
    assert len(res_34.samples) == 3
    assert isclose(res_34.samples[-1].position_m[2], 0.5 * 0.125 * (1e-8 ** 2), rel_tol=1e-6)

    # 3. Viewers reject fabricated acceleration/force under exact cancellation (q=2^22, g_z=2^40)
    plane0 = ConductingPlane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), Grounded())
    prob_c = ElectrostaticProblem(
        (PointCharge(float(2**22), (0.0, 0.0, 2.0)),),
        (plane0,),
        permittivity_f_per_m=1.0 / (4.0 * pi),
    )
    g_c = UniformGravity((0.0, 0.0, float(2**40)), reference_position_m=(0.0, 0.0, 2.0))
    b1_c = sample_motion_1d(
        Motion1DSpec(prob_c, 0, 1.0, (0.0, 0.0, 1.0), 0.0, 0.1, 0.025, gravity=g_c),
        GroundedPlaneBackend(), visual_frames=2, resolution=9,
    )
    b2_c = sample_motion_2d(
        Motion2DSpec(prob_c, 0, 1.0, (0.0, 0.0, 0.0), 0.1, 0.025, gravity=g_c),
        GroundedPlaneBackend(), visual_frames=2, resolution=9,
    )
    assert "<canvas" in render_html(b1_c, title="Valid 1D exact cancellation")
    assert "<canvas" in render_html(b2_c, title="Valid 2D exact cancellation")
    b1_c["motion_1d"]["samples"][1]["acceleration_m_s2"] = 0.025
    with pytest.raises(ValueError, match="acceleration"):
        render_html(b1_c, title="Forged 1D cancellation acc")
    b2_c["motion_2d"]["samples"][1]["force_n"] = [0.0, 0.0, 0.001]
    b2_c["motion_2d"]["samples"][1]["acceleration_m_s2"] = [0.0, 0.0, 0.001]
    with pytest.raises(ValueError, match="total force"):
        render_html(b2_c, title="Forged 2D cancellation force+acc")

    # 4. Both viewers reject 1-ULP electric_interaction_energy_j edit at unchanged position + 0.125 J drift
    q_big = float(2**24)
    m_big = float(2**44)
    prob_u = ElectrostaticProblem(
        (PointCharge(q_big, (0.0, 0.0, 2.0)),),
        (plane0,),
        permittivity_f_per_m=1.0 / (4.0 * pi),
    )
    g_u = UniformGravity((0.0, 0.0, 1.0), reference_position_m=(0.0, 0.0, 2.0))
    b1_u = sample_motion_1d(
        Motion1DSpec(prob_u, 0, m_big, (0.0, 0.0, 1.0), 0.0, 0.1, 0.025, gravity=g_u),
        GroundedPlaneBackend(), visual_frames=2, resolution=9,
    )
    b2_u = sample_motion_2d(
        Motion2DSpec(prob_u, 0, m_big, (0.0, 0.0, 0.0), 0.1, 0.025, gravity=g_u),
        GroundedPlaneBackend(), visual_frames=2, resolution=9,
    )
    for b_u, key in ((copy.deepcopy(b1_u), "motion_1d"), (copy.deepcopy(b2_u), "motion_2d")):
        s1 = b_u[key]["samples"][1]
        s1["electric_interaction_energy_j"] = nextafter(-float(2**45), float("inf"))
        s1["energy_drift_j"] = 0.125
        b_u[key]["max_abs_energy_drift_j"] = 0.125
        with pytest.raises(ValueError, match="repeated particle position|interaction energy"):
            render_html(b_u, title=f"Forged 1-ULP rest edit {key}")


def test_m45_sol_v12_collinear_1d_eq_velocity_exact_cross_1ulp_gforce_and_repeated_pos():
    """Regression suite for all 4 Sol v12 findings."""
    import copy
    from math import nextafter

    # 1. Exactly collinear (1, 2, 2) plane/axis/gravity with a=2^40/3 finds the unstable equilibrium near s=0
    plane_122 = ConductingPlane((0.0, 0.0, 0.0), (1.0, 2.0, 2.0), Grounded())
    a_big = float(2**40) / 3.0
    spec_122 = Motion1DSpec(
        ElectrostaticProblem(
            (PointCharge(float(2**22), (2.0 / 3.0, 4.0 / 3.0, 4.0 / 3.0)),),
            (plane_122,),
            permittivity_f_per_m=1.0 / (4.0 * pi),
        ),
        0, 1.0, (1.0, 2.0, 2.0), 0.0, 0.125, 0.025,
        gravity=UniformGravity((a_big, 2.0 * a_big, 2.0 * a_big)),
    )
    eq_122 = find_1d_equilibria(
        spec_122, GroundedPlaneBackend(), displacement_bounds_m=(-0.001, 0.001), scan_samples=17,
    )
    assert len(eq_122) == 1
    assert abs(eq_122[0].displacement_m) < 1e-12
    assert eq_122[0].stability == "unstable"

    # 2. Valid 2D plane selected by nearly-parallel launch velocity (1, 2, 3) vs (1.0000000001, 2, 2.9999999998)
    plane_123 = ConductingPlane((0.0, 0.0, 0.0), (1.0, 2.0, 3.0), Grounded())
    vec_123 = (1.0000000001, 2.0, 2.9999999998)
    spec_123 = Motion2DSpec(
        ElectrostaticProblem((PointCharge(1e-6, (2.0, 2.0, 2.0)),), (plane_123,)),
        0, 1.0, vec_123, 0.001, 0.0005,
        gravity=UniformGravity(vec_123),
    )
    b_123 = sample_motion_2d(spec_123, GroundedPlaneBackend(), visual_frames=2, resolution=9)
    assert "<canvas" in render_html(b_123, title="Valid 123 velocity-selected 2D plane")

    # 3. Both viewers reject 1-ULP gravitational_force_n edit (2^40 -> nextafter) + 2^-12 total force/acc
    plane0 = ConductingPlane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), Grounded())
    prob_c = ElectrostaticProblem(
        (PointCharge(float(2**22), (0.0, 0.0, 2.0)),),
        (plane0,),
        permittivity_f_per_m=1.0 / (4.0 * pi),
    )
    g_c = UniformGravity((0.0, 0.0, float(2**40)), reference_position_m=(0.0, 0.0, 2.0))
    b1_c = sample_motion_1d(
        Motion1DSpec(prob_c, 0, 1.0, (0.0, 0.0, 1.0), 0.0, 0.1, 0.025, gravity=g_c),
        GroundedPlaneBackend(), visual_frames=2, resolution=9,
    )
    b2_c = sample_motion_2d(
        Motion2DSpec(prob_c, 0, 1.0, (0.0, 0.0, 0.0), 0.1, 0.025, gravity=g_c),
        GroundedPlaneBackend(), visual_frames=2, resolution=9,
    )
    b1_c["motion_1d"]["samples"][1]["gravitational_force_n"] = nextafter(float(2**40), float("inf"))
    b1_c["motion_1d"]["samples"][1]["force_n"] = float(2**-12)
    b1_c["motion_1d"]["samples"][1]["acceleration_m_s2"] = float(2**-12)
    with pytest.raises(ValueError, match="gravitational force"):
        render_html(b1_c, title="Forged 1D 1-ULP g_force")
    b2_c["motion_2d"]["samples"][1]["gravitational_force_n"] = [0.0, 0.0, nextafter(float(2**40), float("inf"))]
    b2_c["motion_2d"]["samples"][1]["force_n"] = [0.0, 0.0, float(2**-12)]
    b2_c["motion_2d"]["samples"][1]["acceleration_m_s2"] = [0.0, 0.0, float(2**-12)]
    with pytest.raises(ValueError, match="gravitational force"):
        render_html(b2_c, title="Forged 2D 1-ULP g_force")

    # 4. Both viewers reject 1-ULP electric energy forgery at a repeated NON-INITIAL position
    q_big = float(2**24)
    m_big = float(2**44)
    prob_u = ElectrostaticProblem(
        (PointCharge(q_big, (0.0, 0.0, 2.0)),),
        (plane0,),
        permittivity_f_per_m=1.0 / (4.0 * pi),
    )
    g_u = UniformGravity((0.0, 0.0, 1.0), reference_position_m=(0.0, 0.0, 2.0))
    b1_m = sample_motion_1d(
        Motion1DSpec(prob_u, 0, m_big, (0.0, 0.0, 1.0), 0.01, 0.1, 0.025, gravity=g_u),
        GroundedPlaneBackend(), visual_frames=2, resolution=9,
    )
    b2_m = sample_motion_2d(
        Motion2DSpec(prob_u, 0, m_big, (0.0, 0.0, 0.01), 0.1, 0.025, gravity=g_u),
        GroundedPlaneBackend(), visual_frames=2, resolution=9,
    )
    for b_m, key in ((copy.deepcopy(b1_m), "motion_1d"), (copy.deepcopy(b2_m), "motion_2d")):
        t2 = b_m[key]["samples"][2]["time_s"]
        dup = copy.deepcopy(b_m[key]["samples"][1])
        dup["time_s"] = t2
        dup["electric_interaction_energy_j"] = nextafter(dup["electric_interaction_energy_j"], float("inf"))
        dup["interaction_energy_j"] = dup["electric_interaction_energy_j"] + dup["gravitational_potential_energy_j"]
        dup["total_energy_j"] = dup["kinetic_energy_j"] + dup["interaction_energy_j"]
        dup["energy_drift_j"] += 0.125
        b_m[key]["samples"][2] = dup
        b_m[key]["max_abs_energy_drift_j"] = max(abs(r["energy_drift_j"]) for r in b_m[key]["samples"])
        with pytest.raises(ValueError, match="repeated particle position"):
            render_html(b_m, title=f"Forged repeated noninitial pos {key}")


def test_m45_sol_v13_collinear_245_coplanar_2d_exact_normal_and_same_pos_ke_drift():
    """Regression suite for all 3 Sol v13 findings."""
    import copy
    from math import nextafter

    # 1. Exact collinear (2, 4, 5) plane/axis/gravity finds equilibrium regardless of duration_s (0.006 vs 0.007)
    plane_245 = ConductingPlane((0.0, 0.0, 0.0), (2.0, 4.0, 5.0), Grounded())
    n_245 = plane_245.normal
    prob_245 = ElectrostaticProblem(
        (PointCharge(float(2**22), tuple(2.0 * c for c in n_245)),),
        (plane_245,),
        permittivity_f_per_m=1.0 / (4.0 * pi),
    )
    g_245 = UniformGravity(tuple(float(2**40) * c for c in n_245))
    for dur in (0.006, 0.007, 1.0):
        spec_245 = Motion1DSpec(prob_245, 0, 1.0, (2.0, 4.0, 5.0), 0.0, dur, dur / 2.0, gravity=g_245)
        eq_245 = find_1d_equilibria(
            spec_245, GroundedPlaneBackend(), displacement_bounds_m=(-0.001, 0.001), scan_samples=17,
        )
        assert len(eq_245) == 1
        assert abs(eq_245[0].displacement_m) < 1e-12
        assert eq_245[0].stability == "unstable"

    # 2. Exactly coplanar 2D gravity proportional to launch velocity (4, 3, 1) passes with exact rational normal
    plane_123 = ConductingPlane((0.0, 0.0, 0.0), (1.0, 2.0, 3.0), Grounded())
    prob_co = ElectrostaticProblem((PointCharge(1e-12, (0.001, 0.001, 0.001)),), (plane_123,))
    g_co = UniformGravity(tuple(float(2**20) * c for c in (4.0, 3.0, 1.0)))
    spec_co = Motion2DSpec(prob_co, 0, 1.0, (4.0, 3.0, 1.0), 1.0, 0.25, gravity=g_co)
    assert spec_co.gravity == g_co

    # 3. Both 1D and 2D viewers reject different drift values at identical (position, kinetic_energy)
    plane0 = ConductingPlane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), Grounded())
    q_h = float(2**47)
    m_h = float(2**90)
    prob_h = ElectrostaticProblem(
        (PointCharge(q_h, (0.0, 0.0, 2.0)),),
        (plane0,),
        permittivity_f_per_m=1.0 / (4.0 * pi),
    )
    g_h = UniformGravity((0.0, 0.0, 1.0), reference_position_m=(0.0, 0.0, 2.0))
    b1_h = sample_motion_1d(
        Motion1DSpec(prob_h, 0, m_h, (0.0, 0.0, 1.0), 0.01, 0.1, 0.025, gravity=g_h),
        GroundedPlaneBackend(), visual_frames=2, resolution=9,
    )
    b2_h = sample_motion_2d(
        Motion2DSpec(prob_h, 0, m_h, (0.0, 0.0, 0.01), 0.1, 0.025, gravity=g_h),
        GroundedPlaneBackend(), visual_frames=2, resolution=9,
    )
    for b_h, key, is_2d in ((copy.deepcopy(b1_h), "motion_1d", False), (copy.deepcopy(b2_h), "motion_2d", True)):
        for dst_idx in (2, 3):
            t_dst = b_h[key]["samples"][dst_idx]["time_s"]
            dup = copy.deepcopy(b_h[key]["samples"][1])
            dup["time_s"] = t_dst
            if is_2d:
                vz = dup["velocity_m_s"][2]
                for _ in range(4):
                    vz = nextafter(vz, float("inf"))
                dup["velocity_m_s"] = [0.0, 0.0, vz]
                dup["kinetic_energy_j"] = 0.5 * m_h * (vz * vz)
            else:
                v_s = dup["velocity_m_s"]
                for _ in range(4):
                    v_s = nextafter(v_s, float("inf"))
                dup["velocity_m_s"] = v_s
                dup["kinetic_energy_j"] = 0.5 * m_h * (v_s * v_s)
            dup["total_energy_j"] = dup["kinetic_energy_j"] + dup["interaction_energy_j"]
            if dst_idx == 3:
                dup["energy_drift_j"] += float(2**26)
            b_h[key]["samples"][dst_idx] = dup
        b_h[key]["max_abs_energy_drift_j"] = max(abs(r["energy_drift_j"]) for r in b_h[key]["samples"])
        with pytest.raises(ValueError, match="identical position and kinetic energy"):
            render_html(b_h, title=f"Forged same pos+KE drift {key}")


def test_m45_sol_v14_exact_rational_1d_collinear_2d_fallback_and_huge_out_of_plane():
    """Regression suite for all 3 Sol v14 findings."""
    # 1. Exact collinear (1, 2, 3) plane/axis/gravity (g = 2^40 * n) is duration-independent (0.001 vs 0.1 s)
    plane_123 = ConductingPlane((0.0, 0.0, 0.0), (1.0, 2.0, 3.0), Grounded())
    n_123 = plane_123.normal
    prob_123 = ElectrostaticProblem(
        (PointCharge(float(2**22), tuple(2.0 * c for c in n_123)),),
        (plane_123,),
        permittivity_f_per_m=1.0 / (4.0 * pi),
    )
    g_123 = UniformGravity(tuple(float(2**40) * c for c in n_123))
    for dur in (0.001, 0.1, 1.0):
        spec_123 = Motion1DSpec(prob_123, 0, 1.0, (1.0, 2.0, 3.0), 0.0, dur, dur / 2.0, gravity=g_123)
        eq_123 = find_1d_equilibria(
            spec_123, GroundedPlaneBackend(), displacement_bounds_m=(-0.001, 0.001), scan_samples=17,
        )
        assert len(eq_123) == 1
        assert abs(eq_123[0].displacement_m) < 1e-12
        assert eq_123[0].stability == "unstable"

    # 2. 2D release from rest with g = 2^20 * n on (1, 2, 3) plane uses exact rational fallback normal
    p0_tiny = tuple(1e-8 * c for c in n_123)
    prob_2d_rest = ElectrostaticProblem((PointCharge(1e-24, p0_tiny),), (plane_123,))
    g_2d_rest = UniformGravity(tuple(float(2**20) * c for c in n_123), reference_position_m=p0_tiny)
    for dur in (1e-4, 1e-3):
        spec_rest = Motion2DSpec(prob_2d_rest, 0, 1.0, (0.0, 0.0, 0.0), dur, dur / 2.0, gravity=g_2d_rest)
        assert spec_rest.gravity == g_2d_rest

    # 3. Huge finite out-of-plane gravity (8e307, 8e307, 8e307) raises UnsupportedProblem (never OverflowError!)
    plane_1m10 = ConductingPlane((0.0, 0.0, 0.0), (1.0, -1.0, 0.0), Grounded())
    prob_ovf = ElectrostaticProblem((PointCharge(1e-12, (1.0, 0.0, 0.0)),), (plane_1m10,))
    g_huge_oop = UniformGravity((8e307, 8e307, 8e307), reference_position_m=(1.0, 0.0, 0.0))
    with pytest.raises(UnsupportedProblem, match="invariant plane"):
        Motion2DSpec(prob_ovf, 0, 1e-300, (1.0, 1.0, -2.0), 1e-3, 5e-4, gravity=g_huge_oop)


def test_m45_sol_v15_near_normal_launch_plane_and_underflow_free_out_of_plane():
    """Regression suite for both Sol v15 findings."""
    plane0 = ConductingPlane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), Grounded())
    prob = ElectrostaticProblem((PointCharge(1e-12, (0.0, 0.0, 1.0)),), (plane0,))

    # 1a. Near-normal launch (0, 5e-11, 100) with g=(0, 0, -100) selects yz plane & passes sample_motion_2d
    spec_yz = Motion2DSpec(
        prob, 0, 1.0, (0.0, 5e-11, 100.0), 1.0, 0.5,
        gravity=UniformGravity((0.0, 0.0, -100.0)),
    )
    b_yz = sample_motion_2d(spec_yz, GroundedPlaneBackend(), visual_frames=2, resolution=9)
    assert abs(b_yz["motion_2d"]["plane_normal"][0]) > 1.0 - 1e-12
    assert "<canvas" in render_html(b_yz, title="Near-normal yz plane")

    # 1b. Near-normal launch (0, 5e-11, 100) with out-of-plane g=(1, 0, -100) is strictly 3D -> refused!
    with pytest.raises(UnsupportedProblem, match="invariant plane"):
        Motion2DSpec(
            prob, 0, 1.0, (0.0, 5e-11, 100.0), 1.0, 0.5,
            gravity=UniformGravity((1.0, 0.0, -100.0)),
        )

    # 2. Huge-parallel + tiny-transverse gravity (1e308, 1e-20, 0) over 1e6 s must NOT underflow 1e-20 m/s^2!
    with pytest.raises(UnsupportedProblem, match="invariant plane"):
        Motion2DSpec(
            prob, 0, 1e-300, (1.0, 0.0, 0.0), 1e6, 5e5,
            gravity=UniformGravity((1e308, 1e-20, 0.0), reference_position_m=(0.0, 0.0, 1.0)),
        )


def test_m45_sol_v16_tiny_yz_velocity_and_exact_rational_displacement_bounds():
    """Regression suite for both Sol v16 findings."""
    plane0 = ConductingPlane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), Grounded())

    # 1. Valid yz trajectory with v=(0, 1e-14, 1) selects yz plane & passes sample_motion_2d + render_html
    prob_yz = ElectrostaticProblem((PointCharge(1e-30, (0.0, 0.0, 1.0)),), (plane0,))
    spec_yz = Motion2DSpec(
        prob_yz, 0, 1.0, (0.0, 1e-14, 1.0), 2.0, 2.0,
        gravity=UniformGravity((0.0, 0.0, -1.0)),
    )
    b_yz = sample_motion_2d(spec_yz, GroundedPlaneBackend(), visual_frames=2, resolution=9)
    assert abs(b_yz["motion_2d"]["plane_normal"][0]) > 1.0 - 1e-12
    assert "<canvas" in render_html(b_yz, title="Valid 1e-14 yz trajectory")

    # 2a. Subnormal transverse gravity 5e-324 over t=1e154 s (0.5*5e-324*1e308 = 2.47e-16 > 1e-17) is refused
    prob_sub_t = ElectrostaticProblem((PointCharge(1e-30, (0.0, 0.0, 1e-8)),), (plane0,))
    with pytest.raises(UnsupportedProblem, match="invariant plane"):
        Motion2DSpec(
            prob_sub_t, 0, 1.0, (1e-154, 0.0, 0.0), 1e154, 1e154,
            gravity=UniformGravity((1e-308, 5e-324, 0.0), reference_position_m=(0.0, 0.0, 1e-8)),
        )
    # Same subnormal-times-t^2 check in 1D on a tilted (all-nonzero) axis so
    # it bypasses both the zero-axis and sine checks and exercises _exceeds_accel_disp directly.
    plane_tilted = ConductingPlane((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), Grounded())
    d_tilt = 1e-8 / sqrt(3.0)
    p_tilt = (d_tilt, d_tilt, d_tilt)
    prob_sub_1d = ElectrostaticProblem((PointCharge(1e-30, p_tilt),), (plane_tilted,))
    g_sub_1d = (1e-308, 1e-308, nextafter(1e-308, inf))
    with pytest.raises(UnsupportedProblem, match="collinear"):
        Motion1DSpec(
            prob_sub_1d, 0, 1.0, plane_tilted.normal, 0.0, 1e154, 1e154,
            gravity=UniformGravity(g_sub_1d, reference_position_m=p_tilt),
        )
    # Shorter duration (1e153 s -> 0.5*a_trans*t^2 ~ 2e-18 m <= 1e-17 m) is accepted.
    spec_sub_1d_ok = Motion1DSpec(
        prob_sub_1d, 0, 1.0, plane_tilted.normal, 0.0, 1e153, 1e153,
        gravity=UniformGravity(g_sub_1d, reference_position_m=p_tilt),
    )
    assert spec_sub_1d_ok.gravity is not None

    # 2b. Huge duration t=1e160 s does NOT disable gravity-based plane selection via t^2 overflow
    spec_1e160 = Motion2DSpec(
        prob_yz, 0, 1.0, (0.0, 0.0, 0.0), 1e160, 1e160,
        gravity=UniformGravity((0.0, 1e-322, 1e-306)),
    )
    assert spec_1e160.gravity is not None


def test_m45_sol_v17_nearly_axial_held_charge_selects_exact_plane():
    """Regression suite for Sol v17: nearly axial held charge selects its exact invariant plane."""
    plane0 = ConductingPlane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), Grounded())

    # 1. Exact Sol v17 reproducer: held charge at (0, 5e-13, 2) determines the yz plane
    prob = ElectrostaticProblem(
        (
            PointCharge(1e-6, (0.0, 0.0, 1.0)),
            PointCharge(-9e-6 / 40.0, (0.0, 5e-13, 2.0)),
        ),
        (plane0,),
    )
    spec = Motion2DSpec(
        prob, 0, 1.0, (0.0, 0.0, 0.0), 0.1, 0.05,
        gravity=UniformGravity((0.0, 0.0, -0.1), reference_position_m=(0.0, 0.0, 1.0)),
    )
    bundle = sample_motion_2d(spec, GroundedPlaneBackend(), visual_frames=2, resolution=9)
    assert abs(bundle["motion_2d"]["plane_normal"][0]) > 1.0 - 1e-12
    assert "<canvas" in render_html(bundle, title="Nearly axial held charge yz plane")

    # 2. Sub-4*ulp held-charge offset (5e-17 m) also selects the exact yz plane via fallback
    prob_tiny = ElectrostaticProblem(
        (
            PointCharge(1e-6, (0.0, 0.0, 1.0)),
            PointCharge(-9e-6 / 40.0, (0.0, 5e-17, 2.0)),
        ),
        (plane0,),
    )
    spec_tiny = Motion2DSpec(
        prob_tiny, 0, 1.0, (0.0, 0.0, 0.0), 0.1, 0.05,
        gravity=UniformGravity((0.0, 0.0, -0.1), reference_position_m=(0.0, 0.0, 1.0)),
    )
    bundle_tiny = sample_motion_2d(spec_tiny, GroundedPlaneBackend(), visual_frames=2, resolution=9)
    assert abs(bundle_tiny["motion_2d"]["plane_normal"][0]) > 1.0 - 1e-12
    assert "<canvas" in render_html(bundle_tiny, title="Tiny held charge yz plane")


def test_m45_sol_v18_exact_displacement_before_plane_cross_product():
    """Regression suite for Sol v18: exact rational displacement prevents float64 subtraction tilt."""
    plane_tilted = ConductingPlane((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), Grounded())
    r = 3.0 * (2.0 ** -55)
    d = 2.0 ** -43
    launch = (r - d, r + d, r)
    prob = ElectrostaticProblem(
        (
            PointCharge(1e-8, launch),
            PointCharge(-1e-8, (1.0, 1.0, 1.0)),
        ),
        (plane_tilted,),
    )
    spec = Motion2DSpec(
        prob, 0, 1.0, (0.0, 0.0, 0.0), 0.1, 0.05,
        gravity=UniformGravity((0.1, -0.1, 0.0), reference_position_m=launch),
    )
    bundle = sample_motion_2d(spec, GroundedPlaneBackend(), visual_frames=2, resolution=9)
    # Exact plane normal is proportional to (1, 1, -2) / sqrt(6)
    n = bundle["motion_2d"]["plane_normal"]
    expected = (1.0 / sqrt(6.0), 1.0 / sqrt(6.0), -2.0 / sqrt(6.0))
    dot_abs = abs(sum(a * b for a, b in zip(n, expected)))
    assert dot_abs > 1.0 - 1e-12
    assert "<canvas" in render_html(bundle, title="Exact coplanar tilted plane")

    # Matching exterior-sphere and spherical-cavity geometry checks with exact rational displacements
    from conductor_sim.motion_2d import _geometry as _m2d_geometry

    sphere = SolidSphere(radius_m=0.5, constraint=Grounded(), center_m=(-1.0, -1.0, -1.0))
    prob_sph = ElectrostaticProblem(
        (
            PointCharge(1e-8, launch),
            PointCharge(-1e-8, (1.0, 1.0, 1.0)),
        ),
        (sphere,),
    )
    spec_sph = Motion2DSpec(
        prob_sph, 0, 1.0, (0.0, 0.0, 0.0), 0.1, 0.05,
        gravity=UniformGravity((0.1, -0.1, 0.0), reference_position_m=launch),
    )
    n_sph = _m2d_geometry(spec_sph).normal
    assert abs(sum(a * b for a, b in zip(n_sph, expected))) > 1.0 - 1e-12

    shell = SphericalShell(inner_radius_m=4.0, outer_radius_m=5.0, constraint=Grounded(), center_m=(-1.0, -1.0, -1.0))
    prob_cav = ElectrostaticProblem(
        (
            PointCharge(1e-8, launch),
            PointCharge(-1e-8, (1.0, 1.0, 1.0)),
        ),
        (shell,),
    )
    spec_cav = Motion2DSpec(
        prob_cav, 0, 1.0, (0.0, 0.0, 0.0), 0.1, 0.05,
        gravity=UniformGravity((0.1, -0.1, 0.0), reference_position_m=launch),
    )
    n_cav = _m2d_geometry(spec_cav).normal
    assert abs(sum(a * b for a, b in zip(n_cav, expected))) > 1.0 - 1e-12


def test_m45_sol_v19_exact_launch_coplanarity_check_for_tiny_offset_sphere():
    """Regression suite for Sol v19: exact rational launch/source coplanarity matches certified normal."""
    sphere = SolidSphere(radius_m=1e-9, constraint=Grounded(), center_m=(-1.0, -1.0, -1.0))
    launch = (-5.0 * (2.0 ** -55), -6.0 * (2.0 ** -55), -3.0 * (2.0 ** -55))
    prob = ElectrostaticProblem((PointCharge(1e-12, launch),), (sphere,))
    spec = Motion2DSpec(
        prob, 0, 1.0, (0.0, 0.0, 0.0), 0.1, 0.05,
        gravity=UniformGravity((0.0, 1.0, 0.0), reference_position_m=launch),
    )
    bundle = sample_motion_2d(spec, ExternalGroundedSphereBackend(), visual_frames=2, resolution=9)
    assert bundle["motion_2d"]["termination"] == "duration"
    assert len(bundle["motion_2d"]["samples"]) == 3
    assert "<canvas" in render_html(bundle, title="Sol v19 tiny sphere exact coplanarity")


def test_m45_sol_v20_out_of_plane_held_charge_and_floorless_transverse_force():
    """Regression suite for Sol v20: out-of-plane held charge near moving charge & floorless transverse force."""
    shell = SphericalShell(inner_radius_m=2.0, outer_radius_m=3.0, constraint=Grounded(), center_m=(0.0, 0.0, 0.0))

    # 1. Exact Sol v20 reproducer: held charge at (1, 0, 5e-10) is directly out of xy plane above (1, 0, 0)
    prob = ElectrostaticProblem(
        (
            PointCharge(1e-27, (1.0, 0.0, 0.0)),
            PointCharge(1e-27, (1.0, 0.0, 5e-10)),
        ),
        (shell,),
    )
    with pytest.raises(UnsupportedProblem, match="invariant plane"):
        Motion2DSpec(
            prob, 0, 1e-20, (0.0, 1e-12, 0.0), 0.01, 0.005,
            charge_guard_m=1e-12,
            gravity=UniformGravity((0.0, 1e-8, 0.0)),
        )

    # 2. Even a 1e-16 m out-of-plane offset directly above moving charge (sin_origin=1e-16, sin_rel=1.0) is refused
    prob_sub_eps = ElectrostaticProblem(
        (
            PointCharge(1e-27, (1.0, 0.0, 0.0)),
            PointCharge(1e-27, (1.0, 0.0, 1e-16)),
        ),
        (shell,),
    )
    with pytest.raises(UnsupportedProblem, match="invariant plane"):
        Motion2DSpec(
            prob_sub_eps, 0, 1e-20, (0.0, 1e-12, 0.0), 0.01, 0.005,
            charge_guard_m=1e-18,
            gravity=UniformGravity((0.0, 1e-8, 0.0)),
        )

    # 3. Floorless transverse-force check rejects 1e-26 N out-of-plane force (below legacy 1e-20 N floor)
    from conductor_sim.motion_2d import integrate_motion_2d

    prob_coplanar = ElectrostaticProblem((PointCharge(1e-27, (1.0, 0.0, 0.0)),), (shell,))
    spec_ok = Motion2DSpec(
        prob_coplanar, 0, 1e-20, (0.0, 1e-12, 0.0), 0.01, 0.005,
        gravity=UniformGravity((0.0, 1e-8, 0.0)),
    )
    # Valid coplanar tiny-charge run completes cleanly
    b_ok = sample_motion_2d(spec_ok, MultiImageShellBackend(), visual_frames=2, resolution=9)
    assert b_ok["motion_2d"]["termination"] == "duration"
    assert "<canvas" in render_html(b_ok, title="Valid tiny-charge coplanar shell")

    # Viewer validation rejects forged tiny (1e-26 N) transverse force or tiny (1e-15 m/s) transverse velocity
    import copy
    tampered_f = copy.deepcopy(b_ok)
    tampered_f["motion_2d"]["samples"][0]["electric_force_n"] = [0.0, 0.0, -3.595e-26]
    tampered_f["motion_2d"]["samples"][0]["force_n"] = [0.0, 1e-28, -3.595e-26]
    with pytest.raises(ValueError, match="planar symmetry"):
        render_html(tampered_f, title="Tampered tiny transverse force")

    tampered_v = copy.deepcopy(b_ok)
    tampered_v["motion_2d"]["samples"][1]["velocity_m_s"] = [0.0, 1e-12, -1e-15]
    with pytest.raises(ValueError, match="planar symmetry"):
        render_html(tampered_v, title="Tampered tiny transverse velocity")


def test_m45_sol_v21_turning_point_velocity_and_saved_held_charge_parity():
    """Regression suite for both Sol v21 findings: turning-point roundoff & exact viewer held-charge check."""
    import copy

    shell = SphericalShell(inner_radius_m=2.0, outer_radius_m=3.0, constraint=Grounded(), center_m=(0.0, 0.0, 0.0))

    # 1. Exact coplanar turning-point trajectory completes, samples, and renders cleanly
    prob_tp = ElectrostaticProblem((PointCharge(1e-30, (1.0, 0.0, 0.0)),), (shell,))
    spec_tp = Motion2DSpec(
        prob_tp, 0, 1e-20, (0.0, 1e-8, 3e-8), 1.0, 0.5,
        gravity=UniformGravity((0.0, -1e-8, -3e-8)),
    )
    b_tp = sample_motion_2d(spec_tp, MultiImageShellBackend(), visual_frames=2, resolution=9)
    assert b_tp["motion_2d"]["termination"] == "duration"
    assert "<canvas" in render_html(b_tp, title="Sol v21 turning point")

    # 2. Saved viewer rejects held charge at (1, 5e-9, 1e-16) matching native Motion2DSpec refusal
    prob_held_bad = ElectrostaticProblem(
        (
            PointCharge(1e-27, (1.0, 0.0, 0.0)),
            PointCharge(1e-27, (1.0, 5e-9, 1e-16)),
        ),
        (shell,),
    )
    with pytest.raises(UnsupportedProblem, match="invariant plane"):
        Motion2DSpec(
            prob_held_bad, 0, 1e-20, (0.0, 1e-12, 0.0), 0.01, 0.005,
            charge_guard_m=1e-12,
            gravity=UniformGravity((0.0, 1e-8, 0.0)),
        )

    prob_held_ok = ElectrostaticProblem(
        (
            PointCharge(1e-27, (1.0, 0.0, 0.0)),
            PointCharge(1e-27, (1.0, 5e-9, 0.0)),
        ),
        (shell,),
    )
    spec_held_ok = Motion2DSpec(
        prob_held_ok, 0, 1e-20, (0.0, 1e-12, 0.0), 0.01, 0.005,
        charge_guard_m=1e-12,
        gravity=UniformGravity((0.0, 1e-8, 0.0)),
    )
    b_held_ok = sample_motion_2d(spec_held_ok, MultiImageShellBackend(), visual_frames=2, resolution=9)
    assert "<canvas" in render_html(b_held_ok, title="Valid coplanar held charge")

    tampered_held = copy.deepcopy(b_held_ok)
    tampered_held["motion_2d"]["stationary_charges"][0]["position_m"] = [1.0, 5e-9, 1e-16]
    for frame in tampered_held["frames"]:
        for ch in frame["charges"]:
            if ch["position_m"] == [1.0, 5e-9, 0.0]:
                ch["position_m"] = [1.0, 5e-9, 1e-16]
    with pytest.raises(ValueError, match="fixed charge lies outside"):
        render_html(tampered_held, title="Tampered held charge 1e-16 z offset")


def test_m45_sol_v22_boundary_normal_parity_and_timestamp_immune_velocity_check():
    """Regression suite for both Sol v22 findings."""
    import copy

    shell = SphericalShell(inner_radius_m=2.0, outer_radius_m=3.0, constraint=Grounded(), center_m=(0.0, 0.0, 0.0))

    # 1. Exact Sol v22 boundary case: native Motion2DSpec and viewer agree (both refuse)
    prob_boundary = ElectrostaticProblem(
        (
            PointCharge(1e-27, (1.0, 0.0, 0.0)),
            PointCharge(1e-27, (1.0, 1.23e-8, 3.689999999999825e-8)),
        ),
        (shell,),
    )
    with pytest.raises(UnsupportedProblem, match="invariant plane"):
        Motion2DSpec(
            prob_boundary, 0, 1e-20, (0.0, 1e-12, 3e-12), 0.01, 0.005,
            charge_guard_m=1e-12,
            gravity=UniformGravity((0.0, 1e-8, 3e-8)),
        )
    # Truly coplanar counterpart passes native integration, sampling, and viewer rendering
    prob_coplanar = ElectrostaticProblem(
        (
            PointCharge(1e-27, (1.0, 0.0, 0.0)),
            PointCharge(1e-27, (1.0, 1.23e-8, 3.69e-8)),
        ),
        (shell,),
    )
    spec_coplanar = Motion2DSpec(
        prob_coplanar, 0, 1e-20, (0.0, 1e-12, 3e-12), 0.01, 0.005,
        charge_guard_m=1e-12,
        gravity=UniformGravity((0.0, 1e-8, 3e-8)),
    )
    b_coplanar = sample_motion_2d(spec_coplanar, MultiImageShellBackend(), visual_frames=2, resolution=9)
    assert "<canvas" in render_html(b_coplanar, title="Coplanar tilted held charge")

    # 2. Forging final timestamp to 1e14 s cannot disable midpoint out-of-plane velocity check
    prob_tp = ElectrostaticProblem((PointCharge(1e-30, (1.0, 0.0, 0.0)),), (shell,))
    spec_tp = Motion2DSpec(
        prob_tp, 0, 1e-20, (0.0, 1e-8, 0.0), 1.0, 0.5,
        gravity=UniformGravity((0.0, -1e-8, 0.0)),
    )
    b_tp = sample_motion_2d(spec_tp, MultiImageShellBackend(), visual_frames=2, resolution=9)
    assert "<canvas" in render_html(b_tp, title="Valid xy turning point")

    tampered = copy.deepcopy(b_tp)
    vx, vy, vz = tampered["motion_2d"]["samples"][1]["velocity_m_s"]
    tampered["motion_2d"]["samples"][1]["velocity_m_s"] = [vx, 0.0, vy]
    tampered["motion_2d"]["samples"][-1]["time_s"] = 1e14
    with pytest.raises(ValueError, match="planar symmetry"):
        render_html(tampered, title="Timestamp-inflated out-of-plane velocity")


def test_m45_sol_v23_forged_acceleration_velocity_check_and_reverse_boundary_parity():
    """Regression suite for both Sol v23 findings."""
    import copy

    shell = SphericalShell(inner_radius_m=2.0, outer_radius_m=3.0, constraint=Grounded(), center_m=(0.0, 0.0, 0.0))

    # 1. Forging electric force/acceleration on another graph-only row cannot hide out-of-plane velocity
    prob_tp = ElectrostaticProblem((PointCharge(1e-30, (1.0, 0.0, 0.0)),), (shell,))
    spec_tp = Motion2DSpec(
        prob_tp, 0, 1e-20, (0.0, 1e-8, 0.0), 1.0, 0.25,
        gravity=UniformGravity((0.0, -1e-8, 0.0)),
    )
    b_tp = sample_motion_2d(spec_tp, MultiImageShellBackend(), visual_frames=2, resolution=9)
    assert "<canvas" in render_html(b_tp, title="Valid 0.25s turning point")

    tampered = copy.deepcopy(b_tp)
    # Move 0.25 s row's y-velocity into z
    vx, vy, _ = tampered["motion_2d"]["samples"][1]["velocity_m_s"]
    tampered["motion_2d"]["samples"][1]["velocity_m_s"] = [vx, 0.0, vy]
    # Inflate a different graph-only row's in-plane electric force/acceleration to 0.1 N
    r2 = tampered["motion_2d"]["samples"][2]
    gf = r2["gravitational_force_n"]
    r2["electric_force_n"] = [0.1, 0.0, 0.0]
    r2["force_n"] = [0.1 + gf[0], gf[1], gf[2]]
    r2["acceleration_m_s2"] = [f / 1e-20 for f in r2["force_n"]]
    with pytest.raises(ValueError, match="planar symmetry"):
        render_html(tampered, title="Acceleration-inflated out-of-plane velocity")

    # 2. Reverse-direction held-charge boundary parity: native Motion2DSpec and viewer agree identically
    prob_rev = ElectrostaticProblem(
        (
            PointCharge(1e-27, (1.0, 0.0, 0.0)),
            PointCharge(1e-27, (1.0, 1.229999999999999e-8, 3.690000000000172e-8)),
        ),
        (shell,),
    )
    spec_rev = Motion2DSpec(
        prob_rev, 0, 1e-20, (0.0, 1e-12, 3e-12), 0.01, 0.005,
        charge_guard_m=1e-12,
        gravity=UniformGravity((0.0, 1e-8, 3e-8)),
    )
    b_rev = sample_motion_2d(spec_rev, MultiImageShellBackend(), visual_frames=2, resolution=9)
    assert "<canvas" in render_html(b_rev, title="Reverse boundary parity")


def test_m45_sol_v24_coordinate_plane_zero_floor_and_causal_tilted_turning_point():
    """Regression suite for Sol v24: coordinate-plane velocity has zero floor; tilted turning point stays valid."""
    import copy

    shell = SphericalShell(inner_radius_m=2.0, outer_radius_m=3.0, constraint=Grounded(), center_m=(0.0, 0.0, 0.0))

    # 1. High-speed XY-plane turning-point bypass: moving 7.28e-9 m/s from Y into Z must be rejected
    plane = ConductingPlane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), Grounded())
    prob_xy = ElectrostaticProblem((PointCharge(1e-30, (1.0, 0.0, 0.0)),), (plane,))
    spec_xy = Motion2DSpec(
        prob_xy, 0, 1e-20, (0.0, 1e6, 0.0), 1.0, 0.5,
        gravity=UniformGravity((0.0, -1e6, 0.0)),
    )
    b_xy = sample_motion_2d(spec_xy, GroundedPlaneBackend(), visual_frames=2, resolution=9)
    assert "<canvas" in render_html(b_xy, title="High-speed XY turning point")

    tampered_xy = copy.deepcopy(b_xy)
    r_mid = tampered_xy["motion_2d"]["samples"][1]
    r_mid["velocity_m_s"] = [0.0, 0.0, 7.28e-9]
    k_mid = 0.5 * 1e-20 * (7.28e-9 ** 2)
    r_mid["kinetic_energy_j"] = k_mid
    e0 = tampered_xy["motion_2d"]["samples"][0]["total_energy_j"]
    r_mid["electric_interaction_energy_j"] = e0 - k_mid - r_mid["gravitational_potential_energy_j"]
    r_mid["interaction_energy_j"] = r_mid["electric_interaction_energy_j"] + r_mid["gravitational_potential_energy_j"]
    r_mid["total_energy_j"] = e0
    r_mid["energy_drift_j"] = 0.0
    with pytest.raises(ValueError, match="planar symmetry"):
        render_html(tampered_xy, title="High-speed XY z-velocity bypass")

    # 2. v23 fixture: inflating another graph-only row's in-plane speed + energy ledger cannot hide z-velocity
    prob_tp = ElectrostaticProblem((PointCharge(1e-30, (1.0, 0.0, 0.0)),), (shell,))
    spec_tp = Motion2DSpec(
        prob_tp, 0, 1e-20, (0.0, 1e-8, 0.0), 1.0, 0.25,
        gravity=UniformGravity((0.0, -1e-8, 0.0)),
    )
    b_tp = sample_motion_2d(spec_tp, MultiImageShellBackend(), visual_frames=2, resolution=9)
    tampered_v23 = copy.deepcopy(b_tp)
    vx, vy, _ = tampered_v23["motion_2d"]["samples"][1]["velocity_m_s"]
    tampered_v23["motion_2d"]["samples"][1]["velocity_m_s"] = [vx, 0.0, vy]
    r2 = tampered_v23["motion_2d"]["samples"][2]
    r2["velocity_m_s"] = [0.0, 1e6, 0.0]
    k2 = 0.5 * 1e-20 * (1e6 ** 2)
    r2["kinetic_energy_j"] = k2
    e0_tp = tampered_v23["motion_2d"]["samples"][0]["total_energy_j"]
    r2["electric_interaction_energy_j"] = e0_tp - k2 - r2["gravitational_potential_energy_j"]
    r2["interaction_energy_j"] = r2["electric_interaction_energy_j"] + r2["gravitational_potential_energy_j"]
    r2["total_energy_j"] = e0_tp
    r2["energy_drift_j"] = 0.0
    with pytest.raises(ValueError, match="planar symmetry"):
        render_html(tampered_v23, title="Inflated in-plane speed bypass")

    # 3. Genuine high-speed tilted-plane turning point still samples and renders cleanly
    spec_tilt = Motion2DSpec(
        prob_xy, 0, 1e-20, (0.0, 1e6, 3e6), 1.0, 0.5,
        gravity=UniformGravity((0.0, -1e6, -3e6)),
    )
    b_tilt = sample_motion_2d(spec_tilt, GroundedPlaneBackend(), visual_frames=2, resolution=9)
    assert "<canvas" in render_html(b_tilt, title="High-speed tilted turning point")


def test_m45_sol_v25_multi_step_tilted_plane_turning_point_compensated_rk4():
    """Regression suite for Sol v25: 100-step and 1000-step tilted-plane turning points stay within ULP bound."""
    plane = ConductingPlane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), Grounded())
    prob = ElectrostaticProblem((PointCharge(1e-30, (1.0, 0.0, 0.0)),), (plane,))

    for dt in (0.01, 0.001):
        spec = Motion2DSpec(
            prob, 0, 1e-20, (0.0, 500.0, 117.0), 1.0, dt,
            gravity=UniformGravity((0.0, -500.0, -117.0)),
        )
        bundle = sample_motion_2d(spec, GroundedPlaneBackend(), visual_frames=2, resolution=9)
        assert bundle["motion_2d"]["termination"] == "duration"
        assert "<canvas" in render_html(bundle, title=f"Sol v25 tilted turning point dt={dt}")


def test_m45_sol_v26_translated_oblique_plane_and_shared_velocity_parity():
    """Regression suite for both Sol v26 findings."""
    from math import ulp

    # 1. Translated oblique plane at (0, 1e6, 2e6) completes, samples, and renders cleanly
    plane_trans = ConductingPlane((0.0, 1e6, 2e6), (1.0, 2.0, 2.0), Grounded())
    prob_trans = ElectrostaticProblem((PointCharge(1e-6, (1.0, 1000002.0, 2000002.0)),), (plane_trans,))
    spec_trans = Motion2DSpec(
        prob_trans, 0, 1e-4, (1.0, 2.0, 2.0), 0.6313519768873583, 0.02,
        gravity=UniformGravity((-1.0, -2.0, -2.0), reference_position_m=(0.0, 1e6, 2e6)),
    )
    b_trans = sample_motion_2d(spec_trans, GroundedPlaneBackend(), visual_frames=2, resolution=9)
    assert b_trans["motion_2d"]["termination"] == "duration"
    assert len(b_trans["motion_2d"]["samples"]) == 33
    assert "<canvas" in render_html(b_trans, title="Sol v26 translated oblique plane")

    # 2. Qualification, native integration, and playback agree on 50-ULP tilted g_z
    plane_x = ConductingPlane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), Grounded())
    prob_x = ElectrostaticProblem((PointCharge(1e-30, (1.0, 0.0, 0.0)),), (plane_x,))
    spec_50ulp = Motion2DSpec(
        prob_x, 0, 1e-20, (0.0, 500.0, 117.0), 1.0, 0.01,
        gravity=UniformGravity((0.0, -500.0, -(117.0 + 50.0 * ulp(117.0)))),
    )
    b_50ulp = sample_motion_2d(spec_50ulp, GroundedPlaneBackend(), visual_frames=2, resolution=9)
    assert b_50ulp["motion_2d"]["termination"] == "duration"
    assert "<canvas" in render_html(b_50ulp, title="Sol v26 50-ULP g_z parity")

    # And a larger drift (e.g. 500 ULPs of v_n_prefix over 1 s) is refused consistently by native integrate_motion_2d
    from conductor_sim.motion_2d import integrate_motion_2d
    spec_500ulp = Motion2DSpec(
        prob_x, 0, 1e-20, (0.0, 500.0, 117.0), 1.0, 0.01,
        gravity=UniformGravity((0.0, -500.0, -(117.0 + 500.0 * ulp(117.0)))),
    )
    with pytest.raises(UnsupportedProblem, match="invariant plane"):
        integrate_motion_2d(spec_500ulp, GroundedPlaneBackend())


def test_m45_sol_v27_translated_plane_energy_and_tangential_anchor_invariance():
    """Regression suite for both Sol v27 findings."""
    from conductor_sim.motion_2d import integrate_motion_2d
    from conductor_sim.observables import force_energy

    # 1. Translated oblique plane force_energy & 2D trajectory match analytic -k q^2 / (4 h)
    plane_t = ConductingPlane((0.0, 1e6, 2e6), (1.0, 2.0, 2.0), Grounded())
    q_val = 1e-18
    pos = (1e-10, 1e6, 2e6)
    prob_t = ElectrostaticProblem((PointCharge(q_val, pos),), (plane_t,))
    sol_t = GroundedPlaneBackend().solve(prob_t)
    h = plane_t.signed_distance_m(pos)
    expected_u = -q_val * q_val / (16.0 * pi * EPSILON_0 * h)
    fe_t = force_energy(sol_t)
    assert isclose(sol_t.interaction_energy_j, expected_u, rel_tol=1e-12, abs_tol=0.0)
    assert isclose(fe_t.energy.induced_j, expected_u, rel_tol=1e-12, abs_tol=0.0)
    spec_t = Motion2DSpec(prob_t, 0, 1.0, (0.0, 0.0, 0.0), 1e-6, 5e-7)
    res_t = integrate_motion_2d(spec_t, GroundedPlaneBackend())
    assert isclose(res_t.samples[0].interaction_energy_j, expected_u, rel_tol=1e-12, abs_tol=0.0)
    # Gravity-coupled translated oblique plane at resolvable displacement also matches analytic U_elec
    pos_g = (1.0, 1000002.0, 2000002.0)
    prob_tg = ElectrostaticProblem((PointCharge(1e-6, pos_g),), (plane_t,))
    h_g = plane_t.signed_distance_m(pos_g)
    expected_ug = -(1e-6) ** 2 / (16.0 * pi * EPSILON_0 * h_g)
    spec_tg = Motion2DSpec(
        prob_tg, 0, 1e-4, (1.0, 2.0, 2.0), 0.1, 0.05,
        gravity=UniformGravity((-1.0, -2.0, -2.0), reference_position_m=(0.0, 1e6, 2e6)),
    )
    res_tg = integrate_motion_2d(spec_tg, GroundedPlaneBackend())
    assert isclose(res_tg.samples[0].electric_interaction_energy_j, expected_ug, rel_tol=1e-12, abs_tol=0.0)

    # 2. Tangential anchor shift does NOT loosen surface_charge_density_at for off-plane queries
    for anchor, normal, charge_pos, half_pos in (
        ((0.0, 1e6, 0.0), (1.0, 0.0, 0.0), (1e-9, 0.0, 0.0), (5e-10, 0.0, 0.0)),
        ((0.0, 1e6, -1e6), (1.0, 2.0, 2.0), (1e-9 / 3.0, 2e-9 / 3.0, 2e-9 / 3.0), (5e-10 / 3.0, 1e-9 / 3.0, 1e-9 / 3.0)),
    ):
        pl = ConductingPlane(anchor, normal, Grounded())
        s_pl = GroundedPlaneBackend().solve(ElectrostaticProblem((PointCharge(1e-9, charge_pos),), (pl,)))
        with pytest.raises(ValueError, match="conducting plane"):
            s_pl.surface_charge_density_at("outer", half_pos)
        with pytest.raises(ValueError, match="conducting plane"):
            s_pl.surface_charge_density_at("outer", charge_pos)


def test_m45_sol_v28_exact_signed_distance_anchor_invariance_and_resolvable_off_plane():
    """Regression suite for both Sol v28 findings."""
    from conductor_sim.motion_2d import integrate_motion_2d
    from conductor_sim.observables import force_energy

    # 1. Exact tangential-anchor invariance of signed_distance_m, force_energy, gravity-2D, and sigma
    p_orig = ConductingPlane((0.0, 0.0, 0.0), (1.0, 2.0, 2.0), Grounded())
    p_shift = ConductingPlane((0.0, 1e7, -1e7), (1.0, 2.0, 2.0), Grounded())
    q_val = 1e-18
    q_pos = (3e-10, 2.0 ** -33, 0.0)
    h_orig = p_orig.signed_distance_m(q_pos)
    h_shift = p_shift.signed_distance_m(q_pos)
    assert h_shift == h_orig
    expected_u = -q_val * q_val / (16.0 * pi * EPSILON_0 * h_orig)

    sol_shift = GroundedPlaneBackend().solve(ElectrostaticProblem((PointCharge(q_val, q_pos),), (p_shift,)))
    fe_shift = force_energy(sol_shift)
    assert isclose(sol_shift.interaction_energy_j, expected_u, rel_tol=1e-12, abs_tol=0.0)
    assert isclose(fe_shift.energy.induced_j, expected_u, rel_tol=1e-12, abs_tol=0.0)

    spec_shift = Motion2DSpec(
        ElectrostaticProblem((PointCharge(q_val, q_pos),), (p_shift,)),
        0, 1.0, (0.0, 0.0, 0.0), 1e-3, 5e-4,
        gravity=UniformGravity((0.0, 0.0, 0.0)),
    )
    res_shift = integrate_motion_2d(spec_shift, GroundedPlaneBackend())
    assert isclose(res_shift.samples[0].electric_interaction_energy_j, expected_u, rel_tol=1e-12, abs_tol=0.0)

    # Exact surface point (-2^-32, 2^-33, 0) accepted; off-plane (0, 2^-33, 0) rejected on shifted anchor
    sol_orig = GroundedPlaneBackend().solve(ElectrostaticProblem((PointCharge(q_val, q_pos),), (p_orig,)))
    surf_pt = (-(2.0 ** -32), 2.0 ** -33, 0.0)
    assert isclose(
        sol_shift.surface_charge_density_at("outer", surf_pt),
        sol_orig.surface_charge_density_at("outer", surf_pt),
        rel_tol=1e-12, abs_tol=0.0,
    )
    with pytest.raises(ValueError, match="conducting plane"):
        sol_shift.surface_charge_density_at("outer", (0.0, 2.0 ** -33, 0.0))

    # 2. Resolvably off-plane query (5e-11, 1e6, 2e6) on anchor (0, 1e6, 2e6) is rejected
    p_trans = ConductingPlane((0.0, 1e6, 2e6), (1.0, 2.0, 2.0), Grounded())
    sol_trans = GroundedPlaneBackend().solve(
        ElectrostaticProblem((PointCharge(1e-6, (1.0, 1000002.0, 2000002.0)),), (p_trans,))
    )
    with pytest.raises(ValueError, match="conducting plane"):
        sol_trans.surface_charge_density_at("outer", (5e-11, 1e6, 2e6))


def test_m45_sol_v29_local_profile_snapping_and_signed_distance_extremes():
    """Regression suite for both Sol v29 findings."""
    # 1. Surface-density profile on anchor (0, 1e6, 2e6) with tiny normal x=1e-14 stays on x=0 slice
    plane_tiny_x = ConductingPlane((0.0, 1e6, 2e6), (1e-14, 1.0, 1.0), Grounded())
    # Place charge at distance sqrt(2) above anchor on x=0 slice
    q_pos = (0.0, 1e6 + 1.0, 2e6 + 1.0)
    prob_tx = ElectrostaticProblem((PointCharge(1e-6, q_pos),), (plane_tiny_x,))
    spec_tx = Motion2DSpec(
        prob_tx, 0, 1e-3, (0.0, 1.0, 1.0), 0.1, 0.05,
        gravity=UniformGravity((0.0, -1.0, -1.0), reference_position_m=(0.0, 1e6, 2e6)),
    )
    b_tx = sample_motion_2d(spec_tx, GroundedPlaneBackend(), visual_frames=2, resolution=9)
    assert "<canvas" in render_html(b_tx, title="Sol v29 tiny-x oblique plane")
    # Verify profile sigma values match analytical -q*h / (2*pi*(h^2 + r^2)^(3/2)) on the x=0 slice!
    profile = b_tx["frames"][0]["profiles"][0]
    h0 = plane_tiny_x.signed_distance_m(q_pos)
    sol_tx = GroundedPlaneBackend().solve(prob_tx)
    peak_analytic = sol_tx.surface_charge_density("outer", 0.0)
    assert min(profile["sigma_c_per_m2"]) < 0.25 * peak_analytic  # strong negative peak near footpoint, not 1e-19!
    # Check exact analytical agreement at every profile coordinate relative to the footpoint
    from conductor_sim.geometry import slice_geometry
    from conductor_sim.visualization import SliceSpec
    sp = SliceSpec(**b_tx["frames"][0]["plane"])
    geom = slice_geometry(plane_tiny_x, sp)
    nx, ny = geom["normal_xy"]
    norm_xy = sqrt(nx * nx + ny * ny)
    tx, ty = (-ny / norm_xy, nx / norm_xy)
    x0, y0 = geom["point_m"]
    for d_coord, sigma_val in zip(profile["coordinate_m"], profile["sigma_c_per_m2"]):
        pt_on_slice = sp.world((x0 + d_coord * tx, y0 + d_coord * ty))
        dist_3d = sqrt(sum((pt_on_slice[i] - q_pos[i]) ** 2 for i in range(3)))
        expected_sigma = -1e-6 * h0 / (2.0 * pi * dist_3d ** 3)
        assert isclose(sigma_val, expected_sigma, rel_tol=1e-6, abs_tol=0.0)

    # 2a. signed_distance_m handles opposing 1e308 coordinates without ValueError in fsum
    p_huge = ConductingPlane((-1e308, 1e308, 0.0), (1.0, 1.0, 0.0), Grounded())
    assert p_huge.signed_distance_m((1e308, -1e308, 0.0)) == 0.0

    # 2b. signed_distance_m preserves subnormal nonzero distances (no premature underflow to 0)
    p_sub = ConductingPlane((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), Grounded())
    assert p_sub.signed_distance_m((5e-324, 5e-324, 0.0)) == 5e-324


def test_m45_sol_v30_shallow_tilt_profile_and_resolvable_small_normal_axis():
    """Regression suite for both Sol v30 findings."""
    from conductor_sim.visualization import SliceSpec, sample_sweep

    # 1. Shallowly tilted plane (1, 0.05, 0) at anchor (1e6, 0, 0): XY slice sweep matches analytical sigma
    p_shallow = ConductingPlane((1e6, 0.0, 0.0), (1.0, 0.05, 0.0), Grounded())
    q_shallow = (1e6 + 1.0, 0.05, 0.0)
    prob_shallow = ElectrostaticProblem((PointCharge(1e-6, q_shallow),), (p_shallow,))
    sp_xy = SliceSpec(
        origin_m=(1e6 + 0.5, 0.0, 0.0), u=(1.0, 0.0, 0.0), v=(0.0, 1.0, 0.0),
        x_range_m=(-1.5, 1.5), y_range_m=(-1.5, 1.5), resolution=9,
    )
    bundle_s = sample_sweep([("shallow", prob_shallow)], GroundedPlaneBackend(), plane=sp_xy, resolution=9)
    prof_s = bundle_s["frames"][0]["profiles"][0]
    assert len(prof_s["sigma_c_per_m2"]) == 91
    h_s = p_shallow.signed_distance_m(q_shallow)
    from conductor_sim.geometry import slice_geometry
    geom_s = slice_geometry(p_shallow, sp_xy)
    nx, ny = geom_s["normal_xy"]
    nxy = sqrt(nx * nx + ny * ny)
    tx, ty = (-ny / nxy, nx / nxy)
    x0, y0 = geom_s["point_m"]
    for d_coord, sigma_val in zip(prof_s["coordinate_m"], prof_s["sigma_c_per_m2"]):
        pt_s = sp_xy.world((x0 + d_coord * tx, y0 + d_coord * ty))
        dist_s = sqrt(sum((pt_s[i] - q_shallow[i]) ** 2 for i in range(3)))
        expected_s = -1e-6 * h_s / (2.0 * pi * dist_s ** 3)
        assert isclose(sigma_val, expected_s, rel_tol=1e-6, abs_tol=0.0)

    # 2. Resolvably off-plane point (1e-9, 1e6, 2e6) on normal (0.05, 1, 1) at anchor (0, 1e6, 2e6) is rejected
    p_05 = ConductingPlane((0.0, 1e6, 2e6), (0.05, 1.0, 1.0), Grounded())
    sol_05 = GroundedPlaneBackend().solve(
        ElectrostaticProblem((PointCharge(1e-6, (0.05, 1e6 + 1.0, 2e6 + 1.0)),), (p_05,))
    )
    with pytest.raises(ValueError, match="conducting plane"):
        sol_05.surface_charge_density_at("outer", (1e-9, 1e6, 2e6))


def test_m45_sol_v31_narrow_slice_sweep_and_adjacent_float_off_plane():
    """Regression suite for both Sol v31 findings."""
    from conductor_sim.visualization import SliceSpec, sample_sweep

    # 1. Narrow XY slice [-1e-4, 1e-4] on plane (1e6, 0, 0) with normal (1, 0.2, 0) returns all 91 samples
    p_narrow = ConductingPlane((1e6, 0.0, 0.0), (1.0, 0.2, 0.0), Grounded())
    q_narrow = (1e6 + 1.0, 0.2, 0.0)
    prob_narrow = ElectrostaticProblem((PointCharge(1e-6, q_narrow),), (p_narrow,))
    sp_narrow = SliceSpec(
        origin_m=(1e6, 0.0, 0.0), u=(1.0, 0.0, 0.0), v=(0.0, 1.0, 0.0),
        x_range_m=(-1e-4, 1e-4), y_range_m=(-1e-4, 1e-4), resolution=9,
    )
    bundle_n = sample_sweep([("narrow", prob_narrow)], GroundedPlaneBackend(), plane=sp_narrow, resolution=9)
    prof_n = bundle_n["frames"][0]["profiles"][0]
    assert len(prof_n["sigma_c_per_m2"]) == 91
    h_n = p_narrow.signed_distance_m(q_narrow)
    expected_center = -1e-6 * h_n / (2.0 * pi * h_n ** 3)
    assert isclose(prof_n["sigma_c_per_m2"][45], expected_center, rel_tol=1e-6, abs_tol=0.0)

    # 2. Adjacent-float off-plane query (1000000.0000000001, 1e6, 0) on (1e6, 1e6, 0) normal (1, 1, 0) is rejected
    p_diag = ConductingPlane((1e6, 1e6, 0.0), (1.0, 1.0, 0.0), Grounded())
    sol_diag = GroundedPlaneBackend().solve(
        ElectrostaticProblem((PointCharge(1e-6, (1e6 + 1.0, 1e6 + 1.0, 0.0)),), (p_diag,))
    )
    with pytest.raises(ValueError, match="conducting plane"):
        sol_diag.surface_charge_density_at("outer", (1000000.0000000001, 1e6, 0.0))


def test_m45_sol_v32_close_charge_far_anchor_narrow_shallow_plane_sweep():
    """Regression suite for Sol v32: close charge on far shallowly tilted plane exports all 91 profile samples."""
    from conductor_sim.visualization import SliceSpec, sample_sweep

    p_far = ConductingPlane((1e7, 1e7, 0.0), (1.0, 0.01, 0.0), Grounded())
    q_far = (1e7 + 0.002, 1e7, 0.0)
    prob_far = ElectrostaticProblem((PointCharge(1e-6, q_far),), (p_far,))
    sp_far = SliceSpec(
        origin_m=(1e7, 1e7, 0.0), u=(1.0, 0.0, 0.0), v=(0.0, 1.0, 0.0),
        x_range_m=(-0.001, 0.001), y_range_m=(-0.001, 0.001), resolution=9,
    )
    bundle_f = sample_sweep([("far-shallow", prob_far)], GroundedPlaneBackend(), plane=sp_far, resolution=9)
    prof_f = bundle_f["frames"][0]["profiles"][0]
    assert len(prof_f["sigma_c_per_m2"]) == 91
    h_f = p_far.signed_distance_m(q_far)
    from conductor_sim.geometry import slice_geometry
    geom_f = slice_geometry(p_far, sp_far)
    nx, ny = geom_f["normal_xy"]
    nxy = sqrt(nx * nx + ny * ny)
    tx, ty = (-ny / nxy, nx / nxy)
    x0, y0 = geom_f["point_m"]
    for d_coord, sigma_val in zip(prof_f["coordinate_m"], prof_f["sigma_c_per_m2"]):
        pt_f = sp_far.world((x0 + d_coord * tx, y0 + d_coord * ty))
        dist_f = sqrt(sum((pt_f[i] - q_far[i]) ** 2 for i in range(3)))
        expected_f = -1e-6 * h_f / (2.0 * pi * dist_f ** 3)
        assert isclose(sigma_val, expected_f, rel_tol=5e-6, abs_tol=0.0)


def test_m45_sol_v33_normal_projection_rejects_off_plane_within_local_limit():
    """Regression suite for Sol v33: off-plane point whose normal projection lies within local_limit is rejected."""
    p_orig_diag = ConductingPlane((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), Grounded())
    sol_od = GroundedPlaneBackend().solve(
        ElectrostaticProblem((PointCharge(1e-6, (1.0, 0.0, 0.0)),), (p_orig_diag,))
    )
    with pytest.raises(ValueError, match="conducting plane"):
        sol_od.surface_charge_density_at("outer", (1.25e-6, 0.0, 0.0))


def test_m45_sol_v34_cancelling_coincident_charges_and_strict_surface_density():
    """Regression suite for both Sol v34 findings."""
    from conductor_sim.visualization import SliceSpec, sample_sweep

    # 1. Adding exact cancelling coincident charges (+1 uC, -1 uC) at (1e7+1e-6, 1e7, 0) does not break sweep
    p_far = ConductingPlane((1e7, 1e7, 0.0), (1.0, 0.01, 0.0), Grounded())
    q_far = (1e7 + 0.002, 1e7, 0.0)
    prob_cancel = ElectrostaticProblem(
        (
            PointCharge(1e-6, q_far),
            PointCharge(1e-6, (1e7 + 1e-6, 1e7, 0.0)),
            PointCharge(-1e-6, (1e7 + 1e-6, 1e7, 0.0)),
        ),
        (p_far,),
    )
    sp_far = SliceSpec(
        origin_m=(1e7, 1e7, 0.0), u=(1.0, 0.0, 0.0), v=(0.0, 1.0, 0.0),
        x_range_m=(-0.001, 0.001), y_range_m=(-0.001, 0.001), resolution=9,
    )
    bundle_c = sample_sweep([("cancel", prob_cancel)], GroundedPlaneBackend(), plane=sp_far, resolution=9)
    prof_c = bundle_c["frames"][0]["profiles"][0]
    assert len(prof_c["sigma_c_per_m2"]) == 91
    h_f = p_far.signed_distance_m(q_far)
    from conductor_sim.geometry import slice_geometry
    geom_c = slice_geometry(p_far, sp_far)
    nx, ny = geom_c["normal_xy"]
    nxy = sqrt(nx * nx + ny * ny)
    tx, ty = (-ny / nxy, nx / nxy)
    x0, y0 = geom_c["point_m"]
    for d_coord, sigma_val in zip(prof_c["coordinate_m"], prof_c["sigma_c_per_m2"]):
        sx, sy = x0 + d_coord * tx, y0 + d_coord * ty
        dx = (sp_far.origin_m[0] - q_far[0]) + sx * sp_far.u[0] + sy * sp_far.v[0]
        dy = (sp_far.origin_m[1] - q_far[1]) + sx * sp_far.u[1] + sy * sp_far.v[1]
        dz = (sp_far.origin_m[2] - q_far[2]) + sx * sp_far.u[2] + sy * sp_far.v[2]
        expected_c = -1e-6 * h_f / (2.0 * pi * (dx * dx + dy * dy + dz * dz) ** 1.5)
        assert isclose(sigma_val, expected_c, rel_tol=1e-12, abs_tol=0.0)

    # 2. Multi-axis ULP off-plane query (1000000.0000000001, 999999.9999999999, 1e6) is strictly rejected
    p_3d = ConductingPlane((1e6, 1e6, 1e6), (1.0, 0.9, 0.3), Grounded())
    sol_3d = GroundedPlaneBackend().solve(
        ElectrostaticProblem((PointCharge(1e-6, (1e6 + 1.0, 1e6, 1e6)),), (p_3d,))
    )
    with pytest.raises(ValueError, match="conducting plane"):
        sol_3d.surface_charge_density_at("outer", (1000000.0000000001, 999999.9999999999, 1000000.0))


def test_m45_sol_v35_exact_offset_profile_at_ulp_height():
    """Regression suite for Sol v35: exact origin+offset profile has zero world-coord rounding at A=1e4 and 1e6."""
    from math import ulp
    from conductor_sim.geometry import slice_geometry
    from conductor_sim.visualization import SliceSpec, sample_sweep

    for anchor_val in (1e4, 1e6):
        u_step = ulp(anchor_val)
        anchor = (anchor_val, anchor_val, 0.0)
        plane_u = ConductingPlane(anchor, (2.0, 1.0, 0.0), Grounded())
        q_pos = (anchor_val - 500.0 * u_step, anchor_val + 1002.0 * u_step, 0.0)
        prob_u = ElectrostaticProblem((PointCharge(1e-12, q_pos),), (plane_u,))
        half_w = 1000.98 * u_step / sqrt(1.6)
        sp_u = SliceSpec(
            origin_m=anchor, u=(1.0, 0.0, 0.0), v=(0.0, 1.0, 0.0),
            x_range_m=(-half_w, half_w), y_range_m=(-half_w, half_w), resolution=9,
        )
        bundle_u = sample_sweep([("ulp-height", prob_u)], GroundedPlaneBackend(), plane=sp_u, resolution=9)
        prof_u = bundle_u["frames"][0]["profiles"][0]
        assert len(prof_u["sigma_c_per_m2"]) == 91
        h_u = plane_u.signed_distance_m(q_pos)
        geom_u = slice_geometry(plane_u, sp_u)
        nx, ny = geom_u["normal_xy"]
        nxy = sqrt(nx * nx + ny * ny)
        tx, ty = (-ny / nxy, nx / nxy)
        x0, y0 = geom_u["point_m"]
        for d_coord, sigma_val in zip(prof_u["coordinate_m"], prof_u["sigma_c_per_m2"]):
            sx, sy = x0 + d_coord * tx, y0 + d_coord * ty
            dx = (anchor[0] - q_pos[0]) + sx
            dy = (anchor[1] - q_pos[1]) + sy
            expected_u = -1e-12 * h_u / (2.0 * pi * (dx * dx + dy * dy) ** 1.5)
            assert isclose(sigma_val, expected_u, rel_tol=1e-12, abs_tol=0.0)


def test_m45_sol_v36_rotated_slice_profile_and_tangential_anchor_region_mask():
    """Regression suite for both Sol v36 findings."""
    from fractions import Fraction
    from conductor_sim.geometry import contains_metal, slice_geometry
    from conductor_sim.visualization import SliceSpec, sample_sweep

    # 1. Rotated slice (1, 1, 0) / (0, 0, 1) with origin (1e6, 1e6, 0) above z=0 plane
    plane_z0 = ConductingPlane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), Grounded())
    # Place 1e-12 C charge 1e-10 m above plane at the endpoint of the rotated slice
    sp_rot = SliceSpec(
        origin_m=(1e6, 1e6, 0.0), u=(1.0, 1.0, 0.0), v=(0.0, 0.0, 1.0),
        x_range_m=(-1.0, 1.0), y_range_m=(-1.0, 1.0), resolution=9,
    )
    geom_rot = slice_geometry(plane_z0, sp_rot)
    nx, ny = geom_rot["normal_xy"]
    nxy = sqrt(nx * nx + ny * ny)
    tx, ty = (-ny / nxy, nx / nxy)
    x0, y0 = geom_rot["point_m"]
    span_rot = sqrt(2.0)
    # Charge near the final sample point (d = +span_rot) at height 1e-10 m
    q_end = (1e6 + span_rot * sp_rot.u[0], 1e6 + span_rot * sp_rot.u[1], 1e-10)
    prob_rot = ElectrostaticProblem((PointCharge(1e-12, q_end),), (plane_z0,))
    bundle_rot = sample_sweep([("rot", prob_rot)], GroundedPlaneBackend(), plane=sp_rot, resolution=9)
    prof_rot = bundle_rot["frames"][0]["profiles"][0]
    for d_coord, sigma_val in zip(prof_rot["coordinate_m"], prof_rot["sigma_c_per_m2"]):
        fx = Fraction(x0) + Fraction(d_coord) * Fraction(tx)
        fy = Fraction(y0) + Fraction(d_coord) * Fraction(ty)
        delta = tuple(
            float(Fraction(sp_rot.origin_m[i]) - Fraction(q_end[i]) + fx * Fraction(sp_rot.u[i]) + fy * Fraction(sp_rot.v[i]))
            for i in range(3)
        )
        expected_rot = -1e-12 * 1e-10 / (2.0 * pi * (sum(c * c for c in delta) ** 1.5))
        assert isclose(sigma_val, expected_rot, rel_tol=1e-12, abs_tol=0.0)

    # 2. Tangential re-anchoring of (2, 1, 0) plane preserves vacuum/metal classification at (1e-11, 0, 0)
    p_orig = ConductingPlane((0.0, 0.0, 0.0), (2.0, 1.0, 0.0), Grounded())
    p_reanch = ConductingPlane((-1e6, 2e6, 0.0), (2.0, 1.0, 0.0), Grounded())
    pt_vac = (1e-11, 0.0, 0.0)
    assert contains_metal(p_orig, pt_vac) is False
    assert contains_metal(p_reanch, pt_vac) is False
    sp_mask = SliceSpec(
        origin_m=pt_vac, u=(1.0, 0.0, 0.0), v=(0.0, 1.0, 0.0),
        x_range_m=(-1e-10, 1e-10), y_range_m=(-1e-10, 1e-10), resolution=9,
    )
    q_m = PointCharge(1e-12, (2e-10, 0.0, 0.0))
    b_orig = sample_sweep([("orig", ElectrostaticProblem((q_m,), (p_orig,)))], GroundedPlaneBackend(), plane=sp_mask, resolution=9)
    b_reanch = sample_sweep([("reanch", ElectrostaticProblem((q_m,), (p_reanch,)))], GroundedPlaneBackend(), plane=sp_mask, resolution=9)
    assert b_orig["frames"][0]["grid"]["region"][4][4] == "vacuum"
    assert b_reanch["frames"][0]["grid"]["region"][4][4] == "vacuum"


def test_m45_sol_v37_exact_slice_intercept_authored_normal_and_boundary_extremes():
    """Regression suite for all three Sol v37 findings."""
    from math import ulp
    from conductor_sim.geometry import PlaneBoundary, contains_metal, physical_boundaries
    from conductor_sim.visualization import SliceSpec, sample_sweep

    # 1. Exact rational slice intercept on rotated slice through origin for x=1e6 plane
    u_1e6 = ulp(1e6)
    p_x1e6 = ConductingPlane((1e6, 0.0, 0.0), (1.0, 0.0, 0.0), Grounded())
    q_1ulp = (1e6 + u_1e6, 1e6, 0.0)
    sp_orig_rot = SliceSpec(
        origin_m=(0.0, 0.0, 0.0), u=(1.0, 1.0, 0.0), v=(0.0, 0.0, 1.0),
        x_range_m=(-1.0, 1.0), y_range_m=(-1.0, 1.0), resolution=9,
    )
    bundle_1e6 = sample_sweep(
        [("rot-x1e6", ElectrostaticProblem((PointCharge(1e-12, q_1ulp),), (p_x1e6,)))],
        GroundedPlaneBackend(), plane=sp_orig_rot, resolution=9,
    )
    prof_1e6 = bundle_1e6["frames"][0]["profiles"][0]
    expected_center = -1e-12 / (2.0 * pi * (u_1e6 ** 2))
    assert isclose(prof_1e6["sigma_c_per_m2"][45], expected_center, rel_tol=1e-12, abs_tol=0.0)

    # 2. Authored (3, 1, 0) normal ratio preserved under tangential re-anchoring (-1e6, 3e6, 0)
    p31_orig = ConductingPlane((0.0, 0.0, 0.0), (3.0, 1.0, 0.0), Grounded())
    p31_shift = ConductingPlane((-1e6, 3e6, 0.0), (3.0, 1.0, 0.0), Grounded())
    pt_pos = (1e-11, 0.0, 0.0)
    pt_neg = (-1e-11, 0.0, 0.0)
    assert contains_metal(p31_orig, pt_pos) is False
    assert contains_metal(p31_shift, pt_pos) is False
    hit_orig = physical_boundaries(p31_orig)[0].first_hit(pt_pos, pt_neg)
    hit_shift = physical_boundaries(p31_shift)[0].first_hit(pt_pos, pt_neg)
    assert isclose(hit_orig, 0.5, rel_tol=1e-12, abs_tol=0.0)
    assert isclose(hit_shift, 0.5, rel_tol=1e-12, abs_tol=0.0)
    sp31 = SliceSpec(
        origin_m=pt_pos, u=(1.0, 0.0, 0.0), v=(0.0, 1.0, 0.0),
        x_range_m=(-1e-10, 1e-10), y_range_m=(-1e-10, 1e-10), resolution=9,
    )
    q31 = PointCharge(1e-12, (2e-10, 0.0, 0.0))
    b31_o = sample_sweep([("o", ElectrostaticProblem((q31,), (p31_orig,)))], GroundedPlaneBackend(), plane=sp31, resolution=9)
    b31_s = sample_sweep([("s", ElectrostaticProblem((q31,), (p31_shift,)))], GroundedPlaneBackend(), plane=sp31, resolution=9)
    assert b31_o["frames"][0]["grid"]["region"][4][4] == "vacuum"
    assert b31_s["frames"][0]["grid"]["region"][4][4] == "vacuum"

    # 3. PlaneBoundary.first_hit avoids underflow and overflow at finite extremes
    pb_x0 = PlaneBoundary((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert pb_x0.first_hit((1e-160, 0.0, 0.0), (1e-200, 0.0, 0.0)) is None
    assert pb_x0.first_hit((1e308, 0.0, 0.0), (-1e308, 0.0, 0.0)) == 0.5


def test_m45_sol_v38_plane_replace_extreme_crossings_shallow_slice_and_equilibrium_band():
    """Regression suite for all five Sol v38 findings."""
    import copy
    import pickle
    from dataclasses import asdict, replace
    from math import sqrt, ulp
    from conductor_sim.geometry import PlaneBoundary, contains_metal, physical_boundaries, slice_geometry
    from conductor_sim.visualization import SliceSpec, sample_sweep

    # 1. dataclasses.replace / copy / pickle preserve exact authored plane geometry,
    #    while replacing normal explicitly updates it and rounded tuples do not compare equal
    p = ConductingPlane((-1e6, 3e6, 0.0), (3.0, 1.0, 0.0), Grounded())
    p_rep = replace(p)
    p_ren = replace(p, name="renamed")
    p_copy = copy.deepcopy(p)
    p_pkl = pickle.loads(pickle.dumps(p))
    pt = (1e-11, 0.0, 0.0)
    expected_d = 3e-11 / sqrt(10.0)
    assert p == p_rep == p_copy == p_pkl
    assert hash(p) == hash(p_rep) == hash(p_copy) == hash(p_pkl)
    for plane_obj in (p, p_rep, p_ren, p_copy, p_pkl):
        assert isclose(plane_obj.signed_distance_m(pt), expected_d, rel_tol=1e-12, abs_tol=0.0)
        assert GroundedPlaneBackend().supports(ElectrostaticProblem((PointCharge(1e-12, pt),), (plane_obj,)))
    # Explicit plain-tuple rounded normal creates a distinct plane (never falsely equal)
    p_rounded = ConductingPlane((-1e6, 3e6, 0.0), tuple(p.normal), Grounded())
    assert p != p_rounded
    # Explicitly replacing normal updates the plane cleanly
    p_new_n = replace(p, normal=(1.0, 0.0, 0.0))
    assert p_new_n.normal == (1.0, 0.0, 0.0)
    assert isclose(p_new_n.signed_distance_m((1e6, 0.0, 0.0)), 2e6, rel_tol=1e-15)
    assert isinstance(asdict(p)["normal"], tuple)

    # 2. PlaneBoundary.first_hit handles overflowing endpoint distances and subnormal underflow-to-zero
    p_ovf = ConductingPlane((1e308, 0.0, 0.0), (1.0, 0.0, 0.0), Grounded())
    assert isclose(
        physical_boundaries(p_ovf)[0].first_hit((-1e308, 0.0, 0.0), (1.5e308, 0.0, 0.0)),
        0.8,
        rel_tol=1e-15,
    )
    u0 = ulp(0.0)
    p_14 = ConductingPlane((0.0, 0.0, 0.0), (1.0, 4.0, 0.0), Grounded())
    assert contains_metal(p_14, (u0, 0.0, 0.0)) is False
    assert isclose(
        physical_boundaries(p_14)[0].first_hit((u0, 0.0, 0.0), (-4.0 * u0, 0.0, 0.0)),
        0.2,
        rel_tol=1e-15,
    )

    # 3. Underflowed normal component (1e300, 1e-100, 0) is preserved in signed_distance_m and contains_metal
    p_tiny_comp = ConductingPlane((0.0, 0.0, 0.0), (1e300, 1e-100, 0.0), Grounded())
    assert p_tiny_comp != ConductingPlane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), Grounded())
    assert replace(p_tiny_comp, normal=(1.0, 0.0, 0.0)) == ConductingPlane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), Grounded())
    assert isclose(p_tiny_comp.signed_distance_m((0.0, 1e300, 0.0)), 1e-100, rel_tol=1e-12, abs_tol=0.0)
    assert contains_metal(p_tiny_comp, (0.0, 1e300, 0.0)) is False
    assert contains_metal(p_tiny_comp, (0.0, -1e300, 0.0)) is True

    # 4. Shallow slice (normal (1, 0, 1e-11), u=(0,0,1), v=(0,1,0)) stays visible with accurate profile
    p_shallow = ConductingPlane((0.0, 0.0, 0.0), (1.0, 0.0, 1e-11), Grounded())
    sp_shallow = SliceSpec(
        origin_m=(0.0, 0.0, 0.0), u=(0.0, 0.0, 1.0), v=(0.0, 1.0, 0.0),
        x_range_m=(-10.0, 10.0), y_range_m=(-1.0, 1.0), resolution=9,
    )
    geom_sh = slice_geometry(p_shallow, sp_shallow)
    assert geom_sh["visible"] is True
    assert geom_sh.get("fills_metal", False) is False
    assert isclose(geom_sh["normal_xy"][0], 1.0, rel_tol=1e-12)
    assert isclose(geom_sh["normal_xy"][1], 0.0, abs_tol=1e-12)
    bundle_sh = sample_sweep(
        [("shallow", ElectrostaticProblem((PointCharge(1e-12, (1.0, 0.0, 0.0)),), (p_shallow,)))],
        GroundedPlaneBackend(), plane=sp_shallow, resolution=9,
    )
    frame_sh = bundle_sh["frames"][0]
    center_row = frame_sh["grid"]["region"][4]
    assert center_row.count("metal") == 4
    assert center_row.count("surface") == 1
    assert center_row.count("vacuum") == 4
    assert frame_sh["geometry"][0]["visible"] is True
    assert len(frame_sh["profiles"]) == 1
    assert isclose(frame_sh["profiles"][0]["sigma_c_per_m2"][45], -1e-12 / (2.0 * pi), rel_tol=1e-10)

    # 5. Tight cancellation band preserves true 2-root and 0-root counts outside ~16 ULP noise
    sphere = SolidSphere(0.1, FixedNetCharge(5e-7), (0.0, 0.0, 0.0), "sphere")
    prob_eq = ElectrostaticProblem((PointCharge(2e-7, (0.0, 0.0, 0.25)),), (sphere,))
    spec_two = Motion1DSpec(
        prob_eq, 0, 5e-4, (0.0, 0.0, 1.0), 0.0, 0.1, 0.01,
        gravity=UniformGravity((0.0, 0.0, -42.64639393301955)),
    )
    eqs_two = find_1d_equilibria(spec_two, ExternalIsolatedSphereBackend(), displacement_bounds_m=(-0.12, 0.10))
    assert len(eqs_two) == 2
    assert isclose(eqs_two[0].position_m[2], 0.171655032154766, rel_tol=1e-9)
    assert isclose(eqs_two[1].position_m[2], 0.171655043115597, rel_tol=1e-9)
    assert eqs_two[0].stability == "unstable"
    assert eqs_two[1].stability == "stable"
    assert eqs_two[1].small_oscillation_angular_frequency_rad_s is not None

    spec_zero = Motion1DSpec(
        prob_eq, 0, 5e-4, (0.0, 0.0, 1.0), 0.0, 0.1, 0.01,
        gravity=UniformGravity((0.0, 0.0, -42.646393933020235)),
    )
    eqs_zero = find_1d_equilibria(spec_zero, ExternalIsolatedSphereBackend(), displacement_bounds_m=(-0.12, 0.10))
    assert eqs_zero == ()


def test_m45_sol_v39_sphere_far_field_cancellation_endpoint_fold_immutable_normal_and_coincident_slice():
    """Regression suite for all four Sol v39 findings."""
    from dataclasses import asdict
    from math import sqrt
    from conductor_sim.geometry import slice_geometry
    from conductor_sim.visualization import SliceSpec, sample_sweep

    # 1. Far-field isolated sphere (R=0.1, Q=2e-13, q=2e-7, z~17.1 m) avoids
    #    Kelvin-vs-compensator cancellation so negative-everywhere net force gives 0 equilibria
    sphere_far = SolidSphere(0.1, FixedNetCharge(2e-13), (0.0, 0.0, 0.0), "sphere")
    prob_far = ElectrostaticProblem((PointCharge(2e-7, (0.0, 0.0, 17.100168823850417)),), (sphere_far,))
    spec_far = Motion1DSpec(
        prob_far, 0, 5e-4, (0.0, 0.0, 1.0), 0.0, 0.1, 0.01,
        gravity=UniformGravity((0.0, 0.0, -1.47532529368e-9)),
    )
    assert find_1d_equilibria(spec_far, ExternalIsolatedSphereBackend(), displacement_bounds_m=(-2.5, 2.5)) == ()

    # 2. Endpoint near/at a cancelling fold peak never returns a bogus root:
    #    -42.64639393301988 gives 0 roots (matching 85-digit Decimal), and an exact
    #    binary64 endpoint zero right at the fold peak is refused explicitly.
    from conductor_sim.observables import force_energy
    sphere_fold = SolidSphere(0.1, FixedNetCharge(5e-7), (0.0, 0.0, 0.0), "sphere")
    prob_fold = ElectrostaticProblem((PointCharge(2e-7, (0.0, 0.0, 0.25)),), (sphere_fold,))
    spec_fold = Motion1DSpec(
        prob_fold, 0, 5e-4, (0.0, 0.0, 1.0), 0.0, 0.1, 0.01,
        gravity=UniformGravity((0.0, 0.0, -42.64639393301988)),
    )
    assert find_1d_equilibria(
        spec_fold,
        ExternalIsolatedSphereBackend(),
        displacement_bounds_m=(-0.09834496336481911, -0.07834496336481911),
    ) == ()
    z_fold = 0.25 - 0.07834496336481911
    fe_fold = force_energy(ExternalIsolatedSphereBackend().solve(
        ElectrostaticProblem((PointCharge(2e-7, (0.0, 0.0, z_fold)),), (sphere_fold,))
    )).charges[0].total_force_n[2]
    spec_fold_zero = Motion1DSpec(
        prob_fold, 0, 5e-4, (0.0, 0.0, 1.0), 0.0, 0.1, 0.01,
        gravity=UniformGravity((0.0, 0.0, -fe_fold / 5e-4)),
    )
    with pytest.raises(UnsupportedProblem, match="near-tangent"):
        find_1d_equilibria(
            spec_fold_zero,
            ExternalIsolatedSphereBackend(),
            displacement_bounds_m=(-0.09834496336481911, -0.07834496336481911),
        )

    # 3. ConductingPlane.normal is strictly immutable (plain floats, AttributeError on _raw/__dict__)
    #    and asdict(p) round-trips the authored direction losslessly via ConductingPlane(**...)
    p_x = ConductingPlane(normal=(1.0, 0.0, 0.0))
    plane_set = {p_x}
    assert all(type(x) is float for x in p_x.normal)
    assert not hasattr(p_x.normal, "__dict__")
    with pytest.raises(AttributeError):
        p_x.normal._raw = (0.0, 1.0, 0.0)
    with pytest.raises(AttributeError):
        _ = p_x.normal.__dict__
    assert p_x in plane_set
    assert p_x.signed_distance_m((1.0, 0.0, 0.0)) == 1.0

    p_31 = ConductingPlane((-1e6, 3e6, 0.0), (3.0, 1.0, 0.0), Grounded())
    d_31 = asdict(p_31)
    p_from_dict = ConductingPlane(**{**d_31, "constraint": Grounded()})
    assert p_from_dict == p_31
    assert isclose(p_from_dict.signed_distance_m((1e-11, 0.0, 0.0)), 3e-11 / sqrt(10.0), rel_tol=1e-12)

    # 4. Authored coincident slice (u=(1,-3,0), v=(0,0,1) on plane normal (3,1,0)) is strictly parallel:
    #    no spurious dividing line and no line profile
    p_orig_31 = ConductingPlane((0.0, 0.0, 0.0), (3.0, 1.0, 0.0), Grounded())
    sp_coinc = SliceSpec(
        origin_m=(0.0, 0.0, 0.0), u=(1.0, -3.0, 0.0), v=(0.0, 0.0, 1.0),
        x_range_m=(-1.0, 1.0), y_range_m=(-1.0, 1.0), resolution=9,
    )
    geom_coinc = slice_geometry(p_orig_31, sp_coinc)
    assert geom_coinc["visible"] is False
    assert geom_coinc["fills_metal"] is True
    bundle_coinc = sample_sweep(
        [("coincident", ElectrostaticProblem((PointCharge(1e-9, (1.0, 0.0, 0.0)),), (p_orig_31,)))],
        GroundedPlaneBackend(), plane=sp_coinc, resolution=9,
    )
    frame_coinc = bundle_coinc["frames"][0]
    assert frame_coinc["geometry"][0]["visible"] is False
    assert frame_coinc["profiles"] == []
    assert all(r == "surface" for r in frame_coinc["grid"]["region"][4])


def test_m45_sol_v40_boundary_root_profile_tilt_sphere_extremes_multi_quadrupole_and_slice_asdict():
    """Regression suite for all five Sol v40 findings."""
    from dataclasses import asdict
    from math import sqrt
    from conductor_sim import ExternalMultiSphereBackend
    from conductor_sim.geometry import slice_geometry
    from conductor_sim.observables import force_energy
    from conductor_sim.visualization import SliceSpec, sample_sweep

    # 1. Resolvable simple equilibrium near upper scan boundary is found (not falsely refused
    #    when an uncounted second root sits just outside the scan bound)
    sphere = SolidSphere(0.1, FixedNetCharge(5e-7), (0.0, 0.0, 0.0), "sphere")
    prob = ElectrostaticProblem((PointCharge(2e-7, (0.0, 0.0, 0.25)),), (sphere,))
    spec = Motion1DSpec(
        prob, 0, 5e-4, (0.0, 0.0, 1.0), 0.0, 0.1, 0.01,
        gravity=UniformGravity((0.0, 0.0, -42.64639393291988)),
    )
    eqs = find_1d_equilibria(
        spec, ExternalIsolatedSphereBackend(),
        displacement_bounds_m=(-0.08, 0.1716550366351809 - 0.25),
    )
    assert len(eqs) == 1
    assert isclose(eqs[0].position_m[2], 0.17165494335056045, rel_tol=1e-9)
    assert eqs[0].stability == "unstable"

    # 2. Surface-density profile uses authored slice axes matching slice_geometry (y=1)
    p_tilt = ConductingPlane((0.0, 1e-19, 0.0), (3.0, 1.0, 0.0), Grounded())
    sp_tilt = SliceSpec(
        origin_m=(0.0, 0.0, 0.0), u=(1.0, -3.0, 0.0), v=(0.0, 1e-19, 1.0),
        x_range_m=(-2.0, 2.0), y_range_m=(-2.0, 2.0), resolution=9,
    )
    bundle_tilt = sample_sweep(
        [("tilt", ElectrostaticProblem((PointCharge(1e-9, (1.0, 0.0, 1.0)),), (p_tilt,)))],
        GroundedPlaneBackend(), plane=sp_tilt, resolution=9,
    )
    f_tilt = bundle_tilt["frames"][0]
    assert f_tilt["geometry"][0]["visible"] is True
    assert isclose(f_tilt["geometry"][0]["point_m"][1], 1.0, rel_tol=1e-12)
    expected_sigma0 = -1e-9 * (3.0 / sqrt(10.0)) / (2.0 * pi)
    assert isclose(f_tilt["profiles"][0]["sigma_c_per_m2"][45], expected_sigma0, rel_tol=1e-10)

    # 3. Fixed-net-charge sphere observables resist d+R overflow and (R/d)^2 underflow
    s_ovf = SolidSphere(8e307, FixedNetCharge(0.0), (0.0, 0.0, 0.0), "big")
    obs_ovf = force_energy(ExternalIsolatedSphereBackend().solve(
        ElectrostaticProblem((PointCharge(1e148, (0.0, 0.0, 1.1e308)),), (s_ovf,))
    ))
    assert isclose(obs_ovf.charges[0].induced_force_n[2], -1.8941030463e-310, rel_tol=1e-7, abs_tol=0.0)
    assert isclose(obs_ovf.energy.induced_j, -0.003335962388, rel_tol=1e-7)

    s_unf = SolidSphere(1e-140, FixedNetCharge(0.0), (0.0, 0.0, 0.0), "tiny")
    obs_unf = force_energy(ExternalIsolatedSphereBackend().solve(
        ElectrostaticProblem((PointCharge(1e130, (0.0, 0.0, 1e25)),), (s_unf,))
    ))
    assert isclose(obs_unf.charges[0].induced_force_n[2], -1.7975103585e-275, rel_tol=1e-7, abs_tol=0.0)
    assert isclose(obs_unf.energy.induced_j, -4.493775896e-251, rel_tol=1e-7, abs_tol=0.0)

    # 4. Multi-source fixed-net-charge sphere preserves symmetric quadrupole residual force
    s_quad = SolidSphere(0.1, FixedNetCharge(0.0), (0.0, 0.0, 0.0), "sphere")
    prob_quad = ElectrostaticProblem(
        (PointCharge(1e-7, (0.0, 0.0, 1e4)), PointCharge(1e-7, (0.0, 0.0, -1e4))),
        (s_quad,),
    )
    obs_quad = force_energy(ExternalMultiSphereBackend().solve(prob_quad))
    pos_site = next(c for c in obs_quad.charges if c.source.position_m[2] > 0)
    assert isclose(pos_site.induced_force_n[2], -5.392531075356704e-37, rel_tol=1e-12, abs_tol=0.0)

    # 5. SliceSpec(**asdict(spec)) preserves authored coincident geometry & equality
    p_orig_31 = ConductingPlane((0.0, 0.0, 0.0), (3.0, 1.0, 0.0), Grounded())
    sp_c = SliceSpec(u=(1.0, -3.0, 0.0), v=(0.0, 0.0, 1.0))
    sp_rt = SliceSpec(**asdict(sp_c))
    assert sp_rt == sp_c
    assert hash(sp_rt) == hash(sp_c)
    assert slice_geometry(p_orig_31, sp_rt)["visible"] is False
    assert slice_geometry(p_orig_31, sp_rt)["fills_metal"] is True


def test_m45_sol_v41_multi_sphere_dipole_cancel_near_contact_and_shallow_profile_overflow():
    """Regression suite for all three Sol v41 findings."""
    from conductor_sim import ExternalMultiSphereBackend
    from conductor_sim.observables import force_energy
    from conductor_sim.visual_adapters import context_for_solution
    from conductor_sim.visualization import SliceSpec, sample_frame

    # 1. Multi-source fixed-Q sphere with exact asymmetric dipole cancellation
    s1 = SolidSphere(0.125, FixedNetCharge(0.0), (0.0, 0.0, 0.0), "sphere")
    p1 = ElectrostaticProblem(
        (PointCharge(2.0**-24, (0.0, 0.0, float(2**24))), PointCharge(25.0 * 2.0**-24, (0.0, 0.0, float(-5 * 2**24)))),
        (s1,),
    )
    obs1 = force_energy(ExternalMultiSphereBackend().solve(p1))
    pos1 = next(c for c in obs1.charges if c.source.position_m[2] > 0)
    assert isclose(pos1.induced_force_n[2], -9.375941921477878e-60, rel_tol=1e-12, abs_tol=0.0)
    assert isclose(obs1.energy.induced_j, -3.1460440564017875e-53, rel_tol=1e-12, abs_tol=0.0)

    # 2. Multi-source fixed-Q sphere near contact retains full binary64 accuracy
    s2 = SolidSphere(1.0, FixedNetCharge(0.0), (0.0, 0.0, 0.0), "sphere")
    p2 = ElectrostaticProblem(
        (PointCharge(1e-8, (0.0, 0.0, 1.0000000000000002)), PointCharge(1e-8, (0.0, 0.0, 1.0000000000000004))),
        (s2,),
    )
    obs2 = force_energy(ExternalMultiSphereBackend().solve(p2))
    outer2 = max(obs2.charges, key=lambda c: c.source.position_m[2])
    assert isclose(outer2.induced_force_n[2], -3.164743173345048e24, rel_tol=1e-12, abs_tol=0.0)

    # 3. Shallow grounded-plane slice with y=-1e110 intercept avoids r**3 overflow in sample_frame
    p_z0 = ConductingPlane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), Grounded())
    prob_sh = ElectrostaticProblem((PointCharge(1e20, (0.0, 0.0, 2.0)),), (p_z0,))
    sol_sh = GroundedPlaneBackend().solve(prob_sh)
    sp_sh = SliceSpec(origin_m=(0.0, 0.0, 1.0), u=(1.0, 0.0, 0.0), v=(0.0, 1.0, 1e-110), resolution=9)
    frame_sh = sample_frame(context_for_solution(prob_sh, sol_sh), sp_sh, seeds=[])
    assert len(frame_sh["profiles"]) == 1
    assert isclose(frame_sh["profiles"][0]["sigma_c_per_m2"][45], -3.183098861838e-311, rel_tol=1e-9, abs_tol=0.0)
