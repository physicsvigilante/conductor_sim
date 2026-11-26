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

    # Huge gravitational reference offset (1e12 m) must not degrade energy_drift_j!
    g_ref0 = UniformGravity((0.0, 0.0, -9.8), reference_position_m=(0.0, 0.0, 0.0))
    g_ref_big = UniformGravity((0.0, 0.0, -9.8), reference_position_m=(0.0, 0.0, 1e12))
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

    # 5. Returning 1D trajectory checks every ordered frame (no duplicate-position overwrite) & checks gravity terms
    spec_ret = Motion1DSpec(
        ElectrostaticProblem((PointCharge(1e-17, (0.0, 0.0, 1.0)),), (plane,)),
        0, 1.0, (0.0, 0.0, 1.0), -1.0, 1.0, 0.25,
        gravity=UniformGravity((0.0, 0.0, 2.0)),
    )
    res_ret = integrate_motion_1d(spec_ret, GroundedPlaneBackend())
    from dataclasses import replace as dc_replace
    forged_samples = (dc_replace(res_ret.samples[0], electric_force_n=99.0), *res_ret.samples[1:])
    forged_res = dc_replace(res_ret, samples=forged_samples)
    with pytest.raises(UnsupportedProblem, match="disagrees"):
        sample_motion_1d(spec_ret, GroundedPlaneBackend(), result=forged_res, visual_frames=5, resolution=9)
    forged_g_samples = (dc_replace(res_ret.samples[0], gravitational_force_n=99.0), *res_ret.samples[1:])
    with pytest.raises(UnsupportedProblem, match="disagrees"):
        sample_motion_1d(spec_ret, GroundedPlaneBackend(), result=dc_replace(res_ret, samples=forged_g_samples), visual_frames=5, resolution=9)

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

    # 7. UniformGravity.potential_energy_at avoids intermediate underflow/overflow
    g_ext = UniformGravity((1e-200, 0.0, 0.0), reference_position_m=(0.0, 0.0, 0.0))
    assert isclose(g_ext.potential_energy_at((1e-200, 0.0, 0.0), mass_kg=1e200), -1e-200, rel_tol=1e-14)
