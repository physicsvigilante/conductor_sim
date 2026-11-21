# An electron crosses into higher potential

Workbook `02_conductors_half_size.pdf`, PDF page 5 (printed 3.37), Single Correct Q52 shows an electron crossing from V₁ to a higher V₂ in a uniform field. Treat the interface as an equipotential plane. The electric field is normal to it, so the force has zero tangential component. **The velocity component parallel to the interface stays constant, option (1).** The electron gains kinetic energy ΔK = e(V₂ − V₁). For an oblique crossing an increasing normal speed can change the direction of travel.

## Detect the crossing in a native trajectory

The source specifies neither the field strength nor the launch velocity. For this teacher example use E = −100 V/m along x, launch the electron at x = 0, and place a passive detector at x = 5 mm. The uniform field extends through the detector. The field and launch velocity both lie in the plane with normal (0, 1, 2), which qualifies the motion for the planar integrator. The example uses nonrelativistic dynamics and the electron’s SI charge and approximate mass.

```pycon
>>> from math import hypot, isclose, sqrt
>>> from conductor_sim import PointCharge, PrescribedField, TestChargeMotionSpec, UniformElectricField
>>> from conductor_sim.motion_events import EventPolicy, EventStudy, PlaneEvent, investigate_motion
>>> from conductor_sim.scenarios import PrescribedScenario
>>> field = PrescribedField((UniformElectricField((-100, 0, 0)),))
>>> q, mass = -1.602176634e-19, 9.1094e-31
>>> start, initial = (0, 0, 0), (2e5, -1e5, 0.5e5)
>>> model = TestChargeMotionSpec(field, PointCharge(q, start), mass, initial, 3e-8, 3e-9,
...                              plane_normal=(0, 1, 2))
>>> scenario = PrescribedScenario("q52:example", model, (("uniform", "Uniform field"),),
...                               "Teacher-selected oblique electron launch")
>>> detector = PlaneEvent("V2", (0.005, 0, 0), (1, 0, 0), "Higher equipotential", direction=1)
>>> policy = EventPolicy(1e-9, 1e-15, 1e-9, 1e-2)
>>> result = investigate_motion(EventStudy(scenario, (detector,), policy))
>>> crossing, = result.events
>>> sample = crossing.sample
>>> isclose(sample.position_m[0], 0.005, abs_tol=1e-9)
True
>>> vx, vy, vz = sample.velocity_m_s
>>> isclose(vy, initial[1]) and isclose(vz, initial[2])
True
>>> delta_v = field.potential_at(detector.point_m) - field.potential_at(start)
>>> delta_k = sample.kinetic_energy_j - mass*hypot(*initial)**2/2
>>> delta_v == 0.5 and isclose(delta_k, -q*delta_v, rel_tol=1e-5, abs_tol=0)
True
>>> expected_vx = sqrt(initial[0]**2 - 2*q*delta_v/mass)
>>> isclose(vx, expected_vx, rel_tol=1e-5) and vx > initial[0]
True

```

The two signed tangential velocity components survive the crossing while the normal velocity increases. Launching exactly along the normal would also preserve the direction of travel. It is the tangential invariance that applies to both launch geometries.

For the single printed item, the model preserves the higher-potential crossing (1/1), reproduces both tangential invariance and the independent work–energy result (1/1), runs a native time-dependent crossing with teacher-selected values (1/1), and identifies the printed answer (1/1). Educational usefulness has not been measured.
