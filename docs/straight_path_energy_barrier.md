# Energy barrier in a straight guide

In `02_conductors_half_size.pdf`, page 3, Single Correct Q35, a unit positive charge of mass m enters a tunnel in a uniformly charged nonconducting sphere of radius R. The diagram places the chord at distance R/2 from the centre. The choices use positive volume charge density ρ. The thin-tunnel idealization neglects removed charge and uses a straight, frictionless guide to constrain transverse motion. A finite bore would require its own field solution.

Let x measure distance along the chord from its midpoint. At the two ends x = ±L, with L² = 3R²/4. Gauss’s law inside the intact sphere gives Eₓ = ρx/(3ε₀). Integrating the opposing force from −L to 0 gives the potential energy rise ΔU = qρL²/(6ε₀) = qρR²/(8ε₀). Although both ends have equal potential, the barrier lies halfway along the chord.

For the printed unit charge the threshold speed is √[ρR²/(4mε₀)], option 1. In an ideal continuous model this speed is an infimum. At equality, on the entry half of the chord, speed is proportional to |x|. The integral of dx/|x| diverges at the midpoint. Finite-time passage requires a strictly higher launch speed.

```python
from math import pi, sqrt
from conductor_sim.path_barriers import straight_path_energy_barrier
from conductor_sim.prescribed_fields import UniformSphereField

radius, density = 2., 1e-9  # teacher-selected metres and C/m³
sphere = UniformSphereField(4*pi*radius**3*density/3, radius)
half_chord = sqrt(3)*radius/2
start, end = (-half_chord, radius/2, 0), (half_chord, radius/2, 0)
barrier = straight_path_energy_barrier((sphere,), 1, start, end)
# barrier.rise_j is the peak energy above the starting value.
# barrier.peak_fraction is 1/2, measured along the guide from its start.
```

Generate the offline teaching comparison with `python -m tools.jee_tunnel_profile review-artifacts/q35-guide/q35.html`. The upper and lower curves compare the original sphere with a teacher-selected axial bias. Both use the same energy scale. The browser check `python -m tools.check_jee_tunnel_profile_browser review-artifacts/q35-guide` checks both curves, their maximum markers, and signed table values in mobile and desktop Chromium.

Use `straight_path_energy_profile` with the same arguments as the barrier calculation to obtain the signed energy change at any segment fraction. The barrier result and the graph use the same qualified profile. Profile position is spatial. It does not represent elapsed time.

Each uniform sphere must contain the entire segment, including its endpoints. The test uses the supplied coordinates without a boundary tolerance. An existing `PrescribedField` can be passed directly when every source qualifies. Its authored permittivity is retained, including when several spheres share a bias field. An empty prescribed field gives zero barrier. Multiple sphere contributions and `AffineFieldAssembly` instances may share the same guide. The caller is responsible for the validity of each prescribed affine field along the guide. The observation maximizes the total energy before rounding and returns the earliest maximizer if there is a tie. It does not integrate motion or determine a finite transit time.

| Evaluation dimension | Status for the one original item | Evidence |
| --- | --- | --- |
| Interpretation fidelity | Pass, 1/1 | Source chord offset, unit charge and mass are preserved. The narrow-bore and guide assumptions are explicit. |
| Physical correctness | Pass, 1/1 | The integrated axial force and the sphere potential independently give the interior barrier. |
| Simulation coverage | Supported under the thin-guide idealization, 1/1 | Static barrier and its location are computed across the complete chord. |
| Educational usefulness | Not evaluated, 1/1 | The example exposes the interior maximum and permits bias fields. No learner study was performed. |
| Textbook-answer completeness | Complete, 1/1 | The threshold gives option 1 with the limiting-speed qualification. |
