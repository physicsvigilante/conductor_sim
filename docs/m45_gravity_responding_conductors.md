# M45 gravity coupled with responding conductors

M45 composes the existing built-in `UniformGravity(acceleration_m_s2, reference_position_m)` component with certified responding-conductor 1D and 2D dynamics (`Motion1DSpec`, `integrate_motion_1d`, `Motion2DSpec`, `integrate_motion_2d`), offline motion viewers (`sample_motion_1d`, `sample_motion_2d`), and a 1D static equilibrium & small-oscillation stability finder (`find_1d_equilibria`).

## Public API

- `Motion1DSpec(..., gravity=UniformGravity(...))` and `Motion2DSpec(..., gravity=UniformGravity(...))`: optional uniform gravitational acceleration coupled with any analytical responding-conductor backend that provides `force_energy` (`GroundedPlaneBackend`, `ParallelPlateBackend`, `ExternalGroundedSphereBackend`, `ExternalIsolatedSphereBackend`, `ExternalMultiSphereBackend`, `ImageChargeShellBackend`, `MultiImageShellBackend`, `NestedShellBackend`).
- `find_1d_equilibria(spec, backend, *, displacement_bounds_m=None, scan_samples=129) -> tuple[ConductingGravityEquilibrium, ...]`: brackets and refines 1D force-balance roots $F_{\text{elec}}(s) + m \mathbf{g}\cdot\hat{\mathbf{u}} = 0$ inside the surface-guard vacuum domain, computes local linear stiffness $k_{\text{eff}} = -\partial F/\partial s = \partial^2 U/\partial s^2$, classifies `stable` / `unstable` / `indeterminate`, and reports $\omega = \sqrt{k_{\text{eff}}/m}$ for stable equilibria.

## Physical & symmetry contract

- **Strict symmetry qualification (never fabricated guide forces):**
  - 1D motion requires $\mathbf{g}$ collinear with the proved 1D symmetry axis; any transverse gravitational component raises `UnsupportedProblem`.
  - 2D motion includes $\mathbf{g}$ when certifying the invariant symmetry plane (allowing release from rest with non-collinear $\mathbf{g}$ to select the physical plane) and raises `UnsupportedProblem` whenever conductors, launch velocity, held charges, and $\mathbf{g}$ do not share one invariant plane.
- **Separated force & conservative energy ledger:**
  - Electrostatic force and energy come exclusively from `force_energy` on the responding conductor at each state; gravitational force is $m\mathbf{g}$ and gravitational potential energy is $U_g(\mathbf{r}) = -m\mathbf{g}\cdot(\mathbf{r} - \mathbf{r}_{\text{ref}})$.
  - When `gravity` is set, each `MotionSample` / `Motion2DSample` populates `electric_force_n`, `gravitational_force_n`, `electric_interaction_energy_j`, and `gravitational_potential_energy_j` alongside total `force_n`, `interaction_energy_j = U_elec + U_grav`, and `total_energy_j = K + U_elec + U_grav`.
  - `energy_drift_j` differences each contribution separately and evaluates $\Delta U_g = -m\mathbf{g}\cdot(\mathbf{r} - \mathbf{r}_0)$ directly so large `reference_position_m` offsets do not introduce cancellation.
- **Explicit refusals & backward compatibility:**
  - Leaving `gravity=None` preserves exact M16/M17/M18 schemas and trajectories.
  - Non-`UniformGravity` objects, underflowing $m\mathbf{g}$, symmetry-breaking $\mathbf{g}$, FEMM backends lacking `force_energy`, and unbounded 1D domains queried without explicit finite `displacement_bounds_m` are refused explicitly.
