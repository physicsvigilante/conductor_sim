# Layered linear dielectrics (M43)

M43 extends the shared M42b gap-elastance and energy-ledger contract (`stack_coupling.py`, `planar_stack.py`, `spherical_stack.py`, `stack_switching.py`) to piecewise-homogeneous linear isotropic dielectric layers ($\kappa \ge 1$).

## Core primitives

- `DielectricLayer(thickness_m, relative_permittivity)`: one homogeneous linear isotropic layer with $\kappa \ge 1$ relative to vacuum $\varepsilon_0$.
- `PlanarStackProblem(..., gap_layers=((...), ...))`: ordered dielectric slabs spanning each planar gap (left to right, thicknesses summing to the gap width; requires `d_left_c_per_m2 = d_right_c_per_m2 = 0` and default vacuum `permittivity_f_per_m`).
- `SphericalStackProblem(..., gap_layers=((...), ...), exterior_layers=(...))`: ordered concentric spherical dielectric shells spanning interior gaps and/or finite dielectric coatings around the outermost conductor (surrounded by `permittivity_f_per_m` to infinity).
- `DielectricLayerField` & `DielectricInterfaceCharge`: per-layer uniform displacement/field/polarization (planar, populated when `gap_layers` is provided) and exact bound surface charge densities $\sigma_b = \mathbf{P}\cdot\hat{\mathbf{n}}$ and bound face charges $Q_b$ at every conductor/dielectric and dielectric/dielectric boundary. Per-layer bound charge sums to zero identically in exact rational arithmetic (and therefore across every source-free interior gap or vacuum-surrounded exterior coating, holding to binary64 rounding in the public `float` fields). When a finite spherical exterior coating sits inside a non-vacuum infinite background ($\kappa_{\text{ext}} > 1$), the finite-radius exterior interface ledger includes the outer dielectric-background boundary and sums to $-(1 - 1/\kappa_{\text{ext}})Q_{\text{enc}}$ (its compensating bound charge resides at infinity).
- `quasistatic_stack_transition(before, after_problem) -> QuasistaticTransitionResult`: reversible endpoint capacitance/work ledger ($W_{\text{supply}} + W_{\text{external}} = \Delta U_{\text{field}}$, `energy_residual_j == 0.0`) for fixed-topology dielectric or spacing changes.

## Physical boundaries and explicit refusals

- **Fixed-geometry switching vs. quasi-static mechanical/dielectric change:** use `switch_conductor_stack` when changing wires or terminal voltages ($W_{\text{external}} = 0$, $E_{\text{residual}} \ge 0$); use `quasistatic_stack_transition` only when electrical connections and terminal voltages remain unchanged ($E_{\text{residual}} = 0$).
- **Dielectric interfaces:** `potential_at` is continuous everywhere; `electric_field_at` raises `SingularField` at any interface where adjacent layers have distinct permittivities (structural discontinuity rule, matching conductor-wall behavior even when the field vanishes in a particular state), and returns the continuous field when adjacent permittivities match (including an outermost spherical coating whose represented permittivity `float(EPSILON_0 * kappa)` equals `permittivity_f_per_m`).
- **Explicit refusals:** $\kappa < 1$, mismatched layer thicknesses, active point charges inside multi-$\kappa$ spherical gaps or outside dielectric-coated spheres, lateral partial-insertion 1D field claims, and non-linear or anisotropic media.
