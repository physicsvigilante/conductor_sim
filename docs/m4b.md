# M4b: composable concentric conducting shells

**Status: accepted for strictly nested concentric conductors on 2026-11-12.** See [M4b numerical evidence](m4b-acceptance.md).

## Supported physical model

Compose several `SphericalShell(inner_radius,outer_radius,constraint,name)`
objects in one `ElectrostaticProblem`. The conductor centres must coincide
exactly, each shell must be wholly outside the previous one, and adjacent
shells must have a nonzero vacuum gap. No physical shell intersections,
eccentricities, or charge positions in inter-shell gaps are currently accepted.
Up to eight physical shells and 16 active axial charge sites are supported
in FEMM as a bounded first implementation.

All physical point-charge locations must lie within the innermost cavity.
The FEMM 5 backend still requires those sites to be on one axis through the
common centre. Independent `Grounded()` and `FixedNetCharge(Q)` constraints
apply to each connected *whole conductor*; a shell's two faces always
share one FEMM conductor property. Additional small finite charged
spheres are numerical source regularizations, never physical conductors.

The compiler builds each concentric surface and each dielectric gap in one
axisymmetric FEMM solve. `conductor_results` retains the authored input
order; `conductor_by_name` identifies any conductor regardless of ordering.
`surface_charge_density(...,conductor_name="inner")` samples FEMM E near
the selected physical face, with an offset constrained to its adjacent
vacuum region. For multiple physical shells, the conductor name is mandatory.

## Independent analytical physics

The `NestedShellBackend` is not a FEMM wrapper. It solves electrostatics
for the same concentric geometry using the radial Gauss-law field in
inter-shell gaps and exact Kelvin-image superposition inside the innermost
cavity. It even supports arbitrary 3D source positions in that cavity;
this does not make general 3D FEMM available.

With `q` the total free charge in the central cavity, `Q_i` the net
charge on shell i in order from inside out and `S_i=q+sum(Q_0..Q_i)`:

- Inner induced total on shell i: `-q-sum(Q_k for k<i)`.
- Outer induced total on shell i: `S_i`.
- Field between shell i and shell i+1: `E_r=S_i/(4*pi*epsilon*r^2)`.
- Field outside outermost shell: `E_r=S_last/(4*pi*epsilon*r^2)`.
- Potential of shell i:
  `V_i=(S_last/b_last + sum(S_j*(1/b_j-1/a_(j+1)),j=i..last-1))/(4*pi*epsilon)`.

The net charge of each grounded conductor is an unknown solved from the
linear system `V_i=0` while fixed-charge conductors use their prescribed
values. The internal cavity potential equals the solved innermost shell
potential plus an image-charge solution with zero on that inner surface.
Within each conductor `E=0`. This provides independent exact predictions
for shell charges, potentials, fields and local surface density.

## Validation requirements (passed for the tested scope)

Unit tests cover charges and potential for mixed and multiple groundings,
two and three shells, reverse authored order, and clear rejection of invalid
geometries. The dedicated real FEMM 5 workflow must compare numerical
conductor charges, shell voltages, cavity/gap/exterior fields and potentials,
sampled inner/outer sigma, mesh/domain sensitivity, and retain all 23 M2–M4a
live regressions. Tolerances are representative acceptance limits, **not**
universal numerical error guarantees.

**Explicitly not supported:** non-concentric shells, point charges in
inter-shell gaps or outside conductors, intersections, arbitrary non-axisymmetric
FEMM setups, direct mesh-integrated charge density, and viewer/scene graph.
