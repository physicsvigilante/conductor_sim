# M21e — Composable electric force and uniform gravity

Approved goal: extend the existing prescribed-field particle engine with reusable conservative force contributions, not a solver keyed to one JEE question. Freeze this scope before implementation.

## Composition contract

- Keep the existing PrescribedField, independent fixed-source identities, TestChargeMotionSpec, adaptive vector integrator, viewer and immutable scenario revisions. Add a separately typed uniform-gravity component and a small bounded electric-plus-gravity composition boundary. No arbitrary callback/plugin system, new ODE method or duplicate motion engine.
- Integrate the full resultant F=qE+mg. Preserve physical E and electric V as electrostatic observations; never disguise mg as a charge-dependent equivalent E.
- Store the two force vectors and electric qV and gravitational U_g=-m g dot (r-r_ref) separately. Total mechanical energy is K+qV+U_g. Energy-reference changes must not change trajectories, forces or work. Keep electric-only exports/backward behavior unchanged when gravity is absent.
- Validate every contribution against the qualified plane and any actual ring/disc axis. No force projection or accidental pointwise cancellation used as a general symmetry proof. Existing source guards and classical speed limits remain.
- Revisions preserve original model and physical IDs; mass/charge/launch and gravity parameter changes independently recertify and reintegrate. Comparisons must report gravity changes and correct energy semantics, or explicitly refuse unsupported comparisons. No hidden reuse of a baseline solve.

## Source-backed and transfer acceptance

Repository workbook `02_conductors_half_size.pdf`, physical page 15, archived IIT-JEE 2012 proton/vertical-plate question: 1 cm separation, proton initially at rest midway, 45 degrees to vertical just after release. Preserve both qE and mg, derive X=mgd/q approximately 1e-9 V. Use standard stated illustrative constant values; the finite preview ends strictly before reaching either plate. The uniform-field idealization omits edge fields, proton-induced conductor response and plate collision. Original source must be checked visually; distinguish exam data from preview time/coordinate choices.

Also demonstrate, without new production code per case: (1) oblique launch with uniform E+g, (2) electric support against gravity, (3) gravity-only projectile using zero electric field, (4) gravity absent recovers existing electric-only motion, (5) nonuniform point-source E+g, and (6) compatible axial ring+g. Include negative off-plane/off-axis requests. A small teacher mass/field/gravity revision must pass through the existing scenario interface.

## Verification

Independent constant-acceleration formulas and energy identities; source voltage/angle, mass and charge dependence, reference-gauge invariance, rotations/translations, nonuniform superposition, equilibrium and negative qualification. Test numerical scale appropriate to proton-sized forces and near-cancelled forces. Verify saved contributions and energy fields cannot be tampered with on export.

Actual offline Chromium acceptance must inspect visible force decomposition, energy labels, physical world-axis labels, saved-time playback and mobile layout. Do not depict plates as solved conductors or presentation interpolation as dynamics. Retain full ordinary Python and real FEMM nonregression; FEMM is not the new force backend.

Excluded: drag, terminal speed, switching protocols, magnetic forces, reacting charge sources, arbitrary 3D motion, conductor-induction plus gravity, or a universal multi-domain framework. Keep these limitations explicit.
