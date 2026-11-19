# M40a — Exact local induced sigma for multiple *spherical* cavities (frozen scientific charter)

**Prerequisite and release identity:** M39b production [PR #98](https://github.com/physicsvigilante/conductor_sim/pull/98) merged as `6257ae9c7d7a0e7da2a12cb01a8519b0f2baa7b0`, verified identical reviewed tree `cb9f9822252f982d5691493b7d54b0fae5917049`, **26/26 completed-success postmerge workflows**. Scientific scope proposal [issue #99](https://github.com/physicsvigilante/conductor_sim/issues/99). This standalone charter MUST be frozen in a distinct Git commit **before any M40a implementation code**.

## Purpose, source honesty and physical regime

M39a/b already compute reliable **entire-face** induced charges for one connected metal with several teacher-declared sealed voids, but cannot compute local sigma without geometry. This milestone admits **one specific physically verified geometry**: a bounded connected ideal electrostatic metal with **one spherical outer surface** and 1–8 strictly sealed, mutually disjoint **spherical** internal voids. All cavity spheres lie strictly inside the outer sphere, with a positive nonzero metal gap between all walls; each existing teacher-assigned source ID has one actual finite 3D position strictly inside the assigned cavity. Physical identities/charges/constraints/permittivity are inherited from the unchanged immutable `GaussCavityAssembly` and existing verified `solve_cavity_gauss` ledger; geometry must not rewrite the source inventory.

No external point charges, external applied fields, other conductors, dielectric interfaces, internal electrodes, apertures, moving charges, time evolution or finite-resistance ground wires. The connected conductor is either **isolated with an independently fixed Q_metal** or **fully grounded at zero V with zero exterior charge/field**. A boundary surface at V(infinity)=0 is assumed, homogeneous positive permittivity throughout. No geometric validation that an originally scanned irregular cavity happens to be spherical.

**Educational provenance:** Entirely teacher-authored geometry and source positions; **zero** newly solved verified original JEE workbook answers. Original workbook p11 linked Q28 and POTENTIAL_CONDUCTORS1 notes p70 irregular multicavity fields/densities REMAIN unsupported. No FEMM representation is asserted for non-axisymmetric multi-cavity geometry. Previous M39a topology-only records MUST NOT acquire new local-field permissions.

## Exact physical sigma theorem and sign

For spherical cavity i of radius a_i and center c_i, a point charge q_j at x_j lies strictly within cavity i with d_j=|x_j-c_i|<a_i. At **unit vector n pointing from cavity center into the metal**, the entire inner wall's local **metal-side surface density** is

\[
\sigma_{\text{inner},i}(\mathbf n)
=-\sum_{j\in i}\frac{q_j(a_i^2-d_j^2)}
 {4\pi a_i\,|a_i\mathbf n-(\mathbf x_j-\mathbf c_i)|^3}
\quad[\mathrm{C/m^2}].
\]

This is the exact Kelvin image-charge/Dirichlet uniqueness distribution for each spherical cavity. The whole connected metal has a *single common constant potential*; that additive constant does **not** alter inner fields or sigma, and other cavities cannot alter each other's inner sigma under electrostatic shielding. Hence the exact surface integral over any entire inner face is `Q_inner_i = -sum(q_j in i)`, matching M39a.

For the **spherical outer boundary** of radius R, no exterior source/field and fixed total outer charge Q_outer, uniqueness gives uniform `sigma_outer(n)=Q_outer/(4 pi R^2)`, independent of n. For grounded metal Q_outer=0 and `sigma_outer=0`; isolated `Q_outer=Q_metal+sum q`. These assertions do NOT apply to an irregular outer boundary. A cavity with positive/negative sources at different positions may have exactly zero net induced face charge but **nonzero local sigma**. Moving a source within the same cavity changes its local profile but not its integrated total. A separate static grounded/isolated revision can change the outer charge but not inner sigma for fixed sources.

Neither an arbitrary-cavity field map nor pointwise cavity E(x), V(x), forces, wire-current history, mechanical motion, energy or FEMM numerical solution is within M40a. Explicit unsupported calls must fail rather than silently use the old one-cavity solver as a fictitious 3D surrogate.

## Typed composable geometry boundary

- Add a distinct immutable geometry/position-authoring module built on **actual existing M39a types**: outer sphere center/radius and one `SphericalCavityGeometry` per original cavity ID; one `NamedSourcePosition` for every originally named point source. Enforce exact one-to-one identities and source ownership, immutable revisions, positive radius, all coordinates/charges finite non-Boolean SI, 1..8 cavities, <=16 sources, and no optional invented geometry flags.
- Validate strict **actual** 3D containment and mutual disjointness: `|x_source-c_i|<a_i`, `|c_i-c_k|>a_i+a_k`, `|c_i-C_outer|+a_i<R_outer`. Add explicitly documented numerical safety clearance around all walls. A valid M39a topology ledger alone never certifies these stronger geometric conditions.
- Query `inner_surface_charge_density(cavity_id, normal)` and `outer_surface_charge_density(normal)` with a **typed finite nearly-unit normal** and returns finite scalar SI C/m²; reflect the correct metal-side sign. Do NOT quietly normalize arbitrary directions or accept Boolean/Floats-as-IDs.
- Physical density outputs use finite binary64 arithmetic. For nonzero theoretical kernel contributions, detect and refuse silent underflow to zero and overflow; do not replace physically nonzero results by theorem zeros. The exact physical theorem does not imply arbitrary machine-precision accuracy near singular/contact configurations. Reject ill-conditioned near-wall geometries and unqualified extreme scales explicitly.
- No geometry-qualified query may be attached to an **unverified M39a ledger**. Rebuild and validate nested immutable authored data before calculation/export, even if a frozen dataclass was altered through low-level Python mutation. Saved and re-exported identity/charge/geometry/sign evidence must be type-sensitive; standalone checkers are not cryptographic source authentication.

## Independent validation and release gates

1. First perform actual **three- to six-cavity** teacher-authoring with signed, empty and nonempty neutral cavities, no geometry overlaps. Independently compute a centered-source uniform sigma, off-center near/far pole contrast and full inner-face integration by separate quadrature, outer uniform density, parent preservation across `dataclasses.replace`, and grounded-vs-isolated invariance of inner profiles. Evaluate enough nonaxial surface normals to prove local 3D orientation is real; test translation/rotation/cavity ordering/source ordering.
2. Independent math reference must not import production physics. Use alternative Kelvin image normal derivative and/or numerical area integration rather than copying the production sigma expression as the sole oracle. Compare the qualified **single-cavity limit** to the existing released `multi_image` backend as a regression control, NOT as a fake multi-metal physical model.
3. Hostile edge cases: extra/missing/swapped physical IDs or placements even when charges cancel, fabricated geometry/grounding provenance, overlapping or barely touching cavities, source on/outside face, cavity escaping outer sphere, invalid unit normal, Boolean and nonfinite position/radius/charge, loss of small nonzero signed density, signed cancellation and outer-surface insufficiency. Reject failed qualification before attempting local field evaluation.
4. Dedicated source-pinned Python 3.10 and 3.12 CI with actual new tests, a **standalone no-production-import mathematical reference**, exact SHA/TREE and original PDF digests. Full inherited Python, real FEMM, original Chromium/source CI must remain green at current exact draft PR HEAD; FEMM is a regression, **not** confirmation that nonconcentric 3D cavities have been numerically meshed.
5. **One** focused fresh-context independent physics/geometry/sigma/provenance review only AFTER developer CI passes, with novel nonconcentric cavity composition and independently sampled near/far density+surface integral. Changes-requested findings trigger **targeted** correction review rather than redoing unchanged historic FEMM/Chromium. Review evidence and all first-attempt reds remain preserved.
6. PR remains DRAFT/UNMERGED until independent PASS and **separate explicit owner merge approval**, followed by expected-head guarded merge, matching reviewer-approved Git tree on main, and successful real postmerge main-push CI. No auto-merge.

This charter permits **exact local inner-wall surface density only for geometry-verified spheres**, a strict small extension of the M39a Gauss topology, not an arbitrary irregular 3D conductor solver.
