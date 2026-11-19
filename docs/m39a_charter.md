# M39a — Composable sealed-cavity Gauss charge ledger (frozen scientific charter)

**Freeze this charter in a distinct Git parent commit before implementation.** Prerequisite fully released M38b+M38c at `main` merge `0dc25ca686fd84891bc0d9291111ee48c4de3a9d`, identical reviewed Git tree `6cb46942d1aba0c1a28fc9696419c08334b441eb` and **22/22 actual post-merge push workflows successful**. M38c's originally failed main-push Chromium run stays historically red. [Issue #95](https://github.com/physicsvigilante/conductor_sim/issues/95) is the scope proposal; this charter is the authoritative M39a contract.

## Motivation and strict educational provenance

The frozen JEE-30 source audit (`docs/m36a_jee30_coverage.json`) retains workbook physical p11 Linked Comprehension Q28 and notes `POTENTIAL_CONDUCTORS1.pdf` p70 Illustration 8/9 as **unsupported for local irregular/nonconcentric cavity fields and surface densities**. However, a Gauss-law integral for *each* sealed cavity surface is exact under declared topology. M39a provides the **qualified Gauss-only sub-analysis**, not a completed replay of any uninspected original answer and not a JEE-30 coverage score increment. All M39a examples are **teacher_hypothetical**. Do not attribute authored numeric charges, cavity geometry or source membership to the original PDFs.

OpenStax University Physics 2 §6.4 gives electrostatic conductor shielding and induced charges; A. Pathak, arXiv:1609.04248 develops uniqueness-theorem separation of cavity/interior and exterior fields. These are physical context, **not** certification of a local 3D boundary solver.

## Explicit topology, not geometry

Author **one single connected homogeneous ideal metal** with 1–8 strictly **sealed**, mutually disjoint, noncontacting voids/cavities. Name the metal and each cavity with stable physical IDs. Declare 0–16 point-source **charge identities** with finite signed SI charges, assigned to exactly one cavity, not on metal walls; no point coordinates, cavity contours, surface areas, radius, field mesh or implicit containment checks. This is a **conditional authored topology assertion**, not a verified physical spatial arrangement. If an author needs actual geometric placement/containment, ask for a different qualified solver; never silently pretend the declarations prove shape or position.

The metal is either `FixedNetCharge(Q_metal)` (isolated) or `Grounded()` connected to an ideal zero-voltage charge reservoir, both as existing constraint vocabulary. Strictly **no exterior charges, external fields, other conductors, dielectric interfaces, open apertures, internal electrodes, contact with metal, transient connections or moving sources**. Homogeneous permittivity epsilon >0. Cavity walls are actual parts of the *same connected metal*, not separately connected shells. No unconstrained ground-wire current, resistance, heat, force or time integral.

## Qualified exact electrostatic integral results

For cavity i, declared source charge sum q_i and the total source charge S = sum_i q_i:
- The **entire inner face integral** is Q_inner,i = -q_i. **Not** its local surface sigma(x).
- For isolated metal with prescribed net metal charge Q_metal, Q_outer = Q_metal + S.
- For grounded metal with zero external source/field and reference V(infinity)=0, electrostatic uniqueness forces no exterior field: Q_outer = 0 and the **equilibrium net charge of the metal** Q_metal = -S.
- The flux of E through a suitable Gaussian surface in the exterior is Phi_E = Q_outer / epsilon (SI V·m); this does not specify the exterior E(x) for isolated irregular geometry. Conductor material has E=0 in static equilibrium.
- Cavity geometry and spatial source placement change **local distributions**, which are intentionally not calculated. For empty cavities with zero enclosed charge, a separate uniqueness theorem can establish E=0 inside, but M39a still provides no spatial field solver.

Results are exact **mathematical Gauss/uniqueness identities conditional on declared electrostatic assumptions**; finite IEEE float outputs require strict stability/overflow checks and are not promises of infinite precision. Grounded equilibrium net metal charge is **not** a measured charge transferred from an unspecified initial state. Comparing two separate teacher-authored static states is **not** a current-versus-time waveform.

## Minimal composable authoring and evidence contract

- Add one separate typed module `conductor_sim.cavity_gauss`: immutable `CavityCharge(source_id,q)`, `SealedCavity(cavity_id,sources)`, `GaussCavityAssembly(conductor_id,cavities,constraint,epsilon)`, `solve_cavity_gauss` and an immutable ledger result. IDs unique globally; reject Boolean/nonfinite charge and epsilon, zero/negative epsilon, unknown constraints, empty/duplicate IDs, non-sequences, too many cavities/sources, external-source claims and arbitrary supplied geometry. Revalidate nested authored values at solve/export, even after `object.__setattr__` tampering; do not trust frozen dataclass as cryptographic protection. Immutable `dataclasses.replace` with authentic IDs is sufficient for teacher scenario revisions; never mutate baseline in place.
- Explicit versioned JSON with all input identity/source ledgers, domain assumptions, each cavity's summed source charge/inner-face charge, metal charge, outer-face charge, full exterior flux, grounded exterior-field theorem flag, SI units, `teacher_hypothetical` and `original_workbook_items_newly_solved=0`. Use typed re-execution and type-sensitive comparisons to refuse forged in-process saved ledgers. Do not assert external-origin attestation or that a sufficiently re-authored consistent JSON is cryptographically authenticated.
- Public `potential_at`, `electric_field_at`, `surface_charge_density_at` or other pointwise/shape-specific calls must clearly **refuse** with unsupported-domain diagnostics; no fake whole-shape Gaussian-flux distribution maps.
- New honest JSON-only teacher example for three cavities (signed +, -, neutral, optional cancellation) with isolated-vs-grounded *separate* scenarios and explicit source-versus-teacher attribution; only true analytic charge/flux ledger, no fake perspective renderer.

## Prespecified physics, numerical, hostile and review acceptance

1. Independently derive with a **stdlib-only** reference checker (NO production imports), preferably using Decimal for algebraic sum cross-checks; verify every genuine SI quantity, signs, exterior flux, theorem domain and source identity. Isolated/grounded variants, 1–8 cavities, empty/neutral cavities, q-sign reversals, cancellation and source order invariance. Global charge conservation across inner/outer faces and guaranteed 0 grounded exterior flux are mandatory.
2. Test changing one source's position **conceptually within the same declared cavity** does not change *integrated* induced charge; since no coordinates are admitted, do not claim to have spatially validated movement. `dataclasses.replace` of an immutable teacher assembly with different source charge/assignment must create a new qualified model; it must not change the parent's results.
3. Reject Boolean `False == 0.0` type attacks, forged face/outer charges, source identities/memberships, unsupported exact-point `E,V,sigma,F` claims, false grounded isolation, nonfinite/overflow, forged units, source-versus-teacher/provenance, and any invented time/current. Native export and standalone numerical checker cover distinct trust boundaries and should not be conflated.
4. Dedicated actual Python 3.10/3.12 CI runs native and independent JSON reference, preserves exact Git HEAD/TREE and original PDF hashes and first genuine red tests; **all historical production Python, FEMM and browser suites** still must pass the exact draft PR source commit. No FEMM claim of validating an intentionally geometry-free theorem.
5. Request **one fresh-context narrow independent JEE/physics/topology/evidence review only AFTER green developer CI**, with an unfamiliar source/cavity permutation and grounded-vs-isolated calculation. No extra broad browser review for unchanged viewers, or false assertion of verified irregular 3D field/sigma. A new source revision with an actual finding gets a targeted correction review only.
6. **PR stays draft/unmerged** until independent PASS and separate explicit owner authorization, then exact-head guarded merge, same reviewed Git tree on main and real postmerge main-push CI. Failure evidence remains in history.

A later M39b may draw a clearly labelled **topology-only schematic** without invented local field lines. A true arbitrary nonconcentric cavity-boundary solver is a separate large milestone, not an accidental side effect of M39a.
