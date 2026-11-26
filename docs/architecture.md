# Architecture: qualified physical reuse, not a universal solver

See the [current release map](current_capabilities.md) for source identity and validation status. This document describes the M40b architecture and the separately identified audit correction; historical M4/M5 notes are no longer the current architecture index.

## Three deliberately different physical contracts

**Responding conductors.** An immutable `ElectrostaticProblem` contains charges, conductor geometry, electrical constraints and SI permittivity. `ElectrostaticBackend.supports/solve` selects only a qualified whole arrangement; `FieldSolution` supplies supported potential and electric-field observations. Analytical image/symmetry solvers and optional FEMM have different domains. Two individually solvable conductors cannot generally be combined by adding their separately solved response fields: the common boundary-value problem changes.

**Prescribed sources and probes.** `PrescribedField` superposes supported nonresponding source providers. Source geometry, guards and field-domain restrictions survive composition. Native test-particle, force/gravity, driven-protocol and event layers reuse these fields, but qualify the complete physical plane/axis and requested investigation. Mechanical energy conventions for prescribed fields and conductor-induced forces remain distinct.

**Multicavity observations.** M39a first proves conditional complete-face Gauss totals from declared sealed-cavity membership. M40a separately adds actual spherical geometry and source containment to compute local surface density. Its surface-density solution intentionally refuses general E/V/force/time queries. This is not a partial `FieldSolution` pretending to support every observation. M39b topology and M40b physical-projection viewers remain correspondingly different.

## Flow of responsibilities

```text
Typed authored physical model and source provenance
  -> whole-model geometry/boundary/domain qualification
  -> qualified backend or native investigation
  -> actual physical result, termination and numerical evidence
  -> validated observations/comparisons
  -> offline presentation of those observations
```

`scenarios.py` and `structural_scenarios.py` add retained named-source identities, explicit physical versus structural revisions and complete requalification. The graph/fork/inquiry layers retain real source lineage and issue new investigations without silently changing old observations. They are useful orchestration tools, not mandatory wrappers for a simple one-sphere calculation. A numerical initial guess, visual annotation or display interpolation must not become a physical initial condition or simulated time evolution.

## Reuse that is justified

An unfamiliar arrangement inside an existing regime should require a new authored model, not a new production force law. Shared geometry, field protocols, event observers and typed revisions support this. The post-M40b audit adds one small private direction-normalization helper because four actual consumers repeated the same demonstrated subnormal arithmetic defect. It preserves ordinary arithmetic and rejection behavior. It must not replace M40a's precision-sensitive original raw query rays.

Do not force all backends into a generic graph compiler, callback registry or universal geometry object. Improve a shared layer only after two or more concrete consumers need the same semantic contract. Source-specific example scripts are appropriate when they only author/label an existing model; they are not appropriate places to conceal a new physical law or silently substitute geometry.

## Numerical/backend boundaries

FEMM remains optional and axisymmetric in the implemented adapter. Its charged numerical cores and mesh are computational devices, not new physical conductors. Global axis/symmetry, concentricity and source-placement restrictions belong to that backend, not to every analytical backend. Several analytical families support exterior spheres, planes or arbitrary 3D charge positions inside a qualified spherical cavity; this does not imply arbitrary irregular 3D conductors or general interacting spheres.

Every revised model must independently meet geometry, SI scale, source guard, conditioning and solver contracts. A valid baseline never licenses all revisions. A guard-truncated trajectory is not a completed requested motion. Point samples are not cell averages; static equilibrium pairs are not current transients. Numerical tolerances are part of each qualified analysis, not permission to hide a reproduced discrepancy.

## Trust and compatibility

Frozen dataclasses prevent ordinary reassignment, not all privileged Python mutation. Issued-result receipts and revalidation enforce scoped in-process consistency; they are not a security sandbox or signatures on third-party files. The M40b JSON checker verifies physical record consistency, not separately edited HTML authenticity.

Older model constructors deliberately retain some Boolean/coercion behavior for compatibility. Strict newer authoring paths validate raw input before that information is lost; M37b explicitly tests this distinction. A strict-constructor migration needs a versioned/deprecation plan and caller characterization, not a blanket constructor change hidden in an audit.

## Maintenance pressure and tests

Milestone layering now creates substantial private-symbol coupling, repeated validation/serialization and a large eager public import surface. These are maintenance risks, not evidence that a universal replacement would be better. Prefer a few tested internal services with preserved record semantics; measure repeated revalidation before caching it across mutable object references.

CI must exercise downstream consumers when upstream source or independent-reference tools change. Audit corrections widen several late-milestone PR filters conservatively without deleting old checks. Further duration/concurrency consolidation should preserve scientific coverage, old red evidence and exact-head provenance. Required release gates remain independent scoped review, explicit owner approval, guarded merge, reviewed-tree equality and actual post-merge CI. See the [audit report](post_m40b_audit.md) for reproducible findings and remaining work.

## Engineering conventions (post-M01)

These conventions govern new work following the M01 collaborative assessment and Phase 0 stabilization; existing legacy backends are preserved unless modified by a separately authorized milestone.

- **Responding-conductor composition.** Responding conductors compose via Faraday region decomposition with one scalar state `(V_m, Q_metal)` per connected conductor, introducing a capacitance/elastance matrix only when two or more coupled conductors require it. Image charges are Green-function terms, never authored sources.
- **Units and gauge.** SI units throughout, one 3D Cartesian world frame, and `V(∞) = 0` for bounded systems. A single infinite conducting plane at nonzero `FixedPotential` is ill-posed under this gauge and is refused.
- **Free-charge bookkeeping.** Authored and reported conductor charges are free charges (`D = εE`). Under `FixedNetCharge`, face totals are accumulated directly from original authored charges without intermediate rounding—never reconstructed from a rounded potential or rounded subtotals.
- **Admission before coalescing.** Validate geometry and assign every source to a qualified nonmetal region before combining coincident sources; sources on walls or inside metal are refused even if their charges cancel.
- **Precision and explicit refusal.** Report an exact zero only from a symmetry/boundary invariant or exact cancellation. Vector accuracy uses the norm of the final field; explicitly reported signed components require a resolved sign. Uncertifiable values return an explicit numerical refusal (distinct from scope refusal and ideal singularities).
- **Truncated series.** Report local values only inside a domain certified by a derived remainder bound on the reported total; never return a silent wrong-sign value.
- **Scale hygiene.** Structure kernel arithmetic to avoid unnecessary intermediate overflow or underflow (`math.hypot`, `k*(q/r)` before multiplying, image radius `a*(a/d)`).
- **Energy reporting.** Report field energy, supply work, external-agent work, and an explicit energy residual separately; label "heat" only under a declared dissipation model.
- **Growth rule.** Author new JEE items against shared layers and extend the smallest shared layer for genuinely missing physics rather than creating per-question modules. Homogeneous capacitors are acceptance cases of radial multi-conductor coupling, not a separate capacitor family.
