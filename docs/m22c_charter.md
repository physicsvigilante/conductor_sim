# M22c — Preselected finite-rod reflection qualification

**Approved extension; charter frozen before any production-code changes.**
Base: `e312c1d53661c125d6677439460141d6d0c1109b`; M22b read-only diagnosis: `M22b_report.md` (local evidence).
The first accepted implementation may extend only the existing prescribed
motion-plane **qualification** and its finite-rod reflection proof. Do not
change electrostatic source formulas, total-force integration, runtime guards,
energy conventions, old scenario schemas or current point-symmetry semantics.

## Implementation-cost hypothesis

A single reusable global reflection certificate over the complete finite
rod inventory should admit physically valid off-plane mirrored rod pairs with
no special production code for each arrangement. Each uniform line source
has fixed signed total charge `Q`; reflection must preserve the *entire
unoriented finite segment* and its signed Q, hence its line-charge density.

The existing motion qualifier already accepts individual rods with both
endpoints in the plane or a symmetric normal rod centred on the plane.
All such legacy cases must continue to qualify without an additional
certificate. A new certificate is required **only** for rods not already
qualified individually. No source positions, charges, force components,
physical planes or numerical tolerances may be projected, snapped or repaired.
Real source guards, full 3D force/energy observations and mathematical
nonrelativistic assumptions remain unchanged.

## Preselected motivating positive cases

1. **P1 (M22b case, parallel mirror pair).** Physical plane y=0; two fixed
   `UniformRodField(+1.5e-9 C)` segments:
   R1=(-.05,+.07,-.02) to (+.05,+.07,-.02) m,
   R2=(-.05,-.07,-.02) to (+.05,-.07,-.02) m.
   Probe starts (0,0,+.10) m and v=(+.03,0,0) m/s; any valid nonzero
   signed charge/mass must be disclosed. Confirm strict reflection, full
   3-vector Coulomb force/potential, numerical motion in y=0, energy,
   source geometry in offline viewer. No artificial guide.
2. **P2 (tilted segments, endpoint reversal).** Plane y=0.
   R1=(-.06,+.085,-.03) to (+.055,+.120,+.04) m,
   R2=(+.055,-.120,+.04) to (-.06,-.085,-.03) m, each -0.8 nC.
   Include a separate in-plane fixed point charge and an in-plane applied
   uniform field; integrate oblique in-plane launch. Reversal of the second
   rod endpoint order must not affect qualification or source physics.
   Verify independently by uniform line-charge quadrature.
3. **R (RESERVED — unexecuted until the implementation candidate is
   frozen).** Multi-pair arrangement with two distinct mirrored finite-rod
   pairs with different signed charges and lengths, an independently
   admissible in-plane finite rod, in-plane fixed point and uniform field,
   plus in-plane uniform gravity. Record exact numerical recipe in
   `tests/m22c_reserved_case.json` **before** production implementation
   and never run during development. Run once against the frozen candidate.
   Any subsequent repair invalidates R as an untouched reserved test.
   This case is explicitly **author-exposed, not blind or independent**.

## Preselected negatives and retained contracts

- Missing pair, unequal signed Q (including one-ulp charge difference),
  endpoint mismatch, unequal segment length and inadequate world-coordinate
  resolution; no pointwise-normal-force-cancellation substitute for geometry.
- One-to-one signed multiplicity including duplicate segments; endpoint
  reversal and source permutation must not change physical admissibility.
- Preserve individually admitted in-plane and normal-centred rods, existing
  point-reflection certificates and their original refusal behavior.
- A mirrored rod pair does **not** justify an off-axis ring/disc evaluation,
  a transverse applied E/gravity, a transverse initial velocity, or a new
  conducting/charge-response model.
- Existing singular/source-segment guards and saved full-vector transverse
  force checks remain effective. Source/particle revision must requalify;
  changing one partner invalidates the assembly.
- Geometry matching tolerance is bounded by floating-point resolution of
  physical coordinates and local span, not the solver ODE tolerance.
- New independent references: signed numerical Coulomb line integrals
  at multiple points (not only probe launch), physical reflection identity,
  `E=-grad(V)`, alternate time integrator, translation/rotation covariance,
  and energy conservation. FEMM is conductor **nonregression**, not a
  numerical reference for these nonresponding fixed rods.
- Existing M5/M19/M20/M21/M22a authored outputs and test hashes must be
  preserved. Real offline Chromium validation must check source identity,
  actual playback, notes for off-slice rods and zero JS/network requests.

## Acceptance and stopping

P1 and P2 require independent physical validation and a real browser view.
The frozen reserved R case must be executed only after candidate freeze,
with a separate audit record. Keep a truthful account of test revisions
and failures; do not report R as untouched if code is changed afterward.
Full ordinary Python suites (3.10/3.12), focused M21c/M22a/rod tests,
97 real FEMM tests and existing browser suites must remain green.
Two positive cases plus R demonstrate **bounded** composability only,
not arbitrary symmetry sources or real exam-coverage statistics.
