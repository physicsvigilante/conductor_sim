# M36e — Preregistered textbook-answer visibility for the authentic p26 solid sphere

**Frozen before implementation.** Base M36d owner-authorized merge on main `8bca8e5d1c157e05b3f04e6185e927bd7ee79941`, reviewed native Git tree `523d02e95dbb4791acd630f9c535eb552262c606`. M36d's true source solid `POTENTIAL_CONDUCTORS1.pdf` p26 Illustration 1 is independently verified; no additional original problem is counted.

## Objective

Address three **presentation-only, nonblocking** findings from independent M36d Chat B R2 [review in review_sandbox PR #3](https://github.com/physicsvigilante/review_sandbox/pull/3):

1. In the real p26 source viewer the printed two requested observable values **V(R/2)** and **V(3R/2)** must be visibly juxtaposed, with units and the actual source/teacher status.
2. The generic legacy label “Induced surface-charge profile” is inaccurate for a lone isolated solid with fixed net Q; **within this viewer only**, use “Surface-charge density” and a physical *outer face* label. Do not relabel unrelated induced-charge scenes globally.
3. Acknowledge jagged/white sampled-grid fill near the circular interface; use a qualified increase in actual grid sampling resolution and compare real browser outputs without inventing interpolation or a new field solver. Do **not** claim pixel-perfect anti-aliasing or actual physical surface subpixel certainty.

## Fixed scientific and UI contract

- Preserve the real M36d public `MixedRadialBackend`, `SolidSphere` and existing M5 viewer; **no production solver/physics/source-PDF mutation, no general renderer changes, no new scene graph or new model shapes**.
- Preserve exactly three actual native source/teacher states: original source-shaped solid `R=.1 m, Q=+1 nC` with explicitly *teacher* numeric values; signed/translated teacher revision `R=.12m, Q=-2nC, c=(.025,0,0)m`; teacher grounded empty-space solid `R=.08m`. Original textbook `R,Q` remain symbolic.
- A **prominent state-aware p26 answer panel** must display simultaneously `V(R/2)` and `V(3R/2)`, from **actual** solver saved native `potential_at` outputs, plus R, charge/source constraint, `E(R/2)`, outer surface density `σ` with signed units, and original-vs-teacher distinction. It must actually update on slider change; static fallback shows the original state. Correctness is NOT certified by merely printing an analytical formula in HTML.
- The reference independent SI checker should continue to validate raw original native observations; new browser tests must compare displayed numeric values for **each frame** against the exact native manifest, and check provenance/physical labels. HTML answer card serialization must safely escape text, never interpret an authored source name as markup or code.
- Rendered embedded frame must remain `SolidSphere`, `inner_radius_m=0`, one outer sigma curve and no source masks, and the actual field grid should have denser *real* solver samples than M36d's 37×37 preview; report its grid size accurately. Increase resolution only while retaining complete offline browser, FEMM/Python and source regression CI.
- Intentionally no current/charge-transport timeline, energy dissipation or new physics. Not an extra original question, not an extrapolated JEE-30 success rate.

## Acceptance and independent review gates

1. Source-pinned standalone runner actually writes real three-state HTML/data and passes untouched M36d separate source-free Gauss reference; 35+ existing tests still pass.
2. Explicit adversarial **display-integrity** checks: tampered or stale answer card values, source/teacher states mismatched, misleading units/charge type, fabricated original source numbers, fake hollow geometry and spurious elapsed-time animation should fail. Browser reads real visible DOM and compares against saved native state values.
3. Actual offline Chromium opens all three source/teacher frames, checks density heading/profiles, label and real animated-state switching **without** time semantics, scans HTTP/JS errors, saves original and teacher screenshot evidence. Where possible evaluate screenshot readability; automated checks do not prove classroom comprehension.
4. Full exact-head Python 3.10/3.12, real FEMM5, historical Chromium, M36b/c/d source regressions successful. Independent separate GPT-6 R2 reviewer on immutable HEAD/TREE authors unfamiliar checks. Preserve first failed attempts.
5. **DRAFT PR, no merge** until source-scoped independent PASS, separate explicit owner merge authorization, expected-head merge guard, verified reviewed Git TREE on main and successful actual main-push CI.

**Denominator:** one previously audited p26 original solid sphere; improved textbook-answer presentation, **zero additional original JEE apparatus claimed**.
