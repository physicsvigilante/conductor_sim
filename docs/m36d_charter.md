# M36d — Preregistered source-authentic standalone solid conductor sphere

**Frozen before implementation.** Released independently reviewed M36c base/main `6e6f53aa6762c327f58ef709379c41b1477c0931`, native Git tree `c43134db1726dc2f1ff48b0b9a560dff090bf56d`. The M36a original 29+1 PDF source inventory and original PDFs remain unchanged. **No broad scene graph, arbitrary 3D induction, FEMM solver change, generic physics framework or time-current model.**

## One printed JEE-anchored physical problem and bounded model

The printed [`POTENTIAL_CONDUCTORS1.pdf`](../POTENTIAL_CONDUCTORS1.pdf) p26 **Illustration 1** depicts a *SOLID* charged conducting sphere radius `R`, net charge `Q`, asking for `V(R/2)`, `V(3R/2)`, and ordinary interior `E=0`. M36c [independent R2 review #02](https://github.com/physicsvigilante/review_sandbox/pull/2) confirmed an existing M6 benchmark draws a *hollow shell proxy* `SphericalShell(.09,.10)` instead. Its radial V/E observers coincide, but its shaded geometry is not the printed source shape.

M36d must render a TRUE `SolidSphere(R,FixedNetCharge(Q),name=...)` using existing physical geometry, actual `MixedRadialSolution` potential, field, true one-sided surface jump and uniform outer charge density. No imaginary inner surface or hollow cavity, no point source, no internal charge, no time-varying charge transfer and no external conducting boundary. `R=.10 m` and `Q=+1 nC` are **teacher numerical illustrations**, not original printed givens. Source is symbolic.

**Qualified backend change:** minimally admit **exactly one** `SolidSphere` with `FixedNetCharge(q)` or `Grounded()` under the already established `MixedRadialBackend`, no other shapes or point sources. Preserve existing 1-solid+1–7-concentric-shell physics and no unrelated single-shell/single-rod admission. Keep current solution type, name identity, existing visualization adapter and renderer. No new standalone solver or decorator.

**Required independent SI reference**, using `k=1/(4π ε)` with source-independent constants: `V(r≤R)=kQ/R`; `V(r>R)=kQ/r`; `E(r<R)=0`; `E(r>R)=kQ(r-c)/|r-c|³`; metal boundary `E` has distinct one-sided limits and is refused exactly at `r=R`; inner face `0` (nonexistent, not a physical inner surface), outer net `Q`; `σ_outer=Q/(4πR²)`; electrostatic exterior self-energy derived independently `U=kQ²/(2R)` (not falsely claimed an engine-computed wire released energy).

## Physical revision and original-source honesty

One genuine original apparatus with its source example `R=.10m, Q=+1 nC`. One **separate teacher hypothetical** at the same stable source name with actual changed radius and signed net charge (e.g. `R=.12m, Q=-2 nC`) is solved anew, not interpolated or counted as a second original JEE question. A no-source grounded sphere `Grounded()` must also be physically verified as zero-charge, zero-field, zero-potential if supported. The original and teacher numerical choices must be visibly labelled and the source's printed solid geometry faithfully drawn with `inner_radius=0`; no hollow-proxy warnings on the new actual-solid viewer. Legacy M6 original historical benchmark and M36c record stay unchanged.

## Test, browser and review gates

1. Clean-source regression tests for **isolated +, isolated −, zero charge, Grounded**, translated centers, physical radial V/E and Gaussian surface density, identity, boundary field refusal, source singularity refusals, and a nonqualifying case `PointCharge` inside/outside standalone solid. Confirm old mixed two-conductor solver still works and M6 historical shell example is **unchanged**.
2. A source-specific **native runner** exports observed values from real `MixedRadialBackend` solves including multi-state source/teacher parameter change, *true geometry metadata* and actual offline HTML, raw sampled field values. It must not export self-manufactured numeric answers as if native observations.
3. A completely separate **stdlib-only reference checker** (no production imports) independently evaluates all source and teacher native values, signed E, potential, σ, surface identities, source geometry, whether the new solid is not a cavity, actual teacher revision; hostile mutated datasets must be refused.
4. Actual offline **Chromium** opens generated solid-sphere viewer and checks real visible labels/graphics and true stored solid cross-section (`inner_radius_m=0`, `outer_radius_m=R`), no network, relevant source/teacher state switching. Screenshot and numerical artifacts retained; no hidden JSON-only source/teacher disclosure.
5. Full current-HEAD Python 3.10/3.12, actual FEMM5, historic Chromium and M36b/M36c CI must PASS. The full solver physical acceptance is original-source fidelity plus qualified numerical Gauss-law; viewer success is an independent presentation gate.
6. Independent separate-context **R2 scientific review** must author/execute unfamiliar qualified standalone solid sphere variations and negative out-of-scope checks; report original p26 PDF and visual source authenticity. PR remains **DRAFT and UNMERGED** pending no-blocker reviewer PASS, separate explicit owner merge approval, expected-head guarded merge, native Git tree equality and all actual main-push CI complete success.

**Denominator:** one original p26 physical source apparatus, several teacher variations, zero additional counted original JEE questions. Prior M36b/M36c selected cohorts remain historic and are not extrapolated to `N/30 solved`.
