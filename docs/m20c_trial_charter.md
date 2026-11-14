# M20c — Predeclared cross-source scenario authoring trial

**This charter is committed before implementation or grading.** It is a
curated, reproducible, developer-authored trial, not a blinded or random JEE
sample. The starting setups are fresh teacher hypotheticals in SI units.
The goal is to challenge M20b outside its Ring A/B example.

## Locked method
- No modifications to production physics, solver, scenario API, renderer,
  validation, numerical defaults or exclusions to make cases pass.
- Build each baseline with existing PrescribedScenario and supported field
  components; use stable source IDs through revise_scenario; independently
  integrate both histories; use unchanged M20a comparison.
- Outcome: paired (two complete verifiable comparable histories); refused
  (unsupported physical/structural request before producing results);
  admitted-not-comparable (valid model but censored run or incompatible
  certified view). Preserve the reason and any exception.
- Independently check analytical constant-field motion, interior volume-sphere
  harmonic motion, shell interior E=0 and qV shift, Coulomb superposition,
  rod quadrature, and exact disc-axis E/V.
- No thin-shell conductor induction, fabricated off-axis disc/ring forces,
  moving sources, or source response.
- Every card must be attempted, including blocked requests, without silently
  dropping a refusal. Numeric settings may be calibrated before evaluation
  only to avoid unintended singularities; disclose changes.

## Six new complete comparison cards
| Card | Teacher-authored physical setup | Requested change | Independent reference |
|---|---|---|---|
| U1 | Transverse uniform-field deflection | Double test mass | Exact parabola |
| V1 | Negative probe inside uniformly charged volume sphere | Increase source charge | Interior harmonic oscillator |
| R1 | Point probe flying near a fixed charged finite rod | Increase rod charge | Coulomb line integral |
| P1 | One fixed point source and one uniform field; 2D motion | Increase point charge | Superposition E/V |
| S1 | Particle entirely inside uniformly charged thin nonconducting shell | Double source charge | Zero E; unchanged path; shifted qV |
| D1 | Fixed uniformly charged disc, axis-only probe | Increase disc radius | Exact axial E and V |

## Six negative/boundary cards
| Card | Request | Expected result |
|---|---|---|
| N1 | Transverse launch in D1 | Refused: axis-only model |
| N2 | Put S1 probe on shell discontinuity | Refused: initial source guard |
| N3 | Add unbound source through U1 revision | Refused: membership fixed |
| N4 | Switch V1 fixed volume charge into grounded conductor | Refused: source-type/constraint out of scope |
| N5 | Change U1 initial position and demand M20a comparison | Admitted, then comparison refused for changed physical slice |
| N6 | Attractive fixed point approach reaches exclusion guard before duration | Admitted, real censored integration; comparison refused |

## Acceptance
Deliver case-by-case reproducible JSON with all twelve classifications, actual
source changes and ancestry, quantitative physics observations, explicit
refusal reasons, and original-model retention. Add ordinary Python regression
tests plus real browser checks on at least three non-Ring comparisons.
Keep FEMM as a nonregression check, not a claimed solver for these sources.
Report interpretation, correctness, coverage, usefulness and answer-completion
separately. These hypotheticals are not verbatim textbook problem solutions.
