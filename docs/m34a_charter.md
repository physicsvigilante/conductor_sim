# M34a — Frozen charter: stable oblique motion-plane normal across source revisions

**Preimplementation freeze.** Base is M33b owner-approved merge SHA 28f93622fd8cfb816e23ef36027992b1a5aaded8, whose reviewed source tree matches independently accepted M33b. Postmerge main checks were still in progress when this charter was first drafted. Inherited defect [issue #77](https://github.com/physicsvigilante/conductor_sim/issues/77) was independently reproduced on both earlier M33a and M33b. Commit this charter alone before source code. This is prospective work, not an implementation or separate review.

## Physical composability defect

For a legitimately supported non-dyadic oblique normal (1, 2, -1), original TestChargeMotionSpec normalizes it approximately to (0.4082482904638631, 0.8164965809277261, -0.4082482904638631). Numerically hypot of this normalized normal can equal 1.0000000000000002. Reconstructing a new source world using dataclasses.replace re-normalizes the already qualified motion-plane normal, changes its floating-point bits and triggers M22a exact unchanged-analyses verification. A real and fully qualified structural addition is rejected as though it changed initial conditions or numerical controls.

This is a false source-world **admission** refusal, not a physics-law/ODE problem or permission to simulate unqualified 3D motion. An unchanged qualified physical plane must survive ordinary structural and physical revisions at exact native tuple bits.

## Bounded numerical contract

- Modify only TestChargeMotionSpec motion-plane normal construction, not all generic unit-vector normalization, source/ring/rod laws, M22a exact source/analysis invariants, event detector, FEMM or scene graph.
- Numerical rule: after exact native finite 3-vector validation, if the norm computed by math.hypot differs from 1.0 by no more than math.ulp(1.0), retain the already-unit vector representation verbatim; otherwise use existing ordinary _unit validation and normalization. Normalizing a clearly non-unit input once must still happen; repeated legitimate constructor requalification must preserve already certified normal bits.
- The allowance is only machine precision and far smaller than existing physical plane-closure tolerances. Run full unchanged _certify, initial-velocity, gravity and source/domain checks on each new whole source world. Never use an unconstrained tolerance on M22a analysis comparisons or quietly accept real out-of-plane sources/initial states.
- All existing public scenario, graph v1/v2/v3, M33b inquiry, source ancestry and in-process evidence receipts remain intact. No silent JSON import or external cryptographic claim.

## Required acceptance

1. Original supported non-dyadic oblique normal (1,2,-1), fully planar three-component E, q/m initial conditions and gravity, genuine native trajectory and M24a signed events. Real M22a add, replace, remove must now construct physically admitted child with unchanged exact native motion plane, genuine source inventory, revision/parent hashes and independently re-integrated dynamics.
2. Structural→physical→structural, repeated revisions, source retirement and independent siblings must preserve ancestral identities and identical plane bits. M20b changes to other genuine physical properties must not corrupt the plane. Inherited source-ID retirement restrictions must still refuse reuse after retirement.
3. Independent direct q ΣE / m + g SI analytic three-component positions, velocities, force and signed root calculation at all actual saved timestamps, not using native field callbacks or RK calculations as the reference. Check true x/y/z differences, not a fictitious projection.
4. Refusals: zero/nonfinite/incorrect-shape normals, sizably nonunit input still normalizes, genuine out-of-plane launch, source or gravity, incompatible source symmetry, fake retained analysis edits and forged evidence. Characterize within and outside the one-ulp normalization window.
5. Reproduce old source failure on pinned released main and new source success with same input, preserve first negative CI evidence. Unaffected legacy exports and existing regression tests remain unchanged. No alteration to other plane normal (e.g. event-detector normal) semantics.
6. Full source-head Python 3.10/3.12, dedicated 97 real FEMM 5, Chromium and independent-source numerical/negative tests; pin exact source SHA/tree and retain artifact SHA256 evidence.

## Explicit non-goals

No expanded field law, source motion, responding conductor, new ODE or root locator, 3D solver, PDF text interpretation, arbitrary callbacks, scene graph or browser-time physics.

## Release process

Work on branch m34a-idempotent-physical-plane-normal from reviewed M33b merge, charter-only commit first, then a minimal normalization source change, targeted tests, example, separately implemented physical reference and CI. Open a draft PR. Require source/tree-pinned full green CI and one separate-context R2 scientific/adversarial review PASS with no blockers, then separate owner expected-HEAD authorization. Verify exact merge tree and all triggered main workflows before release.
