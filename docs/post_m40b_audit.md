# Post-M40b project audit: defects, composability and JEE relevance

**Developer audit and correction candidate — not independent PASS or release authorization.** Baseline is the released M40b main commit `4bae2f6be5a1689b9e1dc80d2dd146b77d366a3e`, native tree `5ea6098d463eec12fbc184ab3a6fca2f27699e9a`. [Frozen correction scope](post_m40b_audit_charter.md). M40c's charter branch is deliberately untouched. This report is a broad, risk-based audit; it is not a proof that every line or every supported configuration is error-free.

## Executive judgement

The project is meaningfully composable **within declared physical regimes**, and its electrostatics teaching direction remains JEE-relevant. The most important architectural success is not forcing responding conductors, prescribed-source superposition, topology-only Gauss totals and spherical local-density observations into one misleading universal solver. The main risk is now maintenance/authoring complexity: many milestone adapters, private imports, repeated validation and stale navigation can make reusable capabilities harder to discover than to implement.

This candidate repairs two reproduced correctness defects, restores current documentation, and closes demonstrated downstream CI-trigger gaps. It does not introduce a general scene graph, a new electrostatic law, an arbitrary-geometry solver, or a new source-PDF solution claim. Further feature work should follow the corrections' independent review and release, rather than building M40c on an acknowledged defective baseline.

## Scope, identity and method

The complete released source was obtained from main workflow `44577369591`, artifact `14556359640`, ZIP SHA256 `3129b8a8e2b61d20dbe600b5be13be15b314a30ff656fa5e62b664d866992c2f`. Reconstructing the extracted Git tree reproduced the native released tree exactly. The baseline inventory contains **87 production Python modules / 22,828 lines, 112 test modules, 87 tools, 61 examples, 156 documentation files and 39 workflows**. Line/module counts are inventory, not test-coverage percentages or quality scores.

The audit inventoried the whole repository and traced representative architecture paths: physical model/backend admission, planes and conductor images, prescribed fields, force/energy, 1D/2D motion, source revisions/structural lineage, graph/fork observations, source-backed replays, multicavity Gauss/sigma and their viewers/checkers. New adversarial tests were executed against the unchanged baseline before corrections. Original tests and selected native source-backed replays were executed locally. Original scanned PDFs were hash-preserved; every problem was **not** reinterpreted from its scan in this audit.

## Reproduced findings and corrections

### A1 — Tiny direction ratios silently distorted physical geometry

**Correctness defect; repaired in candidate.** A valid direction `(ulp(0.0), ulp(0.0), 0)` describes the same ray as `(1,1,0)`. The old plane, 1D motion-axis, 2D direction and slice helpers divided by a heavily rounded subnormal `hypot`, producing `(1,1,0)` as an allegedly unit vector. Its length is square-root two. This is not just a displayed rounding issue: plane image locations, potential, electric field, force, surface density and projection metre scales can change.

For a grounded plane with that raw normal, a +1 nC source at `(1,1,0)` m and observation `(2,2,1)` m, the old potential was `2.6203043124868897 V`, versus independent mirror geometry `3.1270796564964125 V`. The force components were a factor of four too small. A slice point authored at `(0.25,-0.125)` m had world distance `0.375 m` rather than `0.2795084971874737 m`.

A small private `unit_direction` helper scales components **only when the computed norm is subnormal**. Four demonstrated consumers now use it. Ordinary arithmetic is retained bit-for-bit; zero/nonfinite/overflowing norms still refuse. Positive and negative source tests derive E/V/force/sigma from an explicitly placed real charge and mirror, rather than calling the production mirror helper for expected values. This is a shared fix for a repeated concrete defect, not speculative abstraction.

M35a had corrected a similar issue in the *prescribed-source* direction path, but explicitly did not cover every independent normalizer. The present correction does not rewrite that history. **M40a original near-unit raw query rays and their Decimal kernel are unchanged**: those have different physical precision semantics and must not be normalized prematurely.

### A2 — Atlas witness could certify an undisclosed missing cavity

**Evidence-completeness defect; repaired in candidate.** Remove the final displayed cavity and its corresponding isolated/grounded face-witness entries from an authentic M40b manifest, but leave the full three-cavity native geometry/source records intact. The old standalone verifier returned success for two cavities. The attack works for an omitted empty cavity and for an omitted charged cavity after valid source/geometry ordering.

The verifier now requires the displayed cavity count to equal both the authoritative native void inventory and native Gauss cavity inventory. Existing per-row identity and physical-value checks remain. Genuine complete manifests pass. This is distinct from the documented limitation that a JSON-only checker cannot authenticate separately edited HTML or the historical origin of a wholly self-consistent rewrite; that limitation remains explicit, not “fixed” by a count check.

### A3 — Current navigation described a release several milestones behind

**Documentation defect; repaired in candidate.** README and the “current” capability map still said released M34b/draft M35a; architecture still called M5 a candidate and M4b the latest FEMM evidence. The map treated the old sweep issue as open although [M35b PR #82](https://github.com/physicsvigilante/conductor_sim/pull/82) is merged.

A shorter teacher-facing README, the current capability map and architecture now distinguish released M40b, this unreleased correction candidate, and charter-only M40c. The former long README is retained byte-for-byte at the repository root as `README-MILESTONE-HISTORY.md`, preserving its relative links. Historical charters, review reports, old pilot counts and original release evidence are not overwritten. Future releases should update the current index once, rather than rewriting every historical milestone document. `0.1.0` alone is not a release-source identity.

### A4 — Newer downstream workflows missed upstream scientific changes

**Validation coverage defect; repaired conservatively in candidate.** Twenty explicit dependency assertions failed on the released workflow path filters. Examples: a change to `model.py` could miss a newer consumer's independent reference workflow; M40b's filter omitted `spherical_multicavity.py`, `cavity_gauss.py` and their standalone oracles; M37b omitted its no-JavaScript hostility checker.

Seven late-milestone workflows now include core source/configuration changes and the relevant upstream reference/checker tools. This is additive: no prior physics, FEMM or browser step was deleted. Twenty regression cases demonstrate the old misses and the corrected coverage. It intentionally prefers extra checks to an unsafe “optimized” dependency graph; a later measured consolidation may reduce duplication without weakening coverage.

## Local execution evidence and release limitations

The correctly installed **unmodified baseline** passed **2,560 non-FEMM tests**, with **97 real-FEMM tests deselected**. Thus existing green tests did not detect the newly reproduced defects. The fresh correctness regressions then produced **17 failures / 6 passes** on that same source; the twenty dependency checks all failed as intended.

The repaired numerical/evidence source passed **2,583 non-FEMM tests**, and the full new targeted set passed **43/43** (23 numerical/evidence/compatibility cases plus 20 workflow-dependency cases). The final full-suite run incorporating the dependency tests completed with **2,603 passed, 97 deselected and 13 collection warnings**, exit code zero. Those warnings concern imported dataclasses whose names start with `Test`, not failed physics assertions. All local runs use **Python 3.13.5**; they do not stand in for required Python 3.10/3.12 hosted acceptance.

The first baseline harness attempt lacked an editable installation and produced two CLI-import failures; that setup error and its log were preserved. After installing correctly, the complete baseline suite passed. A local Chromium attempt was blocked by managed `ERR_BLOCKED_BY_ADMINISTRATOR`; browser security/policies were not modified. Consequently this audit does **not** claim new locally successful browser acceptance or real FEMM execution. Hosted current-head Python, real FEMM and Chromium remain mandatory before independent PASS/release.

Fresh source-backed generation plus standalone reference commands succeeded for M36b, M36c, M36d and M37b. M36b checked seven scoped apparatus; M36c checked seven apparatus/eight source anchors, including honest partial/refused controls; M36d checked one actual solid-sphere source apparatus plus two teacher revisions; M37b checked 31 prescribed Kelvin states for one source apparatus. These are **different, partly overlapping denominators**, not additive unique-question coverage. The 30-case inventory check verifies inventory/source metadata, not thirty successful simulations.

Reproduce the new contracts from an installed checkout:

```sh
python -m pip install -e ".[test]"
python -m pytest tests/test_post_m40b_audit.py tests/test_post_m40b_workflow_dependencies.py -q
python -m pytest -m "not femm5" -q
```

The dedicated audit workflow checks out literal candidate HEAD, preserves source/tree/PDF hashes, replays the expected red tests against the pinned baseline, then requires all 43 corrected tests to pass on Python 3.10 and 3.12. Expected baseline failures are negative-control evidence, not successful baseline scientific results. The normal inherited workflows remain separate gates.

## Is composability sensible rather than forced?

**Overall: yes, at the physics and model-revision layers; caution at the orchestration/maintenance layer.** `ElectrostaticProblem` plus backend qualification permits new authored conductor arrangements within each supported domain. `PrescribedField` supports legitimate fixed-source superposition. Named-source revision/reassembly retains identity and requalifies whole systems. Motion/event observers and offline viewers consume actual physical results rather than substituting browser calculations. M39a/M40a deliberately distinguish topology from geometrically qualified local physics.

These are useful forms of reuse. A teacher changing charge, location, launch, source inventory or observation should not have to rewrite a force law. Conversely, an arrangement requiring new boundary conditions or a new mathematical solver is **not** merely a missing list-composition operation. Prescribed fields and responding conductor solutions must not be naively added; field-energy conventions cannot be unified just because both APIs contain a quantity called “energy”.

The inventory found **234 relative private-symbol imports**. That count signals coupling, not 234 defects. Several milestone adapters repeat result validation, serialization and dispatcher logic; the public import surface also eagerly imports presentation/orchestration. The right response is limited internal services with explicit contracts and compatibility tests, not a universal physics graph rewrite. Graph histories are valuable for real multi-stage teaching inquiries but should remain optional for a simple sphere or plate example.

Use this rule for the next refactor: identify at least two actual consumers, prove they share physical/evidence semantics, preserve valid outputs, and measure the maintenance or runtime benefit. Otherwise leave qualified modules separate. The four-consumer normalization correction meets that test. A broad scene graph, generic plugin registry or merged solver does not currently have enough demonstrated benefit to justify its risk.

## Is the project staying JEE-relevant?

**Yes for a substantial, explicitly bounded electrostatics teaching subset; not as a complete JEE simulator or measured question solver.** The official [JEE Main 2026 syllabus](https://jeemain.nta.nic.in/document/syllabus-2026/) and [JEE Advanced 2026 syllabus](https://jeeadv.ac.in/documents/jee-advanced-2026-syllabus.pdf) include Coulomb fields/superposition, Gauss law, potential/energy, conductors and capacitance. These are direct matches for the project's core questions. The syllabi also include dielectric and capacitor-network topics; current conductor examples are not a general solver for all of those configurations. This comparison uses the explicitly named 2026 documents, not an unverified claim about a newer syllabus.

M39a/M40a/M40b remain relevant when used to teach shielding, induced whole-face charge versus local density, signed-neutral cavities, off-centre sources and grounding. Their sophisticated near-wall numerical work protects truthful output but does not itself broaden original-question coverage. The project correctly claims **zero new original scanned JEE questions** from those teacher-only milestones. M40c's source relocation is educationally justified if kept as a small static what-if feature, not expanded into another general orchestration framework.

The next teaching priorities should be: make existing source-backed demonstrations easy to find/run; retain source-given symbolic facts separately from teacher numerical choices; expose simple charge/position/grounding comparisons with readable SI and genuine before/after physics; maintain a current topic-to-demo map with explicit partial/refusal boundaries. Only then choose the next physical domain from recurring unmet textbook questions. Generic capacitor networks/dielectric configurations need their own qualified physical contract, not visual substitution by an existing shell model.

## Open risks and sensible follow-up

- **Legacy Boolean coercion:** some older constructors intentionally accept/coerce Boolean SI values, with tests depending on that behavior. New strict factories and observers reject raw invalid input, but information can be lost after coercion. Do not silently break those constructors. Plan a versioned strict-authoring migration and characterize callers first.
- **CI efficiency:** the baseline has 37 of 39 workflows without a concurrency block and nine without explicit workflow-level permissions; omission alone does not establish an overprivileged token. General Python and visualization workflows repeat the full suite. Measure job costs, retain one full regression gate plus independently scoped scientific/browser/FEMM gates, and introduce safe superseded-PR cancellation without deleting first-failure artifacts. Do not optimize by reducing verified physics coverage.
- **Release protection:** the live main-branch response reported `protected=false`. Exact-head process safeguards are helpful but not equivalent to server enforcement. Required checks/review policy and protection against accidental direct pushes need owner configuration; this audit does not change administration settings.
- **External evidence:** post-export HTML authenticity and historical third-party JSON origin require a trusted anchor if promised. Internal consistency is not cryptographic provenance. The current boundary should remain visible to consumers.
- **Coverage/maintenance:** other independent normalizers, extreme coordinate conditioning and adapter contracts are not exhaustively proved by these focused repairs. Historical drafts, version labeling and source-coverage aggregation deserve housekeeping, but deleting old branches/evidence or changing schemas is outside this correction.

**Recommendation:** release this bounded correction after one independent exact-source review and separate approval; then resume the narrow M40c teaching inquiry. Favor real teacher-facing exploration and consolidation with demonstrated reuse over further generic framework layers. Never infer “error-free” from a green suite.
