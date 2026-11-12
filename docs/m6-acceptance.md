# M6a acceptance — source-anchored JEE authorability benchmark

**Accepted for the bounded curated-case scope on 2026-11-12.**

## Exact source

- Repository: `physicsvigilante/conductor_sim`.
- Source: `POTENTIAL_CONDUCTORS1.pdf` at repository root, 71 physical pages.
- Verified SHA-256: `8d0d9b5cf821219058cfc4b7685a39b777e460944df44c3f9cffcb68157a60e5`.
- [Curated casebook](m6-benchmark.md) anchors each case to an original physical PDF page and illustration/derivation.

## What is accepted

- **19 distinct referenced problems:** 13 were authored using existing physical components and solutions; 6 were explicitly classified as unsupported, with missing-capability reasons.
- The 13 supported examples generated **43 independent analytical observations** that all passed. Observables include conductor charge, induced inner/outer totals, potential, field and local induced surface density.
- **Both analytical and FEMM capability checks** accept the authored physical examples (the latter's numerical results are independently tested for five representative cases).
- Invalid or unsupported examples terminate with explicit diagnostics rather than silently moving charges or swapping geometries.
- The benchmark is reusable from a CLI (`list`, `check`, `render`) and can export solver-independent offline HTML and sampled JSON with PDF provenance.
- The finite-thickness shell interpretation has independently checked Gauss-law potential corrections instead of pretending the shells have zero thickness.

## GitHub acceptance gates

- [Python checks on PR #7](https://github.com/physicsvigilante/conductor_sim/actions/runs/43234955519): success on Python 3.10 and 3.12, including source SHA-256 validation and new catalog tests.
- [Visualization and source checks on PR #7](https://github.com/physicsvigilante/conductor_sim/actions/runs/43234955681): success, including independent analytical benchmark and Chromium viewer interactions for a PDF-authored two-state induction example.
- [Pinned real FEMM 5 validation on PR #7](https://github.com/physicsvigilante/conductor_sim/actions/runs/43234955683): **40 passed, zero skipped**. This includes **35 previously accepted FEMM physics/visualization regressions** plus **5 new PDF-derived live cases**:
  - PDF p.26: charged conductor potential.
  - PDF p.29: outer-shell charging does not change the inner-to-outer voltage difference.
  - PDF p.62: grounded outer shell.
  - PDF p.63: grounded inner shell.
  - PDF p.69: central versus off-centre cavity charge.
- Local non-FEMM suite: **168 passed**, with 40 FEMM-only cases deselected. Two independently generated PDF-based HTML exports passed Chromium controls, hover, state reset, no external requests/JS errors and mobile-overflow checks.

All FEMM tests ran the **actual pinned `femm-qt v5.0.20261101`** Linux headless solver. Python-only tests and source snapshots are **not** represented as numerical FEMM evidence.

## Scope limitations

This benchmark is curated manually by an LLM assistant using typed Python. It does **not** prove that an autonomous natural-language parser can read arbitrary textbook diagrams or that a new LLM succeeds on unseen questions. It does not add new numerical physics, charge dynamics, or a scene graph.

Each runnable case is a *physics-faithful parameterized analogue*, not necessarily a digit-for-digit transcription of the original PDF. The source's zero-thickness conductor diagrams are represented by finite-thickness, nonintersecting metal shells and are checked against the correct finite geometry. Some static parameter changes are represented as precomputed separate equilibria; **no browser-time FEMM recomputation or charge motion occurs**.

The six unsupported cases were not simulated: charged rings, uniformly charged dielectric solid spheres, particle trajectories, charges exterior to grounded spheres, connected distant spheres, and multiple cavities in one irregular conductor. Their required new components/geometry/physics remain in the backlog.

## Next measured milestone

M6b: Ask an LLM to author previously unseen **held-out** JEE problems in a fresh Python module using only existing typed physical components, observation requests and the M5 export API. Score success, limitations detected, and number of solver/renderer source changes. Begin with external charges near grounded spheres as a new separately validated physics milestone after the authorability assessment.
