# JEE-12 — frozen source-fidelity, physical-answer and educational-usefulness evaluation charter

**PREIMPLEMENTATION CHARTER ONLY.** Tracking [issue #108](https://github.com/physicsvigilante/conductor_sim/issues/108). This document commits the case selection and scoring semantics before any per-case evaluation implementation. It is *not* a claim that twelve original questions have been simulated, solved or independently reviewed.

## Precise release baseline and separation

Base = fully verified post-M40b audit [PR #104](https://github.com/physicsvigilante/conductor_sim/pull/104) merged to main at `337001f2e0d1ce62c87fa4352993c4a074cfbde6`, native Git TREE `25e91e0fd53749bf6c6c578583310b219e1ecb4f`; main postmerge 30/30 workflows and 43/43 check runs completed successful. M40c [PR #105](https://github.com/physicsvigilante/conductor_sim/pull/105) has independent scoped PASS at `33a1c677fd9c68404fc64b44496c08fb2c3cee7b`, native tree `02688024e08f8f0726b5399233e50daa0744c855`, but remains **draft/unmerged pending separate owner approval**. JEE-12 must NOT assume M40c is released or silently merge it. Keep this assessment an independent branch and a distinct PR.

Historical coverage source = `docs/m36a_jee30_coverage.json`, schema `conductor_sim.m36a_jee30_candidate_coverage.v1`. Its 30-case `supported_scoped/partial/unsupported` routing is **historical planning-only, not achieved verified case coverage**. Later historical M36b/M36c/M36d/M36e/M37b claimed independent source replays within explicit bounds, sometimes on shared apparatus. Do not rewrite their evidence or add overlapping counts into a success score.

Frozen source-document bytes:
- `02_conductors_half_size.pdf`: SHA256 `9dff3c02a9f1848d2c3d99cb1a6e7a44dc7f5a51f7a3a2b8cc92d10ce1fbca5d`; workbook, not automatically an official JEE exam paper.
- `POTENTIAL_CONDUCTORS1.pdf`: SHA256 `8d0d9b5cf821219058cfc4b7685a39b777e460944df44c3f9cffcb68157a60e5`; notes/illustrations, not automatically official JEE exam problems.

## Exactly 12 original source-anchor IDs, historical source order

| ID | Original printed anchor | Reason for inclusion (not a success claim) |
| --- | --- | --- |
| `p01-single-08` | Workbook p1 Single Correct Q8 | uniform-field work and signs |
| `p02-single-19` | Workbook p2 Single Correct Q19 | real stopping event vs one-sided sheet approximation |
| `p03-single-27` | Workbook p3 Single Correct Q27 | actual solid and concentric shell static energy release |
| `p07-multiple-13` | Workbook p7 Multiple Correct Q13 | whole-face Gauss vs unknown local sigma |
| `p08-multiple-21` | Workbook p8 Multiple Correct Q21 | partial grounded-sphere approach/current domain |
| `p11-linked-28` | Workbook p11 Linked Q28 | refusal of irregular nonconcentric cavities despite topology-level Gauss |
| `p12-linked-40` | Workbook p12 Linked Q40 | ring potential/work and transverse field honesty |
| `p16-archive-numerical-01` | Workbook p16 archived 2020 question | disc field and oil-drop switching partiality |
| `p26-conductor-potential` | Notes p26 Illustration 1 | real solid-sphere p26 versus old hollow proxy |
| `p27-cavity-two-shells` | Notes p27 Illustration 4 | exact point charge and two concentric shells |
| `p69-cavity-central-vs-shifted` | Notes p69 Illustration 7 | named source shift and honest teacher-only hypothetical boundary |
| `p70-three-cavities` | Notes p70 Illustration 9 | physical spherical qualified subset versus actual irregular/unspecified source geometry |

The set intentionally includes cases historically labeled `supported_scoped`, `partial` and `unsupported`. Do not cherry-pick only successes or silently reinterpret the original geometry to improve a metric. Retain all twelve IDs regardless of eventual evaluation outcome.

## Qualified answers, visual evidence and interpretation

For every case, independently inspect the original PDF page's *actual* printed conditions/figure, identify source-given symbolic and numerical facts separately from any teacher-specified values, and preserve stable original page/question/answer identity. Record a source anchor, PDF byte digest, precise requested observable(s), geometry and physical assumptions. If unclear, label `interpretation_unverified` with the page-specific reason; no fabricated solution.

Separate these **orthogonal** axes, never a single unsupported success percentage:
1. `interpretation_fidelity`: original full problem vs verified subset vs ambiguous/unverified;
2. `physical_correctness`: independently checked actual native SI/equations vs not independently checked vs fails;
3. `original_answer_completeness`: all source asked quantities verified vs subset vs no source-specific answer vs unknown;
4. `rendered_observability`: genuine HTML physical geometry/labels/answer visible vs only native data vs unverified;
5. `educational_utility`: reasoned visible teaching goal/limitations (qualitative, evidence-grounded);
6. `admissibility_refusal`: correct unsupported-source or mathematical domain refusals, no silent proxy claims.

Do not call a `supported_scoped` M36a routing a verified original problem replay; no arbitrary result or browser preview counts as a solved original answer. Do not sum multiple linked questions as separate *apparatus* or distinct scenarios as separate *original case* successes. Preserve hypothetical static scenarios as teacher-authored, not workbook-given. Distinguish 3D disjoint sphere projection from 2D cut plane and static before/after from charge motion/current.

## Engineering, composability and safety

Use existing qualified public APIs and the source-backed M36b/M36c/M36d/M36e/M37b tools when they actually match. Prefer a **small evaluation harness/manifest** over a generic authoring framework or scene graph. Nothing in the charter authorizes new source physics, invented FEMM 3D, irregular BEM, E/V map or recasting real conductor geometry as spherical without a disclosed proxy.

Keep original PDFs, frozen charters, released solver/runtime, historical reviews and source records unchanged. A first implementation should build a machine-readable evaluation cohort with explicit `not_evaluated` statuses, exact source ancestry, independent replay provenance and fail-closed schema checks. Advance the twelve one case at a time with actual page inspections and original-output verification. Preserve failed attempts, don't label a pending test PASS. Keep test budgets small with actual dependencies.

Only merge after fully successful exact-head Python 3.10/3.12, applicable browser/FEMM/source CI, one tightly scoped new reviewer-owned source-fidelity review, **separate explicit owner approval**, expected-head guarded merge, exact reviewed Git-tree equality and fully successful actual postmerge main-push checks. A charter alone has no claim of evaluation results.
