# M35a — Audit corrections and composable multi-stage physical exploration

**Preimplementation charter.** Base: released M34b `a5113fa4cc924ac10cbb35b2f2d7ac80e11f1bde`, Git tree `d0646c9e4d518008784c42c1fec1c48c81901a09`. This charter is committed alone before production edits. User requested a thorough project audit, correction or a plan for findings, then M35a. Source remains a separate draft branch; independent R2 and explicit owner merge approval are required.

## Audit evidence motivating corrections

A genuine source-pinned baseline audit, review_sandbox Actions run 44098354148, executed these reproductions on released code:

1. A native physical node mutated before a fork request can contain observation='typo-event'; the plan accepts it and issues a rejected-event-study record instead of rejecting malformed authoring before execution.
2. A native WorldComparison mutated before issuing a fork can reference the same new world on both sides and be exported as compared. The public constructor originally prohibits this; parent plans do not revalidate nested requests.
3. A finite nonzero motion normal `(ulp(0), ulp(0), 0)` becomes `(1,1,0)` with length sqrt(2), then changes when reconstructed. Normalization loses accuracy when its computed length is subnormal. M34a's one-ULP retention does not fix the first inaccurate normalization.

Additional audit work will examine numerical/field relationships, nonuniform motion, provenance, API boundaries, CI coverage/cost, packaging, stale release documentation, and repeated validation/serialization. Distinguish reproduced defects from design risks and untested coverage. Preserve baseline failures as evidence.

## Correction boundaries

- Revalidate nested native world nodes, structural cases, edits, comparison requests and typed event policy/questions at public graph/fork/inquiry construction and before execution. Do not normalize caller-owned objects a second time while validating them; validation must not mutate source evidence. Existing genuine outputs must remain unchanged.
- Correct or explicitly refuse under-resolved subnormal direction normalization. Preserve ordinary finite normal arithmetic and existing M34a one-ULP qualified motion-plane retention. No broad relaxation of physical symmetry, off-plane force/velocity/gravity, source singularity or M22a source-invariance contracts.
- Correct stale documentation and add a current capability/verification map. Document CI dependency coverage and duplication; avoid deleting historical validation in the same scientific change.

## Multi-stage exploration API

Extend the existing GraphWorldForkPlan.source_graph input to accept either an authentic UnifiedObservationRun OR an authentic previously completed GraphWorldForkRun. Do not fabricate a new UnifiedObservationRun, replay earlier integrations, flatten away source history, or reconstruct physics from JSON. The old direct-graph M34b API and v1 outputs remain byte-identical for unaffected inputs.

A new stage may select the original baseline, any earlier admitted ancestor-stage world, or an earlier new world in the same stage as parent. All inherited model, source assembly, native trajectory/event and ancestry objects are reused exactly. Only genuinely new admitted worlds are qualified and integrated. A rejected physical model yields skipped descendants; a physically admitted model with refused observations remains a valid physical parent.

Use one bounded, iterative source-resolution/validation path shared by forks and event inquiries. Avoid recursive exponential evidence reconstruction. Detect object cycles, forged prior runs and malformed chains before numerical work; independently validate every ancestor receipt and native observation on export. Never persist a mutable validation cache across calls.

Direct graph forks retain original limits: 1..5 new nodes, total 12, source-root depth 6, twelve physical edits per stage. For explicitly multi-stage inputs: at most eight fork stages including the new stage, 32 cumulative non-baseline worlds, physical ancestry depth 12, 1..5 new nodes and twelve edits per stage. Count all requested worlds, including refusals; budgets cannot reset on each stage. World IDs and fork IDs cannot collide anywhere in the inherited chain. Independent separate exploration branches can reuse names; no global registry.

Retain complete branch-local introduced/current/ever-seen/retired source IDs across all structural/physical/stage boundaries. A retired ID cannot be resurrected later. Every real source/model revision retains the actual immediate native parent, not an earlier substituted world. Comparison signs are right-minus-left; literal common saved SI times only. Root shifts require the same full typed questions/policy and two duration-complete unique same-direction native roots. Guard/terminal/unresolved/multiple/opposite roots stay unpaired.

Multi-stage forks use separate opt-in graph_world_fork_plan.v2 / graph_world_fork.v2 evidence, binding the immediate source-run digest, original graph digest, ordered stage lineage and cumulative scope. The first direct fork still exports v1. A genuine graph event inquiry may also target an authentic fork result, allowing fresh M24a questions about worlds created in previous stages; its new opt-in v2 evidence preserves original graph inquiry v1 unchanged.

This is repeated exploration calls in one Python process. Durable storage, process-restart restoration, trusted JSON import, merging two divergent stage histories, concurrency/shared editing and natural-language authoring are NOT included.

## Required acceptance

- Red-before/green-after regressions for reproduced audit defects. No source or caller evidence mutation during validation.
- Three successive exploration stages with structural-to-physical-to-structural changes, comparison to a root/old world, retained retired-ID refusal, unaffected sibling, fresh inquiry on latest stage, no earlier integration/reassembly and original objects/bytes unchanged.
- A previously unseen oblique three-component qE+mg analytic trajectory/root reference for every actual saved timestamp; separately expressed variable-field checks (finite rod/ring quadrature, E=-grad V, Coulomb circular motion) to broaden beyond uniform-field-only transfers.
- Hostile request, timestamp Boolean/float substitution, root/bracket tamper, old source mutation, forged receipts, node/stage collisions, cycles, cumulative budgets and failed parent histories reject safely. Finite detached JSON changes cannot modify native evidence.
- Actual released/candidate original graph v1/v2/v3, direct M34b fork v1 and graph event inquiry v1 byte equality on unchanged inputs.
- Python 3.10/3.12 full tests, real FEMM and historical Chromium, aggregate independent-form scientific references on PR and main, source SHA/tree evidence. Local interpreter results are supplemental and must not be mislabeled as the CI matrix.

## Release

Publish a detailed audit with priorities, reproduced findings, corrections, deferred work and explicit untested limits. Separate correction and feature commits. Open draft PR; all current-head CI and independent separate-context R2 review must pass on the exact final SHA/tree. Do not merge without separate owner expected-head authorization; verify exact merge tree and all actual postmerge main workflows before release.
