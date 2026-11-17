# M33a — Frozen charter: explicit cross-branch physical-world comparisons

**Preimplementation freeze.** Base is released M32b `ad99239faf6a4a2229134b95f82c599f7be9d520` (PR #74, independently reviewed and 15/15 postmerge workflows reported green). Commit this charter alone before production edits. This is a contract, not an implementation or independent review.

## Educational goal

A teacher may ask **which of two independently authored, independently qualified and genuinely integrated alternative worlds** has a different position, velocity, or detector event time, without forcing one world to be the immediate physical parent of the other. This enables siblings and nonadjacent branches from one immutable baseline. A pairwise comparison is *not* a physical transition between the worlds, a source-motion law, a force/work integral, or evidence that a textbook transcription is authenticated.

## Minimal opt-in native interface

- Add an immutable `WorldComparison` request with exact bounded ASCII `comparison_id`, `left_world_id`, `right_world_id`. Names refer only to `baseline` or existing `UnifiedObservationPlan.nodes`, and must differ. `left` is reference, `right` is variation; numerical differences are **right minus left**. Reject bad types, unknown IDs, same-world pairs, repeated IDs, repeated unordered pairs and excessive requests (at most eight).
- Add exactly one optional last field `comparisons: tuple[WorldComparison, ...] = ()` to `UnifiedObservationPlan`. Nonempty comparisons require `retain_source_ancestry=True` so each admitted world has runner-issued complete branch-local source provenance. Existing unmodified M32a v1 and M32b v2 plans, exported bytes/shapes, statuses, and immediate-parent comparisons must remain unchanged when no comparison is requested.
- Only nonempty explicit comparisons switch plan/result to `conductor_sim.unified_observation_plan.v3` and `conductor_sim.unified_observation_graph.v3`. The detached result contains ordered `cross_world_comparisons`, with each request's ID, side IDs, actual admitted revision IDs or null, source ancestries or null, physical/observation statuses, explicit `compared` or `unpaired` status and explanatory reason.
- A comparison never triggers a new integration: reuse actual separately solved and validated native observations for both graph worlds. It is an analysis of *two different qualified worlds* not a synthetic model/physical edge.

## Scientific comparison limits

- For two successful same-mode observations, compare actual **identical saved SI times only**, using the unchanged M28a `_shared_rows` native 3D position and velocity differences. No interpolated times, index pairing, extrapolation past guards, invented empty-horizon match, or implicit time shifting.
- If both are native `events` runs for the **same plan questions and numerical policy**, use unchanged M28b `_root_pair`: an event time shift may be paired only for one same-direction located root in each of **two duration-complete** actual event histories. Multiple or bidirectional ambiguous roots, guard/terminal/unresolved horizons and unlocated events remain explicitly unpaired per event.
- Mismatched observation modes, missing native observations, physically rejected nodes, integration refusals or a lack of shared native saved times yield an `unpaired` comparison with an explicit reason, not fabricated values. A physically admitted but event-study-refused world is still physically admitted; report its model lineage honestly.
- Different held-source inventories and probe conditions can be compared as *alternative trajectories* in one world coordinate frame. Never advertise a cross-inventory force/energy difference as physical work, never imply sources dynamically changed between alternatives. Preserve both actual source inventories and unique branch ancestry; do not merge or globally retire source IDs.

## Evidence and acceptance

1. A graph has two sibling structural worlds using different fixed held fields, plus nonadjacent physical descendants. Show right-minus-left genuine 3D trajectory deltas and detector times for an original vs sibling, sibling vs sibling and distant cousin pair.
2. Independently express at least one analytic Newtonian uniform-field reference (SI x/y/z acceleration, every *actual saved* timestamp, and independent detector-root calculation), not by calling production comparison code to derive expected values. A distinct tilted fixed-field or point-source case should ensure axes and directions are not silently projected onto a 1D picture.
3. An event comparison with different native observation modes, physically invalid world, admitted model with refused event study, source guard, terminal/unresolved event, multiple roots, opposite directions and disjoint saved timestamps must not produce a false paired time shift. Preserve M28b root qualification and M28a no-interpolation semantics.
4. Different branches may legitimately reuse the same newly introduced source ID. Their source certificate paths remain independent. Both revision IDs and source-ancestry histories must appear on the correct side of every comparison.
5. Reject malformed/duplicate/unknown requests, Boolean/non-native IDs, false source provenance, forged status, forged native or comparison result, mutated time, and float-to-Boolean zero timestamp substitutions. `UnifiedObservationRun` construction and every `as_dict` independently reconstruct native comparisons and check the runner-issued receipt. This is an *in-process evidence boundary*, not cryptographic security against privileged private Python access.
6. Old M32a and M32b tests and byte-shape snapshots pass unmodified. New v3 data are JSON-finite and detached; altering an export cannot alter the native result or a subsequent export. Avoid novel solvers, FEMM, source dynamics, scene graphs, browser-time physics and new Python dependencies.

## Process

Implement only on `m33a-cross-branch-world-comparisons`. Preserve original frozen charter commit. Add one small native comparison primitive (or bounded additive code to the existing graph module), tests, a complete executable example, documentation and a separately expressed numerical reference. Run source-HEAD Python 3.10/3.12, dedicated real FEMM, historical Chromium and adversarial checks with exact SHA/tree evidence. Open a **draft** PR; request an independent separate-context scientific/evidence R2 review only on the exact validated head. Merge using expected HEAD only after a published no-blocker independent PASS, all required CI green, and owner authorization. Verify postmerge main and every triggered workflow before claiming release.
