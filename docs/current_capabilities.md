# Current capability and validation map

**Released baseline at this audit:** M34b, merge `a5113fa4cc924ac10cbb35b2f2d7ac80e11f1bde`, tree `d0646c9e4d518008784c42c1fec1c48c81901a09`. [Release evidence](https://github.com/physicsvigilante/review_sandbox/blob/main/projects/conductor_sim/pr-79/STATUS.md): independently reviewed, merged, 15/15 main-push workflows passed; Python 3.10/3.12 each 1,991 tests, 97 dedicated real FEMM tests and historical Chromium acceptance. Counts are historical baseline evidence, not a claim that a later draft has passed.

**Draft development:** [M35a multi-stage exploration](m35a.md) and [project audit/corrections](m35a_audit.md). Not released until new exact-head CI, independent R2 review, owner approval and postmerge verification.

| Layer | Reusable API / examples | Qualified scope and limits |
| --- | --- | --- |
| Conductor electrostatics | Analytical backends, FEMM5, `force_energy`; older M1–M18 examples | Explicit grounded/isolated spheres, shells/cavities, planes, plates and supported connected-conductor configurations. Backend domains differ; not arbitrary conductor geometry with universal dynamics. |
| Prescribed fields | `PrescribedField`, fixed point, rod, ring, disk-axis, sphere/shell and uniform field providers | Fixed nonresponding source distributions, not automatically conductors. M27 full ring supports off-axis E/V; disk and old ring-axis providers remain axis-qualified. |
| Particle dynamics | `TestChargeMotionSpec`, native integration, uniform gravity and qualified driven/event APIs | Genuine elapsed-time numerical motion in certified axes/planes, source guards and explicit terminal events; not unrestricted 3D or moving-source many-body dynamics. |
| Physical revisions | `PrescribedScenario`, M20b revision, M22a `reassemble_scenario` | Stable IDs; complete-model requalification; structural changes are not source motion. M34a retains valid oblique plane bits. |
| Scenario families | Parameter and event sweeps, structural/physical families | Actual independent solves and comparisons; legacy result wrappers have the open [issue #80](https://github.com/physicsvigilante/conductor_sim/issues/80) export-consistency limitation. |
| Verified world graphs | `UnifiedObservationPlan/Run`, `SourceWorldAncestry`, `WorldComparison` | Graph v1/v2/v3, real branch-local source retirement, exact saved-time comparisons and qualified event shifts. Receipt-based in-process integrity, not external signatures. |
| New measurement questions | `GraphEventInquiryPlan/Run` | Released M33b accepts authentic graph worlds and performs fresh native M24a studies. Draft M35a additionally accepts fork history. |
| Incremental physical hypotheses | `GraphWorldForkPlan/Run` | Released M34b adds 1–5 new worlds to one original graph without replaying prior integrations. Draft M35a permits subsequent fork stages under cumulative bounds. |
| Visualization | Existing offline HTML/SVG viewers and browser acceptance | Actual saved physical states, source geometry where qualified, graphs and playback. No universal scene graph, no browser ODE, no automatic display for every new orchestration API. |
| LLM authorability | Typed Python models and executable examples | Codex/LLM can write supported models; no automatic arbitrary-text/PDF interpretation or universal JEE answer engine. |

## Where to start

- [Physical/scientific distinction](m19a.md): prescribed sources versus responding conductors.
- [Physical source reassembly](m22a.md), [cross-branch comparison](m33a.md), [fresh event inquiries](m33b.md), [incremental worlds](m34b.md).
- [Draft M35a usage and limits](m35a.md), [audit findings and prioritized work](m35a_audit.md).

Milestone charters and early implementation documents preserve their historical development wording. This index and the source-pinned release status are the current navigation/status layer; a historical “candidate” paragraph is not evidence that an already merged release was unreviewed. Conversely, a candidate on a branch is not released just because the earlier main revision is green.

## Verification contract

Report separately: faithful authored interpretation, numerical/physical correctness, native observation completeness, provenance/in-process evidence integrity, educational usefulness and optional textbook-answer completeness. Do not count blocked source questions or skipped real-FEMM tests as completed simulations. Compare exact source HEAD/tree before review and merge; inspect actual main-push runs afterwards. A separate-context LLM review is valuable scrutiny but is not human scientific certification.
