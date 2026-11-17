# M35a project audit, corrections and prioritized plan

**Audit baseline:** released M34b commit `a5113fa4cc924ac10cbb35b2f2d7ac80e11f1bde`, Git tree `d0646c9e4d518008784c42c1fec1c48c81901a09`. **Candidate status:** audit corrections and M35a are development work until exact-head CI, a new separate-context R2 report and owner-authorized merge/postmerge verification. This is an implementation-side audit, not that independent review.

## Executive assessment

The project has a useful physical core and substantial genuine validation: reusable electrostatic field/potential providers, explicit conductor versus prescribed-source semantics, qualified particle-motion planes, typed event studies, physical revision APIs and honest refusal boundaries. Its recent graph APIs preserve original worlds, source identities and numerical observations instead of disguising visual edits as physics. M34b is a meaningful reusable engine, not merely a set of animations.

The main weakness is **uneven validation and architecture across generations**. A result originating in an older sweep API does not carry all the integrity guarantees of a newer issued graph. Frozen dataclasses are also not a substitute for validating the current state of nested requests at use time. Documentation and CI have accumulated milestone by milestone; that provides useful historical evidence but makes current capabilities, transitive test coverage and costs difficult to understand.

This audit reproduced two closely related request-validation defects, one extreme-scale numerical normalization defect and one separate legacy-export consistency gap. The first three are corrected in this candidate. The legacy gap is explicitly tracked in [issue #80](https://github.com/physicsvigilante/conductor_sim/issues/80) with a separate hardening plan. No broad claim that the project is error-free is justified.

## Audit method and scope

The complete original repository was obtained from a pinned `git archive`; the downloaded ZIP's SHA256 and each retained file digest were independently checked. Rebuilding the Git tree from the extracted files reproduced the exact released tree above. No repository `AGENTS.md` was present. Original baseline reproducibility evidence is retained in review_sandbox runs [44098354148](https://github.com/physicsvigilante/review_sandbox/actions/runs/44098354148) and [44108932124](https://github.com/physicsvigilante/review_sandbox/actions/runs/44108932124); the charter-only commit `ba19e12c519df2338df5dbb9293e9fa7ebd611a8` precedes corrections.

The inventory at that baseline contained 77 production modules, 19,798 production lines, 94 test modules and 25 workflows. AST inspection counted 222 relative imports of private/internal symbols. Counts describe the source, not defect severity. Review concentrated on source fields, motion/event numerical boundaries, source ancestry, recent graph/fork/inquiry request/result validation, old sweep serialization, workflow dependency coverage, packaging and release documentation.

The original local non-FEMM suite passed **1,991 tests**, with 97 real-FEMM-only tests deselected and 17 warnings. That run used **Python 3.13.5**. It is supplemental execution, not a substitute for the requested CI Python 3.10/3.12 matrix, real FEMM installation or Chromium. No original PDF page was reinterpreted during this audit. No new independent human certification, exhaustive state-space proof, GUI screenshot audit or unrestricted-3D qualification is claimed.

Reproduce the targeted numerical and authoring audit with:

```sh
python tools/m35a_audit.py --expect candidate --output output/m35a-audit.json
```

The report deliberately ends with `AUDIT_COMPLETE_WITH_DOCUMENTED_OPEN_FINDINGS`, not a blanket all-clear. Running the identical audit script against released source with `--expect released` preserves red-before/green-after evidence.

## Findings and dispositions

### A-01 — Nested typed requests were not revalidated before issuing evidence

**Priority: high; reproduced; fixed in candidate.**

`WorldComparison` originally rejects comparing a world to itself. However, after constructing a valid instance, an ordinary `object.__setattr__` mutation could change its left ID to its right ID. A surrounding M34b fork plan checked its type and referenced IDs but did not rerun the comparison's invariant checks. The real runner could therefore issue an apparently valid `compared` record for the same world on both sides. A native world node whose observation mode was similarly changed to `typo-event` could pass a parent plan and become an event-study refusal instead of malformed authoring being rejected before execution.

These are **request-boundary and evidence-meaning defects**, not demonstrated force-law or ODE errors. They nevertheless matter for LLM/application authoring and stale mutable references. `frozen=True` prevents normal attribute assignment, not every mutation available in Python.

The correction revalidates current native node IDs, observation modes, structural requests, physical edits, comparison pairs and typed event policy/questions at graph/fork/inquiry plan construction and execution. This is done without reconstructing physical scenarios or renormalizing already issued event directions. Valid legacy exports remain unchanged. New regressions explicitly exercise malformed nested requests both before and after plan construction, and ensure rejection occurs before numerical work.

**Acceptance:** old source reproduces self-pair/mode acceptance; corrected source rejects it; genuine same-mode comparisons and compatible events still succeed; caller objects and normal bits are not modified by validation.

### A-02 — Smallest-subnormal finite direction normalization could produce a non-unit vector

**Priority: medium for an extreme numerical boundary; reproduced; fixed in candidate.**

For `(ulp(0.0), ulp(0.0), 0.0)`, the floating-point `hypot` result is itself subnormal and cannot retain enough relative precision. Dividing each component by that rounded length produced `(1,1,0)`, whose norm is square-root two. Reconstructing a motion specification then changed it again. M34a's one-ULP retention of an already-unit normal cannot repair an inaccurate first normalization.

The fix scales components before normalization **only when the calculated length is subnormal**, then normalizes the safely scaled vector. Ordinary non-subnormal arithmetic remains the original expression; zero/nonfinite/overflowing norms retain refusal behavior. The existing full physical plane, force, gravity and launch qualification is not relaxed, and M34a's already-unit one-ULP rule remains intact.

**Acceptance:** smallest-subnormal vectors become genuinely unit within the expected rounding allowance; repeated physical reconstruction is stable; 200 ordinary random normals have exactly the original arithmetic result; zero, invalid and overflowing directions still refuse. This correction covers the prescribed-source direction helper used by the audited motion/event path, not every independent normalization helper in the entire repository.

### A-03 — Older sweep output types permit contradictory status/native evidence

**Priority: high for consumers treating exported records as authoritative; reproduced; OPEN.**

Using public `dataclasses.replace`, `EventSweepRun` can export `status='complete'` with both its scenario and native event run removed. `TrajectorySweepRun` has the corresponding complete-without-trajectory combination. The numerical runner originally computed real data; the defect is that these older result wrappers do not enforce consistency when subsequently replaced and exported.

The newer graph/fork/inquiry APIs check issued receipts and native bindings. The two older sweep types are **not accepted as M35a source histories**, so M35a does not import this weaker evidence boundary into its stage chain. That containment does not fix the historical export APIs and must not be described as doing so.

[Issue #80](https://github.com/physicsvigilante/conductor_sim/issues/80) retains exact reproduction and the separate corrective plan: characterize legacy callers; add consistent status/model/native-observation checks at construction and each export; preserve actual refusal and censoring cases; bind genuine issued results or explicitly define an equivalent trust boundary; test Boolean clocks, roots and statuses; verify renderer and valid-output byte compatibility. Audit related older family wrappers before generalizing a fix. Keep this migration separate from M35a's new multi-stage feature to make review scope credible.

### A-04 — Current capability/release documentation had drifted

**Priority: medium; fixed as navigation/status clarification.**

The README foregrounded older milestones while newer released milestone documents still described themselves as unreviewed candidates. These are historical development notes, not reliable current release indicators. This candidate adds a [current capability map](current_capabilities.md), a clearly marked [M35a contract](m35a.md), and a README entry point. Recently released notes retain their scientific text but point readers to the capability/release index. The index explicitly distinguishes released M34b from draft M35a.

A future release checklist should update the index in the release process, rather than rewriting every old scientific charter. Exact Git SHA/tree and release verification in review_sandbox remain authoritative; `pyproject.toml`'s historical `0.1.0` alone does not identify a validated milestone.

### A-05 — CI has both redundant work and transitive coverage gaps

**Priority: medium; safety-net addition in candidate; consolidation deferred.**

The 25 baseline workflows retain useful original physics/FEMM/browser acceptance. They also repeat full pytest runs in general Python and visualization workflows. All lacked a concurrency declaration, and eight lacked an explicit workflow-level permissions declaration. Omitted permissions do not by themselves prove a write token: organization/repository defaults also matter. Several newer independent-reference workflows trigger only on selected top-level adapter files and PR events; a change in a transitive core helper can therefore miss the most relevant reference workflow or not rerun it after merge.

The candidate adds one **dependency-aware aggregate scientific gate on PR and main push**. It reruns the M32a–M35a native/independently expressed references when core source, tests, tools or examples change. It uses read-only contents permission, retains failed evidence, pins actual source HEAD/tree and cancels superseded PR runs only. Existing real FEMM and historical Chromium checks are not deleted.

Next CI step: measure workflow/job duration and overlap, define fast unit, numerical transfer, real FEMM and browser tiers, remove duplicated full-suite execution only after proving equivalent coverage, and add explicit least-privilege permissions and suitable concurrency to each historical workflow. Do not reduce scientific coverage simply to make status counts look smaller.

### A-06 — Release discipline relies on process rather than server-enforced branch protection

**Priority: medium; owner configuration required, not silently changed.**

The live branch API reported `main` as unprotected, with no required status contexts in that response. The current human/assistant workflow uses source-pinned review and an expected-head safeguard, which is valuable but is not equivalent to server-enforced branch protection. Recommend requiring the agreed status checks and review policy, disallowing accidental direct main pushes and ensuring stale approvals cannot be reused after source changes. The owner must configure this; no repository-administration setting was modified by the audit.

Also review the still-open historical draft PR #59 for supersession by the accepted M25a work. Its existence is backlog ambiguity, not evidence that current main is missing that feature. No old branch or PR was deleted/closed here.

### A-07 — Growing private coupling and repeated validation/serialization

**Priority: medium; targeted improvement in M35a, broader refactor deferred.**

Milestone-specific adapters import private functions from one another and repeat world resolution, comparison and record construction. A naive recursive multi-stage extension would repeatedly reconstruct entire ancestor records; caching by object identity across calls would be unsafe because source objects can be mutated.

M35a introduces a shared **bounded iterative per-call source-history resolver** for forks and event inquiries. It detects cycles and malformed chain objects before numerical work, validates the root and each prior stage chronologically, preserves native identities, and discards its context afterwards. A regression checks one validation visit per previous stage for a fork export. Total work still depends on saved samples and source ancestry; this is not a constant-time export or a broad performance theorem. Inquiry plan/run checks can perform multiple bounded passes, so further optimization remains measurable future work.

Later, extract public internal services for source resolution, native observation validation and comparison only behind full byte-compatible tests. Do not replace functioning physical solvers during an orchestration cleanup. Version the record contracts and make unsupported combinations explicit.

## Additional physical checks performed

The audit did not rely solely on uniform-field examples. A separately expressed finite-difference check of `E = -grad(V)` passed for fixed point, finite rod, full ring, sphere and shell sources away from their singular/discontinuous boundaries. Relative field discrepancies were approximately 6e-13 to 8e-12 for those selected cases.

Independent midpoint Coulomb quadrature with 16,000 source elements checked the rod and tilted ring: rod relative potential/field discrepancies about 4.1e-10 / 2.4e-9; ring about 1.3e-16 / 1.7e-16. An attractive inverse-square Coulomb circular orbit was checked against its analytic position/velocity for 65 actual native saved samples over a full period: maximum position discrepancy about 5.3e-9 m and velocity discrepancy 1.4e-8 m/s in the authored SI case.

These checks found **no additional field-law defect in the tested cases**. They are finite developer-authored transfers, not an exhaustive proof of all geometries, new conductor-induced dynamics, point/rod near-singularity guarantees or a separate-context reviewer verdict. Real FEMM and Chromium coverage must be verified through actual CI.

## M35a deliverable and honest boundary

M35a permits a new physical fork to use an authentic previously completed fork as its source, then permits fresh typed event inquiries over any world in that retained history. Only newly admitted physical worlds are integrated during forking. The executable example uses three stages with structural/physical/replacement/removal operations, a refused retired-ID resurrection and an unaffected sibling, then four genuinely new event inquiries.

Continuation opts into fork plan/run v2; queries on fork history opt into inquiry plan/run v2. Original graph v1/v2/v3, direct M34b fork v1 and original graph inquiry v1 have **complete byte-identical output** on the selected unchanged test fixture. Multi-stage limits are eight fork stages, 32 cumulative requested nonbaseline worlds and physical depth 12; rejected nodes count too, preventing budget resets. Same typed question/policy and duration-complete unique same-direction event rules remain.

This is repeated calls in **one live Python process**. It does not restore native models after restarting a program, authenticate saved JSON, merge divergent histories, support concurrent editing, or automatically interpret arbitrary JEE questions. A durable session/persistence format is a separate future contract.

## Prioritized next actions

1. Finish exact-head full CI and a genuinely new separate-context R2 review for these audit fixes and M35a before owner-authorized merge. Preserve old failures and fixture corrections.
2. Address legacy export consistency in issue #80 with its own compatibility characterization and review; avoid blanket integrity claims meanwhile.
3. Owner enables server-enforced release gates; review superseded draft PRs and establish one current capability/release index.
4. Consolidate CI only after measured coverage/duration comparison; retain nonuniform physics transfers and source-HEAD plus main checks.
5. Extract stable source/observation/comparison services, then design durable exploration sessions if needed. Expand physics/JEE coverage using explicit new regimes and independent references, not by loosening plane or source guards.

## Reproduction and evidence distinctions

```sh
python -m pytest -m 'not femm5' -q
python -m pytest tests/test_m35a_audit_corrections.py tests/test_m35a_multistage_exploration.py -q
python -m examples.m35a_multistage_exploration --output output/m35a
python tools/m35a_independent_multistage.py --input output/m35a --output output/m35a-independent.json
python tools/check_m35a_references.py --output output/m35a-references
python tools/m35a_legacy_snapshot.py --output output/legacy-candidate
```

The standalone M35a numerical checker imports no `conductor_sim`; it validates ten admitted worlds, 210 actual saved 3D samples, 126 exact-time comparison rows, four fresh native inquiries and seven altered-evidence controls. The native execution/example imports production as expected. The audit tool imports production to obtain actual values but uses separate gradient/quadrature/orbit equations for expectations. These distinctions should remain explicit in reviewer and release reports.

The first multi-stage example fixture reused a source display label and was correctly refused by existing unique-label validation. That failed attempt is retained; the fixture was corrected, not the physical source validator. An incomplete timed-out test invocation was not counted as a pass. Source-pinned CI and independent review evidence will be recorded in the PR/review packet; this audit is not itself a release certificate.
