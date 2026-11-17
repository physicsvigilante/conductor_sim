# M36a — Preregistered JEE-30 physics coverage reassessment

**Frozen before M36a benchmark implementation.** Source baseline: released `main` merge `73e076cf7eaf7fefbae2b24afc1f54ff8f0b6fc6` (M35b), Git tree `89363a917e3862d663d934501a1b5cd8a22a84c0`. The original [M21a source-anchored 29-case cohort](m21a_coverage.json) and its [source audit](m21a.md) are immutable historical inputs. M21a tiers describe *the M20c engine*, not the current engine.

## Educational objective

Determine which of **30 preregistered physical systems** can be faithfully represented now, what is only approximated or refused, and which reusable physical capability unlocks the most meaningful additional JEE simulation. This is a **coverage reassessment**, not an LLM-written answer benchmark, a textbook-question frequency estimate or a universal problem parser.

The cohort consists of all 29 existing M21a question anchors, in exactly their original order, plus **one** previously independently documented official JEE Advanced 2021 Paper 1 Q9–10 charge-pair zero-equipotential *apparatus* (one physical system, not two counted question successes). Its origin and certified example are recorded in [M21b](m21b.md). No other cases may be added to improve a score after inspecting results. The original source PDFs have recorded SHA-256 in M21a. Do not claim a new visual reading of the PDF pages merely from this reassessment.

## What the audit must distinguish

1. **Current *candidate route***: documented present-day qualified engine capability (`supported_scoped`), partial representation (`partial`), or missing physics (`unsupported`). These are *triage hypotheses*, not executed test verdicts. Explicitly name physical assumptions and source fidelity concerns per case.
2. **Independent case execution**: source-model interpretation, genuinely authored model, actual native computation (or truthful static observation), separately expressed physics reference, and rendered/viewer inspection are different pieces of evidence. Record missing pieces. No numerical simulation PASS may be inferred from a document, prior CI for a related hypothetical, or an API's existence.
3. **Five-axis teaching evaluation**: use [evaluation_contract.md](evaluation_contract.md): interpretation fidelity, physical correctness, simulation coverage, educational usefulness, and optional textbook-answer completeness. Separate supported partials from refusals and count denominators explicitly. A static endpoint-work calculation is not elapsed-time motion; a uniform sheet approximation is not induced conductor response.
4. **High-priority source-backed probes (preselected before coding)**:
   - workbook p2 Q15 far-separated joined conductors: `Q_i\propto R_i` and `E_1/E_2=R_2/R_1` in explicit self-capacitance asymptotic regime, not concentric wiring;
   - workbook p2 Q19 electron reversal near an ideal charged metal plane: one-sided `E=\sigma/\epsilon_0` with an actual numerical reversal event and no microscopic surface collision;
   - workbook p3 Q30 electron plate transit: true elapsed-time traversal, clearly disclosed assumed uniform field and terminal observation, no invented contact response;
   - workbook p5 Q47 rigid dipole: static orientation energy ranking `\Delta U=pE(1-\cos\theta)`; do not infer inertia or genuine rotation dynamics from an energy-only question;
   - workbook p12 Q40 rings with upward transverse E: `q(V_A-V_B)\approx3.5950207 J` without imposing an imaginary free axial trajectory;
   - workbook p15 IIT-JEE 2012 proton: `qE=mg`, voltage about `1.04375e-9 V` using explicitly rounded constants, zero initial speed;
   - official 2021 Paper 1 Q9–10: real `V=0` locus `(x-3)^2+y^2=3` for charges `-Q` and `+Q/\sqrt 3` at x=0 and x=2 m, masked charge singularities;
   - workbook p70 three nonconcentric cavities: **deliberate refusal control** of local 3D induced charge/field visualization; Gauss-law total-charge facts alone do not permit invented local detail.
   These are coverage-probe targets; subsequent independently tested executions must be labeled as such. The initial inventory may only label existing documentation, not claim these eight probes ran.
5. **Boundaries**: old `uniform-ring-axis` limitation is historically superseded by M27 off-axis rings, but a charged ideal ring is **not a responding conductor**; M21d's `\sum R_i/d_{ij}\le0.05` threshold is an admission heuristic, not a measured 5% error guarantee; M24b's oil-drop transient requires teacher-supplied drag, switch time and launch data absent from the official question; genuine irregular multicavity local distributions remain unsupported. All borrowed teacher values must be explicitly separated from source-given quantities.

## Deliverable and acceptance

- Add `docs/m36a_jee30_coverage.json` as a 30-item **provisional source-pinned capability inventory**, preserving the 29 original anchors, tiers, PDF-page identifiers and order. State the 30th's official source and apparatus grouping. Every case has a current documented route, assumptions, excluded physics, citations to actual repository docs and `case_replay_status=pending`.
- Add a deterministic, clean-checkout offline auditor at `tools/m36a_jee30_audit.py` that compares all original 29 identities against their historical inventory, checks the new anchor, case uniqueness and allowed triage labels, validates actual referenced docs, makes no production physics calls, and emits honest separate denominators. Add tests for mislabeled proven PASS, altered historical anchor, duplicate case, bogus source link and unsupported case being counted as simulated.
- Do **not** edit production physics or claim new native JEE simulations merely by passing inventory tests. The next step is independent source-anchored native replay/observer verification, then a separate reviewer review. Any newly discovered coverage gap should become a small physical work item, not an excuse for a broad scene graph.
- PR remains draft and unmerged until independently checked, exact-head CI passes and owner explicitly authorizes release.

**Anti-score rule:** no `N/30 solved` or “coverage achieved” based on these proposed capability labels. Until the source-specific probes are *actually executed*, the measured native-replay denominator is zero, even if the documented qualified-route candidate count is large.
