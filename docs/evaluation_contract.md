# Evaluation contract — teaching accompaniment, not answer reproduction

**Normative project requirement.** `conductor_sim` is a teaching
accompaniment. Its primary objective is **faithful physical interpretation**
and **independently verified simulation of systems within the explicitly
supported domain**. Reproducing a textbook's final numerical answer, answer
option, formatting, or full derivation/solution sequence is optional. Those
attributes must never be used as proxies for physical correctness or as
unannounced conditions for accepting an educational simulation.

This contract governs future authoring, source-anchored/PDF, correctness,
visualization and JEE problem evaluations (including M19c). Existing
historical grading artifacts keep their original definitions and counts;
they are not retroactively relabeled or presented as scores under this rubric.

## Five **separately reported** dimensions

Each item receives an explicit status per dimension: `pass`, `partial`,
`fail`, `not_evaluated` or `not_applicable`, accompanied by concise
evidence, relevant limitations and a **clearly stated denominator**.
Do not collapse the dimensions into a single pass rate or numerical score
without publishing the aggregation rules and the individual results.

| Dimension | What it evaluates | Evidence and non-substitutes |
|---|---|---|
| **1. Interpretation fidelity** | Correct identification of physical objects, stationary/moving sources, constraints, geometry, charge signs, initial/boundary conditions, assumptions and the requested *phenomenon*. | Model/inventory and problem-statement audit; distinguish an honestly bounded interpretation from a changed physical question. A matching textbook answer does not prove fidelity. |
| **2. Physical correctness** | Correct potential, field, force, energy, signs/units, conservation, time evolution, interfaces and domain behavior *for the model actually claimed*. | Independent analytical or numerical reference, invariants, metamorphic checks, solver/trajectory consistency and actual boundary tests. A good-looking picture or the textbook's option alone is not proof. |
| **3. Simulation coverage** | Which parts of the original physical situation are genuinely represented, which are omitted or approximated, and whether all claims stay in the supported computational domain. | Explicit `supported` / `partially_supported` / `unsupported` classification and included/excluded-phenomenon inventory. Never imply a complete question solution when only a subset is simulated. |
| **4. Educational usefulness** | Whether the output makes the supported physics understandable and investigable: meaningful motion, readable quantities and units, clearly identified idealizations, stable axes and source geometry, informative E/V/energy graphs and usable playback/controls. | Actual saved viewer/data inspection and where appropriate Chromium/browser acceptance, including honest gaps/guards. Pedagogy cannot excuse incorrect physics. |
| **5. Textbook-answer completeness (optional)** | Whether the final textbook value/option, answer convention, or full written solution sequence was reproduced. | Report `not_attempted`, `complete`, `partial`, `mismatch`, or `not_applicable` **separately**. No penalty to dimensions 1–4 when an answer is intentionally not generated. |

The fifth dimension is **not** a hard gate. If the simulator *does* assert a
numerical answer or comparison, that assertion must be correct and explicitly
qualified; demonstrably wrong **displayed physics** still fails dimension 2.
Not matching a book's wording, answer format, rounding convention or step
order is not, by itself, a physics or simulation failure.

## Claim-level scientific acceptance

- **Interpretation and truthfulness are non-negotiable.** Wrong physical
  geometry, silently substituted conductors, fictitious off-axis ring/disc
  fields, dropped sources, wrong force/potential-energy convention or invented
  collisions cannot be accepted merely because a final answer matches.
- **A faithful partial simulation may pass.** Example: animate charge
  motion along a ring's symmetry axis and graph `V(x)`, `E_x(x)`,
  `K(t)` and `qV(t)`, while explicitly not claiming a general 3D
  ring field, an exact textbook turning-time expression or a full
  written solution. Report `simulation_coverage=partial` if other parts
  of the question are outside scope; this can still be educationally strong.
- **An appropriate refusal is a valid outcome.** If a requested phenomenon
  is unsupported, explain what is unsupported, why the attempted model
  would be misleading, and optionally identify a faithfully representable
  subproblem. Grade the refusal's interpretation/honesty and coverage
  separately; it does **not** count as a fully simulated problem.
- **Independent verification is required for claims of correctness.**
  Verify `E=-grad(V)` where defined, `F=qE` and `U=qV` for
  prescribed fields; use the correct effective mechanical energy convention
  when conductor-induced forces and fixed-voltage supplies are involved.
  Test units, source identity, boundary/guard conditions and analytic
  limits. Avoid scoring the simulator against reference code that supplies
  its own production physics.
- **Graphical honesty is required.** Show true discontinuity gaps and
  unresolved regions instead of inventing values. Static position sweeps
  must not be called time dynamics; sparse paths and approach cutoffs must
  not imply exact collisions. Labels, scales and units must be meaningful.

## Required report structure for new evaluation suites

Reports should include the following separate keys (or explicitly
documented semantically equivalent fields), plus benchmark-provenance and
evidence references:

```json
{
  "evaluation_contract": "teaching-accompaniment.v1",
  "case_id": "example-ring-axis",
  "source_type": "original_prompt_or_page_anchored_question",
  "support_classification": "partially_supported",
  "interpretation_fidelity": {
    "status": "pass",
    "evidence": ["correct ring geometry, source sign and initial conditions"]
  },
  "physical_correctness": {
    "status": "pass",
    "evidence": ["independent axial ring formula and mechanical-energy checks"]
  },
  "simulation_coverage": {
    "status": "partial",
    "represented": ["on-axis test-charge dynamics"],
    "excluded": ["off-axis ring field", "full textbook derivation"]
  },
  "educational_usefulness": {
    "status": "pass",
    "evidence": ["time playback with readable V, E, kinetic and potential energy"]
  },
  "textbook_answer_completeness": {
    "status": "not_attempted",
    "required_for_simulator_acceptance": false
  }
}
```

The example is a *schema illustration*, not a claim about an executed JEE
question. Different trials may supply additional dimensions for authoring
reliability, robustness, refusal appropriateness or numerical uncertainty,
but must preserve this separation.

For aggregate reports, show **distinct denominators**: prompts assessed;
accurately interpreted cases; simulations actually run; verified physical
simulations; accurately scoped partial simulations; justified refusals;
educationally usable visualizations; and optional textbook-answer matches.
An unsupported correct refusal must not inflate the count of successful
simulations. The reference source (textbook page, user prompt or original
authoring problem) and whether authorship was blinded or openly curated
must also be reported.

## M19c implementation implication

Evaluate newly authored fixed-field JEE accompaniments on all four primary
dimensions independently. Sample correct ring-axis, sphere/shell, uniform
field, rod and composite-source setups; include purposely unsupported
off-axis cases. Grade actual model interpretation and independent physics,
source-appropriate visualization and honest scope. **Do not require a final
textbook answer or complete worked-solution sequence for a case to be
accepted.** Retain the repo's Python, Chromium, FEMM non-regression and
explicit-refusal gates; this policy does not broaden any solver's domain.
