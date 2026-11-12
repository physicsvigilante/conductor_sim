# M6b — held-out authorability test (single assistant trial)

**Status: accepted for the one open assistant authoring trial.** See [M6b acceptance](m6b-acceptance.md).

## Question and evidence separation

M6a curated 19 exercises directly from `POTENTIAL_CONDUCTORS1.pdf`. M6b
introduces **15 previously unimplemented, differently parameterized JEE-style
questions**. The prompts in [`m6b/questions.py`](../m6b/questions.py) are
separate from the assistant-authored physical compositions in
[`m6b/attempts.py`](../m6b/attempts.py), which in turn do not import
[`m6b/reference.py`](../m6b/reference.py). The independent grading key
contains full model geometry/constraint fingerprints and *closed-form*
Gauss-law reference observations, not expected values copied from the engine.

This was authored by the assistant within the same development environment
**after inspecting the public APIs**. It is **not** a blind independent model
comparison, a statistical success-rate estimate, an automatic PDF parser, or
proof that an arbitrary language model will generalize to novel problems.
The held-out distinction is from the earlier **M6a curated casebook**, not
from model training data or the eventual publicly committed grading key.

## Submission contract

An author writes only a Python module with a `SUBMISSIONS` dictionary mapping
question IDs to `Submission` objects (`m6b.protocol`). Each supported
submission contains one or more `ElectrostaticProblem` equilibrium states
and requested `Probe` observations. A genuinely unsupported case supplies
an explicit capability explanation and **no fictitious problem/simulation**.

For this trial, **no physics engine or viewer implementation source was
changed**. Geometry validation, FEMM capability detection, observation
sampling, and HTML export reuse M1–M6a public APIs. The new M6b files are
benchmark/submission/scoring glue, not a new physics backend.

The grader checks all of the following independently:

1. Exact structure of every physical problem: charge signs/locations,
   conductor names/radii/constraints, original authored order, medium and
   equilibrium-state count.
2. Complete requested observation set (with correct probe locations and
   conductor identities); no duplicate or missing measurements.
3. Exact analytic potentials/field components/conductor charges compared to
   separately calculated Gauss-law answers. Geometry/expected values are
   not inferred from a generated visual.
4. Honest backend capability (`both`, `analytical-only`, `unsupported`).
   **Non-collinear** cavity charges can be handled analytically, but the
   axisymmetric FEMM backend rejects them explicitly. External sources,
   rings, gap charges, connected separated spheres and charge-motion problems
   fail closed.
5. Valid offline M5 HTML/JSON exports for runnable states, with original
   question, source backend, and precomputed-equilibrium metadata.

The actual FEMM 5 acceptance gate samples five M6b configurations and
checks a distinct two-state grounded-conductor sweep, retaining all 40
earlier FEMM regression tests. These use the pinned official FEMM 5 headless
engine, **not** fake client recordings.

## Reproduce or submit a different LLM attempt

From the repository root:

```bash
python -m m6b.evaluate list
python -m m6b.evaluate grade --output output/m6b-evaluation.json
python -m m6b.evaluate render grounded-two-state-sweep --output output/m6b-sweep.html
python -m m6b.evaluate render noncollinear-math-only --output output/m6b-analytical.html
```

To evaluate **another** LLM's independently written submission, put its
`SUBMISSIONS` mapping in a trusted importable Python module and run:

```bash
python -m m6b.evaluate --submission other_attempts grade
```

**Security:** importing a submission executes arbitrary Python code. This is
for trusted/local use only; it is *not* a sandbox for untrusted agent code.
Do not offer arbitrary web users direct module import. The ordinary grade
command uses exact analytical references and does **not** require FEMM.

The explicit FEMM renderer must be requested with `--backend femm`; missing
FEMM installation raises an error and never substitutes analytical samples.
The output is offline, precomputed equilibria, **not** continuous charge
movement or browser-time numerical solving.

## Evaluation cases

| Case group | Count | Capability |
| --- | ---: | --- |
| Recharged, grounded, nested, shifted and axially multicharged shells | 9 | FEMM and exact analytical |
| Two genuinely non-collinear charges inside a spherical shell | 1 | Exact analytical only |
| External charge, ring, inter-shell-gap charge, connected separated spheres, particle trajectory | 5 | Unsupported |

The independently measured local grading result was 15/15, including
all requested probes. No physical solver or viewer source modifications
were needed. This is **one open authored trial**; follow-up evaluations
should sample independently selected unseen questions, distinct models,
independent grading criteria, and report first-pass and repair counts.

## Next physics opportunity

Outside charges near grounded spheres, isolated charge rings, and charge
motion remain unsupported. Each needs its own reusable physics component or
backend, rather than extending the viewer with fake effects.
