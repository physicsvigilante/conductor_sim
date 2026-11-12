# M6b acceptance — openly documented held-out JEE authoring trial

**Accepted 2026-11-12 for the strictly bounded M6b evaluation, not for arbitrary autonomous natural-language authoring.**

## Questions, attempted solutions and grading

- **15** JEE-style question cards absent from the curated M6a PDF source catalog; new geometry/charge values and compound constraints, not directly copied M6a implementations.
- **One assistant-authored candidate** using the already existing `PointCharge`, `SphericalShell`, `Grounded`, `FixedNetCharge`, `ElectrostaticProblem` and requested `Probe`s. Authored code does not import the grading reference or implement physical laws, solve PDEs or render graphics.
- **15/15 evaluated correctly**, separated into **9 analytical + FEMM-capable**, **1 analytical-only non-collinear source arrangement**, and **5 unsupported, explicitly explained**. The author did not invent replacement geometry for ring, external sphere charge, inter-shell-gap charge, separated wired conductors or charged-particle motion.
- **29 independently checked observations** and **all exact input model fingerprints** passed: SI charge signs/locations, conductor radii, names, grounding and charge constraints, authored ordering and free-space permittivity. Missing/misplaced observations, wrong charge geometry and fabricated unsupported models fail grader regression tests.
- The scorer is open and reproducible. A different trusted, importable submission module can be passed via `--submission MODULE`. There is **no isolation of imported Python**. Do not run untrusted authored code directly.

**Methodological limitation:** This is *not* a blind benchmark of an independent unfamiliar model. The assistant writing the attempt knew the public engine API and had previously worked on the repository, and the grader keys are committed openly. The 15/15 outcome measures **this one manually inspected composition trial** and cannot be interpreted as a general LLM success probability. Future controlled studies should hide independent reference keys from the author, report first-token attempts, failures, retries and time/cost budgets, and sample independently selected problems.

## Independent physics examples

| Previously unimplemented prompt | Independent exact prediction |
| --- | ---: |
| Two isolated shells, inner +1 nC, outer −2 nC; gap E at x=0.15 m | +399.446746 V/m |
| Same arrangement, exterior potential x=0.30 m | −29.958506 V |
| Inner +0.6 nC shell and grounded outer, inner potential | +35.685867 V |
| Grounded outer shell with inner +1 nC / +3 nC over two separate states, inner potential | +31.774173 V / +95.322519 V |
| Non-collinear cavity charges, outer net total +0.1 nC, exterior V at x=0.30 m | +2.995851 V; analytical only |

The independent expected values are algebraic Gauss-law/Kelvin-image references; they are **not copied from the solver outputs**. Numerical FEMM tolerances remain representative benchmark checks and are not uniform 3D error guarantees.

## GitHub exact-head acceptance evidence

Candidate commit: `8c4917c416cef26636de3227e7cd1b4bb503c82f` in [PR #8](https://github.com/physicsvigilante/conductor_sim/pull/8).

- [Python 3.10/3.12 workflow](https://github.com/physicsvigilante/conductor_sim/actions/runs/43240533266): **192 passed**, **47 FEMM-dependent skipped** in normal Python CI. This includes 24 new M6b authored-model/negative-case/CLI/export checks.
- [Chromium source and visualization workflow](https://github.com/physicsvigilante/conductor_sim/actions/runs/43240533264): passed. It evaluated the new 15-task report, exported a new two-state shell simulation, exercised slider/controls/probe/provenance behavior and had no external resource requests or JavaScript errors under the existing browser contract.
- [Actual FEMM 5 workflow](https://github.com/physicsvigilante/conductor_sim/actions/runs/43240533547): **47 passed, 0 skipped**, consisting of **40 retained numerical/visualization/PDF regressions** and **7 new M6b tests**. Five authored scenarios check real FEMM field/potential and individual net charges, one confirms non-collinear FEMM rejection, and one solves two independently authored grounded-shell equilibrium states. It also exports a real FEMM HTML/JSON artifact for the newly authored two-source configuration.
- FEMM version is the pinned `femm-qt v5.0.20261101` (`7c31ccfa3d4bcda8123d822cdda31fce929f672e`) on Ubuntu 24.04, with actual headless FEMM and electrostatics solver, **not** the fake command recorder.

**No modifications were made to the physics solver, geometry compiler, physical models or M5 viewer implementation.** The M6b change set contains only authoring submissions, prompts, independent grading, tests, docs and workflow steps. A source diff must be checked before merging.

## Supported deliverables and limitations

- Reusable `m6b.protocol` for typed question attempts, and `m6b.evaluate` CLI for listing questions, grading a local trusted Python module, and exporting analytical/FEMM samples to offline self-contained HTML and JSON.
- Static field maps/lines, charge profiles and change-of-charge states are **precomputed equilibria**. No real-time PDE solve in browser, charge trajectories, ring physics, arbitrary external induction or general non-axisymmetric FEMM.
- Analytically solvable non-collinear charges are **not** silently sent to FEMM. A blocked case never produces a fake visual.
- The numerical point-source core and surface-charge density limitations documented in M2–M5 remain unchanged.

Next engineering priority: a reusable physics capability for exterior sources near a grounded conducting sphere (independent Kelvin-image formula, then FEMM/live convergence checks), after a separate external blinded authoring study if model-level performance must be measured.
