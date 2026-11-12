# M6a — JEE problem authorability benchmark from `POTENTIAL_CONDUCTORS1.pdf`

**Status:** M6a accepted for the curated scope on 2026-11-12; [numerical and browser evidence](m6-acceptance.md).
**Source:** repository root `POTENTIAL_CONDUCTORS1.pdf`, **71 physical PDF pages**.
Exact checked-in file SHA-256:
`8d0d9b5cf821219058cfc4b7685a39b777e460944df44c3f9cffcb68157a60e5`.
Pages below are **PDF page numbers (starting at 1)**, not video lesson numbers.

## What this tests

We chose **19 identifiable teaching illustrations or derivations** from the PDF.
For each one we ask whether the *currently implemented* engine can author an
unambiguous physical model **without changing any solver or renderer code**.
Thirteen are runnable using existing `PointCharge`, `SphericalShell`,
`Grounded`, `FixedNetCharge`, `NestedShellBackend`, `Femm5Backend`, and the M5
visualization pipeline. Six require new capabilities. The registry contains
both executable model factories and explicit failure explanations. A supported
case must pass independent analytical checks, not merely return a solution.

The benchmark is **manually curated** and does not claim automatic parsing of
textbook figures, LLM success rates, or unsupervised question interpretation.
It is the first reproducible authorability baseline; a later M6 step should
ask the LLM to author *previously unseen* problems without editing the engine.

## Case inventory

| ID | PDF page / anchor | Classification | What is checked or missing |
| --- | --- | --- | --- |
| `p26-conductor-potential` | 26 / Illustration 1 | Runnable | Sphere V at R/2 and 3R/2, interior E=0 |
| `p26-centre-to-three-r` | 26 / Illustration 3 | Runnable | Centre-to-3R potential and E at 3R |
| `p27-cavity-two-shells` | 27 / Illustration 4 | Runnable | Cavity charge, two shells, induced charge and potential |
| `p27-two-shell-potentials` | 27 / Illustration 5 | Runnable | Potentials inside two nested charged shells |
| `p28-two-shell-q-2q` | 28 / Illustration 6 | Runnable | Radii a,3a; charge Q,2Q; cavity and gap values |
| `p28-three-shells` | 28 / Illustration 7 | Runnable | Three shell superposition in multiple radial regions |
| `p29-shell-potential-difference` | 29 / Illustration 10 | Runnable | Voltage difference between nested shells |
| `p29-outer-charge-change` | 29 / Illustration 11 | Runnable | Change outer net charge; verify voltage difference invariance |
| `p30-inner-charge-change` | 30 / Illustration 12 | Runnable | Quadruple inner charge; verify 4× voltage difference |
| `p62-ground-outer` | 62 / Illustration 2 | Runnable | Charge and exterior shielding of grounded outer shell |
| `p63-ground-inner` | 63 / Illustration 4 | Runnable | Charge on grounded inner shell in presence of outer Q |
| `p69-cavity-central-vs-shifted` | 69 / Illustration 7 | Runnable | Uniform vs nonuniform inner density; surface totals |
| `p70-cavity-offset` | 70 / Illustration 8(i) | Runnable | Spherical neutral conductor with off-centre cavity charge |
| `p20-ring-axis` | 20 / ring potential derivation | New component | Uniform charged ring potential/field primitive missing |
| `p25-insulating-sphere` | 25 / nonconducting sphere | New component | Volumetric fixed charge density/insulating sphere missing |
| `p13-projected-charge` | 13 / Illustration 7 | New dynamics | Particle motion/turning-point solver missing |
| `p61-external-grounded-sphere` | 61 / Illustration 1 | New solver domain | Charge outside a grounded conductor not compiled |
| `p68-connected-separated-spheres` | 68 / Illustration 4 | New constraint/geometry | Two separated spheres electrically connected by a wire |
| `p70-three-cavities` | 70 / Illustration 9 | New solver domain | Several nonconcentric cavities within one irregular conductor |

The final two illustrations are **not interchangeable with one concentric
spherical cavity**. They must remain blocked even if a plausible-looking
rendering could be produced from a simplified model.

## Faithfulness and numerical assumptions

The benchmark reproduces the **physical relationships** in these PDF items,
not necessarily their literal numerical parameters or every pictured probe.
All runnable fixtures use specified SI quantities, generally `Q = 1 nC`
and characteristic radius `a = 0.10 m`. Some vary positions or total charges
as separate static equilibrium states. They are not time-dependent simulations.

**Important finite-thickness correction:** the diagrams often idealize a shell
as a surface with zero metal thickness. Our `SphericalShell` has distinct
inner and outer radii. For example, for inner shell outer radius `a`, outer
shell inner radius `c` and outer radius `b`, with shell net charges `Q1,Q2`
and no cavity charge:

- Outer conductor: `V2 = k(Q1+Q2)/b`.
- Inner conductor: `V1 = V2 + k Q1(1/a - 1/c)`.
- Hence `V1 - V2 = k Q1(1/a - 1/c)`, not `k Q1(1/a-1/b)` unless `c=b`
  in the ideal thin-sheet limit.

Our checks use the authored *finite* geometry and independently derived
Gauss-law reference equations. We do **not** silently substitute ideal
thin-sheet answers or relax test tolerances to make mismatched geometry pass.
The p.63 grounded-inner example is similarly corrected for the finite wall
thickness of the outer shell.

The PDF is used as a **source of problem types and exercises**, not as an
unquestioned source of physical truth. Potential/field observations in code
have independent closed-form expectations. FEMM source cores are finite
charged spheres; live numerical tolerances remain looser than analytical
checks and must be reported separately.

## Reproduce the benchmark

From a local checkout on `main` after M6a is merged:

```bash
python -m pip install -e ".[test]"
python -m conductor_sim.pdf_benchmark list
python -m conductor_sim.pdf_benchmark check --output output/m6-report.json
python -m conductor_sim.pdf_benchmark render p29-outer-charge-change --output output/m6-voltage.html
python -m conductor_sim.pdf_benchmark render p69-cavity-central-vs-shifted --output output/m6-induction.html
```

Open the generated `.html` files in a browser. Each has adjacent `.json`
with raw samples and a PDF page/sha citation. The browser slider selects
**precomputed equilibrium states**; it does not animate a particle or rerun a
solver. Rendering with `--backend femm` requires a real compatible FEMM 5
installation and never silently falls back to analytical values.

Trying to render a blocked case exits with a clear error and does **not**
write a simulation file:

```bash
python -m conductor_sim.pdf_benchmark render p61-external-grounded-sphere
```

## Acceptance gates and next step

1. The exact PDF SHA is checked so a revised PDF cannot silently reuse old
   page anchors. All 19 cases must have unique IDs and source anchors.
2. All 13 runnable factories must be accepted by both the current analytical
   and FEMM capability checks. Independent analytical observations must pass.
3. All six unsupported cases must return explicit, stable errors. No
   fabricated substitute geometry or analytical fallback is allowed.
4. At least two exported PDF exercises must pass Chromium viewer checks,
   including state changes, source labels, controls and no external requests.
5. Representative PDF exercises must pass **real FEMM 5** tests, in addition
   to all existing numerical regression tests.

Next: run an **unseen-question authorability evaluation**. Present fresh PDF
exercises to an LLM, have it write only composition code and observation
requests, and score whether a solver/editor source change was necessary. Do
not conflate the present curated benchmark with an autonomous text-to-simulation
system. After that, expand reusable physics for blocked domains, starting
with an exterior charge near a grounded sphere and charged rings.
