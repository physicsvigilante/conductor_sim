# conductor_sim

Composable, Python-authored electrostatics for JEE conductor problems. Physical
components, numerical/analytical solvers, and visualization are separate layers.

**M1–M4 physics baseline:** independently grounded or fixed-charge concentric
spherical shells, with point charges in the innermost cavity. FEMM supports
collinear sources on a common axis through the shell centre; independent exact
analytical backends also support non-collinear cavity sources.

**M5 visualization:** an offline procedural viewer with potential maps,
equipotentials, directed field lines, conductor results, surface-density profiles,
and stored parameter sweeps. Review and live numerical/browser evidence are
recorded in [docs/m5.md](docs/m5.md) and [the acceptance scope](docs/m5-acceptance.md). No scene graph has been introduced.

## M6a: PDF-grounded JEE authorability benchmark

A traceable catalog of 19 problems from the checked-in
[`POTENTIAL_CONDUCTORS1.pdf`](POTENTIAL_CONDUCTORS1.pdf) is now available:
13 were runnable at the M6a acceptance date; 6 were then **explicitly
blocked**. The exterior-grounded-sphere case is now supported by **M7**;
the M6a counts remain a frozen historical authorability baseline. The benchmark uses
independent analytical checks and can render supported cases with the M5 viewer.
It is manually authored and is **not yet** an automatic natural-language parser.

```bash
python -m conductor_sim.pdf_benchmark list
python -m conductor_sim.pdf_benchmark check --output output/m6-report.json
python -m conductor_sim.pdf_benchmark render p69-cavity-central-vs-shifted --output output/m6-induction.html
```

See [M6 benchmark casebook](docs/m6-benchmark.md) for original PDF page
anchors and finite-thickness corrections; [M6a acceptance](docs/m6-acceptance.md)
records 40 actual FEMM tests and browser verification of the curated scope.

## M6b: held-out JEE authoring trial

A [15-question open authorability trial](docs/m6b.md) tests whether an
LLM can compose *new* charge/shell configurations and request measurements
using the existing engine unchanged. Nine questions are supported by both
FEMM 5 and the analytical reference; one non-collinear cavity arrangement
requires the analytical backend; five require missing physics and must be
refused rather than simulated with invented geometry.

```bash
python -m m6b.evaluate list
python -m m6b.evaluate grade --output output/m6b-report.json
python -m m6b.evaluate render grounded-two-state-sweep --output output/m6b.html
```

The [accepted M6b trial](docs/m6b-acceptance.md) passed 192 Python tests
and 47 actual FEMM tests, including 7 M6b numerical/diagnostic cases.
This is reproducible but **not** a blinded benchmark of arbitrary LLMs or
an automatic textbook parser. The browser slider selects stored equilibria.

## M7: live-validated external charge near a grounded solid conducting sphere

A new **`SolidSphere`** physical component represents metal without a hollow
cavity. `ExternalGroundedSphereBackend` uses the exact external Kelvin image
charge; `Femm5ExternalSphereBackend` creates a separate real FEMM 5 axial solve
with an independently charged numerical source core. Both expose potential,
field, conductor charge and outer induced surface-charge density through the
existing sampling/viewer interfaces. The exact reference also reports the
attractive force on the external point charge and its interaction energy.

```powershell
python examples/femm5_external_sphere.py --output output/external.html
Start-Process .\output\external.html
# With FEMM 5 configured:
python examples/femm5_external_sphere.py --backend femm --resolution 25 --output output/external_femm.html
```

These states represent **precomputed static equilibria**, not moving charges
or time-domain particle dynamics. Current M7 scope is one nonzero point charge
strictly outside **one grounded solid sphere**. Existing 
`Femm5Backend` continues to handle the earlier *interior-shell* configurations;
no automatic backend fallback is introduced. **M7 passed 52 actual FEMM 5 integration tests**, including five new external-grounded-sphere cases. See [design](docs/m7.md) and [numerical acceptance](docs/m7-acceptance.md).

## M8 candidate: isolated sphere with prescribed net charge

`ExternalIsolatedSphereBackend` adds the exact two-image solution for one
external charge near an isolated **solid** conductor with `FixedNetCharge(Q)`.
A second compensating *central* image ensures that the sphere's net charge
is exactly Q while its potential is calculated, not grounded. The shared
`ExternalSphereBackend` handles both grounded and isolated constraints
without changing authored physics. The same `Femm5ExternalSphereBackend`
geometry now supports fixed-total-charge conductor properties.

The existing M5 HTML viewer can compare four solved static equilibria
(grounded, isolated neutral, isolated positive-Q, isolated negative-Q):

```powershell
python examples/femm5_isolated_sphere.py --output output/isolated_sphere.html
Start-Process .\output\isolated_sphere.html
# With FEMM 5 configured for a genuine numerical solve:
python examples/femm5_isolated_sphere.py --backend femm --resolution 25 --output output/isolated_sphere_femm.html
```

**M8 is not accepted until actual pinned FEMM 5 and browser checks pass.**
The force and interaction energy displayed by the example come from the
*analytical reference*, not FEMM force post-processing. The finite numerical
source core and far computational boundary remain approximations.
See [M8 physics and verification](docs/m8.md).

## First visual preview (no FEMM installation required)

```bash
python -m pip install -e ".[test]"
python -m pytest -m "not femm5"
python examples/visualize_conductors.py --output output/conductors.html
```

Open `output/conductors.html` in a browser. It is self-contained and works without
an internet connection. The example generates three **analytical** solutions for
two charges inside a neutral shell surrounded by a grounded shell. The slider
selects stored solutions; it does **not** recompute FEMM or animate charge motion.
The adjacent JSON file preserves sampled SI values, geometry, masks and provenance.

For genuine FEMM data, configure [FEMM 5](docs/femm5.md), then run:

```bash
python examples/visualize_conductors.py --backend femm --resolution 31 --output output/femm.html
python -m pytest -m femm5 -vv
```

A FEMM execution error is not replaced by an analytical solution.

## Compose a problem

```python
from conductor_sim import (
    ElectrostaticProblem, PointCharge, SphericalShell,
    FixedNetCharge, Grounded, NestedShellBackend,
)
from conductor_sim.visualization import SliceSpec, sample_sweep, write_visualization

problem = ElectrostaticProblem(
    charges=(PointCharge(1e-9, (0, 0, 0.035)),
             PointCharge(0.45e-9, (0, 0, -0.025))),
    conductors=(SphericalShell(0.10, 0.12, FixedNetCharge(0), name="inner"),
                SphericalShell(0.20, 0.22, Grounded(), name="outer")),
)
plane = SliceSpec.meridian(problem, extent_m=0.30, resolution=61)
bundle = sample_sweep([("Two charges, two shells", problem)],
                      NestedShellBackend(), plane=plane)
write_visualization(bundle, "output/my_problem.html")
```

Use `Femm5Backend()` instead of `NestedShellBackend()` for numerical solves.
All quantities are SI. Source positions are world-space 3D coordinates, not
FEMM's cylindrical coordinates. Changing supported components does not require
writing new FEMM commands or renderer code.

## Supported domains and important limits

FEMM: up to eight strictly separated, concentric spherical shells and 16 active
collinear source sites **inside the innermost cavity**. These are implementation
limits, not a claim that every combination or near-wall configuration has been
numerically validated. The live benchmarks include representative two-source,
one/two/three-shell cases and sensitivity checks.

Analytical: `ConcentricShellBackend`, `ImageChargeShellBackend`,
`MultiImageShellBackend`, and `NestedShellBackend` expose increasingly general
exact references within their documented spherical domains.

FEMM point sources are finite charged-sphere approximations. Its spatial surface
density is estimated from one-sided near-surface fields. Integrated inner/outer
charge totals are inferred from Gauss's law using measured net charges, not
independent surface integrals. The viewer labels these distinctions. It excludes
source cores, stops traces at metal boundaries, and does not interpolate through
invalid samples. Non-planar fields stop 2D field tracing rather than becoming
mislabelled projected field lines. Surface profiles are great-circle samples,
not automatically inferred area integrals or an assertion of axisymmetry.

Not supported numerically: eccentric/touching shells, charges in inter-shell gaps,
non-collinear FEMM sources, arbitrary 3D conductor geometry. Production-grade
process isolation and hard session timeouts remain outstanding. Use only trusted,
bounded authored models.

## Evidence and architecture

[Architecture](docs/architecture.md) · [Roadmap](docs/roadmap.md) ·
[M4a acceptance](docs/m4a-acceptance.md) · [M4b acceptance](docs/m4b-acceptance.md) ·
[M5 review and visualization](docs/m5.md)

Normal Python CI tests compiler contracts and analytical physics. The separate
FEMM workflow builds the pinned real headless solver and runs numerical tests.
A skipped FEMM test is not numerical evidence. Browser checks exercise the exported
viewer independently of the solver. Numerical tolerances are benchmark criteria,
not uniform error bounds throughout every supported configuration.
