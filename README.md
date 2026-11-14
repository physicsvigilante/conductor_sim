# conductor_sim

## M19a — Prescribed fixed fields and test-charge motion

For fixed-source JEE field/energy problems, use `PrescribedField` with
`UniformElectricField`, `FixedPointField`, `UniformRingAxisField`,
`UniformDiskAxisField`, `UniformRodField`, `UniformSphereField` or
`UniformShellField`, and a separate `TestChargeMotionSpec`. The sources do
not respond to the moving particle: force is qE and potential energy is qV.
The existing M17/M18 adaptive RK4 controller is reused; conductor-induced
motion remains available through its original API.

Rings/discs are **axis-only** in this first milestone. Full-spatial sources
support motion in a physically certified plane. Thin-surface jumps and
singular sources stop at an approach guard; a smooth fixed volume-sphere
field can be traversed. No artificial conductor, projected force, or
invented off-axis field is substituted.

```powershell
python examples/test_charge_motion.py --case ring-turning --output output/m19a-ring.json
python examples/test_charge_motion.py --case uniform-deflection --output output/m19a-uniform.json
```

This stage exports numerical JSON, not a new HTML viewer. See
[M19a scope, source domains and equations](docs/m19a.md). The prescribed-source
viewer is the next stage; all existing conductor viewers remain supported.


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

## M19b — Prescribed-source interactive physics viewer

M19a's immutable fields and test-charge trajectories can now be explored in
the **same standalone offline HTML viewer** as the conductor engine. The
M19b adapter reuses its playback/time slider, vector arrows, path and energy
graphs. Rings and discs display **only actual symmetry-axis** electric-field
and potential profiles, with explicit gaps at undefined ideal surface values.
Other fixed sources show `E(t)` and `V(t)` at real stored test-particle
positions; the source geometry panel is explicitly **schematic**, not a
fabricated 2D field map. No extra numerical integrator, conductor induction,
source response or new physics formula is added.

```powershell
python examples/test_charge_viewer.py --case ring-turning --output output/m19b-ring.html
Start-Process .\output\m19b-ring.html
python examples/test_charge_viewer.py --case uniform-deflection --output output/m19b-uniform.html
```

All eight M19a cases are supported. See [M19b methods and limitations](docs/m19b.md).


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

## M8: live-validated isolated sphere with prescribed net charge

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

**M8 accepted for its stated limited domain: 59 actual FEMM 5 tests passed, with no skips.**
The force and interaction energy displayed by the example come from the
*analytical reference*, not FEMM force post-processing. The finite numerical
source core and far computational boundary remain approximations.
See [M8 physics and verification](docs/m8.md) and [M8 numerical acceptance](docs/m8-acceptance.md).

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


## M9: compose multiple external charges (real-FEMM validated)

Author several `PointCharge` objects outside the same `SolidSphere`, with
`Grounded()` or `FixedNetCharge(Q)`. `ExternalSphereBackend` dispatches
to an exact multi-image analytical reference for arbitrary source positions
in 3D. `Femm5ExternalSphereBackend` compiles separately charged source
cores into one FEMM 5 solve only when source sites share a line through the
sphere centre; **non-collinear numerical cases are rejected explicitly**.

Run `python examples/femm5_multi_external.py --output output/m9.html`
for analytical 4-state HTML/JSON, or configure FEMM 5 and add
`--backend femm --resolution 25`. The slider plays stored equilibria,
not continuous charge motion. See [M9 design](docs/m9.md) for the
force/energy convention and acceptance limits. **M9 passed 68 actual FEMM tests** (including all 59 earlier regressions),
plus independently validated analytical superposition and Chromium review.
See [M9 acceptance](docs/m9-acceptance.md) for scoped evidence and limitations.

## M10 accepted baseline: a charge above an infinite grounded conducting plane

`ConductingPlane(point_m, normal, Grounded())` defines an infinite metal
half-space in the same typed physical problem model. The independent exact
`GroundedPlaneBackend` supports arbitrary 3D charge placements above one plane;
At M10, `Femm5PlaneBackend` supported one source numerically using a finite
axisymmetric upper-half-space geometry. M11 extends it to multiple charge sites
sharing the **same plane-normal axis**. Non-axisymmetric numerical configurations
are still rejected, not silently approximated or replaced by analytic results.

The existing M5 viewer now renders a true planar boundary and a signed-distance
surface-charge profile, with no scene graph. Try an **analytical offline preview**:

```bash
python examples/femm5_grounded_plane.py --output output/grounded_plane.html
```

Real FEMM 5 requires `FEMM5_BINDING`/`FEMM_DIR` as documented in
[the FEMM setup](docs/femm5.md):

```bash
python examples/femm5_grounded_plane.py --backend femm --resolution 25 \
    --output output/grounded_plane_femm.html
```

**M10c was accepted with 75 real FEMM 5 tests.** The infinite
plane is approximated by a *finite numerical domain*. FEMM reports charge on
the truncated modeled plane; this must be compared to the exact finite-disk
integral, not to the charge on the infinite plane. See
[M10 design](docs/m10.md) and [acceptance evidence](docs/m10-acceptance.md).

## M11 — Multiple external charges above one grounded plane

The same physically authored `ElectrostaticProblem`, `ConductingPlane`,
`PointCharge`, and `Grounded` components can now use **multiple distinct
FEMM 5 source spheres** on one plane-normal axis. The independent exact
image backend remains fully three-dimensional, including non-collinear
positions. Axisymmetric FEMM cannot represent charges at different lateral
projections; it rejects them before starting a session. Coincident charges
combine exactly, while distinct cancelling sources remain separately solved.

```bash
python examples/femm5_multi_plane.py --output output/m11-plane.html
python examples/femm5_multi_plane.py --backend femm --output output/m11-femm.html
```

The second command requires the pinned FEMM 5 binding. Each output includes
three stored equilibria and the plane's spatial density profile. A real FEMM
conductor integrates charge on a **finite disk**, not the ideal infinite plane: see
[physical/numerical distinctions](docs/m11.md) and [M11 numerical acceptance](docs/m11-acceptance.md)
(83 real FEMM 5 tests, zero skipped). This does not expand numerical
capability to arbitrary 3D, parallel plates, or browser-time solving.


## M12a — Parallel plates (analytical only)

Compose **two facing** `ConductingPlane` components with `Grounded()`
or `FixedPotential(volts)` and any supported point charges strictly between
them. `ParallelPlateBackend(image_order=48)` combines the exact uniform-field
solution with a truncated image-charge series for charged arrangements.
Infinite plates at different potentials have undefined/unbounded integrated
charge totals; these are displayed as `n/a`, while local surface density
remains defined. The existing viewer renders both planar metal faces.

```bash
python examples/parallel_plates.py --output output/parallel_plates.html
```

The M12a backend itself remains **analytical only**. The separately
validated M12b FEMM numerical backend appears below; it reuses the same
authored models. See [M12a scope](docs/m12a.md).


## M12b — finite-cylinder FEMM 5 parallel-plate numerical acceptance

The same M12a authored capacitor can be solved numerically with
`Femm5ParallelPlateBackend()` for **zero or up to 16 axial active charge
sites**, preserving the independent analytical backend. FEMM computes finite
electrode-disk potentials/charges and the separate charged source-core
properties. The outer cylindrical side has **zero radial electric flux**, not
an infinite-plate boundary. Numerical local V/E/sigma require convergence
checks, and non-axisymmetric sources are refused before opening FEMM.

```bash
python examples/parallel_plates.py --backend femm --resolution 21 --output output/m12-femm.html
```

**M12b numerical acceptance passed: 90 actual FEMM 5 tests, zero skipped;
303 ordinary Python tests on each supported version, and Chromium on an
actual FEMM-generated HTML preview.** See [M12b design and finite-disk
identity](docs/m12b.md) and [real numerical acceptance](docs/m12b-acceptance.md).
This is not a uniform numerical-error guarantee or support for non-axial FEMM.


## M13b — Reusable analytical position-sweep explorer

The existing viewer can display precomputed **one-charge straight-line position sweeps**
from the same typed physical model: a selected authored charge moves while all
other charges, conductors and voltage constraints remain fixed. The same
`PositionSweepSpec` and `sample_position_sweep` functions work with analytical
grounded planes, exterior spheres (grounded or fixed Q), and two-plate systems.
The viewer adds a selected-charge force arrow, a force-component-versus-position
graph, a mechanical-interaction-energy graph, a breakdown of direct/induced/
background terms, and the quasistatic work relative to the initial position.
The position slider is **not time**; all states were solved when generating HTML.

```bash
python examples/force_energy_explorer.py --case plates-grounded --output output/m13b.html
python examples/force_energy_explorer.py --case sphere-isolated --output output/m13b-sphere.html
```

The calculations remain analytical and explicit: an unsupported FEMM force
observation is rejected, not substituted with an image result. Existing
visualization exports without this metadata retain their original interface.
See [M13b design and limitations](docs/m13b.md).


## M13c — open authorability and source-anchored conductor PDF audit

A separate `m13c` evaluation suite authors ten analytical force/energy
position sweeps from fresh JEE-style prompts and explains eight unsupported
requests **without editing the electrostatics engine**. Its independent
image-charge scoring also checks geometry, individual force/energy components
and work, not just rendered pixels. This is one **open assistant attempt**;
it is not a blind LLM-generalization measurement.

```bash
python -m m13c.evaluate grade --output output/m13c.json
python -m m13c.evaluate render plates-grounded --output output/m13c.html
```

The new **17-page scanned** `02_conductors_half_size.pdf` is now SHA-pinned
and visually audited. Of **18 selected PDF questions**, two are independently
checked conductor demonstrations (grounded sphere p8 Q21; neutral tangent
sphere p9 Q27), two are static-potential-only, and 14 are outside the current
force/energy explorer. This is distinct from the 18 original open prompts.
Both analytical PDF explorers reuse the existing viewer; no new solver code.

```bash
python -m m13c.pdf_cases grade --output output/m13c-pdf-audit.json
python -m m13c.pdf_cases render p09-multiple-27 --output output/m13c-neutral-sphere.html
```

See [M13c source audit and limitations](docs/m13c.md).

## M14a — connected concentric conductors, without case-specific solver code

Add named `ElectricalLink` connections to a `ConnectionPlan` containing an
ordinary `ElectrostaticProblem`. `ConnectedShellBackend` solves the equal-
potential/charge-conservation constraints once, then reuses the **existing**
`NestedShellBackend` for the before/after equilibria. It returns each shell's
charge transfer and the electrostatic field-energy decrease, excluding the
unchanged inner-cavity field contribution when point sources exist.

This directly models the idealized **page 14, Numerical Q7** of
`02_conductors_half_size.pdf`: connecting shells of radii R and 2R charged
q and 2q does **not** change the outer shell potential (`ΔV = 0`).

```bash
python examples/connected_shells.py --case pdf-p14-q7 --output output/connected_shells.html
```

The HTML reuses the same two-state viewer and is not a current/time simulation.
Other composable examples are `--case ground-link`, `--case cavity-charge`, and
`--case three-shell`. Scope is analytical, for 2–8 strictly concentric
nonintersecting **shells**, with floating or grounded disjoint wire groups;
not a 3D finite-wire or standalone FEMM wiring solver. See [M14a](docs/m14a.md).

## M14b — A solid metal sphere inside connected concentric shells

The same `ConnectionPlan` and `ElectricalLink` introduced in M14a now accept
one innermost `SolidSphere` surrounded by strictly separated concentric
`SphericalShell` objects, with arbitrary valid floating/grounded link groups.
An exact new `MixedRadialBackend` handles that radial, charge-free geometry
without treating the solid as a fictitious hollow shell; the existing
nested-shell solver and HTML viewer are reused where applicable.

```bash
python examples/solid_sphere_connected.py --case pdf-p03-q27 --resolution 11 --line-steps 20 --output output/m14b-solid.html
python examples/solid_sphere_connected.py --case pdf-p10-q13-q14 --resolution 11 --line-steps 20 --output output/m14b-shells.html
```

From `02_conductors_half_size.pdf`, page 3 Q27's +20 μC solid sphere (10 cm)
joined to a 20 cm enclosing shell releases **approximately 9 J**; page 10
linked Q13–14's oppositely charged shells release **approximately 0.09 J**.
Both are source-anchored, validated **static** results, not simulations of wire
current. The imported spheres/shells must be concentric with no charges in the
new hybrid solid+shell configuration. See [M14b](docs/m14b.md) for capability
boundaries and the independent energy derivations.

## M15a/b — Electrostatic forces and mechanical energy inside cavities

The existing analytical shell/cavity solvers now expose the same
`force_energy(solution)` observations as grounded planes, exterior spheres,
and parallel plates. For one or more point charges *inside* a single spherical
shell or the innermost cavity of nested concentric shells, M15 calculates
self-field-free forces, exact Kelvin-induced interaction energies, conductor
background terms, and quasistatic work as a charge changes its position.
Grounded and fixed-net-charge shell constraints are supported; FEMM force
observations remain explicitly unsupported.

The same `PositionSweepSpec` and offline force/energy viewer handle valid
straight-line paths **inside the cavity**, including an initial position at
the exact centre (zero induced force and no arrow). No dynamics or scene graph
is added. Five working examples extend the existing command:

```bash
python examples/force_energy_explorer.py --case cavity-grounded --output output/cavity.html
python examples/force_energy_explorer.py --case cavity-nested --output output/nested-cavity.html
python examples/force_energy_explorer.py --case cavity-multi --output output/multi-cavity.html
```

Other choices are `cavity-isolated` and `cavity-centered`. See the
[M15a/b physics, interaction-energy and acceptance contract](docs/m15.md).

## M16a — True 1D electrostatic particle motion (analytical)

A single charge with mass and initial velocity can now evolve under the same
self-field-free electrostatic force/energy API used by M13–M15.
`Motion1DSpec` proves a physically invariant 1D direction for normal plane
motion, radial exterior-sphere motion or motion along a line through a cavity
centre. `integrate_motion_1d()` returns elapsed-time states, velocity,
acceleration, kinetic and interaction energy, turning points and guarded
surface-approach events. It rejects an unsupported transverse force rather
than simulating a fictitious straight track. No numerical FEMM force or
browser-time animation has been added.

```bash
python examples/particle_motion_1d.py --case cavity --output output/m16-cavity.json
```

See [M16a equations, constraints and validation](docs/m16a.md).

## M16b — Standalone animated elapsed-time electrostatics

The M16a numerical motion result can now be displayed with **real time-based
playback** in the same offline HTML viewer (not the M13–M15 quasistatic
position slider). The exporter preserves every numerical sample for
position/velocity/kinetic-and-interaction-energy plots, plus a bounded number
of separately solved potential/field visual snapshots at those real times.
Play, pause, restart, speed, sliders and the existing potential/field-line
controls all work without a web server or browser-time physics solver.

```powershell
python examples/particle_motion_viewer.py --case cavity --output output/m16b-cavity.html
Start-Process .\output\m16b-cavity.html
```

Other `--case` choices include `plane`, `plates`, `sphere-exterior` and
`turning`. For an elementary electron between biased plates, supply its
charge, mass, duration and sample interval explicitly. See
[M16b implementation and scope](docs/m16b.md).

## M17a — True planar curved trajectories

`Motion2DSpec` and `integrate_motion_2d` extend M16 dynamics to
three-component velocities and positions, while proving a physically
invariant plane for fixed parallel conductors or spherical symmetry.
Angled launches now produce genuine curved time trajectories. The existing
analytical `force_energy` provides forces and mechanical interaction
energies; an adaptive RK4 stepper is shared with M16a. No browser-side
integration or FEMM force substitution is introduced.

```powershell
python examples/particle_motion_2d.py --case plates --output output/m17a-plates.json
python examples/particle_motion_2d.py --case sphere --output output/m17a-sphere.json
```

M17a exports numerical trajectory data as JSON; M17b now renders it in the
same reusable offline HTML field viewer. See [M17a scope](docs/m17a.md).

## M17b — Interactive curved-trajectory explorer

The existing standalone offline viewer now understands `sample_motion_2d`:
real curved particle paths, projected vector velocity/force arrows and two
planar displacement/velocity component plots, alongside M16b time controls,
shared potential/field-line/density maps and K/U/K+U energy graphs. The browser
**never integrates electrostatic dynamics**. Field maps are independently solved
at selected actual M17a times (at most 24), and every full numerical time
sample remains in the trajectory plot. Unsupported FEMM force remains an
explicit refusal, not a fallback.

```powershell
python examples/particle_motion_2d_viewer.py --case plates --output output/m17b-plates.html
Start-Process .\output\m17b-plates.html
```

Further cases: `plane`, `sphere`, `cavity`, `center` and `guard`.
See [M17b design and scientific constraints](docs/m17b.md).

## M17c — Curved-motion authoring and correctness review

The original open motion-authoring trial has **12 prompts: 8 runnable
trajectories and 4 justified refusals**. It independently checks authored
models, units, time evolution, conductor forces and mechanical invariants.
This is not blind LLM accuracy, an automatic prompt parser or new PDF coverage.
Rotated radial viewing, saved-data validation and sparse paths near metal
have also been corrected without changing the force laws or integrator.

```powershell
python -m m17c.evaluate grade --output output/m17c-grade.json
python -m m17c.evaluate render electron-plates --output output/m17c-electron.html
Start-Process .\output\m17c-electron.html
```

See [M17c references, repairs and verification](docs/m17c.md).

## M18a — One moving point charge with other charges held fixed

The original `Motion2DSpec` and `integrate_motion_2d` now accept an authored
inventory with **one selected moving source and additional externally held
stationary sources**. The only new public choice is the selected authored
charge index and optional `charge_guard_m`: the electrostatic models,
`force_energy` providers and adaptive RK4 kernel are reused. Held sources
produce real direct and conductor-induced forces; their positions are never
integrated, and energy differences still use the mechanical interaction
potential. All sources and conductors must preserve a physical symmetry
plane; unsupported layouts or pairwise singular contact fail closed.

```powershell
python examples/particle_motion_fixed_charges.py --case plane --output output/m18a-plane.json
python examples/particle_motion_fixed_charges.py --case sphere --output output/m18a-sphere.json
```

This milestone produces numerical time-series **JSON**. The older M17b
single-charge HTML viewer explicitly refuses multi-charge motion rather than
omitting stationary sources; rendering *all* charges is M18b's task.
See [M18a domain and scientific limitations](docs/m18a.md).

## M18b — Interactive motion with held point charges

The existing `sample_motion_2d` exporter now displays every M18a source, with
stable original indices, a moving circle and held-source squares. It reuses
the time controls, field maps, force/velocity arrows and energy graphs. Every
field snapshot is solved with the complete charge inventory at its actual
saved time; there is no second integrator or omitted stationary source.

```powershell
python examples/particle_motion_fixed_viewer.py --case plane --output output/m18b-plane.html
Start-Process .\output\m18b-plane.html
```

Other cases: `plates`, `sphere`, `cavity`, `nested`, `guard`. See
[M18b scope, corrections and acceptance](docs/m18b.md). The earlier M18a
viewer refusal is superseded; its JSON-only CLI remains unchanged.
