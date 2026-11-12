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
