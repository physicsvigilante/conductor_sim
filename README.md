# conductor_sim

A composable, Python-authored electrostatics engine for **JEE-level conductor problems**.
Physical components are defined independently of the numerical solver and eventual viewer.

**Status: M1–M4a accepted for documented domains; M4b nested conductors is next.**
The analytical reference solves one spherical shell with charges at its center. The
M2a FEMM 5 backend was validated for an empty isolated charged shell.
M2b adds **centered point charge(s)** and **grounding**. M3 adds **a single off-centre source location** inside the spherical cavity, with rotation of the symmetry axis automatically handled by the backend. Multiple distinct source positions, non-spherical conductors, and arbitrary 3D geometries are not implemented.

## Quick start

```bash
python -m pip install -e '.[test]'
pytest
python examples/concentric_shell.py
```

On Windows PowerShell, use `python -m pip install -e ".[test]"` if needed.

## Author a problem

```python
from conductor_sim import (
    ConcentricShellBackend, ElectrostaticProblem,
    FixedNetCharge, PointCharge, SphericalShell,
)

problem = ElectrostaticProblem(
    charges=(PointCharge(charge_c=1e-9),),
    conductors=(
        SphericalShell(
            inner_radius_m=0.1,
            outer_radius_m=0.12,
            constraint=FixedNetCharge(charge_c=0.0),
        ),
    ),
)
solution = ConcentricShellBackend().solve(problem)
print(solution.conductor_results)
print(solution.potential_at((0.05, 0.0, 0.0)))
```

Switch to `Grounded()` to ground the shell; no solver geometry script changes.
All quantities use SI units. `PointCharge.position_m` and `SphericalShell.center_m` are **3D positions**, not FEMM's axisymmetric `(r,z)` coordinates.

## Design principles

- **Physical semantics first:** typed charges, connected conductors and electrical constraints, with immutable problem definitions.
- **Explicit solver capability:** unsupported configurations fail clearly; never silently pretend a general 3D solution exists.
- **Independent backends:** the `ElectrostaticBackend` / `FieldSolution` protocols permit a numerical FEMM 5 backend or a future 3D backend without changing problem definitions.
- **Viewer independence:** field values and conductor charge/potential are accessible without a rendering engine.
- **Physics validation:** analytical solutions are baselines for future FEMM numerical regression tests.

See [architecture](docs/architecture.md) and [roadmap](docs/roadmap.md).

## FEMM 5 installation (optional numerical backend)

FEMM 5 provides a separate Python binding from FEMM 4.2. **Do not assume** that `pip install pyfemm` installs the FEMM 5 binding; follow the official binding setup instructions at https://femm.info/doku/doku.php?id=bindingsetup and https://www.femm.info/doku/doku.php?id=pyfemm-qt .

The M2a adapter handles the supported axisymmetric empty-shell problem. Extending it to ideal point charges requires validated source modeling and convergence checks.

## M2a: live-validated limited FEMM 5 backend

A **limited FEMM 5 adapter** is now implemented for an empty, isolated
spherical conducting shell with prescribed net charge. It uses the same
`ElectrostaticProblem` definition as the analytical backend and an
axisymmetric FEMM 5 geometry compiler. The **real FEMM 5.0.20261101** integration suite passed on Ubuntu 24.04.
See [the accepted M2a evidence](docs/m2a-acceptance.md). The normal Python CI
uses command recorders; a **separate FEMM 5 workflow** builds and runs the real solver.

```python
from conductor_sim import Femm5Backend, ElectrostaticProblem, SphericalShell, FixedNetCharge

problem = ElectrostaticProblem(conductors=(
    SphericalShell(0.1, 0.12, FixedNetCharge(1e-9)),
))
with Femm5Backend().solve(problem) as solution:
    print(solution.conductor_results)
    print(solution.potential_at((0.24, 0, 0)))
```

Requires local FEMM 5 and `FEMM5_BINDING` configured to the **FEMM 5**
`pyfemm/femm.py` file (this is not the older PyPI `pyfemm` package).
See [setup, limitations, and live validation](docs/femm5.md).

## M2b: live-validated centered charge and grounded shell

Construct the **same** `ElectrostaticProblem` as in the analytical example
above, then use `with Femm5Backend().solve(problem) as solution:`.
The numerical backend translates centered point charge(s) into a small
concentric charged source sphere, enforcing their net charge and disallowing
field/potential sampling inside the artificial source core.
For M2b source, grounded and isolated-shell examples see
[docs/m2b.md](docs/m2b.md) and
[examples/femm5_centered_charge.py](examples/femm5_centered_charge.py).
**Real FEMM 5 numerical acceptance passed:** nine live tests (no skips).
See [M2b acceptance evidence](docs/m2b-acceptance.md).

## M3: live-validated off-centre charge induction

The **accepted M3 numerical backend** adds an axisymmetric off-center point-charge approximation: a finite charged sphere positioned along the automatically chosen symmetry axis. This supports a point charge at any orientation inside the spherical cavity, with grounded or fixed-net-charge shell constraints, while rejecting multiple distinct charge locations. `ImageChargeShellBackend` provides an **independent exact Kelvin image-charge reference** for potential, field and induced surface-charge density.

Numerical `solution.surface_charge_density("inner", theta_rad)` estimates the **local induced surface density from FEMM's one-sided near-surface electric field**, not from an analytical formula or direct boundary-element extraction. The artificial source size and mesh must be sensitivity-checked. See [M3 design](docs/m3.md) and [real FEMM acceptance evidence](docs/m3-acceptance.md). **17 live FEMM tests passed, without skips**, including eight new off-centre cases and nine retained M2 regressions.

## M4a candidate: compose multiple independent charge sites

An arbitrary list of `PointCharge` components can be passed to the same
`ElectrostaticProblem`. In the experimental M4a FEMM backend, all active
charge sites must lie on **one line through the spherical shell's centre**;
the engine detects that line and rotates it to FEMM's symmetry axis.
Each distinct site becomes an independently charged numerical source core;
coincident charges are combined, and zero-net locations are omitted.
No per-charge FEMM code is required from the author.

`MultiImageShellBackend` separately evaluates the **exact Kelvin-image
superposition** for multiple charges anywhere in the spherical cavity,
including non-collinear 3D arrangements. FEMM does **not** accept those
non-collinear cases, and reports an explicit capability error.

For a multi-site FEMM solution, use `numerical_sources` to inspect source
positions, radii and measured charges. The legacy `source_radius_m` field
remains defined only for a single numerical source. FEMM density profiles
remain one-sided near-surface estimates, not direct boundary-element data.

**M4a accepted after 23 actual FEMM 5 tests passed.** See [M4a design](docs/m4a.md) and [M4a acceptance](docs/m4a-acceptance.md).
