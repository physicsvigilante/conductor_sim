# conductor_sim

A composable, Python-authored electrostatics engine for **JEE-level conductor problems**.
Physical components are defined independently of the numerical solver and eventual viewer.

**Status: M1, M2a, and M2b accepted for their explicitly limited domains.**
The analytical reference solves one spherical shell with charges at its center. The
M2a FEMM 5 backend was validated for an empty isolated charged shell.
M2b adds **centered point charge(s)** and **grounding**, with all nine real FEMM 5 numerical tests passing.
Off-center and arbitrary 3D geometries are not implemented.

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
