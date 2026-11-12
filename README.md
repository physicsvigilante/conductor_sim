# conductor_sim

A composable, Python-authored electrostatics engine for **JEE-level conductor problems**.
Physical components are defined independently of the numerical solver and eventual viewer.

**Status: M1 foundation, with an analytical spherical-shell reference solver.**
This initial commit does **not** yet run FEMM 5, perform numerical meshing, solve off-centre charges, or render field lines. Those are future milestones; the present analytical backend is deliberately limited to one conducting spherical shell with charges at its centre.

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

## FEMM 5 installation note (not required for M1)

FEMM 5 provides a separate Python binding from FEMM 4.2. **Do not assume** that `pip install pyfemm` installs the FEMM 5 binding; follow the official binding setup instructions at https://femm.info/doku/doku.php?id=bindingsetup and https://www.femm.info/doku/doku.php?id=pyfemm-qt .

The future FEMM backend will validate axial symmetry before compilation, document the representation of ideal point charges and conductor charge constraints, and check output against this analytical backend.

## M2a: experimental FEMM 5 numerical adapter

A **limited FEMM 5 adapter** is now implemented for an empty, isolated
spherical conducting shell with prescribed net charge. It uses the same
`ElectrostaticProblem` definition as the analytical backend and an
axisymmetric FEMM 5 geometry compiler. **Real FEMM integration still needs to
be executed and accepted**; default GitHub CI checks the Python command
contract, not the numerical solver.

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
