# conductor_sim

Composable electrostatics for JEE conductor simulations, with Python-authored physical components separate from solvers and visualization.

**Current milestone: M1, analytical baseline only.** This code does not yet integrate FEMM 5, generate meshes, solve off-center charges, or render animations.

## Quick start

```bash
python -m pip install -e ".[test]"
python -m pytest
python examples/concentric_shell.py
```

## Author a problem

```python
from conductor_sim import (
    ConcentricShellBackend, ElectrostaticProblem,
    FixedNetCharge, PointCharge, SphericalShell,
)

problem = ElectrostaticProblem(
    charges=(PointCharge(1e-9),),
    conductors=(SphericalShell(0.1, 0.12, FixedNetCharge(0.0)),),
)
solution = ConcentricShellBackend().solve(problem)
print(solution.conductor_results)
print(solution.potential_at((0.05, 0.0, 0.0)))
```

Change `FixedNetCharge(0.0)` to `Grounded()` to ground the shell. All units are SI.

The reference solver supports a *single* spherical shell with centered charge(s); it explicitly rejects off-center and general multi-conductor arrangements. Its field and potential interfaces will be shared with future FEMM 5 and other numerical backends.

See [architecture](docs/architecture.md) and [roadmap](docs/roadmap.md). A FEMM 5 integration must be validated against actual installed FEMM 5 bindings and real solver runs; `pyfemm` assumptions from older versions are not sufficient.
