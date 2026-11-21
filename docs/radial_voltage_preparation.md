# Preparing concentric conductors at specified potentials

Workbook page 3, Single Correct Q32 gives two concentric conducting shells with radii a₁ < a₂ and potentials V₁ and V₂. It asks for the charge on the inner shell. Assume vacuum and zero potential at infinity. There is no charge in the central cavity.

Gauss's law in the gap gives V₁ − V₂ = Q₁(1/a₁ − 1/a₂)/(4πε₀). Hence Q₁ = 4πε₀a₁a₂(V₁−V₂)/(a₂−a₁), printed option (1). The common potential offset affects exterior charge even though the inner charge depends only on the potential difference.

## Prepare the shells and close an earth switch

The workbook gives symbolic radii and voltages. In this teacher example the gap runs from 0.10 m to 0.30 m, the inner shell is held at +10 V and the outer shell at +3 V. The finite shell widths are teacher choices. The inner charge depends on the two faces that bound the gap.

```python
from conductor_sim import (
    ConnectionPlan, ConnectedShellBackend, ElectricalLink, ElectrostaticProblem,
    Grounded, SphericalShell, prepare_radial_voltages,
)

geometry = ElectrostaticProblem(conductors=(
    SphericalShell(.09, .10, Grounded(), name="inner"),
    SphericalShell(.30, .32, Grounded(), name="outer"),
))
isolated = prepare_radial_voltages(geometry, {"inner": 10., "outer": 3.})
print(isolated.conductors[0].constraint.charge_c)  # coulombs
earth = ElectricalLink(("inner",), mode="grounded")
connected = ConnectedShellBackend().solve(ConnectionPlan(isolated, (earth,)))
print(connected.charge_to_ground_c(earth))  # positive for delivery to earth
```

Preparation supersedes the input conductor constraints. The returned problem preserves conductor names and ordering, dimensions, medium and cavity sources. Its fixed net charges describe the equilibrium immediately after disconnecting the voltage supplies. A grounding or wiring operation can then change that equilibrium.

For a shell enclosing a point charge, Gauss's law fixes the charge on the inner face; the prescribed voltage differences fix the flux in each gap. A filled innermost sphere uses the existing solid-sphere backend. The relevant radial backend re-solves the isolated configuration to check its potentials, gap voltages and enclosed charges. The operation also checks the charge inventory using the individual cavity sources and returned conductor charges. A nonzero target potential, gap voltage or enclosed charge must agree to relative tolerance 10⁻¹¹. A zero conductor-potential target allows at most 10⁻¹² V absolute error; zero enclosed charge and zero gap voltage require exact zero. Preparations outside these bounds are refused. It supplies static endpoint states only.
