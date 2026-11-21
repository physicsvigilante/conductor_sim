# What a spherical outer boundary determines

Workbook `02_conductors_half_size.pdf`, PDF page 3, Single Correct Q34 places q inside a cavity of a neutral conductor whose **outer** surface is spherical. It asks for the potential at an exterior point a distance r from the outer sphere's centre. The question gives no cavity shape or source offset.

Gauss's law fixes the cavity-wall total at −q and the outer-surface total at +q. The spherical boundary is equipotential. In a homogeneous, source-free unbounded exterior with zero potential at infinity, uniqueness gives V = kq/r, printed option 1. The same argument gives **E = kq r⃗/r³**. Gauss's law alone, without the spherical boundary condition, would only give the total flux.

```python
from conductor_sim.cavity_gauss import CavityCharge, GaussCavityAssembly, SealedCavity
from conductor_sim.model import FixedNetCharge
from conductor_sim.spherical_exterior import SphericalCavityExterior

# Teacher-selected SI values. The sealed topology declares source membership.
cavity = SealedCavity("void", (CavityCharge("q", 2e-9),))
model = GaussCavityAssembly("metal", (cavity,), FixedNetCharge(0))
outside = SphericalCavityExterior(model, radius_m=2)
volts = outside.potential_at((3, 0, 0))
field_v_m = outside.electric_field_at((3, 0, 0))
```

The outer-radius declaration adds one geometric fact to the existing ledger. Cavity shapes, positions and containment are author declarations. The ledger records binary64 SI values and refuses integer inputs that would lose precision on conversion. Charge sums are accumulated exactly before rounding each output. Exterior probes must lie strictly outside the sphere. The same construction supports several sealed cavities within one connected metal. A grounded shell has zero exterior field and potential in this source-free environment. Neighbouring electrodes, external charges and a nonspherical outer surface require another boundary solution.

## Adding charge to the metal

Page 4, Single Correct Q42 specifies a **centred** source q inside a neutral hollow conducting sphere and a point p in the cavity. Adding positive charge Q to the metal leaves the cavity field unchanged, printed option 1. The cavity-wall charge remains −q. The added charge goes to the outer surface, shifting the metal and cavity potential up by kQ/R, where R is the outer radius.

For this interior observation use the existing `ConcentricShellBackend`. The regression performs two complete solves at unchanged geometry. With teacher-selected inner and outer radii 1 and 2 m, source +2 nC, deposit +3 nC and p at x = 0.5 m, both fields equal k(2 nC)/(0.5 m)² along +x, while the potential increases by k(3 nC)/(2 m). The native exterior observations before and after also agree with the spherical exterior model. The topology-only model continues to refuse interior probes.

| Dimension | Status for two workbook items | Evidence |
| --- | --- | --- |
| Interpretation fidelity | Pass, 2/2 | Q34 preserves unspecified cavity geometry. Q42 keeps the stated central charge and interior probe. |
| Physical correctness | Pass, 2/2 | Exterior uniqueness and Gauss's law, plus the constant cavity-potential shift at unchanged field. |
| Simulation coverage | Supported, 2/2 | Exterior observation for Q34 and complete centred single-shell equilibria for Q42. |
| Educational usefulness | Not evaluated, 2/2 | The two observations distinguish field and potential changes. |
| Textbook-answer completeness | Complete, 2/2 | Both original questions give option 1. |
