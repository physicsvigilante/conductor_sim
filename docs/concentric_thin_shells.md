# Charges on ideal concentric conducting shells

Workbook `02_conductors_half_size.pdf`, page 3, Single Correct Q36 gives radii r, 2r and 3r. The inner and outer shells are earthed. The printed q₁, q₂ and q₃ are the net charges **after** earthing. The middle shell remains isolated.

At the grounded outer surface the potential is k(q₁+q₂+q₃)/(3r) = 0, taking zero at infinity. Thus **q₁ + q₃ = −q₂**, printed option 1. At the grounded inner surface the condition is q₁ + q₂/2 + q₃/3 = 0 after multiplying by r/k. Together they give q₁ = −q₂/4 and q₃ = −3q₂/4. The two conditions apply to the same equilibrium.

```python
from fractions import Fraction
from conductor_sim import ElectricalLink, FixedNetCharge
from conductor_sim.thin_shells import ConcentricThinShell, solve_thin_shell_charges

# Teacher-selected r = 1 m and q₂ = 4 nC.
nc = Fraction(1, 10**9)
shells = (ConcentricThinShell("inner", 1, FixedNetCharge(0)),
          ConcentricThinShell("middle", 2, FixedNetCharge(4*nc)),
          ConcentricThinShell("outer", 3, FixedNetCharge(0)))
links = (ElectricalLink(("inner",), "grounded"), ElectricalLink(("outer",), "grounded"))
q = solve_thin_shell_charges(shells, links)
assert q["inner"] == -nc and q["outer"] == -3*nc
```

The two initial zeros in this example are teacher-selected. Ground fixes the final potentials, so those initial charges do not enter the result. The solver also accepts existing grounded constraints and disjoint floating groups of initially isolated shells. Floating groups conserve their combined signed charge. Each named result is a `Fraction` in coulombs, exact for the supplied numbers. The charge equations use the same gap coefficients and wire constraints as the finite-thickness radial solver, with both faces now at the printed radius. A uniform dielectric factor cancels from the zero-voltage and equal-voltage equations.

The scope is 2–8 distinct spherical surfaces sharing a centre, an empty central cavity, no external sources and static equilibrium. For shells with a metal thickness, the two bounding face radii determine each gap. For instance, the finite-shell solver with outer radii 1, 2, 3 m, 0.01 m thicknesses and q₂ = 4 nC gives about −0.999 nC on the inner shell. Its split depends on the teacher-selected thickness.

| Dimension | Status for the one source item | Evidence |
| --- | --- | --- |
| Interpretation fidelity | Pass, 1/1 | Printed radii, final charge labels and simultaneous earth connections are preserved. |
| Physical correctness | Pass, 1/1 | Both surface-potential conditions independently give the signed charge split. |
| Simulation coverage | Supported, 1/1 | The named charges are computed for the ideal thin surfaces. |
| Educational usefulness | Not evaluated, 1/1 | Example allows independent changes to radii, charge and switches. |
| Textbook-answer completeness | Complete, 1/1 | Option 1 follows from grounding the outer surface. |
