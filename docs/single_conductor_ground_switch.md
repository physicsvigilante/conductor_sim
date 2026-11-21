# Grounding the inner conductor of an isolated pair

Workbook page 14, Numerical Value Type Q2 shows two concentric thin conducting shells of radii r and 3r. The outer shell carries q. The inner shell starts neutral and a switch connects it to earth. The question asks for n when the charge flowing from the inner shell to earth is q/n.

Let x be the final inner-shell charge. For ideal thin shells the potential on the inner shell is k(x/r + q/(3r)). Grounding sets this to zero, so x = −q/3. The charge delivered to earth is the opposite change, +q/3, and **n = 3**. Reversing the sign of q reverses the signed transfer. The result compares static equilibria and gives no current waveform.

## Reproduce the thin-shell limit

The native radial backend models conductors with finite thickness. This example uses teacher-selected outer radii 0.10 m and 0.30 m, and q = +3 nC. It reduces the prescribed thickness while keeping both outer radii fixed.

```python
from conductor_sim import (
    ConnectedShellBackend, ConnectionPlan, ElectricalLink, ElectrostaticProblem,
    FixedNetCharge, SphericalShell,
)

q = 3e-9
switch = ElectricalLink(("inner",), mode="grounded")
for width in (1e-3, 1e-5, 1e-7):
    shells = (
        SphericalShell(.1-width, .1, FixedNetCharge(0), name="inner"),
        SphericalShell(.3-width, .3, FixedNetCharge(q), name="outer"),
    )
    plan = ConnectionPlan(ElectrostaticProblem(conductors=shells), (switch,))
    run = ConnectedShellBackend().solve(plan)
    print(width, q / run.charge_to_ground_c(switch))
```

For finite thickness, let a be the inner conductor’s outer radius, b the outer conductor’s inner radius and c its outer radius. Gauss’s law gives enclosed charge x in the gap and x+q outside. Integrating inward from infinity gives zero inner potential when (x+q)/c + x(1/a−1/b) = 0. The finite-thickness ratio is 1+c(1/a−1/b), which tends to 3. All shell thicknesses in the example are teacher choices.

The same switch can share a plan with a disjoint floating wire group. Each group retains its own boundary condition. The transfer observation requires an initially isolated group so the new wire defines an unambiguous earth path.
