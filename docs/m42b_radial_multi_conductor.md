# Concentric conductors and planar stacks

The stack API uses SI units and one `ConductorGroup` for each independent
electrical node. A singleton group specifies an individual conductor. A group
with multiple member IDs models an ideal wire at a common potential. Use
`Grounded()`, `FixedPotential(volts)`, or `FixedNetCharge(coulombs)` for a static
solve. A fixed charge belongs to the entire group, and the solver distributes
it among the members. Authored fixed charges use the decimal spelling of the SI
number, so `1e-9`, `2e-9`, and `-3e-9` form an exactly neutral planar stack.

## Concentric conductors

Author the conductors in radial order. An innermost inner radius of zero makes
a solid sphere. Equal positive inner and outer radii give an ideal thin shell
with two independently reported face totals. A single world-frame center and
uniform permittivity apply to all regions. The voltage reference is infinity.

```python
from dataclasses import replace

from conductor_sim import (
    ConcentricConductor, ConductorGroup, FixedNetCharge, FixedPotential,
    SphericalStackProblem, StackPointCharge, solve_spherical_stack,
)

problem = SphericalStackProblem(
    conductors=(
        ConcentricConductor("core", 0, 1),
        ConcentricConductor("middle", 3, 4),
        ConcentricConductor("outer", 6, 7),
    ),
    groups=(
        ConductorGroup(("core",), FixedNetCharge(1e-9)),
        ConductorGroup(("middle",), FixedNetCharge(2e-9)),
        ConductorGroup(("outer",), FixedNetCharge(-3e-9)),
    ),
    center_m=(13, -7, 4),
    sources=(
        StackPointCharge("annular", 7e-9, (13, -5, 4)),
        StackPointCharge("outside", -4e-9, (13, -7, 18)),
    ),
)
initial = solve_spherical_stack(problem)
connected = solve_spherical_stack(replace(problem, groups=(
    ConductorGroup(("middle",), FixedPotential(-5)),
    ConductorGroup(("core", "outer"), FixedNetCharge(-2e-9)),
)))
gap_field = connected.electric_field_at((18, -7, 4))
```

`conductor_results` follows geometry order and gives each voltage, free net
charge, and inner and outer face totals. `source_regions` maps the authored
source IDs to their regions. The cavity and exterior use Kelvin solutions.
In a source-free annulus the field is radial. In a charged annulus the static
solution includes the two complete face totals from Green's reciprocity.
Local potential and field queries there raise `UnsupportedProblem`. Active
charge sites and bare wall field queries raise `SingularField`.

## Planar capacitors and switches

The planar API describes ideal plates normal to the x axis and reports the
signed x component of the field. Ordered x positions specify the gap widths.
The common area converts uniform face densities to charge totals. The model
requires explicit zero exterior displacement on both sides. An all-floating
stack requires neutral charge and a `PlanarVoltageReference` on a named plate.
Grounded or fixed-potential terminals supply the reference when present.

```python
from conductor_sim import (
    ConductorGroup, FixedNetCharge, FixedPotential, Floating, Grounded,
    PlanarStackProblem, solve_planar_stack, switch_conductor_stack,
)

initial = solve_planar_stack(PlanarStackProblem(
    plate_ids=("left", "middle", "right"),
    plate_positions_m=(0, 0.002, 0.005),
    area_m2=0.1,
    groups=(
        ConductorGroup(("left",), Grounded()),
        ConductorGroup(("middle",), FixedNetCharge(0)),
        ConductorGroup(("right",), Grounded()),
    ),
    d_left_c_per_m2=0,
    d_right_c_per_m2=0,
))
switch = switch_conductor_stack(initial, (
    ConductorGroup(("left",), FixedPotential(12)),
    ConductorGroup(("middle",), Floating()),
    ConductorGroup(("right",), Grounded()),
))
densities = switch.after.plate_results[1]
fields = switch.after.gap_fields_v_per_m
```

`switch_conductor_stack` also accepts source-free spherical solutions. A
final `Floating()` group inherits the sum of its members' initial charges.
Pass the final solution to the next switch to chain steps. A final
`FixedNetCharge` must match the conserved group charge. A charge-constrained
planar final state requires its voltage reference as a switch argument.

The ledger reports charge gained by each conductor and the signed charge
received by each final supply or ground terminal. For a planar stack those
terminal transfers sum to zero. Supply work sums each terminal voltage times
the charge gained by its group. The energy residual is supply work plus
external work minus the field energy change. Fixed geometry gives zero
external work. The switch models static endpoints, so the residual carries
no claim about heating or switching dynamics. Active point charges make the
total field energy undefined and preclude this ledger.

## Coupling and scope

`capacitance_matrix_f` returns the symmetric Maxwell matrix in geometry order
before electrical groups are applied. Each gap contributes the inverse of its
elastance to the two diagonal entries and its negative to the two mutual
entries. The spherical exterior adds a diagonal term from the outermost
conductor to infinity. A held source adds a grounded face-charge offset. The
capacitance matrix gives the voltage-dependent part of free conductor charge.

The homogeneous spherical gap elastance is
`(1/a - 1/b) / (4*pi*epsilon)` for facing radii `a < b`. The planar value is
`d / (epsilon*A)`. For charge-free layers of linear dielectric, a geometry
adapter can sum the elastances of the layers spanning the gap and reuse the
same constraint and energy equations. Local field evaluation and source
partitions in a layered gap require the corresponding dielectric solution.

The planar model covers the fringe-free one-dimensional field. Nonzero
exterior displacement, arbitrary circuit netlists, and other geometries fall
outside the admitted stack inputs.
