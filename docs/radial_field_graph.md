# From a radial potential to a Cartesian field

Workbook `02_conductors_half_size.pdf`, PDF page 4, Single Correct Q38 gives V = 10r and asks for the field at the position vector (3, 4, −5). Its radius is √50. Away from the origin E = −∇V = −10 r⃗/r, so **E = −√2 (3, 4, −5)**, printed option 2. The z component is positive.

The field graph stores the signed *outward* radial component. Its potential reference is the potential at its first radius. This example takes metres and volts as SI units and chooses 8 m as the outer limit of the numerical example. The source law gives no upper cutoff.

```python
from conductor_sim.axial_field_graph import AxialFieldGraph
from conductor_sim.radial_field_graph import RadialFieldGraph

graph = AxialFieldGraph(((0, -10), (8, -10)), reference_potential_v=0)
field = RadialFieldGraph(graph)
volts = field.potential_at((3, 4, -5))
components = field.electric_field_at((3, 4, -5))
```

For Eᵣ = mr+b on a segment starting at a, V(r) = V(a) − m(r²−a²)/2 − b(r−a). In Cartesian coordinates E = (m+b/r)(x−x₀), where x−x₀ is the displacement vector from the common centre. The solver tests squared distances against squared knot radii and evaluates the radical before rounding each requested observation. It accepts the supplied field values at the endpoints. All probes must be inside the supplied radial interval.

For several fields sharing one centre and medium, use `field.superpose(other_radial_field)` to check the centres and combine their signed radial components and separate potential references on the common interval. A translated centre is a teacher choice. Combining radial functions about different centres would need a three-dimensional field assembly. If the interval includes r = 0, potential is defined there. A finite nonzero limiting radial component has no single vector direction at the centre. A component tending to zero gives the zero vector. This model prescribes a field; it does not infer a charge distribution or solve electrode boundary conditions.

| Dimension | Status for one original item | Evidence |
| --- | --- | --- |
| Interpretation fidelity | Pass, 1/1 | The signed gradient and the complete off-axis position vector are retained. |
| Physical correctness | Pass, 1/1 | Independent Cartesian differentiation gives all three signed components. |
| Simulation coverage | Supported, 1/1 | The requested probe lies within the example interval. |
| Educational usefulness | Not evaluated, 1/1 | The example allows signed profile, gauge and centre changes. |
| Textbook-answer completeness | Complete, 1/1 | The derived vector gives printed option 2. |
