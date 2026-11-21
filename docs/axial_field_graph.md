# Potential from an axial electric-field graph

Workbook `02_conductors_half_size.pdf`, PDF page 4, Single Correct Q40 specifies Eᵧ = E𝓏 = 0, V(0) = 10 V and a graph of Eₓ versus x. Between 0 and 2 m the line falls from 0 to −20 N/C. The signed area is −20 V, so V(2 m) = 10 V − (−20 V) = **30 V**, option 4.

```python
from conductor_sim.axial_field_graph import AxialFieldGraph

# Read the original graph through its last numbered tick at x = 5 m.
graph = AxialFieldGraph(((0, 0), (2, -20), (4, 20), (5, 20)), reference_potential_v=10)
assert graph.potential_v(2) == 30
assert graph.axial_field_v_m(3) == 0
assert graph.potential_v(3) == 40

# Teacher revision: overlay a +2 V/m field on [1, 4] m, with its own V(1) = 0 V.
from conductor_sim.affine_fields import AffineFieldAssembly
from conductor_sim.prescribed_fields import UniformElectricField
source = UniformElectricField((2, 0, 0), reference_point_m=(1, 0, 0))
bias = AxialFieldGraph.from_affine(AffineFieldAssembly((source,)), 1, 4)
combined = graph.superpose(bias)
assert combined.potential_v(2) == 28
```

An `AxialFieldGraph` is a prescribed continuous field component along one Cartesian x axis. Each pair gives position in metres and signed field in V/m (equivalently N/C). The first knot sets the location of the potential reference. The engine integrates each linear segment and includes all intermediate knots. At the zero crossing in Q40 the potential has a maximum. The zero crossing alone cannot determine its value.

Superposition assumes both graphs use the same oriented axis, coordinate origin and medium. It preserves their separate potential references, uses every slope change within their common interval and refuses extrapolation. The affine adapter restricts an existing three-dimensional assembly to (x, 0, 0) and carries over its potential reference. Transverse field components do no work along this path. A field discontinuity requires a different model. `field_work_j(q, start, end)` gives signed work by the field, q∫Eₓ dx, even if a large arbitrary potential reference makes the two displayed potentials round to the same value. Internal arithmetic uses exact rational values for the supplied numbers.

| Evaluation dimension | Status for one workbook item | Evidence |
| --- | --- | --- |
| Interpretation fidelity | Pass, 1/1 | Original signed field, distance axis and reference potential are retained. |
| Physical correctness | Pass, 1/1 | Independent triangular area gives the potential change. Reversing endpoints reverses field work. |
| Simulation coverage | Supported, 1/1 | The original observation lies inside the authored graph. |
| Educational usefulness | Not evaluated, 1/1 | The example permits a change of field and reference. |
| Textbook-answer completeness | Complete, 1/1 | 30 V is printed option 4. |
