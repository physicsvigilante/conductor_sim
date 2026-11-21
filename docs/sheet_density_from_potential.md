# Inferring sheet charge from a potential graph

Workbook `02_conductors_half_size.pdf`, PDF page 5, Single Correct Q49 shows V(x) for two similarly charged infinite sheets. The figure labels the **left** sheet σ₂ and the **right** sheet σ₁. Both corners turn the potential slope downward as x increases. Since Eₓ = −dV/dx and σ = ε(Eₓ,right − Eₓ,left), both sheets have positive charge density.

In the gap the potential decreases toward the right, so Eₓ is positive. For the isolated two-sheet system the gap field is (σ₂ − σ₁)/(2ε). Hence **σ₂ > σ₁**, printed option 1. A separately applied uniform field would also contribute to the gap slope. With such a bias, use the two individual slope jumps to infer the densities.

```python
from conductor_sim.parallel_sheets import ParallelSheetField

# Teacher-selected knots in m and V, and permittivity 2 F/m for simple arithmetic.
# These numbers illustrate the printed slope ordering.
model = ParallelSheetField.from_potential_segments(
    ((-2, 0), (-1, 3), (1, 1), (2, -2)), permittivity_f_m=2)
assert model.sheets == ((-1, 8), (1, 4))  # C/m², left then right
assert model.bias_v_m == 0
assert model.gap_graph(-0.5, 0.5).axial_field_v_m(0) == 1
```

Author a continuous, piecewise-linear potential by giving successive (x,V) knots in one homogeneous medium. For a slope change from L to R (V/m), the inferred charge density is ε(L−R). A collinear knot introduces no sheet. The fitted uniform bias is the mean of the first and last segment fields. The returned `ParallelSheetField` uses the first knot as its potential reference and can compute work or yield a gap graph for superposition.

This construction assumes the outermost supplied segments continue linearly beyond the interval. The inferred interior slope jumps follow from the graph itself. Measurements on a finite interval cannot uniquely determine remote sheets or the exterior continuation. Isolated sample values also cannot establish that the interpolating potential really is piecewise-linear.

| Evaluation dimension | Status for one original item | Evidence |
| --- | --- | --- |
| Interpretation fidelity | Pass, 1/1 | Both sheet labels and the sign of the gap slope follow the printed graph. |
| Physical correctness | Pass, 1/1 | The field discontinuities follow the planar Gauss boundary condition. |
| Simulation coverage | Supported within the prescribed uniform-sheet model, 1/1 | The native inverse and forward calculations reproduce a teacher-selected graph with the source ordering. |
| Educational usefulness | Not evaluated, 1/1 | The example can vary slope, gauge and uniform bias. |
| Textbook-answer completeness | Complete, 1/1 | σ₂ > σ₁ gives option 1. |
