# Potential of a position-dependent field

In `02_conductors_half_size.pdf`, page 3, Single Correct Q28 gives **E = a(y, x, 0)** and asks for its potential. Integrating ∂V/∂x = −ay gives V = −axy + f(y,z). The remaining components require both derivatives of f to vanish. The printed answer is option 2, **V = −axy + C**. The question leaves the potential reference free.

```python
from conductor_sim.affine_fields import AffineFieldAssembly, ConservativeAffineField

# Teacher-selected SI example: a = 3 V/m² and V(origin) = 7 V.
field = ConservativeAffineField(((0, 3, 0), (3, 0, 0), (0, 0, 0)),
                               reference_potential_v=7)
field.potential_at((2, -4, 10))  # 31 V
assembly = AffineFieldAssembly((field,))
assembly.field_work_j(-2, (0, 0, 0), (2, -4, 10))  # +48 J
```

For a displacement d from the reference point, E = E₀ + Gd and V = V₀ − E₀·d − d·Gd/2. Symmetry of G is the zero-curl condition. The trace can be nonzero, so a prescribed affine field can model a region containing volume charge. The caller is responsible for restricting points and paths to a region where the prescribed law applies.

Combine multiple affine fields and existing `UniformElectricField` objects using `AffineFieldAssembly`. Each term keeps its own potential reference. Signed work by the field on a fixed test charge is q times the line integral of E. For affine E the midpoint value integrates exactly along the segment. The implementation evaluates the supplied numbers as exact rationals internally and rounds each requested observation to a float. Overflow and nonzero results that round to zero raise a qualification error.

This qualifies static scalar potential, vector field and endpoint work. Electrode boundary conditions, particle motion and browser rendering are outside this milestone. The original symbolic answer and one teacher-selected numerical example cover one workbook item. The evaluation denominator is one original workbook item.

| Dimension | Status | Evidence |
| --- | --- | --- |
| Interpretation fidelity | Pass, 1/1 | Both coordinate-dependent components and the free potential reference are preserved. |
| Physical correctness | Pass, 1/1 | Independent integration of each gradient component and a signed numerical check. |
| Simulation coverage | Supported, 1/1 | Static potential throughout the prescribed domain and both nonzero field components. |
| Educational usefulness | Not evaluated, 1/1 | The example allows reference and coefficient changes. Student outcomes were not measured. |
| Textbook-answer completeness | Complete, 1/1 | Printed option 2 includes the additive constant. |
