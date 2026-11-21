# Choosing the zero of potential

Workbook `02_conductors_half_size.pdf`, PDF page 5 (printed 3.37), Single Correct Q54 and Q56 use the same sign convention: V(B) − V(A) is the negative line integral of the prescribed electric field from A to B. For an affine field the mean field along a straight segment is exactly the midpoint field. Because the gradient matrix is symmetric, any path within the prescribed domain gives the same difference.

**Q54.** The uniform field is (4, 5, 0) V/m and the potential at the origin is V₀. The result is **V(x,y) = V₀ − 4x − 5y, option (1)**. The following numerical probe uses teacher-selected coordinates and V₀.

```pycon
>>> from conductor_sim import UniformElectricField
>>> from conductor_sim.affine_fields import AffineFieldAssembly, ConservativeAffineField
>>> uniform = UniformElectricField((4, 5, 0), reference_potential_v=12)
>>> model = AffineFieldAssembly((uniform,))
>>> model.potential_at((2, -1, 0))
9.0
>>> model.potential_difference_v((0, 0, 0), (2, -1, 0))
-3.0

```

**Q56.** The field is (10x, −20, 0) in SI units. The reference A is (30, −20) m and the question asks for V at B = (−20, 30) m when V(A) = 0. Integrating along x gives −5(x_B² − x_A²) = +2500 V. Integrating along y gives +20(y_B − y_A) = +1000 V. The sum is **+3500 V, option (4)**.

```pycon
>>> varying_x = ConservativeAffineField(((10, 0, 0), (0, 0, 0), (0, 0, 0)))
>>> constant_y = UniformElectricField((0, -20, 0))
>>> field = AffineFieldAssembly((varying_x, constant_y))
>>> A, B = (30, -20, 0), (-20, 30, 0)
>>> field.potential_difference_v(A, B)
3500.0
>>> field.potential_difference_v(B, A)
-3500.0

```

The assembly accepts independently referenced source potentials. The endpoint difference depends on their fields and can be reported even when a large common voltage offset makes the two displayed endpoint potentials round to the same number.

Across these two items, the models retain the printed fields and references (2/2), match independent line integrals (2/2), support the requested static voltage observations (2/2), and recover the two textbook choices (2/2). Educational usefulness has not been measured.
