# Equal potential drops in a quadratic field

Workbook `02_conductors_half_size.pdf`, PDF page 5 (printed 3.37), Single Correct Q55 shows three successive equipotential planes on the positive x axis in the field Eₓ = 2x², with x in metres and Eₓ in V/m. The field points right and grows with x. The two potential drops are equal. The positive area under the Eₓ graph gives each drop. A second interval as wide as the first would have strictly more area, so the second must be shorter: **x₁ > x₂, option (2)**. Here x₁ and x₂ denote the spacings shown in the diagram.

## Compose an authored field profile

Supply the endpoint positions and fields, and one coefficient of x² (in V/m³) per segment. The coefficient and endpoint values determine the polynomial throughout each segment. Omitting the coefficients keeps the linear interpolation of workbook Q40. The model reports fields, potentials relative to an authored reference, and signed field work within the specified domain.

These plane positions are a teacher example. The first two lie at x = 1 m and 2 m. Integrating 2x² gives the next position at the cube root of 15 metres when x is expressed numerically in SI units.

```pycon
>>> from math import isclose
>>> from conductor_sim.axial_field_graph import AxialFieldGraph
>>> field = AxialFieldGraph(((1, 2), (3, 18)), quadratic_v_m3=(2,))
>>> third = 15**(1/3)
>>> isclose(field.potential_v(1) - field.potential_v(2), 14/3)
True
>>> isclose(field.potential_v(2) - field.potential_v(third), 14/3) and third-2 < 1
True

```

A teacher may add a constant or piecewise-linear field with `superpose`, or combine two quadratic profiles with different breakpoints. The result keeps the summed field and potential references on the common interval. A radial interpretation requires a separately authored spherically symmetric field. The three planes in Q55 use the Cartesian model.

For the single workbook item the positive-axis geometry is retained (1/1), the area comparison establishes the physical result (1/1), the native model supplies both signed potential drops for the example (1/1), and the printed answer follows (1/1). Educational usefulness has not been measured.
