# External work when point charges change configuration

Workbook page 14, Numerical Value Type Q5 gives three point charges −2q, −2q and +q at the vertices of an equilateral triangle of side a. It asks for the work by an external force as the separation increases to 2a. Take the kinetic energy to be the same at both endpoints, as in a quasistatic rearrangement.

Each distinct pair contributes k qᵢqⱼ/rᵢⱼ to the interaction energy. At side a, the three contributions have the common factor kq²/a and coefficients +4, −2 and −2. They sum to zero. Doubling every separation halves each contribution, so the final sum is also zero. **The total external work is 0 J.** Each pair contributes a nonzero change.

```python
from conductor_sim import ElectrostaticProblem, PointCharge, rearrange_point_charges

# Teacher-selected SI values. These vertices form an equilateral triangle of side √2 m.
points = ((1, 1, 0), (1, 0, 1), (0, 1, 1))
charges = (-2e-9, -2e-9, 1e-9)
initial = ElectrostaticProblem(tuple(PointCharge(q, p) for q, p in zip(charges, points)))
positions = tuple(tuple(2*x for x in point) for point in points)
expanded = rearrange_point_charges(initial, positions)
print(expanded.external_work_j)  # 0.0
```

Supply one final position per source in its original order. Passing the returned `final` problem into another rearrangement preserves the charges and medium. Replacing +q with +2q is a teacher variation that gives positive external work during the same expansion.

The model uses a uniform permittivity, isolated point charges, and electrostatic pair interaction energies. It excludes divergent point self energies and rejects coincident nonzero charges at either endpoint. The endpoints determine net work at unchanged kinetic energy. They do not determine a path, duration or force history. The signed result is rounded after summing pair changes. An unresolvable result raises an explicit numerical error.
