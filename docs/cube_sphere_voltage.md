# A neutral sphere inside a cube of point charges

Workbook `02_conductors_half_size.pdf`, PDF page 6 (printed 3.38), Single Correct Q63 fixes one charge q at each vertex of a cube of side a. A neutral solid conducting sphere of radius a/3 shares the cube’s centre. The question asks for the sphere’s potential relative to infinity.

The equilibrium potential is constant throughout the metal, so evaluate it at the centre. Every induced surface charge element lies one sphere radius away. Their potential contributions sum to zero because their net charge is zero. Each cube vertex is at distance √3 a/2, giving **V = 16kq/(√3 a), option (2)**. Neutrality fixes the net induced charge, while the local surface density can vary.

## Run the eight-source equilibrium

The example chooses a = 0.30 m and q = 1 nC. All eight point charges are held fixed in vacuum, and the conducting sphere responds to them together. The example probes two opposite surface points along a body diagonal. The original cube symmetry requires their densities to agree.

```pycon
>>> from dataclasses import replace
>>> from itertools import product
>>> from math import isclose, pi, sqrt
>>> from conductor_sim import ElectrostaticProblem, ExternalMultiSphereBackend, FixedNetCharge, PointCharge, SolidSphere
>>> a, q = 0.30, 1e-9
>>> sphere = SolidSphere(a/3, FixedNetCharge(0))
>>> corners = tuple(product((-a/2, a/2), repeat=3))
>>> problem = ElectrostaticProblem(conductors=(sphere,), charges=tuple(PointCharge(q, p) for p in corners))
>>> solver = ExternalMultiSphereBackend()
>>> original = solver.solve(problem)
>>> k = 1/(4*pi*problem.permittivity_f_per_m)
>>> isclose(original.conductor_potential_v, 16*k*q/(sqrt(3)*a))
True
>>> near = (sphere.radius_m/sqrt(3),)*3
>>> far = tuple(-x for x in near)
>>> def density(solution, point):
...     return solution.surface_charge_density_at("outer", point)
>>> isclose(density(original, near), density(original, far), rel_tol=1e-10)
True

```

## Change the corner charges

As a teacher revision, transfer half of one corner’s charge to the diagonally opposite corner, keeping all positions and the sphere fixed. Every source still lies the same distance from the centre, so the sphere voltage stays the same. The induced density becomes more negative at the surface point facing the strengthened positive source and more positive at the opposite probe, relative to the original readings.

```pycon
>>> charges = list(problem.charges)
>>> charges[0] = replace(charges[0], charge_c=q/2)
>>> charges[-1] = replace(charges[-1], charge_c=3*q/2)
>>> revised = solver.solve(replace(problem, charges=tuple(charges)))
>>> isclose(revised.conductor_potential_v, original.conductor_potential_v)
True
>>> density(revised, near) < density(original, near) and density(revised, far) > density(original, far)
True

```

For the one original item, the model preserves all eight sources and the conductor constraint (1/1), agrees with the independent centre-potential sum (1/1), computes the requested static voltage (1/1), and recovers the printed choice (1/1). Educational usefulness has not been measured. The two surface samples compare static equilibria and do not describe a charging trajectory.
