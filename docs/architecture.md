# Architecture

LLM-authored Python -> immutable ElectrostaticProblem -> solver backend -> FieldSolution -> procedural viewer.

- Models are in SI units and 3D coordinates, independent of FEMM (r,z).
- Grounded and fixed-net-charge refer to entire connected conductors.
- Solver declares support and refuses unsupported geometry.
- M1 ships exact concentric spherical shell reference physics only.
- FEMM 5 numerical backend, viewer, off-center charges and nested shells are **not yet implemented**.

For a shell of inner radius a and outer radius b, central charge q, net conductor charge Q:
inner induced charge = -q; outer charge = Q+q; shell potential = (Q+q)/(4*pi*epsilon*b).
For grounded shell, Q = -q.
Within the cavity, V(r)=V_shell + q/(4*pi*epsilon)*(1/r-1/a).
In the metal, V=V_shell. Outside, V=(Q+q)/(4*pi*epsilon*r).

Beware that an axisymmetric FEMM point away from the axis represents a ring of revolution, **not** a point charge in 3D. Numerical source regularization and validation are required before off-center cases.

At a charged surface, E has a jump; surface samples are deliberately rejected.
