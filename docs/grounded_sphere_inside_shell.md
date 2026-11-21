# Ground a solid sphere inside a charged conducting shell

Workbook `02_conductors_half_size.pdf`, PDF page 5 (printed 3.37), Single Correct Q53 puts an earthed solid sphere of radius R inside a concentric, isolated thick conducting shell. Its inner and outer radii are 2R and 3R. The shell carries +10Q. Find the charge on the inner sphere.

Let s be the sphere charge. A Gaussian surface within the shell metal encloses zero net charge, so the inner shell face carries −s. Charge conservation on the isolated shell then puts 10Q+s on its outer face. With zero potential at infinity, earthing the inner sphere imposes

`0 = k[s/R − s/(2R) + (10Q+s)/(3R)]`.

This gives **s = −4Q, option (1)**. The shell faces carry +4Q and +6Q. Although the sphere is at zero potential, it exchanges charge with earth. The ideal ground connection sets the electrical boundary condition in the spherically symmetric model.

## Solve the combined boundary conditions

Use teacher-selected SI scales R = 0.1 m and Q = 1 nC. The public mixed radial solver handles both conductors together. In the second solve, increase only the shell’s outer radius to 4R. Keeping the inner face at 2R and the shell total fixed changes the ground-potential equation and gives s = −10Q/3.

```pycon
>>> from dataclasses import replace
>>> from math import isclose
>>> from conductor_sim import ElectrostaticProblem, FixedNetCharge, Grounded, MixedRadialBackend, SolidSphere, SphericalShell
>>> R, Q = 0.1, 1e-9
>>> sphere = SolidSphere(R, Grounded(), name="inner sphere")
>>> shell = SphericalShell(2*R, 3*R, FixedNetCharge(10*Q), name="outer shell")
>>> def charges_in_Q(outer):
...     problem = ElectrostaticProblem(conductors=(sphere, outer))
...     solution = MixedRadialBackend().solve(problem)
...     solid = solution.conductor_by_name(sphere.name)
...     faces = solution.conductor_by_name(outer.name)
...     assert abs(solid.potential_v) < 1e-10
...     assert isclose(faces.net_charge_c, 10*Q)
...     return tuple(charge/Q for charge in (solid.net_charge_c,
...                  faces.inner_surface_charge_c, faces.outer_surface_charge_c))
>>> original = charges_in_Q(shell)
>>> all(isclose(a, b) for a, b in zip(original, (-4, 4, 6)))
True
>>> revised = charges_in_Q(replace(shell, outer_radius_m=4*R))
>>> all(isclose(a, b) for a, b in zip(revised, (-10/3, 10/3, 20/3)))
True

```

For this one item the model retains the source geometry and separate conductor constraints (1/1), satisfies the ground potential and face charge equations (1/1), simulates the requested equilibrium charge (1/1), and recovers the printed choice (1/1). Educational usefulness has not been measured. The revised shell is a separate static equilibrium; the example supplies no charging time or wire current.
