# Voltage of an isolated conducting sphere near charged arcs

Workbook `02_conductors_half_size.pdf`, PDF page 5 (printed 3.37), Single Correct Q51 gives a neutral conducting sphere and a nonuniformly charged ring. Their centres are 3 m apart on the ring axis, the radius given is 4 m and the total ring charge is 1 μC. Every element of the ring is 5 m from the sphere centre.

In electrostatic equilibrium the solid sphere has one potential throughout its interior. Evaluate it at the centre. Each element of the induced surface charge is one sphere radius R away, so its combined contribution is kQ_sphere/R regardless of how that charge redistributes. Each exterior arc i contributes kQ_i/d_i at that same point. Here k = 1/(4πε), d_i is the distance from the sphere centre to any point on arc i and the potential reference is infinity. For Q51 the sphere term vanishes and the voltage is approximately **1.8 kV, option (3)**.

```pycon
>>> from conductor_sim import FixedNetCharge, SolidSphere
>>> from conductor_sim.coaxial_arcs import CircularArcCharge, CoaxialArcAssembly
>>> ring = CoaxialArcAssembly((CircularArcCharge(1e-6, 4, 0, 360, center_z_m=3),))
>>> round(ring.isolated_sphere_voltage_v(SolidSphere(4, FixedNetCharge(0))))
1798

```

The observable accepts one solid conducting sphere with fixed net charge and its centre on the common axis of prescribed arcs, all strictly exterior, in one homogeneous medium. A teacher may split the ring into unequal arcs with different signed totals, add another ring at a different distance, or change the sphere’s charge. Angular charge density on each arc may vary freely. The data determine the conductor voltage. Calculating its induced surface density or the full exterior field would require additional angular information.

For this one source item the model preserves the printed geometry and charge (1/1), reproduces the independently derived equilibrium voltage (1/1), supplies the requested scalar observation (1/1), and gives the textbook choice (1/1). Educational usefulness has not been measured.

For a three-dimensional point-source comparison, [workbook Q63](cube_sphere_voltage.md) uses the multisite Kelvin solver to compare local polarization at fixed sphere voltage.
