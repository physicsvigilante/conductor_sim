# Work inside and outside a uniformly charged sphere

Workbook `02_conductors_half_size.pdf`, PDF page 5 (printed page 3.37), Single Correct Q50 places A at the centre, B on the surface at radius R, and C at radius 2R on the same ray. Charge Q is uniform throughout the volume. The question asks for the ratio of electric work on probe q on the two successive legs.

Gauss’s law gives radial fields kQr/R³ inside and kQ/r² outside, where k = 1/(4πε). Integrating qE from 0 to R gives kQq/(2R). Integrating from R to 2R gives the same signed work. For nonzero Q and q the ratio is **1, option (1)**. Reversing the sign of q reverses both works.

## Run the two observations

The example uses teacher-selected SI values Q = 2 nC, q = −3 nC and R = 0.5 m in vacuum. The charge signs exercise the signed result that a ratio alone would hide. Each call to the public endpoint observation uses the same physical source and probe. The checks compare its potential-based result with the independent field integrals above.

```pycon
>>> from math import isclose, pi
>>> from conductor_sim.jee_potential_explorer import endpoint_field_work
>>> from conductor_sim.prescribed_fields import PrescribedField, UniformSphereField, UniformShellField
>>> Q, q, R = 2e-9, -3e-9, 0.5
>>> A, B, C = (0, 0, 0), (R, 0, 0), (2*R, 0, 0)
>>> field = PrescribedField((UniformSphereField(Q, R),))
>>> def work(model, start, end):
...     return endpoint_field_work(model, source_annotations=(("body", "Charged body"),),
...                                probe_charge_c=q, start_m=start, end_m=end)["field_work_j"]
>>> expected = Q*q/(8*pi*field.permittivity_f_m*R)
>>> ab, bc = work(field, A, B), work(field, B, C)
>>> ab < 0 and bc < 0 and all(isclose(w, expected) for w in (ab, bc))
True
>>> isclose(ab/bc, 1) and isclose(ab + bc, work(field, A, C))
True

```

## Change the charge distribution

As a teacher revision, move all Q onto a concentric, uniformly charged thin shell of the same radius. Keep the probe, endpoints and source identity. The interior field now vanishes. The exterior field is identical to the original sphere’s, so the ratio becomes zero. Potential is continuous at B and the work remains defined there even though the ideal shell’s pointwise surface field jumps.

```pycon
>>> shell = PrescribedField((UniformShellField(Q, R),))
>>> work(shell, A, B) == 0 and isclose(work(shell, B, C), expected)
True

```

For the one source item, the model reproduces the printed geometry (1/1), the signed field integrals (1/1), the requested static work observations (1/1), and the textbook answer (1/1). Educational usefulness has not been measured. These observations assume a fixed charge distribution and a test probe. They supply no transit times or mechanical trajectories.
