# Successive grounding of distant conducting spheres

Workbook `02_conductors_half_size.pdf`, page 3, Single Correct Q31 places three initially neutral spheres A, B and C of radius a at the corners of an equilateral triangle of side d, with d ≫ a. A fourth identical sphere carrying q touches A and is removed far away. B is grounded and disconnected. C is then grounded. The observable is the charge on C.

Under the textbook equal-sharing approximation the contact prepares A with q/2. The engine starts from this stated post-contact charge. It does not solve the touching pair in the presence of B and C. For each subsequent switch, approximate the potential of each other sphere by that of its net charge at its centre. With only sphere i connected to earth, the zero-potential condition becomes

    Q_i = −R_i Σ (Q_j/d_ij),  j ≠ i.

Consequently B retains −qa/(2d) after disconnection. When C is grounded its charge is −(a/d)(q/2 − qa/(2d)), giving **−qa(d−a)/(2d²), option 3**. B contributes to the potential at C even though its switch is now open. Opening C's switch after it settles preserves the requested instantaneous charge.

Use `ground_and_disconnect_distant_sphere(plan, approximation="centre-monopole")` from `conductor_sim.distant_spheres`. Create a `ConnectionPlan` with one `ElectricalLink(("B",), "grounded")`, then use the returned immutable `ElectrostaticProblem` as the initial state of the next plan. Untouched conductors retain their identities and charge constraints. Each stored charge is rounded to binary64. A table of repeated stages therefore represents operations on the stored charge state.

The solver qualifies 2–8 isolated spheres or empty spherical shells in one homogeneous dielectric with no independent point charges. Each row must satisfy Σ R_i/d_ij ≤ 0.05. This is a geometric admission threshold without a relative-error guarantee. The dielectric factor cancels at zero prescribed potential. Surface polarization, variation of potential across a sphere due to higher multipoles, finite earth leads, transient currents and contact dynamics lie outside the approximation. Simultaneous grounded targets would require a coupled solution.

| Evaluation dimension | Status for the one original item | Evidence |
| --- | --- | --- |
| Interpretation fidelity | Pass, 1/1 | The contact preparation, equilateral geometry and switch order are explicit. |
| Physical correctness | Pass within the stated approximation, 1/1 | Each switch enforces the centre-monopole zero-potential condition with other net charges fixed. |
| Simulation coverage | Partial, 1/1 | Both grounding equilibria execute. The preceding contact is supplied as the textbook preparation. |
| Educational usefulness | Not evaluated, 1/1 | Reversing the order or omitting B's switch changes the result. No learner study was performed. |
| Textbook-answer completeness | Complete, 1/1 | The derived charge gives printed option 3. |
