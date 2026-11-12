# Milestones

M1 — **Complete, GitHub CI green:** solver-independent immutable models,
analytical solution for a centered charge in one conducting shell, tests.

M2a — **Accepted for empty-shell scope; real FEMM 5 Linux benchmark passed:** FEMM 5 binding/session isolation,
axisymmetric empty isolated charged shell, fixed conductor charge, Robin
outer boundary, numerical field sampling and live comparative tests. See
[`femm5.md`](femm5.md) for setup and [M2a acceptance](m2a-acceptance.md)
for run evidence. Normal Python CI uses fake command recorders; the dedicated
FEMM workflow builds and executes a real solver.

M2b — Grounded empty conductor and centered point-charge source; regularization
and convergence checks before accepting charge-specific numerical accuracy.

M3 — Off-center axisymmetric point charge; nontrivial induced charge density
and Gauss-law/analytical validation, with conductor/cavity surface identity.

M4 — Multiple compatible conductors and geometry composability with explicit
capability checks.

M5 — Procedural visualization separate from FEMM (potential maps, field lines,
surface charge density, parameter sweeps).

M6 — LLM authoring in terms of physical components; backend selection and
observability without embedding FEMM commands in authored problems.
