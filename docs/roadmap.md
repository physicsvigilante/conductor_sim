# Milestones

M1 — **Complete, GitHub CI green:** solver-independent immutable models,
analytical solution for a centered charge in one conducting shell, tests.

M2a — **Accepted for empty-shell scope; real FEMM 5 Linux benchmark passed:** FEMM 5 binding/session isolation,
axisymmetric empty isolated charged shell, fixed conductor charge, Robin
outer boundary, numerical field sampling and live comparative tests. See
[`femm5.md`](femm5.md) for setup and [M2a acceptance](m2a-acceptance.md)
for run evidence. Normal Python CI uses fake command recorders; the dedicated
FEMM workflow builds and executes a real solver.

M2b — **Accepted for centered-charge scope; 9 live FEMM 5 tests passed:** grounded empty
shell and centered point-charge source core, fixed-shell charge, charge
and potential agreement, and mesh/source-radius sensitivity. See [M2b](m2b.md).
M2a remains a separately accepted baseline; M2b does not extend support to off-center charges.

M3 — **Accepted for single off-centre source location; 17 live FEMM 5 tests passed:** automatically chosen 3D symmetry axis, independent Kelvin image-charge reference, numerical cavity/metal/exterior potential and field, near-surface induced sigma(theta) profile and integral. See [M3 design](m3.md) and [M3 acceptance](m3-acceptance.md). The 17 tests include the 9 previously accepted M2 benchmarks.

M4a — **Candidate pending live FEMM acceptance:** distinct charge positions along one common symmetry axis in one spherical shell; independent numerical source cores, exact analytical superposition, and explicit rejection of non-collinear FEMM problems. See [M4a](m4a.md).

M4b — Compatible *concentric nested spherical shells* with independent grounding/net-charge constraints and rigorous live numerical checks; this is a separate expansion, not implied by M4a acceptance.

M5 — Procedural visualization separate from FEMM (potential maps, field lines,
surface charge density, parameter sweeps).

M6 — LLM authoring in terms of physical components; backend selection and
observability without embedding FEMM commands in authored problems.
