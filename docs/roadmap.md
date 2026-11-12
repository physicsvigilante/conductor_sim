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

M4a — **Accepted (23 real FEMM 5 tests, including all 17 M2/M3 regressions):** distinct charge positions along one common symmetry axis in one spherical shell; independent numerical source cores, exact analytical superposition, and explicit rejection of non-collinear FEMM problems. See [M4a](m4a.md) and [numerical acceptance](m4a-acceptance.md).

M4b — **Accepted (32 real FEMM 5 tests, including 9 new nested-shell cases):** strictly nested concentric spherical shells, independently grounded/fixed-net-charge, source charges inside innermost cavity, with an independent exact radial/image reference and explicit capability checks. See [M4b](m4b.md) and [numerical acceptance](m4b-acceptance.md).

M5 — **Accepted (148 ordinary tests, 35 real FEMM tests and browser checks):** shared slice sampling, potential maps, contours, directed field traces, surface profiles and offline precomputed sweep playback. Includes M4 sampler/input corrections. See [M5](m5.md). No scene graph or browser-time FEMM recomputation.

M6a — **Accepted (19 page-anchored problems, 13 runnable and 6 blocked; 40 actual FEMM tests including 5 PDF cases):** 19 anchored JEE cases from
POTENTIAL_CONDUCTORS1.pdf, 13 runnable using existing physical models,
6 explicit capability blockers. Physics expectations are independent of
solver output. See [M6 benchmark](m6-benchmark.md) and [acceptance](m6-acceptance.md).

M6b — **Accepted for one open JEE authoring trial (192 Python + 47 real FEMM tests):** 15 new JEE-style authoring
prompts with a separate assistant-written composition module and independent
Gauss-law grading keys. Nine numerical-capable, one analytical-only, five
correctly unsupported; solver/viewer internals unchanged. See [M6b](m6b.md) and [acceptance evidence](m6b-acceptance.md).

M7 — **Accepted for one external charge near a grounded solid sphere (52 actual FEMM tests):** one external point
charge near one grounded solid spherical conductor. Adds `SolidSphere`, an
independent exact Kelvin image-charge/force reference and a separate FEMM 5
axisymmetric backend; integrates into existing offline viewer. See [M7](m7.md) and [M7 numerical acceptance](m7-acceptance.md).

M8 — **Candidate pending pinned real FEMM acceptance:** one external source
near an isolated solid conducting sphere with prescribed net charge Q, exact
Kelvin image+central compensator, calculated sphere potential, measured
numerical E/V/sigma, force and energy from the independent analytical
reference. Grounded and isolated cases use the same FEMM compiler.
See [M8 design](m8.md). All M7 regressions retained.

M9 — Extend external geometry/charge-site composition only after validating
M8: multiple independent external sources on the axis, a new conducting plane
component, and charge/ring field components. Charged-particle dynamics remain
a separate physical engine.
