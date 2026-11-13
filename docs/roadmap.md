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

M8 — **Accepted (59 real FEMM 5 tests, including 7 new fixed-net-charge exterior sphere tests):** one external source
near an isolated solid conducting sphere with prescribed net charge Q, exact
Kelvin image+central compensator, calculated sphere potential, measured
numerical E/V/sigma, force and energy from the independent analytical
reference. Grounded and isolated cases use the same FEMM compiler.
See [M8 design](m8.md) and [numerical acceptance](m8-acceptance.md). All 52 earlier numerical regressions retained.

M9 — **Accepted for multiple external charges (68 real FEMM tests, including 9 M9 cases):** exact Kelvin image superposition for arbitrary source locations outside one grounded/isolated solid sphere, separate FEMM charged source cores for axisymmetric sites, strict non-collinear numerical refusal and offline four-state visualization. See [M9 design](m9.md) and [acceptance](m9-acceptance.md).

M10 — **Accepted (75 real FEMM tests, including 6 new grounded-plane cases):** grounded infinite conducting half-space,
exact multiple-charge mirror-image analytical solution, reusable plane/sphere
sampling geometry, flat viewer rendering/profiles, and actual axisymmetric
FEMM 5 approximation for one charge with finite-domain sensitivity checks.
See [M10 design](m10.md) and [M10 acceptance](m10-acceptance.md).
M11 subsequently extends the axial numerical capability.

M11 — **Numerical acceptance passed (83 actual FEMM tests, including 8 new M11 cases):** multiple distinct point charges above a single
grounded conducting plane. Real FEMM 5 uses up to 16 independent source cores
sharing one normal-line footpoint, with no false 2D rings or silent fallback.
Non-collinear charges remain analytically supported. See [M11 design](m11.md)
and [M11 numerical acceptance](m11-acceptance.md).


M12a — **Accepted analytical two-plate reference (297 ordinary tests, retained 83 real-FEMM regressions, Chromium):** two
facing infinite conducting half-spaces, with `Grounded`/`FixedPotential`
boundary values, zero or multiple point sources in the slab, and the existing
viewer. Exact capacitor background plus finite symmetric image series for
sources; transparent `None` for undefined infinite-plate total charge.
M12b subsequently added separately validated finite-radius FEMM support.
See [M12a](m12a.md) and [PR #15](https://github.com/physicsvigilante/conductor_sim/pull/15).


M12b — **Numerical acceptance passed (90 actual FEMM 5 tests, including seven new plate-pair cases):**
reuse the same authored two-plane capacitor to compile separately constrained
finite electrode disks, charged axial spherical FEMM source cores and an
artificial zero-radial-flux cylindrical outer boundary. Independently measure
all FEMM electrode/source charges; test the finite-cylinder weighting-potential
charge identities, local analytic V/E/sigma and convergence. Keep arbitrary
3D source configurations analytical-only. See [M12b](m12b.md) and [M12b acceptance](m12b-acceptance.md).


M13a — **Analytical force/energy observation layer:** optional `force_energy(solution)`
returns separately labeled direct, induced and conductor-background forces and
mechanical interaction energies for grounded planes, exterior grounded/fixed-Q
spheres and parallel plates. The plate image kernel now preserves reciprocity
and plate-swap symmetry at finite order. Legacy APIs are retained; FEMM and
cavity force observations are explicitly unsupported, not replaced by a fallback.
See [M13a methods and acceptance contract](m13a.md). Exact-head acceptance is
recorded in the associated pull request. **M13b force/energy visualization and
M13c expanded JEE authoring validation are separate, not included here.**


M13b — **Implementation candidate; separate acceptance required:** a reusable
analytical position sweep of one explicitly tracked authored point charge,
precomputed static equilibria, in-slice force arrow, force-versus-position
and mechanical-energy graphs, and external quasistatic work readouts in the
existing offline viewer. Preserve all other charges, conductor geometry,
constraints and voltage gauge. Refuse unsupported FEMM force observations.
See [M13b design](m13b.md).


M13c (provisional open authorability trial) — **18 fresh open JEE-style prompts**
composed using only M13a/M13b public APIs: ten analytical supported sweeps,
eight explicit and physically justified refusals. Independently check geometry,
SI forces, interaction-energy components and displacement work, then render
supported cases through the existing viewer. This is *not* a blind trial.
The new scanned `02_conductors_half_size.pdf` has since been uploaded to
`main`: 17 pages visually reviewed, 18 selected source anchors classified,
**2 independently validated** existing analytical conducting-sphere examples,
2 static-only and 14 unsupported-force/energy categories. The two PDF-authored
demonstrations are distinct from 18 original open assistant prompts.
See [M13c PDF audit and limitations](m13c.md).

M14a — **Connected concentric-shell analytical composition candidate.** A
reusable `ElectricalLink` + `ConnectionPlan` solves equal-potential and
linked-group charge conservation/grounding without modifying the existing
spherical solver or scene viewer. It compiles into familiar `FixedNetCharge`
and `Grounded` shell conditions and returns two equilibria, per-shell
charge transfers and radial field-energy release. This newly supports the
ideal concentric-shell wiring class of `02_conductors_half_size.pdf`, including
page 14 Numerical Q7 with exact `delta V_outer = 0 V`. No transient current,
solid inner sphere, separated wired spheres or FEMM wire-geometry accuracy is
claimed. See [M14a design](m14a.md); acceptance subject to complete CI.

M14b — **Exact analytical mixed concentric conductors candidate:** extend
M14a's `ConnectionPlan` to one solid conducting sphere enclosed by one or
more separated concentric shells, grounded/floating links and radial energy
release. The new shared `MixedRadialBackend` reuses the M14a capacitance
response matrix but has **no fictitious cavity** in the solid. The accepted
M14a all-shell domain remains intact. Reproduce JEE page 3 Q27 (about 9 J
released after joining sphere and shell) and page 10 Linked Q13–14 (about
0.09 J initially stored and fully released upon linking opposite shells).
No FEMM wire geometry, off-center point charges in the mixed domain, or
transient current. See [M14b](m14b.md); require exact-head regression gates.


M15a/b — **Analytical cavity force and reusable position-sweep candidate:**
provide exact self-field-free Kelvin-image force, induced/background and
mechanical interaction energies for centered/one/multiple point charges in a
single spherical cavity and the innermost cavity of concentric nested shells.
The existing `PositionSweepSpec` now accepts strictly interior straight paths,
including paths starting at the centre. The unchanged offline viewer shows
force arrows, graphs, work and field/potential/density overlays. No FEMM force,
physical time motion or new viewer renderer. See [M15](m15.md); exact-head
Python 3.10/3.12, browser and retained real-FEMM regression acceptance required.


M16a — **One-dimensional electrostatic dynamics candidate:** integrate a
single charge's actual elapsed-time motion through the existing analytical
force/energy provider on provably symmetric lines, with adaptive RK4,
turning-point detection, conductor-surface guard, sample-level kinetic and
interaction energies and independent energy-conservation checks. Includes
normal plane/capacitor, radial exterior sphere, and spherical cavity domains.
No phantom constraint forces, numerical FEMM force, physical surface
collisions, true 3D particle dynamics or HTML time animation. M16b will add
a distinct time-playback viewer only after the dynamics validation. See
[M16a](m16a.md); acceptance subject to exact-head GitHub CI.

M16b — **Offline time-based playback over accepted M16a dynamics:** reuse
`sample_motion_1d` and the existing field-map viewer with a synchronized
elapsed-time slider, play/pause/restart, particle and signed velocity arrows,
position/velocity-versus-time, kinetic/interaction/total energy traces, and
the original force/energy-drift readouts. Every curve uses saved actual M16a
samples, while a bounded selection of true time snapshots each has a matching
solved electrostatic field; no browser ODE, time-proxy position sweep or FEMM
force substitution. See [M16b](m16b.md). Full Python 3.10/3.12,
Chromium and retained real-FEMM validation required before acceptance.

M17a — **Accepted in PR #30: planar vector electrostatic dynamics:** reusable
`Motion2DSpec` and `integrate_motion_2d` over existing analytical
`force_energy` providers; true 3-vector positions/velocities confined to a
physically invariant plane by symmetry, not by a fabricated constraint.
Supports oblique launches between infinite conducting plates, central-force
exterior-sphere and spherical cavity motion, conservation diagnostics and
vacuum-side approach guards. M16a retains its original API and controller
but shares the dimension-independent RK4 stepping kernel with M17a.
No new FEMM force, numerical fallback, multi-moving-charge or browser
curved-path viewer. See [M17a](m17a.md); exact-head Python 3.10/3.12,
retained Chromium/authorability and all real-FEMM regressions required before
merge. M17b will add curved-trajectory visualization.

M17b — **Accepted in PR #31: offline curved-trajectory time viewer:** reuses M17a's
`Motion2DResult` and M16b's field-frame selection/sampling pipeline, offline
HTML renderer, elapsed-time controls and energy graphs. Adds the complete
curved path, synchronous vector force and velocity arrows and two-component
position/velocity-time charts in the proven physical plane. All field frames
are solved only at exact saved trajectory times; up to 24 field snapshots
but up to 2001 authentic physics samples. No browser ODE, FEMM force
fallback or fabricated out-of-plane constraints. Original static, sweep and
M16b time modes remain distinct. See [M17b](m17b.md); acceptance requires
exact-head Python 3.10/3.12, full retained Chromium/authorability regressions
and all 97 real FEMM non-regression tests with zero skips.

M17c — **Correctness review and original curved-motion authoring trial:**
fix rotated radial slice degeneracy, validate every saved planar graph row
and field-frame identity, and omit sparse display chords crossing metal.
Twelve original prompts comprise eight supported trajectories (168 checked
states) and four explicit refusals. Independent test-only velocity-Verlet
and closed-form references do not import the production engine. Existing
M17a physics, M17b renderer and public model components are reused; no new
geometry/force solver or browser ODE. See [M17c](m17c.md) for scope, evidence,
limitations and exact-head Python/browser/97-real-FEMM acceptance gates.

M18a — **Accepted in PR #33: one moving charge plus externally held fixed charges:**
reuse the original immutable authoring components, M17 vector RK4 and
self-field-free `force_energy`, identify the moving component by its original
authored index independently of sorted grouped sites, and preserve all held
charge locations across every solver call. Admit only a proved reflection-
symmetry plane through the motion and *all* held sources. Record mechanical
energy drift and terminate at a guarded conductor or fixed point-charge
approach, without claiming impact. M18a exports genuine numerical motion as
JSON only; this milestone deliberately deferred multi-source HTML. Accepted
M18b now shows and validates the complete inventory. New independent mirror/
Kelvin, finite-difference and refusal tests. See [M18a](m18a.md); exact-head
Python 3.10/3.12, all retained browser/authoring gates and 97 real-FEMM
non-regression tests with zero skips required before acceptance.

M18b — **Accepted in PR #34: held-source viewer and output correctness:** extend the shared
`sample_motion_2d`/HTML pipeline to preserve and label every moving/held
source, certify full field inventories and keep all sources inside a fixed
physical slice. Reuse elapsed-time controls, vector arrows and energy graphs.
Correct duplicate final timestamps in scalar/vector motion; reject invalid
saved speed/guard metadata; omit sparse chords through held-charge guards.
No new force law, RK4 implementation or multi-moving-charge support. See
[M18b](m18b.md); acceptance requires all retained and new Python/browser
workflows and 97 real FEMM regressions with zero skips.

M18c — **Independent held-source motion authoring trial:** twelve original
textual prompts, eight supported trajectories (168 independently checked
states) and four justified refusals. Public-API authored models are separate
from standard-library boundary Green functions, a shell-constraint linear
solve and step-halved test-only velocity-Verlet. Includes source-identity,
unit, energy-convention and corrupted-output mutations, rotation/source-order/
voltage-gauge invariance, and production-only multi-charge browser exports.
No production solver, integrator or renderer changes. See [M18c](m18c.md);
require exact-candidate Python/browser and 97 real-FEMM gates before acceptance.
