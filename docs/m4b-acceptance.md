# M4b acceptance: independent concentric conducting shells

**Accepted 2026-11-12 for up to eight nonintersecting concentric shells, each with an independent Grounded or FixedNetCharge constraint, and up to 16 FEMM-axis-collinear source sites inside the innermost cavity.**

## Real FEMM evidence

- [Pinned FEMM 5 integration run](https://github.com/physicsvigilante/conductor_sim/actions/runs/43214380394) on candidate commit `f8d37155fbebf636d186774c7f153e4a51c168d3`: **32 passed, zero failures, zero skipped**, running the actual FEMM 5 headless electrostatics solver.
- **Nine new M4b live cases**: 4 grounded/fixed-charge combinations for two shells with two displaced sources, numerical inner/outer near-surface sigma, reversed shell ordering, empty charged inner shell with grounded outer shell, 3-shell mixed constraints, and numerical mesh/domain sensitivity.
- 23 previous M2–M4a real FEMM tests retained and passed.
- Ordinary Python jobs for Python 3.10 and 3.12: **80 passed, 32 FEMM-dependent skipped**; all FEMM tests are executed separately in the live job.
- Upstream FEMM 5 tag `v5.0.20261101`, pinned source commit `7c31ccfa3d4bcda8123d822cdda31fce929f672e`, compiled on Ubuntu 24.04.

## Verified sample (all SI units)

Two physical shells: inner (radius 0.10–0.12 m, isolated neutral), outer (radius 0.20–0.22 m, grounded). Two numerical source spheres approximate +1 nC at z=+0.035 m and +0.45 nC at z=−0.025 m inside the innermost cavity.

| Observable | Independent nested-shell reference | Real FEMM 5 |
| --- | ---: | ---: |
| Inner shell voltage | +43.439834 V | +43.245767 V |
| Outer shell voltage | 0 V | 0 V |
| Inner shell net charge | 0 C | 0 C |
| Outer shell net charge | −1.450000000 × 10⁻⁹ C | −1.449999999914 × 10⁻⁹ C |
| Potential at radial x=0.055 m | 119.923755 V | 118.428321 V |
| Potential at x=0.16 m (inter-shell gap) | 16.289938 V | 16.315264 V |
| Potential at x=0.21 m (outer metal) | 0 V | 0 V |
| Potential at x=0.30 m (grounded exterior) | 0 V | 0 V |

The FEMM solution independently solves the shell potentials and net charges
from one axisymmetric geometry; the exact reference is derived from radial
Gauss-law equations and Kelvin-image superposition, **not** FEMM output.

The shell surface-charge **totals** are inferred from FEMM-measured net
charge and enclosed source charge using Gauss's law. Numerical angular
sigma samples are derived from near-surface one-sided FEMM fields; they
depend on mesh and sample offset and are not direct surface integrals.
Finite-radius charged cores are not exact point charges.

## Acceptance thresholds and explicit exclusions

The real tests allow up to 20% relative plus 1.5 V absolute error on sampled
potential, 45% relative plus 180 V/m absolute on field, 7–10% relative plus
30–40 pC absolute error on net conductor charge, 60% relative error on
sampled density, 40% relative error on near-field sigma integral, and
12% relative plus 0.2 V absolute sensitivity for the sampled mesh/domain
comparison. These are **representative benchmark tolerances**, not proven
bounds throughout the domain.

Not accepted: nonconcentric, intersecting or touching physical shells;
charge sites in inter-shell gaps or exterior space; non-axisymmetric FEMM
sources; genuinely arbitrary 3D FEMM; local high-resolution charge-density
integration; procedural viewer or scene graph; production-safe process
supervision/timeouts. M5 visualization is a separate milestone.
