# M4a acceptance: composition of multiple axial charge sites

**Accepted 2026-11-12 for up to 16 active point-charge locations on a common axis passing through the centre of ONE spherical shell.**

## Actual execution evidence

- [Real FEMM 5 integration, PR #4](https://github.com/physicsvigilante/conductor_sim/actions/runs/43212540764): **23 passed, 0 failed, 0 skipped** in tests/test_femm5_live.py plus tests/test_m4_live.py.
- **6 new M4a live tests**: two independently charged source sites with grounded, neutral and charged shell configurations; comparison of numerical field/potential with exact Kelvin-image superposition, induced sigma profile and angular integration, world-axis covariance, core-radius sensitivity.
- 17 previously accepted M2a/M2b/M3 tests also passed.
- GitHub Actions ordinary Python 3.10 and 3.12 jobs both passed. Python 3.12 test output: **64 passed, 23 skipped**; the 23 skipped require real FEMM and are executed in the separate live job.
- Pinned upstream FEMM 5 release `v5.0.20261101`, commit `7c31ccfa3d4bcda8123d822cdda31fce929f672e`, compiled on Ubuntu 24.04 with `femm_headless` and `belasolv`.

## Scientific interpretation

Each distinct, nonzero-net source is an independently charged *finite spherical conductor* placed on the symmetry axis, not a point charge or a ring. Source core radii adapt to the wall and neighbouring cores. FEMM measures each core charge separately, then the physical shell's connected conductor charge/potential. Integrated inner/outer physical shell charge totals are inferred using Gauss's law. Spatial `sigma` is estimated using one-sided FEMM field sampling (not an exact surface element integral).

The `MultiImageShellBackend` is an independent exact **analytical** reference using linear Kelvin image-charge superposition, including **non-collinear 3D source locations**. That broader analytical capability does NOT imply the FEMM backend can solve non-collinear configurations.

The M4a numerical tests allow up to 18% relative (plus 1.5 V absolute) potential error, 42% relative (plus 160 V/m absolute) field error, 55% relative local sigma error, 35% relative inner sigma-integral error, and 14% relative source-size sensitivity. These are narrow acceptance tolerances, **not general error bounds**. The tests use representative sources at axial offsets +3.5 cm and −2.5 cm in a 10 cm spherical cavity.

## Not accepted

Multiple physical conductors (M4b); distinct off-axis source positions in the FEMM backend; arbitrary 3D numerical field solving; touching/conducting source cores; high-precision local induced-charge density; scene graph; procedural viewer; untrusted FEMM process timeout control. The ordinary Python tests alone are not numerical validation.
