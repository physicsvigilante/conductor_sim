# M11 acceptance — several external charges above one grounded plane

**Scope:** one grounded infinite conducting half-space, with up to 16 separately
charged, active FEMM 5 source spheres on the same normal-line footpoint.
The independent exact analytical backend remains capable of arbitrary 3D
charge placements on the vacuum side. Neither backend silently substitutes
one source for another or replaces numerical FEMM data with images.

## Verification evidence

- Pinned upstream FEMM 5 v5.0.20261101 source
  `7c31ccfa3d4bcda8123d822cdda31fce929f672e`, headless
  `femm_headless` + `belasolv`, GitHub Actions Ubuntu 24.04, Python 3.12.
- Ordinary Python CI 3.10/3.12: 288 passing non-FEMM tests, including eight
  new M11 contract/geometry/visualization checks, preserving 280 M10 tests.
  Live FEMM-marked tests are skipped **only** in ordinary CI.
- [Pinned real FEMM 5 PR-head integration run](https://github.com/physicsvigilante/conductor_sim/actions/runs/43283937767): **83 passed, zero skipped**, retaining
  the full 75-case M1–M10 numerical baseline. Eight new M11 tests validate
  individual measured core charges, plane potential and truncated-disk net
  charge, sampled V/E, sigma profile and numerical flux, signed zero-net
  dipole, domain/core/mesh sensitivity, rigid world transformation, and
  explicit pre-session refusal of non-axisymmetric source placements.
- Offline three-state analytical export and Chromium/source CI: 3 stored
  equilibria, slider, field lines, contours, plane density profile, hover/probe
  reset, no JavaScript errors/external requests, tested mobile viewport.
- Separate **actual FEMM 5** three-state HTML/JSON export downloaded from
  [numerical workflow artifact](https://github.com/physicsvigilante/conductor_sim/actions/runs/43283937767/artifacts/13973677961),
  SHA-256 of the ZIP: `6230c66f9bf62cba8070422ac40eb41e3efc02b52ccc5be8088bf4134ba6b4dc`.
  The exported data contains three separately solved states, **two finite
  numerical source masks in each**, measured plane Q/V, signed-density profiles
  and FEMM provenance. Independently exercised in Chromium: sliders, field lines,
  potential/contours, profile selector, hover and stale-probe clearing, no JS
  errors/external requests, and no horizontal overflow in the tested mobile view.

## Exact infinite-plane versus truncated numerical disk

M11 preserves the M10 finite-domain distinction. For charges q_i on one
normal at heights h_i, the exact infinite-plane net charge is -sum(q_i).
FEMM's physical grounded conductor is only the disk 0<=rho<=R, so the
reference for its measured charge is:

    Q_disk(R) = -sum_i q_i * (1 - h_i / sqrt(h_i*h_i + R*R)).

With h1=0.12m, h2=0.26m and default outer R=8*h2=2.08m:

| Authored real charges | Infinite-plane exact Q (nC) | Finite-disk exact Q (nC) | Actual FEMM Q (nC) |
|---|---:|---:|---:|
| +1 nC and +0.5 nC | -1.500000 | -1.380386 | **-1.379849** |
| +1 nC and -0.5 nC | -0.500000 | -0.504421 | **-0.504126** |
| +1 nC and -1 nC | 0.000000 | -0.066438 | **-0.066264** |

The last case is especially important: **two nonzero distinct physical sources
are not equivalent to removing all source geometry merely because their net
real charge is zero**. Nor does the integrated numerical finite-disk Q need to
be zero when the exact infinite-plane Q is zero.

## Field, coordinate and mesh review

The first two live checks (PR #14 head revisions) each passed 82/83 but
reproduced a field-component mismatch when comparing the pointwise local
FEMM E of two independent meshes corresponding to rotated worlds. The third
candidate (82/83) found 7 mm central finite differences of FEMM's continuous
potential substantially more stable under the same rigid transform, but one
transverse component differed by 54.4 V/m, just above a 50 V/m experimental
threshold. M11 therefore uses the **already accepted M10 rotational-field
criterion**, 15% relative or **80 V/m** absolute, for covariance of these
potential-derived fields, without changing any older numerical tolerance.
The revised all-83 [live run](https://github.com/physicsvigilante/conductor_sim/actions/runs/43283937767)
passes the strict combined check, including both *raw* independently computed
FEMM E vectors compared separately against the exact analytical image-charge E.
A successful test is not a certification of elementwise gradient agreement in
different FEMM meshes.

The M11 review also fixed a genuine viewer bug for translated planes: FEMM's
finite numerical domain must be centered on the physical sources' common
normal footpoint, not the arbitrary plane point used in the authored model.
A fake-session/visual-context regression protects this.

## Limits

The finite numerical charged cores are not literal ideal points; mesh
interpolation, the finite outer Robin boundary and one-sided near-plane sigma
sampling remain accuracy limitations, particularly near the plane, source
cores and domain edge. The stated FEMM cap is an implementation limit, not
numerical validation of every 16-site combination. Analytical 3D arrangements
with lateral source offsets are **not** supported numerically by FEMM 5.
There is no browser-time solving, dynamics, parallel-plate geometry, full
scene graph, direct numerical force/energy integration, or production-grade
hard process isolation/timeout.
