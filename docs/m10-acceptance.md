# M10 acceptance: point charge over a grounded conducting plane

**Scope: one grounded infinite conducting half-space; exact analytical image
superposition, plane-aware viewer, and a finite-domain FEMM 5 numerical
comparison for one nonzero point charge.**

## Evidence and provenance

- [First actual FEMM 5 numerical run](https://github.com/physicsvigilante/conductor_sim/actions/runs/43275080726): **75 passed, zero skipped** — 69 existing numerical regressions and six new M10 live scenarios (positive/negative charge, finite-radius charge, V/E and sigma, finite disk charge, mesh/outer-domain/core sensitivity, rotation/translation and explicit multi-source refusal). This is native FEMM 5.0.20261101 headless and belasolv built on Ubuntu 24.04 from the pinned upstream commit `7c31ccfa3d4bcda8123d822cdda31fce929f672e`, not mock-solver output.
- [Python checks](https://github.com/physicsvigilante/conductor_sim/actions/runs/43275080504): 279 non-FEMM tests passed on the initial PR run; a subsequent viewer-provenance regression increases the final candidate to **280 tests**. The 75 live-FEMM tests intentionally skip in ordinary Python CI.
- [Chromium/source check](https://github.com/physicsvigilante/conductor_sim/actions/runs/43275080637): offline 3-state plane analytical viewer, controls, profile, hover reset, mobile layout, no external requests or JavaScript errors.
- [Review PR #13](https://github.com/physicsvigilante/conductor_sim/pull/13): repeat all checks on the final candidate head before merging. The final-run links are in the PR discussion.

## Finite-plane versus infinite-plane charge

An ideal infinite grounded plane has **exact total charge `Q_infinity=-q`** for
one charge q a distance h above its surface. FEMM models only a plane segment
from radial coordinate 0 to the finite computational radius R, so its **FEMM
reported net charge is a truncated-disk charge**, not `Q_infinity`.

The exact image-charge density integrated only to radius R yields:

`Q_disk(R) = -q * (1 - h / sqrt(h² + R²))`.

At the default `R = 8h` and `q = +1 nC`:

| Quantity | Exact reference | Actual FEMM |
|---|---:|---:|
| Conductor V | 0 V (grounded) | 0 V |
| Charge **on modeled disk only** | **−0.875965 nC** | **−0.875639 nC** (h=12 cm) |
| Complete mathematical infinite plane | −1.000000 nC | *not directly represented* |

The ~12.4% difference between `Q_disk` and `Q_infinity` represents the
**unmodeled infinite tail**, not a 12.4% FEMM discretization error. The
actual finite-disk discrepancy in this sample is about 0.04%. Numerical
mesh/Robin errors are tested separately. Values for other three-state heights
scale with h while `R/h` stays constant (as expected).

## What the implementation actually supports

- Physical `ConductingPlane` is solver-independent, with a 3D point and unit
  positive-normal direction into vacuum. The metal half-space extends on the
  negative side; only `Grounded()` makes physical sense for an ideal
  infinite conductor. It is not a two-sided zero-thickness sheet.
- Exact `GroundedPlaneBackend` reflects each point charge to a fictitious
  opposite-sign charge and superposes arbitrary, potentially non-collinear
  sources on the vacuum side. Tested boundary V=0, E=-grad(V), source force,
  energy, surface-charge distribution, total exact charge and rotated worlds.
- `Femm5PlaneBackend` independently compiles a **finite axisymmetric upper
  hemisphere**, a **finite charged numerical source core**, and a fixed-zero
  conductor segment on the plane. The far Robin boundary is matched to the
  ideal **dipolar** leading decay. It reads FEMM conductor charge, numerical
  source charge and sampled V/E; near-surface sigma is calculated using FEMM's
  one-sided E, not substituted from the analytical reference.
- Only **one point-charge site** can be solved by the M10 numerical FEMM
  backend. It may appear anywhere in world coordinates above the plane:
  the local symmetry axis is automatically translated and rotated.
  Multiple or non-axisymmetric sites remain **analytical-only** and the
  numerical backend rejects them before opening a FEMM session.
- M5 rendering uses general shape boundary intersection, renders an actual
  half-space in 2D cross-section, masks metal and stops field lines at the
  physical line. Planar sigma profiles show signed distance in metres,
  spherical profiles still show angle. No scene graph or live browser solver.

## Numerical acceptance tolerances and honest limitations

The M10 live tests permit up to 6% relative (plus 20 pC absolute) error for
**truncated-disk charge**, 5% relative for measured source charge, 24% + 2.5 V
for potential, 44% + 200 V/m for field, 65% + 0.4 nC/m² for near-plane sigma,
38% for the near-plane field-integral charge over a partial disk, 9% for
domain/mesh disk-charge variants, 22% for sampled potential sensitivity and
15% for orientation field comparison. These acceptance tolerances are
**not universal certified numerical error bounds**.

FEMM source cores are finite conducting spheres rather than literal points;
field and sigma near the source or at the plane depend on the mesh, the
near-plane sampling offset and the finite computational boundary. At extremely
small height or far outside the computational hemisphere, the numerical
approximation is not validated. The physical infinite-plane solution has no
finite prescribed conductor net charge; the view is precomputed equilibria,
not motion or live solving. Further planes/plates and non-axisymmetric 3D
numerical arrangements need new independent acceptance.
