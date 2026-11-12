# FEMM 5 setup, supported domains and numerical evidence

**Status: M2–M9 accepted within documented geometries and tested tolerances.**
A dedicated GitHub Actions workflow compiles the pinned real FEMM 5 headless
solver, runs its electrostatics solver, and compares actual results against
independent analytical references. The accepted M7 run passed **52 real FEMM numerical tests** on Ubuntu 24.04.
M8 passed 59 tests; M9 expanded the real live numerical suite to **68 tests**, including 9 multi-source external cases. See [M8 acceptance](m8-acceptance.md) and [M9 acceptance](m9-acceptance.md).

## Installation

Use [FEMM 5](https://femm.info/doku/doku.php?id=download5), **not FEMM 4.2**.
The FEMM 5 `pyfemm-qt` TCP binding shares the Python import name `femm`
with FEMM 4.2's incompatible PyPI binding. Set `FEMM5_BINDING` to the
official FEMM 5 `pyfemm/femm.py` file. The adapter privately loads this
file and checks for the `FemmSession` class.

On Windows PowerShell after installing FEMM 5:

```powershell
git clone https://github.com/physicsvigilante/conductor_sim.git
cd conductor_sim
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
$env:FEMM5_BINDING = 'C:\femm-qt\pyfemm\femm.py'
# Set FEMM_DIR if the installed FEMM binaries are not on PATH:
# $env:FEMM_DIR = 'C:\femm-qt\build\bin'
python examples/femm5_nested_shells.py
python -m pytest -m femm5 -vv
```

Adjust installation paths to the actual FEMM 5 locations. On Linux/macOS,
use upstream's [binding documentation](https://femm.info/doku/doku.php?id=pyfemm-qt).
The default Python-only CI uses fake FEMM clients; a skipped live test is
not proof of numerical correctness.

## Numerical solver capabilities

- Zero or more point-charge components *inside the innermost cavity*.
  FEMM only accepts distinct nonzero-net source locations on one common
  axis through the shared conductor centre, and uses small independently
  charged spherical conductors to approximate each ideal point. The core
  radius is bounded by nearest-source and cavity-wall clearance. Sampling
  on/inside any source core is rejected.
- One or more **strictly nonintersecting, concentric** spherical conducting
  shells with separately assigned `Grounded()` or `FixedNetCharge(Q)`
  constraints. Each connected metal shell uses ONE FEMM conductor property
  on both inner and outer physical faces. Up to eight shells and 16 active
  axial source sites are allowed in the present bounded backend.
- Field/potential queries in world-space 3D coordinates are transformed to
  the automatically selected axisymmetric FEMM half-plane and back.
- `conductor_results` is in authored order, `conductor_by_name` finds
  an individual shell, and `numerical_sources` reports measured FEMM
  source-core charge and geometry. `source_radius_m` is retained for
  single-source compatibility; for multi-site problems use
  `numerical_sources` instead.
- `surface_charge_density("inner"|"outer", theta,conductor_name="...")`
  estimates local sigma using the appropriate **one-sided FEMM electric
  field** just outside the chosen metal surface. The nearest-surface
  sampling offset respects the available dielectric gap. For a single
  physical shell, `conductor_name` can be omitted. For several shells, it
  is required. Local sigma is not a direct boundary-element extraction.

FEMM directly measures physical conductor **net charge and potential**,
plus the numerical source-core charges. Reported *integrated*
inner/outer surface-charge totals are inferred using Gauss's law; they
are not separately measured integrations.

Each finite numerical source core is a **conducting sphere, not an exact point**.
Off-centre source cores polarize, and near-surface sigma depends on
mesh and offset. Core-radius, mesh and domain checks assess representative
sensitivity but are not universal guarantees. Field sampling *exactly*
on a conducting surface is rejected because it has two one-sided limits.

## Exterior solid-sphere (M7, M8 and M9)

The separate `Femm5ExternalSphereBackend` uses the same headless numerical
session and a genuine `SolidSphere` physical conductor, distinct from the
interior-shell compiler. It supports one or more active external point-charge sites with either
`Grounded()` or `FixedNetCharge(Q)`. **FEMM requires all sites collinear with
the sphere centre**, accepting up to 16 distinct grouped sites. The physical
sphere is one fixed-voltage (grounded) or fixed-total-charge (isolated) FEMM
conductor; each external source is a separate small fixed-Q conductor on the
symmetry axis. In the isolated case FEMM must **solve the metal potential**,
not force it to zero. The outer Robin boundary is approximate for multipoles;
source-core and computational-domain sensitivity checks remain necessary.
The numerical sphere's total/outer charge and potential are direct FEMM
conductor observations; its local density is a near-surface E estimate.
The analytical Kelvin image references separately compute induced sigma,
force and interaction energy. **Force is not measured by FEMM here.**

## Independent analytic references

- `ConcentricShellBackend`: central charges and one shell.
- `ImageChargeShellBackend`: one point charge at any position inside
  one spherical cavity (Kelvin image method).
- `MultiImageShellBackend`: arbitrary number of charges, even at
  **non-collinear 3D positions**, inside one spherical cavity.
- `ExternalGroundedSphereBackend`: one external charge and grounded solid sphere (M7).
- `ExternalIsolatedSphereBackend`: one external charge and fixed-Q solid sphere (M8).
- `ExternalMultiSphereBackend`: multiple external charges anywhere in 3D outside
  a grounded or fixed-Q solid sphere; exact Kelvin image superposition, induced
  density, per-source forces and interaction energy (M9).
- `ExternalSphereBackend`: exact dispatch between the M7/M8 single-charge and
  M9 multi-charge analytical references.
- `NestedShellBackend`: nested concentric shells with mixed grounding
  and fixed-charge constraints; arbitrary 3D point-charge positions in
  the innermost cavity. It solves the concentric radial equations and
  uses Kelvin-image superposition in the innermost cavity.

The analytical backends' general 3D charge arrangement capability does
**not** extend FEMM's axisymmetric numerical geometry support.

## Acceptance and remaining work

- [M2a numerical acceptance](m2a-acceptance.md)
- [M2b numerical acceptance](m2b-acceptance.md)
- [M3 numerical acceptance](m3-acceptance.md)
- [M4a numerical acceptance](m4a-acceptance.md)
- [M4b numerical acceptance](m4b-acceptance.md)
- [M7 acceptance](m7-acceptance.md)
- [M8 acceptance](m8-acceptance.md)
- [M9 acceptance](m9-acceptance.md)
- [Pinned FEMM live CI workflow](../.github/workflows/femm5-live.yml)

Unsupported (rather than silently approximated): eccentric or intersecting
spherical conductors, charges between shells or outside the innermost cavity,
non-collinear numerical FEMM arrangements (including multiple external sites),
arbitrary engineering 3D, general planar geometry and a scene graph.
The offline procedural viewer is already supported as M5.

The session has no built-in hard process timeout for arbitrary untrusted
problems; GitHub Actions live tests use an external deadline. This engine
should not yet run uncontrolled LLM-authored models in production.

## M10 numerical grounded-plane candidate

A separate `Femm5PlaneBackend` uses a finite quarter-disk axisymmetric
meridian with one grounded plane segment, one finite charged source core,
and a dipole-matched outer Robin boundary. The plane is **infinite only in
the analytical ideal**; numerical net charge and surface-density profiles
must be assessed for sensitivity to the outer domain and sampling offset.
The numeric capability is limited to one active external source above one
plane; `GroundedPlaneBackend` supports non-collinear source superposition
analytically. See [M10](m10.md). Initial pinned real FEMM CI passed 75 tests; see [M10 acceptance](m10-acceptance.md).

### M11: multiple sources on the normal to one grounded plane

`Femm5PlaneBackend` now supports up to 16 distinct active point-charge
sites on a shared normal to one grounded plane, with a separately prescribed
and measured charged finite FEMM core per site. The plane symmetry axis is
**perpendicular** to the plane; collinearity on an inclined line does not
suffice. Non-collinear numerical cases fail before launching the solver.
For a finite modeled disk of radius R, compare FEMM plane charge with
`-sum(q_i * (1 - h_i/sqrt(h_i**2+R**2)))`, rather than the infinite-plane
`-sum(q_i)`. The M10 single-source backend and zero-net **signed dipole**
regressions are retained. [M11 design and limits](m11.md).
