# FEMM 5 (femm-qt) setup and live benchmark

**Status: M2a, M2b and M3 accepted in their documented limited domains.** The real FEMM 5
headless solver was built from the pinned November 1, 2026 release. All **17 real FEMM numerical tests passed** in GitHub Actions (Ubuntu 24.04). See
[M2a acceptance](m2a-acceptance.md), [M2b acceptance](m2b-acceptance.md), and [M3 acceptance](m3-acceptance.md). The ordinary Python job still
uses fake clients; only the separate **FEMM 5 real-solver validation** workflow
provides numerical evidence.

## Installation

Use [FEMM 5](https://femm.info/doku/doku.php?id=download5), not FEMM 4.2.
FEMM 5 provides a TCP Python binding called `pyfemm-qt`. It has the same
import name `femm` as FEMM 4.2's incompatible `pyfemm` package. The adapter
requires the path to **FEMM 5's own `femm.py`** and loads it under a distinct
private name, avoiding pollution of global `import femm`. It requires the
`FemmSession` class documented by FEMM 5.

After installing FEMM 5, in a PowerShell window:

```powershell
git clone https://github.com/physicsvigilante/conductor_sim.git
cd conductor_sim
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
$env:FEMM5_BINDING = 'C:\femm-qt\pyfemm\femm.py'
# Set FEMM_DIR only if FEMM 5 is not in its default installation directory.
# $env:FEMM_DIR = 'C:\femm-qt\build\bin'
python examples/femm5_empty_shell.py
python -m pytest -m femm5 -v
```

Adjust `FEMM5_BINDING` and `FEMM_DIR` to **actual installed locations**.
On Linux/macOS use shell environment variables and a FEMM 5 build, as described
by the upstream [binding instructions](https://femm.info/doku/doku.php?id=pyfemm-qt).

## Explicitly supported physics

- M2a (accepted): exactly one isolated conducting spherical shell with **fixed net charge** and **no point charges**.
- M2b (live-validated): the same shell with **Grounded()** or **FixedNetCharge**, plus only centered point charges; finite concentric source core with measured source charge and excluded interior. See [M2b design](m2b.md).
- M3 (live-validated): one off-centre point-charge location anywhere inside the spherical cavity, transformed to a local FEMM axis, with `Grounded()` or `FixedNetCharge`. The finite charged source sphere approximates an ideal point; see [M3 design](m3.md).
- Uniform permittivity throughout space. Independent analytical benchmarks use `ConcentricShellBackend` for centred sources and `ImageChargeShellBackend` for one displaced source.
- The sphere center can be anywhere in world coordinates: compilation translates it
  to the symmetry origin; field samples transform axisymmetric (E_r,E_z) back to 3D.
- The outer spherical boundary is assigned a Robin/mixed condition
  `epsilon*dV/dn + (epsilon/R)*V = 0`, appropriate to a monopolar field.
  It does not inject the reference potential into the model.
- Both inner and outer spherical surfaces share **one FEMM conductor property**.
  The interior metal is meshed as source-free dielectric but held equipotential
  at its surfaces; this should give zero field there.

Total inner/outer surface charges in `ConductorResult` are inferred from Gauss's law using FEMM's measured source and shell total charges, not from a direct surface integral. M3 separately adds `Femm5Solution.surface_charge_density(surface, theta_rad)`: a **near-boundary estimate from FEMM's one-sided field**, compared to the independent image-charge density and numerically integrated in live tests. This is not a direct boundary-element density extraction, and results depend on mesh and sampling offset.

## Geometry and sampling

FEMM is built on a 2D half-plane `(r,z)` and rotates it around the `z` axis.
Three two-arc semicircles represent shell inner radius `a`, outer radius `b`,
and computational radius `R`. The symmetry axis is subdivided at each interface.
The cavity, metal, and exterior regions are meshed with uniform permittivity.
The FEMM session remains open while the returned solution is used:

```python
with Femm5Backend().solve(problem) as solution:
    potential = solution.potential_at((0.18, 0, 0))
    field = solution.electric_field_at((0.18, 0, 0))
```

Do not sample the electric field **on** a conductor surface; there is a
jump between one-sided limits. Samples beyond the computational boundary
are rejected. To clean up resources predictably, always use the `with` block.
FEMM is given a fresh high TCP port for each session. **There is not yet a
hard external timeout** for hung solver processes; one will be needed for
untrusted/batch problem execution. For now run only trusted, bounded models.

## Acceptance criteria and limitations

The tests in `tests/test_femm5_adapter.py` validate FEMM command construction,
version collision avoidance, response handling, unsupported geometry, and
failure cleanup, without running FEMM. Tests in `tests/test_femm5_live.py`
require a real FEMM 5 installation and compare potential, field, charge, and
sensitivity to mesh/domain settings. These tests are skipped by ordinary CI unless `FEMM5_BINDING` is configured.
In the dedicated FEMM CI job, the real solver and binding are built/present
and all tests run (no skips). M2a, M2b and M3 acceptance is limited to their
specified geometries and stated numerical tolerances; it does not establish
arbitrarily fine mesh convergence or correctness for unsupported arrangements.

M2b validated centred charges and grounding; [M3](m3-acceptance.md) adds a single off-centre charge position and numerical density profiles. The backend **does not** solve multiple distinct charge locations, multiple physical conductors, general planar geometry, or arbitrary 3D. These are rejected rather than silently approximated; near-wall charge positions require additional source-core and mesh validation.

Upstream references: [FEMM 5 pyfemm-qt](https://femm.info/doku/doku.php?id=pyfemm-qt),
[FEMM electrostatics tutorial](https://femm.info/doku/doku.php?id=electrostaticstutorial),
[FEMM command reference](https://femm.drsz.pl/doku.php/users_manual).
