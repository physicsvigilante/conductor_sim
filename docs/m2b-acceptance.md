# M2b numerical acceptance — centered point charge and conducting shell

**Accepted: 2026-11-12, for the explicitly supported spherical/centered domain only.**

## Evidence

- [GitHub Actions FEMM 5 real-solver execution](https://github.com/physicsvigilante/conductor_sim/actions/runs/43204319468): job `femm5-empty-shell` reported **success**.
- Upstream FEMM 5 source: [femm-qt](https://github.com/dcm3c/femm-qt), release tag `v5.0.20261101`, verified commit `7c31ccfa3d4bcda8123d822cdda31fce929f672e`.
- Linux host `ubuntu-24.04`, Python 3.12; actual `femm_headless` and `belasolv` executables built from the pinned upstream tree.
- **9 passed, 0 failed, 0 skipped** in `tests/test_femm5_live.py` (including the two already-accepted M2a regression tests). All ordinary Python checks also passed.
- Code candidate: `0cea2f842fc39094ac4bf62e8f2542e21dcf47bc` on PR #2.

## Independently verifiable numerical sample

Geometry: inner radius 0.10 m, outer radius 0.12 m, centered charge +1 nC, vacuum throughout, numerical source-core radius 0.01 m, outer Robin boundary at 0.96 m.

| Observable | Analytical reference | Real FEMM 5 |
| --- | ---: | ---: |
| Isolated shell net charge | 0 C | 0 C |
| Isolated shell potential | 74.896265 V | 74.614275 V |
| Isolated cavity potential at r=0.05 m | 164.771783 V | 162.886310 V |
| Isolated cavity E_r at r=0.05 m | 3595.020717 V/m | 3855.613660 V/m |
| Grounded shell potential | 0 V | 0 V |
| Grounded shell net charge | −1.0 × 10⁻⁹ C | −1.000000000014705 × 10⁻⁹ C |
| Grounded shell outer charge (Gauss inference) | 0 C | −1.47 × 10⁻²⁰ C |
| FEMM measured numerical-source net charge | +1.0 × 10⁻⁹ C | +1.0 × 10⁻⁹ C |

The full sample tables are printed by the standalone `examples/femm5_centered_charge.py` program in the workflow logs. The finite-core approximation is used *only outside the core*; no FEMM result is substituted with the analytical reference.

## What was actually tested

- Isolated neutral, isolated prescribed nonzero-charge, and grounded shells with centered source.
- Both positive and negative central point charges; empty grounded shell.
- FEMM-reported source and physical shell total charges and shell potentials.
- Cavity, metal, and exterior potential and field samples compared against independent exact solutions.
- Source-core radius and mesh-parameter sensitivity; previous M2a mesh/domain sensitivity is retained.
- Demonstrated numerical agreement at the tested sample points, **not** a general convergence proof.

Tests permit up to 10% relative error on nonzero sampled potential (plus 0.65 V absolute), 20% on sampled field (plus 2 V/m absolute), 5% on shell charge (plus 20 pC absolute), and 4% on measured source charge; the sensitivity test requires all tested source-radius/mesh variants to remain within 10% of the reference potential and within 10% of each other. The reported cavity-field example differs by approximately 7.25%.

The physical shell's **inner and outer total surface charges are inferred by Gauss's law** from FEMM-measured source and shell total charges, not local `sigma(r,z)` measurements. The numerical source is a separate FEMM charged conductor not part of the authored physical geometry.

## Not part of M2b acceptance

Off-center point charges, multiple shells, arbitrary 3D/non-axisymmetric configurations, surface-charge-density maps, full scene graph/visualization, external solver process termination guarantees, or universal mesh convergence. These require additional milestones and their own numerical evidence.
