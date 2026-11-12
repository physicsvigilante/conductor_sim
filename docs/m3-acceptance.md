# M3 numerical acceptance — off-centre charge induction

**Accepted on 2026-11-12 for one source location (or coincident sources) strictly inside one spherical conducting shell, in a uniform medium.**

## Verifiable execution

- [Successful real FEMM 5 GitHub Actions run](https://github.com/physicsvigilante/conductor_sim/actions/runs/43207940483), job `femm5-conductor-physics`.
- All **17 live numerical tests passed, 0 failed, 0 skipped**: eight new M3 cases plus nine M2a/M2b regression tests. Ordinary Python CI on supported Python versions passed separately (**52 pure Python tests**, live integration skipped when FEMM is not installed).
- FEMM 5.0.20261101 from the pinned [femm-qt](https://github.com/dcm3c/femm-qt) release (verified upstream commit `7c31ccfa3d4bcda8123d822cdda31fce929f672e`), built and executed on Ubuntu 24.04 with a true headless solver and electrostatics backend.
- Proposed code: [PR #3](https://github.com/physicsvigilante/conductor_sim/pull/3), initial validation commit `5398c10ea32fcd4d59ebb7421581db4246070227`.

## Independent reference versus real FEMM results

Sample problem: shell inner radius 0.10 m, outer radius 0.12 m, vacuum, point charge +1 nC located at `(0,0,0.04)` m, isolated-neutral shell, numerical source-core radius 0.01 m. Polar angle `theta=0` points toward the displaced source.

| Observable | Exact Kelvin image-charge reference | FEMM 5 numerical result |
| --- | ---: | ---: |
| Shell potential | 74.896265 V | 74.589172 V |
| Potential at `(0,0,-0.06)` m | 92.291526 V | 92.184274 V |
| Potential at `(0.05,0,0)` m | 127.128046 V | 126.623787 V |
| Potential at `(0,0,0.075)` m | 203.289862 V | 206.937362 V |
| Inner sigma at 8.6 degrees | −2.982357 × 10⁻⁸ C/m² | −3.332131 × 10⁻⁸ C/m² |
| Inner sigma at 25.8 degrees | −2.293085 × 10⁻⁸ C/m² | −2.458513 × 10⁻⁸ C/m² |
| Inner sigma at 90 degrees | −5.350353 × 10⁻⁹ C/m² | −5.891567 × 10⁻⁹ C/m² |
| Inner sigma at 171.4 degrees | −2.452889 × 10⁻⁹ C/m² | −2.668071 × 10⁻⁹ C/m² |

The angular nonuniformity is large and correctly signed. `sigma` is **computed from FEMM's one-sided field sampled a small distance into the cavity**, not copied from the analytical formula. The local-density test also integrates 60 angular FEMM samples and confirms that total inner charge is within 30% of `−1 nC`; this is a coarse near-boundary mesh/offset agreement test, not a direct surface-element integral. The exact analytic profile integrates to `−q` independently.

The live suite additionally validated grounded and prescribed-net-charge shells, signed positive/negative off-centre sources, 3D world-axis rotation covariance, FEMM potential and field versus the image-charge solution, and variation of finite-source radius. Numeric tolerance limits used in M3 are in `tests/test_femm5_live.py`: V relative 16% (plus 1.5 V absolute), field relative 32% (plus 130 V/m), density relative 55%, induced-density integral relative 30%, and source-radius potential stability relative 14%.

## Validity and remaining limitations

- Exactly one **distinct** point-charge location; coincident charge components can combine into a net source. The physical shell may be grounded or isolated.
- The finite charged sphere is an approximation to an ideal displaced point charge; unlike a true point charge, it polarizes. Core-radius and mesh sensitivity were tested for representative configurations, **not** proven for every charge displacement or near-wall gap.
- Inner/outer **total** induced charges reported in `ConductorResult` come from Gauss-law inference using FEMM-measured conductor/source totals. Angular sigma is separately estimated from numerical fields with mesh/offset dependence.
- No general multiple-charge 3D, multiple conductors, local high-resolution surface-charge mesh integration, scene graph, production solver timeout, or field-line viewer is accepted as part of M3.
