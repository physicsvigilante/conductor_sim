# M9 acceptance — multiple external charges and one solid conducting sphere

**Accepted for the limited M9 domain on 2026-11-12, subject to final source-head and post-merge CI gates.**

## Source and real FEMM numerical verification

- [M9 candidate's first real FEMM 5 run](https://github.com/physicsvigilante/conductor_sim/actions/runs/43260227644): **68 passed, zero skipped or failed**. Nine new M9 live cases (four charged/grounded configurations, angular sigma and flux, zero-total-charge external dipole, rotation/translation, mesh/domain/core sensitivity, and explicit non-collinear refusal), plus **all 59 M1–M8 numerical regressions**.
- FEMM source: [dcm3c/femm-qt](https://github.com/dcm3c/femm-qt) tag `v5.0.20261101`, commit **`7c31ccfa3d4bcda8123d822cdda31fce929f672e`**, independently checked by the GitHub workflow. Solver `femm_headless` and electrostatic `belasolv` built on Ubuntu 24.04, Python 3.12.
- [Python 3.10 and 3.12 jobs](https://github.com/physicsvigilante/conductor_sim/actions/runs/43260227901): 230 ordinary tests passed on the initial M9 code. Later exact-force 3D regression and internal source-core provenance refinement increase the expected ordinary count to **231**; the final-head workflow must pass again.
- [Offline Chromium/source review](https://github.com/physicsvigilante/conductor_sim/actions/runs/43260227783): passed. A four-state HTML/JSON example was generated from actual FEMM results, downloaded, and separately exercised with Chromium: state controls, surface profiles, hover, stale-probe reset, zero external requests/JS errors, and no tested mobile horizontal overflow.
- Analytic reference and numerical implementation are separate; FEMM never substitutes Kelvin images for measured field, potential, source charges or physical sphere net charge.

## Independently checked example

One solid sphere of radius `a=0.12 m` at the origin; two real external charges **`+1 nC` at `(0,0,+0.27) m`** and **`−0.5 nC` at `(0,0,−0.32) m`**. The numerical FEMM model uses TWO independent finite charged source cores (typically radius `0.012 m`), both outside the sphere. Uniform vacuum permittivity. All values are SI.

| Conductor constraint | Exact sphere Q (nC) | FEMM Q (nC) | Exact metal V (V) | FEMM metal V (V) |
| --- | ---: | ---: | ---: | ---: |
| Grounded | −0.256944 | −0.258935 | 0.00000 | 0.00000 |
| Isolated neutral, Q=0 | 0.000000 | 0.000000 | +19.24418 | +19.30554 |
| Isolated, Q=+1.2 nC | +1.200000 | +1.200000 | +109.11970 | +108.77442 |
| Isolated, Q=−0.8 nC | −0.800000 | −0.800000 | −40.67283 | −40.34038 |

The grounded charge differs by approximately 0.77%, the neutral metal potential by approximately 0.32%, and the prescribed-Q potentials by approximately 0.32–0.82% in the sampled configurations. FEMM's prescribed net-Q values are *inputs*, so the independent conductor potential, electric field and local induced density checks are essential additional evidence.

## Analytical physics checks and significance

The external multi-charge analytical reference is exact for **arbitrary 3D arrangements of point charges** strictly outside one grounded or isolated spherical conductor. Each charge gets its own Kelvin image, and fixed-net-charge metal adds one central compensating image. The result computes conductor equipotential and net charge, local surface density, E/V and forces on each *active grouped* external source. `force_on_charge_n(i)` includes Coulomb forces from other **real** charges; `conductor_force_on_charge_n(i)` includes the conductor's field alone. The interaction energy includes the factor-1/2 grounded image term and isolated sphere charging term, excluding constant self-energy; checked independently by `F_i=-grad_{r_i}(U)` for both collinear and non-collinear sources.

The analytic net surface density integrates to the exact conductor charge, verified by full polar/azimuthal quadrature for non-collinear cases. Non-collinear external-source arrangements cannot be numerically represented in FEMM 5's axisymmetric mode. The numerical backend rejects them **before** opening FEMM rather than showing a physically different revolved-ring model.

The real suite additionally checks individual measured source charges, opposite-signed local density and its coarse flux integral, source/core radius sensitivity, finite outer boundary sensitivity and world-axis rotation covariance. A zero-net external ±dipole retains **both nonzero physical source cores**, not zero source geometry.

## Numerical tolerances and remaining limitations

Tests have bounded, representative criteria (not global numerical error guarantees): up to **20% relative + 3 V absolute** for sampled potential; **43% + 170 V/m absolute** for field; **14% relative + 20 pC absolute** for physical grounded net charge; **3% relative + 1 pC absolute** for individual core charge; **70% + 1.5 nC/m² absolute** for sampled local sigma; **42% + 60 pC absolute** for the coarse near-surface sigma integral; plus stated mesh/domain and rotation checks in `tests/test_m9_live.py`. Not an engineering-grade convergence certification.

The point charges in FEMM are **finite charged spherical conductors**, not ideal point charges; they polarize and must remain clear of the metal and each other. The Robin outer boundary approximates infinity and multipole decay. The analytic force and energy are *not* measured by FEMM. The viewer is an offline snapshot player, not particle dynamics or continuous recomputation.

**Not accepted:** non-collinear numerical FEMM external-charge configurations; general 3D conductor geometry; multiple physical spheres; conducting planes; rings; touching/near-contact high-precision limits; production-grade execution isolation; field-line flux weighting; or a full scene graph.
