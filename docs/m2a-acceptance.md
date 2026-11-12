# M2a acceptance — numerical empty-shell FEMM 5 baseline

**Status: accepted for the explicitly limited M2a scope on 2026-11-12.**

- [Real FEMM 5 GitHub Actions execution](https://github.com/physicsvigilante/conductor_sim/actions/runs/43201326048), job `femm5-empty-shell`, conclusion **success**.
- Upstream solver: [dcm3c/femm-qt](https://github.com/dcm3c/femm-qt), tag `v5.0.20261101`, verified source commit `7c31ccfa3d4bcda8123d822cdda31fce929f672e`.
- Host: GitHub Actions `ubuntu-24.04`, Python 3.12, native `femm_headless` and `belasolv` compiled from upstream source.
- Real tests: `test_real_femm5_charged_empty_shell` **PASSED**; `test_real_femm5_mesh_and_outer_domain_stability` **PASSED**. Log states **2 passed, 0 skipped**. Python-only tests are separate and do not count as numerical evidence.
- Experiment: one isolated conducting spherical shell, inner radius `0.10 m`, outer radius `0.12 m`, total charge `1 nC`, no free charges elsewhere, uniform permittivity.
- Observables: conductor potential and total charge; potential and field at points inside cavity, in metal and outside.
- Numerical acceptance tolerances used: potential within 8%, conductor charge within 4%, exterior electric field within 15%; mesh/domain sensitivity within 8% of analytical target. These are *benchmark acceptance tolerances*, not a demonstrated general error bound.
- The boundary is finite and modeled using a mixed/Robin condition; this test checks several outer-domain and mesh configurations, not a full asymptotic proof of convergence.

The FEMM solution uses actual geometry, meshing, solve and FEMM post-processing; it is not substituted by the analytical backend. The two geometric shell surfaces share one conductor property. The reported inner/outer surface-charge split is inferred from Gauss's law, **not directly measured** as local FEMM surface-charge density.

**Not accepted here:** grounded-shell solver, central/off-center point charges, multiple conductors, arbitrary 3D, spatial surface-density maps, GUI, and production-grade process timeout/kill isolation. M2b should address grounded constraints and point-source regularization with new live numerical tests.

CI definition: [`femm5-live.yml`](../.github/workflows/femm5-live.yml). Release-pin changes require explicit new numerical acceptance.
