# M8 numerical acceptance — prescribed-net-charge isolated solid sphere

**Accepted 2026-11-12 for one nonzero external point source and one isolated
`SolidSphere` with `FixedNetCharge(Q)`, using the exact analytical Kelvin-image
reference and the pinned actual FEMM 5 solver.** The earlier grounded M7
baseline is preserved and retested.

## Real-solver and browser evidence

- [Real FEMM 5 integration on PR #10](https://github.com/physicsvigilante/conductor_sim/actions/runs/43253055381): **59 passed, zero skipped, zero failures**, including **7 new M8 numerical cases** and **52 retained M1–M7 numerical regressions**.
- FEMM 5 `v5.0.20261101` from [femm-qt](https://github.com/dcm3c/femm-qt) at verified commit `7c31ccfa3d4bcda8123d822cdda31fce929f672e`, built as `femm_headless` + `belasolv` on Ubuntu 24.04. This is a real FEMM solve, not a fake client or analytical substitution.
- [Python 3.10 and 3.12 PR workflow](https://github.com/physicsvigilante/conductor_sim/actions/runs/43253055994): **220 passed, 59 FEMM-dependent skipped** in the ordinary Python job. This *does not* replace the real numerical result above.
- [Chromium export/browser PR workflow](https://github.com/physicsvigilante/conductor_sim/actions/runs/43253055699): successful. The new four-state analytical viewer passed page switching, field/contour toggles, sigma profiles, hover/probe state reset, solver provenance, mobile horizontal-overflow check, no external requests and no JavaScript errors.
- Downloaded the **actual FEMM-produced four-frame HTML/JSON artifact** from the real solver run, and independently exercised its HTML in Chromium: four states; all controls passed; zero external requests and JavaScript errors. This is genuine FEMM-generated sampled data, not the analytical export. The numerical HTML is available as a GitHub Actions artifact during its retention window.

## Independently checked numerical sample

Problem: solid metal sphere radius `a=0.12 m`, external source `q=+1 nC` at `d=0.27 m`, homogeneous vacuum. Four distinct static equilibria using the **same** 
physical source position and geometry:

| Electrical constraint | Sphere net charge exact / FEMM (nC) | Exact sphere voltage (V) | FEMM voltage (V) |
| --- | ---: | ---: | ---: |
| Grounded, V=0 | -0.444444 / -0.447368 | 0 | 0 |
| Isolated neutral Q=0 | 0 / 0 | +33.2872 | +33.3707 |
| Isolated +2 nC | +2.000000 / +2.000000 | +183.0798 | +182.5573 |
| Isolated -0.6 nC | -0.600000 / -0.600000 | -11.6505 | -11.3853 |

The **fixed-net-charge input** is prescribed in FEMM, so equality of the net
charges alone is *not* validation. The decisive independent measurements are
the solved **nonzero metal voltages**, external potential and electric field,
and sign-changing near-surface sigma in the isolated-neutral configuration.
The new numerical tests cover fixed Q=0, +1 nC and -0.75 nC, plus two
boundary-condition comparisons, a neutral sigma near-surface angular integral,
mesh/domain/source-radius sensitivity and world rotation/translation.

In the exact independent reference, the Kelvin image
`q'=-q*a/d` and central compensator `q0=Q+q*a/d` together yield the prescribed
sphere net charge Q, and its equipotential equals `V=k*q0/a`.
The exact total sphere-on-charge forces for the four example states are,
in order: `-8.509e-8 N`, `-3.030e-8 N`, `+2.163e-7 N`, and `-1.043e-7 N`
(along the positive source axis). This exhibits attractive polarization of a
**neutral** isolated conductor and repulsion for a sufficiently positive Q.
**These force and energy predictions come from the analytical reference;
FEMM did not independently calculate force or energy.**

## Validation tolerances and limitations

- Numerical metal voltage: 16% relative plus 2 V absolute to the exact fixed-Q reference. Individual external V samples: 25% relative plus 3.5 V; fields: 40% relative plus 130 V/m; local sigma: 70% relative plus 2.5 nC/m². These are *benchmark acceptance criteria*, not claimed worst-case errors across the geometry.
- Physical conductor net Q: within 8% relative plus 20 pC absolute (although Q is an *input* constraint, not an independent prediction). Numerical source core charge: 5% relative. Neutral sigma angular integral: absolute residual smaller than 0.35 nC, from one-sided FEMM near-field samples (not direct boundary integration).
- Domain/core-size/mesh sensitivity compares three configurations with 26% relative plus 2.5 V sample tolerance and less than 22% relative spread. Grounded and M1–M7 real numerical regressions retain their original acceptance gates.
- FEMM uses a small charged conducting **source sphere** to approximate an external point charge; off-centre source spheres polarize and require convergence checks. Its finite outer Robin boundary approximates multipolar decay. High-precision near-contact behavior is not certified.

**Not accepted:** multiple distinct external charges, isolated sphere without
any external source, arbitrary solid shapes, nonaxisymmetric FEMM geometry,
charge transport, force/stress-tensor postprocessing directly from FEMM,
particle dynamics, a scene graph, or production-grade external-process isolation.
The visualization slider selects stored equilibria rather than rerunning the
numerical solver continuously.

See [M8 physics and authoring](m8.md) and the [real FEMM workflow](../.github/workflows/femm5-live.yml). Post-merge default-branch workflow links can be added to the PR's acceptance comment.
