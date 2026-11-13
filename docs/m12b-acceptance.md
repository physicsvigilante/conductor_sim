# M12b — numerical and viewer acceptance record

## Status and source pin

**Physics-code candidate validated:** `3bb02faf53818b9796c42e8e880158ecd88cee49`
(the M12b compiler and regression changes). Verified on actual
`femm-qt v5.0.20261101`, upstream revision
`7c31ccfa3d4bcda8123d822cdda31fce929f672e`, built in GitHub Actions
Ubuntu 24.04 with headless `femm_headless` + `belasolv`.

* [Real FEMM 5 numerical workflow](https://github.com/physicsvigilante/conductor_sim/actions/runs/43350476985):
  **90 passed, zero skipped**, consisting of all **83 retained** M1–M11 live
  regressions and seven new `test_m12b_live.py` tests. Real measured conductor
  and source properties; no analytical fallbacks.
* [Python checks](https://github.com/physicsvigilante/conductor_sim/actions/runs/43350477020):
  **303 passed** on Python 3.10 and 3.12. **90 FEMM-dependent tests skipped
  only in ordinary CI**, which does not provide numerical validation.
* [Visualization/source export](https://github.com/physicsvigilante/conductor_sim/actions/runs/43350476983):
  green, including the M12a analytical three-state Chromium regression.
* The **real FEMM-produced** `live-visualization/parallel_plates.html`
  (three stored equilibria), matching JSON, Chromium screenshot `m12b.png`
  and browser assertions `m12b-browser.json` were archived by the live
  numerical workflow. Browser checked the actual numerical export for
  slider, plot toggles, both density profiles, probe, state reset,
  mobile overflow and no external requests/JavaScript errors.

These exact-head workflow links above cover the physics code before this
documentation-only acceptance commit. **Merge condition:** Python 3.10/3.12,
live 90-case FEMM and Chromium/source checks must remain green on the final
documentation head; record those final run IDs in PR #16 before merging.

## Measured finite electrode disks (illustrative real FEMM outputs)

The authored plates are parallel, face across `d=0.20 m`, and the FEMM
computational cylinder has `R=1.60 m` (default radial factor 8). Each
electrode disk's voltage and charge are measured separately using
`eo_getconductorproperties`. All source-core Q values likewise come from
the actual solver, and the independently calculated expectations are not
injected into numerical output.

| Authored state | Exact lower disk Q | FEMM lower disk Q | Exact upper disk Q | FEMM upper disk Q |
| --- | ---: | ---: | ---: | ---: |
| Empty slab, V_lower=0V, V_upper=20V | -7.120960355 nC | -7.120960361 nC | +7.120960355 nC | +7.120960361 nC |
| Both grounded, +1 nC at 0.065 m | -0.675000 nC | -0.675197017 nC | -0.325000 nC | -0.324802982 nC |
| Both grounded, +1 nC at 0.065 m and -0.5 nC at 0.14 m | -0.525000 nC | -0.524905548 nC | +0.025000 nC | +0.024905546 nC |

The numerical +1 nC source core measured +1 nC. In the mixed-sign case the
two cores measured +1 nC and -0.5 nC **individually**, not merely their
net +0.5 nC. The signed zero-total-charge dipole is separately tested and
retains both numerical source cores.

The reference identities use the artificial **insulating cylinder** side
condition and have finite area `pi R^2`: with
`C=eps*pi*R^2/d`, `Q_lower=C(V_lower-V_upper)-sum(q_i(1-h_i/d))` and
`Q_upper=C(V_upper-V_lower)-sum(q_i h_i/d)`.
The finite-disk capacitance `C=0.356048 nF` is **not** a finite fringe
capacitance measurement for two isolated physical plates.

## Scope, fidelity and independent checks

1. **Supported FEMM domain:** two facing, ideal parallel conducting
   halfspaces held at known voltages, zero or 1–16 active charge sites on
   one common plane-normal line; charges lie strictly inside the slab.
   Supports rotated/translated world-space axes and coincident-site
   cancellation. Nonaxisymmetric 3D point placements remain
   analytical-only and are rejected *before* launching FEMM.
2. **Independent tests:** exact uniform bias reference for charge-free
   sample V/E/disk Q and sigma; nonzero and signed source Q/plate Q,
   Gauss-law sums, sampled V/E against M12a truncated-image reference,
   one-sided local sigma, zero-net-charge pair, rigid transformation,
   numerical core size, mesh and R sensitivity, strict refusals.
3. **Viewer:** same pair of plane halfspaces, source masks, density
   profiles, potential contours and directed field traces. A new
   `CylindricalDomain` and side intersection enforce the real FEMM
   finite extent instead of M11's old spherical-domain mask. Playback
   switches among *precomputed static states*, not particle dynamics.
4. **Limits:** FEMM source spheres approximate ideal point charges;
   local E gradients are elementwise and near-face sigma depends on
   offset and mesh. The outer boundary forces zero *radial* flux at
   finite R; no uniform error guarantee for fields, no arbitrary
   3D numerical case, no finite-fringe plate geometry, no FEMM-derived
   force/energy, and no production process isolation.

**Critical distinction:** for ideal infinite biased plates, the source-free
uniform background has infinite integrated electrode charge. M12a correctly
uses `None`/n/a for those ideal infinite totals. The nonzero measured
charges reported here belong only to the finite FEMM electrode disks.
