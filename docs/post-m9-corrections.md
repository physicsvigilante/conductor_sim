# Post-M9 correctness review

This is a bug-fix pass on the M9 baseline, not a new physics milestone. The
starting commit is `3ee186b883e6a85fae761f3e82706405941faf70`; its downloaded
source archive reproduced Git tree `b5a6ef8d9e42fa7884731e204dae0a29951afa13`.
All 231 existing non-FEMM tests passed before changes.

## Reproduced faults and corrections

1. **Near-surface analytical density.** The squared angular separation
   `a*a + d*d - 2*a*d*cos(theta)` rounded to zero for distinct source and
   surface positions (tested at a 1e-10 m gap with a 0.1 m sphere). The exact
   inner/outer image-charge APIs then divided by zero. Use the algebraically
   equivalent `hypot(a-d, 2*sqrt(a)*sqrt(d)*sin(theta/2))` and factored
   differences of squares. Apply the same factored numerator to multi-source
   density references. This does NOT certify FEMM accuracy near contact.

2. **Charge summation.** Running nested-shell charge subtotals lost a small
   residual when later large charges cancelled. Re-sum original source/net-Q
   terms with `math.fsum`, including FEMM's Gauss-law-derived surface totals.
   The central-shell reference also explicitly uses `fsum` for consistent
   behavior on Python 3.10 (rather than relying on newer `sum` compensation).
   The underlying physical inputs and FEMM-measured charges are not altered.

3. **Error masking and cleanup.** A failed FEMM solve, or an exception in a
   `with solution` body, was replaced by a secondary session-close exception.
   Both numerical backends now attempt session and workspace cleanup, preserve
   the original error/cause, and attach secondary diagnostics in
   `cleanup_errors` (also exception notes on Python 3.11+). Cleanup failure
   after a successful operation still raises. Cancellation also cleans up.
   This is NOT hard timeout/process-isolation support.

4. **FEMM input contracts.** `<None>` is FEMM's no-conductor sentinel and
   cannot be an authored conductor name in a numerical solve. Both backends
   reject it before launching FEMM; analytical labels remain solver-independent.
   The shell numerical sampler now uses the shared exactly-three-finite-
   coordinates validator instead of leaking TypeError/IndexError.

5. **Silently clamped settings.** The pinned FEMM implementation clamps arc
   segment size to 10 degrees and solver precision to [1e-16, 1e-8]. Our old
   accepted ranges incorrectly implied these settings would be honored.
   Requests outside the supported ranges now raise ValueError; existing
   defaults and tested settings are unchanged. This follows the pinned
   [FEMM source](https://github.com/dcm3c/femm-qt/blob/7c31ccfa3d4bcda8123d822cdda31fce929f672e/libfemm/py_elec_editor.cpp)
   (`ei_setarcsegmentprop`, `ei_probdef`).

6. **Sweep generators.** A one-shot iterable of field-line seeds was consumed
   by the first frame, leaving later frames without the requested traces.
   Materialize at most 97 seeds once, reject more than 96 before opening a
   solver, and reuse the bounded collection across frames.

7. **Default view bounds.** The default extent used only conductor radii and
   cropped out an exterior source. Automatic bounds now include active source
   distances and conductor extents, with margin. Explicit user bounds remain
   unchanged. A sweep still uses the first state's fixed plane; author an
   explicit plane for sweeps that change relevant geometry or orientation.
   Malformed meridian axes now fail with ValueError.

8. **Viewer readouts.** Moving into a plot margin or leaving the canvas now
   clears the previous point's probe. Formatting retains tiny nonzero samples
   rather than replacing all values below 1e-10 by a displayed zero. Raw
   exported samples are unchanged by formatting.

## Regression and acceptance scope

The first 32-case reproduction suite failed 23 tests against the original
source. The final regression file adds 38 cases; all 269 non-FEMM tests pass
locally. Chromium checks exercise plot-exit probe clearing and tiny-value
formatting alongside existing slider/profile/offline/mobile checks.

The numerical gate retains all 68 prior FEMM integration cases and adds one
real exterior-sphere export regression for automatic bounds, seed replay,
physical charge agreement and owned-session closure (69 live cases total).
Exact commit/workflow outcomes are recorded in the pull-request discussion;
local fake-session tests do not establish numerical accuracy.

This bounded review does not claim to find every possible defect. Supported
geometries, finite-source approximation, near-field density tolerances,
analytical-only force/energy and offline playback limitations remain as
previously documented. No conducting plane or scene graph was added.
