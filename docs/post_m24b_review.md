# Post-M24b correctness review and bounded public-interface trial

Baseline: `5eb12091839e1325278a40b0927d2fb56a81bae8` (M24b accepted main).
This is a corrective pass and teaching-model authoring check, not a new physics
milestone or a claim of an exhaustive error-free repository.

## Reproduced defects and the bounded fixes

1. **Uncertain terminal event cluster rejected in driven motion.** Two detector
   planes at x=.11 and .1100000001 m, crossed at about 1 m/s, can have overlapping
   numerical root brackets and slightly different representative times. The
   shared locator correctly retained both and marked order unresolved, but the
   driven validator rejected the later representative as outside the stopped
   interval. The event viewer also inserted every such representative into
   playback. Both errors were behind the earlier validation failure.

   A shared horizon check now permits only an explicitly order-unresolved
   candidate within time tolerance whose bracket overlaps the retained terminal
   bracket. Ordinary saved states remain strictly inside the stopped interval.
   The driven viewer excludes alternative states after the retained stop from
   playback, retaining them only in the labelled event evidence table. It does
   not choose a physically exact priority or extend the trajectory past a stop.
   This aligns with the conservative event viewer's existing playback policy.

2. **Incomplete conservative event-observation validation.** The M24a event
   validator checked total force and total energy but ignored displacement,
   electric potential and separate electric/gravitational force and energy
   contributions. Corrupted values or undeclared gravity could survive export.
   All these observations are now re-evaluated from the retained physical model;
   absent contributions must remain absent. Checks of source guard, full vectors
   and original scalar precision are preserved, not loosened.

3. **Boolean event numeric metadata.** Python equality allowed False to stand
   for a zero residual and True for a time of 1 s. Both conservative and driven
   event validators now require finite real numeric brackets/residuals, excluding
   booleans. This is strict saved-data validation, not a change to root finding.

None of these repairs changes an electrostatic force law, integration formula,
source guard, physical qualification, schedule, work state or numerical tolerance.
Only `motion_events.py`, `driven_motion.py` and `driven_viewer.py` change in
production; other production files retain their previous bytes.

## Reproductions and preservation

`tests/test_post_m24b_review.py` contains the targeted counterexamples, controls
and public-interface checks. The pre-fix run had **35 failed regressions and two
passing preservation/boundary controls**. A still earlier run also included a
review-test-only decimal multiplication equality error; it was changed to a
1e-14 relative check and retained in the local first-attempt evidence. No
production tolerance was changed to make a regression pass.

The initial corrective set is 37 parameterized outcomes; three public-interface
checks add two authored families and one expected refusal. Existing M24a/M24b
regressions remain required. Twelve earlier examples (five M24a, five M24b, one
static dipole and one rotating dipole) were regenerated under the old and revised
source: all **24 HTML/JSON files are byte-identical**. The affected previously
failing near-coincident export is new acceptance evidence, not a rewritten
historical success.

## Continue with a bounded authoring trial

The exact cards were recorded before their first execution. Both use public
Python model/force/schedule/event/revision interfaces; no additional production
code was written for either arrangement.

| Card | Authored model and requested investigations | Observed scope |
|---|---|---|
| T1 `mirror-rod` | Two equal held point charges mirrored across y=0, an in-plane finite rod, uniform gravity, linear drag, three applied-field phases. Ask x reversals and return through the initial-x plane; separately ask only about reversals; derive a doubled-drag child. | All three runs reach the requested finite duration. No requested event is located in that interval. The independent numerical reference agrees; absence over all time is not claimed. |
| T2 `coaxial-rings` | Two differently signed coaxial rings, axial gravity, drag and two applied-field phases. Ask x reversals and crossings of x=.17 m; separately ask only about reversals; derive a doubled-drag child. | All three runs reach duration. The detector is crossed in the baseline and child. No reversal is located in the tested interval. |
| B1 | Revise T2's second field phase to include a transverse component. | Rejected during physical qualification, before integration. The existing axial-only ring law is not bypassed. |

The same immutable model object supports both query inventories. A drag change
creates a child with retained parent and computed changes, not a new special-case
solver. Trial hypotheses do not force a desired motion outcome by tuning source
values after seeing the results. All values are teacher hypotheticals, not
purported past-paper reproductions.

**Exposure:** this is a preselected, same-developer public-interface transfer
check, not an independent or blinded LLM authoring benchmark. The author has
prior implementation knowledge. The first scripts and execution outputs are
retained. No claim of classroom effectiveness or statistical JEE coverage follows.

Independent local reference code uses direct point-charge/ring formulas, Gauss
line quadrature for the rod, and separate DOP853 state/work integration. Those
checks are distinct from byte-preservation checks. SciPy is a reference tool,
not a new runtime or test dependency. CI independently rechecks source-force sums
using direct formulas and midpoint line quadrature, plus existing analytic
schedule/work/event references.

## Observation limits and actual browser checks

The near-coincident detector preview verifies actual final playback time,
unresolved-order notice, event table, canvas path, controls, 390px layout, and
zero JavaScript/console errors or external requests. Both authored trial viewers
use the retained driven-browser checker. These are separate from numerical
verification. Detector separation may be below visual pixel resolution.

The driven viewer is an explicitly labelled projection, not a full apparatus
renderer. Coaxial-ring sources are identified in the retained source inventory
but ring geometry is not drawn by this viewer. Thus T2's numerical investigation
passes, while a complete ring apparatus depiction remains an observation gap.
Nothing is silently replaced by an off-axis ring field or a fake source shape.

```powershell
git pull --ff-only
python -m pip install -e ".[test]"
python -m pytest tests/test_post_m24b_review.py -q
python examples/public_api_event_trial.py --case mirror-rod --output output/review-mirror-rod.html
python examples/public_api_event_trial.py --case coaxial-rings --output output/review-rings.html
Start-Process .\output\review-mirror-rod.html
```

The existing observation workflow is extended; there is no new permanent CI
workflow. Historical transfer steps are now labelled as retained regressions,
not repeated first executions. Ordinary Python matrices and actual FEMM remain
nonregression gates; FEMM does not independently validate the event/drag physics.
