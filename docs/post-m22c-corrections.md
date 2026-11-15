# Post-M22c — observation and saved-data correctness fixes

Baseline: `22eb9f398a76f3223efa13a869dd799b7f31f01f`, tree
`9da0dbb5e7896fcd5bdd99b30b3c736f4e6ed37e`.

This is a bounded corrective review, not a new physics milestone or a claim
that the entire repository is error-free. The user mentioned an older review
but no separate review text/attachment was available with the request; these
findings were independently reproduced on the current source. No earlier
review is silently substituted or claimed to have been verified.

## Four confirmed failures

1. **Saved off-plane rod indices accepted booleans.** Python considers
   `[False, True] == [0, 1]`, but the browser's index lookup uses strict
   equality. A corrupted record could pass validation and then draw an
   off-plane rod at a fictitious projected position. The saved omission list
   must now contain actual integers and match the reconstructed inventory.
   Legitimate M22c geometry, reflection certificates and outputs are unchanged.

2. **Static-field validation accepted incomplete E vectors.** The old `zip`
   comparison ignored absent/extra components, including an empty vector.
   Every unmasked E observation now requires exactly three finite real
   components; potential values reject booleans/non-numbers/nonfinite data.
   Region row widths are also checked before indexing. This is consistency
   validation, not authentication of arbitrary imported results.

3. **Defined work was blocked by an undefined endpoint E.** The observer
   calculated finite endpoint potentials but then raised at the electric-field
   jump on an ideal shell/disc surface. Work remains `q(V_start - V_end)`.
   An unavailable optional endpoint field is now `null`, with an explicit
   `unavailable_endpoint_fields` explanation and visible caveat; it is never
   zero. Undefined potential and off-axis ring/disc requests still fail.
   This does not authorize dynamics across the source guard or conductor
   induction, and does not supply a mechanical trajectory.

4. **A spatial plot bridged jumps missed by its sampling grid.** For example,
   a shell radius of 0.101 m with 21 samples from 0 to 0.24 m has no sample
   on its surface. The old graph joined the inner zero E to the exterior E
   with a spurious slope. Profile intervals now use existing geometric
   intersection/distance services to identify shell/disc jumps and point/rod
   poles. Optional per-row `break_before` keys prevent plotting across those
   intervals. Original samples and their values are retained: no artificial
   sample or null measurement is inserted. A continuous shell/disc potential
   is not interrupted merely because E jumps. An exactly sampled undefined
   point already supplies a gap and needs no new metadata.

The helper separates sampled quantities from safe connecting lines. It is not
an exact event locator, a continuous-field certificate or a replacement for
adequate spatial resolution. Geometric distances retain floating-point
resolution limitations; a plotted gap is not a new physical exclusion radius.

## Preservation and prospective correction

No electrostatic law, ODE controller, symmetry-admission rule, conductor
backend, numerical tolerance, motion source guard, initial condition or
historical evidence was changed. Correct existing outputs keep their meaning.
Previously accepted malformed data is now rejected; previously unjustified
observation failures and misleading plot connections are corrected.

The three accepted M21b exports (2021 circle, 2012 shell, ring work) and two
M22c exports (parallel and tilted rods) were regenerated before and after the
fix: their HTML and canonical numerical/audit JSON SHA-256 values match
exactly. These fingerprints establish preservation, not independent physics.

## Verification and reuse

Thirty-five new regression cases cover the four defects, signs, surface-work
reversal and superposition, missing fields at both endpoints, unsupported
geometry, unsampled double shell crossings, point/rod poles and smooth-source
controls. On the unmodified baseline, 24 failed and 11 passed. All 35 pass
with the correction. Independent work expectations use the shell/disc
potential formulas and declared physical constants, not a copied result.

Local ordinary Python baseline: 1,394 passed, 97 FEMM-dependent skipped.
Local corrected ordinary suite: 1,429 passed, 97 FEMM-dependent skipped.
Focused new + M21b + M22c tests: 102 passed. These counts overlap.
FEMM is separately required for retained conductor nonregression, not as the
physics reference for these prescribed-source observations.

The existing M21b workflow now includes the new regressions and two actual
Chromium examples. They check separate E polylines, continuous V, unchanged
sample counts, correct surface work, explicit null-field explanation, offline
operation and mobile width. The two new scenes are teacher hypotheticals,
not claimed examination questions or evidence of classroom effectiveness.

```powershell
python -m pytest tests/test_post_m22c_review.py -q
python tools/check_post_m22c_observations.py --output output/post-m22c
Start-Process .\output\post-m22c\unsampled-shell-jump.html
```

The correction is shared by all supported authored profiles and work requests;
no branch keyed to an exam question, example name or selected charge value is
added to production. Final PR CI status and exact source bindings are recorded
in the PR rather than inferred from local tests or screenshots.
