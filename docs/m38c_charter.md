# M38c — Stabilize genuine M38b playback acceptance after failed main-push CI (frozen maintenance charter)

**Trigger and truth:** independently reviewed M38b PR #92 was merged with exact-head safeguard as `44e0c50f3258452d112ab7a16b87759063054832`, whose native Git tree `547b2206f35b43beb8e0c38c461f5c73364cf88b` exactly matches the R2 independently reviewed source. Actual postmerge `main` workflow [#44471869184](https://github.com/physicsvigilante/conductor_sim/actions/runs/44471869184) **FAILED** in the M38b Chromium playback acceptance check. Its saved source and independent Kelvin SI were genuine. The failure was `#m38b-current-marker.cy` not matching a separately sampled moving `selectedIndex` after clicking Play. Preserve this original red run and [issue #93](https://github.com/physicsvigilante/conductor_sim/issues/93); do not call the M38b release fully green yet.

## Physical/integrity boundary

The browser uses `setInterval` every 100 ms to display **already Python-computed** M38a snapshots. The current `tools/check_m38b_line_browser.py` tester reads `window.m38bViewer.selectedIndex`, then separately performs many Playwright DOM calls; the live player can move to the next frame between those calls. Thus the test can compare a genuine new current marker with an older selected record. This is a *concurrent verification* defect in the checker, not authorization to loosen physical tolerances or change M38a Kelvin laws.

## Minimal corrective contract

- Modify only `tools/check_m38b_line_browser.py` and narrowly scoped regression tests, not the actual `src/conductor_sim/prescribed_line_viewer.py`, source PDF, native Kelvin/charge/force equations, current sign, original M37b viewer, or the independently checked HTML/CSS canonical defense.
- **First witness real Play** by waiting for a saved index strictly greater than zero; then **atomically** pause the existing presentation timer using the slider's official `input` handler with the *current* index unchanged. Only then inspect the full timestamp, world coordinates, SI and scene/chart marker values for that one frozen physically computed frame. Assert no index advancement across the verification. Test more than one actual play transition, including after Restart, to prevent this race from silently recurring.
- Retain complete genuine browser JS-off original-frame inspection, mobile/HTTP/JS error checks, canonical whole-HTML authenticity and **26 genuine HTML-only hostiles**. Do not weaken any visual proof or accept a merely stale marker or an invented frame.
- A typed/standalone SI validation must continue independently of the browser. This patch is a test synchronisation fix, not a simulation change; a valid negative-q teacher world must retain the same qualified checks.

## Release and review-cost boundary

- Commit this charter **as a separate parent before code**. Keep next PR draft and unmerged, preserve the failed post-merge M38b run plus any new red attempts, verify same scientific model source bytes, run focused real Chromium + adversarial + Python 3.10/3.12 and full exact-HEAD CI.
- Request at most **one narrow fresh-context independent review** only of playback race correction after developer CI green; reuse existing R2 evidence for unchanged source physics and canonical page-integrity boundary. For a narrow test-only timing correction, one review is sufficient; no broad extra physics review.
- Separate explicit owner authorization before merging the hotfix. Use expected-head guarded merge; verify identical reviewed Git tree on main and **all actual main-push CI**, especially M38b workflow, before declaring M38b completely released. The original failed postmerge run remains historical evidence.
