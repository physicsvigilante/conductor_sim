# M5 acceptance scope and evidence

M5 adds a solver-independent offline visualization pipeline, not a scene graph or
a browser-time electrostatics solver. The reviewed implementation is PR #6. Merge
is gated on Python, real FEMM, and browser checks on the exact final head.

## Recovered baseline

The saved candidate b346e85655b2a9ede9ab093765d76efcfefe5b09 was recovered from
the exact source archive in GitHub Actions run 43221990033. Its archive digest
matched the connector artifact digest. Its three workflows succeeded:

- Python checks: run 43221999762.
- Real FEMM 5: run 43221999951; numerical export is the femm-visualization artifact.
- Visualization and source checks: run 43221990033, including archived source,
  analytical HTML/JSON, screenshot and Chromium interaction results.

The recovered source was rerun locally: 146 non-FEMM tests passed and the
Chromium check passed. Live FEMM was executed in GitHub Actions, not claimed
as installed or executed in this chat container.

## Final review additions

A real browser reproduced stale probe values after a stored-state change. The
probe now clears until a new point is sampled. Missing density profiles clear
the old provenance label, and custom-solver conductor results carry explicit
metadata instead of being called analytical by a backend-name heuristic.
The generic slice heading is now planar rather than meridional.

After these additions: 148 local non-FEMM tests passed; 35 FEMM-dependent cases
were deselected locally and must pass separately in the live workflow. Chromium
checks exercise frame pixel changes, probe reset, missing-profile reset, explicit
provenance, field-line toggle, density selector, hover values, no external network
requests, no JS errors, and mobile horizontal overflow.

Final exact-head workflow IDs and merge verification are recorded in PR #6.
No earlier successful workflow is used as a substitute for final-head acceptance.

## What the implementation verifies

Samples match the supplied solution. Unknown numerical errors propagate; no
failed FEMM solve is replaced by an analytic answer. Source interiors/outside
domains are masked as null, not zero. Directed bounded traces terminate at metal,
source masks, domain boundaries, zero fields, or out-of-plane fields. Exact
segment/sphere checks prevent trace steps tunnelling through thin metal.
Contours exclude invalid regions. Solver ownership is closed on success/failure.
HTML labels and script-embedded JSON are escaped. Fixed color scales across stored
frames do not alter the exported numeric values.

The added real FEMM cases check sampling against the same live FEMM solution,
sweep cleanup, and each signed core charge in a zero-net dipole. The dipole field
comparison uses mesh refinement without widening its original absolute tolerance.
The earlier 32 numerical regressions remain unchanged.

## Explicit limitations

Playback is of precomputed states, not continuous motion or browser-side solving.
Field lines are midpoint geometric traces with bounded steps, not adaptive
error-controlled trajectories or counts proportional to flux. Density profiles are
great-circle samples, not spherical integrals. FEMM point sources still use finite
charged cores; density still uses mesh/offset-dependent near-surface fields.
The visual work does not increase numerical precision or extend supported physical
geometries. Browser checks are Chromium checks, not full accessibility, touch, or
cross-browser certification. Production process isolation remains separate work.
