# M26a — Driven structural reassembly: frozen acceptance charter

**Base:** M25a merge `c5cc9bf818f1c8d83901bcadef745cc29f6d1675` on `main`.
**Purpose:** close one missing *cross-layer authoring* seam, not add a new physical law.
This developer-selected trial is not a blinded LLM benchmark or representative JEE coverage.

## Scope and immutable boundaries

Extend the existing M22a stable-ID `reassemble_scenario` operation to a
retained M24b `DrivenModel`: atomically add/remove/replace fixed source components,
and add/remove the optional uniform-gravity component, with a complete
structural parent/child ledger. Reuse `ScenarioReassembly`,
`DrivenModel` qualification **for every applied-field phase**, and
`DrivenStudy`/`investigate_driven_motion` without modifying their
integrators, field/force laws, guards, event search, work accounts or HTML
viewer. The drag and the scheduled uniform E are preserved exactly in this
structural operation. Existing `revise_driven_model` remains parameter-only;
a new source ID always denotes a new component, not silent identity reuse.
Original inputs and committed historical exports must remain unchanged.

The direct parent must be `DrivenModel` or a previously returned
`DrivenReassembly`. Keep original driven and base records in ancestry.
Each new child model and its base scenario have explicit new IDs and
teacher-hypothetical provenance; retain ancestry to reject recycling retired
source IDs within the supplied chain. A structural reassembly is *not* a
numerical result. It never infers initial conditions, plane/axis changes,
applied switch, numerical tolerance, new detector, physical contact or source
response. A different mathematical investigation may reuse the resulting
model but must be authored separately.

No special per-card production code, arbitrary callbacks, graph compiler,
conductors in prescribed inventories, off-axis ring/disc motion, forced 3D
integration, browser calculations or implicit solver fallback.

## Preselected positive / transfer cases

- P1 — Add an in-plane fixed point source to a stationary-medium driven
  system with scheduled uniform fields. Independently check initial Coulomb
  force at the actual launch point and preserve schedule, drag, IDs and
  left/right switch observations.
- P2 — Add a complete reflection-matched off-plane pair atomically. Independently
  check the symmetry-cancelled transverse force; check an actual driven run.
- P3 — Add uniform gravity to a driven zero-source fixed-field experiment;
  independent vector exponential solution for constant E+g and linear drag.
- P4 — Remove a fixed source and gravity together, preserving the unchanged
  applied force and drag work semantics. Independent constant-field reference.
- P5 — Replace a held point with a fixed uniformly charged volume sphere
  under a new ID. Check finite interior Coulomb field against its independent
  Gauss-law expression and retain provenance.
- U1 — Introduce a coaxial charged ring into a qualifying axial-driven
  scenario with a longitudinal multi-phase schedule. Independent axial E/V
  at an admissible point and one accepted run.

All are declared teacher hypotheticals with SI physical inputs. The transfer
case U1 must use the exact same public authoring path after implementation.
The cases must *not* be misrepresented as sourced exam problems.

## Preselected negative / boundary cases

- N1 — Unpaired out-of-plane point or rod source refuses full-system symmetry.
- N2 — A ring whose axis conflicts with a *later* applied phase refuses even
  if the initial phase would qualify.
- N3 — A new source ID colliding with scheduled-field, drag, particle or
  other retained identity is refused; retired IDs cannot be recycled.
- N4 — Unknown, repeated, conflicting removals/replacements, or a no-op
  structural request is rejected without mutating the parent.
- N5 — A forged structural ledger that mismatches its parent or silently
  edits drag/schedule, particle, initial state, medium or clock is refused.
- N6 — Existing parameter-revision API still rejects changing source
  membership; no accidental acceptance of structural differences by
  parameter-only comparison.
- N7 — The qualified source guard can truncate a child motion; it is not
  converted into a successful duration-complete run.

## Evidence / acceptance

Separate outcomes for source/membership authoring, final whole-model
qualification, numerical completion/guard, independently checked physics,
and availability of event/visual observations. Correct refusals are
**not** successful simulations. Verify parent purity, stable source IDs,
complete schedules, retained model/base ancestry and detached audit JSON.
Include a separate real Chromium offline run for at least one newly composed
model if an available environment supports it. Before claiming acceptance,
require complete Python 3.10/3.12 suites, existing FEMM nonregression and
relevant browser checks. A separate independent second-ChatGPT review should
inspect the exact candidate source for provenance/identity and physics
boundaries before merge. No merger without separate owner approval.

Any amendments or first failed executions must be documented; do not
retroactively change this frozen charter or existing evidence.
