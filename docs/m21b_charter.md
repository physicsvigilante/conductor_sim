# M21b — Real JEE Advanced electrostatic potential/field visualization

This is an approved small, JEE-focused observer/viewer milestone based on
the source-anchored M21a audit. Its evaluation cases and refusal contracts
are frozen **before** writing the new implementation.

## Three source anchors
1. Official JEE Advanced 2021 Paper 1, Physics linked Q9–10:
   source `https://jeeadv.ac.in/past_qps/2021_1_English.pdf`,
   PDF page 4 (index 3), fixed charges -Q at (0,0) m and
   +Q/sqrt(3) at (2,0) m. The exact V=0 locus is the circle centered
   at (3,0) m with radius sqrt(3) m. The original Q is symbolic;
   any illustrative numerical value of Q must be labelled.
2. `02_conductors_half_size.pdf`, physical PDF p15, archived IIT-JEE
   2012 thin uniformly charged *nonconducting* spherical shell question
   (the E(r)/V(r) graph). Preserve V continuous at r=R, E=0 inside,
   E~1/r² outside and **E undefined on the ideal surface**.
   Do not mislabel this as induced conducting-shell charge.
3. Same scanned workbook p12, linked Q40–41, uniformly charged
   rings A/B of radius 0.30 m separated by 0.40 m, charges +10 and
   -20 microcoulombs, test charge +10 microcoulombs from A to B,
   external E=1e5 N/C upward transverse to the horizontal ring axis.
   Q40's field work is +3.595 J approximately. This is endpoint work,
   not evidence of a free axial trajectory; Q41's mass is symbolic.

## Scope gates
- Reuse existing `PrescribedField.potential_at` and
  `electric_field_at`. No new electrostatic law or ODE integration.
- A **physical** 2D slice can display sampled V equipotentials and E
  lines only for a certified symmetry-preserving source inventory.
  Mask point singularities rather than assign V=0 or bridge them.
  Prefer adapting existing M5 sampling/contouring/canvas renderer.
- Ring/disc formulae remain axis-only, never used off-axis.
  Show radial/axial profiles from actual field evaluations, with
  nulls/gaps at singularities and jumps. No interpolated bogus E
  at the shell surface.
- Source IDs and labels are author-supplied and preserved; source
  quantities and bounds must be reproducible JSON.
- Compute endpoint work from source potentials
  W_field=q(V_A - V_B), with per-source contributions and genuine
  fixed external uniform fields. Do not infer time, mass, or guide
  constraints.
- No universal scene graph, extra arbitrary field engine, browser ODE,
  conductor boundary solver, conductor induction or time-varying E.
- Regression: correct charge signs/positions, reference circle, masked
  source vicinity, field gradients and invariance; exact 2012 radial
  jumps; source-faithful two-ring work including **zero transverse-field
  contribution** on horizontal displacement and explicit nonzero
  perpendicular force; correct unsafe-request refusals; original viewer
  outputs unchanged by default.
- Acceptance: independent references and genuine offline Chromium
  views for each case, zero JS errors, no external requests and readable
  graphs/geometry. Standard repo Python/FEMM workflows must remain green.
  State clearly which pieces are source-faithful versus illustrative.
