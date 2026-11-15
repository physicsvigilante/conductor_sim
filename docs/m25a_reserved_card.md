# M25a R2 — frozen postimplementation, pre-execution transfer recipe

**Do not execute this test before the final code is committed.** Source-families
and acceptance objectives were chosen in the preimplementation charter. Exact
numeric values are fixed here before the first reserved run (a narrower claim
than preimplementation freezing of the entire recipe).

This is a *same-developer, author-exposed* teacher hypothetical, not an
independent or blinded LLM authoring test.

- Source `ring:reserved`: uniformly charged **fixed thin ring**, +2.0e-12 C,
  radius 0.15 m, centre (0,0,0), axis (0,0,+1). Only axial fields qualified.
- Source `field:fixed`: uniform fixed E=(0,0,+0.01) V/m.
- Particle: q=-1.0e-9 C; m=1.0e-9 kg; r0=(0,0,0.12) m;
  v0=(0,0,-0.01) m/s; normal to motion plane (0,+1,0), length scale 0.3 m.
- Gravity (0,0,-0.03) m/s2. Stationary-medium linear drag b=3.0e-10 kg/s.
- Scheduled applied E: phase 0 at t=0, E=(0,0,+0.04) V/m;
  phase 1 at t=0.23 s, E=(0,0,-0.07) V/m.
- Duration 0.7 s, output sampling 0.05 s, relative tolerance 1e-10.
- Investigation: first downward detector crossing of z=0.10 m, normal +z,
  direction -1, nonterminal, EventPolicy max_step=.01 s, time tol=1e-9 s,
  position tol=1e-10 m, velocity tol=1e-8 m/s.
- Independently integrate the true axial ring Coulomb formula, work by each
  component, and scheduled fixed-step RK4 over *exactly separated switch phases*.
  Compare position, velocity, work, candidate event time and continuity.
- Real Chromium acceptance: ring depicted as world-z-axis edge-on projection;
  no invented off-axis field; fixed uniform-field glyphs and source legends;
  source toggle alters painted canvas; standard controls, 390px layout and no
  JavaScript errors or external requests.

No parameters may be retuned after seeing the first run to manufacture event
success. If a genuine physical/precision issue is discovered, report it and
freeze a *new, separately identified* candidate rather than rewriting this
card's exposure history.
