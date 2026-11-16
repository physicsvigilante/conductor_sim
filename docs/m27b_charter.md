# M27b — Off-axis ring particle dynamics under certified planar symmetry (frozen charter)

**Base:** independently reviewed M27a merge `4a967f469df999bc4a97c434dfdd6b69f49893ba` on `main`.
**Risk:** R2, because admission of a new physical motion domain affects
trajectory validity and singularity guards. This charter is frozen
**before implementing production changes**; no physics source-law revision
is authorized. Developer-selected cards are not a blind JEE coverage study.

## Objective and physical definition

Enable classical motion of **one separately authored test charge** (nonresponsive
fixed sources) under the existing real 3D `UniformRingField` E/V law and
superpositions with established built-in sources, genuine uniform electric
fields and optional uniform gravity; reuse the existing vector adaptive
RK4, `F=qE+mg`, `U=qV+Ug`, event investigations, drag/scheduled-field
driven investigations, immutable scenarios and structural reassembly.

Do **not** create a fake guide, force the trajectory onto an axis,
project away normal forces, insert browser-side ODE integration or infer a
mass/initial condition from a symbolic source statement.

### Qualified physical motion planes

A static ideal circular ring is reflection-invariant under:
1. Any **meridian plane containing its symmetry axis and centre**
   (`axis dot plane_normal=0`);
2. Its **equatorial plane** (ring plane) through the centre, whose
   normal is parallel or antiparallel to the ring axis.

Only these are accepted for each new full-ring source individually. The
ring centre must genuinely belong to the physical motion plane, to strict
geometry/roundoff tolerance based on its radius and world coordinates,
not an arbitrarily enlarged numerical `length_scale_m`. Validate the
**entire source inventory**, particle launch velocity, gravity and every
applied scheduled-field phase. Existing M21c/M22c reflection-pair point/rod
qualification and legacy axis-only source rules stay intact.

Paired off-plane rings and their more general collective reflection
symmetries are deliberately not introduced in M27b. A **new full-ring type
mixed with legacy axis-only ring/disc types is refused** until a separate
whole-system axis-contract proof: a genuine general off-axis trajectory
cannot be fed into an axis-only field. All original legacy-only trajectories
must remain admitted and unchanged.

### Ideal wire guard and numerical completion

`UniformRingField.clearance_at` is the 3D distance to the ideal thin
charged-wire singularity, NOT a material collision radius. The existing
test-charge guard must enforce launch clearance and sampled-stage clearance.
A conservative **whole chord** exclusion lower bound, using the 1-Lipschitz
distance-to-ring set, must prevent accepted RK step chords from leaping
through a wire within the declared guard. If resolution is inadequate,
subdivide or terminate/refuse; never cross a pole or report a successful
duration after a source guard. This is a numerical chord/stage guarantee,
not a theorem about unsampled exact trajectories.

The full physical E (and qE+mg if active), not its planar projection, is
integrated. The original transverse-force residual runtime guard remains
active. Preserve true signed force, electric/gravitational energies,
source guards, elapsed-time samples and event-outcome censorship. No new
relativistic, contact, absorption, rebound, moving-source or induction law.

### Truthful viewer and authoring

Reuse M24a event viewer and M24b driven viewer with the existing qualified
real-time samples. Extend the shared M25a source-geometry ledger **only for
the new UniformRingField**; show the true orthographic 3D rim, stable
ring IDs, source axes and whether the qualified physical plane is meridian
or equatorial, not a charged rod. Do not silently alter old source geometry
entries, historical HTML/JSON, source labels or viewer semantics.

M19b's old standalone prescribed-source viewer supports its historical
geometry/axis contract. If it cannot accurately display a new full-ring
arrangement, it must **explicitly refuse** it rather than omit the ring or
invent axial-only profiles. The opt-in M24a/M24b viewers are the supported
M27b playback path. M27a static observation remains available separately.

## Frozen positive and novel cases

- **P1:** One world-z ring, meridian x–z plane, oblique launch off axis.
  Compare initial full E and energy at independently quadrature-checked
  points and compare saved motion to a step-halved independently implemented
  velocity-Verlet reference; no force projection.
- **P2:** One world-z ring in its equatorial x–y plane, oblique launch
  away from the ideal wire; verify field normal zero, actual curvature and
  parent model stability.
- **P3:** Two identical-axis but physically separated fixed charged rings
  plus a transverse uniform field, in a genuine common meridian plane.
  Verify actual transverse displacement and conserved `K+qV`; source
  inventory IDs and full E preserved.
- **P4:** Add optional uniform gravity in the same qualified plane and
  check separate electric and gravitational energies; no doubled qV.
- **P5:** Same base model with two independently specified typed event
  questions (passive detector, velocity reversal), retaining the
  original physical problem, and render saved real-time positions in
  Chromium; no invented event after a guard.
- **P6:** Driven model with linear drag and multi-phase uniform applied E,
  all phases preserving the same physical plane; check switch left/right
  accelerations, physical work/dissipation and full input provenance.
- **P7:** M26a stable-ID structural reassembly can add/replace an
  off-axis full-ring source and requalifies all phases with no
  problem-specific production path; invalid changes refuse.
- **U1 (unfamiliar transfer):** Rigidly rotate+translate the source
  arrangement, launch and physical plane together, then compare
  world-covariant trajectory, field, forces and energy at saved times.
- **W1 workbook-authored geometry:** `02_conductors_half_size.pdf`
  physical p12 linked Q40–41 has Ring A/B axes world x, ring charges
  +10 μC / −20 μC, radius .30 m each, separation .40 m, and +y applied
  electric field 10^5 V/m. Preserve these **source-given** values.
  A real charge moved freely in world x–y may acquire +y displacement:
  specify test-particle charge, numerical mass, initial position, velocity,
  duration and guards as **explicit teacher hypotheticals**.
  Do not claim this executes the original Q41's unspecified mass or that
  the horizontal x axis is a mechanical guide.

## Frozen negative and boundary cases

- N1: Ring axis oblique to proposed plane (neither meridian nor
  equatorial) or centre off-plane, even when a locally zero transverse
  E happens at launch; refuse before integration.
- N2: Incompatible gravity, a uniform source E normal to plane, or a
  **later scheduled phase** with transverse E refuses complete model.
- N3: Mixed legacy axis-only ring/disc with full ring refuses, while
  original legacy source results and schemas remain unchanged.
- N4: Initial point within wire guard refuses, and an approaching
  trajectory can terminate at `source-guard`; no crossing/contact,
  no post-guard events or duration-complete misreporting.
- N5: Coarse numerical step whose chord nearly intersects the ideal
  wire is subdivided/refused using a conservative distance bound;
  no false assertion that a straight chord is the true curved path.
- N6: Source ID/geometry, qualified plane, saved E/energy, event/motion
  record and geometry-bound imports cannot be silently forged; no
  untrusted HTML is accepted as a validated simulation.
- N7: Unsupported old M19b viewer path must refuse new source rather
  than fabricate a source drawing; old viewers unchanged by default.
- N8: Rotation/axis normalization and near-singularity numerical
  imprecision must fail closed, not silently relax the physical plane.

## Acceptance and evidence

Distinguish authored physical model, globally certified geometry,
actual RK completion/guard, independent E/V and alternative integration
checks, event outcomes, viewer correctness and source provenance.
Independent mathematical checks must not rely only on the production
elliptic evaluator; use separate direct ring-line Coulomb quadrature for
sampled fields and an independently implemented time-integrator for
at least one genuine off-axis trajectory. All source guards, energy
drift, normal force and motion-plane residuals remain visible.

Reserve first-run transfer evidence before execution. Use bounded standard-
library testing, no mandatory new dependency. Run complete Python 3.10/3.12,
97 real FEMM conductor nonregression tests and real Chromium offline
playback checks of a **new off-axis curved trajectory** plus existing
views. Preserve earlier numerical/HTML fingerprints and original failures.
Any first failed execution must be recorded; never redefine the frozen
cards retroactively.

After exact-head source/CI freeze, require one independently authored R2
ChatGPT review of the physical admission, wire guard and actual native
motion before separate owner authorization for merge. No merge is implied
by the user's approval to begin M27b. Unbounded free 3D trajectories,
off-axis disc fields, induced conductor responses, cross-ring collective
symmetry, actual contact and current waveforms remain separate work.
