# M27a — off-axis static electric field of ideal uniformly charged rings: frozen acceptance

**Base:** verified M26a `main` merge `923ac2c49b23439cc9c76604bfd2a945e984d221`. This charter is committed
**before production implementation**. The task is bounded new electrostatic physics,
not broader scenario/scene-graph refactoring. Developer-authored test cards are
not a blinded model-authorability or representative JEE Advanced benchmark.

## Physical definition and API boundaries

Introduce a **new**, immutable `UniformRingField` built-in fixed-source type:
an infinitesimally thin *nonconducting*, uniformly charged circular ring with
total signed charge Q, physical radius a, arbitrary 3D centre and 3D axis.
Its `potential_at(p)`, `electric_field_at(p)`, `clearance_at(p)`, and
`as_dict()` obey the existing SI `PrescribedField` protocol with homogeneous
permittivity. E is the true negative gradient of V in free space outside
the charged wire. Charge never redistributes due to the probe or to another
fixed source. Allow source superposition with other built-in fixed fields,
including a transverse uniform field. Support positive and negative charges,
axial coordinates, general off-axis locations, translation, rigid orientation
changes and linear permittivity scaling.

**Preserve legacy** `UniformRingAxisField` as strictly axis-only, including
existing input/output, exceptions and serialized contracts; do not silently
extend or weaken its motion qualification. M27a makes **static** 3D E/V
observations only. A `TestChargeMotionSpec` containing the *new*
off-axis source must be **explicitly refused** for time integration until
a later qualified M27b motion-plane/guard extension. No new ODE/integrator,
force/energy ledger, event search or browser-side solver is authorized here.
No conductor induction, free-moving source, disc physics or off-axis field
evaluation from the legacy axial-only class.

Use a documented, stable analytical or numerically certified complete-elliptic
integral formulation (standard library only); no introduced mandatory runtime
dependency. At the ideal wire, V and E are undefined/divergent: raise a
typed `SingularField` rather than returning 0, a finite regularization or
a computational error. Near-wire numerical stability, on-axis limits,
axis reversal, near-axis radial cancellation and floating-point representability
must have explicit guards or verification. A `clearance_at` value is
distance to an ideal field singularity, not physical collision/metal contact.

## Static teaching observer and drawing

Provide an opt-in 2D **static observation slice** of the real 3D E and V
for new full-ring sources and already-supported uniform E, with honest
physical world axes, source IDs, source outlines/projections, displayed
out-of-plane field-component information, singularity exclusion and stored
E/V numbers. The slice is not a mechanically enforced trajectory plane;
an off-plane E component may be present, and cannot be silently deleted
in the physics. Display generated precomputed physics, not browser field
solving, future motion or uncomputed field lines. Avoid modifying existing
M21b/static or M19b/motion export schemas as a hidden migration.
Reusable source and observation APIs should be independent of the
original workbook's exact numbers.

## Preselected positive cards and independent checks

- **P1** — recover the original axial `kQ/sqrt(a²+z²)` and
  `kQz/(a²+z²)^(3/2)`, including both sides and centre, for arbitrary
  axis orientation, signed Q and permittivity.
- **P2** — off-axis (`rho` both inside/outside the ring, z signed) E
  and V agree with an *independently written direct Coulomb line integral*
  (not with the new elliptic implementation), including no spurious
  azimuthal field. Check `E=-grad V` by finite differences.
- **P3** — rotate, translate and reverse the physical ring axis;
  numerical potentials invariant under rotation and E vectors covariant.
  Verify negative charges invert V/E.
- **P4** — compose **two actual horizontal world-x rings** plus upward
  `UniformElectricField`. Evaluate genuine off-axis E/V on a physical xz
  observation slice; reveal transverse electric force without claiming an
  on-axis free trajectory. Original workbook parameters, if used, must be
  identified separately from chosen view/sampling inputs.
- **P5** — a second unfamiliar multi-ring/point/uniform-E assembly uses
  the unchanged `PrescribedField` superposition and stable source names.
- **R1 (reserved)** — frozen parameter recipe: ring
  `Q=-2.3e-9 C`, `a=.17 m`, `centre=(.08,-.06,.11) m`,
  `axis=(2,-1,3)`; observe at `(.155,.047,.227) m`, permittivity
  `1.35*EPSILON_0`; compare full E and V to direct 1D circular-line
  quadrature and a finite-difference gradient. This is author-exposed,
  not a blind, independently authored transfer.

## Negative and boundary/refusal cards

- **N1** — exact and poorly resolved wire singularity raises
  `SingularField`; near-wire checks cannot silently return finite zeros,
  NaN or infinite fields.
- **N2** — invalid charge/radius/axis/centre/permittivity, overflow and
  physically indistinguishable/poorly resolved ring geometry fail closed.
- **N3** — legacy axial-only ring still refuses off-axis evaluation and
  existing saved/axis-only trajectories remain unchanged.
- **N4** — new 3D field does NOT imply qualified particle dynamics:
  `TestChargeMotionSpec` must reject it before any run.
- **N5** — a masked static grid cell must not be painted as a trustworthy
  physical zero or interpolated through; display must separate real 3D E
  from the 2D arrow projection. An unsupported source/viewer combination
  refuses rather than distorting the original source.
- **N6** — badly forged static source identity, mismatched saved E/V or
  artificially added physical-time data fails validation.

## Acceptance

Freeze source physics references before production calculations; keep
reserved R1 first-run evidence and initial errors if any. Test independent
Coulomb quadrature, K/E or finite-difference relationships, symmetries,
false motions and source masks, negative cases, model purity and old hashes.
Require complete Python 3.10/3.12, actual FEMM 5 nonregression and real
Chromium offline acceptance of at least the two-ring transverse-field
example. All accepted rows must distinguish physical qualification,
independent numerical validation, observer correctness, visible static
education and optional exam-answer completeness. No claims of automatic
PDF interpretation. A new R2 independent ChatGPT reviewer must examine
the final exact source SHA/tree and new physical law before owner approval
for any merge. No merge until separately authorized.
