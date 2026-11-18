# M37a — Grounded-sphere wire-current observer (frozen charter)

**Base:** released M36e `dc347251c14e449c53d51820b4481d1bfbee2cfb` (native Git tree `60b4f21aa88a2031ec50c5d1e7c15a5135a7becf`). Freeze this charter in a standalone commit **before implementation**. M36e's actual postmerge main-push CI is a separate release gate.

## Source-backed educational gap

The provisional M36a JEE-30 audit calls **`02_conductors_half_size.pdf` physical p8, Multiple Correct Q21** (charge approaching a grounded sphere) only **PARTIAL**. M7/M13 already calculate static Kelvin response, M16a actual radial Newtonian motion and M16b playback. Their existing results do **not** yet report the signed **ground-wire current** from the varying net induced charge. Do not count this as a solved original paper item until its exact printed question/figure and requested quantities have been checked independently.

## Actual qualified physics

One **nonzero** point charge `q` at `r>a` outside one fixed **grounded solid spherical conductor** of radius `a`, homogeneous linear permittivity, static perfect equipotential `V=0` at each state. No cavity, extra charges/conductors, isolated net-Q sphere, finite response time or non-quasistatic electromagnetic claims.

Let `d = |x-c| > a`, `n=(x-c)/d` and the actual instantaneous radial velocity `v_r=v·n`. Existing `ExternalGroundedSphereBackend` must supply the **actual** static induced net charge `Q_sphere = -q*a/d` and existing `force_energy` must supply the **actual** source's force and *induced* mechanical energy. Derive the wire current solely by differentiating the moving Kelvin equilibrium:

- `dQ_sphere/dt = (q*a/d**2)*v_r` (conventional charge flowing **from ground into sphere**, signed).
- `I_sphere_to_ground = -dQ_sphere/dt = -(q*a/d**2)*v_r` (opposite signed direction). These are **ideal charge-flux rates**, not microscopic conduction velocities or a resistive circuit.
- Inward `q>0,v_r<0` implies `Q_sphere` becomes more negative, so **positive conventional current sphere→ground**; negative source reverses current direction. Tangential velocity induces changing surface density but **zero net current through the ground wire** at that instant.

Expose **two distinct authoring modes** without confusing them:
1. **Instantaneous prescribed approach:** source/teacher supplies an actual velocity vector. Report a local derivative/current at this state; **do not invent a time trajectory** or assume constant speed/acceleration.
2. **True Newtonian trajectory:** consume a **genuine `Motion1DResult`** from the existing `integrate_motion_1d` supported radial sphere configuration, retaining its input mass, real initial conditions, elapsed timestamps, energy ledger, termination and surface guard. At each saved actual state derive `I(t)`, while re-solving the existing static Kelvin equilibrium; a frame is **not** a fake dynamical conductor PDE.

No new solver or RK integrator, no FEMM dynamic-current claim, no collision/contact, no fixed-Q sphere wire current, no vector-current interpretation for local induced surface redistribution. This is quasi-electrostatic, nonrelativistic, no radiation or wire resistance/inductance. A moving charge induces a changing surface profile even for tangential movement, but the grounded sphere's **net** `Q` depends only on `d`.

## Expected public work product

A small independent observer module, public exports, minimal example(s), JSON schema documenting units/sign/origin, tests, and one dedicated source-pinned CI lane. **No viewer/animation yet**; retain existing M16b visualization, plan truthful synchronized ground-current overlay separately. Keep current under an explicit `observed` derivation rather than silently inserting it into old M16/M13 schemas.

## Acceptance / adversarial proof

- Independent analytic `Q=-qa/d`, force `F_r=-kq²ad/(d²-a²)²`, induced mechanical `U=-kq²a/[2(d²-a²)]` with mandatory half factor; analytic `I=-qa v_r/d²`. Native Kelvin charge/force/energy must match.
- Separate numerical derivative check on **position-varying static equilibria**, with central finite differences at safely interior distances, and `I=-dQ/dt` (not `-d(qV)/dt`).
- Positive and negative q, inward/outward, tangential-zero, translated/rotated axes and charge sign, signed actual M16a motion states and genuine surface-guard termination, no post-guard sample.
- Refuse wrong constraint (isolated), shell proxy, inside/touching source, multiple sources/metals, zero q, nonfinite/malformed vector, initial/source-state inconsistency, forged motion velocity/position, field-energy/current substitution, and unreliable claimed original-PDF attribution.
- Verify distinct **prescribed-state** vs **actual elapsed-time** metadata, stable conductor identity, pure offline/export behavior, and all unaffected historical tests.
- Independent fresh-context reviewer must author unfamiliar tests against **exact reviewed HEAD + native Git tree** and read actual p8 PDF/figure before claiming question completeness. No automatic merge. Owner separately approves, expected-head guarded merge, matching native Git tree on main and all actual postmerge workflows SUCCESS.

**Scientific ceiling:** a new signed grounded-wire-current *observable* and a genuine **already-supported** motion-derived trace, not new general conductors, circuits, moving boundaries, self-consistent Maxwell transients or a newly solved original JEE item by default.
