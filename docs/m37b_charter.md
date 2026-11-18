# M37b — Source-faithful constant-speed grounded-sphere current explorer (frozen charter)

**Freeze before implementation.** Base is *merged* M37a `133acf97a123e6167aeaac3ca2a2b2094af1d36f`, native Git tree `c728041ca5118013c6979d120c8805f31ee09fa0`, independently reviewed M37a source HEAD `73d7daee9f31f6feee3e6305205aa28abc3e48af`. Real M37a postmerge main-push CI remains a separate verified release gate.

## Genuine original JEE-source anchor and distinct physics protocols

The independently visually inspected, unchanged `02_conductors_half_size.pdf` **physical p8, Multiple Correct Q21** depicts **a positive external charge q** approaching **a grounded metallic sphere of radius R** at **externally maintained constant small speed v**. The source's p17 answer key gives **(1),(3)**: conventional current direction in the grounding wire and increasing magnitude. These exact original-source observations were established by M37a independent R1/R2/R4 (retained in review_sandbox); do not confuse the teacher's illustrative numerical values with original symbolic givens. This is a workbook exercise, not an asserted official JEE past paper.

The M37a source physics is already independently qualified:
```text
d(t) = d0 - v*t,   with q>0, v>0, d0 - v*T > R > 0
Q_sphere(t) = -q*R/d(t)
I_sphere_to_ground(t) = q*R*v/d(t)**2    (positive conventional charge flow)
dI_sphere_to_ground/dt = 2*q*R*v**2/d(t)**3 > 0
```
A speed supplied by the original problem is a **prescribed physical kinematic input**, not a force-determined Newtonian trajectory. The externally maintained speed requires an unmodelled support/actuator exerting counterforce; its mechanics, electrical source, resistive wire, radiation and microscopic electron drift are outside scope. M16a's freely accelerating orbit is **not** an authentic replay of this original question.

## Minimal reusable authoring and display contract

- A frozen, typed `PrescribedSphereRadialSpec` takes one actual `ElectrostaticProblem` with one signed nonzero external `PointCharge` and one fixed **grounded `SolidSphere`**, plus a finite **signed radial velocity** `v_r`, prescribed elapsed `duration_s`, discrete `sample_count`, and a strictly positive source/surface approach guard. Initial radial axis derives from actual immutable authored geometry. Only radial fixed-speed motion is qualified, inward or outward, and the entire segment must remain outside metal plus guard. Speed is nonrelativistic and no cancellation/zero-speed route is faked.
- At every **actual prescribed time** use `d(t)=d0+v_r t` and a fresh immutable point-source placement to invoke public `observe_grounded_sphere_wire`. Preserve the solved signed native `Q_sphere`, `I_sphere_to_ground`, opposite terminal sign, physical force and induced mechanical energy. **No new electrostatic law, solver, RK4, FEMM time model or browser-side physics.**
- Preserve original electrical source name and centre, numerical units, signed q and v, real positions/time, one source apparatus, and explicit `prescribed_constant_speed` status. Distinguish source-given **symbols/signs/protocol** from every **teacher-chosen numerical input**. Allow teacher hypotheses with q<0 or outward motion, without falsely counting them as additional original JEE problems. A standalone JSON export has a strict schema with all relevant physical inputs; it is not a claim of cryptographic provenance.
- Display: **one simple, offline, dependency-free, accessible HTML/SVG viewer** that plays or selects exclusively precomputed snapshots. A fixed, honestly projected sphere/point/ground schematic, a current-versus-**prescribed physical time** graph, and signed numerical readouts remain synchronized. No HTML/JS/Canvas field computation, no M16a ODE, no fake current flow simulation in wires. A **JavaScript-disabled original-first-sample fallback** remains meaningful. The authentic original symbolic question and answer options (1),(3), together with unambiguous *teacher numerical example* labels, must be visible.
- A source-specific example and an **independent standard-library-only physics/export checker** must recompute sampled time/geometry, expected conventional current, induced charge, force/energy and option direction/trend from Gauss/Kelvin formulas. It must refuse forged signed q, radius, v, time, sample order, current direction, geometry, provenance, units and impossible guard claims. Also test independent native static-Q finite differences, translation/rotation invariance and a negative-q/outward teacher hypothetical for composability; keep original-source fixture q>0,inward.
- Browser verification: real offline Playwright Chromium checks embedded vs saved independently validated JSON, visible original-vs-teacher attribution, selected frame readouts, chart marker, fixed physical scene and no-JS fallback. Screenshot evidence retained. Import/export validation is **not** established by a browser screenshot alone.

## Release gates without redundant reviews

1. Freeze this charter in a **separate parent commit**, before source code.
2. Create implementation and test source branch/draft PR, run new dedicated source-pinned Python 3.10/3.12, independent stdlib physics checker, real offline Chromium, plus all unchanged full Python, real FEMM and historical browser regressions. Keep failed attempts historically red.
3. **One fresh-context independent SOURCE-AND-UI review only after developer-controlled tests are all green and changes frozen.** Reviewer may reuse M37a independently verified Kelvin mathematical results if unchanged, but must execute novel q/v/current/metadata/viewer attacks against exact PR HEAD/TREE. Avoid generic extra reviews if targeted fixes suffice.
4. No owner-independent merge. Require explicit separate owner approval, `expected_head_sha` guarded merge, identical reviewed source Git TREE on `main`, and actual post-merge main push CI success.

**Ceiling:** One original workbook apparatus with an authentic externally prescribed constant-speed protocol and recomputed static conductor observations at authored times, not a general LLM autonomous parser, a new field solver, a true transient Maxwell conductor, a released original JEE-30 coverage percentage, or a physical driven-wire circuit simulation.
