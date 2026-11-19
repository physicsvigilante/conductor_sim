# M38a — Qualified 3D straight-line prescribed motion near one grounded sphere (frozen charter)

**Prerequisite released base:** M37b PR #90 merged to main as `f187dede846d441dd5928ccb6521d67443ead72b`; identical independently reviewed tree `44b754e005c3d720cb3e59c1debd363ca2d8d639`. Exactly 22/22 post-merge main-push workflows succeeded. Freeze this charter as a distinct commit BEFORE any implementation.

## JEE relevance, source honesty, and physical distinction

The unchanged `02_conductors_half_size.pdf`, physical p8 Multiple Correct Q21, gives a **radially approaching, externally maintained constant small speed** positive source near a grounded solid sphere. M37b already reproduces its signed grounding current and trend in a verified original-source viewer. **Do not recast that original exercise as an off-axis charge path.** M38a teaches a new *teacher-authored, transferable extension* of the same Kelvin apparatus: a charge crosses along a straight 3D line at prescribed constant Cartesian velocity, possibly passing closest to the sphere and reversing the sign of the conventional current. It is **not** a new independently sampled original JEE question or an upgrade to the frozen 30-case coverage fraction.

The one continuous, differentiable physical trajectory is
```text
x(t)=x0+v*t,    0 <= t <= T
r(t)=x(t)-c,    d(t)=||r(t)||,    v_r(t)=(r(t).v)/d(t)
Q_sphere(t) = -q*a/d(t)
I_sphere_to_ground(t) = -q*a*v_r(t)/d(t)^2
t_closest = clamp(-((x0-c).v)/||v||^2,0,T)
```
The zero at an interior closest approach is a **true kinematic radial-velocity zero**, not a numerical turning event or a conductor-current relaxation law. The source continues moving linearly, although its Kelvin force is reported. A physically unspecified external actuator maintains that path; it is not the M16a free-charge orbit. For tangential motion at an instant the **net ground-wire current is zero** but local induced surface charge may redistribute. Do not claim microscopic currents, radiation, resistive-wire current density, electrical power dissipation or circuit transients.

## Minimal reusable qualified authoring contract

- Introduce a new typed `PrescribedSphereLineSpec` and `sample_prescribed_sphere_line` in a separate module. Take an unchanged `ElectrostaticProblem` containing exactly one nonzero external `PointCharge` and exactly one stationary physically **solid** `SolidSphere(Grounded())`; accept real 3-vector Cartesian velocity, genuine prescribed duration, bounded sample count and a positive surface approach guard. Preserve stable sphere name, original immutable problem and SI coordinates. Reject Boolean/nonfinite raw velocity/time/guard and source values at the new boundary, unsupported electrostatic constraints and zero/relativistic speed. Use the existing M37b raw-authorship factory where untrusted raw coordinates must be checked *before* legacy constructors normalize them.
- Check the entire **continuous line segment**, not just endpoints or saved frames: compute the constrained interior closest approach and reject `min d <= a+guard` with a conservative floating-point separation margin. Cases whose stored frames are all outside the metal but whose between-frame path hits it must be refused. Reject numerical underflow, overflow and inadequately resolved world-space translation rather than fabricate equal positions. No collision model, reflected path or post-contact extension.
- For each saved exact prescribed time, create a fresh immutable point-source position and invoke the **unchanged** public M37a `observe_grounded_sphere_wire` (Kelvin charge + native force/energy). Do not add a competing solver, re-integrate Newtonian motion, reuse M37b's radial-only spec, or compute charge/force/energy in the browser. Report both signed terminal currents, radial speed, distance, world geometry, source force and induced mechanical interaction energy in a new versioned JSON schema.
- Ensure result/export integrity at a typed boundary. An altered native sample or Boolean timestamp must not be accepted merely because Python treats `False == 0.0`; recheck the qualified input and authentic saved observations, with strict types. An in-process recomputation check is not a cryptographic external signature. Original p8 Q21 remains an existing distinct M37b source answer; mark all off-axis examples `teacher_hypothetical`.
- No new renderer yet: preserve the verified M37b **radial** source page rather than falsely depicting a 3D off-axis line on its radial schematic. A later separate M38b can provide an honestly projected 2D/3D path and current-versus-time viewer, with real Chromium/no-JS integrity checks. M38a provides native data for that layer.

## Prespecified acceptance and independent references

1. Genuine radial inward/outward line motions reproduce the existing signed M37b Kelvin current, charge, force and energy at matching times; neither history is a Newtonian solve.
2. An off-axis line across the near side has strictly positive ground current on approach (q>0), **zero** at the true closest-approach instant, and negative current while receding; q<0 flips signs. Include exact analytic `t_closest` and smallest separation in output, regardless of whether the closest instant is a saved frame.
3. A deliberately hidden between-sample intersection with the metal/guard must be rejected; just-outside near tangent accepted only when its full continuous path is demonstrably clear. Include shifted/rotated source arrangements, positive/negative charge, endpoint/near-endpoint closest approaches and nonzero initial tangential velocity.
4. An independently written **stdlib-only** reference checker, with no production physics imports, recalculates position/time, radial velocity, closest approach and all saved `Q=-qa/d`, `I=-qa v_r/d^2`, induced Kelvin force `-k q^2 a d/(d^2-a^2)^2` and interaction energy `-k q^2 a/[2(d^2-a^2)]`. Test forged signs, current, geometry, sample order, time, provenance, units and world-space/source identifiers. Verify native `-dQ/dt` using independent nearby static Kelvin solves when far from guards.
5. A small real Python-authored example exports the actual teacher off-axis line and source-specific evidence as JSON. Dedicated Python 3.10/3.12 CI runs native and reference checks, preserves literal exact Git commit/tree and test evidence. Full historical Python/FEMM/Chromium suites stay mandatory at the exact PR source HEAD. No browser image claims for a JSON-only milestone.

## Release gates and review cost control

Implement on the same new branch only AFTER this charter commit is frozen. Keep one scoped draft PR unmerged. Run source-focused tests and full existing CI first, preserve any genuinely failed attempts, and request **one** fresh-context independent JEE/physics/evidence-integrity review only after all developer checks are green. The independent reviewer should attack an unfamiliar off-axis path and a between-frame crossing, rederive current/force independently and verify source-versus-teacher labels; do not re-review unchanged M37b original radial browser/physics from scratch. A separate owner decision and expected-head guarded merge, matching reviewed Git tree on main and successful actual postmerge CI remain mandatory. No release without those gates.
