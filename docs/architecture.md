# conductor_sim architecture

`ElectrostaticProblem` is an immutable collection of physical charges,
whole connected conductors, and electrical constraints, expressed in SI
world-space 3D coordinates. It is independent of FEMM and any viewer.

```text
LLM-authored typed Python models
  -> geometry/symmetry capability validation
  -> ElectrostaticBackend (analytical or FEMM 5)
  -> FieldSolution: V(x), E(x), measured/predicted conductor properties
  -> procedural visualization (M5 candidate)
```

## Composition

- `PointCharge` components may be combined without authoring FEMM commands.
  Same-position charges combine exactly. FEMM accepts up to 16 distinct active
  sites only when they lie on a **common line through the shell centre**.
- Multiple physical `SphericalShell` components can be composed with
  independent `Grounded()` and `FixedNetCharge(Q)`. FEMM requires concentric
  nonintersecting shells, up to eight, and all sources in the innermost cavity.
- For FEMM, each active site becomes a small separately charged spherical
  numerical core. These are not physical model conductors and are excluded
  from interior field sampling.
- One FEMM conductor property controls BOTH faces of each physically connected
  metal shell. FEMM computes each physical conductor's potential and net charge.
  Inner/outer integrated surface-charge results are subsequently inferred using
  Gauss's law; local density is separately estimated from near-surface numerical E.
- World-space sampling is projected to FEMM's axisymmetric `(r,z)` plane
  using a deterministic common axis. Non-axisymmetric arrangements are
  rejected, not silently approximated. The FEMM solver remains optional and
  is used in headless TCP mode through the FEMM 5 binding.
- Numerical source core radius, mesh scale and outer computational boundary
  are tunable for convergence/sensitivity experiments.

## Independent physics

`ConcentricShellBackend`, `ImageChargeShellBackend`,
`MultiImageShellBackend`, and `NestedShellBackend` compute independent
analytical reference solutions in their supported domains.

`NestedShellBackend` solves the coupled radial shell charge equations,
including mixed independent grounded/fixed-charge conductors, with
Kelvin images used for fields inside the innermost spherical cavity.
Unlike axisymmetric FEMM, the reference can superpose charges at arbitrary
3D locations in that cavity.

## Verification

Each incremental acceptance milestone has pinned real FEMM 5
numerical tests **separate** from mocked command-contract tests.
See [numerical evidence](m4b-acceptance.md) for the latest set and its
limitations. FEMM numerical acceptance uses finite mesh/source-radius
tolerances; it is not an analytic error bound.

**M5 candidate:** solver-independent slice sampling, contours, bounded field-line tracing, surface profiles and self-contained HTML sweep playback. See [M5](m5.md).

**Not implemented:** a full scene graph, genuinely general 3D FEMM backend,
eccentric/nonconcentric conductors, sources in gaps or outside shell nests,
and production-grade isolation/timeouts for untrusted authored problems.
