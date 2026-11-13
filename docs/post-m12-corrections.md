# Post-M12 correctness review

## Baseline and evidence

Reviewed main commit `77208fa4c8e485557a55bae77cb975b6bd0502e5`.
Downloaded its source artifact, verified artifact SHA-256
`a91239ac954cf2034035c1b8e8e5a64aa224a35d79daad47d78234d2055ec054`,
and reconstructed Git tree `80a90430aea81ed8c7167a0c6b5b4c3a69996d46`.
All **303 existing non-FEMM tests passed** before edits. A separate set of
27 regression checks then produced **23 failures and 4 passes** on that
unchanged baseline. This is a bounded review, not proof that every possible
problem or input is correct.

## Corrections

1. **Analytical cancellation:** the grounded-plane image potential used a
   subtraction of almost equal reciprocals. For a +1 nC source at z=0.15 m,
   it returned exactly zero at (1e8,0,0.07) m instead of approximately
   1.887385876e-25 V, and at (0.05,0,1e-18) m instead of approximately
   6.821072221e-16 V. The tangential field could also incorrectly vanish.
   Rationalize using `r_image^2-r_source^2=4*h*h_source`, including a stable
   reciprocal-cube difference for E. Tests independently evaluate the
   original image formula with 70-digit Decimal arithmetic. The physical
   model, image charges, force/energy conventions and existing tolerances
   are unchanged. This is not an unlimited-range floating-point guarantee.
2. **Near-start geometry intersections:** sphere, plane and cylindrical-side
   trace barriers discarded valid positive intersections at t<=1e-10.
   Keep every root in (0,1] and use stable quadratic roots/factored distance
   differences for spherical and cylindrical boundaries.
3. **Automatic planar views:** use an active charge's normal-line footpoint
   rather than an arbitrary plane anchor for the default meridian. A second
   source's transverse direction selects a useful slice where possible;
   genuinely non-planar fields still stop tracing, not project silently.
   Bounds use perpendicular distances to other planes, not distances to
   physically irrelevant tangential anchor coordinates.
4. **Analytical display exclusions:** compute active grouped sites before
   choosing masks. Bound masks by clearance to every conductor surface and
   neighbouring active site. A canceled near-wall site no longer shrinks
   unrelated masks, and a charge just beneath the upper plate no longer
   gets an exclusion disc crossing into that electrode. These are display
   masks, not physical source cores or new solver approximations.
5. **Empty capacitor field lines:** automatically seed the vacuum gap when
   two planes have no source seeds. Real E determines direction and stopping;
   a zero field still produces no traces. No imaginary charges are added.
6. **Wide numerical profiles:** add typed `OutsideDomain(UnsupportedProblem)`
   for finite planar FEMM domains. Density samples outside the mesh, including
   a one-sided probe just beyond the hemispherical edge, are exported as JSON
   null. The viewer leaves gaps rather than drawing false zeros or connecting
   across missing data. Unexpected solver errors still propagate. An entirely
   excluded profile is labeled as unavailable in this view.
7. **Plane-parallel slices:** mark an entire slice inside a conducting
   half-space as metal even when no boundary line intersects that slice.
   Previously the region data said metal but the renderer omitted the fill.
8. **Cleanup and export contracts:** preserve a primary sampling error even
   when cleanup raises KeyboardInterrupt, and retain cleanup diagnostics using
   the existing shared resource helper. The parallel-plate example now actually
   writes the advertised adjacent JSON file; earlier M12 documentation claimed
   a separate JSON artifact although the example only wrote embedded HTML data.

## Validation and merge gate

Local Python 3.13: **330 non-FEMM tests passed**, including all 27 new
regressions. No existing numerical tolerance was relaxed. Chromium passed the
three-state analytical export checks and separate tests for automatic capacitor
lines, sparse/all-null profiles, true gaps, parallel-slice metal fill, mobile
layout, and zero JavaScript errors/external requests.

Three additional actual-FEMM regressions are included in `test_review_live.py`:
two wide-profile cases (single plane and plate pair), and a translated
single-plane default meridian. The dedicated workflow must pass **all 93
actual FEMM tests with zero skips**, while Python 3.10/3.12 and source/browser
checks must pass on the exact final candidate. Final run IDs and results are
recorded in the pull request; local mock-client tests are not live numerical
solver evidence.

## Limits retained

No new physics milestone: arbitrary 3D FEMM, finite-plate fringing, particle
dynamics, production process isolation and a general scene graph remain out of
scope. M12a's charged-slab image series is still finite-order, not an exact
or uniformly error-bounded field. M12b still uses the explicitly artificial
insulated finite cylinder and measured finite electrode-disk charges.
