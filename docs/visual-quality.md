# Smooth potential and progressive field-line density

This is a visualization update, not a new electrostatics solver or time-domain
simulation. It reuses the physical models, numerical meshes and field APIs.

## Potential display

The viewer interpolates **scalar potential before mapping to colour**. Ordinary
vacuum cells use bilinear interpolation. At planar metal interfaces it clips
boundary triangles to the vacuum and uses the stored conductor voltage on the
boundary. It does not blur the finished picture or invent additional precision.

Null samples are not zeros and are never replaced by their nearest valid
neighbour. A stencil intersecting a source mask or spherical metal interface is
rejected. This also protects thin shells or masks contained inside a grid cell
whose corners are all in vacuum. Conservative grid-scale blank areas can remain
near unresolved curved boundaries and source masks; increase export resolution
rather than filling those gaps with fictitious data.

The shared sign-preserving **asinh colour scale** uses the largest stored
potential magnitude (including conductor voltages), not the old 95th-percentile
saturation limit. Blue remains negative, neutral remains zero and red remains
positive. Five legend ticks invert the same mapping used by the raster. The
colour transform is explicitly labeled and does not modify stored SI values,
contour levels, probes or any solver output. Only the current frame's raster is
cached, so density changes do not resample or repaint the potential pixel grid.

## Field-line density

The keyboard-accessible slider selects Sparse, Light, Medium, Dense or Full:
15%, 30%, 50%, 75% or 100% of the available stored pool (rounded up). A numeric
count is shown. Higher settings add curves without moving or replacing existing
ones. The selection persists across stored states and is disabled when field
lines are hidden or no traces are available.

Automatic seed families are generated around **each active signed source** and
on each visible conductor face. A deterministic dyadic sequence fills angular
or lateral gaps; round-robin selection prevents one source from consuming the
96-seed budget. Nearly identical complete paths are suppressed without moving
retained paths. An empty capacitor uses one face to avoid duplicate copies of
the same uniform-field lines. All traces use the existing live solver field and
its existing geometric/out-of-plane termination rules. Line count is **not a
flux measure** or a quantitative comparison of source magnitudes.

An explicitly authored `seeds=` iterable still overrides automatic seeding;
`seeds=[]` means no traces. Older visualization.v1 bundles can be rendered
without mutating their stored data, but a new renderer cannot manufacture extra
traces from an old fixed pool. Regenerate an export to obtain the new pool.

The parallel-plate example now uses the common automatic policy rather than ten
fixed midpoint traces, with a 61 by 61 default grid. All sampling/tracing still
happens at export time. Browser controls do not run FEMM or solve physics.

## Acceptance

`tests/test_visual_quality.py` exercises conservative stencils, exact linear
fields, boundary values, mask/thin-shell exclusion, progressive distribution,
explicit seeds, cancellation, old bundles and unchanged numerical samples.
`tools/check_visual_quality.py` exercises actual browser pixels, nested density
levels, keyboard/state persistence, raster caching, nonlinear legend ticks,
null/domain exclusion and 320/390-pixel layouts without network requests.

The existing Chromium suite also tests density controls on its actual exports,
including the FEMM-produced capacitor. Two new real FEMM tests exercise the
progressive pool and interpolation metadata for one plane and a plate pair.
Required gates are the full ordinary suite, full browser workflow and the
pinned real-FEMM suite with no skipped numerical tests. Check the PR acceptance
comment for exact source pins and executed counts; local mock tests alone are
not FEMM numerical evidence.
