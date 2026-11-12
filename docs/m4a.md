# M4a: multiple distinct charge sites in a spherical cavity

**Status: accepted on 2026-11-12 for multiple axial charge sites in one spherical shell.** See [M4a acceptance](m4a-acceptance.md).

## Composable authored model

Author a single `ElectrostaticProblem` with one `SphericalShell` and
as many `PointCharge` components as needed. For the FEMM 5 backend, the
nonzero source sites must be collinear *with the sphere centre*, up to a
1e-11 relative-radius floating-point tolerance. Any one point and the
shell centre define an axis. For multiple distinct points there must be
**one shared axis**; a line between two charges not passing through the
shell centre is *not* an axis of rotational symmetry.

The engine chooses a deterministic axis, combines exactly coincident
charges by summing their strengths, discards zero-net groups, and creates
one separate small charged spherical FEMM conductor for every active site
(up to 16 sites). Source names are internal, not physical conductor IDs.
No author should write per-source FEMM commands.

Each numerical source's radius is constrained by the configured regularization
fraction, the distance to the nearest source, and the closest cavity wall.
The artificial cores are non-overlapping and are excluded from sampling.
`solution.numerical_sources` returns the independent, FEMM-measured source
charges and core positions. The physical shell still appears as **one**
connected conductor in `conductor_results`. Its inner/outer *total* charges
are Gauss-law inferences using FEMM-measured conductor/source charges.

## General analytical reference

`MultiImageShellBackend` computes an **exact** reference for **any number of
point charges inside one spherical cavity in arbitrary 3D positions**,
not merely collinear configurations. It superposes Kelvin image pairs,
handles the centred source limit, and applies the common conductor potential
from the grounded or fixed-net-charge condition. It provides exact potential,
field and `surface_charge_density_at(surface, position_m)`.

This reference is deliberately more general than the axisymmetric FEMM
backend; passing an analytical solution does not license FEMM to solve a
non-collinear 3D configuration. Its conductor charge totals use Gauss's law.

## Live acceptance (passed for documented scope)

Compare `Femm5Backend` to `MultiImageShellBackend` for distinct axial
charge sites and grounded, isolated-neutral and nonzero-net-charge shells.
Verify FEMM-measured individual core charge, shell charge/potential,
cavity/metal/exterior potential and E, induced surface-charge-density
profiles, numerical sigma integration, world-frame rotation covariance and
source-core/mesh sensitivity. Retain all M2/M3 numerical regressions.

All finite-core and one-sided near-surface results have discretization
error. Test thresholds are acceptance tolerances rather than universal
accuracy guarantees. Reject non-collinear source sites and unsupported
physics before launching FEMM. Multiple *physical conductors* are M4b.
