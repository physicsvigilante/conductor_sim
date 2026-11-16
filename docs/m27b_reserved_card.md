# M27b — first-run reserved transfer card (frozen before candidate CI)

**Purpose:** a *single*, developer-exposed previously unexecuted
test of the shared off-axis full-ring planar-motion API. Distinct from
the workbook two-ring world-x/y case and from primary development cards.
Do not change inputs or acceptance bounds to obtain a pass. Preserve the
first CI result, failures and any later corrections. This is not a blind
LLM-authorability test.

## Physical SI model

The physical plane is x–z: `plane_normal=(0,1,0)`.
Components are fixed, nonresponding ideal charged rings:

| Stable physical ID | Source | Total charge | Radius | World centre | World axis |
| --- | --- | --- | --- | --- | --- |
| `ring:alpha` | `UniformRingField` | +3.0e-9 C | 0.16 m | (0.05,0,-0.07) m | (0,0,1) |
| `ring:beta` | `UniformRingField` | -1.5e-9 C | 0.11 m | (-0.1,0,0.12) m | (1,0,0) |
| `field:held` | `UniformElectricField` | — | — | — | (0.5,0,-0.3) V/m |

Gravity `g=(0,0,-0.12) m/s²`, teacher-hypothetical.
Probe `q=-4.0e-10 C`, `mass=3.0e-7 kg`, initial position
`(0.08,0,0.04) m`, velocity `(0.1,0,-0.015) m/s`,
duration `0.07 s`, saved physical sample interval `0.007 s`,
numerical `relative_tolerance=1e-9`, `length_scale_m=.4`,
source guard `.008 m`.

The source ring axes are **nonparallel** yet each lies within the same
physical xz meridian plane, so complete source reflection symmetry is
provable without a collective paired-ring model.

## Independent reference and acceptance

1. Instantiate through public `PrescribedField` + `TestChargeMotionSpec`
   only (no reserved-case production physics).
2. Independently form ring positions with trigonometric world-basis
   parameterization, sum Coulomb contributions of 512 midpoint charges
   per ring at every reference RHS calculation. Include held uniform E
   and exact uniform gravity separately. Do not call production
   `UniformRingField.electric_field_at` inside the reference.
3. Integrate a separate velocity-Verlet trajectory using **700 equal
   timesteps**. Compare the final position to within `3e-6 m` and velocity
   within `2e-5 m/s` (componentwise) against the unchanged production
   vector RK4. Compare initial full force with independent Coulomb
   line integral to `rel=5e-8` or `abs=1e-12 N`.
4. The real run should terminate `duration`, keep normal world-y
   position/velocity/force within `1e-10`, report finite values, save
   genuine t=0 and t=0.07 s samples and conserve `K+qV+Ug`
   with maximum absolute drift less than `5e-12 J`.
5. Explicitly serialize immutable source IDs, true sources and teacher
   provenance, observed runtime and independent reference errors. A pass
   is not an event-location proof, external source authentication,
   contact simulation or JEE-source-given worked answer.

This card is sealed by its first commit to the M27b source branch,
and its first run must be associated with the exact source head and
GitHub Actions job/artifact. No first-run reclassification on correction.
