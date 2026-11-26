# M44 uniform background field on a spherical conductor

M44 solves one connected metal body with a spherical outer boundary and any
number of strictly disjoint, sealed spherical cavities. The exterior and all
cavities have the same homogeneous linear permittivity, which defaults to
`EPSILON_0`. Held point charges may occupy the exterior or any cavity.

```python
from conductor_sim import Grounded, SphericalConductorProblem, UniformBackgroundField
from conductor_sim import solve_spherical_conductor

solution = solve_spherical_conductor(SphericalConductorProblem(
    1.0, Grounded(), center_m=(2.0, 0.0, 0.0),
    background_field=UniformBackgroundField(
        electric_field_v_per_m=(30.0, 0.0, 0.0),
        reference_point_m=(0.0, 0.0, 0.0),
    ),
))
sigma = solution.outer_surface_charge_density_at((3.0, 0.0, 0.0))
```

The background potential is $V_{bg}(\mathbf y)=-\mathbf E_0\cdot(\mathbf y-\mathbf r_{ref})$.
The reference defaults to the sphere center $\mathbf c$. The reference fixes
the zero of the imposed background. The total potential includes the conductor
response and the held charges. The exterior has this affine background plus a
decaying perturbation. `Grounded()` sets $V_m=0$, `FixedPotential(V)` prescribes $V_m$,
and `FixedNetCharge(Q)` prescribes the total free charge on the metal. An
explicit zero background field also permits nonzero `FixedPotential`.

## Field and charge

Let $\mathbf x=\mathbf y-\mathbf c$, $r=|\mathbf x|$, and $R$ be the outer radius.
In the exterior the potential is

$$
V=V_{bg}(\mathbf c)-\mathbf E_0\cdot\mathbf x\,[1-(R/r)^3]
  +(V_m-V_{bg}(\mathbf c))R/r+V_{Kelvin}.
$$

$V_{Kelvin}$ sums each exterior charge and its grounded-sphere image. For a
charge $q_i$ at distance $d_i$, the image has charge $-q_i R/d_i$. Thus
$Q_{outer}=4\pi\epsilon R(V_m-V_{bg}(\mathbf c))-\sum_i q_i R/d_i$.
The grounded, exterior-source-free example above has
$Q_{outer}=4\pi\epsilon R\,\mathbf E_0\cdot(\mathbf c-\mathbf r_{ref})$.
Each cavity face carries the negative of its enclosed free charge. The fields
inside the cavities depend only on the sources in those cavities. Their
potentials shift by the change in $V_m$. Metal has zero field, and potential at
either kind of wall is exactly $V_m$.

On the outer wall, the free charge density at outward normal $\hat{\mathbf n}$ is

$$
\sigma=\frac{Q_{outer}}{4\pi R^2}+3\epsilon\,\mathbf E_0\cdot\hat{\mathbf n}
 +\sum_i\frac{q_i}{4\pi R}
  \left[\frac{1}{d_i}-\frac{d_i^2-R^2}{|\mathbf y-\mathbf y_i|^3}\right].
$$

The cosine and bracketed terms each have zero surface integral. The dipole
induced by the uniform field alone, about the sphere center, is
`background_induced_dipole_c_m`, equal to $4\pi\epsilon R^3\mathbf E_0$.

## Wall observables and refusals

`outer_wall_electric_field_at` returns the exterior-side field
$\sigma\hat{\mathbf n}/\epsilon$. `outer_wall_pressure_at` gives the outward
Maxwell traction magnitude $\sigma^2/(2\epsilon)$ in pascals. These methods
and `outer_surface_charge_density_at` require an exactly on-wall Cartesian
point. `electric_field_at` raises `SingularField` at a wall. Field and potential
are singular at active point sites.

`outer_wall_force_n` integrates that traction over the outer face and returns
$Q_{outer}\mathbf E_0$ in the absence of active exterior point charges. It also
gives the exterior electrostatic resultant on the combined metal and enclosed
charges. The force on the metal alone with held cavity charges would additionally
require inner-wall traction. The API refuses the outer-face force with active
exterior charges. It also refuses the generic `force_energy` query, including
total energy in the unbounded background and point-charge self-energies.

Wall or metal point sources, touching cavities, multiple conductors, dielectric
layering and transient motion are outside this model. M44 classifies its input
coordinates exactly and checks range at the observable boundary. A nonzero
result that rounds to zero, an overflowing result, or an unresolved rounding
interval raises `UnsupportedProblem`. Omitting `background_field` uses the M42a
boundary condition and its conservative near-wall and near-source guards.
