# Conducting drops in separate static equilibria

Workbook PDF page 14, Numerical Value Type Q1 gives 27 identical conducting-fluid drops of radius r and common potential V₁. It asks for V₂/V₁ after they combine, assuming unchanged fluid density. The source gives no numerical radius, charge, or voltage.

Model each initial sphere in isolation with its potential referenced to infinity. Assume no charge or material is exchanged with the surroundings, a common homogeneous permittivity, and a spherical final drop in static equilibrium. Initial mutual induction, the flow during contact, and drop stability are outside the model.

The composition takes independently charged solid spheres, sums their charges and volumes, and solves the final sphere with the same radial backend. For equal drops the final radius is 3r and the charge is 27 times the original charge. Since an isolated sphere has potential proportional to charge divided by radius, the nonzero-potential ratio is 9. A teacher may also supply unequal radii or signed charges; in that case each initial sphere has its own potential.

Independent checks must compare the result with charge conservation, volume conservation and the isolated-sphere potential, including a neutral final state and a composition performed in two stages.

## Reproduce the workbook ratio

The radius and charge below are teacher-selected SI values. Changing the sign of every charge reverses both potentials and preserves the ratio.

```python
from conductor_sim import FixedNetCharge, SolidSphere, combine_isolated_drops

one_drop = SolidSphere(radius_m=0.002, constraint=FixedNetCharge(2e-12))
equilibria = combine_isolated_drops((one_drop,) * 27)
initial_v = equilibria.initial[0].potential_at((0, 0, 0))
final_v = equilibria.final.potential_at((0, 0, 0))
print(f"V₂/V₁ = {final_v / initial_v:.8g}")  # V₂/V₁ = 9
```

Pass a previous `equilibria` result directly to `combine_isolated_drops((equilibria, another_drop))`. This retains each constituent charge and volume through later cancellation. Extracting only the final sphere rounds these quantities to the displayed equilibrium. Use the same permittivity at each stage when modelling one fluid in the same surroundings. This operation checks the equilibrium radius, conserved charge and conductor potential. The returned radial solutions also expose the backend’s spatial probes. At extreme scales those inherited potential, field and surface-density probes have their own numerical limits.
