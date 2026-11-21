# Interaction energy and bringing a charge from infinity

In a homogeneous dielectric, the electrostatic interaction energy of held point charges is U = k Σᵢ<ⱼ qᵢqⱼ/rᵢⱼ. The reference is infinite separation. A single or empty configuration has zero interaction energy. Bringing one charge from infinity while holding the others fixed requires external work equal to the sum of its new pair terms for unchanged kinetic energy. Insertion and rearrangement return a result whose `.final` problem can feed the next operation. `interaction_energy_j` returns the energy in joules. Stored positions use binary64 SI coordinates. An integer position that would change on conversion is refused. Singular coincident nonzero charges and responding conductors fall outside this model.

## Workbook page 4, Single Correct Q37

In `02_conductors_half_size.pdf`, the two upper corners of a square carry +q and the two lower corners −q. The four side contributions cancel. Both diagonals connect opposite charges, so U(a) = −√2 kq²/a. Reducing a makes U more negative. For slow motion with unchanged kinetic energy, the external agent does negative work, option 3. The regression uses teacher-selected sides 2 m and 1 m and checks both endpoint energies and their signed change.

## Workbook page 4, Single Correct Q44

The two fixed +Q charges have coordinates (−a/2,0,0) and (+a/2,0,0). For a third charge −Q at distances r₋ and r₊, the combined energy is kQ²(1/a − 1/r₋ − 1/r₊). Its full three-dimensional zero locus satisfies **1/r₋ + 1/r₊ = 1/a**. The question gives no restriction on x.

On the perpendicular bisector plane x = 0 both distances must equal 2a, giving **y²+z² = 15a²/4**. This is printed option 3 only as a section in that plane. The full locus also crosses the x axis at x = ±a(1+√5/2), where y = z = 0. None of the three printed equations describes the full locus, so the literal unrestricted question gives option 4. No answer-key intent is assumed.

```python
from conductor_sim.model import ElectrostaticProblem, PointCharge
from conductor_sim.point_charge_work import insert_point_charge, interaction_energy_j

# Teacher-selected a = 2 m, Q = 1 nC, and a trial position on the x axis.
pair = ElectrostaticProblem(charges=(PointCharge(1e-9, (-1,0,0)), PointCharge(1e-9, (1,0,0))))
trial = insert_point_charge(pair, PointCharge(-1e-9, (5,0,0)))
assert trial.external_work_j < 0
assert interaction_energy_j(trial.final) > 0
```

The work of inserting −Q is negative at every finite nonsingular position for this pair. The zero-locus condition includes the pair's positive initial energy. Native tests bracket the bisector and axial crossings with adjacent floating-point positions and test a point on the cylinder of option 3 outside the true locus. Signed sums use the shared reciprocal-distance calculation, including cancellation before rounding.

| Dimension | Status for two workbook items | Evidence |
| --- | --- | --- |
| Interpretation fidelity | Pass, 2/2 | Square charge arrangement and unrestricted Q44 probe position are preserved. |
| Physical correctness | Pass, 2/2 | Counted side and diagonal pairs, both locus sections and a counterexample to the cylinder. |
| Simulation coverage | Supported, 2/2 | Static energy and quasistatic work at authored positions. The locus equation is derived analytically. |
| Educational usefulness | Not evaluated, 2/2 | Source cases distinguish interaction energy from incremental work. |
| Textbook-answer completeness | Q37 complete, 1/2. Q44 qualified, 1/2. | Q37 matches option 3. Q44 distinguishes the literal answer from the restricted bisector section. |
