# Potential across parallel charged sheets

Workbook `02_conductors_half_size.pdf`, PDF page 4, Single Correct Q46 places two large identical charged plates at x=d and x=2d. The answer graphs continue onto PDF page 5. Their total charges are Q and −5Q, and V(0)=V₀. Let A denote their common area and take Q positive as in the choices. For small separations near their central regions, use the uniform infinite-sheet approximation with no external field. Edge fields and metal thickness are outside this approximation.

Each sheet contributes σ/(2ε) on its right and −σ/(2ε) on its left. Superposing the two sheets gives Eₓ = (2, 3, −2) Q/(Aε) in the three successive regions. V is continuous and its slopes are the negatives of these fields. Writing h = Qd/(Aε), the four potentials at 0, d, 2d and 3d are V₀, V₀−2h, V₀−5h and V₀−3h. The potential falls, falls more steeply, then rises with the magnitude of the initial slope. Printed graph 4 has this qualitative ordering. Reading the grid literally gives slope ratios −1:−2:+1 in graph 4, whereas the ideal sheet law gives −2:−3:+2. The option is a qualitative match only.

```python
from conductor_sim.parallel_sheets import ParallelSheetField
from conductor_sim.model import EPSILON_0

# Teacher example: d=1 m, Q/(A ε₀)=1 V/m, V₀=8 V.
field = ParallelSheetField(((1, EPSILON_0), (2, -5*EPSILON_0)), reference_potential_v=8)
values = [field.potential_at((x,0,0)) for x in range(4)]  # approximately [8, 6, 3, 5] V
interior = field.gap_graph(1.25, 1.75)  # bounded graph can feed axial field superposition
```

On PDF page 5, Single Correct Q48 places a uniformly charged sheet in the yz plane. A positive test charge moves along a circular arc from A=(10a,2a,0) to B=(10a,0,2a), then to C=(10a,0,0). The prompt calls it a plane sheet and draws a patch, without giving lateral dimensions or explicitly saying that it is infinite. Under the infinite-sheet idealization V depends only on x. The field then does zero work between each pair of endpoints, so the final energy is U₀, printed option 2. For a finite sheet the potential can vary with y and z even at fixed x. For example, a uniformly positive disk of radius a centred at the origin is admissible as a finite sheet. For a source element at (0,y,z), its squared distance to A exceeds that to C by 4a(a−y), positive almost everywhere on the disk. Thus V(C)>V(A) in that example. The given information does not determine the final energy for a finite sheet.

To recover signed densities and an equivalent uniform bias from a continuous piecewise-linear potential graph, see [workbook Q49](sheet_density_from_potential.md).

Supply prescribed signed densities in C/m² and sheet positions in metres. A declared uniform axial bias adds to the symmetric isolated-sheet solution. The potential reference is at a finite x because the field need not decay at infinity. Coincident densities add before a field jump is tested. At a charged sheet the model permits a potential observation and refuses a single electric-field vector. A continuous gap graph requires a closed interval free of charged sheets. These results do not determine a finite plate's detailed surface density or fringing field.

| Dimension | Status for two original items | Evidence |
| --- | --- | --- |
| Interpretation fidelity | Pass, 2/2 | Original plate order, charge ratio, reference and Q48 path endpoints are retained. |
| Physical correctness | Pass, 2/2 | Sheet jumps, signed region slopes, potential continuity and zero tangential work. |
| Simulation coverage | Conditional support, 2/2 | Q46 has piecewise potentials in the large-plate limit. Q48 has endpoint work for the described paths under the infinite-sheet assumption. The finite-sheet case is undetermined. |
| Educational usefulness | Not evaluated, 2/2 | Students can change density, reference, bias and endpoints. |
| Textbook-answer completeness | Qualified matches, 2/2 | Q46 graph 4 has the correct slope ordering. Q48 option 2 follows in the infinite-sheet interpretation. |
