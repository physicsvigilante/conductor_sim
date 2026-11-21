# From field area to potential

On page 4 of `02_conductors_half_size.pdf`, Single Correct Q39 asks for a potential difference in an inclined uniform field. Single Correct Q40 gives the potential at the origin and asks for the potential at 2 m in a position-dependent field. The offline page plots the native axial field and potential against the same x coordinate. Shading stops at the position named in each question.

| Source | Signed area ∫Eₓ dx | Potential change | Requested observation |
| --- | --- | --- | --- |
| Q39, A to O | +100 V | −100 V | V(O) − V(A) = −100 V, option 4 |
| Q40, 0 to 2 m | −20 V | +20 V | V(2 m) = 10 V + 20 V = 30 V, option 4 |

For Q39 the positive field component is 100 cos 60° = 50 V/m. The example places A at −2 m and O at 0 and chooses V(A) = 0 V for the plotted potential. The requested difference is independent of that choice. For Q40, integrating Eₓ = −10x from 0 to 2 m gives the signed triangular area. The plotted Q40 domain ends at the last numbered tick at 5 m.

Run `python -m tools.jee_field_area review-artifacts/field-area/field-area.html` to generate the page, or `python -m tools.check_jee_field_area_browser review-artifacts/field-area` to repeat the mobile and desktop Chromium checks. Both plots use static native observations. The field graph has 21 samples for Q39 and 61 for Q40. The potential curves use the same positions.

| Evaluation dimension | Status for two original questions | Evidence |
| --- | --- | --- |
| Interpretation fidelity | Pass, 2/2 | Both prompts, field directions, reference choices and requested observations are identified. |
| Physical correctness | Pass, 2/2 | Independent projected-field and signed triangular-area calculations. |
| Simulation coverage | Supported, 2/2 | Both full requested paths are inside the qualified static fields. |
| Educational usefulness | Not evaluated, 2/2 | Plots and signed observations were checked in Chromium. Learner outcomes were not measured. |
| Textbook-answer completeness | Complete, 2/2 | Both original signed answers give printed option 4. |
