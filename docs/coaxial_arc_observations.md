# Potential on the axis of charged arcs

Workbook page 3, Single Correct Q33 shows three circular arcs of common radius R with total charges +Q, −2Q and +3Q. It asks for the electric potential at their common centre. Every charge element is at distance R, so its contribution is k dq/R. Adding the three totals gives **V = 2kQ/R = Q/(2πε₀R), option (1)**. Multiplying the given total charges by their angular fractions would count the arc lengths twice.

In this teacher-selected SI example, angles are measured counterclockwise from +x. The three arcs occupy the xy plane.

```python
from conductor_sim import CircularArcCharge, CoaxialArcAssembly

q, radius = 3e-9, 0.20
arcs = CoaxialArcAssembly((
    CircularArcCharge(q, radius, 90, 45),
    CircularArcCharge(-2*q, radius, 135, 135),
    CircularArcCharge(3*q, radius, 270, 120),
))
print(arcs.potential_v(0))  # about +269.6 V
```

For a teacher variation, place an arc in a different xy plane using `center_z_m`, or change its radius, total charge or angular span. Each source keeps its own geometry. For an axial point z the source distance is √(R²+(z−z₀)²), independent of angle. This also determines the signed axial field component, kQ(z−z₀)/distance³. At the centre of a single arc that component vanishes, while its transverse field can be nonzero. The angular charge distribution would be needed to infer the full field from arc totals alone.

`field_work_j(probe_charge_c, start_z_m, end_z_m)` sums the contributions to q(Vstart−Vend) before rounding. It assumes fixed source charges and one uniform permittivity, with potential referenced to infinity. This endpoint observation requires no trajectory or travel time. All three operations take axial coordinates only.
