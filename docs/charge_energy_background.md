# Held point charges in an external field

Workbook `02_conductors_half_size.pdf`, PDF page 6 (printed 3.38), Single Correct Q61 places two point charges a distance r apart in an external uniform electric field. V₁ and V₂ are the potentials due to the external sources alone at the two sites. The energy of the held charges is **q₁V₁ + q₂V₂ + q₁q₂/(4πε₀r), option (3)**. Each charge couples to the prescribed field once. The mutual pair also occurs once. A factor of one half would undercount the external contribution.

The engine combines an `ElectrostaticProblem` of held point charges with an independently authored `AffineFieldAssembly` in one homogeneous medium. The pair convention uses infinite separation. The background retains its specified potential reference. A constant shift C in that reference adds C times the total held charge to the result. Self-energies, the energy of creating the background, and the response of external sources are excluded.

```pycon
>>> from conductor_sim import ElectrostaticProblem, PointCharge, UniformElectricField
>>> from conductor_sim.affine_fields import AffineFieldAssembly
>>> from conductor_sim.point_charge_work import configuration_energy_j
>>> sites = ElectrostaticProblem(charges=(PointCharge(2e-9), PointCharge(-3e-9, (3, -4, 0))))
>>> field = AffineFieldAssembly((UniformElectricField((1, 0, 0), reference_potential_v=7),))
>>> round(configuration_energy_j(sites, field) * 1e9, 4)  # nJ
-8.7851

```

All numerical values are teacher-selected. The site separation is 5 m, the external potentials are 7 V and 4 V, and the two external energy contributions are +14 nJ and −12 nJ. The attractive pair contributes about −10.7851 nJ. Changing the field or adding another held charge re-evaluates the complete sum.

For the one source item, the external-only interpretation of the potentials is preserved (1/1), independent term counting gives the physical energy (1/1), the model evaluates the requested static observable for a numerical instance (1/1), and the expression matches the printed choice (1/1). Educational usefulness has not been measured.
