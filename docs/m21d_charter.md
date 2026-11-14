# M21d — Distant connected spherical conductors: acceptance charter

Approved scope: a reusable **explicit leading self-capacitance approximation**, not an exact separated-conductor boundary solver. Freeze these criteria before implementation.

## Source anchors inspected
- Official IIT-JEE 2011 Paper 1 Q31, physical PDF p13: https://jeeadv.ac.in/past_qps/2011_1.pdf. A larger metal shell A and smaller solid sphere B, empty interiors, initially +Q each, far apart and connected by a thin wire. Verify all four stated relations: empty interior E=0, QA>QB, sigmaA/sigmaB=RB/RA, EA<EB. The paper prints ABCD. The numbers in a preview must be labeled illustrative.
- Workbook `02_conductors_half_size.pdf` p2 Single Q15: distant spherical conductors, surface-field ratio R2/R1 after connecting. Symbolic source; numerical initialization illustrative.
- Notes `POTENTIAL_CONDUCTORS1.pdf` p68 Illustration 4: shell radii 0.10/0.20 m and initial +0.5/+0.7 mC; final +0.4/+0.8 mC. Illustration 5: radii R,2R with initially equal surface density; final larger density 5sigma/6.
- PDF pins unchanged: workbook SHA256 9dff3c02a9f1848d2c3d99cb1a6e7a44dc7f5a51f7a3a2b8cc92d10ce1fbca5d; notes 8d0d9b5cf821219058cfc4b7685a39b777e460944df44c3f9cffcb68157a60e5. Spherical-shell wall thickness and finite demonstration distances are explicitly additional modeling choices.

## Physics contract
Reuse immutable SolidSphere/SphericalShell, ElectrostaticProblem, ConnectionPlan and named ElectricalLink components. A separately named backend requires explicit `self-capacitance-leading-order` selection. For empty cavities, homogeneous permittivity, 2–8 disjoint spheres and initially FixedNetCharge conductors, use Ci=4*pi*epsilon*Ri, Vi=Qi/Ci and U=sum Qi^2/(2Ci). Floating disjoint groups conserve their own total charge and share a common potential; unlinked conductors retain their charge. Source membership and identities persist in the resolved immutable problem, which can be disconnected/reconnected through a new plan.

Quantify geometric separation with max_i sum_(j!=i) Ri/dij <= 0.05. This is a conservative **admission policy**, not a proved 5% physical error bound. Report neglected mutual-monopole potential estimates and interaction-energy magnitude, not fake exact local fields. Zero/cancelled net charge and disparate source charges do not acquire a relative-accuracy guarantee. Wire capacitance, finite-wire geometry, surface polarization, close-range induction, nearby charges, grounding, voltage sources and dielectrics are excluded. Refuse invalid geometry/constraints explicitly; no automatic fallback to a concentric solver.

Energy release uses a nonnegative charge-redistribution identity; do not erase small release by subtracting large stored energies. The release is electrostatic energy available on settling, not a computed current, heating rate, discharge duration or guarantee that all energy goes into the wire.

## Verification and display
Independent capacitance/charge/energy checks, source numbers and four exam relations; stable identity, groups, reordering, transforms, permittivity, signed and zero total charge, disconnected continuation, invalid requests and numerical overflow. Retain existing engine tests and real FEMM nonregression (not claimed to validate the new approximation).

Offline before/after schematic and SI tables, explicit approximation banner, shell versus solid depiction and source/hypothetical labeling. No fabricated global field map or charge-flow animation. Actual Chromium tests check visible numbers, source IDs, controls, mobile layout, no external requests/JS errors. Keep implementation bounded: no generic scenario architecture or new transient solver.
