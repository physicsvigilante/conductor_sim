# M23a — Static rigid electric dipoles: frozen acceptance

Approved scope: a reusable finite rigid pair (+q, -q), existing externally prescribed field components, static force/torque/energy observations and independently authored orientations. No new electrostatic source formula, time integrator, dynamics, damping, polarizability, responding source or universal compiler.

Base main: 355e1a53c1f0764931086bab09944ff60c4a21b9.

## Source check before implementation

The actual checked-in `02_conductors_half_size.pdf`, SHA-256 9dff3c02a9f1848d2c3d99cb1a6e7a44dc7f5a51f7a3a2b8cc92d10ce1fbca5d, physical page 5 (printed page 3.37), Q47, was rendered and visually inspected. A dipole begins parallel to a uniform E and is rotated to four depicted orientations: (i) acute, (ii) perpendicular, (iii) obtuse, (iv) antiparallel. It asks for a ranking of potential-energy changes. No numerical angles, q, separation, E, mass, pivot or moment of inertia are specified. Chosen display values must be explicitly illustrative. It is a workbook exercise, not certified as an official JEE past-paper item.

## Preselected positive arrangements

P1: uniform E along +x, fixed centre; reproduce the four qualitative directions and rank delta U. Representative angles 40,90,115,180 degrees, q=1 nC, separation=.04 m, E=120 V/m are illustrative only. Verify F_net=0, torque=p cross E, U=-p dot E and stable/unstable rotational orientations.
P2: finite dipole near a separately fixed point charge; evaluate qE at both real endpoints and q(V_plus-V_minus), not the local point-dipole approximation. Verify independent Coulomb reference and torque about centre versus an explicit off-centre pivot; changes in centre under rotation about that pivot must be real.
T1: a newly authored mixed finite rod + fixed volume sphere + uniform field. Same observer and orientation sampler, no arrangement-specific production path. Independently verify rod quadrature, sphere formula and energy/torque derivative. This is preselected and same-author exposed, not a blinded transfer trial.

## Qualification and investigation rules

Use complete three-component forces and torques. Do not hide a perpendicular component in a physical-slice viewer. Source identities and immutable parents survive parameter variations. Dipole moment is derived q*separation*direction, not an independently inconsistent input. Interaction energy is EXTERNAL-field interaction only; mutual binding energy is constant for a fixed rigid pair and excluded. Changing separation is a new model, not rotational work.

Saved angle selections are independent static observations, not elapsed time, a free trajectory or proof that every intervening configuration is admissible. Unsupported or singular observations must be explicit gaps/refusals. Potential-only work can remain defined when optional field/torque observations are undefined. Work endpoints must refer to the same external sources, q, separation and pivot policy. Analytic equilibrium classification is restricted to a globally uniform field; nonuniform global root completeness is not claimed.

## Preselected negative and preservation tests

Reject malformed vectors/nonfinite data, zero/negative q or length, aliased or duplicate source IDs, poorly resolved geometry, conductor objects, unsupported off-axis ring/disc observations, singular endpoint potentials, invalid revisions and mixed-model work claims. A shell/disc field jump is not zero force. No angular dynamics without inertia/constraints/initial conditions. No forced projection or automatic reorientation of fields.

Independent tests: Coulomb endpoint sums, uniform exact laws, per-source superposition, gauge invariance, translation/rotation covariance, torque-origin change, minus dU/dtheta, fixed-pivot centre motion, signed work/reversal, exact workbook ordering and trusted numerical rod integration. Preserve all historical accepted source files and evidence; extend existing CI only where appropriate. Actual offline Chromium must verify saved-angle controls, displayed vectors/units, source notes, mobile width, no JS/network errors. Full ordinary Python and existing actual FEMM nonregression remain gates; FEMM is not the reference for this finite-dipole observer.
