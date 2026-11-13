"""Frozen task prompts, separate from authored models and independent grading keys.

These parameterized prompts are fresh *open* M13c tasks; they are not claimed
as PDF transcriptions or a statistically blind test of language models.
"""
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Question:
    id: str
    prompt: str

QUESTIONS = (
    Question('plane-approach',
        'A +1 nC point charge above an infinite grounded conducting z=0 plane '
        'moves quasistatically from z=18 cm to z=6 cm, staying on the normal. '
        'Show seven equilibria, its induced force, mechanical interaction energy and external work.'),
    Question('plane-two-signed',
        'Above grounded z=0 metal, a +0.8 nC charge goes from (0,0,6) cm to '
        '(0,0,13) cm along a straight path. Keep a separate -0.35 nC charge '
        'fixed at (4.5,0,8) cm. Render seven positions, full force, energy and work.'),
    Question('plane-rotated',
        'An infinite grounded conductor lies at x=20 cm with vacuum on +x. '
        'Move -0.6 nC from (30,2,0) cm to (40,2,0) cm along the plane normal, '
        'in five static states. Plot total force and interaction energy.'),
    Question('sphere-grounded',
        'A grounded solid metal sphere of radius 11 cm is centred at origin. '
        'Move +1 nC on its +z axis from z=17 cm to z=30 cm across seven '
        'equilibria; calculate induced force, energy, and external work.'),
    Question('sphere-isolated',
        'A solid isolated conducting sphere has radius 12 cm, net charge '
        '+1.8 nC and centre at origin. An external +0.7 nC charge moves '
        'from z=22 cm to z=35 cm (seven equilibria). Separate background '
        'fixed-Q repulsion from induced polarization and show the work.'),
    Question('sphere-two-sources',
        'A grounded conducting solid sphere of radius 10 cm has an external '
        'moving +0.8 nC point charge from z=19 cm to z=30 cm; keep a '
        '-0.2 nC charge fixed at (24,10,3) cm. Author seven static states '
        'and show the force on the selected moving charge.'),
    Question('plates-grounded',
        'Two infinite grounded metal plates face each other at z=0 and '
        'z=20 cm. Move +1 nC along their common normal from z=5 cm to '
        'z=15 cm in nine steps. Examine midpoint force, energy maximum '
        'and external quasistatic work.'),
    Question('plates-biased-negative',
        'Between planes z=0 (grounded) and z=24 cm (+20 V), move '
        '-0.8 nC from z=6 cm to z=18 cm in seven stored states; display '
        'separately the induced attraction and imposed-voltage force.'),
    Question('plates-oblique-mixed',
        'Two parallel plates are grounded at z=0 and 20 cm. A +0.65 nC '
        'charge moves from (-4,0,5) cm to (4.5,0,14) cm in seven steps; '
        'a -0.23 nC charge remains at (-2,6,11) cm. Calculate path-projected '
        'force, mechanical energy and work without treating the second charge as moving.'),
    Question('plates-dual-bias',
        'Two infinite parallel plates at z=0 and 20 cm are kept at -4 V '
        'and +8 V. Move +0.9 nC along the z axis from z=6 cm to z=14 cm '
        'in five snapshots, retaining the same voltage gauge and both voltage sources.'),
    Question('dipole-torque',
        'A rigid dipole rotates in a uniform electric field. Show continuous '
        'torque, angle, angular velocity and rotational energy over time; '
        'do not substitute a single translating point charge.'),
    Question('ring-field',
        'A uniformly charged insulating circular ring of radius 10 cm holds '
        '+2 nC. Animate the potential, force and energy of an axial point '
        'charge approaching the centre; do not fake a ring with a conducting sphere.'),
    Question('insulating-volume',
        'A uniformly charged non-conducting solid sphere has prescribed '
        'volume density. Move an external test charge toward it and report '
        'the volume-field work, not the field of a conducting surface.'),
    Question('voltage-change-during-motion',
        'Move a charge between parallel plates while the upper plate voltage '
        'is independently changed from 0 to 30 V in the same sweep. Report '
        'electrode-source electrical work as well as mechanical work for this '
        'time-varying voltage protocol.'),
    Question('connected-spheres',
        'Two spatially separated conducting spheres of different radii are '
        'connected by a wire and share net charge. Track a nearby moving '
        'external charge and calculate the coupled redistribution and work.'),
    Question('time-trajectory',
        'Release a proton near a grounded metal plane with specified mass '
        'and initial velocity. Plot the actual acceleration, velocity, '
        'collision time and position as functions of elapsed time.'),
    Question('offaxis-femm-force',
        'Use real axisymmetric FEMM 5 to compute the instantaneous force '
        'and mechanical energy of two external point charges at distinct '
        'off-axis positions around a grounded sphere; no analytical substitution.'),
    Question('charge-in-cavity-motion',
        'Move an off-centre charge within the cavity of a thick metal shell '
        'and calculate its induced force/interaction energy; do not move '
        'it to the outside of a solid conductor.'),
)
BY_ID = {q.id: q for q in QUESTIONS}
