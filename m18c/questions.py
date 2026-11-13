"""Original textual tasks, not extracted textbook questions or a blind trial."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Question:
    id: str
    prompt: str

COMMON = (' Use SI output with 21 equally spaced states including t=0 and the stated final time. '
          'All conductors and all charges described as held remain stationary. '
          'Report the moving particle position, velocity, force, acceleration, kinetic energy, '
          'full-configuration mechanical interaction energy and energy drift. '
          'Point charges are ideal; do not simulate collisions. Use vacuum permittivity '
          '8.8541878128e-12 F/m. Coordinates are in metres and velocities in m/s. ')
QUESTIONS = (
    Question('plane-three', 'An infinite grounded plane is z=0, vacuum z>0. Author charges in this order: '
        '+0.4 nC held at (-0.055,0,0.14); -0.7 nC, mass 1 mg, moving from (0.01,0,0.10) '
        'with velocity (0.11,0,-0.015); -0.2 nC held at (0.075,0,0.16). Follow for 0.08 s.'+COMMON),
    Question('tilted-plane', 'A grounded plane passes through (0.4,-0.2,0.3), vacuum normal (0,0.6,0.8). '
        'Author -0.3 nC held at (0.35,-0.11,0.42), then +0.5 nC of mass 2 mg at '
        '(0.42,-0.14,0.38) with velocity (0.09,-0.006,-0.008). Follow for 0.10 s.'+COMMON),
    Question('grounded-sphere', 'A grounded solid conducting sphere has radius 0.06 m and centre (0.2,-0.1,0.3). '
        'Author +0.6 nC moving at (0.37,-0.1,0.3), mass 1 mg, velocity (0.015,0.12,0); '
        'then -0.3 nC held at (0.43,-0.02,0.3). Follow for 0.08 s.'+COMMON),
    Question('isolated-sphere', 'A solid conducting sphere at the origin, radius 0.055 m, is isolated with '
        'net charge +1.2 nC, not grounded. Author +0.3 nC held at (0.22,-0.05,0); '
        '-0.5 nC moving at (0.16,0,0), mass 1 mg, velocity (0.02,0.10,0); '
        '-0.2 nC held at (-0.19,0.07,0). Follow for 0.06 s.'+COMMON),
    Question('cavity-three', 'A grounded spherical shell at the origin has inner radius 0.12 m and outer '
        'radius 0.15 m. Author +0.2 nC held at (-0.045,0.02,0); -0.15 nC held at '
        '(0.04,-0.03,0); +0.4 nC moving at (0.01,0.035,0), mass 1.5 mg, '
        'velocity (0.08,-0.02,0). Follow for 0.08 s.'+COMMON),
    Question('nested-cavity', 'Two concentric shells are centred at (-0.3,0.2,0.1). The inner shell '
        'has radii (0.12,0.15) m and fixed net charge +0.4 nC; the outer shell has '
        'radii (0.23,0.27) m and is grounded. Author -0.2 nC held at (-0.34,0.22,0.1), '
        'then +0.5 nC moving at (-0.28,0.21,0.1), mass 1 mg, velocity (0.04,0.07,0). '
        'Follow for 0.10 s.'+COMMON),
    Question('biased-plates', 'Infinite conducting planes at z=0 and z=0.20 m are maintained at +3 V '
        'and +23 V, respectively; the vacuum is between them. Author +0.08 nC held at '
        '(-0.04,0,0.13), then -0.12 nC moving at (0.01,0,0.08), mass 1 mg, '
        'velocity (0.11,0,0.02). Follow for 0.12 s. Retain induction, not just a uniform field.'+COMMON),
    Question('centre-release', 'A grounded shell at the origin has radii (0.12,0.16) m. Author '
        '+0.06 nC held at (0,0.04,0), then -0.05 nC held at (0,0,0.05), then '
        '+0.10 nC moving at the centre, initially at rest with mass 1 mg. Follow for 0.20 s.'+COMMON),
    Question('two-moving', 'Release two finite-mass charges simultaneously above a grounded plane. '
        'Compute their coupled motion and induced conductor response. Do not hold either charge fixed.'),
    Question('nonplanar', 'Near a sphere centred at the origin, launch a charge from (0.2,0,0) '
        'with velocity (0,0.1,0), while a nonzero held charge is at (0.2,0.04,0.03). '
        'Use the planar motion engine without imposing a physical guide or projecting away force.'),
    Question('numerical-femm', 'Use only numerical FEMM forces to move a charge in the presence of '
        'a held charge and conductor. No analytical force substitution is permitted.'),
    Question('varying-voltage', 'Move a charge between plates whose applied voltage changes with time '
        'while a second charge stays held. Use the current autonomous static-electrostatics motion API.'),
)
BY_ID = {q.id:q for q in QUESTIONS}
