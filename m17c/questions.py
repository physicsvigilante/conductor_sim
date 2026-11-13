"""Original authoring prompts. No solver code or grading outputs are supplied."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Question:
    id: str
    prompt: str

QUESTIONS = (
    Question('tilted-plane',
        'An infinite grounded conductor passes through (0.4,-0.2,0.3) m and has vacuum normal '
        '(0,0.6,0.8). A -0.8 nC particle of mass 1 milligram starts 0.12 m along that normal '
        'and 0.03 m in the +x direction from this point. Its initial velocity is '
        '(0.13,0.009,0.012) m/s. Model its free motion for 0.16 s, sampled every 0.008 s; '
        'show the deflection, velocity and mechanical energy without a guide or constraint.'),
    Question('electron-plates',
        'Infinite parallel plates at z=0 and z=0.2 m are held at 0 and 20 V. An electron '
        '(q=-1.602176634e-19 C, mass=9.1093837139e-31 kg) starts at (0,0,0.1) m with '
        'velocity (800000,200000,-100000) m/s. Calculate 50 ns of motion in 2.5 ns output '
        'intervals. Include conductor induction in the engine; check the uniform-field '
        'parabolic limit where induction is negligible. The plates have no finite edges.'),
    Question('grounded-sphere',
        'A grounded solid conducting sphere has radius 0.07 m and centre (0.2,-0.3,0.1) m. '
        'A -1 nC particle, mass 1 milligram, starts at (0.38,-0.26,0.1) m with velocity '
        '(-0.02,0.18,0.06) m/s. Integrate 0.12 s with output every 0.006 s. Show the curved '
        'trajectory and check angular momentum about the sphere centre and energy.'),
    Question('isolated-sphere',
        'An isolated solid sphere of radius 0.06 m centred at the origin carries +1.6 nC '
        'net charge. A +0.7 nC particle of mass 1 milligram starts at (0.15,0,0) m with '
        'velocity (-0.04,0.11,0) m/s. Model 0.08 s with output every 0.004 s, retaining '
        'both polarization and the prescribed sphere charge. Do not ground the sphere.'),
    Question('cavity-oblique',
        'A grounded concentric spherical shell has inner and outer radii 0.12 and 0.15 m. '
        'A -1.1 nC particle of mass 1 milligram starts at (0.025,-0.01,0.005) m with '
        'velocity (0.02,0.11,-0.04) m/s. Model 0.1 s in 0.005 s output intervals. Show the '
        'curved trajectory inside the cavity, not a prescribed chord.'),
    Question('nested-cavity',
        'Two isolated concentric conducting shells centred at (-0.5,0.25,0.1) m have '
        'radii (0.12,0.15) and (0.24,0.28) m and fixed net charges -0.2 nC and +1 nC. '
        'A +0.6 nC particle of mass 1 milligram starts at (-0.48,0.26,0.07) m with '
        'velocity (0.06,-0.04,0.03) m/s. Calculate 0.08 s with 0.004 s output spacing, '
        'including the fixed-shell contribution to the interaction energy.'),
    Question('rotated-radial',
        'A grounded solid sphere of radius 0.01 m is centred at the origin. A +1 nC '
        'particle of mass 1 milligram starts at (0.1,0.2,0.3) m with velocity '
        '(-0.01,-0.02,-0.03) m/s. Integrate 0.06 s sampled every 0.003 s and render it '
        'with the planar viewer. This radial special case must not acquire artificial '
        'angular momentum or fail because of the coordinate rotation.'),
    Question('centre-crossing',
        'An isolated concentric spherical shell, radii 0.12 and 0.16 m, carries +0.3 nC. '
        'A +0.2 nC particle of mass 1 milligram starts at (0.02,0,0) m and moves with '
        'velocity (-0.25,0,0) m/s. Model 0.12 s in 0.006 s output intervals, including '
        'passage through the cavity centre. The centre is not a conducting boundary.'),
    Question('two-moving',
        'Release two separate charged particles near a grounded sphere and calculate '
        'both mutually coupled trajectories with the existing M17 motion API.'),
    Question('numerical-femm',
        'Use numerical FEMM force, not an analytical image-force substitute, to animate '
        'an oblique particle trajectory near a sphere with the existing motion API.'),
    Question('finite-plate-edge',
        'Launch an electron beyond the edge of two finite square plates and simulate '
        'fringing-field deflection with the current infinite-plate components.'),
    Question('varying-voltage',
        'Drive a capacitor plate with a sinusoidal voltage during particle flight and '
        'simulate the time-dependent trajectory with the existing fixed-voltage M17 API.'),
)
BY_ID = {q.id:q for q in QUESTIONS}
