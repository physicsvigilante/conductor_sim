"""Fresh JEE-style tasks, kept separate from authored submissions and grading keys.

This is a reproducible *open* hold-out from the curated M6a PDF cases, not an
independently blinded model benchmark. These prompts were not part of M6a.
"""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Question:
    id: str
    prompt: str


QUESTIONS = (
    Question('two-shell-negative-net',
        'Two concentric metal shells have inner/outer radii 8/10 cm and 19/21 cm. '
        'The inner shell has +1 nC; the outer has -2 nC. There is no free '
        'charge in the cavity. Find outer-shell net charge, E_x in the gap at '
        'x=15 cm, and potential at x=30 cm.'),
    Question('grounded-outer-shield',
        'A neutral-cavity metal shell (6–8 cm) carries +0.6 nC and is '
        'surrounded by a grounded 17–20 cm shell. Find the outer shell’s '
        'net charge, the inner-shell voltage and the exterior field at 30 cm.'),
    Question('offcentre-negative-grounded',
        'A -1.4 nC point charge sits at z=+2.8 cm inside a grounded 10–13 cm '
        'spherical conductor. Find its potential, total induced conductor '
        'charge and exterior electric field at x=25 cm.'),
    Question('central-source-and-charged-shell',
        'A +1.2 nC central point charge is enclosed by an isolated metal '
        'shell spanning 9–11.5 cm, whose prescribed net charge is +0.3 nC. '
        'Find induced charge on both faces and exterior potential at x=23 cm.'),
    Question('noncollinear-math-only',
        'A 12–15 cm metal sphere shell has fixed charge -0.5 nC. Two point '
        'charges inside its cavity sit at (2.0,1.3,0) cm with +1 nC and '
        '(-1.6,0,2.4) cm with -0.4 nC. Find outer induced charge and '
        'potential at (30,0,0) cm. Can the axisymmetric FEMM backend solve '
        'this non-collinear problem, or is the exact analytical solver needed?'),
    Question('reordered-three-shells',
        'Three isolated concentric shells, ordered physically from inside out '
        'as 5–6 cm (+0.4 nC), 11.5–13 cm (-0.9 nC), and 19–21 cm (+0.2 nC), '
        'have no free cavity charges. Author them in REVERSE order and find '
        'the outermost face charge, E_x at x=16 cm, and outer shell net charge.'),
    Question('cancelled-central-charges',
        'At the exact centre of a grounded shell spanning 10–12 cm sit two '
        'separate but coincident authored charges +1.2 nC and -1.2 nC. '
        'What are the grounded shell potential, its net charge and E_x '
        'at x=5.5 cm? Avoid inventing a nonzero net source.'),
    Question('translated-rotated-cavity-charge',
        'A shell centred at (0.4,-0.2,0.1) m has radii 10–12 cm and net '
        'charge +1 nC. A -0.4 nC charge lies 3 cm along +x and 4 cm along '
        '+y from its centre. Find outer-face induced charge and potential '
        'at a point 25 cm along +x from the shell centre. FEMM must select '
        'the appropriate symmetry axis itself.'),
    Question('two-positive-separated-sources',
        'A 10–12 cm neutral isolated shell encloses a +0.7 nC source '
        'at z=+2.7 cm and a separate +0.3 nC source at z=-2.1 cm. '
        'Find the induced total inner charge, outer charge and exterior '
        'potential at x=24 cm. These must be two distinct charge sites.'),
    Question('grounded-two-state-sweep',
        'A metal shell spanning 9–11 cm carries first +1 nC, then +3 nC '
        'in two static configurations. It is surrounded by a grounded '
        '18–20 cm shell, and there are no cavity charges. Produce two '
        'states, recording inner-shell potential in each and the outer '
        'shell’s E_x at x=30 cm in both.'),
    Question('exterior-charge-grounded-sphere',
        'Place a +1 nC charge 30 cm outside the centre of a grounded '
        'conducting spherical shell of radius 10 cm. Determine its '
        'nonuniform induced surface charge and force. Do not move '
        'the source into a cavity to make the geometry fit.'),
    Question('charged-ring-on-axis',
        'A uniformly charged thin circular ring, radius 10 cm, has +2 nC. '
        'Find the potential on its axis at z=5 cm and render its field. '
        'Do not replace the ring by a spherical conductor.'),
    Question('charge-between-conductors',
        'Two concentric isolated shells occupy 5–6 cm and 14–16 cm. '
        'Put a point charge at z=10 cm in the VACUUM GAP between them. '
        'Compute their induced surface-charge patterns.'),
    Question('connected-separated-spheres',
        'Two conducting spheres of radii 5 and 10 cm are separated by '
        '1 metre and joined by a thin wire. A total charge +3 nC '
        'is shared at equal potential. Compute each sphere’s charge. '
        'These are not nested concentric shells.'),
    Question('moving-charge-trajectory',
        'A proton initially at rest is released near an isolated conducting '
        'spherical shell. Produce continuous acceleration, velocity, '
        'time-of-flight and a moving particle visualization (not a '
        'slider between static electrostatic equilibria).'),
)

BY_ID = {q.id:q for q in QUESTIONS}
