"""Original prompts, not copied from the conductor PDFs or existing examples."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Question:
    id: str
    prompt: str

QUESTIONS = (
    Question('centre-outward', 'A +1.2 nC point charge moves quasistatically from the centre to (0,0,7) cm inside a grounded spherical cavity of radius 12 cm; the metal outer radius is 15 cm. Show five equilibria, force, mechanical interaction energy and external work.'),
    Question('negative-inward', 'Inside a grounded spherical shell with inner/outer radii 11/14 cm, move -0.7 nC from (8,0,0) cm to (2,0,0) cm in five static states. Resolve the force along this inward path and the external work.'),
    Question('translated-isolated', 'An isolated shell centred at (0.8,-0.4,0.2) m has inner/outer radii 12/16 cm and net +1.5 nC. A +0.9 nC source follows a straight chord, relative to that centre, from (2,-1,1) cm to (6,2,4) cm in five states. Retain the shell charge and voltage reference.'),
    Question('signed-reordering', 'A grounded cavity has inner/outer radii 10/13 cm. Move +0.8 nC from (-3,0,-3) cm to (4,0,4) cm; keep -0.45 nC fixed at (1,3,1.5) cm. Show five states and track the moving source even when sorted source order changes.'),
    Question('nested-ground', 'An inner conducting shell has radii 12/15 cm and net +0.4 nC; a concentric outer shell has radii 45/50 cm and is grounded. Move +1 nC from (0,0,-4) cm to (0,0,8) cm in five static states. Include both shells in the view.'),
    Question('inner-ground-shield', 'Three concentric shells have radii 10/12, 20/22 and 30/32 cm. The innermost is grounded, the middle carries -0.3 nC and the outer +0.8 nC. Move +0.6 nC from (0,0,2) cm to (0,0,6) cm in five states. Verify that the exterior charges do not exert a cavity force.'),
    Question('connect-disconnect', 'Two concentric shells have radii 10/13 and 22/25 cm, initial net charges +0.3 and -0.1 nC, and a +0.6 nC point charge at (0,0,2) cm in the inner cavity. Join the shells by an isolated ideal wire, let them equilibrate, then REMOVE the wire. Move the point charge to (0,0,7) cm in five states, preserving the newly isolated shell charges.'),
    Question('zero-net-sources', 'An isolated shell with radii 12/15 cm has net +0.2 nC. Keep -1 nC at (3,0,0) cm while +1 nC moves from (0,0,-5) cm to (0,0,5) cm, in five static states. The net source charge is zero, but the individual induced forces are not generally zero.'),
    Question('cross-metal', 'A charge starts at (0,0,2) cm inside a shell of radii 10/12 cm. Use a continuous vacuum-only electrostatic position sweep ending at (0,0,14) cm. Do not ignore the intervening metal.'),
    Question('source-collision', 'Inside a grounded cavity of radius 10 cm, sweep +1 nC from (0,0,-4) to (0,0,4) cm through a fixed -1 nC point charge at the centre. Keep both ideal point charges as separately identifiable particles.'),
    Question('cavity-time-motion', 'Release a massive charged particle inside an off-centre conducting cavity and produce elapsed-time position, velocity and a wall-impact time. A static position sweep is not the requested dynamics.'),
    Question('numerical-cavity-force', 'Measure the force and mechanical interaction energy of an internal off-axis point charge using the numerical FEMM solution itself, without substituting Kelvin analytical observations.'),
)
BY_ID = {q.id:q for q in QUESTIONS}
