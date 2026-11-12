"""Live numerical empty-shell benchmark. Requires actual FEMM 5 installation.

Set FEMM5_BINDING to its pyfemm/femm.py and (if needed) FEMM_DIR to the
executable folder. Run: python examples/femm5_empty_shell.py
"""
from math import pi

from conductor_sim import ConcentricShellBackend, ElectrostaticProblem, FixedNetCharge, SphericalShell
from conductor_sim.femm5 import Femm5Backend


problem = ElectrostaticProblem(
    conductors=(SphericalShell(.1, .12, FixedNetCharge(1e-9)),),
)
reference = ConcentricShellBackend().solve(problem)
with Femm5Backend().solve(problem) as numerical:
    print("Conductor: analytic", reference.conductor_results[0])
    print("Conductor: FEMM 5 ", numerical.conductor_results[0])
    print("radius (m)   analytic V     FEMM V       analytic E_r      FEMM E_r")
    for r in (.05, .11, .18, .24, .36):
        print(
            f"{r:10.3f} {reference.potential_at((r,0,0)):12.6f} "
            f"{numerical.potential_at((r,0,0)):12.6f} "
            f"{reference.electric_field_at((r,0,0))[0]:16.6f} "
            f"{numerical.electric_field_at((r,0,0))[0]:16.6f}"
        )
