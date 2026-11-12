"""Run after installation: python examples/concentric_shell.py."""
from conductor_sim import (
    ConcentricShellBackend, ElectrostaticProblem, FixedNetCharge,
    Grounded, PointCharge, SphericalShell,
)
for constraint in (FixedNetCharge(0.0), Grounded()):
    problem = ElectrostaticProblem(
        charges=(PointCharge(1e-9),),
        conductors=(SphericalShell(0.1, 0.12, constraint),),
    )
    solution = ConcentricShellBackend().solve(problem)
    print(type(constraint).__name__, solution.conductor_results[0])
    for radius in (0.05, 0.11, 0.2):
        print(radius, solution.potential_at((radius, 0.0, 0.0)))
