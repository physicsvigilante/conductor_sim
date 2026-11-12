"""Compare a real FEMM 5 axisymmetric centered source with exact Gauss-law data.

Requires FEMM_DIR and FEMM5_BINDING to point to FEMM 5 (see docs/femm5.md).
The source is regularized as a small charged sphere; sampling is permitted only
outside it. All three-dimensional input and result positions are SI coordinates.
"""

from conductor_sim import (
    ConcentricShellBackend, ElectrostaticProblem, Femm5Backend,
    FixedNetCharge, Grounded, PointCharge, SphericalShell,
)

for constraint in (FixedNetCharge(0.0), Grounded()):
    problem = ElectrostaticProblem(
        charges=(PointCharge(1e-9),),
        conductors=(SphericalShell(0.1, 0.12, constraint),),
    )
    reference = ConcentricShellBackend().solve(problem)
    with Femm5Backend().solve(problem) as numerical:
        print(f"\n{type(constraint).__name__}")
        print("Reference shell:", reference.conductor_results[0])
        print("FEMM 5 shell:   ", numerical.conductor_results[0])
        print("FEMM source radius:", numerical.source_radius_m)
        print("FEMM measured source charge:", numerical.numerical_source_charge_c)
        print("r (m)      exact V      FEMM V       exact E_r       FEMM E_r")
        for radius in (.03, .05, .075, .11, .18, .30):
            point = (radius, 0.0, 0.0)
            print(
                f"{radius:6.3f} {reference.potential_at(point):12.5f} "
                f"{numerical.potential_at(point):12.5f} "
                f"{reference.electric_field_at(point)[0]:14.5f} "
                f"{numerical.electric_field_at(point)[0]:14.5f}"
            )
