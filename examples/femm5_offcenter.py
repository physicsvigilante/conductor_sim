"""M3: solve an off-center point charge and compare its induced sigma profile.

Requires the real FEMM 5 TCP binding, e.g. FEMM5_BINDING=/path/to/pyfemm/femm.py.
"""
from math import pi
from conductor_sim import (
    ElectrostaticProblem, Femm5Backend, FixedNetCharge, Grounded,
    ImageChargeShellBackend, PointCharge, SphericalShell,
)


def show(constraint):
    problem = ElectrostaticProblem(
        charges=(PointCharge(1e-9, (0, 0, .04)),),
        conductors=(SphericalShell(.1, .12, constraint),),
    )
    analytic = ImageChargeShellBackend().solve(problem)
    print('\n',type(constraint).__name__)
    with Femm5Backend().solve(problem) as numeric:
        print('Analytical shell:', analytic.conductor_results[0])
        print('FEMM 5 shell:    ', numeric.conductor_results[0])
        print('FEMM numerical source radius:', numeric.source_radius_m)
        print('Angle (deg)  Analytical inner sigma  FEMM near-surface sigma [C/m^2]')
        for theta in (0.15, 0.45, 0.90, pi/2, 2.30, pi-0.15):
            print(f'{theta*180/pi:10.1f}  {analytic.surface_charge_density("inner",theta):22.6e}  '
                  f'{numeric.surface_charge_density("inner",theta):22.6e}')
        print('Position (m)        Analytic V       FEMM V')
        for point in ((0,0,-.06), (.05,0,0), (0,0,.075), (.2,0,0)):
            print(f'{point!s:20s} {analytic.potential_at(point):13.6f} '
                  f'{numeric.potential_at(point):13.6f}')


if __name__ == '__main__':
    show(FixedNetCharge(0))
    show(Grounded())
