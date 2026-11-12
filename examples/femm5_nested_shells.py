"""M4b live comparison: nested isolated inner shell and grounded outer shell.

Requires the real FEMM 5 installation and FEMM5_BINDING.
"""
from conductor_sim import (
    ElectrostaticProblem, FixedNetCharge, Grounded,
    PointCharge, SphericalShell, Femm5Backend, NestedShellBackend,
)
problem=ElectrostaticProblem(
    charges=(PointCharge(1e-9,(0,0,.035)),PointCharge(.45e-9,(0,0,-.025))),
    conductors=(
        SphericalShell(.1,.12,FixedNetCharge(0),name="inner"),
        SphericalShell(.2,.22,Grounded(),name="outer"),
    ),
)
reference=NestedShellBackend().solve(problem)
with Femm5Backend().solve(problem) as result:
    for name in ("inner","outer"):
        print(name,"exact",reference.conductor_by_name(name))
        print(name,"FEMM ",result.conductor_by_name(name))
    for x in (.055,.11,.16,.21,.3):
        pt=(x,0,0)
        print("x=",x," V_exact=",reference.potential_at(pt),
              " V_FEMM=",result.potential_at(pt))
    for name in ("inner","outer"):
        print(name,"numerical sigma inner:",result.surface_charge_density(
            "inner",1.,conductor_name=name))
