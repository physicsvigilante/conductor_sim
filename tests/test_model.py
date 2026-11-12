import math
import pytest
from conductor_sim import ElectrostaticProblem, FixedNetCharge, Grounded, PointCharge, SphericalShell

def test_problem_copies_mutable_input():
    charges = [PointCharge(1.), PointCharge(-1.)]
    p = ElectrostaticProblem(charges=charges)
    charges.clear()
    assert len(p.charges) == 2

@pytest.mark.parametrize("inner,outer", [(0,1), (1,1), (2,1), (-1,1), (1,math.inf)])
def test_invalid_shell(inner,outer):
    with pytest.raises(ValueError):
        SphericalShell(inner,outer,Grounded())

def test_nonfinite_and_bad_permittivity():
    with pytest.raises(ValueError):
        PointCharge(float("nan"))
    with pytest.raises(ValueError):
        FixedNetCharge(float("inf"))
    with pytest.raises(ValueError):
        ElectrostaticProblem(permittivity_f_per_m=0)

def test_duplicate_names():
    a = SphericalShell(.1,.2,Grounded(),name="same")
    b = SphericalShell(.3,.4,FixedNetCharge(0),name="same")
    with pytest.raises(ValueError):
        ElectrostaticProblem(conductors=(a,b))
