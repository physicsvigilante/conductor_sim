"""Regression tests for problems found during the M4/M5 review."""
from math import nan, inf
import pytest
from conductor_sim import (
    ElectrostaticProblem, PointCharge, SphericalShell, Grounded,
    ConcentricShellBackend, ImageChargeShellBackend, MultiImageShellBackend, NestedShellBackend,
)


def problem(d=0):
    return ElectrostaticProblem((PointCharge(1e-9,(0,0,d)),),(SphericalShell(.1,.12,Grounded()),))


@pytest.mark.parametrize('backend',[ConcentricShellBackend,ImageChargeShellBackend,MultiImageShellBackend,NestedShellBackend])
@pytest.mark.parametrize('point',[(nan,0,0),(inf,0,0),(0,0),(1,2,3,4),None,('x',0,0)])
def test_all_analytical_samplers_reject_invalid_coordinates(backend,point):
    sol=backend().solve(problem())
    for method in (sol.potential_at,sol.electric_field_at):
        with pytest.raises(ValueError,match='three finite coordinates'):
            method(point)


@pytest.mark.parametrize('backend',[ImageChargeShellBackend,MultiImageShellBackend,NestedShellBackend])
@pytest.mark.parametrize('d',[1e-150,1e-155,1e-160])
def test_kelvin_center_limit_no_overflow_or_nan(backend,d):
    actual=backend().solve(problem(d))
    reference=ConcentricShellBackend().solve(problem())
    point=(.05,0,0)
    assert actual.potential_at(point)==pytest.approx(reference.potential_at(point),rel=1e-12)
    assert actual.electric_field_at(point)==pytest.approx(reference.electric_field_at(point),rel=1e-12,abs=1e-12)


@pytest.mark.parametrize('name',[None,5,'','  ','";command()','new\nline','bad\\escape','zero\x00byte'])
def test_unsafe_femm_conductor_names_rejected_at_model_boundary(name):
    with pytest.raises(ValueError,match='conductor name'):
        SphericalShell(.1,.12,Grounded(),name=name)


def test_unicode_and_spaces_remain_valid_names():
    assert SphericalShell(.1,.12,Grounded(),name='inner shell α').name=='inner shell α'


def test_stable_kelvin_field_matches_negative_potential_gradient():
    sol=MultiImageShellBackend().solve(problem(.04))
    p=(.03,.02,-.01);h=1e-7
    gradient=[]
    for axis in range(3):
        plus=list(p);minus=list(p);plus[axis]+=h;minus[axis]-=h
        gradient.append(-(sol.potential_at(plus)-sol.potential_at(minus))/(2*h))
    assert sol.electric_field_at(p)==pytest.approx(gradient,rel=1e-8)
