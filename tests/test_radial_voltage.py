from math import nextafter, pi

import pytest

from conductor_sim import (EPSILON_0, ConnectedShellBackend, ConnectionPlan, ElectricalLink,
                           ElectrostaticProblem, FixedNetCharge, Grounded, MixedRadialBackend,
                           NestedShellBackend, PointCharge, SolidSphere, SphericalShell,
                           prepare_radial_voltages)


def shells():
    return (SphericalShell(.09,.1,Grounded(),name="inner"),
            SphericalShell(.3,.32,FixedNetCharge(1e-9),name="outer"))


@pytest.mark.parametrize("v1,v2", [(10.,3.),(-10.,3.),(3.,3.),(0.,3.),(103.,96.)])
def test_workbook_q32(v1, v2):
    inner, outer = shells()
    original = ElectrostaticProblem(conductors=(outer,inner))
    prepared = prepare_radial_voltages(original, {"inner":v1, "outer":v2})
    outer_final, inner_final = prepared.conductors
    assert outer_final.name == "outer"
    q = 4*pi*EPSILON_0*.1*.3*(v1-v2)/(.3-.1)
    assert inner_final.constraint.charge_c == pytest.approx(q, rel=1e-13, abs=0)
    assert outer_final.constraint.charge_c + q == pytest.approx(4*pi*EPSILON_0*.32*v2, rel=1e-13)
    solved = NestedShellBackend().solve(prepared)
    assert solved.conductor_by_name("inner").potential_v == pytest.approx(v1, abs=1e-12)
    assert solved.conductor_by_name("outer").potential_v == pytest.approx(v2, abs=1e-12)
    link = ElectricalLink(("inner",), mode="grounded")
    switched = ConnectedShellBackend().solve(ConnectionPlan(prepared, (link,)))
    assert switched.after.conductor_by_name("outer").net_charge_c == outer_final.constraint.charge_c
    assert switched.after.conductor_by_name("inner").potential_v == pytest.approx(0, abs=1e-12)
    assert isinstance(original.conductors[1].constraint, Grounded)


def test_cavity_source_and_solid():
    inner, outer = shells()
    source = PointCharge(4e-11, (.025,0,0))
    cavity = ElectrostaticProblem((source,), (inner,outer), permittivity_f_per_m=2*EPSILON_0)
    prepared = prepare_radial_voltages(cavity, {"outer":3., "inner":3.})
    assert prepared.conductors[0].constraint.charge_c == -source.charge_c
    assert prepared.charges == cavity.charges
    solid = ElectrostaticProblem(conductors=(SolidSphere(.1,Grounded(),name="ball"),outer))
    solved = MixedRadialBackend().solve(prepare_radial_voltages(solid, {"outer":0.,"ball":5.}))
    assert solved.conductor_by_name("ball").net_charge_c == pytest.approx(4*pi*EPSILON_0*.15*5, abs=0, rel=1e-13)


def test_rounded_cavity_total_cannot_hide_residual_flux():
    inner, outer = shells()
    sources = (PointCharge(.01, (.02,0,0)), PointCharge(1.602176634e-19, (-.01,0,0)))
    original = ElectrostaticProblem(sources, (inner,outer))
    with pytest.raises(ValueError, match="radial flux"):
        prepare_radial_voltages(original, {"inner":0., "outer":0.})


def test_unresolved_one_ulp_gap_voltage_refused():
    original = ElectrostaticProblem(conductors=shells())
    with pytest.raises(ValueError, match="gap voltage"):
        prepare_radial_voltages(original, {"inner":10., "outer":nextafter(10., 0.)})


def test_ground_target_has_an_absolute_voltage_bound():
    original = ElectrostaticProblem(conductors=shells())
    with pytest.raises(ValueError, match="conductor potential"):
        prepare_radial_voltages(original, {"inner":0., "outer":1e26})


def test_bad_targets_and_geometry():
    inner, outer = shells()
    problem = ElectrostaticProblem(conductors=(inner,outer))
    for targets in ({"inner":1.}, {"inner":True,"outer":1.}, {"inner":1.,"outer":float("nan")}):
        with pytest.raises(ValueError):
            prepare_radial_voltages(problem, targets)
    outside = ElectrostaticProblem((PointCharge(1,(.2,0,0)),), (inner,outer))
    with pytest.raises(ValueError, match="concentric"):
        prepare_radial_voltages(outside, {"inner":1.,"outer":2.})
