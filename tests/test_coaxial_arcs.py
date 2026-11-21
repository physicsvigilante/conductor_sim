from math import cos, hypot, pi, radians, sin

import pytest

from conductor_sim import CircularArcCharge, CoaxialArcAssembly, EPSILON_0


def test_workbook_q33_totals_are_not_angular_densities():
    radius, q = .2, 3e-9
    arcs = (CircularArcCharge(q,radius,90,45), CircularArcCharge(-2*q,radius,135,135),
            CircularArcCharge(3*q,radius,270,120))
    scene = CoaxialArcAssembly(arcs)
    assert scene.potential_v(0) == pytest.approx(q/(2*pi*EPSILON_0*radius), rel=2e-15)
    assert scene.axial_field_v_m(0) == 0.
    # Moving the arc around the axis retains its contribution even when its span changes.
    assert CoaxialArcAssembly((CircularArcCharge(q,radius,0,7),)).potential_v(0) == (
        CoaxialArcAssembly((arcs[0],)).potential_v(0))


def test_shifted_unequal_arcs_against_nonuniform_point_distributions():
    arcs = (CircularArcCharge(1e-9,.2,10,80,-.1), CircularArcCharge(-4e-9,.7,170,140,.4))
    epsilon = 2*EPSILON_0
    scene = CoaxialArcAssembly(arcs,epsilon)
    def reference(z):
        potential = ez = 0.
        for arc in arcs:
            for location, fraction in ((.05,.25),(.35,.125),(.88,.625)):
                theta = radians(arc.start_deg + location*arc.sweep_deg)
                dz = z-arc.center_z_m
                distance = hypot(arc.radius_m*cos(theta),arc.radius_m*sin(theta),dz)
                factor = arc.charge_c*fraction/(4*pi*epsilon)
                potential += factor/distance
                ez += factor*dz/distance**3
        return potential, ez
    for z in (-.9,.1,.85):
        v, ez = reference(z)
        assert scene.potential_v(z) == pytest.approx(v, rel=1e-13, abs=0)
        assert scene.axial_field_v_m(z) == pytest.approx(ez, rel=1e-13, abs=0)
    q = -3e-12
    assert scene.field_work_j(q,-.9,.85) == pytest.approx(q*(reference(-.9)[0]-reference(.85)[0]), rel=1e-13, abs=0)
    assert scene.field_work_j(q,-.9,.85) == -scene.field_work_j(q,.85,-.9)


def test_zero_total_and_domain_boundaries():
    arcs = (CircularArcCharge(1,.2,0,30), CircularArcCharge(-1,.2,180,10))
    scene = CoaxialArcAssembly(arcs)
    assert scene.potential_v(.3) == scene.axial_field_v_m(.3) == 0
    with pytest.raises(ValueError):
        scene.potential_v((.1,0,0))
    for radius, sweep in ((0,5),(1,-1),(1,361)):
        with pytest.raises(ValueError):
            CircularArcCharge(1,radius,0,sweep)


def test_nonzero_endpoint_work_when_rounded_potentials_match():
    scene = CoaxialArcAssembly((CircularArcCharge(1.,1.,0.,90.),))
    z = 2.**-30
    assert scene.potential_v(0.) == scene.potential_v(z)
    # Rationalize 1 - 1/sqrt(1+z²) to retain the small signed change.
    h = hypot(1.,z)
    expected = z*z / (4*pi*EPSILON_0*h*(1+h))
    assert scene.field_work_j(1.,0.,z) == pytest.approx(expected, rel=2e-15, abs=0)


def test_large_integer_axis_preserves_local_offsets():
    center = 2**53 + 1
    translated = CoaxialArcAssembly((CircularArcCharge(1.,1.,0.,90.,center),))
    local = CoaxialArcAssembly((CircularArcCharge(1.,1.,0.,90.),))
    assert translated.axial_field_v_m(center) == 0.
    assert translated.potential_v(center) == local.potential_v(0)
    assert translated.field_work_j(1.,center-1,center) == local.field_work_j(1.,-1,0)


def test_integer_probe_charge_preserved_in_work():
    from fractions import Fraction

    # With these source and medium values, the Coulomb factor cancels.
    scene = CoaxialArcAssembly((CircularArcCharge(4*pi,1.,0.,90.),), 1.)
    charge = 2**53 + 1
    assert scene.field_work_j(charge,0.,.75) == float(Fraction(charge,5))
