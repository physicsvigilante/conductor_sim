"""M21a source-relevant independent physical spot checks and audit integrity.

The inspected questions are not all solved by these selected spot checks.
Do not create synthetic viewer samples for unsupported source situations.
"""
from __future__ import annotations

from math import cos, hypot, pi, sin, sqrt

import pytest

from conductor_sim import (
    EPSILON_0, PointCharge, PrescribedField,
    UniformRingAxisField, UniformElectricField, UniformShellField,
    FixedPointField, TestChargeMotionSpec,
    ElectrostaticProblem, SolidSphere, SphericalShell, FixedNetCharge,
    ConnectionPlan, ElectricalLink, ConnectedShellBackend,
    UnsupportedProblem,
)
from conductor_sim.solver import SingularField
from tools.m21a_audit import validate

K = 1/(4*pi*EPSILON_0)


def test_all_preselected_source_anchors_retain_original_hashes():
    report = validate()
    assert report["cases_examined"] == 29
    assert report["source_counts"] == {
        "02_conductors_half_size.pdf":18,
        "POTENTIAL_CONDUCTORS1.pdf":11}
    assert report["tier_counts"] == {"A":15,"B":5,"C":9}
    assert len(set(report["case_ids"])) == 29


def test_no_source_claims_past_paper_authenticity_without_evidence():
    report = validate()
    by_id = {c["id"]:c for c in report["source_cases"]}
    assert "externally corroborated" in by_id["p15-archive-02"]["provenance"]
    assert "official" not in by_id["p02-single-15"]["provenance"]
    assert all("teaching illustration" in c["provenance"]
               for c in report["source_cases"]
               if c["source"] == "POTENTIAL_CONDUCTORS1.pdf")


def test_advanced_2012_uniform_surface_shell_e_v_graph_reference():
    # Workbook p15 archive IIT-JEE 2012 Q1, a separately observed archive
    # outside the fixed 29-case cohort. A THIN UNIFORMLY CHARGED SHELL,
    # not a volumetric charge, has a discontinuous ideal E at its surface.
    charge, radius = 2e-9, .1
    shell = UniformShellField(charge, radius)
    inside, outside = (.03,0.,0.), (.2,0.,0.)
    assert shell.potential_at(inside) == pytest.approx(K*charge/radius)
    assert shell.electric_field_at(inside) == (0.,0.,0.)
    assert shell.potential_at(outside) == pytest.approx(K*charge/.2)
    assert shell.electric_field_at(outside)[0] == pytest.approx(K*charge/.2**2)
    assert shell.potential_at((radius,0.,0.)) == pytest.approx(K*charge/radius)
    with pytest.raises(SingularField):
        shell.electric_field_at((radius,0.,0.))


def test_advanced_2021_two_point_charge_equipotential_circle():
    # Workbook p16 2021 Paper 1 Q9-10; independently located source in the
    # 2021 JEE Advanced Paper 1 archive. The potential provider already has
    # the physical solution; a 2D equipotential viewer does not.
    field = PrescribedField((
        FixedPointField(-1e-9,(0.,0.,0.)),
        FixedPointField(1e-9/sqrt(3),(2.,0.,0.)),
    ))
    # From V=0: r1 = sqrt(3) r2 -> (x-3)^2+y^2=3.
    R,b = sqrt(3),3.
    for theta in (0,pi/6,pi/3,pi/2,pi,3*pi/2,11*pi/6):
        x,y = b+R*cos(theta),R*sin(theta)
        assert field.potential_at((x,y,0.)) == pytest.approx(0.,abs=1e-12)
        d1 = hypot(x,y)
        d2 = hypot(x-2,y)
        assert d1 == pytest.approx(sqrt(3)*d2,rel=1e-12)
    assert R == pytest.approx(1.7320508075688772)
    assert b == 3.


def test_pdf_p12_Q40_static_ring_work_and_transverse_field_caveat():
    # Both centres lie along world x but applied E is world +y. It does
    # ZERO work between endpoints on y=0 even though a free particle would
    # *not* remain on the ring axis without a physical constraint.
    charge = 10e-6
    field = PrescribedField((
        UniformRingAxisField(+10e-6,.3,center_m=(0.,0.,0.),axis=(1.,0.,0.)),
        UniformRingAxisField(-20e-6,.3,center_m=(.4,0.,0.),axis=(1.,0.,0.)),
        UniformElectricField((0.,1e5,0.)),
    ))
    Ca,Cb=(0.,0.,0.),(.4,0.,0.)
    work = charge*(field.potential_at(Ca)-field.potential_at(Cb))
    # Independent direct ring potential reference, not an integrated path.
    expected = K*charge*((10e-6/.3-20e-6/.5) -
                         (10e-6/.5-20e-6/.3))
    assert work == pytest.approx(expected,rel=2e-13)
    assert work == pytest.approx(3.6,rel=0.002)
    assert field.electric_field_at(Ca)[1] == pytest.approx(1e5)
    with pytest.raises(UnsupportedProblem):
        TestChargeMotionSpec(field,PointCharge(charge,Ca),
                             .1,(0.,0.,0.),.2,.01)


def test_pdf_p03_Q27_connected_solid_sphere_and_shell_9_j_reference():
    # Source-given Q=20 microcoulombs, radii 10 and 20 cm, wire connection.
    # The accepted M14b solver reuses the exact radial field-energy domain.
    q=20e-6
    first=ElectrostaticProblem(conductors=(
        SolidSphere(.1,FixedNetCharge(q),name="inner"),
        SphericalShell(.199999,.2,FixedNetCharge(0.),name="outer"),
    ))
    plan=ConnectionPlan(first,(ElectricalLink(("inner","outer")),))
    computed=ConnectedShellBackend().solve(plan)
    # In the ideal thin-shell limit the released energy is the initial
    # vacuum-gap energy k Q^2/2(1/R - 1/(2R)) = about 9 J.
    expected=K*q*q/2*(1/.1-1/.2)
    assert computed.released_field_energy_j == pytest.approx(expected,rel=2e-4)
    assert computed.released_field_energy_j == pytest.approx(9.,rel=0.002)


def test_pdf_p09_Q27_tangent_neutral_sphere_is_already_analytically_supported():
    from m13c.pdf_cases import check_one
    result=check_one("p09-multiple-27")
    assert result["passed"] is True, result
    assert result["status"] == "checked"


def test_pdf_p08_Q21_grounded_sphere_is_existing_analytical_explorer():
    from m13c.pdf_cases import check_one
    result=check_one("p08-multiple-21")
    assert result["passed"] is True, result
    # A static position sweep does not compute a current-vs-time waveform.
    assert result["frames"] == 7


def test_iit_jee_2012_plates_requires_gravity_not_only_qE():
    # Independent source check from original 2012 vertical plate Q. The
    # real initial acceleration is (qE/m in x, -g in y), at 45 degrees
    # when these magnitudes are equal. This is a gap demonstration, not
    # a claim the production integrator includes gravity.
    mass,gravity,spacing,e = 1.67e-27,10.,.01,1.6e-19
    voltage = mass*gravity*spacing/e
    assert voltage == pytest.approx(1.04375e-9,rel=1e-12)
    assert voltage == pytest.approx(1e-9,rel=0.05)
