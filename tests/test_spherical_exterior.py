from dataclasses import replace
from math import pi

import pytest

from conductor_sim.analytical import ConcentricShellBackend
from conductor_sim.model import ElectrostaticProblem, PointCharge, SphericalShell

from conductor_sim.cavity_gauss import CavityCharge, GaussCavityAssembly, SealedCavity, solve_cavity_gauss
from conductor_sim.model import FixedNetCharge, Grounded
from conductor_sim.solver import UnsupportedProblem
from conductor_sim.spherical_exterior import SphericalCavityExterior
from tools.m39a_gauss_reference import check_gauss_record


def assembly(net, *charges):
    return GaussCavityAssembly("shell", tuple(
        SealedCavity(f"void{i}", (CavityCharge(f"source{i}", q),)) for i, q in enumerate(charges)
    ), net)


def test_q34_neutral_shell_external_potential_and_signed_vector():
    model = assembly(FixedNetCharge(0), -2e-9)
    exterior = SphericalCavityExterior(model, 2, (2, -1, 3))
    kq = -2e-9/(4*pi*model.permittivity_f_per_m)
    probe = (5, 3, 3)
    assert exterior.potential_at(probe) == pytest.approx(kq/5)
    assert exterior.electric_field_at(probe) == pytest.approx((3*kq/125, 4*kq/125, 0))


def test_q42_deposit_changes_cavity_potential_and_preserves_field():
    q, added = 2e-9, 3e-9
    shell = SphericalShell(1, 2, FixedNetCharge(0), name="metal")
    initial = ElectrostaticProblem(charges=(PointCharge(q, (0,0,0)),), conductors=(shell,))
    revised = replace(initial, conductors=(replace(shell, constraint=FixedNetCharge(added)),))
    before, after = (ConcentricShellBackend().solve(state) for state in (initial, revised))
    probe = (0.5,0,0)
    k = 1/(4*pi*initial.permittivity_f_per_m)
    assert before.electric_field_at(probe) == after.electric_field_at(probe)
    assert after.electric_field_at(probe) == pytest.approx((4*k*q, 0, 0))
    assert after.potential_at(probe)-before.potential_at(probe) == pytest.approx(k*added/2)
    for metal_q, native in ((0,before),(added,after)):
        exterior = SphericalCavityExterior(assembly(FixedNetCharge(metal_q), q), 2)
        assert exterior.potential_at((3,0,0)) == pytest.approx(native.potential_at((3,0,0)))


def test_multiple_voids_isolated_versus_grounded_and_interior_refusal():
    isolated = SphericalCavityExterior(assembly(FixedNetCharge(-2e-9), 3e-9, 5e-9), 1)
    earth = SphericalCavityExterior(assembly(Grounded(), 3e-9, 5e-9), 1)
    assert isolated.potential_at((2,0,0)) == pytest.approx(6e-9/(8*pi*isolated.assembly.permittivity_f_per_m))
    assert earth.electric_field_at((2,0,0)) == (0,0,0)
    for point in ((0,0,0),(1,0,0)):
        with pytest.raises(UnsupportedProblem, match="strictly outside"):
            earth.potential_at(point)


def test_independent_export_checker_rejects_a_rounded_away_authored_charge():
    doc = solve_cavity_gauss(assembly(FixedNetCharge(2**53), -2**53)).as_dict()
    doc["conductor"]["authored_isolated_metal_charge_c"] = 2**53 + 1
    with pytest.raises(ValueError, match="exactly representable"):
        check_gauss_record(doc)


def test_lossy_integer_input_cannot_silently_cancel_an_exact_source():
    with pytest.raises(ValueError, match="loses precision"):
        assembly(FixedNetCharge(2**53 + 1), -2**53)


@pytest.mark.parametrize("charges", [(1e308,1e308,-1e308),(1e308,-1e308,1e308)])
def test_face_sum_is_order_independent_when_intermediate_binary64_addition_overflows(charges):
    void = SealedCavity("void", tuple(CavityCharge(f"source{i}", q) for i,q in enumerate(charges)))
    model = GaussCavityAssembly("metal", (void,), FixedNetCharge(0), permittivity_f_per_m=1e308)
    exterior = SphericalCavityExterior(model, 1)
    assert exterior.ledger.outer_face_charge_c == 1e308
    assert exterior.potential_at((2,0,0)) == pytest.approx(1/(8*pi))


@pytest.mark.parametrize("sign", (-1, 1))
def test_outer_charge_preserves_residual_after_cancellation(sign):
    model = assembly(FixedNetCharge(-sign), sign, sign*2**-54)
    ledger = solve_cavity_gauss(model)
    assert ledger.outer_face_charge_c == sign*2**-54
    check_gauss_record(ledger.as_dict())
    potential = SphericalCavityExterior(model, 1).potential_at((2,0,0))
    assert potential == pytest.approx(sign*2**-54/(8*pi*model.permittivity_f_per_m), abs=0)
    assert ledger.exterior_electric_flux_v_m == ledger.outer_face_charge_c/model.permittivity_f_per_m
