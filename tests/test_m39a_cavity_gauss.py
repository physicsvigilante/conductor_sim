"""M39a: new composable sealed-cavity Gauss integrals, not local 3D fields."""
from copy import deepcopy
from dataclasses import replace
from math import isclose

import pytest

from conductor_sim.model import EPSILON_0, FixedNetCharge, Grounded
from conductor_sim.solver import UnsupportedProblem
from conductor_sim.cavity_gauss import (
    CavityCharge, SealedCavity, GaussCavityAssembly,
    CavityFaceTotal, GaussCavityLedger, solve_cavity_gauss,
    validate_cavity_gauss_ledger,
)
from tools.m39a_gauss_example import sample_assembly, run_all
from tools.m39a_gauss_reference import check_gauss_record


def test_real_authored_3cavity_isolated_signed_totals_and_separate_grounded(tmp_path):
    original=sample_assembly()
    outputs=run_all(tmp_path)
    isolated=outputs["isolated"]
    grounded=outputs["grounded"]
    for label in ("isolated","grounded"):
        oracle=check_gauss_record(outputs[label])
        assert oracle["cavities_independently_checked"]==3
        assert oracle["point_source_identities_checked"]==4
        assert oracle["original_workbook_items_newly_solved"]==0
        assert oracle["local_3d_fields_qualified"] is False
        assert oracle["physical_containment_geometrically_verified"] is False
    assert isolated["total_interior_source_charge_c"]==pytest.approx(+2e-9)
    assert [c["cavity_id"] for c in isolated["cavities"]]==["cavity_A","cavity_B","cavity_C"]
    assert isolated["cavities"][0]["induced_inner_face_total_charge_c"]==0.
    assert isolated["cavities"][1]["induced_inner_face_total_charge_c"]==pytest.approx(+1e-9)
    assert isolated["cavities"][2]["induced_inner_face_total_charge_c"]==pytest.approx(-3e-9)
    assert isolated["conductor"]["equilibrium_net_metal_charge_c"]==pytest.approx(+1e-9)
    assert isolated["outer_face_total_charge_c"]==pytest.approx(+3e-9)
    assert isolated["exterior_electric_flux_v_m"]==pytest.approx(3e-9/EPSILON_0)
    assert isolated["exterior_electric_field_zero_theorem"] is False
    assert grounded["conductor"]["equilibrium_net_metal_charge_c"]==pytest.approx(-2e-9)
    assert grounded["outer_face_total_charge_c"]==0.
    assert grounded["exterior_electric_flux_v_m"]==0.
    assert grounded["exterior_electric_field_zero_theorem"] is True
    assert grounded["conductor"]["authored_isolated_metal_charge_c"] is None
    assert original.constraint.charge_c==pytest.approx(+1e-9)
    assert original.cavities[0].sources[0].charge_c==pytest.approx(+2e-9)


@pytest.mark.parametrize("n",range(1,9))
def test_independent_gauss_for_one_to_eight_cavities(n):
    cs=tuple(SealedCavity(f"cavity_{i}",(
        CavityCharge(f"q_{i}_plus",(i+1)*1e-9),
        CavityCharge(f"q_{i}_minus",-(i+1)*1e-9),
    )) for i in range(n))
    model=GaussCavityAssembly("metal_1",cs,FixedNetCharge(+4e-9))
    ledger=solve_cavity_gauss(model)
    assert all(c.enclosed_source_charge_c==0 and
               c.induced_inner_face_charge_c==0 for c in ledger.faces)
    assert ledger.outer_face_charge_c==pytest.approx(4e-9)
    assert ledger.total_source_charge_c==0.
    assert check_gauss_record(ledger.as_dict())["point_source_identities_checked"]==2*n
    grounded=solve_cavity_gauss(replace(model,constraint=Grounded()))
    assert grounded.exterior_field_zero_theorem is True
    assert grounded.outer_face_charge_c==0.


def test_empty_cavities_and_one_source_are_genuine_and_conserve_charge():
    spec=GaussCavityAssembly("metal_X",(
        SealedCavity("empty_1"),
        SealedCavity("charged", (CavityCharge("q",-.5e-9),)),
        SealedCavity("empty_2"),
    ),FixedNetCharge(+2e-9))
    result=solve_cavity_gauss(spec)
    assert result.faces[0].induced_inner_face_charge_c==0.
    assert result.faces[1].induced_inner_face_charge_c==pytest.approx(+.5e-9)
    assert result.faces[2].induced_inner_face_charge_c==0.
    assert result.outer_face_charge_c==pytest.approx(+1.5e-9)
    assert result.net_metal_charge_c==pytest.approx(+2e-9)
    check_gauss_record(result.as_dict())


def test_true_immutable_teacher_revision_changes_source_membership_without_time_history():
    parent=sample_assembly()
    baseline=solve_cavity_gauss(parent).as_dict()
    a,b,c=parent.cavities
    moved=b.sources[0]
    child=replace(parent,cavities=(
        a,replace(b,sources=()),replace(c,sources=c.sources+(moved,))
    ))
    updated=solve_cavity_gauss(child).as_dict()
    assert baseline["cavities"][1]["source_charge_sum_c"]==-1e-9
    assert updated["cavities"][1]["source_charge_sum_c"]==0.
    assert updated["cavities"][2]["source_charge_sum_c"]==pytest.approx(+2e-9)
    assert updated["outer_face_total_charge_c"]==pytest.approx(baseline["outer_face_total_charge_c"])
    assert solve_cavity_gauss(parent).as_dict()==baseline
    assert updated["time_evolution_defined"] is False
    assert check_gauss_record(updated)["local_3d_fields_qualified"] is False


def test_permutation_of_order_of_sources_cannot_change_integrated_total():
    parent=sample_assembly()
    reversed_cavity=replace(parent.cavities[0],sources=tuple(reversed(parent.cavities[0].sources)))
    new=replace(parent,cavities=(reversed_cavity,)+parent.cavities[1:])
    original=solve_cavity_gauss(parent)
    other=solve_cavity_gauss(new)
    assert other.total_source_charge_c==original.total_source_charge_c
    assert other.outer_face_charge_c==original.outer_face_charge_c
    for x,y in zip(original.faces,other.faces):
        assert x.induced_inner_face_charge_c==y.induced_inner_face_charge_c
    check_gauss_record(other.as_dict())


@pytest.mark.parametrize("call",[
    lambda x:x.electric_field_at((0.,0.,0.)),
    lambda x:x.potential_at((0.,0.,0.)),
    lambda x:x.surface_charge_density_at("cavity_A",(0.,0.,0.)),
    lambda x:x.force_on_source("source_A_positive"),
    lambda x:x.ground_wire_current(1.),
])
def test_no_fictional_geometry_field_force_or_chronology(call):
    solution=solve_cavity_gauss(sample_assembly())
    with pytest.raises(UnsupportedProblem,match="topology-only|static Gauss ledgers"):
        call(solution)


@pytest.mark.parametrize("charge",[
    True,False,float("nan"),float("inf"),float("-inf"),"1e-9",None
])
def test_invalid_raw_source_charges_rejected(charge):
    with pytest.raises((TypeError,ValueError)):
        CavityCharge("q",charge)


@pytest.mark.parametrize("eps",[
    False,True,0.,-1.,float("nan"),float("inf"),"8.85e-12"
])
def test_invalid_homogeneous_permittivity_rejected(eps):
    with pytest.raises((TypeError,ValueError)):
        GaussCavityAssembly("M",(SealedCavity("A"),),Grounded(),eps)


@pytest.mark.parametrize("ident",["", "1q", "with space", "a/b", "a<script>", "q\nx", 3])
def test_source_identity_requires_stable_id(ident):
    with pytest.raises((TypeError,ValueError)):
        CavityCharge(ident,1e-9)


def test_invalid_topology_and_constraints_refused():
    with pytest.raises(UnsupportedProblem):
        GaussCavityAssembly("M",(),Grounded())
    with pytest.raises(UnsupportedProblem):
        GaussCavityAssembly("M",tuple(SealedCavity(f"C{i}") for i in range(9)),Grounded())
    with pytest.raises((TypeError,UnsupportedProblem)):
        GaussCavityAssembly("M",(SealedCavity("A"),),object())
    with pytest.raises(ValueError):
        GaussCavityAssembly("M",(SealedCavity("A"),SealedCavity("A")),Grounded())
    with pytest.raises(ValueError):
        GaussCavityAssembly("M",(SealedCavity("M"),),Grounded())
    with pytest.raises(ValueError):
        GaussCavityAssembly("M",(SealedCavity("A",(CavityCharge("A",1e-9),)),),Grounded())
    with pytest.raises(ValueError):
        GaussCavityAssembly("M",(
            SealedCavity("A",(CavityCharge("q",1e-9),)),
            SealedCavity("B",(CavityCharge("q",-1e-9),)),
        ),Grounded())
    with pytest.raises(UnsupportedProblem):
        GaussCavityAssembly("M",(
            SealedCavity("A",tuple(CavityCharge(f"q{i}",1e-9) for i in range(16))),
            SealedCavity("B",(CavityCharge("q_extra",1e-9),)),
        ),Grounded())
    with pytest.raises(TypeError):
        GaussCavityAssembly("M","cavity_A",Grounded())
    with pytest.raises(TypeError):
        GaussCavityAssembly("M",(SealedCavity("A"),),Grounded(), exterior_charges=(1e-9,))


def test_legacy_boolean_fixed_net_constraint_refused_by_new_boundary():
    with pytest.raises(ValueError,match="isolated metal charge"):
        GaussCavityAssembly("M",(SealedCavity("A"),),FixedNetCharge(False))


def test_overflowing_charge_or_exterior_flux_must_be_refused():
    with pytest.raises(UnsupportedProblem,match="representable"):
        solve_cavity_gauss(GaussCavityAssembly("M",(
            SealedCavity("A",(CavityCharge("q1",1e308),CavityCharge("q2",1e308))),
        ),Grounded()))
    with pytest.raises(UnsupportedProblem,match="flux"):
        solve_cavity_gauss(GaussCavityAssembly("M",(
            SealedCavity("A",(CavityCharge("q1",1e308),)),
        ),FixedNetCharge(0.),1e-320))


def test_mutated_nested_frozen_source_cannot_bypass_solve_validation():
    spec=sample_assembly()
    object.__setattr__(spec.cavities[0].sources[0],"charge_c",False)
    with pytest.raises(ValueError,match="charge"):
        solve_cavity_gauss(spec)


@pytest.mark.parametrize("key,value",[
    ("total_source_charge_c",False),
    ("total_source_charge_c",0.),
    ("net_metal_charge_c",False),
    ("outer_face_charge_c",0.),
    ("exterior_electric_flux_v_m",float("nan")),
    ("exterior_field_zero_theorem",True),
])
def test_native_gauss_saved_result_integrity_reexecutes_and_refuses(key,value):
    result=solve_cavity_gauss(sample_assembly())
    forged=replace(result,**{key:value})
    with pytest.raises(ValueError):
        validate_cavity_gauss_ledger(forged)
    with pytest.raises(ValueError):
        forged.as_dict()


def test_forged_native_face_charge_refused():
    authentic=solve_cavity_gauss(sample_assembly())
    row=authentic.faces[1]
    fake=replace(row,induced_inner_face_charge_c=False)
    forged=replace(authentic,faces=(authentic.faces[0],fake,authentic.faces[2]))
    with pytest.raises(ValueError):
        forged.as_dict()


@pytest.mark.parametrize("fake_id",["cavity_A","nonexistent_cavity","source_B"])
def test_forged_native_face_identity_refused_even_when_charge_totals_match(fake_id):
    authentic=solve_cavity_gauss(sample_assembly())
    altered=replace(authentic.faces[1],cavity_id=fake_id)
    forged=replace(authentic,faces=(authentic.faces[0],altered,authentic.faces[2]))
    with pytest.raises(ValueError,match="face identities/order"):
        validate_cavity_gauss_ledger(forged)
    with pytest.raises(ValueError,match="face identities/order"):
        forged.as_dict()


def test_missing_or_extra_saved_cavity_face_refused_before_serialization():
    authentic=solve_cavity_gauss(sample_assembly())
    truncated=replace(authentic,faces=authentic.faces[:2])
    with pytest.raises(ValueError,match="face identities/order"):
        truncated.as_dict()


@pytest.mark.parametrize("path,bad",[
    (("source_role",),"original_workbook"),
    (("membership_provenance",),"all placements geometrically certified"),
    (("original_workbook_items_newly_solved",),True),
    (("units","charge"),"nC"),
    (("geometry_verified",),True),
    (("time_evolution_defined",),True),
    (("exterior_electric_field_zero_theorem",),True),
    (("cavities",0,"sealed"),False),
    (("cavities",0,"source_charge_sum_c"),1e-9),
    (("cavities",1,"induced_inner_face_total_charge_c"),False),
    (("cavities",1,"authored_sources",0,"charge_c"),True),
    (("cavities",1,"authored_sources",0,"source_id"),"source_A_positive"),
    (("total_interior_source_charge_c",),False),
    (("outer_face_total_charge_c",),0.),
    (("exterior_electric_flux_v_m",),False),
    (("conductor","equilibrium_net_metal_charge_c"),2e-9),
    (("conductor","constraint"),"grounded_zero_potential"),
])
def test_independent_checker_rejects_forged_math_provenance_and_identities(path,bad):
    genuine=solve_cavity_gauss(sample_assembly()).as_dict()
    fake=deepcopy(genuine)
    target=fake
    for part in path[:-1]:
        target=target[part]
    target[path[-1]]=bad
    with pytest.raises(ValueError):
        check_gauss_record(fake)


def test_grounded_checker_rejects_fake_charge_and_false_outer_field():
    original=solve_cavity_gauss(replace(sample_assembly(),constraint=Grounded())).as_dict()
    check_gauss_record(original)
    original["outer_face_total_charge_c"]=1e-12
    with pytest.raises(ValueError):
        check_gauss_record(original)


def test_distinguishes_integral_charge_from_unknown_local_sigma_and_no_jee_count():
    actual=solve_cavity_gauss(sample_assembly()).as_dict()
    assert actual["source_role"]=="teacher_hypothetical_topology_gauss_only"
    assert any("bounded connected" in phrase for phrase in actual["assumptions"])
    assert actual["original_workbook_items_newly_solved"]==0
    assert "no geometry or containment was verified" in actual["membership_provenance"]
    assert "surface charge density sigma(x)" in actual["unsupported"][2]
    assert actual["geometry_verified"] is False
    assert actual["time_evolution_defined"] is False
    assert "source_position_defined" in actual
    assert actual["source_position_defined"] is False
