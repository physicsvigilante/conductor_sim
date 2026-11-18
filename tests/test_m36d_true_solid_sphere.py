"""M36d: source-authentic single solid conductor and independent JEE p26 checks."""
from __future__ import annotations

from copy import deepcopy
from math import pi

import pytest

from conductor_sim import (
    ConnectedShellBackend, ElectrostaticProblem, FixedNetCharge, Grounded,
    MixedRadialBackend, PointCharge, SolidSphere, SphericalShell,
)
from conductor_sim.model import EPSILON_0
from conductor_sim.solver import SingularField, UnsupportedProblem
from conductor_sim.geometry import contains_metal, physical_boundaries, slice_geometry
from conductor_sim.visualization import SliceSpec
from conductor_sim.pdf_benchmark import load_case
from examples.solid_sphere_connected import scenario as old_connected_scenario
from tools.m36d_true_solid_sphere import run_all
from tools.m36d_solid_reference import check as reference_check


K = 1/(4*pi*EPSILON_0)


def model(q=1e-9, R=.1, center=(0.,0.,0.)):
    return ElectrostaticProblem(conductors=(
        SolidSphere(R, FixedNetCharge(q), center_m=center, name="Sphere A"),
    ))


def test_true_standalone_solid_sphere_r_field_sigma_and_no_inner_face():
    p=model()
    backend=MixedRadialBackend()
    assert backend.supports(p)
    sol=backend.solve(p)
    assert len(sol.radial_shells)==1
    solid=sol.radial_shells[0]
    assert isinstance(solid,SolidSphere)
    assert solid.inner_radius_m==0.
    assert len(physical_boundaries(solid))==1
    assert physical_boundaries(solid)[0].radius_m==.1
    assert contains_metal(solid,(0,0,0)) and contains_metal(solid,(.05,0,0))
    assert not contains_metal(solid,(.15,0,0))
    c=sol.conductor_by_name("Sphere A")
    assert c.net_charge_c==pytest.approx(1e-9)
    assert c.inner_surface_charge_c==0.
    assert c.outer_surface_charge_c==pytest.approx(1e-9)
    assert c.potential_v==pytest.approx(K*1e-9/.1)
    for point in [(0,0,0),(.05,0,0),(0,0,.03)]:
        assert sol.electric_field_at(point)==(0.,0.,0.)
        assert sol.potential_at(point)==pytest.approx(K*1e-9/.1)
    assert sol.potential_at((.15,0,0))==pytest.approx(K*1e-9/.15)
    assert sol.electric_field_at((.15,0,0))==pytest.approx((K*1e-9/.15**2,0,0))
    assert sol.electric_field_at((-.2,0,0))==pytest.approx((-K*1e-9/.2**2,0,0))
    for surface_point in ((.1,0,0),(0,0,.1),(0,-.1,0)):
        assert sol.surface_charge_density_at("Sphere A","outer",surface_point)==pytest.approx(1e-9/(4*pi*.1**2))
    with pytest.raises(ValueError,match="no inner surface"):
        sol.surface_charge_density_at("Sphere A","inner",(.1,0,0))
    with pytest.raises(ValueError,match="named conductor surface"):
        sol.surface_charge_density_at("Sphere A","outer",(.15,0,0))
    with pytest.raises(KeyError):
        sol.conductor_by_name("unknown")
    for x in (-.1,+.1):
        with pytest.raises(SingularField,match="one-sided"):
            sol.electric_field_at((x,0,0))


@pytest.mark.parametrize("charge",[-2e-9,0.,+1e-9])
def test_signed_sphere_and_true_zero_model(charge):
    sol=MixedRadialBackend().solve(model(q=charge))
    assert sol.conductor_by_name("Sphere A").net_charge_c==pytest.approx(charge)
    assert sol.potential_at((0.,0.,0.))==pytest.approx(K*charge/.1)
    assert sol.potential_at((.2,0,0))==pytest.approx(K*charge/.2)
    assert sol.electric_field_at((.2,0,0))[0]==pytest.approx(K*charge/.2**2)
    assert sol.surface_charge_density_at("Sphere A","outer",(.1,0,0))==pytest.approx(charge/(4*pi*.1**2))


def test_true_grounded_no_point_source_is_charge_free_not_fake_fixed_charge():
    p=ElectrostaticProblem(conductors=(
        SolidSphere(.08,Grounded(),name="Sphere A"),
    ))
    backend=MixedRadialBackend()
    assert backend.supports(p)
    result=backend.solve(p)
    c=result.conductor_by_name("Sphere A")
    assert c.potential_v==0.
    assert c.net_charge_c==0.
    assert c.inner_surface_charge_c==0.
    assert c.outer_surface_charge_c==0.
    for r in (0,.04,.12,.36):
        assert result.potential_at((r,0,0))==0.
        assert result.electric_field_at((r,0,0))==(0.,0.,0.)
    assert result.surface_charge_density_at("Sphere A","outer",(.08,0,0))==0.


def test_translated_true_solid_charge_geometry_and_directed_field():
    centre=(.025,-.013,.04)
    R=.12;q=-2e-9
    p=model(q=q,R=R,center=centre)
    r=MixedRadialBackend().solve(p)
    assert r.potential_at(centre)==pytest.approx(K*q/R)
    assert r.potential_at((centre[0]+1.5*R,centre[1],centre[2]))==pytest.approx(K*q/(1.5*R))
    assert r.electric_field_at((centre[0]-2*R,centre[1],centre[2]))==pytest.approx((-K*q/(2*R)**2,0,0))
    assert r.surface_charge_density_at("Sphere A","outer",(centre[0],centre[1]+R,centre[2]))==pytest.approx(q/(4*pi*R**2))
    geom=slice_geometry(p.conductors[0],SliceSpec.meridian(p,extent_m=.3))
    assert geom["kind"]=="sphere" and geom["inner_radius_m"]==0
    assert geom["outer_radius_m"]==pytest.approx(R)


@pytest.mark.parametrize("bad",[
    ElectrostaticProblem(conductors=(SphericalShell(.09,.1,FixedNetCharge(1e-9),name="shell"),)),
    ElectrostaticProblem(conductors=(SolidSphere(.1,FixedNetCharge(1e-9),name="Sphere A"),),
                         charges=(PointCharge(1e-9,(0,0,0)),)),
    ElectrostaticProblem(conductors=(SolidSphere(.1,FixedNetCharge(1e-9),name="Sphere A"),),
                         charges=(PointCharge(1e-9,(0,0,.25)),)),
    ElectrostaticProblem(conductors=(SolidSphere(.1,FixedNetCharge(1e-9),name="Sphere A"),
                                    SolidSphere(.2,FixedNetCharge(0),name="Sphere B"))),
])
def test_invalid_standalone_solid_domains_remain_refused(bad):
    backend=MixedRadialBackend()
    assert not backend.supports(bad)
    with pytest.raises(UnsupportedProblem):
        backend.solve(bad)


def test_historical_m14b_and_m6_hollow_source_remain_separate():
    # New one-sphere admission must not change already qualified mixed
    # connected sphere plus shell physics or rewrite historical M6.
    plan=old_connected_scenario("pdf-p03-q27")
    backend=MixedRadialBackend()
    assert backend.supports(plan.initial)
    connection=ConnectedShellBackend().solve(plan)
    assert connection.released_field_energy_j==pytest.approx(8.9875,rel=4e-4)
    old_case=load_case("p26-conductor-potential")
    authored=old_case.author()
    assert len(authored.states)==1
    assert isinstance(authored.states[0][1].conductors[0],SphericalShell)
    assert old_case.source_anchor=="Illustration 1"


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    folder=tmp_path_factory.mktemp("m36d-authentic-solid-sphere")
    data=run_all(folder)
    return folder,data


def test_source_authentic_real_viewer_and_three_actual_native_worlds(native):
    folder,data=native
    assert data["original_source_apparatus_count"]==1
    assert data["teacher_hypothetical_worlds"]==2
    assert [x["case_role"] for x in data["native_states"]]==[
        "source_original_shape_teacher_numeric_example",
        "teacher_changed_radius_charge_and_centre",
        "teacher_grounded_no_external_sources",
    ]
    import json
    assert json.loads((folder/"native-solid-sphere.json").read_text())==data
    html=(folder/data["viewer_file"]).read_text(encoding="utf-8")
    assert '<section id="m36d-source-authentic-solid"' in html
    assert "genuinely SOLID conducting sphere" in html
    assert "NOT the historical hollow-shell proxy" in html
    assert "TEACHER HYPOTHETICAL" in html
    assert "canvas" in html
    assert all(f["inner_radius_m"]==0. for f in data["actual_rendered_geometry"])


def test_all_three_real_native_solids_pass_source_free_SI_reference(native):
    _folder,data=native
    r=reference_check(data)
    assert r["qualified_original_jee_solid_sphere_apparatus"]==1
    assert r["teacher_real_physical_revisions"]==2
    assert r["actual_solid_source_states_independently_checked"]==3


@pytest.mark.parametrize("state_index,path,bad",[
    (0,("physical_shape_type",),"SphericalShell"),
    (0,("physical_geometry","inner_radius_m"),.09),
    (0,("observed_conductor_results",0,"inner_surface_charge_c"),1e-9),
    (0,("observed_conductor_results",0,"outer_surface_charge_c"),-1e-9),
    (0,("native_samples","half_radius_potential_v"),-100.),
    (0,("native_samples","three_half_radius_potential_v"),0.),
    (0,("native_samples","metal_half_radius_e_v_m",0),20.),
    (0,("native_samples","external_neg_x_e_v_m",0),500.),
    (0,("native_samples","outer_sigma_c_m2"),0.),
    (1,("case_role",),"source_original_shape_teacher_numeric_example"),
    (1,("physical_geometry","center_m",0),0.),
    (1,("authored_charge_c",),2e-9),
    (1,("native_samples","external_pos_x_e_v_m",0),+500.),
    (2,("constraint_type",),"FixedNetCharge"),
    (2,("observed_conductor_results",0,"net_charge_c"),1e-9),
])
def test_source_free_checker_rejects_forged_solid_geometry_field_and_attribution(
    native,state_index,path,bad,
):
    _folder,data=native
    forged=deepcopy(data)
    target=forged["native_states"][state_index]
    for key in path[:-1]:
        target=target[key]
    target[path[-1]]=bad
    with pytest.raises(AssertionError):
        reference_check(forged)


@pytest.mark.parametrize("path,bad",[
    (("printed_source","page"),27),
    (("printed_source","original_shape"),"hollow sphere"),
    (("original_source_apparatus_count",),3),
    (("physical_time_evolution",),True),
    (("actual_rendered_geometry",0,"inner_radius_m"),.09),
    (("actual_rendered_geometry",1,"outer_radius_m"),.10),
    (("actual_rendered_geometry",1,"name"),"Fake source"),
])
def test_manifest_original_jee_source_or_drawn_solid_forgery_refused(native,path,bad):
    _folder,data=native
    forged=deepcopy(data)
    target=forged
    for key in path[:-1]:
        target=target[key]
    target[path[-1]]=bad
    with pytest.raises(AssertionError):
        reference_check(forged)
