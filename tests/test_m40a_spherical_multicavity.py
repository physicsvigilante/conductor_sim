"""M40a geometry-certified genuine multicavity induced sigma, SI and refusals."""
from copy import deepcopy
from dataclasses import replace
from math import hypot, pi, sqrt

import pytest

from conductor_sim.cavity_gauss import CavityCharge, SealedCavity, GaussCavityAssembly
from conductor_sim.model import (
    EPSILON_0, ElectrostaticProblem, FixedNetCharge, Grounded,
    PointCharge, SphericalShell,
)
from conductor_sim.multi_image import MultiImageShellBackend
from conductor_sim.solver import UnsupportedProblem
from conductor_sim.spherical_multicavity import (
    NamedSourcePosition, SphericalVoid, SphericalMetalGeometry,
    SphericalSigmaModel, solve_spherical_sigma,
)
from tools.m40a_spherical_example import teacher_model, run_all
from tools.m40a_spherical_reference import check_record


def test_actual_three_spherical_cavities_and_two_genuine_constraints(tmp_path):
    parent=teacher_model()
    outputs=run_all(tmp_path)
    for mode in ("isolated","grounded"):
        record=outputs[mode]
        witness=check_record(record)
        assert witness["cavities_checked"]==3
        assert witness["point_sources_checked"]==4
        assert witness["sampled_local_surface_densities_checked"]==7
        assert record["original_workbook_items_newly_solved"]==0
        assert record["geometry_qualified_spherical_only"] is True
        assert record["irregular_3d_fields_qualified"] is False
        assert record["gauss"]["source_role"]=="teacher_hypothetical_topology_gauss_only"
    isolated,grounded=outputs["isolated"],outputs["grounded"]
    assert isolated["gauss"]["total_interior_source_charge_c"]==pytest.approx(-2e-9)
    assert isolated["gauss"]["outer_face_total_charge_c"]==pytest.approx(-1e-9)
    assert grounded["gauss"]["outer_face_total_charge_c"]==0.0
    assert grounded["samples"][-1]["sigma_c_per_m2"]==0.
    assert isolated["samples"][0]["sigma_c_per_m2"]!=0.
    assert isolated["samples"][1]["sigma_c_per_m2"]!=0.
    assert isolated["samples"][0]["sigma_c_per_m2"]!=isolated["samples"][1]["sigma_c_per_m2"]
    assert isolated["samples"][4]["sigma_c_per_m2"]==0.
    assert isolated["samples"][0]["sigma_c_per_m2"]==grounded["samples"][0]["sigma_c_per_m2"]
    assert parent.gauss_assembly.constraint.charge_c==1e-9
    assert solve_spherical_sigma(parent).gauss_ledger.assembly==parent.gauss_assembly


def test_centered_and_displaced_poles_against_separate_analytic_limit():
    q=2e-9
    radius=.6
    cavity=SealedCavity("A",(CavityCharge("s",q),))
    gauss=GaussCavityAssembly("Metal",(cavity,),FixedNetCharge(0.))
    hole=SphericalVoid("A",(0.,0.,0.),radius,(NamedSourcePosition("s",(0.,0.,0.)),))
    geometry=SphericalMetalGeometry((0.,0.,0.),3.,(hole,))
    uniform=solve_spherical_sigma(SphericalSigmaModel(gauss,geometry))
    sigma=-q/(4*pi*radius**2)
    assert uniform.inner_surface_charge_density("A",(1.,0.,0.))==pytest.approx(sigma,rel=2e-15)
    assert uniform.inner_surface_charge_density("A",(0.,0.,1.))==pytest.approx(sigma,rel=2e-15)
    shifted=solve_spherical_sigma(SphericalSigmaModel(
        gauss,replace(geometry,voids=(replace(hole,positions=(NamedSourcePosition("s",(radius/2,0.,0.)),)),))
    ))
    assert shifted.inner_surface_charge_density("A",(1.,0.,0.))==pytest.approx(
        -6*q/(4*pi*radius**2),rel=2e-14
    )
    assert shifted.inner_surface_charge_density("A",(-1.,0.,0.))==pytest.approx(
        -(2/9)*q/(4*pi*radius**2),rel=2e-14
    )
    assert shifted.gauss_ledger.faces[0].induced_inner_face_charge_c==-q


def test_independent_angle_quadrature_integrates_complete_inner_face_to_gauss():
    # Independent analytic expression is evaluated with midpoint surface
    # integration; no production local-sigma function computes expected area.
    radius=.7; q=3e-9; displacement=.31
    cavity=SealedCavity("C",(CavityCharge("q",q),))
    gauss=GaussCavityAssembly("Metal",(cavity,),FixedNetCharge(+1e-9))
    geometry=SphericalMetalGeometry((0.,0.,0.),3.,(
        SphericalVoid("C",(0.,0.,0.),radius,
                       (NamedSourcePosition("q",(displacement,0.,0.)),)),
    ))
    solution=solve_spherical_sigma(SphericalSigmaModel(gauss,geometry))
    N=2400
    expected=0.
    for k in range(N):
        mu=-1+(k+.5)*2/N
        separation=sqrt(radius*radius+displacement*displacement-2*radius*displacement*mu)
        density=-q*(radius**2-displacement**2)/(4*pi*radius*separation**3)
        expected+=density*2*pi*radius**2*(2/N)
    assert expected==pytest.approx(-q,rel=3e-7,abs=0.)
    assert solution.gauss_ledger.faces[0].induced_inner_face_charge_c==-q


def test_single_cavity_limit_matches_preexisting_released_kelvin_backend():
    q=4e-9; center=(.1,.2,-.3);a=.6;source=(.28,.24,-.2)
    gauss=GaussCavityAssembly("Metal",(SealedCavity("A",(CavityCharge("q",q),)),),
                              FixedNetCharge(0.))
    geometry=SphericalMetalGeometry(center,3.,(
        SphericalVoid("A",center,a,(NamedSourcePosition("q",source),)),
    ))
    new=solve_spherical_sigma(SphericalSigmaModel(gauss,geometry))
    old=MultiImageShellBackend().solve(ElectrostaticProblem(
        (PointCharge(q,source),),
        (SphericalShell(a,3.,FixedNetCharge(0.),center_m=center),),
        EPSILON_0,
    ))
    assert new.inner_surface_charge_density("A",(1.,0.,0.))==pytest.approx(
        old.surface_charge_density_at("inner",(center[0]+a,center[1],center[2])),
        rel=3e-13,
    )


def test_source_order_and_parent_preserved_and_signed_neutral_not_empty():
    parent=teacher_model()
    before=solve_spherical_sigma(parent)
    initial=before.inner_surface_charge_density("NeutralButNonuniform",(1.,0.,0.))
    void=parent.geometry.voids[0]
    revision=replace(parent,geometry=replace(parent.geometry,voids=(
        replace(void,positions=tuple(reversed(void.positions))),
    )+parent.geometry.voids[1:]))
    assert solve_spherical_sigma(revision).inner_surface_charge_density(
        "NeutralButNonuniform",(1.,0.,0.))==initial
    assert solve_spherical_sigma(parent).inner_surface_charge_density(
        "NeutralButNonuniform",(1.,0.,0.))==initial
    assert initial!=0.
    assert before.inner_surface_charge_density("EmptyCavity",(1.,0.,0.))==0.
    assert before.gauss_ledger.faces[0].induced_inner_face_charge_c==0.


def test_outer_sphere_uniformity_and_grounded_invariance_of_inner_sigma():
    parent=teacher_model()
    isolated=solve_spherical_sigma(parent)
    grounded=solve_spherical_sigma(replace(parent,gauss_assembly=replace(
        parent.gauss_assembly,constraint=Grounded()
    )))
    assert isolated.outer_surface_charge_density((1.,0.,0.))==pytest.approx(-1e-9/(4*pi*25),rel=1e-14)
    assert isolated.outer_surface_charge_density((0.,0.,1.))==isolated.outer_surface_charge_density((1.,0.,0.))
    assert grounded.outer_surface_charge_density((0.,1.,0.))==0.
    for void in parent.geometry.voids:
        assert isolated.inner_surface_charge_density(void.cavity_id,(1.,0.,0.))==grounded.inner_surface_charge_density(void.cavity_id,(1.,0.,0.))


@pytest.mark.parametrize("normal",[
    (True,0.,0.),(0.,0.,0.),(2.,0.,0.),(float("nan"),0.,0.),
    (float("inf"),0.,0.),("1",0.,0.),(1.,0.),
])
def test_invalid_nonunit_or_nonfinite_or_boolean_normals_are_rejected(normal):
    solution=solve_spherical_sigma(teacher_model())
    with pytest.raises(ValueError):
        solution.inner_surface_charge_density("NegativeCavity",normal)
    with pytest.raises(ValueError):
        solution.outer_surface_charge_density(normal)


def test_unknown_cavity_and_local_field_force_time_queries_fail_closed():
    solution=solve_spherical_sigma(teacher_model())
    with pytest.raises(ValueError):
        solution.inner_surface_charge_density("unknown",(1.,0.,0.))
    for query in (solution.electric_field_at,solution.potential_at,
                  solution.force_on_source,solution.ground_wire_current):
        with pytest.raises(UnsupportedProblem):
            query("unsupported")


def test_geometry_rejects_missing_or_swapped_source_ids_even_with_equal_total():
    parent=teacher_model()
    a=parent.geometry.voids[0]
    bad=replace(a,positions=(
        NamedSourcePosition("plus_B",a.positions[0].position_m),
        a.positions[1],
    ))
    with pytest.raises(ValueError,match="source placements"):
        SphericalSigmaModel(parent.gauss_assembly,replace(parent.geometry,voids=(bad,)+parent.geometry.voids[1:]))
    with pytest.raises(ValueError,match="identities"):
        SphericalSigmaModel(parent.gauss_assembly,replace(parent.geometry,voids=(
            replace(a,cavity_id=parent.geometry.voids[1].cavity_id),)+parent.geometry.voids[1:]))


@pytest.mark.parametrize("mutate",[
    lambda g:replace(g,voids=(replace(g.voids[0],center_m=(4.7,0.,0.)),)+g.voids[1:]),
    lambda g:replace(g,voids=(replace(g.voids[0],center_m=g.voids[1].center_m),)+g.voids[1:]),
    lambda g:replace(g,voids=(replace(g.voids[0],positions=(
        NamedSourcePosition("plus_A",(0.,0.,0.)),)+g.voids[0].positions[1:]),)+g.voids[1:]),
])
def test_nonphysical_containment_collision_or_escaped_source_refused(mutate):
    parent=teacher_model()
    with pytest.raises(UnsupportedProblem):
        SphericalSigmaModel(parent.gauss_assembly,mutate(parent.geometry))


def test_frozen_saved_native_source_position_tampering_is_revalidated():
    sol=solve_spherical_sigma(teacher_model())
    object.__setattr__(sol.model.geometry.voids[0].positions[0],"position_m",(True,0.,0.))
    with pytest.raises(ValueError):
        sol.as_dict()


@pytest.mark.parametrize("path,new",[
    (("source_role",),"verified_original_jee_solution"),
    (("original_workbook_items_newly_solved",),True),
    (("geometry_qualified_spherical_only",),False),
    (("irregular_3d_fields_qualified",),True),
    (("units","surface_density"),"C"),
    (("samples",0,"sigma_c_per_m2"),0.),
    (("samples",0,"sigma_c_per_m2"),False),
    (("samples",0,"surface_id"),"EmptyCavity"),
    (("voids",0,"source_positions",0,"source_id"),"minus_B"),
    (("voids",0,"source_positions",0,"position_m",0),False),
])
def test_independent_si_checker_refuses_forged_density_geometry_provenance(path,new):
    genuine=solve_spherical_sigma(teacher_model()).as_dict((
        ("NeutralButNonuniform",(1.,0.,0.)),("outer",(1.,0.,0.)),
    ))
    assert check_record(genuine)["sampled_local_surface_densities_checked"]==2
    forged=deepcopy(genuine)
    item=forged
    for part in path[:-1]:
        item=item[part]
    item[path[-1]]=new
    with pytest.raises(ValueError):
        check_record(forged)


def test_subnormal_nonzero_source_denied_by_narrow_local_density_regime():
    original=teacher_model()
    cav=original.gauss_assembly.cavities
    revised=replace(cav[0],sources=(CavityCharge("plus_A",5e-324),cav[0].sources[1]))
    with pytest.raises(UnsupportedProblem,match="magnitude"):
        SphericalSigmaModel(replace(original.gauss_assembly,cavities=(revised,)+cav[1:]),
                            original.geometry)


def test_independent_image_checker_remains_stable_for_nearly_centered_sources():
    # The real Kelvin image lies far outside the cavity for a tiny off-center
    # position. Ordinary binary64 explicit-image algebra would overflow or
    # divide by zero, even though the physical wall density is well behaved.
    source=GaussCavityAssembly("Metal",(SealedCavity("C",(CavityCharge("q",2e-9),)),),Grounded())
    for displacement in (5e-324,1e-250,1e-160,1e-50,1e-12):
        geo=SphericalMetalGeometry((0.,0.,0.),3.,(
            SphericalVoid("C",(0.,0.,0.),.5,(
                NamedSourcePosition("q",(displacement,0.,0.)),
            )),
        ))
        saved=solve_spherical_sigma(SphericalSigmaModel(source,geo)).as_dict((
            ("C",(1.,0.,0.)),("C",(-1.,0.,0.)),
        ))
        assert check_record(saved)["sampled_local_surface_densities_checked"]==2
        sigma=-2e-9/(4*pi*.5**2)
        if displacement < 1e-12:
            assert saved["samples"][0]["sigma_c_per_m2"]==pytest.approx(sigma,rel=2e-12)


def test_real_3d_rotation_translation_and_cavity_order_symmetry():
    base=teacher_model()
    before=solve_spherical_sigma(base)
    q=before.inner_surface_charge_density("NegativeCavity",(1.,0.,0.))
    def rotate(p):
        return (-p[1],p[0],p[2])
    def translate(p):
        return (p[0]+.3,p[1]-.4,p[2]+.2)
    def shift_model(transform):
        voids=tuple(SphericalVoid(v.cavity_id,transform(v.center_m),v.radius_m,tuple(
            NamedSourcePosition(s.source_id,transform(s.position_m)) for s in v.positions
        )) for v in base.geometry.voids)
        return replace(base,geometry=SphericalMetalGeometry(
            transform(base.geometry.outer_center_m),base.geometry.outer_radius_m,voids,
        ))
    turned=solve_spherical_sigma(shift_model(rotate))
    moved=solve_spherical_sigma(shift_model(translate))
    assert turned.inner_surface_charge_density("NegativeCavity",(0.,1.,0.))==pytest.approx(q,rel=5e-13)
    assert moved.inner_surface_charge_density("NegativeCavity",(1.,0.,0.))==pytest.approx(q,rel=5e-13)
    assert moved.gauss_ledger.outer_face_charge_c==before.gauss_ledger.outer_face_charge_c
    order=(2,0,1)
    reordered_gauss=replace(base.gauss_assembly,cavities=tuple(base.gauss_assembly.cavities[i] for i in order))
    reordered_geo=replace(base.geometry,voids=tuple(base.geometry.voids[i] for i in order))
    reordered=solve_spherical_sigma(SphericalSigmaModel(reordered_gauss,reordered_geo))
    assert reordered.inner_surface_charge_density("NegativeCavity",(1.,0.,0.))==q
    assert solve_spherical_sigma(base).inner_surface_charge_density("NegativeCavity",(1.,0.,0.))==q


def test_reserved_outer_is_not_a_cavity_id():
    c=SealedCavity("outer",())
    model=GaussCavityAssembly("Metal",(c,),Grounded())
    geom=SphericalMetalGeometry((0.,0.,0.),3.,(SphericalVoid("outer",(0.,0.,0.),.4),))
    with pytest.raises(ValueError,match="reserved"):
        SphericalSigmaModel(model,geom)


def test_reviewer_b1_admitted_nearunit_pole_is_not_solved_off_surface():
    """Independent review #01 B1: same radial direction near 1e-8 wall gap."""
    radius=.85
    source_q=3.1e-9
    displacement=radius*(1-1.2e-8)
    gauss=GaussCavityAssembly("M",(SealedCavity("V",(
        CavityCharge("q",source_q),
    )),),Grounded())
    geom=SphericalMetalGeometry((0.,0.,0.),3.,(
        SphericalVoid("V",(0.,0.,0.),radius,(
            NamedSourcePosition("q",(displacement,0.,0.)),
        )),
    ))
    result=solve_spherical_sigma(SphericalSigmaModel(gauss,geom))
    exact=(1.,0.,0.)
    near=(1.+9e-13,0.,0.)
    true_pole=-source_q*(radius+displacement)/(4*pi*radius*(radius-displacement)**2)
    exact_value=result.inner_surface_charge_density("V",exact)
    near_value=result.inner_surface_charge_density("V",near)
    assert exact_value==pytest.approx(true_pole,rel=1e-13)
    assert near_value==exact_value
    assert result.outer_surface_charge_density(near)==result.outer_surface_charge_density(exact)==0.
    assert abs(near[0]-1.)>0
    assert check_record(result.as_dict((("V",near),("outer",near))))["sampled_local_surface_densities_checked"]==2
    exported=result.as_dict((("V",near),))
    assert exported["samples"][0]["unit_normal"]==[1.,0.,0.]
    assert exported["samples"][0]["input_direction"]==list(near)
    assert exported["schema"]=="conductor_sim.m40a_spherical_multicavity_sigma.v2"
    assert exported["samples"][0]["sigma_c_per_m2"]==exact_value
    # The originally accepted-but-incorrect off-sphere value must be refused
    # even if a hostile JSON editor supplies a self-consistent nearunit normal.
    falsified=deepcopy(exported)
    old_offwall=-source_q*((radius-displacement)*(radius+displacement))/(4*pi*radius*(radius*near[0]-displacement)**3)
    assert abs(old_offwall/exact_value-1)>2e-4
    falsified["samples"][0]["sigma_c_per_m2"]=old_offwall
    with pytest.raises(ValueError,match="independent surface density"):
        check_record(falsified)
    # Valid alternative raw near-unit representation carries the same
    # physical axis and must be accepted if the true wall value is preserved.
    falsified["samples"][0]["input_direction"]=[1.+8e-13,0.,0.]
    falsified["samples"][0]["sigma_c_per_m2"]=exact_value
    assert check_record(falsified)["sampled_local_surface_densities_checked"]==1
    falsified["samples"][0]["unit_normal"]=[1.+8e-13,0.,0.]
    with pytest.raises(ValueError,match="unit-normal display"):
        check_record(falsified)
    with pytest.raises(ValueError,match="unit"):
        result.inner_surface_charge_density("V",(1.+2e-12,0.,0.))


def test_reviewer_b1_oblique_nearunit_directions_and_signed_static_variants():
    """Normalizing cannot be limited to world-axis pole special cases."""
    radius=.85
    direction=(.6,.8,0.)
    near=tuple(k*(1.+9e-13) for k in direction)
    source_pos=tuple(k*radius*(1-1.2e-8) for k in direction)
    for charge,constraint in ((3.1e-9,Grounded()),(-3.1e-9,FixedNetCharge(1e-9))):
        gauss=GaussCavityAssembly("M",(SealedCavity("V",(
            CavityCharge("q",charge),
        )),),constraint)
        geo=SphericalMetalGeometry((0.,0.,0.),3.,(
            SphericalVoid("V",(0.,0.,0.),radius,(
                NamedSourcePosition("q",source_pos),
            )),
        ))
        sol=solve_spherical_sigma(SphericalSigmaModel(gauss,geo))
        a=sol.inner_surface_charge_density("V",direction)
        b=sol.inner_surface_charge_density("V",near)
        assert b!=0. and a!=0.
        # Near-wall *raw* directions can differ slightly in their precise
        # radial ray even when produced by a simple float scaling, so their
        # native values are validated separately against the standalone
        # real+image normal-field checker, not required to be bit-equal.
        # Separate 90-digit Poisson-image-field physical value derived from
        # the admitted exact binary64 source positions. A regression here
        # must never be hidden by widening the standalone checker tolerance.
        assert check_record(sol.as_dict((("V",direction),)))["sampled_local_surface_densities_checked"]==1
        assert (a<0)==(charge>0)
        assert check_record(sol.as_dict((("V",near),("outer",near))))["sampled_local_surface_densities_checked"]==2
        n=sol.as_dict((("V",near),))["samples"][0]["unit_normal"]
        assert hypot(*n)==pytest.approx(1.,abs=3e-16)


@pytest.mark.parametrize("axis,charge,isolated,independent_raw",[
    ((.6,.8,0.),3.1e-9,False,-3532142.26790142),
    ((.6,.8,0.),-2.4e-9,True,+2734561.7557946476),
    ((.48,-.64,.6),3.1e-9,False,-4242770.360599676),
    ((.48,-.64,.6),-2.4e-9,True,+3284725.440464265),
])
def test_review02_b1_physical_raw_direction_and_truthful_json(
    axis,charge,isolated,independent_raw,
):
    """Chat B #02 independent 105-digit raw-real+image-field challenge.

    These are the four separately computed mathematical results from the
    preserved independent failed report (review_sandbox PR24). Inputs
    intentionally preserve the original *binary64 encoded raw direction*;
    prematurely normalizing in ordinary double precision is not allowed.
    """
    radius=.85
    pos=tuple(x*radius*(1.-1.2e-8) for x in axis)
    moved=tuple(axis[i]+(5e-9,-2.5e-9,1e-9)[i] for i in range(3))
    magnitude=hypot(*moved)
    unit=tuple(x/magnitude for x in moved)
    near=tuple(x*(1.+9e-13) for x in unit)
    assert abs(hypot(*near)-1.) <= 1e-12
    metal=FixedNetCharge(1.1e-9) if isolated else Grounded()
    gauss=GaussCavityAssembly("M",(SealedCavity("V",(
        CavityCharge("q",charge),
    )),),metal)
    geom=SphericalMetalGeometry((0.,0.,0.),3.,(
        SphericalVoid("V",(0.,0.,0.),radius,(
            NamedSourcePosition("q",pos),
        )),
    ))
    sol=solve_spherical_sigma(SphericalSigmaModel(gauss,geom))
    raw_sigma=sol.inner_surface_charge_density("V",near)
    assert raw_sigma==pytest.approx(independent_raw,rel=2e-11,abs=0.)
    record=sol.as_dict((("V",near),("outer",near)))
    assert record["schema"]=="conductor_sim.m40a_spherical_multicavity_sigma.v2"
    assert record["samples"][0]["input_direction"]==list(near)
    assert record["samples"][0]["sigma_c_per_m2"]==raw_sigma
    assert record["direction_interpretation"]
    assert check_record(record)["sampled_local_surface_densities_checked"]==2
    # A saved density forged from the OLD prematurely-rounded direction must
    # fail at the originally declared, unrelaxed 2e-10 numerical tolerance.
    old={
        ((.6,.8,0.),False):-3532142.2752839886,
        ((.6,.8,0.),True):+2734561.761510185,
        ((.48,-.64,.6),False):-4242770.356589729,
        ((.48,-.64,.6),True):+3284725.43735979,
    }[(axis,isolated)]
    false_record=deepcopy(record)
    false_record["samples"][0]["sigma_c_per_m2"]=old
    with pytest.raises(ValueError,match="independent surface density"):
        check_record(false_record)
    false_record=deepcopy(record)
    false_record["samples"][0]["input_direction"]=[True,0.,0.]
    with pytest.raises(ValueError):
        check_record(false_record)
    false_record=deepcopy(record)
    false_record["samples"][0]["unit_normal"]=[1.,0.,0.]
    with pytest.raises(ValueError,match="unit-normal display"):
        check_record(false_record)
    false_record=deepcopy(record)
    false_record["direction_interpretation"]="misleading rounded source direction"
    with pytest.raises(ValueError,match="direction provenance"):
        check_record(false_record)


def test_review02_near_wall_nonorigin_coordinates_preserve_encoded_si():
    """3D absolute coordinates must not be rounded into false near-wall gaps."""
    radius=.85
    center=(2.3,-1.7,.45)
    q=3.1e-9
    position=(center[0]+radius*(1.-1.2e-8),center[1],center[2])
    gauss=GaussCavityAssembly("M",(SealedCavity("V",(
        CavityCharge("q",q),
    )),),Grounded())
    geo=SphericalMetalGeometry((0.,0.,0.),5.,(
        SphericalVoid("V",center,radius,(
            NamedSourcePosition("q",position),
        )),
    ))
    sol=solve_spherical_sigma(SphericalSigmaModel(gauss,geo))
    almost=(1.+9e-13,0.,0.)
    rec=sol.as_dict((("V",almost),))
    assert check_record(rec)["sampled_local_surface_densities_checked"]==1
    assert rec["samples"][0]["input_direction"]==list(almost)
    assert rec["samples"][0]["sigma_c_per_m2"]<0.
