"""M40b: honest physically located offline atlas over independently qualified M40a."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import json

import pytest

from conductor_sim.cavity_gauss import CavityCharge, SealedCavity, GaussCavityAssembly
from conductor_sim.model import FixedNetCharge, Grounded, EPSILON_0
from conductor_sim.spherical_multicavity import (
    NamedSourcePosition, SphericalVoid, SphericalMetalGeometry,
    SphericalSigmaModel, solve_spherical_sigma,
)
from conductor_sim.spherical_sigma_atlas import (
    SCHEMA, ROLE, _si, surface_atlas_manifest, render_surface_atlas,
)
from tools.m40a_spherical_example import teacher_model
from tools.m40b_spherical_atlas_example import run_all
from tools.m40b_spherical_atlas_reference import check_manifest


def pair(model=None):
    model = model or teacher_model()
    other = replace(model, gauss_assembly=replace(
        model.gauss_assembly, constraint=Grounded(),
    ))
    return solve_spherical_sigma(model), solve_spherical_sigma(other)


def test_authentic_three_cavity_example_and_independent_reference(tmp_path: Path):
    doc = run_all(tmp_path)
    verdict = check_manifest(doc)
    assert verdict["cavities_checked"] == 3
    assert verdict["sources_checked"] == 4
    assert verdict["signed_inner_samples_verified_by_images"] == 3 * 24 * 2
    assert verdict["new_source_jee_items_solved"] == 0
    assert doc["schema"] == SCHEMA and doc["source_role"] == ROLE
    assert doc["original_workbook_items_newly_solved"] == 0
    assert len(doc["native_m40a_v2_witness_records"]["isolated"]) == 4
    assert json.loads((tmp_path / "m40b-teacher-atlas.json").read_text()) == doc
    assert (tmp_path / "m40b-teacher-atlas.html").read_text() == render_surface_atlas(*pair())


def test_teacher_physics_signed_neutral_and_empty_cavity():
    doc = surface_atlas_manifest(*pair())
    neutral, negative, empty = doc["cavities"]
    assert neutral["source_total_charge_c"] == 0.0
    assert neutral["whole_inner_face_charge_c"] == 0.0
    vals = [s["isolated_sigma_c_per_m2"] for s in neutral["samples"]]
    assert any(v > 0 for v in vals) and any(v < 0 for v in vals)
    assert any(s["isolated_sigma_c_per_m2"] != 0 for s in negative["samples"])
    assert all(s["isolated_sigma_c_per_m2"] == 0 for s in empty["samples"])
    assert all(s["grounded_sigma_c_per_m2"] == s["isolated_sigma_c_per_m2"]
               for row in doc["cavities"] for s in row["samples"])
    assert doc["outer"]["grounded_uniform_sigma_c_per_m2"] == 0.0
    assert doc["outer"]["isolated_uniform_sigma_c_per_m2"] != 0.0


def test_visual_is_honest_true_3d_coordinates_and_nojs():
    doc = surface_atlas_manifest(*pair())
    html = render_surface_atlas(*pair())
    assert 'Content-Security-Policy' in html
    assert "script-src &#39;none&#39;" in html
    assert "<script" not in html.lower()
    assert "not area averages" in html
    assert "0 newly solved original JEE" in html
    assert "XY ORTHOGRAPHIC PROJECTIONS" in html
    assert "z IS LISTED" in html
    assert "+x" in html and "+y" in html
    assert "θ" in html and "φ" in html
    assert "m40b-source-1-0" in html and "m40b-void-2" in html
    for cav in doc["cavities"]:
        assert cav["cavity_id"] in html
        for source in cav["sources"]:
            assert source["source_id"] in html
            assert repr(source["position_m"][2]) in html
            assert source["position_m"] in [
                row["position_m"] for row in cav["sources"]
            ]
        assert len(cav["samples"]) == 24
    assert html.count('class="tile ') == 72
    assert "original 3D" not in html  # no false 3D field claim


def test_near_wall_raw_direction_is_retained_for_real_local_sigma():
    radius = .85
    gap = 1.2e-8
    m = SphericalSigmaModel(
        GaussCavityAssembly("M",(
            SealedCavity("A",(CavityCharge("q",3.1e-9),)),
        ),FixedNetCharge(1.1e-9),EPSILON_0),
        SphericalMetalGeometry((0.,0.,0.),5.,(
            SphericalVoid("A",(0.,0.,0.),radius,(
                NamedSourcePosition("q",(radius*(1-gap),0.,0.)),
            )),
        )),
    )
    exact = surface_atlas_manifest(*pair(m),directions=((1.,0.,0.),))
    raw = surface_atlas_manifest(*pair(m),directions=((1.+9e-13,0.,0.),))
    x = exact["cavities"][0]["samples"][0]
    y = raw["cavities"][0]["samples"][0]
    assert x["isolated_sigma_c_per_m2"] == y["isolated_sigma_c_per_m2"]
    assert y["input_direction"] == [1.+9e-13,0.,0.]
    assert y["unit_normal"] == [1.,0.,0.]
    assert check_manifest(raw)["signed_inner_samples_verified_by_images"] == 2
    assert "1.0000000000009" in render_surface_atlas(*pair(m),directions=((1.+9e-13,0.,0.),))


@pytest.mark.parametrize("field,value",[
    ("schema","conductor_sim.m40b_forged.v1"),
    ("original_workbook_items_newly_solved",True),
    ("source_role","original_JEE"),
    ("irregular_3d_fields_qualified",True),
    ("time_evolution_defined",True),
    ("sample_tiles_are_point_observations_only",False),
    ("projection_is_2d_field_or_cross_section",True),
    ("warning","This diagram is a real 3D field solution"),
])
def test_forged_major_claims_refused(field,value):
    doc = deepcopy(surface_atlas_manifest(*pair(),directions=((1.,0.,0.),)))
    doc[field] = value
    with pytest.raises((ValueError,TypeError)):
        check_manifest(doc)


@pytest.mark.parametrize("path,value",[
    (("cavities",0,"whole_inner_face_charge_c"),False),
    (("cavities",0,"sources",0,"charge_c"),False),
    (("cavities",0,"sources",0,"source_id"),"wrong"),
    (("cavities",1,"center_m",0),4.0),
    (("cavities",1,"sources",0,"position_m",2),500.0),
    (("cavities",0,"samples",0,"isolated_sigma_c_per_m2"),False),
    (("cavities",0,"samples",0,"isolated_sigma_c_per_m2"),1e-12),
    (("cavities",0,"samples",0,"phi_deg_display"),92.0),
    (("outer","grounded_uniform_sigma_c_per_m2"),1.0),
    (("native_m40a_v2_witness_records","isolated",0,"samples",0,"sigma_c_per_m2"),False),
    (("native_m40a_v2_witness_records","isolated",0,"samples",0,"input_direction",0),False),
    (("native_m40a_v2_witness_records","isolated",0,"samples",0,"unit_normal",0),0.3),
])
def test_tampered_visible_values_and_witness_refused(path,value):
    doc = deepcopy(surface_atlas_manifest(*pair(),directions=((1.,0.,0.),)))
    ref = doc
    for k in path[:-1]:
        ref = ref[k]
    ref[path[-1]] = value
    with pytest.raises((ValueError, TypeError, KeyError, OverflowError)):
        check_manifest(doc)


@pytest.mark.parametrize("kind",[
    "identical-constraint", "grounded-first", "different-source-charge",
    "different-geometry", "different-source-position",
])
def test_refuse_wrong_or_unrelated_static_pair(kind):
    iso, grd = pair()
    if kind == "identical-constraint":
        args = (iso,iso)
    elif kind == "grounded-first":
        args = (grd,iso)
    else:
        original = teacher_model()
        if kind == "different-source-charge":
            cav = original.gauss_assembly.cavities[0]
            changed = replace(cav, sources=(replace(cav.sources[0],charge_c=3e-9),)+cav.sources[1:])
            gg = replace(original.gauss_assembly,
                         cavities=(changed,)+original.gauss_assembly.cavities[1:],
                         constraint=Grounded())
            altered = replace(original,gauss_assembly=gg)
        else:
            void = original.geometry.voids[0]
            if kind == "different-geometry":
                changed = replace(void,radius_m=.7)
            else:
                p = void.positions[0]
                changed = replace(void,positions=(replace(p,position_m=(-1.45,0.,0.)),)+void.positions[1:])
            altered = replace(original,geometry=replace(
                original.geometry,voids=(changed,)+original.geometry.voids[1:]),
                gauss_assembly=replace(original.gauss_assembly,constraint=Grounded()))
        args = iso,solve_spherical_sigma(altered)
    with pytest.raises((ValueError,TypeError)):
        surface_atlas_manifest(*args,directions=((1.,0.,0.),))


@pytest.mark.parametrize("directions",[
    (), ((),), ((False,0.,0.),), ((0.,0.,0.),),
    ((2.,0.,0.),), ((float("nan"),0.,0.),),
    ((float("inf"),0.,0.),), ((1.,0.),),
    tuple((1.,0.,0.) for _ in range(49)),
])
def test_unqualified_atlas_rays_refused(directions):
    with pytest.raises((ValueError, TypeError)):
        surface_atlas_manifest(*pair(),directions=directions)


def test_source_si_numerics_are_not_forged_as_zero_or_boolean():
    with pytest.raises(ValueError):
        _si(False,"C")
    with pytest.raises(ValueError):
        _si(float("nan"),"m")
    assert _si(0.0,"C/m²") == "0 C/m²"
    assert _si(-1.23e-9,"C") == "−1.23e-09 C"
