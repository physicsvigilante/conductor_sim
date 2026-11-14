"""M21b: source-given JEE physics and honest static visual acceptance."""
from copy import deepcopy
from math import cos, hypot, isclose, pi, sin, sqrt

import pytest

from conductor_sim import (
    EPSILON_0, FixedPointField, PrescribedField,
    UniformElectricField, UniformRingAxisField,
)
from conductor_sim.jee_potential_explorer import (
    endpoint_field_work, sample_fixed_potential_map, sample_static_profile,
    validate_fixed_potential_map, write_observation,
)
from conductor_sim.solver import UnsupportedProblem
from conductor_sim.visualization import SliceSpec
from examples.jee_potential_explorer import make_2021, make_2012, make_ring, make_case

K=1/(4*pi*EPSILON_0)


@pytest.fixture(scope="module")
def circle():
    return make_2021()


@pytest.fixture(scope="module")
def shell():
    return make_2012()[1]


@pytest.fixture(scope="module")
def rings():
    return make_ring()[1]


def test_official_source_2021_charge_geometry_and_identity(circle):
    a=circle["static_fixed_source_view"]
    sources=a["field"]["sources"]
    assert a["schema"]=="conductor_sim.fixed_field_observer.v1"
    assert a["mode"].startswith("static")
    assert a["source_annotations"][0]["id"] == "charge:negative"
    assert a["source_annotations"][1]["id"] == "charge:positive"
    assert sources[0]["kind"]==sources[1]["kind"]=="FixedPointField"
    assert sources[0]["position_m"] == [0.,0.,0.]
    assert sources[1]["position_m"] == [2.,0.,0.]
    assert sources[0]["charge_c"] < 0 < sources[1]["charge_c"]
    assert sources[1]["charge_c"] == pytest.approx(-sources[0]["charge_c"]/sqrt(3))
    assert circle["jee_anchor"]["independent_circle_center_m"]==[3.,0.]
    assert circle["jee_anchor"]["independent_radius_m"]==pytest.approx(sqrt(3))
    assert validate_fixed_potential_map(circle) is True
    assert circle["frames"][0]["geometry"] == []
    assert circle["frames"][0]["conductor_results"] == []
    assert not any(k in circle for k in ("motion_1d","motion_2d","prescribed_motion","position_sweep"))


def test_2021_analytic_circle_independent_coulomb_reference():
    Q=1e-9
    field=PrescribedField((FixedPointField(-Q),
                           FixedPointField(Q/sqrt(3),(2.,0.,0.))))
    for angle in [2*pi*i/48 for i in range(48)]:
        x=3+sqrt(3)*cos(angle)
        y=sqrt(3)*sin(angle)
        r1=hypot(x,y); r2=hypot(x-2,y)
        expected=K*(-Q/r1+Q/sqrt(3)/r2)
        assert abs(expected) < 1e-11
        assert field.potential_at((x,y,0.))==pytest.approx(expected,abs=2e-11)
    # Independent Q9 and Q10 are *geometrical*, not fitted from a sampled chart.
    assert (3-sqrt(3)) > 0 and (3+sqrt(3))<5
    assert field.potential_at((3.,sqrt(3),0.))==pytest.approx(0.,abs=1e-11)


def test_highlighted_v_zero_contour_is_sampled_and_near_analytic(circle):
    frame=circle["frames"][0]; meta=circle["static_fixed_source_view"]
    highlight=[x for x in frame["contours"] if x.get("highlight") is True]
    assert len(highlight)==1 and highlight[0]["level_v"]==0.
    assert len(highlight[0]["segments_m"])>100
    assert len(highlight[0]["segments_m"])==meta["highlighted_segment_count"]
    # Actual contour vertices are approximate triangulated grid intersections.
    # World x = slice origin x + local u; exact circle (x-3)^2+y^2=3.
    for seg in highlight[0]["segments_m"]:
        for u,v in seg:
            x=u+frame["plane"]["origin_m"][0]
            assert abs(hypot(x-3.,v)-sqrt(3)) < .085
    assert frame["grid"]["potential_v"] and frame["field_lines"]
    assert len(frame["field_lines"])>3


def test_2021_source_singularities_are_null_not_interpolated(circle):
    frame=circle["frames"][0]
    g=frame["grid"]; meta=circle["static_fixed_source_view"]
    assert len(frame["source_masks"])==2
    for sx in (0.,2.):
        i=min(range(len(g["x_m"])),key=lambda k:abs(g["x_m"][k]+2.5-sx))
        j=min(range(len(g["y_m"])),key=lambda k:abs(g["y_m"][k]))
        assert g["region"][j][i]=="source-mask"
        assert g["potential_v"][j][i] is None
        assert g["electric_field_v_per_m"][j][i] is None
    assert all(m["kind"]=="display-exclusion" for m in frame["source_masks"])
    assert meta["source_exclusion_radius_m"]==.11


def test_grid_tampering_fails_closed(circle):
    altered=deepcopy(circle)
    g=altered["frames"][0]["grid"]
    for j,row in enumerate(g["region"]):
        if "vacuum" in row:
            i=row.index("vacuum")
            g["potential_v"][j][i]+=1.
            break
    with pytest.raises(ValueError,match="disagrees"):
        validate_fixed_potential_map(altered)
    altered=deepcopy(circle)
    altered["frames"][0]["contours"][-1]["level_v"]=17.
    with pytest.raises(ValueError,match="highlighted"):
        validate_fixed_potential_map(altered)
    altered=deepcopy(circle)
    altered["frames"][0]["contours"][-1]["segments_m"][0][0][0]+=0.25
    with pytest.raises(ValueError,match="highlighted contour"):
        validate_fixed_potential_map(altered)


def test_real_2012_shell_radial_jump_and_analytic_limits(shell):
    anchor=shell["anchor"]; profile=shell["profile"]
    rows=profile["rows"]
    assert anchor["pdf_page"]==15
    assert profile["coordinate"]=="radial"
    assert len(rows)==241
    assert anchor["original_given"].startswith("Uniform positive")
    R=.1;Q=1e-9
    for row in rows:
        r=row["coordinate_m"]
        V=K*Q/max(r,R)
        assert row["potential_v"]==pytest.approx(V,rel=1e-12)
        if isclose(r,R,abs_tol=1e-14):
            assert row["electric_field_v_m"] is None
            assert row["field_parallel_v_m"] is None
            assert row["field_undefined"] is True
        elif r<R:
            assert row["field_parallel_v_m"] == 0.
            assert row["electric_field_v_m"] == [0.,0.,0.]
        else:
            assert row["field_parallel_v_m"]==pytest.approx(K*Q/r**2,rel=1e-12)
    assert sum(row["field_undefined"] for row in rows)==1
    assert rows[0]["potential_v"]==pytest.approx(K*Q/R)
    assert rows[-1]["potential_v"]==pytest.approx(K*Q/.24)


def test_shell_html_contains_genuine_surface_gap_not_zero(shell):
    html,audit=make_2012()
    assert html.count('id="physical-surface"')==2
    assert "E undefined" in html
    assert '<polyline' in html
    assert 'CONDUCTOR_SIM / JEE STATIC SOURCE OBSERVER' in html
    assert '"field_parallel_v_m":null' in html  # genuine undefined value at surface
    assert '"mode":"static spatial samples' in html
    assert "No time evolution" in html


def test_workbook_ring_geometry_work_per_source_and_transverse_force(rings):
    anchor=rings["anchor"]; work=rings["endpoint_work"];profile=rings["profile"]
    assert anchor["pdf_page"]==12
    assert work["schema"]=="conductor_sim.fixed_endpoint_work.v1"
    assert work["start_m"]==[0.,0.,0.] and work["end_m"]==[.4,0.,0.]
    assert work["probe_charge_c"]==10e-6
    source=work["field"]["sources"]
    assert source[0]["charge_c"]==10e-6 and source[1]["charge_c"]==-20e-6
    assert source[0]["radius_m"]==source[1]["radius_m"]==.3
    assert source[0]["axis"]==source[1]["axis"]==[1.,0.,0.]
    assert source[2]["field_v_m"]==[0.,1e5,0.]
    assert anchor["original_mass"].startswith("symbolic")
    assert [x["id"] for x in work["source_work_components"]]==[
        "ring:A","ring:B","field:vertical"]
    expected=10e-6*K*((10e-6)*(1/.3-1/.5)+(-20e-6)*(1/.5-1/.3))
    assert work["field_work_j"]==pytest.approx(expected,rel=1e-12)
    assert work["field_work_j"]==pytest.approx(3.5950207169,rel=1e-9)
    assert work["source_work_components"][-1]["field_work_j"]==0.
    for p in (work["start_electric_field_v_m"],work["end_electric_field_v_m"]):
        assert p[1]==1e5
    assert max(abs(r["field_transverse_abs_v_m"]-1e5) for r in profile["rows"])<1e-8
    assert not any(r["field_undefined"] for r in profile["rows"])


def test_ring_html_is_static_with_no_mass_or_browser_ode():
    html,audit=make_ring()
    assert "No physical guide" in html
    assert "source-geometry" in html
    assert "Ring A" in html and "Ring B" in html
    assert "3.5950207" in html
    assert 'id="work-summary"' in html
    assert 'probe' not in audit["endpoint_work"] # probe charge is a scalar, not invented moving object


def test_gauge_shifts_do_not_change_field_work():
    base=PrescribedField((
        UniformRingAxisField(10e-6,.3,axis=(1.,0.,0.)),
        UniformRingAxisField(-20e-6,.3,center_m=(.4,0.,0.),axis=(1.,0.,0.)),
        UniformElectricField((0.,1e5,0.)),
    ))
    labels=(("a","A"),("b","B"),("e","E"))
    shifted=PrescribedField(base.sources[:-1]+(
        UniformElectricField((0.,1e5,0.), reference_potential_v=17.),))
    A=(0.,0.,0.); B=(.4,0.,0.)
    a=endpoint_field_work(base,source_annotations=labels,probe_charge_c=1e-5,start_m=A,end_m=B)
    b=endpoint_field_work(shifted,source_annotations=labels,probe_charge_c=1e-5,start_m=A,end_m=B)
    assert a["field_work_j"]==pytest.approx(b["field_work_j"],abs=1e-11)
    assert b["start_potential_v"]-a["start_potential_v"]==pytest.approx(17.)
    assert a["source_work_components"][-1]["field_work_j"]==0.


@pytest.mark.parametrize("field,plane",[
    (PrescribedField((UniformRingAxisField(1e-9,.2),)),
     SliceSpec(origin_m=(0.,0.,0.),u=(1.,0.,0.),v=(0.,0.,1.),resolution=11)),
    (PrescribedField((FixedPointField(1e-9,(0.,0.,.1)),)),
     SliceSpec(origin_m=(0.,0.,0.),u=(1.,0.,0.),v=(0.,1.,0.),resolution=11)),
    (PrescribedField((UniformElectricField((0.,0.,3.)),)),
     SliceSpec(origin_m=(0.,0.,0.),u=(1.,0.,0.),v=(0.,1.,0.),resolution=11)),
])
def test_outside_qualified_planar_map_is_rejected(field,plane):
    with pytest.raises(UnsupportedProblem):
        sample_fixed_potential_map(
            field,plane,source_annotations=tuple(
                (f"source:{i}",f"Source {i}") for i in range(len(field.sources))),
            source_mask_m=.01)


def test_offaxis_ring_endpoint_refused_and_no_fictional_axis():
    field=PrescribedField((UniformRingAxisField(1e-9,.2),))
    with pytest.raises(UnsupportedProblem,match="axis"):
        endpoint_field_work(field,source_annotations=(("a","Ring A"),),
                            probe_charge_c=1e-9,start_m=(0.,0.,.1),end_m=(.1,0.,.1))
    with pytest.raises(UnsupportedProblem,match="axis"):
        sample_static_profile(field,source_annotations=(("a","Ring A"),),
                              origin_m=(0.,0.,0.),axis=(1.,0.,0.),
                              start_m=0.,end_m=.4)


@pytest.mark.parametrize("case",["2021-equipotential","2012-shell","ring-work"])
def test_all_three_standalone_html_and_json(tmp_path,case):
    html,audit=make_case(case)
    path,sidecar=write_observation(tmp_path/f"{case}.html",html,audit)
    assert path.is_file() and sidecar.is_file()
    assert "<!doctype html>" in html.lower()
    assert "src=\"https://" not in html  # a source URL in metadata is not a network request
    assert "fetch(" not in html
    assert '"schema"' in sidecar.read_text(encoding="utf-8")
    assert len(html)>10000
