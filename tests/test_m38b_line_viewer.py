"""M38b: truthful local physical plane, SI readouts and source-safe playback."""
from copy import deepcopy
from dataclasses import replace
import json
from math import hypot

import pytest

from conductor_sim.model import EPSILON_0
from conductor_sim.prescribed_ground_wire import author_prescribed_grounded_sphere
from conductor_sim.prescribed_ground_line import (
    PrescribedSphereLineSpec, sample_prescribed_sphere_line,
)
from conductor_sim.prescribed_line_viewer import (
    SCHEMA, line_view_manifest, render_prescribed_line_viewer,
)
from tools.m38b_line_viewer_example import example_trace, run_all
from tools.m38b_line_view_reference import check_line_view
from tools.check_m38b_line_browser import _verify_canonical_html, _freeze_actual_playback_frame, LABELS
from tools.m38a_line_reference import check_line_record


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    target=tmp_path_factory.mktemp("m38b-true-line")
    manifest=run_all(target)
    return target,manifest


def test_actual_teacher_line_si_viewer_and_html_agree_at_every_frame(generated):
    folder, doc=generated
    out=check_line_view(doc)
    assert out["independently_checked_display_rows"]==41
    assert out["independently_checked_kelvin_states"]==41
    assert out["additional_original_jee_items"]==0
    assert doc["schema"]==SCHEMA
    assert doc["source"]["role"]=="teacher_hypothetical"
    assert doc["trace"]["source_role"].startswith("teacher_hypothetical")
    assert doc["trace"]["closest_approach_time_s"]==pytest.approx(2.)
    assert doc["trace"]["closest_approach_distance_m"]==pytest.approx(.2)
    assert doc["readouts"][0]["current_label"].startswith("8.944")
    assert doc["readouts"][20]["current_label"]=="0 A"
    assert doc["trace"]["samples"][40]["wire_current_sphere_to_ground_a"]<0
    assert json.loads((folder/"m38b-line-physics.json").read_text())==doc["trace"]
    assert json.loads((folder/"m38b-line-viewer.json").read_text())==doc
    html=(folder/"m38b-line.html").read_text(encoding="utf-8")
    for token in (
        '<svg id="m38b-scene"', '<svg id="m38b-current-chart"',
        '<script id="m38b-data" type="application/json">',
        '<noscript><p id="m38b-nojs">',
        'id="m38b-charge"', 'id="m38b-sphere"', 'id="m38b-path"',
        'id="m38b-world"', 'id="m38b-radial-v"', 'id="m38b-opposite"',
        "NOT world x-y", "TEACHER-AUTHORED", "NOT a free Newtonian orbit",
        'value="0" disabled', 'id="m38b-current-line"',
    ):
        assert token in html
    assert "https://" not in html and "http://" not in html
    assert "setInterval" in html
    assert "window.m38bViewer" in html


def test_local_plane_is_fixed_true_metre_scale_and_world_xyz_not_faked():
    t=example_trace()
    m=line_view_manifest(t)
    r=m["layout"]
    a=m["readouts"]
    p=m["trace"]
    assert r["sphere_cx_px"]==360.
    assert r["sphere_cy_px"]==190.
    assert r["sphere_r_px"]/r["pixels_per_meter"]==pytest.approx(p["sphere_radius_m"])
    assert r["path_start_x_px"]==a[0]["scene_x_px"]
    assert r["path_end_x_px"]==a[-1]["scene_x_px"]
    assert a[0]["scene_x_px"]<r["sphere_cx_px"]<a[-1]["scene_x_px"]
    assert a[0]["scene_y_px"]==pytest.approx(a[-1]["scene_y_px"])
    assert a[20]["scene_x_px"]==pytest.approx(r["sphere_cx_px"])
    assert a[20]["chart_y_px"]==pytest.approx(r["chart_zero_y_px"])
    for k in (0,1,9,20,31,40):
        x=a[k]
        raw=p["samples"][k]
        assert x["world_position_label"].startswith("[")
        assert x["time_label"].endswith(" s")
        assert x["radial_velocity_label"].endswith(" m/s")
        assert x["current_label"].endswith(" A")
        assert x["energy_label"].endswith(" J")
        assert x["scene_x_px"]!=float(k)  # no faked frame index-to-position map
        assert hypot(*raw["source_position_m"])==pytest.approx(raw["radial_distance_m"])
    check_line_view(m)


def test_unfamiliar_translated_noncoplanar_negative_charge_transfer():
    c=(.75,-.45,1.2)
    initial=(.35,-.19,1.38)
    problem=author_prescribed_grounded_sphere(
        -1.7e-9,initial,.1,sphere_center_m=c,sphere_name="Off-axis <B>",
        permittivity_f_per_m=1.3*EPSILON_0
    )
    trace=sample_prescribed_sphere_line(PrescribedSphereLineSpec(
        problem,(.19,-.06,.04),3.6,sample_count=23,surface_guard_m=.002
    ))
    manifest=line_view_manifest(trace)
    expected=check_line_view(manifest)
    assert expected["independently_checked_display_rows"]==23
    assert manifest["trace"]["source_charge_c"]<0
    assert manifest["trace"]["samples"][0]["wire_current_sphere_to_ground_a"]<0
    assert manifest["trace"]["samples"][-1]["wire_current_sphere_to_ground_a"]>0
    assert manifest["readouts"][0]["world_position_label"].startswith("[0.35")
    assert "Off-axis &lt;B&gt;" in render_prescribed_line_viewer(trace)
    assert "<B>" not in render_prescribed_line_viewer(trace).split("</header>")[0]
    u=manifest["layout"]["local_u_world"]
    w=manifest["layout"]["local_w_world"]
    assert abs(sum(x*y for x,y in zip(u,w)))<2e-12


def test_radial_qualified_line_uses_honest_deterministic_transverse_axis():
    problem=author_prescribed_grounded_sphere(1e-9,(0.,0.,.4),.1)
    trace=sample_prescribed_sphere_line(PrescribedSphereLineSpec(
        problem,(0.,0.,-.05),2.,sample_count=11,surface_guard_m=.01
    ))
    m=line_view_manifest(trace)
    assert check_line_view(m)["independently_checked_display_rows"]==11
    assert all(r["scene_y_px"]==pytest.approx(m["layout"]["sphere_cy_px"])
               for r in m["readouts"])
    assert m["layout"]["local_u_world"]==pytest.approx([0.,0.,-1.])
    assert m["source"]["newly_solved_original_workbook_items"]==0


def test_manually_forged_native_frame_never_renders():
    genuine=example_trace()
    row=genuine.samples[0]
    fake=replace(genuine,samples=(replace(row,time_s=False),)+genuine.samples[1:])
    with pytest.raises(ValueError):
        line_view_manifest(fake)
    with pytest.raises(ValueError):
        render_prescribed_line_viewer(fake)


@pytest.mark.parametrize("keys,bad",[
    (("source","role"),"original_workbook_teacher_example"),
    (("source","heading"),"PRINTED WORKBOOK Q21"),
    (("source","original_warning"),"This is official JEE"),
    (("source","teacher_notice"),"Original values copied verbatim"),
    (("source","newly_solved_original_workbook_items"),False),
    (("layout","projection"),"world xy is the real 3D plane"),
    (("layout","sphere_r_px"),1.),
    (("layout","local_u_world",0),True),
    (("layout","local_w_world",1),.7),
    (("layout","path_start_y_px"),555.),
    (("layout","path_end_x_px"),555.),
    (("layout","chart_peak_abs_a"),1.),
    (("readouts",0,"scene_x_px"),600.),
    (("readouts",0,"sign_y_px"),300.),
    (("readouts",0,"time_label"),"999 s"),
    (("readouts",0,"current_label"),"-9 A"),
    (("readouts",0,"world_position_label"),"[0, 0, 0] m"),
    (("readouts",20,"chart_y_px"),50.),
    (("readouts",30,"energy_label"),False),
    (("trace","samples",0,"time_s"),False),
    (("trace","samples",1,"wire_current_sphere_to_ground_a"),1.),
])
def test_reference_refuses_false_native_projection_and_readout_claims(keys,bad):
    m=json.loads(json.dumps(line_view_manifest(example_trace())))
    target=m
    for key in keys[:-1]:
        target=target[key]
    target[keys[-1]]=bad
    with pytest.raises(ValueError):
        check_line_view(m)


def test_unsafe_authored_name_cannot_execute_active_html():
    problem=author_prescribed_grounded_sphere(
        1e-9,(-.4,.2,0),.1,sphere_name='S</script><script>alert(1)</script>'
    )
    trace=sample_prescribed_sphere_line(PrescribedSphereLineSpec(
        problem,(.2,0.,0.),4.,sample_count=11,surface_guard_m=.01
    ))
    html=render_prescribed_line_viewer(trace)
    assert 'S&lt;/script&gt;&lt;script&gt;alert' in html
    assert "<script>alert(1)</script>" not in html
    assert "\\u003c" in html
    assert check_line_view(line_view_manifest(trace))["independently_checked_kelvin_states"]==11



def test_independently_verified_native_renderer_refuses_css_svg_forgery(tmp_path):
    """Actual canonical native-HTML integrity, without an installed browser.

    Real Chromium separately proves CSS attacks are physically visible.
    This test establishes that the source-boundary additionally rejects
    entire-document forged rendering, not just individual DOM values.
    """
    manifest=run_all(tmp_path)
    target=tmp_path/"m38b-line.html"
    real=target.read_bytes()
    assert check_line_view(manifest)["independently_checked_display_rows"]==41
    _verify_canonical_html(manifest,target)
    attacks=(
        '<style>#m38b-heading::after{content:"OFFICIAL PRINTED"}</style>',
        '#m38b-sphere{fill:none!important;stroke:none!important}',
        'transform:scaleX(-1)!important',
    )
    for alteration in attacks:
        try:
            if alteration.startswith("<style>"):
                false=real.decode("utf-8").replace("</head>",alteration+"</head>")
            else:
                false=real.decode("utf-8").replace("</style>",alteration+"</style>")
            assert false.encode("utf-8")!=real
            target.write_text(false,encoding="utf-8")
            with pytest.raises(AssertionError,match="noncanonical visible HTML/CSS"):
                _verify_canonical_html(manifest,target)
        finally:
            target.write_bytes(real)
    _verify_canonical_html(manifest,target)


def test_m38c_atomic_play_frame_captures_unmodified_science_before_pausing(generated):
    """A fake slider index with stale live marker is NOT made honest by redraw.

    The actual browser workflow separately performs real Playwright playback
    and samples the untouched running DOM. This unit regression checks the
    precise source-binding and detection contract without requiring Chromium
    in the ordinary full Python test matrix.
    """
    _,manifest=generated
    i=1
    row=manifest["readouts"][i]
    payload={
        "index":i,"pausedIndex":i,"sliderIndex":i,
        "visible":{key:row[key] for key in LABELS.values()},
        "coords":{key:row[key] for key in (
            "scene_x_px","scene_y_px","sign_x_px","sign_y_px",
            "chart_x_px","chart_y_px",
        )},
    }

    class CapturedPlayFrame:
        def __init__(self, result):
            self.result=result
        def evaluate(self, source_js, labels):
            assert labels==LABELS
            # Critical: the displayed science must be captured before the
            # normal input event stops and redraws the current true index.
            assert source_js.index("const visible=")<source_js.index("slider.dispatchEvent")
            assert source_js.index("const coords=")<source_js.index("slider.dispatchEvent")
            return deepcopy(self.result)

    assert _freeze_actual_playback_frame(CapturedPlayFrame(payload),manifest)==i

    fake=deepcopy(payload)
    fake["coords"]["chart_y_px"]=999.
    with pytest.raises(AssertionError,match="LIVE playback scene/current marker"):
        _freeze_actual_playback_frame(CapturedPlayFrame(fake),manifest)

    fake=deepcopy(payload)
    fake["visible"]["current_label"]="-9 A"
    with pytest.raises(AssertionError,match="LIVE playback SI/world label"):
        _freeze_actual_playback_frame(CapturedPlayFrame(fake),manifest)

    fake=deepcopy(payload)
    fake["pausedIndex"]=i+1
    with pytest.raises(AssertionError,match="pause changed"):
        _freeze_actual_playback_frame(CapturedPlayFrame(fake),manifest)
