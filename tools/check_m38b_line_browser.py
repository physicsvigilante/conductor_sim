"""M38b: REAL offline Chromium checks of actual physical HTML and JS-off first frame.

The independent stdlib-only M38a/M38b checkers prove SI/geometry; this
browser layer proves what the STUDENT actually sees, including with no JS.
"""
from __future__ import annotations

import argparse
import json
from math import isclose, isfinite
from pathlib import Path

from tools.m38b_line_view_reference import check_line_view
from tools.m38a_line_reference import check_line_record


def _verify_canonical_html(manifest: dict, target: Path) -> None:
    """Bind *all* visible HTML/CSS/SVG to separately SI-checked native physics.

    Per-element DOM value tests cannot detect CSS pseudo-text, opacity,
    stroke removal, clipping or transformed world axes. Re-generate the exact
    deterministic production HTML from the verified native physical inputs
    and require byte-for-byte identity; then independently test real Chromium
    JS-on AND no-JS DOM. This is a source-to-presentation consistency gate,
    not a cryptographic proof of the origin of arbitrary historical exports.
    """
    from conductor_sim.prescribed_ground_wire import author_prescribed_grounded_sphere
    from conductor_sim.prescribed_ground_line import (
        PrescribedSphereLineSpec, sample_prescribed_sphere_line,
    )
    from conductor_sim.prescribed_line_viewer import (
        line_view_manifest, render_prescribed_line_viewer,
    )

    native = manifest["trace"]
    problem = author_prescribed_grounded_sphere(
        native["source_charge_c"],
        native["initial_source_position_m"],
        native["sphere_radius_m"],
        sphere_center_m=native["sphere_center_m"],
        sphere_name=native["sphere_name"],
        permittivity_f_per_m=native["permittivity_f_per_m"],
    )
    spec = PrescribedSphereLineSpec(
        problem,
        native["constant_cartesian_velocity_m_s"],
        native["duration_s"],
        sample_count=native["sample_count"],
        surface_guard_m=native["surface_guard_m"],
    )
    genuine = sample_prescribed_sphere_line(spec)
    regenerated = line_view_manifest(genuine)
    def serialized(record):
        return json.dumps(record, ensure_ascii=False, sort_keys=True,
                          allow_nan=False, separators=(",", ":"))
    if serialized(regenerated) != serialized(manifest):
        raise AssertionError(
            "canonical rendered viewer source disagrees with independently verified native SI/geometry"
        )
    if target.read_bytes() != render_prescribed_line_viewer(genuine).encode("utf-8"):
        raise AssertionError(
            "noncanonical visible HTML/CSS: source, signed SI, sphere, path, "
            "current-axis or physical projection differs from the verified native renderer"
        )


LABELS = {
    "#m38b-time": "time_label",
    "#m38b-distance": "distance_label",
    "#m38b-radial-v": "radial_velocity_label",
    "#m38b-Q": "induced_charge_label",
    "#m38b-current": "current_label",
    "#m38b-opposite": "opposite_label",
    "#m38b-world": "world_position_label",
    "#m38b-world-v": "world_velocity_label",
    "#m38b-force": "force_label",
    "#m38b-energy": "energy_label",
}


def _string(page, selector, expected, reason):
    loc=page.locator(selector)
    if not loc.is_visible() or loc.inner_text().strip()!=expected:
        raise AssertionError(reason+" "+selector)


def _attr(page, selector, attribute, expected, reason):
    node=page.locator(selector)
    if node.count()!=1 or not node.evaluate(
        "(el)=>{const s=getComputedStyle(el);return s.display!=='none' && s.visibility!=='hidden' && Number(s.opacity)>0;}"
    ):
        raise AssertionError(reason+" "+selector+" is hidden or absent")
    value=node.get_attribute(attribute)
    try:
        actual=float(value) if value is not None else float("nan")
    except (TypeError,ValueError):
        actual=float("nan")
    if not isfinite(actual) or not isclose(actual,expected,rel_tol=4e-11,abs_tol=2e-7):
        raise AssertionError(reason+" "+selector+"."+attribute)


def _check_static(page, data, *, no_js):
    """Check visible DATA and geometry independently of JavaScript startup."""
    s=data["source"]
    l=data["layout"]
    rows=data["readouts"]
    native=data["trace"]
    for selector,key in (
        ("#m38b-heading","heading"),
        ("#m38b-original-warning","original_warning"),
        ("#m38b-teacher-notice","teacher_notice"),
        ("#m38b-protocol","protocol_notice"),
    ):
        _string(page,selector,s[key],"false visible teacher-vs-original source attribution")
    _string(page,"#m38b-projection",l["projection"],"false visible local physical plane label")
    for selector,key in (
        ("#m38b-u","u_label"),("#m38b-w","w_label"),("#m38b-normal","normal_label")
    ):
        _string(page,selector,l[key],"false visible real world local-plane basis")
    _string(page,"#m38b-sphere-name",
            "Physical conductor: "+native["sphere_name"]+
            " (grounded solid sphere). Conventional positive current is sphere → ground. All 3D world coordinates are SI metres.",
            "false visible 3D physical source identity")
    for attr,key in (("cx","sphere_cx_px"),("cy","sphere_cy_px"),("r","sphere_r_px")):
        _attr(page,"#m38b-sphere",attr,l[key],"false visible fixed physical sphere geometry")
    for attr,key in (
        ("x1","path_start_x_px"),("y1","path_start_y_px"),
        ("x2","path_end_x_px"),("y2","path_end_y_px")
    ):
        _attr(page,"#m38b-path",attr,l[key],"false visible complete 3D projected source path")
    for a in ("y1","y2"):
        _attr(page,"#m38b-zero",a,l["chart_zero_y_px"],"false visible signed-current zero axis")
    if not page.locator("#m38b-scene").is_visible() or not page.locator("#m38b-current-chart").is_visible():
        raise AssertionError("physical scene or signed current graph is not actually visible")
    curve=page.locator("#m38b-current-line")
    if not curve.is_visible():
        raise AssertionError("visible signed-current graph is hidden")
    poly=curve.get_attribute("points")
    if poly is None or len(poly.split())!=len(rows):
        raise AssertionError("visible full signed-current graph dropped genuine physical samples")
    for i, (token,row) in enumerate(zip(poly.split(),rows)):
        try:
            xy=token.split(",")
            if len(xy)!=2:
                raise ValueError("non-xy")
            vals=[float(x) for x in xy]
        except ValueError as ex:
            raise AssertionError("visible plotted current point format incorrect") from ex
        for j,k in enumerate(("chart_x_px","chart_y_px")):
            if not isclose(vals[j],row[k],rel_tol=4e-10,abs_tol=2e-7):
                raise AssertionError("visible plotted current point "+str(i)+" contradicts verified Kelvin SI")
    expected_chart=[
        "I (A), positive sphere → ground",
        "t=0 s",
        "t="+format(native["duration_s"],".11g")+" s",
    ]
    if page.locator("#m38b-current-chart text").all_text_contents()!=expected_chart:
        raise AssertionError("visible current/physical-time axis label falsified")
    sign="+q" if native["source_charge_c"]>0 else "−q"
    # Diagnose falsified physical q sign before the broader scene-caption
    # attribution check. These are two independently meaningful boundaries.
    if not page.locator("#m38b-q-sign").is_visible() or page.locator("#m38b-q-sign").text_content().strip()!=sign:
        raise AssertionError("visible source charge sign contradicts physical signed q")
    expected_scene=[
        sign,"Grounded","+u","+w",
        "One fixed equal-metre scale in both axes; no world-xy claim"
    ]
    if page.locator("#m38b-scene text").all_text_contents()!=expected_scene:
        raise AssertionError("visible local source physical axes falsified")
    slider=page.locator("#m38b-index")
    if (slider.input_value()!="0" if no_js else False):
        raise AssertionError("no-JS first physical time index falsely selected")
    if slider.get_attribute("min")!="0" or slider.get_attribute("max")!=str(len(rows)-1):
        raise AssertionError("false physical time slider range")
    if no_js:
        if not page.locator("#m38b-nojs").is_visible():
            raise AssertionError("no-JS fallback not visibly honest")
        if not slider.is_disabled() or not page.locator("#m38b-play").is_disabled() or not page.locator("#m38b-reset").is_disabled():
            raise AssertionError("no-JS first physical time controls must remain disabled")
    else:
        if slider.is_disabled() or page.locator("#m38b-play").is_disabled() or page.locator("#m38b-reset").is_disabled():
            raise AssertionError("genuine JS physical snapshot controls not available")


def _check_frame(page, data, index, *, no_js):
    r=data["readouts"][index]
    for selector,key in LABELS.items():
        _string(page,selector,r[key],
                ("no-JS first SI/world label mismatch" if no_js else "visible physical SI/world label mismatch"))
    for selector,attr,key in (
        ("#m38b-charge","cx","scene_x_px"),
        ("#m38b-charge","cy","scene_y_px"),
        ("#m38b-q-sign","x","sign_x_px"),
        ("#m38b-q-sign","y","sign_y_px"),
        ("#m38b-current-marker","cx","chart_x_px"),
        ("#m38b-current-marker","cy","chart_y_px"),
    ):
        _attr(page,selector,attr,r[key],
              "no-JS first physical scene/current marker mismatch" if no_js else
              "visible selected physical scene/current marker mismatch")
    if page.locator("#m38b-index").input_value()!=str(index):
        raise AssertionError("physical time slider differs from selected genuine sample")


def check(directory: Path):
    from playwright.sync_api import sync_playwright

    path=directory/"m38b-line-viewer.json"
    manifest=json.loads(path.read_text(encoding="utf-8"))
    oracle=check_line_view(manifest)
    native=json.loads((directory/"m38b-line-physics.json").read_text(encoding="utf-8"))
    if check_line_record(native)["independently_checked_kelvin_states"]!=len(manifest["readouts"]):
        raise AssertionError("separate original physical SI history is incomplete")
    if native!=manifest["trace"]:
        raise AssertionError("viewer embedded physics is not identical to actual native Kelvin SI record")
    target=directory/"m38b-line.html"
    if not target.is_file():
        raise AssertionError("offline student viewer HTML missing")
    pictures=directory/"screenshots"
    pictures.mkdir(parents=True,exist_ok=True)
    requests=[]
    errors=[]
    inspected=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True)
        try:
            page=browser.new_page(viewport={"width":1250,"height":900},device_scale_factor=1)
            page.on("request",lambda request: requests.append(request.url)
                    if request.url.startswith(("https://","http://")) else None)
            page.on("pageerror",lambda exc: errors.append(str(exc)))
            page.goto(target.resolve().as_uri(),wait_until="load",timeout=45000)
            embedded=page.evaluate("() => JSON.parse(document.getElementById('m38b-data').textContent)")
            if embedded!=manifest or check_line_view(embedded)!=oracle:
                raise AssertionError("embedded current viewer payload contradicts independent native SI")
            if page.evaluate("() => window.m38bViewer?.sampleCount")!=len(manifest["readouts"]):
                raise AssertionError("browser did not load the genuine real sampled Kelvin history")
            if page.evaluate("() => window.m38bViewer?.sourceRole")!="teacher_hypothetical":
                raise AssertionError("off-axis browser is misattributed to the original workbook")
            _check_static(page,manifest,no_js=False)
            for i in sorted(set((0,1,len(manifest["readouts"])//4,
                                 len(manifest["readouts"])//2,
                                 3*len(manifest["readouts"])//4,
                                 len(manifest["readouts"])-1))):
                page.locator("#m38b-index").evaluate(
                    "(element,i) => {element.value=String(i);element.dispatchEvent(new Event('input',{bubbles:true}));}",
                    i
                )
                if page.evaluate("() => window.m38bViewer?.selectedIndex")!=i:
                    raise AssertionError("script did not select actual expected Kelvin time index")
                _check_frame(page,manifest,i,no_js=False)
                if i in (0,len(manifest["readouts"])//2,len(manifest["readouts"])-1):
                    pic=pictures/("time-"+str(i)+".png")
                    page.screenshot(path=str(pic),full_page=True)
                    inspected.append({"index":i,
                                     "time_s":manifest["trace"]["samples"][i]["time_s"],
                                     "signed_current_a":manifest["trace"]["samples"][i]["wire_current_sphere_to_ground_a"],
                                     "screenshot":pic.name})
            page.locator("#m38b-reset").click()
            _check_frame(page,manifest,0,no_js=False)
            page.locator("#m38b-play").click()
            page.wait_for_function("() => window.m38bViewer && window.m38bViewer.selectedIndex>0",
                                   timeout=5000)
            _check_frame(page,manifest,page.evaluate("() => window.m38bViewer.selectedIndex"),no_js=False)
            page.locator("#m38b-reset").click()
            _check_frame(page,manifest,0,no_js=False)
            mobile=browser.new_page(viewport={"width":390,"height":740})
            mobile.goto(target.resolve().as_uri(),wait_until="load",timeout=45000)
            width=mobile.evaluate("document.body.scrollWidth")
            if width>395:
                raise AssertionError("M38b teacher viewer overflows 390px mobile")
            mobile.screenshot(path=str(pictures/"mobile-first.png"),full_page=True)
            mobile.close()
            nojs_context=browser.new_context(java_script_enabled=False,
                                            viewport={"width":920,"height":780})
            try:
                first=nojs_context.new_page()
                first.goto(target.resolve().as_uri(),wait_until="load",timeout=45000)
                _check_static(first,manifest,no_js=True)
                _check_frame(first,manifest,0,no_js=True)
                first.screenshot(path=str(pictures/"no-js-original-first.png"),full_page=True)
                first.close()
            finally:
                nojs_context.close()
            page.close()
        finally:
            browser.close()
    if requests or errors:
        raise AssertionError("offline physics viewer emitted HTTP requests or JavaScript errors: "+
                             repr(requests)+" / "+repr(errors))
    # The untouched JS-disabled DOM is checked ABOVE; this second necessary
    # gate rejects an entire class of forged *rendered* CSS and SVG that
    # nominal text/coordinate DOM checks alone cannot observe.
    _verify_canonical_html(manifest, target)
    result={
        "schema":"conductor_sim.m38b_real_browser_check.v1",
        "original_workbook_cases_newly_solved":0,
        "physics_states_independently_checked":oracle["independently_checked_kelvin_states"],
        "visible_scripted_physical_states":len(inspected),
        "javascript_disabled_first_frame":True,
        "mobile_390px_verified":True,
        "external_http_requests":0,
        "javascript_errors":0,
        "screenshots":inspected,
    }
    (directory/"m38b-chromium-report.json").write_text(
        json.dumps(result,indent=2)+"\n",encoding="utf-8"
    )
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path("output/m38b"))
    args=parser.parse_args()
    print(json.dumps(check(args.output_dir),sort_keys=True))


if __name__=="__main__":
    main()
