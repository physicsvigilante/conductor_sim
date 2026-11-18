"""Real offline Chromium verification of M37b original-source constant-speed viewer.

The standalone Kelvin SI checker is the physical oracle, not Chromium.
Browser evidence checks the *actual embedded payload*, visible original/
teacher attribution, all stored-state controls and no-JS first snapshot.
"""
from __future__ import annotations

import argparse
import json
from math import isclose
from pathlib import Path

from tools.m37b_source_reference import check as independent_jee_si



def _verify_nojs_first_frame(page, manifest):
    """Attest the literal, never-JS-repaired first frame in REAL Chromium.

    Full visible physical state and drawn geometry must match the separately
    independently SI-checked native record, not merely an embedded JSON blob.
    JS-enabled startup calls show(0) and can conceal false ORIGINAL HTML
    attributes; checking the no-JS DOM catches that distinct forgery.
    """
    first = manifest["readouts"][0]
    layout = manifest["layout"]
    native = manifest["trace"]
    if not page.locator("#m37b-nojs").is_visible():
        raise AssertionError("no-JS static fallback description missing or hidden")
    for name in ("#m37b-source", "#m37b-original-statement",
                 "#m37b-teacher-attribution", "#m37b-scene",
                 "#m37b-current-chart"):
        if not page.locator(name).is_visible():
            raise AssertionError(f"no-JS source/teacher/physical scene is hidden: {name}")
    if page.locator("#m37b-source h1").inner_text().strip() != "PRINTED WORKBOOK p8 · Multiple Correct Q21":
        raise AssertionError("no-JS workbook source heading falsified")
    if page.locator("#m37b-original-statement").inner_text().strip() != (
            "SOURCE-GIVEN: positive q approaches a grounded metallic sphere of radius R "
            "at externally maintained constant small speed v. Workbook answer key: "
            "options (1) and (3), conventional sphere→ground current increasing."):
        raise AssertionError("no-JS original source claim falsified")
    if page.locator("#m37b-teacher-attribution").inner_text().strip() != (
            "ALL numbers (q, R, d₀, v, duration, ε) displayed here are TEACHER-CHOSEN "
            "illustrations, NOT printed workbook numerical givens."):
        raise AssertionError("no-JS teacher numerical provenance falsified")
    slider = page.locator("#m37b-index")
    if slider.input_value() != "0" or slider.get_attribute("min") != "0" or (
            slider.get_attribute("max") != str(len(manifest["readouts"])-1)):
        raise AssertionError("no-JS first physical time index/slider range is falsified")
    if not slider.is_disabled() or not page.locator("#m37b-start").is_disabled() or (
            not page.locator("#m37b-play").is_disabled()):
        raise AssertionError("no-JS inert playback controls must all remain disabled")
    for selector,key in (
        ("#m37b-time","time_label"),
        ("#m37b-distance","distance_label"),
        ("#m37b-Q","induced_charge_label"),
        ("#m37b-current","wire_current_label"),
        ("#m37b-opposite","opposite_current_label"),
    ):
        if page.locator(selector).inner_text().strip() != first[key]:
            raise AssertionError(f"no-JS first physical SI label {selector} falsified")
    expected_sign = "+q" if native["source_charge_c"] > 0 else "−q"
    if page.locator("#m37b-q-sign").text_content().strip() != expected_sign:
        raise AssertionError("no-JS source charge sign label contradicts physical q")
    for selector,attribute,physical_value in (
        ("#m37b-sphere","cx",layout["sphere_cx_px"]),
        ("#m37b-sphere","cy",layout["sphere_cy_px"]),
        ("#m37b-sphere","r",layout["sphere_radius_px"]),
        ("#m37b-charge","cx",first["charge_x_px"]),
        ("#m37b-charge","cy",layout["charge_cy_px"]),
        ("#m37b-q-sign","x",first["charge_x_px"]),
        ("#m37b-current-marker","cx",first["chart_x_px"]),
        ("#m37b-current-marker","cy",first["chart_y_px"]),
    ):
        raw = page.locator(selector).get_attribute(attribute)
        try:
            valid = (raw is not None and
                     isclose(float(raw),physical_value,rel_tol=2e-12,abs_tol=5e-9))
        except (ValueError,OverflowError,TypeError):
            valid = False
        if not valid:
            raise AssertionError(
                f"no-JS physical scene/graph coordinate {selector}.{attribute} "
                "does not match the verified first saved Kelvin sample"
            )
    # The no-JS entire drawn curve is itself an executable teaching claim.
    # Verify its 31 actual values rather than relying on JS startup behavior.
    points = page.locator("#m37b-current-line").get_attribute("points")
    plotted = points.split() if points is not None else []
    if len(plotted) != len(manifest["readouts"]):
        raise AssertionError("no-JS physical current curve length falsified")
    for i,(point,row) in enumerate(zip(plotted,manifest["readouts"])):
        xy = point.split(",")
        try:
            correct = len(xy)==2 and all(
                isclose(float(xy[coord]),row[key],rel_tol=3e-10,abs_tol=2e-7)
                for coord,key in ((0,"chart_x_px"),(1,"chart_y_px"))
            )
        except (ValueError,OverflowError,TypeError):
            correct = False
        if not correct:
            raise AssertionError(f"no-JS physical current graph point {i} falsified")

def check(directory: Path):
    from playwright.sync_api import sync_playwright

    manifest=json.loads((directory/"p8-ground-current.json").read_text(encoding="utf-8"))
    oracle=independent_jee_si(manifest)
    target=directory/"p8-ground-current.html"
    if not target.is_file():
        raise AssertionError("source-specific offline viewer absent")
    screenshots=directory/"screenshots"
    screenshots.mkdir(parents=True,exist_ok=True)
    views=[]
    external_requests=[]
    js_errors=[]
    with sync_playwright() as browser_tool:
        browser=browser_tool.chromium.launch(headless=True)
        try:
            page=browser.new_page(viewport={"width":1210,"height":840},device_scale_factor=1)
            page.on("request",lambda r: external_requests.append(r.url)
                    if r.url.startswith(("https://","http://")) else None)
            page.on("pageerror",lambda e: js_errors.append(str(e)))
            page.goto(target.resolve().as_uri(),wait_until="load",timeout=45000)
            if not page.locator("#m37b-source").is_visible():
                raise AssertionError("original workbook/source role not visible")
            if page.locator("#m37b-source h1").inner_text()!="PRINTED WORKBOOK p8 · Multiple Correct Q21":
                raise AssertionError("visible source page and printed question false")
            if not page.locator("#m37b-original-statement").is_visible():
                raise AssertionError("original constant-speed protocol hidden")
            source_text=page.locator("#m37b-original-statement").inner_text().strip()
            exact_original=(
                "SOURCE-GIVEN: positive q approaches a grounded metallic sphere of radius R "
                "at externally maintained constant small speed v. Workbook answer key: "
                "options (1) and (3), conventional sphere→ground current increasing."
            )
            if source_text!=exact_original:
                raise AssertionError("visible original Q21 symbolic source/current claim altered")
            if not page.locator("#m37b-teacher-attribution").is_visible():
                raise AssertionError("source versus teacher numeric attribution hidden from students")
            attribution=page.locator("#m37b-teacher-attribution").inner_text().strip()
            exact_teacher=(
                "ALL numbers (q, R, d₀, v, duration, ε) displayed here are TEACHER-CHOSEN "
                "illustrations, NOT printed workbook numerical givens."
            )
            if attribution!=exact_teacher:
                raise AssertionError("visible teacher-source numeric attribution altered or falsified")
            if not page.locator("#m37b-scene").is_visible() or not page.locator("#m37b-current-chart").is_visible():
                raise AssertionError("physical radial scene/current curve not actually visible")
            embedded=page.evaluate("() => JSON.parse(document.getElementById('m37b-data').textContent)")
            if embedded!=manifest:
                raise AssertionError("embedded viewer data differs from verified native SI JSON")
            if independent_jee_si(embedded)!=oracle:
                raise AssertionError("browser embedded SI proof failed")
            if page.evaluate("() => window.m37bPrescribedViewer?.sampleCount")!=31:
                raise AssertionError("not 31 actual Python-solved prescribed physical states")
            if "teacher-prescribed constant-speed" not in page.evaluate(
                "() => window.m37bPrescribedViewer?.mode"
            ):
                raise AssertionError("time slider misrepresented free Newtonian integration")
            slider=page.locator("#m37b-index")
            if slider.is_disabled():
                raise AssertionError("script-enabled offline slider still disabled")
            positions=manifest["readouts"]
            # The entire plotted current curve and conductor circle are
            # actual DOM geometry, not trustworthy just because the embedded
            # JSON is correct. Check them before any slider interaction.
            phys=manifest["layout"]
            for name,attr,key in (
                ("#m37b-sphere","cx","sphere_cx_px"),
                ("#m37b-sphere","cy","sphere_cy_px"),
                ("#m37b-sphere","r","sphere_radius_px"),
                ("#m37b-charge","cy","charge_cy_px"),
            ):
                value=page.locator(name).get_attribute(attr)
                if value is None or not isclose(float(value),phys[key],rel_tol=3e-12,abs_tol=1e-9):
                    raise AssertionError(f"visible drawn physical geometry {name}.{attr} contradicts native model")
            points=page.locator("#m37b-current-line").get_attribute("points")
            if points is None:
                raise AssertionError("missing actual drawn current-versus-prescribed-time curve")
            plotted=points.split()
            if len(plotted)!=len(positions):
                raise AssertionError("graph lost genuine stored current observations")
            for i,(point,row) in enumerate(zip(plotted,positions)):
                xy=point.split(",")
                if len(xy)!=2 or any(not isclose(float(xy[j]),row[k],
                                                rel_tol=3e-10,abs_tol=2e-7)
                                      for j,k in ((0,"chart_x_px"),(1,"chart_y_px"))):
                    raise AssertionError(f"drawn current graph point {i} contradicts verified physical series")
            if page.locator("#m37b-q-sign").text_content().strip()!="+q":
                raise AssertionError("source says q positive but actual plotted charge sign differs")
            for i in (0,1,10,15,25,30):
                slider.evaluate("(el,i) => {el.value=String(i);el.dispatchEvent(new Event('input',{bubbles:true}));}",i)
                if page.evaluate("() => window.m37bPrescribedViewer?.selectedIndex")!=i:
                    raise AssertionError("slider did not select the requested real source index")
                expected=positions[i]
                labels=(
                    ("#m37b-time","time_label"),
                    ("#m37b-distance","distance_label"),
                    ("#m37b-Q","induced_charge_label"),
                    ("#m37b-current","wire_current_label"),
                    ("#m37b-opposite","opposite_current_label"),
                )
                for locator,key in labels:
                    if page.locator(locator).inner_text().strip()!=expected[key]:
                        raise AssertionError(f"visible snapshot #{i} {key} mismatches native Kelvin SI value")
                for locator,axis,key in (
                    ("#m37b-charge","cx","charge_x_px"),
                    ("#m37b-current-marker","cx","chart_x_px"),
                    ("#m37b-current-marker","cy","chart_y_px"),
                    ("#m37b-q-sign","x","charge_x_px"),
                ):
                    raw=page.locator(locator).get_attribute(axis)
                    if raw is None or not isclose(float(raw),expected[key],rel_tol=2e-12,abs_tol=5e-9):
                        raise AssertionError(f"visible scene/chart element {locator}.{axis} not at actual Python display position")
                if i in (0,15,30):
                    img=screenshots/f"prescribed-time-{i:02}.png"
                    page.screenshot(path=str(img),full_page=True)
                    views.append({"sample_index":i,"time_s":manifest["trace"]["samples"][i]["time_s"],
                                  "current_a":manifest["trace"]["samples"][i]["wire_current_sphere_to_ground_a"],
                                  "screenshot":img.name})
            page.locator("#m37b-start").click()
            if page.locator("#m37b-index").input_value()!="0":
                raise AssertionError("restart does not return to actual t=0")
            small=browser.new_page(viewport={"width":390,"height":720},device_scale_factor=1)
            small.goto(target.resolve().as_uri(),wait_until="load",timeout=45000)
            body_width=small.evaluate("document.body.scrollWidth")
            if body_width>395:
                raise AssertionError("source viewer overflows a 390px mobile screen")
            small.screenshot(path=str(screenshots/"mobile-first.png"),full_page=True)
            small.close()
            offline=browser.new_context(java_script_enabled=False,
                                        viewport={"width":940,"height":740})
            try:
                fallback=offline.new_page()
                fallback.goto(target.resolve().as_uri(),wait_until="load",timeout=45000)
                _verify_nojs_first_frame(fallback,manifest)
                fallback.screenshot(path=str(screenshots/"no-javascript-first.png"),full_page=True)
            finally:
                offline.close()
            page.close()
        finally:
            browser.close()
    if external_requests or js_errors:
        raise AssertionError(f"offline browser had external HTTP or JS errors: {external_requests} / {js_errors}")
    result={
        "native_source_scenarios":1,
        "independently_verified_sample_count":oracle["verified_python_kelvin_states"],
        "actual_playwright_chromium_states":len(views),
        "no_javascript_original_state":True,
        "mobile_render_checked":True,
        "external_http_requests":0,
        "javascript_errors":0,
        "browser_views":views,
    }
    (directory/"chromium-source-report.json").write_text(
        json.dumps(result,indent=2)+"\n",encoding="utf-8")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path("output/m37b"))
    args=parser.parse_args()
    print(json.dumps(check(args.output_dir),sort_keys=True))


if __name__=="__main__":
    main()
