"""Actual offline Chromium mobile/desktop JS-on/off truth for both JEE pages."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tools.jee12_walkthrough_reference import verify

CASES = (
    ("p1-external-work", "p01-single-08",
     ("p1-external-ev", "p1-field-ev", "p1-teacher-external-ev")),
    ("p3-solid-shell-energy", "p03-single-27",
     ("p3-before-j", "p3-after-j", "p3-released-j", "p3-ideal-j",
      "p3-teacher-released-j")),
)


def check(out: Path) -> dict:
    from playwright.sync_api import sync_playwright

    verify(out)
    observed = {}
    screenshots = out / "screenshots"
    screenshots.mkdir(exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            for name, case, selectors in CASES:
                record = json.loads((out/(name+".json")).read_text(encoding="utf-8"))
                assert record["source"]["case_id"] == case
                if case == "p01-single-08":
                    expected = [
                        record["native"]["external_work_j"] / 1.602176634e-19,
                        record["native"]["field_work_j"] / 1.602176634e-19,
                        record["teacher_hypothetical"]["native_external_work_j"] / 1.602176634e-19,
                    ]
                else:
                    expected = [record["native"]["energy_before_j"],
                                record["native"]["energy_after_j"],
                                record["native"]["energy_released_j"],
                                record["printed_answer"]["ideal_thin_limit_j"],
                                record["teacher_hypothetical"]["native_energy_released_j"]]
                for label, width, js in (
                    ("desktop-js", 1280, True), ("mobile-js", 375, True),
                    ("desktop-nojs", 1280, False), ("mobile-nojs", 375, False),
                ):
                    context = browser.new_context(
                        viewport={"width":width,"height":850 if width==1280 else 720},
                        device_scale_factor=1, java_script_enabled=js)
                    page = context.new_page()
                    requests, errors = [], []
                    page.on("request", lambda request:
                            requests.append(request.url)
                            if request.url.startswith(("http://","https://")) else None)
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    try:
                        page.goto((out/(name+".html")).resolve().as_uri(),
                                  wait_until="load", timeout=45000)
                        body = page.locator("body").inner_text()
                        assert "Original requested answer" in body
                        assert "TEACHER HYPOTHETICAL" in body
                        assert record["source"]["anchor"].split()[-1] in body
                        assert len(body) > 400
                        actual = []
                        for selector, reference in zip(selectors, expected):
                            item = page.locator("#"+selector)
                            assert item.is_visible(), (name, label, selector)
                            value = float(item.get_attribute("data-si"))
                            assert math.isclose(value, reference, rel_tol=3e-11, abs_tol=1e-16)
                            assert item.inner_text().strip(), (name, label, selector)
                            actual.append(value)
                        svgs = page.locator("svg[role=img]")
                        assert svgs.count() == (1 if case == "p01-single-08" else 2)
                        if case == "p03-single-27":
                            for svg in svgs.all():
                                rings = svg.locator("circle")
                                assert [rings.nth(i).get_attribute("r") for i in (0,1)] == ["110","55"]
                            assert "line width is not the 1 µm material thickness" in body
                            assert "filled solid" in body
                        else:
                            assert "Work by external agent (+)" in body
                            assert "Work by electric field" in body
                        dims=page.evaluate("() => ({scroll:document.documentElement.scrollWidth,viewport:innerWidth})")
                        assert dims["scroll"] <= dims["viewport"] + 1, (name,label,dims)
                        assert not requests and not errors, (name,label,requests,errors)
                        filename = name+"-"+label+".png"
                        page.screenshot(path=str(screenshots/filename), full_page=True,
                                        animations="disabled")
                        observed[name+"/"+label] = dict(case_id=case, values_si=actual,
                            svg_count=svgs.count(), horizontal_overflow_px=0,
                            no_external_requests=True, no_js_page_errors=True,
                            screenshot="screenshots/"+filename)
                    finally:
                        context.close()
        finally:
            browser.close()
    result = {"schema":"conductor_sim.jee12_walkthrough_browser.v1",
              "eight_actual_chromium_modes":len(observed),
              "observed":observed,
              "warning":"chromium observation is not independent physics verification"}
    (out/"chromium-observations.json").write_text(
        json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir",type=Path,required=True)
    args=ap.parse_args()
    print(json.dumps({"validated_browser_modes":check(args.input_dir)["eight_actual_chromium_modes"]}))


if __name__=="__main__":
    main()
