"""Actual Chromium acceptance: M40b true geometry, signed sigma and JS-off fallback."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

from tools.m40b_spherical_atlas_example import run_all
from tools.m40b_spherical_atlas_reference import check_manifest


def check_browser(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    data = run_all(output_dir)
    check_manifest(data)
    page_file = output_dir / "m40b-teacher-atlas.html"
    url = page_file.resolve().as_uri()
    checks = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        for javascript, width, height in (
            (True, 1220, 820), (False, 1220, 820),
            (True, 390, 844), (False, 390, 844),
        ):
            context = browser.new_context(
                viewport={"width":width,"height":height},
                java_script_enabled=javascript,
                device_scale_factor=1,
            )
            page = context.new_page()
            requests = []
            page.on("request",lambda request: requests.append(request.url))
            page.goto(url,wait_until="load")
            assert page.title() == "M40b real spherical sigma teaching atlas"
            assert "TEACHER HYPOTHETICAL" in page.locator("#m40b-role").inner_text()
            assert "0 newly solved original JEE" in page.locator("#m40b-role").inner_text()
            assert "z IS LISTED" in page.locator("#m40b-warning").inner_text()
            assert page.locator("svg[role=img]").count() == 1
            assert page.locator("circle[id^=m40b-void-]").count() == 3
            assert page.locator("circle[id^=m40b-source-]").count() == 4
            assert page.locator('[id^="m40b-tile-"]').count() == 72
            assert "0 C" in page.locator("#m40b-q-outer-grounded").inner_text()
            # Compare the entire exact round-trip SI value, never a hand-typed
            # decimal. A true Gauss ledger can give -9.999999999999999e-10 C
            # rather than the rounded -1e-09 C; both mean the same physics.
            from conductor_sim.spherical_sigma_atlas import _si
            assert page.locator("#m40b-q-outer-isolated").inner_text() == _si(
                data["outer"]["isolated_whole_face_charge_c"], "C"
            )
            assert page.locator("#m40b-q-outer-grounded").inner_text() == _si(
                data["outer"]["grounded_whole_face_charge_c"], "C"
            )
            for i,cav in enumerate(data["cavities"]):
                block = page.locator("#m40b-cavity-{}".format(i))
                assert cav["cavity_id"] in block.inner_text()
                for source in cav["sources"]:
                    assert source["source_id"] in block.inner_text()
                    assert repr(source["position_m"][2]) in block.inner_text()
                for j,sample in enumerate(cav["samples"]):
                    tile = page.locator("#m40b-tile-{}-{}".format(i,j))
                    sigma = sample["isolated_sigma_c_per_m2"]
                    assert repr(abs(sigma)) in tile.inner_text() if sigma else "0 C/m²" in tile.inner_text()
                    assert "raw n=" in tile.inner_text()
            svg_circles = [page.locator("#m40b-void-{}".format(i)) for i in range(3)]
            outer = data["outer_sphere"]
            for i,cav in enumerate(data["cavities"]):
                expected_x = 425.+(cav["center_m"][0]-outer["center_m"][0])*375./outer["radius_m"]
                expected_y = 425.-(cav["center_m"][1]-outer["center_m"][1])*375./outer["radius_m"]
                assert abs(float(svg_circles[i].get_attribute("cx"))-expected_x) <= 1e-8
                assert abs(float(svg_circles[i].get_attribute("cy"))-expected_y) <= 1e-8
            assert all(req.startswith("file://") for req in requests), requests
            filename = "m40b-{}-{}px.png".format("js-on" if javascript else "js-off",width)
            page.screenshot(path=str(output_dir / filename),full_page=False)
            checks.append({"javascript_enabled":javascript,"viewport_width":width,
                           "actual_cavities":3,"actual_sources":4,
                           "actual_sigma_point_tiles":72,"remote_requests":0})
            context.close()
        # A deliberately forged script in this same standalone HTML must be blocked
        # by CSP rather than being executed as an added source or numerical result.
        forged = output_dir / "m40b-forged-script.html"
        source = page_file.read_text(encoding="utf-8")
        assert "<script" not in source.lower()
        forged.write_text(source.replace(
            "</body>","<script>window.__m40b_hacked__ = true;</script></body>"),
            encoding="utf-8")
        context = browser.new_context(java_script_enabled=True)
        page = context.new_page()
        page.goto(forged.resolve().as_uri(),wait_until="load")
        assert page.evaluate("window.__m40b_hacked__ === true") is False
        context.close()
        browser.close()
    outcome = {"result":"PASS","actual_chromium_js_on_off_mobile":checks,
               "CSP_script_injection_executed":False,
               "surface_samples_independently_verified":3*24*2,
               "original_jee_items_solved":0}
    (output_dir/"m40b-browser-evidence.json").write_text(
        json.dumps(outcome,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path("output/m40b-browser"))
    args = parser.parse_args()
    print(json.dumps(check_browser(args.output_dir),indent=2))


if __name__ == "__main__":
    main()
