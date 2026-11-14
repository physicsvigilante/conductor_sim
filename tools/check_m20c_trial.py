"""M20c real Chromium review of six independent non-Ring paired exports."""
from __future__ import annotations

import argparse
import json
import re
from math import hypot, isclose
from pathlib import Path
import shutil

from playwright.sync_api import Error as PlaywrightError, sync_playwright

EXPECTED = {
    "U1": ("field:U1",),
    "V1": ("sphere:V1",),
    "R1": ("rod:R1",),
    "P1": ("point:P1", "field:P1"),
    "S1": ("shell:S1",),
    "D1": ("disc:D1",),
}


def check(path: Path, *, case: str, screenshot: Path | None = None):
    if case not in EXPECTED:
        raise ValueError("unknown M20c browser case")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    errors, console, requests = [], [], []
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except PlaywrightError:
            binary = shutil.which("chromium") or shutil.which("chromium-browser")
            if binary is None:
                raise
            browser = p.chromium.launch(headless=True, executable_path=binary,
                                        args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 970})
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: console.append(m.text) if m.type == "error" else None)
        page.on("request", lambda r: requests.append(r.url))
        page.set_content(path.read_text(encoding="utf-8"), wait_until="load")
        page.wait_for_function(
            "window.conductorComparison && window.conductorComparison.loaded",
            timeout=20000)
        manifest = page.evaluate(
            'JSON.parse(document.getElementById("comparison-data").textContent)')
        assert manifest["schema"] == "conductor_sim.scenario_comparison.v1"
        assert tuple(manifest["source_ids"]) == EXPECTED[case]
        assert len(manifest["revisions"]) == 1
        assert manifest["time_interpolation"] is False
        assert len(manifest["frame_times_s"]) == 9
        assert manifest["physical_sample_count"] >= 16
        assert manifest["frame_times_s"][0] == 0.
        assert manifest["frame_times_s"][-1] > 0.
        assert "Teacher hypothetical" in " ".join(manifest["provenance"])
        assert page.evaluate("window.conductorComparison.frameCount") == 9
        assert page.locator("#compare-time").input_value() == "0"
        views = page.evaluate("""() => Array.from(document.querySelectorAll('iframe')).map(f => {
            const w=f.contentWindow, d=w.document;
            const c=d.getElementById('scene');
            const ctx=c.getContext('2d');
            const pixels=ctx.getImageData(0,0,c.width,c.height).data;
            let ink=0;
            for (let i=0;i<pixels.length;i+=4) {
                if (pixels[i+3]>0 && pixels[i]+pixels[i+1]+pixels[i+2]<650) ink++;
            }
            return {mode:w.conductorViewer.mode,frames:w.conductorViewer.frameCount,
                sampleCount:w.conductorViewer.motionSampleCount,
                sources:d.getElementById('sources').innerText,painted:ink,
                startIndex:Number(d.getElementById('frame').value)};
        })""")
        assert len(views) == 2
        assert all(v["mode"] == "elapsed-time-prescribed" for v in views)
        assert all(v["frames"] == 9 and v["startIndex"] == 0 for v in views)
        assert all(v["sampleCount"] == manifest["physical_sample_count"] for v in views)
        assert all(v["painted"] > 50 for v in views)
        for view in views:
            for source_id in EXPECTED[case]:
                assert "[" + source_id + "]" in view["sources"], view["sources"]
        assert all(v["field_frame_count"] == 9 for v in (
            page.evaluate("""() => Array.from(document.querySelectorAll('iframe')).map(f => (
                {field_frame_count:f.contentWindow.conductorViewer.frameCount}))""")))
        for idx in (1, 4, 8):
            page.locator("#compare-time").fill(str(idx))
            assert page.evaluate("window.conductorComparison.frameIndex") == idx
            actual = page.evaluate(
                "() => [...document.querySelectorAll('iframe')].map("
                "f => Number(f.contentWindow.document.getElementById('frame').value))")
            assert actual == [idx, idx]
            t = manifest["frame_times_s"][idx]
            assert isclose(float(page.locator("#compare-clock").inner_text().split()[0]),
                           t, rel_tol=1e-4, abs_tol=1e-6)
            assert "Physical t = " in page.locator("#readout-0").inner_text()
            assert "Physical t = " in page.locator("#readout-1").inner_text()
            dp = hypot(*(manifest["position_m"][idx][0][j] -
                         manifest["position_m"][idx][1][j] for j in range(3)))
            dv = hypot(*(manifest["velocity_m_s"][idx][0][j] -
                         manifest["velocity_m_s"][idx][1][j] for j in range(3)))
            actual_delta = page.locator("#compare-delta").inner_text()
            assert "|r(variation)" in actual_delta
            # JavaScript toPrecision(5) can choose decimal while Python .5g
            # chooses exponent notation. Parse numerical values, never text format.
            shown = re.findall(
                r"=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*m(?:/s)?",
                actual_delta,
            )
            assert len(shown) == 2, actual_delta
            assert isclose(float(shown[0]), dp, rel_tol=5e-5, abs_tol=1e-12)
            assert isclose(float(shown[1]), dv, rel_tol=5e-5, abs_tol=1e-12)
        if case == "S1":
            assert all(x == y for x, y in manifest["position_m"])
            assert all(x == y for x, y in manifest["velocity_m_s"])
            assert manifest["potential_energy_j"][-1][0] != manifest["potential_energy_j"][-1][1]
        else:
            assert manifest["position_m"][-1][0] != manifest["position_m"][-1][1]
        page.locator("#compare-reset").click()
        assert page.locator("#compare-time").input_value() == "0"
        # The existing M20a player normalizes one sweep to four wall-clock
        # seconds at 1x, independent of the physical duration. Use its
        # established 4x/310 ms acceptance cadence, not physical seconds.
        page.locator("#compare-speed").select_option("4")
        page.locator("#compare-play").click()
        page.wait_for_timeout(310)
        assert page.evaluate("window.conductorComparison.frameIndex") > 0
        assert page.evaluate("window.conductorComparison.playing") is True
        page.locator("#compare-play").click()
        assert page.evaluate("window.conductorComparison.playing") is False
        if screenshot:
            screenshot.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(screenshot), full_page=False)
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth<=window.innerWidth")
        assert page.locator("#viewer-0").is_visible() and page.locator("#viewer-1").is_visible()
        assert not errors, errors
        assert not console, console
        assert not requests, requests
        browser.close()
    return dict(case=case, source_ids=list(EXPECTED[case]),
                independently_solved_panels=2,
                physical_samples=manifest["physical_sample_count"],
                real_saved_snapshots=9,
                painted_scene_pixels=[v["painted"] for v in views],
                identical_world_plane=True,
                playback_scrub_pause_reset="passed",
                mobile_width_px=390, javascript_errors=errors,
                external_requests=requests)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", type=Path)
    parser.add_argument("--case", choices=sorted(EXPECTED), required=True)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.html, case=args.case, screenshot=args.screenshot),
                     indent=2))


if __name__ == "__main__":
    main()
