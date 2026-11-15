"""M26a Chromium verification of a newly composed physical model.

The Python engine precomputes physics. Browser verification checks source
identity, saved physical time, switches and offline presentation, not a new
force law or independent integrator.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

from examples.m26a_driven_composition import export


def check(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    html_path = output / "m26a-driven.html"
    metadata = export(html_path)
    html = html_path.read_text(encoding="utf-8")
    assert len(metadata["model"]["changes"]) == 1
    assert metadata["model"]["changes"][0]["after"]["id"] == "point:A"
    assert metadata["numerical_result"]["termination"] == "duration"
    assert len(metadata["numerical_result"]["switches"]) == 1

    page_errors, console_errors, requests = [], [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 850})
        page.on("pageerror", lambda e: page_errors.append(str(e)))
        page.on("console", lambda e: console_errors.append(e.text) if e.type == "error" else None)
        page.on("request", lambda r: requests.append(r.url))
        page.set_content(html, wait_until="load")
        page.wait_for_function("window.conductorDrivenViewer && window.conductorDrivenViewer.loaded")
        viewer = page.evaluate("""() => ({
            source_ids: data.source_geometry.entries.map(e => e.source_id),
            statuses: data.source_geometry.entries.map(e => e.status),
            shapes: data.source_geometry.entries.map(e => e.shape),
            saved_time: window.conductorDrivenViewer.physicalTime,
            switch_times: data.switches.map(s => s.time_s),
            source_annotations: data.study.model.base.source_annotations,
            saved_sample_times: data.display_states.map(s => s.time_s)
        })""")
        assert viewer["source_ids"] == ["point:A"], viewer
        assert viewer["source_annotations"] == [["point:A", "Held point A"]]
        assert viewer["statuses"] == ["in-slice"] and viewer["shapes"] == ["point"]
        assert viewer["saved_time"] == 0
        assert viewer["switch_times"] == [.02]
        assert page.locator(".source-geometry-entry").count() == 1
        assert page.locator("#source-geometry").is_checked()
        assert "Held point A" in page.locator("#sources").inner_text()
        page.locator("#frame").evaluate("""(el) => {
            el.value = el.max;
            el.dispatchEvent(new Event('input', {bubbles:true}));
        }""")
        at_last = page.evaluate("window.conductorDrivenViewer.physicalTime")
        assert at_last == viewer["saved_sample_times"][-1]
        page.locator("#reset").click()
        assert page.evaluate("window.conductorDrivenViewer.physicalTime") == 0
        page.locator("#source-geometry").uncheck()
        assert not page.locator("#source-geometry").is_checked()
        page.locator("#source-geometry").check()
        page.screenshot(path=str(output / "m26a-wide.png"), full_page=True)

        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(output / "m26a-mobile.png"), full_page=True)
        viewport = page.evaluate("""() => ({
            inner_width: window.innerWidth,
            document_scroll_width: document.documentElement.scrollWidth,
            scene_width: document.getElementById('scene').getBoundingClientRect().width
        })""")
        assert viewport["inner_width"] == 390
        assert viewport["scene_width"] <= 390
        assert not page_errors and not console_errors and not requests, (
            page_errors, console_errors, requests
        )
        browser.close()

    report = dict(
        status="pass",
        model_revision=metadata["model"]["child_model_revision_id"],
        root_parent=metadata["model"]["parent_model_revision_id"],
        html_sha256=sha256(html.encode("utf-8")).hexdigest(),
        physical_termination=metadata["numerical_result"]["termination"],
        source_id="point:A",
        switch_s=.02,
        last_saved_time_s=at_last,
        browser_errors=page_errors,
        console_errors=console_errors,
        external_requests=requests,
        mobile_viewport=viewport,
        scope="Real Chromium and saved time/source UI only, not a second force solver.",
    )
    (output / "m26a-browser.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/m26a-review"))
    args = parser.parse_args()
    print(json.dumps(check(args.output), sort_keys=True))


if __name__ == "__main__":
    main()
