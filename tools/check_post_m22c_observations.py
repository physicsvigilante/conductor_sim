"""Browser regressions for unsampled field jumps and defined surface work."""
from __future__ import annotations

import argparse
import json
from math import pi
from pathlib import Path
import shutil

from playwright.sync_api import Error as PlaywrightError, sync_playwright

from conductor_sim import EPSILON_0, PrescribedField, UniformDiskAxisField, UniformShellField
from conductor_sim.jee_potential_explorer import (
    endpoint_field_work, render_static_profile, sample_static_profile)


def check(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    results = []
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except PlaywrightError:
            binary = shutil.which("chromium") or shutil.which("chromium-browser")
            if binary is None:
                raise
            browser = p.chromium.launch(headless=True, executable_path=binary,
                                        args=["--no-sandbox"])
        for case in ("unsampled-shell-jump", "unsampled-disc-jump"):
            shell = case == "unsampled-shell-jump"
            source = UniformShellField(1e-9, .101) if shell else UniformDiskAxisField(1e-9, .08)
            field = PrescribedField((source,))
            labels = (("source:fixed", "Prescribed shell" if shell else "Prescribed disc"),)
            profile = sample_static_profile(field, source_annotations=labels,
                origin_m=(0., 0., 0.), axis=(1., 0., 0.) if shell else (0., 0., 1.),
                start_m=0. if shell else -.13, end_m=.24 if shell else .17,
                sample_count=21, coordinate="radial" if shell else "axial")
            start = (.101, 0., 0.) if shell else (0., 0., 0.)
            end = (.2, 0., 0.) if shell else (0., 0., .15)
            work = endpoint_field_work(field, source_annotations=labels,
                probe_charge_c=1e-9, start_m=start, end_m=end)
            html = render_static_profile(profile, work=work,
                title="Field jump between saved samples — " + ("shell" if shell else "disc"),
                source_note="Teacher-authored regression: the E graph must not bridge a jump missed by the sample grid.",
                surface_coordinate=.101 if shell else None)
            html_path = output / (case + ".html")
            html_path.write_text(html, encoding="utf-8")
            (output/(case+".json")).write_text(json.dumps({"profile": profile, "work": work},
                indent=2, allow_nan=False)+"\n", encoding="utf-8")
            page = browser.new_page(viewport={"width": 1360, "height": 1000})
            errors, requests = [], []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda e: errors.append(e.text) if e.type == "error" else None)
            page.on("request", lambda r: requests.append(r.url))
            page.set_content(html, wait_until="load")
            assert page.locator("svg.plot").count() == 2
            counts = page.locator("svg.plot").evaluate_all(
                "els => els.map(e=>e.querySelectorAll('polyline').length)")
            assert counts == [2, 1], counts
            assert "Lines also stop between samples" in page.locator("main").inner_text()
            assert "Endpoint E is undefined" in page.locator("#work-summary").inner_text()
            record = page.evaluate("JSON.parse(document.getElementById('observation-data').textContent)")
            assert record["endpoint_work"]["start_electric_field_v_m"] is None
            assert "start" in record["endpoint_work"]["unavailable_endpoint_fields"]
            assert len(record["profile"]["rows"]) == 21
            assert not any(r["field_undefined"] for r in record["profile"]["rows"])
            k = 1 / (4*pi*EPSILON_0)
            expected = (1e-18*k*(1/.101 - 1/.2) if shell else
                        2*k*1e-18*(1/.08-1/((.08**2+.15**2)**.5+.15)))
            assert abs(record["endpoint_work"]["field_work_j"]-expected) < 1e-20
            page.screenshot(path=str(output/(case+".png")), full_page=True)
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth+1")
            assert not errors and not requests, (errors, requests)
            results.append(dict(case=case, physical_samples=21,
                field_polylines=counts[0], potential_polylines=counts[1],
                field_work_j=work["field_work_j"], start_field="undefined, not zero",
                javascript_errors=errors, external_requests=requests, mobile_width_px=390))
            page.close()
        browser.close()
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("output/post-m22c"))
    args = ap.parse_args()
    result = check(args.output)
    (args.output/"browser-results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
