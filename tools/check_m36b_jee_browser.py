"""M36b: REAL offline Chromium inspection of actual seven JEE source viewers.

No browser-side physics calculation. The separate source-free checker verifies
native outputs; this only checks meaningful rendered teaching content.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED_PHRASES = {
    "p02-single-19": ("sheet-stop", "Physical t"),
    "p03-single-30": ("plate-transit", "Physical t"),
    "p05-single-47": ("dipole energy Q47",),
    "p12-linked-40": ("Ring A", "Ring B", "Upward"),
    "p15-archive-02": ("2012 proton",),
    "p02-single-15": ("far-separated connected spheres",),
    "jee2021-p1-q09-q10-equipotential": ("2021 Q9",),
}


def check_viewers(output_dir: Path) -> dict:
    from playwright.sync_api import sync_playwright

    manifest=json.loads((output_dir/"native-replays.json").read_text(encoding="utf-8"))
    entries=manifest["source_apparatus"]
    if set(entries) != set(EXPECTED_PHRASES):
        raise ValueError("missing preregistered source viewer")
    (output_dir/"screenshots").mkdir(exist_ok=True)
    results={}
    with sync_playwright() as playwright:
        browser=playwright.chromium.launch(headless=True)
        try:
            for id_, phrases in EXPECTED_PHRASES.items():
                page=browser.new_page(viewport={"width":1280,"height":800},device_scale_factor=1)
                external=[]
                page.on("request",lambda request: external.append(request.url)
                        if request.url.startswith(("https:","http:")) else None)
                file=output_dir/entries[id_]["viewer_file"]
                if not file.is_file():
                    raise ValueError("real HTML viewer not found for "+id_)
                try:
                    page.goto(file.resolve().as_uri(),wait_until="load",timeout=45000)
                    body=page.locator("body").inner_text(timeout=15000)
                    for phrase in phrases:
                        if phrase.casefold() not in body.casefold():
                            raise AssertionError(f"missing visible {id_} physics label: {phrase}")
                    if len(body.strip())<100:
                        raise AssertionError("empty/truncated visual content "+id_)
                    if external:
                        raise AssertionError(f"viewer made unexpected network requests {id_}: {external}")
                    if page.locator("svg, canvas").count() < 1:
                        raise AssertionError("source viewer has no rendered physical graphic "+id_)
                    path=output_dir/"screenshots"/(id_+".png")
                    page.screenshot(path=str(path),full_page=False,animations="disabled")
                    results[id_]={
                        "viewer":file.name,
                        "visible_required_phrases":list(phrases),
                        "visible_character_count":len(body),
                        "svg_and_canvas_elements":page.locator("svg, canvas").count(),
                        "network_requests_to_http":0,
                        "screenshot":str(path.relative_to(output_dir)),
                        "status":"passed_real_chromium_offline",
                    }
                finally:
                    page.close()
        finally:
            browser.close()
    report={
        "schema":"conductor_sim.m36b_chromium_source_viewers.v1",
        "tested_original_source_apparatus":len(results),
        "passed_actual_browser_viewers":len(results),
        "cases":results,
        "do_not_confuse_with_independent_numerical_physics":True,
        "unsupported_local_3d_case_never_rendered":True,
    }
    (output_dir/"browser-checks.json").write_text(
        json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8"
    )
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    x=check_viewers(args.output_dir)
    print("M36b genuine Chromium viewers inspected:",x["passed_actual_browser_viewers"])


if __name__=="__main__":
    main()
