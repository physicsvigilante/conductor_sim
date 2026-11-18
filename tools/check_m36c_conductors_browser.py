"""M36c: visually check REAL source-conductor HTML in offline Chromium.

Labels and screenshots alone are NOT SI physics evidence. Numeric physical
claims are independently tested in m36c_jee_conductor_reference.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED=(
    "p03-single-27","p10-linked-13+p10-linked-14","p14-numerical-07",
    "p26-conductor-potential","p27-cavity-two-shells",
    "p69-cavity-central-vs-shifted","p09-multiple-27",
)
SOURCE_MARKERS={
    # These are actual visible title/labels in the accepted M5 offline viewer,
    # not hidden source JSON metadata or guessed descriptive prose.
    "p03-single-27":("p03-single-27","Stored state"),
    "p10-linked-13+p10-linked-14":("p10-linked-13+p10-linked-14","Stored state"),
    "p14-numerical-07":("p14-numerical-07","Stored state"),
    "p26-conductor-potential":("PDF p.26","Potential"),
    "p27-cavity-two-shells":("PDF p.27","Potential"),
    "p69-cavity-central-vs-shifted":("PDF p.69","Potential"),
    "p09-multiple-27":("p09-multiple-27","Potential"),
}


def check(directory:Path)->dict:
    from playwright.sync_api import sync_playwright

    manifest=json.loads((directory/"native-conductors.json").read_text(encoding="utf-8"))
    sources=manifest["native_source_apparatus"]
    if tuple(sources)!=EXPECTED:
        raise ValueError("source physical apparatus were altered")
    screenshots=directory/"screenshots"
    screenshots.mkdir(exist_ok=True)
    completed={}
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        try:
            for name in EXPECTED:
                case=sources[name]
                file=directory/case["viewer_file"]
                if not file.is_file():
                    raise AssertionError("missing real source-conductor HTML: "+name)
                page=browser.new_page(viewport={"width":1280,"height":800},device_scale_factor=1)
                requests=[]
                errors=[]
                page.on("request",lambda r: requests.append(r.url)
                        if r.url.startswith(("http://","https://")) else None)
                page.on("pageerror",lambda e: errors.append(str(e)))
                try:
                    page.goto(file.resolve().as_uri(),wait_until="load",timeout=45000)
                    content=page.locator("body").inner_text(timeout=15000)
                    if len(content)<120:
                        raise AssertionError("source teaching viewer empty: "+name)
                    graphics=page.locator("svg,canvas").count()
                    if graphics<1:
                        raise AssertionError("no real conductor physical graphic: "+name)
                    if requests or errors:
                        raise AssertionError("offline browser network or JS error: "+
                                             name+" "+str((requests,errors)))
                    for keyword in SOURCE_MARKERS[name]:
                        if keyword.casefold() not in content.casefold():
                            raise AssertionError("missing source interpretation label "+
                                                 name+": "+keyword)
                    shot=screenshots/(name+".png")
                    page.screenshot(path=str(shot),full_page=True,animations="disabled")
                    completed[name]={
                        "source_case_id":name,
                        "graphics":graphics,
                        "visible_characters":len(content),
                        "expected_source_labels":list(SOURCE_MARKERS[name]),
                        "http_requests":0,
                        "js_errors":0,
                        "screenshot":str(shot.relative_to(directory)),
                        "viewer_mode":"actual native STATIC conductor field/equilibrium",
                        "result":"success_real_chromium",
                    }
                finally:
                    page.close()
        finally:
            browser.close()
    return {
        "schema":"conductor_sim.m36c_jee_conductor_chromium.v1",
        "inspected_distinct_source_apparatus":len(completed),
        "source_case_results":completed,
        "field_physics_verification":"separate source-free SI check, not a browser observation",
        "non_success_controls_not_rendered":2,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    a=parser.parse_args()
    data=check(a.output_dir)
    (a.output_dir/"chromium-viewers.json").write_text(
        json.dumps(data,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("M36c actual source conductor viewers inspected in Chromium:",
          data["inspected_distinct_source_apparatus"])


if __name__=="__main__":
    main()
