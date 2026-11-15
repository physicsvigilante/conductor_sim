"""M27a actual offline Chromium check of a new static full-ring field map.

Validates source shapes/IDs, proper wire masking, nonzero out-of-plane E,
world coordinates, saved-physics-only SVG, mobile width and zero requests.
No browser E/V calculations or motion integration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

from examples.m27a_ring_field import make_case
from conductor_sim.ring_field_observer import (
    write_ring_field_slice, validate_ring_field_slice,
)


def check(output: Path):
    output.mkdir(parents=True,exist_ok=True)
    record=make_case("workbook-two-rings")
    assert validate_ring_field_slice(record)
    path=output/"m27a-two-rings.html"
    write_ring_field_slice(path,record,title="M27a · Real off-axis two-ring E/V")
    masked=sum(c["region"]=="source-exclusion" for c in record["grid"]["cells"])
    assert masked>0
    errors=[]; console=[]; requests=[]
    with sync_playwright() as playwright:
        browser=playwright.chromium.launch(headless=True)
        page=browser.new_page(viewport={"width":1330,"height":930})
        page.on("pageerror",lambda e:errors.append(str(e)))
        page.on("console",lambda e:console.append(e.text) if e.type=="error" else None)
        page.on("request",lambda e:requests.append(e.url))
        page.set_content(path.read_text(encoding="utf-8"),wait_until="load")
        assert page.locator("#static-ring-map").count()==1
        assert page.locator(".ring-source-projection").count()==2
        assert page.locator(".field-arrow").count()>20
        assert page.locator("#play").count()==0
        assert page.locator("#frame").count()==0
        payload=page.evaluate("""() => JSON.parse(
            document.getElementById("ring-field-data").textContent)""")
        assert payload==record
        assert not payload["time_evolution"] and not payload["motion_qualified"]
        assert [s["source_id"] for s in payload["ring_projections"]]==["ring:A","ring:B"]
        assert payload["plane"]["u"]==[1.,0.,0.] and payload["plane"]["v"]==[0.,0.,1.]
        cells=payload["grid"]["cells"]
        assert sum(c["region"]=="source-exclusion" for c in cells)==masked
        assert all(c["potential_v"] is None and c["electric_field_v_m"] is None
                   for c in cells if c["region"]=="source-exclusion")
        nonmasked=[c for c in cells if c["region"]=="physical-observation"]
        assert all(abs(c["field_normal_v_m"]+1e5)<1e-5 for c in nonmasked)
        assert not page.locator("#static-ring-map").evaluate(
            "(svg) => svg.outerHTML.includes('javascript:')")
        page.screenshot(path=str(output/"m27a-wide.png"),full_page=True)
        page.set_viewport_size({"width":390,"height":844})
        page.screenshot(path=str(output/"m27a-mobile.png"),full_page=True)
        mobile=page.evaluate("""() => ({
            viewport:window.innerWidth,
            document_width:document.documentElement.scrollWidth,
            svg_width:document.querySelector('#static-ring-map').getBoundingClientRect().width
        })""")
        assert mobile["viewport"]==390
        assert mobile["document_width"]<=390, mobile
        assert 0<mobile["svg_width"]<=390
        assert not errors and not console and not requests,(errors,console,requests)
        browser.close()
    report=dict(result="pass",scope="static source E/V data and real SVG display, not motion",
                ring_ids=["ring:A","ring:B"],mask_count=masked,
                full_transverse_field_retained=True,mobile=mobile,
                browser_errors=errors,console_errors=console,network_requests=requests,
                source_provenance="Workbook p12 Q40–41; 2D observation plane, grid and mask teacher-authored")
    (output/"m27a-browser.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,default=Path("output/m27a-browser"))
    args=ap.parse_args()
    print(json.dumps(check(args.output),sort_keys=True))


if __name__=="__main__":
    main()
