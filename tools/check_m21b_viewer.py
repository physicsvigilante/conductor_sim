"""Real offline Chromium acceptance for M21b's three JEE static observations."""
from __future__ import annotations

import argparse
import json
from math import hypot, isclose, sqrt
from pathlib import Path
import shutil
from playwright.sync_api import sync_playwright, Error as PlaywrightError


def check(html_path, *, case, screenshot=None):
    html_path=Path(html_path)
    if not html_path.is_file():
        raise FileNotFoundError(html_path)
    page_errors=[]; errors=[];external=[]
    with sync_playwright() as p:
        try:
            browser=p.chromium.launch(headless=True)
        except PlaywrightError:
            exe=shutil.which("chromium") or shutil.which("chromium-browser")
            if not exe: raise
            browser=p.chromium.launch(headless=True,executable_path=exe,args=["--no-sandbox"])
        page=browser.new_page(viewport={"width":1440,"height":970})
        page.on("pageerror", lambda e: page_errors.append(str(e)))
        page.on("console", lambda e: errors.append(e.text) if e.type=="error" else None)
        page.on("request", lambda r: external.append(r.url))
        page.set_content(html_path.read_text(encoding="utf-8"),wait_until="load")
        if case=="2021-equipotential":
            page.wait_for_function("window.conductorViewer && window.conductorViewer.mode==='static'")
            assert page.locator('header .kicker').inner_text()=="CONDUCTOR_SIM / M21B VERIFIED STATIC FIELD"
            data=page.evaluate('JSON.parse(document.getElementById("simulation-data").textContent)')
            assert data["schema"]=="conductor_sim.visualization.v1"
            assert len(data["frames"])==1
            meta=data["static_fixed_source_view"]
            assert meta["highlight_potential_v"]==0.
            assert meta["source_annotations"][0]["id"]=="charge:negative"
            assert meta["source_annotations"][1]["id"]=="charge:positive"
            assert meta["highlighted_segment_count"]>100
            assert len(data["frames"][0]["field_lines"])>3
            assert page.locator('#conductor-panel').is_hidden()
            assert page.locator('#density-panel').is_hidden()
            assert page.locator('#frame').is_hidden()
            assert page.locator('#heat').is_checked()
            assert page.locator('#contours').is_checked()
            assert page.locator('#lines').is_checked()
            text=page.locator('#notes').inner_text()
            assert 'JEE Advanced 2021 Paper 1 Q9' in text
            assert 'charge:negative' in text and 'charge:positive' in text
            canvas=page.locator('#scene')
            assert canvas.is_visible()
            source_pixels=page.evaluate("""() => {
              const c=document.getElementById('scene'),d=c.getContext('2d').getImageData(0,0,c.width,c.height).data;
              let dark=0;for(let i=0;i<d.length;i+=4)if(d[i]+d[i+1]+d[i+2]<550)dark++;
              return dark;
            }""")
            assert source_pixels>700,source_pixels
            original=page.evaluate("document.getElementById('scene').toDataURL()")
            page.locator('#contours').uncheck()
            changed=page.evaluate("document.getElementById('scene').toDataURL()")
            assert original!=changed
            page.locator('#contours').check()
            page.locator('#heat').uncheck()
            assert page.evaluate("document.getElementById('scene').toDataURL()") != original
            page.locator('#heat').check()
            assert page.evaluate("document.getElementById('scene').toDataURL()")==original
            assert not page.locator('#motion-controls').is_visible()
            physics=dict(
                mode="verified static physical plane",
                highlighted_segments=meta["highlighted_segment_count"],
                fixed_sources=len(meta["field"]["sources"]),
                painted_dark_pixels=source_pixels,
                canvas_visibility="passed",toggle_contours_heat="passed",
                original_charge_geometry=meta["field"]["sources"])
        else:
            data=page.evaluate('JSON.parse(document.getElementById("observation-data").textContent)')
            assert data["schema"]=="conductor_sim.jee_static_explorer.v1"
            assert page.locator('svg.plot').count()==2
            assert page.locator('.plot polyline').count()>=(3 if case=='2012-shell' else 2)
            assert page.locator('header .kicker').inner_text()=="CONDUCTOR_SIM / JEE STATIC SOURCE OBSERVER"
            assert "No time evolution" in page.locator("main").inner_text()
            profile=data["profile"]
            if case=="2012-shell":
                rows=profile["rows"]
                assert profile["coordinate"]=="radial"
                assert page.locator("#physical-surface").count()==2
                undefined=[r for r in rows if r["field_undefined"]]
                assert len(undefined)==1
                assert isclose(undefined[0]["coordinate_m"],.1,abs_tol=1e-13)
                assert undefined[0]["potential_v"] is not None
                assert page.locator('#work-summary').count()==0
                physics=dict(mode="radial surface",radial_rows=len(rows),
                             undefined_surface_field_count=len(undefined),
                             surface_markers=2)
            elif case=="ring-work":
                assert page.locator('#source-geometry svg').count()==1
                assert page.locator('#work-summary').is_visible()
                assert "No physical guide" in page.locator('#source-geometry').inner_text() or \
                       "No physical guide" in page.locator('#source-geometry svg').get_attribute("aria-label") or \
                       "No physical guide" in html_path.read_text(encoding="utf-8")
                work=data["endpoint_work"]
                assert isclose(work["field_work_j"],3.5950207169044694,rel_tol=1e-10)
                assert len(work["source_work_components"])==3
                assert work["source_work_components"][-1]["field_work_j"]==0.
                assert work["start_electric_field_v_m"][1]==1e5
                assert work["end_electric_field_v_m"][1]==1e5
                assert all(isclose(x["field_transverse_abs_v_m"],1e5,abs_tol=1e-8)
                           for x in profile["rows"])
                assert "3.5950207" in page.locator('#work-summary').inner_text()
                physics=dict(mode="static source-given endpoint work",field_work_j=work["field_work_j"],
                             transverse_e_v_m=work["start_electric_field_v_m"][1],
                             transverse_work_j=work["source_work_components"][-1]["field_work_j"])
            else:
                raise ValueError("unknown browser case")
        if screenshot:
            screenshot=Path(screenshot)
            screenshot.parent.mkdir(parents=True,exist_ok=True)
            page.screenshot(path=str(screenshot),full_page=False)
        page.set_viewport_size({"width":390,"height":844})
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
        assert page.locator('main').is_visible()
        assert not page_errors,page_errors
        assert not errors,errors
        assert not external,external
        browser.close()
        return dict(case=case,actual_browser="Chromium",physics=physics,mobile_px=390,
                    console_errors=errors,javascript_errors=page_errors,external_requests=external)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("html",type=Path)
    ap.add_argument("--case",required=True,choices=["2021-equipotential","2012-shell","ring-work"])
    ap.add_argument("--screenshot",type=Path)
    args=ap.parse_args()
    print(json.dumps(check(args.html,case=args.case,screenshot=args.screenshot),indent=2))


if __name__=="__main__":
    main()
