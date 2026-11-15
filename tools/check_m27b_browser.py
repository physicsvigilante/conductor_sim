"""M27b real Chromium check of newly admitted off-axis ring particle motion.

The Python engine validates and numerically integrates first. Browser paints
only saved physical trajectory, source geometry, force/energy and event state.
A 390px viewport check and canvas path-toggle check are included.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

from examples.m27b_offaxis_motion import run_case


def check(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    paths={}
    examples={}
    for case in ("workbook-xy","driven-switch"):
        html,record=run_case(case)
        path=output/(case+".html")
        path.write_text(html,encoding="utf-8")
        path.with_suffix(".json").write_text(
            json.dumps(record,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        paths[case]=path
        examples[case]=record

    errors=[];consoles=[];requests=[]
    result={}
    with sync_playwright() as playwright:
        browser=playwright.chromium.launch(headless=True)
        page=browser.new_page(viewport={"width":1360,"height":900})
        page.on("pageerror",lambda e:errors.append(str(e)))
        page.on("console",lambda e:consoles.append(e.text) if e.type=="error" else None)
        page.on("request",lambda r:requests.append(r.url))
        for case in ("workbook-xy","driven-switch"):
            page.set_content(paths[case].read_text(encoding="utf-8"),wait_until="load")
            viewer="conductorEventViewer" if case=="workbook-xy" else "conductorDrivenViewer"
            root_id="event-data" if case=="workbook-xy" else "driven-data"
            page.wait_for_function("window."+viewer+" && window."+viewer+".loaded")
            data=page.evaluate("""(node) => JSON.parse(document.getElementById(node).textContent)""",root_id)
            geometry=data["source_geometry"]
            assert [e["source_id"] for e in geometry["entries"]]==[
                "ring:A","ring:B","field:up"]
            assert [e["motion_symmetry_mode"] for e in geometry["entries"][:2]]==[
                "meridian","meridian"]
            assert geometry["u"]==[1.,0.,0.]
            assert geometry["v"]==[0.,1.,0.]
            assert page.locator(".source-geometry-entry").count()==3
            assert page.locator("#source-geometry").is_checked()
            assert page.locator("#scene").count()==1
            assert page.locator("#play").count()==1 and page.locator("#reset").count()==1
            assert page.evaluate("window."+viewer+".physicalTime")==0
            assert page.locator("#path").is_checked()
            assert "Ring A" in page.locator("#sources").inner_text()
            assert "Ring B" in page.locator("#sources").inner_text()
            assert "full 3D off-axis E/V" in page.locator("#sources").inner_text()
            if case=="workbook-xy":
                rows=data["trajectory"]["samples"]
                expected_termination=data["trajectory"]["termination"]
                assert expected_termination=="duration"
                assert rows[-1]["position_m"][1]>rows[0]["position_m"][1]+1e-4
                assert rows[-1]["position_m"][0]>rows[0]["position_m"][0]
                assert len(data["events"])>=1
            else:
                rows=data["samples"]
                expected_termination=data["termination"]
                assert expected_termination=="duration"
                assert len(data["switches"])==1
                assert data["switches"][0]["time_s"]==.008
                assert rows[-1]["position_m"][1]>rows[0]["position_m"][1]+1e-4
            assert len(data["display_states"])>2
            # Canvas drawing is actually triggered by saved-time slider.
            page.evaluate("""() => {
                const ctx=document.querySelector('#scene').getContext('2d');
                const original=ctx.lineTo.bind(ctx);
                window.__segments=0;
                ctx.lineTo=function(...args){ window.__segments++; return original(...args) };
            }""")
            def select_last():
                page.locator("#frame").evaluate("""el => {
                    el.value=el.max;
                    el.dispatchEvent(new Event('input',{bubbles:true}));
                }""")
            select_last()
            end=page.evaluate("window."+viewer+".physicalTime")
            assert end==data["display_states"][-1]["time_s"]
            lines_on=page.evaluate("window.__segments")
            page.evaluate("window.__segments=0")
            page.locator("#path").uncheck()
            lines_off=page.evaluate("window.__segments")
            assert lines_on-lines_off>=len(data["display_states"])-1,(
                case,lines_on,lines_off,len(data["display_states"]))
            page.locator("#path").check()
            page.screenshot(path=str(output/(case+"-wide.png")),full_page=True)
            page.locator("#reset").click()
            assert page.evaluate("window."+viewer+".physicalTime")==0
            page.set_viewport_size({"width":390,"height":844})
            page.screenshot(path=str(output/(case+"-mobile.png")),full_page=True)
            mobile=page.evaluate("""() => ({
                width:window.innerWidth,
                document_width:document.documentElement.scrollWidth,
                canvas_width:document.querySelector('#scene').getBoundingClientRect().width
            })""")
            assert mobile["width"]==390 and mobile["document_width"]<=390,mobile
            assert mobile["canvas_width"]<=390
            result[case]=dict(
                actual_termination=expected_termination,
                saved_physical_end_time_s=end,
                source_ids=["ring:A","ring:B","field:up"],
                genuine_2d_trajectory=True,
                canvas_segments_with_path=lines_on,
                canvas_segments_without_path=lines_off,
                mobile=mobile)
            page.set_viewport_size({"width":1360,"height":900})
        browser.close()
    assert not errors and not consoles and not requests,(errors,consoles,requests)
    report=dict(status="pass",mode="Python-validated numerical physical time; browser saved-state playback only",
                cases=result,errors=errors,console_errors=consoles,network_requests=requests,
                source="Workbook ring geometry Q40–41; teacher-hypothetical charge/mass/launch/time")
    (output/"m27b-browser.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("output/m27b-browser"))
    args=parser.parse_args()
    print(json.dumps(check(args.output),sort_keys=True))


if __name__=="__main__":
    main()
