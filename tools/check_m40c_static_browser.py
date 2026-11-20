"""Real Chromium offline JS-on/off and mobile checks for M40c static source comparison."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

from tools.m40c_static_example import run_all
from tools.m40c_static_reference import check_static_comparison
from conductor_sim.spherical_sigma_atlas import _si


def check_browser(outdir: Path):
    outdir.mkdir(parents=True,exist_ok=True)
    evidence=run_all(outdir)
    check_static_comparison(evidence)
    url=(outdir/"m40c-static-teacher.html").resolve().as_uri()
    checks=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(
            headless=True,
            **({"executable_path": os.environ["M40C_BROWSER_EXECUTABLE"]}
               if os.environ.get("M40C_BROWSER_EXECUTABLE") else {}),
        )
        for javascript, width in ((True,1280),(False,1280),(True,390),(False,390)):
            context=browser.new_context(viewport={"width":width,"height":820},java_script_enabled=javascript)
            page=context.new_page()
            requested=[]
            page.on("request",lambda req: requested.append(req.url))
            page.goto(url,wait_until="load")
            assert page.title()=="M40c static spherical-source comparison"
            assert "TEACHER HYPOTHETICAL" in page.locator("#m40c-role").inner_text()
            assert "STATIC" in page.locator("#m40c-no-chronology").inner_text()
            assert "No time" in page.locator("#m40c-no-chronology").inner_text() or "time" in page.locator("#m40c-no-chronology").inner_text()
            assert page.locator('svg[role="img"]').count()==2
            assert page.locator('circle[id^="m40b-void-"]').count()==6
            assert page.locator('circle[id^="m40b-source-"]').count()==8
            assert page.locator('tr[id^="m40c-"]').count()==18
            assert page.locator("#m40c-source").inner_text()=="plus_A"
            body=page.locator("body").inner_text()
            assert "(-1.45, 0.04, 0.01) m" in body
            assert "(-1.4, 0.0, 0.0) m" in body
            assert "unchanged whole-face Gauss Q: 0 C" in body
            for cav in evidence["cavities"]:
                for j,sample in enumerate(cav["samples"]):
                    row=page.locator("#m40c-{}-{}".format(cav["cavity_id"],j))
                    assert row.count()==1
                    assert _si(sample["sigma_baseline_c_per_m2"],"C/m²") in row.inner_text()
                    assert _si(sample["sigma_revised_c_per_m2"],"C/m²") in row.inner_text()
                    if sample["delta_c_per_m2"] is not None:
                        assert _si(sample["delta_c_per_m2"],"C/m²") in row.inner_text()
            # Both viewports show same true XY projection with one metre scale.
            for i,cav in enumerate(evidence["baseline_voids"]):
                circle=page.locator("#m40b-void-{}".format(i)).first
                outer=evidence["outer_sphere"]
                x=425.+(cav["center_m"][0]-outer["center_m"][0])*375./outer["radius_m"]
                y=425.-(cav["center_m"][1]-outer["center_m"][1])*375./outer["radius_m"]
                assert abs(float(circle.get_attribute("cx"))-x)<1e-10
                assert abs(float(circle.get_attribute("cy"))-y)<1e-10
            assert all(v.startswith("file://") for v in requested),requested
            page.screenshot(path=str(outdir/("m40c-{}-{}px.png".format(
                "js-on" if javascript else "js-off",width))),full_page=False)
            checks.append({"js":javascript,"width":width,"source_positions_visible":8,
                           "signed_point_rows_visible":18,"remote_requests":0})
            context.close()
        # CSP stops post-hoc script insertion even if the imported HTML was edited.
        tampered=outdir/"m40c-injected-script.html"
        original=(outdir/"m40c-static-teacher.html").read_text(encoding="utf-8")
        assert "<script" not in original.lower()
        tampered.write_text(original.replace("</body>",
            "<script>window.__m40c_injected__=true</script></body>"),encoding="utf-8")
        ctx=browser.new_context(java_script_enabled=True)
        page=ctx.new_page()
        page.goto(tampered.resolve().as_uri(),wait_until="load")
        assert page.evaluate("window.__m40c_injected__===true") is False
        ctx.close()
        browser.close()
    result={"status":"PASS","browser_modes":checks,"csp_script_injection_executed":False,
            "source_jee_items_solved":0,"signed_physics_native_not_browser_generated":True}
    (outdir/"m40c-browser-evidence.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir",type=Path,default=Path("output/m40c-browser"))
    args=ap.parse_args()
    print(json.dumps(check_browser(args.output_dir),indent=2))

if __name__=="__main__":main()
