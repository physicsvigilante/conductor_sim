"""Real Chromium checks of structurally authored models in unchanged viewers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from playwright.sync_api import sync_playwright, Error as PlaywrightError

EXPECTED = {
    "U1": ("point:A", "point:B", "point:C"),
    "U2": ("rod:A", "point:B"),
    "U3": ("ring:A", "ring:B", "ring:C", "field:axial"),
    "A1": ("point:mirror-up", "point:mirror-down"),
}


def check(path, *, case, screenshot=None):
    path = Path(path)
    audit = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    structure = audit["assembly"]
    assert structure["schema"] == "conductor_sim.scenario_reassembly.v1"
    assert structure["origin"] == "teacher-hypothetical"
    assert structure["parent_revision_id"] == structure["parent_model"]["revision_id"]
    assert structure["child_revision_id"] == structure["child_model"]["revision_id"]
    errors, console, requests = [], [], []
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except PlaywrightError:
            executable = shutil.which("chromium") or shutil.which("chromium-browser")
            if not executable:
                raise
            browser = p.chromium.launch(headless=True, executable_path=executable, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width":1400, "height":1000})
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda e: console.append(e.text) if e.type == "error" else None)
        page.on("request", lambda r: requests.append(r.url))
        page.set_content(path.read_text(encoding="utf-8"), wait_until="load")
        page.wait_for_function("window.conductorViewer && window.conductorViewer.mode==='elapsed-time-prescribed'")
        data = page.evaluate("data")
        assert data == audit["viewer"]
        motion = data["prescribed_motion"]
        assert motion["field"] == structure["child_model"]["spec"]["field"]
        identities = [a["id"] for a in motion["source_annotations"]]
        assert tuple(identities) == EXPECTED[case]
        assert motion["termination"] == "duration"
        assert len(motion["samples"]) == 25 and len(data["frames"]) == 13
        assert all(a["grid"]["potential_v"] == [[None,None],[None,None]] for a in data["frames"])
        assert page.locator("#scene").is_visible()
        assert page.locator("#conductor-panel").is_hidden()
        source_text = page.locator("#sources").inner_text()
        assert all(identity in source_text for identity in identities)
        assert "Structural reassembly" in page.locator("#notes").inner_text()
        assert "Teacher hypothetical" in page.locator("#notes").inner_text()
        painted = page.evaluate("""() => {
            const points=[],ellipses=[],texts=[];
            const arc=ctx.arc,ellipse=ctx.ellipse,text=ctx.fillText,pl=frame.plane;
            ctx.arc=function(...a){points.push(a);return arc.apply(this,a)};
            ctx.ellipse=function(...a){ellipses.push(a);return ellipse.apply(this,a)};
            ctx.fillText=function(t,...a){texts.push(String(t));return text.call(this,t,...a)};
            const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
            const project=p=>[pl.u,pl.v,cross(pl.u,pl.v)].map(
                a=>a.reduce((s,v,i)=>s+v*(p[i]-pl.origin_m[i]),0));
            try{drawPrescribedSources(pl,x=>x,y=>y,project)}
            finally{ctx.arc=arc;ctx.ellipse=ellipse;ctx.fillText=text;drawScene()}
            return {points,ellipses,texts};
        }""")
        if case == "U1":
            assert len(painted["points"]) == 3
        if case == "U2":
            assert len(painted["points"]) == 1
            assert any("rod" in t.lower() for t in painted["texts"])
        if case == "U3":
            # Existing axis-only renderer shows the two meridian intersections
            # per ring, not a fabricated full off-axis ellipse/field.
            assert len(painted["points"]) == 6 and painted["ellipses"] == []
            assert all(any(label in t for t in painted["texts"]) for label in ("Ring A","Ring B","Ring C"))
            assert motion["axis_display"] == "horizontal"
        if case == "A1":
            assert painted["points"] == [] and painted["texts"] == []
            notes = page.locator("#notes").inner_text()
            assert notes.count("signed normal offset") == 2
            assert "No in-plane replacement source" in notes
        if motion.get("gravity") is not None:
            # Reuse the existing independent real force-arrow/energy checker.
            from check_m21e_viewer import _panel
            _, force_evidence = _panel(page)
        else:
            assert "U=qV" in page.locator("#motion-summary").inner_text()
            force_evidence = {"electric_only":True}
        for index in (0, 6, 12):
            page.locator("#frame").fill(str(index))
            page.locator("#frame").dispatch_event("input")
            assert page.evaluate("frame.motion_sample_index") == motion["frame_sample_indices"][index]
        image = page.evaluate("document.getElementById('scene').toDataURL()")
        page.locator("#trajectory").uncheck()
        assert image != page.evaluate("document.getElementById('scene').toDataURL()")
        page.locator("#trajectory").check()
        page.locator("#motion-reset").click()
        page.locator("#motion-speed").select_option("4")
        page.locator("#motion-play").click()
        page.wait_for_function("Number(document.getElementById('frame').value)>=2", timeout=5000)
        if page.evaluate("window.conductorViewer.playing"):
            page.locator("#motion-play").click()
        paused = page.locator("#frame").input_value()
        page.wait_for_timeout(80)
        assert paused == page.locator("#frame").input_value()
        page.locator("#motion-reset").click()
        assert page.locator("#frame").input_value() == "0"
        page.locator("#frame").fill("6")
        assert page.evaluate("data") == data
        if screenshot:
            Path(screenshot).parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(screenshot), full_page=True)
        page.set_viewport_size({"width":390, "height":844})
        assert page.evaluate("document.documentElement.scrollWidth<=window.innerWidth+1")
        assert page.locator("#motion-controls").is_visible()
        assert not errors and not requests and not console, (errors, requests, console)
        evidence = dict(case=case, browser=browser.version,
            source_ids=identities, source_drawings=painted, forces=force_evidence,
            physical_samples=25, snapshots=13, immutable_saved_data=True,
            playback="passed", mobile_width_px=390, javascript_errors=errors,
            external_requests=requests, console_errors=console)
        browser.close()
        return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", type=Path)
    parser.add_argument("--case", choices=EXPECTED, required=True)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.html, case=args.case, screenshot=args.screenshot), indent=2))


if __name__ == "__main__":
    main()
